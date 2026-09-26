#!/usr/bin/env python3
"""Serial full-scene acceptance runner; not an independent Mission verdict."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from usegeo_mesh_benchmark import benchmark as mesh
from usegeo_mesh_benchmark import official_compat


PROTECTED = {
    "code/usegeo_benchmark/README.md": "e5c9363c94dadcfbe5f48ed9dfa8c8d873b3e27e5bf40685aa2345b32e1d713a",
    "code/usegeo_benchmark/benchmark.py": "df94a4205219035d5e63bb9205306ae9fcca480c63b55e3168495ad3b35eb627",
    "code/usegeo_benchmark/protocol_v1.json": "418bcad7baca0e70e9aeeb353582bf12ea7cc88bfea37e1adbdcf1cf6385078c",
    "code/usegeo_benchmark/requirements.txt": "8c5a9ebb2bc6ea793bbc8cc2422f5d081106fc2921eedcdb15c610e1676d6667",
    "code/usegeo_benchmark/tests/test_benchmark.py": "462eb97850192b10076ef0c8416fc5a94a8fb9601d06078b2687baf39b979fae",
    "code/usegeo_benchmark/vendor/usegeo-aa689753/LICENSE": "ca7f0c8e79deb90183b6e8afc61bd1000b1a114214a552bb9393ffc2616158f2",
    "code/usegeo_benchmark/vendor/usegeo-aa689753/SOURCE.json": "ca86b983ce8d8e06b45a4e73bb4a52ed98d995fe458d13344eb943bc6f12e633",
    "code/usegeo_benchmark/vendor/usegeo-aa689753/eval_pointcloud.py": "6f391199dd90fe2e649bf305909ee3772a3bf307585ba42d4f3dae2a3882473c",
    "code/usegeo_benchmark/vendor/usegeo-aa689753/filter_ground_truth_pointcloud.py": "9666a490cddccbec5a8fda69f8905ba506b1fe09200ab9136429d1c148c2718f",
    "code/usegeo_benchmark/vendor/usegeo-aa689753/requirements.txt": "555a128255e998dad41ae7b223d611a2bfa6dbfea0a204ab8fc1df11bbf47ea9",
    "out/usegeo_benchmark/prepared/v1/manifest.json": "b4a99f53ab7297a159913776dd3ce97fcf8741f364cf713dda9fdd1a35771e40",
    "out/usegeo_benchmark/prepared/v1.lock.json": "bdff58a760d90327b5ed40876670cf3b2bcf38e08ed189d56529919dd738fe27",
    "out/usegeo_validation/official_evaluator/eval_mesh.py": "55f52e5c5674da6cb9dd081889975fa7484c8d199b7aa44db9874558492cb194",
}


def protected_report(root: Path) -> dict[str, Any]:
    observed: dict[str, str | None] = {}
    for relative in PROTECTED:
        path = root / relative
        observed[relative] = mesh.sha256_file(path) if path.is_file() and not path.is_symlink() else None
    mismatches = {key: {"expected": PROTECTED[key], "observed": value}
                  for key, value in observed.items() if value != PROTECTED[key]}
    return {"status": "valid" if not mismatches else "invalid", "observed": observed, "mismatches": mismatches}


def semantic(score: dict[str, Any]) -> dict[str, Any]:
    return {key: score[key] for key in (
        "status", "track", "scene_id", "configuration_id", "alignment", "counts",
        "surface_metrics", "structure_diagnostics", "bindings",
    )}


def load_bound_result(path: Path, bundle: Path, bundle_lock: Path,
                      scene: str, submission: Path, configuration_id: str) -> dict[str, Any]:
    """A valid result is reusable only for this exact input and submission."""
    score, _ = mesh._load_result(path)
    expected = mesh._bindings(bundle, bundle_lock, scene, {
        "contract_sha256": mesh.sha256_file(submission / "submission.json"),
        "mesh_sha256": mesh.sha256_file(submission / "mesh.ply"),
    })
    if (score["scene_id"] != scene or score["configuration_id"] != configuration_id
            or score["bindings"] != expected):
        raise mesh.EvaluatorFailure("completed result does not match this acceptance attempt")
    return score


def write_tap(path: Path, assertions: list[tuple[bool, str]]) -> None:
    lines = ["TAP version 13", f"1..{len(assertions)}"]
    lines.extend(f"{'ok' if passed else 'not ok'} {index} - {name}"
                 for index, (passed, name) in enumerate(assertions, 1))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _process_start_token(pid: int) -> str | None:
    """Return an OS start-time token; it prevents treating PID reuse as death."""
    try:
        return subprocess.check_output(
            ["ps", "-o", "lstart=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _owner_interrupted(state: dict[str, Any]) -> bool:
    owner = state.get("owner")
    if (not isinstance(owner, dict) or not isinstance(owner.get("pid"), int)
            or isinstance(owner.get("pid"), bool) or owner["pid"] <= 0
            or not isinstance(owner.get("start"), str) or not owner["start"].strip()):
        return False
    pid = owner["pid"]
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    # A live PID, a reused PID, or unavailable identity is not proof of death.
    return False


@contextmanager
def output_ownership(output: Path):
    """Hold an OS-released lock across preparation, execution and terminal writes."""
    output = output.absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise mesh.EvaluatorFailure("acceptance output contains a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the sibling inode: unlinking a flock file permits two separate owners.
    lock_path = output.parent / f".{output.name}.acceptance.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise mesh.EvaluatorFailure("acceptance output is owned by another process") from exc
        yield
    finally:
        os.close(descriptor)


def prepare_output(output: Path, root: Path | None = None) -> None:
    """Create a fresh attempt or archive terminal evidence for a safe retry."""
    if not output.exists():
        output.mkdir(parents=True)
        return
    if output.is_symlink() or not output.is_dir() or (output / "acceptance-summary.json").exists():
        raise mesh.EvaluatorFailure(f"acceptance output is not safely retryable: {output}")
    state_path = output / "execution-state.json"
    state = mesh._strict_json(state_path, "ACCEPTANCE_STATE_INVALID") if state_path.is_file() else {}
    if state.get("status") in {"running", "completed"}:
        if root is None or not _owner_interrupted(state):
            raise mesh.EvaluatorFailure(f"acceptance output has a live or unverified owner: {output}")
        observed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        after = protected_report(root)
        mesh._write_json(output / "protected-after.json", {
            "observed_at_recovery": observed_at, "report": after,
        })
        mesh._write_json(output / "acceptance-error.json", {
            "status": "interrupted", "detail": "previous owner ended before terminal evidence",
            "protected_after_observed_at": observed_at,
        })
        state.update({"status": "failed", "interrupted": True,
                      "protected_after_sha256": mesh.sha256_file(output / "protected-after.json")})
        mesh._write_json(state_path, state)
    if not (output / "acceptance-error.json").is_file() or not (output / "protected-after.json").is_file():
        raise mesh.EvaluatorFailure(f"acceptance output is not safely retryable: {output}")
    history = output / "attempt-history"
    attempt = history / f"attempt-{len(list(history.glob('attempt-*'))) + 1:04d}"
    attempt.mkdir(parents=True)
    for name in ("acceptance-error.json", "protected-after.json", "execution-state.json"):
        if (output / name).is_file():
            shutil.copy2(output / name, attempt / name)


def _run_attempt(bundle: Path, bundle_lock: Path, output: Path) -> int:
    root = Path(__file__).resolve().parents[3]
    protocol = mesh.load_protocol()
    state = {"schema_version": "usegeo-mesh-acceptance-state-1.0", "status": "running",
             "completed_formal_results": [],
             "owner": {"pid": os.getpid(), "start": _process_start_token(os.getpid())}}
    mesh._write_json(output / "execution-state.json", state)
    plan = {
        "schema_version": "usegeo-mesh-acceptance-plan-1.0",
        "scenario": "SCENARIO-1", "validator_type": "acceptance",
        "satisfies": ["OUT-1", "OUT-2", "OUT-3", "OUT-4", "OUT-5"],
        "protocol_sha256": mesh.sha256_file(mesh.PROTOCOL_PATH),
        "runtime": protocol["runtime"], "fixture_algorithm": protocol["fixture"]["algorithm_id"],
        "commands": {
            "fixtures": "generate-real-fixtures --configuration-id publisher-mvs-grid-1m-v1",
            "formal": "score each Dataset-1..3 serially twice, then aggregate each pass",
            "official": "independent official-compat campaign after formal TAP",
        },
    }
    mesh._write_json(output / "execution-plan.json", plan)
    before = protected_report(root)
    mesh._write_json(output / "protected-before.json", before)
    if before["status"] != "valid":
        raise mesh.EvaluatorFailure("protected point-cloud authority differs before acceptance")

    fixtures = output / "fixtures"
    fixture_manifest = mesh.generate_real_fixtures(
        bundle, bundle_lock, fixtures, protocol["fixture"]["configuration_id"],
        ["real_acceptance.py", "generate-real-fixtures"],
    )
    fixture_assertions: list[tuple[bool, str]] = []
    for scene in sorted(protocol["scenes"]):
        vertices, triangles = mesh.read_ply(fixtures / scene / "mesh.ply")
        fixture_assertions.append((len(vertices) > 0 and len(triangles) > 0, f"{scene} fixture is a triangle mesh"))
        fixture_assertions.append((fixture_manifest["scenes"][scene]["label"] == protocol["fixture"]["label"],
                                   f"{scene} fixture provenance label"))

    scores: dict[str, list[dict[str, Any]]] = {"pass-1": [], "pass-2": []}
    formal_assertions = fixture_assertions[:]
    resource_rows: list[dict[str, Any]] = []
    for pass_name in ("pass-1", "pass-2"):
        for scene in sorted(protocol["scenes"]):
            result_dir = output / "formal" / pass_name / scene
            if result_dir.exists():
                score = load_bound_result(result_dir, bundle, bundle_lock, scene,
                                          fixtures / scene, protocol["fixture"]["configuration_id"])
                code = 0
            else:
                code, score = mesh.score_submission(
                    bundle, bundle_lock, scene, fixtures / scene, result_dir,
                    ["real_acceptance.py", pass_name, scene],
                )
            state["completed_formal_results"].append(f"{pass_name}/{scene}")
            mesh._write_json(output / "execution-state.json", state)
            scores[pass_name].append(score)
            run_manifest = mesh._strict_json(result_dir / "run_manifest.json", "RESULT_RUN_MANIFEST_INVALID")
            resource_rows.append({"pass": pass_name, "scene": scene,
                                  "elapsed_seconds": run_manifest["elapsed_seconds"],
                                  "peak_worker_rss_bytes": run_manifest["peak_worker_rss_bytes"]})
            formal_assertions.append((code == 0 and score.get("status") == "valid", f"{pass_name} {scene} valid formal score"))
            formal_assertions.append((all(isinstance(value, (int, float)) and math.isfinite(value)
                                          for value in score.get("surface_metrics", {}).values()),
                                      f"{pass_name} {scene} finite metrics"))
            formal_assertions.append((run_manifest["elapsed_seconds"] < protocol["limits"]["scene_timeout_seconds"]
                                      and run_manifest["peak_worker_rss_bytes"] < protocol["limits"]["max_peak_rss_bytes"],
                                      f"{pass_name} {scene} resource bounds"))
        mesh.aggregate_scores(
            bundle, bundle_lock, protocol["fixture"]["configuration_id"],
            [output / "formal" / pass_name / scene for scene in sorted(protocol["scenes"])],
            output / "formal" / f"aggregate-{pass_name}.json",
        )

    for first, second in zip(scores["pass-1"], scores["pass-2"]):
        formal_assertions.append((semantic(first) == semantic(second), f"{first['scene_id']} repeat semantic equality"))
    aggregate_one = mesh._strict_json(output / "formal" / "aggregate-pass-1.json", "AGGREGATE_INVALID")
    aggregate_two = mesh._strict_json(output / "formal" / "aggregate-pass-2.json", "AGGREGATE_INVALID")
    formal_assertions.append((aggregate_one["macro_mean"] == aggregate_two["macro_mean"], "aggregate repeat equality"))

    neutral = output / "label-neutral-submission"
    if not neutral.exists():
        shutil.copytree(fixtures / "Dataset-1", neutral)
        contract = mesh._strict_json(neutral / "submission.json", "MALFORMED_CONTRACT")
        contract["method_metadata"] = {"name": "same bytes alternate label", "family": "Agent"}
        mesh._write_json(neutral / "submission.json", contract)
    neutral_output = output / "formal" / "label-neutral"
    if neutral_output.exists():
        neutral_score = load_bound_result(neutral_output, bundle, bundle_lock, "Dataset-1",
                                          neutral, protocol["fixture"]["configuration_id"])
        code = 0
    else:
        code, neutral_score = mesh.score_submission(
            bundle, bundle_lock, "Dataset-1", neutral, neutral_output,
            ["real_acceptance.py", "label-neutral"],
        )
    baseline = scores["pass-1"][0]
    formal_assertions.append((code == 0 and neutral_score["surface_metrics"] == baseline["surface_metrics"]
                              and neutral_score["structure_diagnostics"] == baseline["structure_diagnostics"],
                              "method label neutrality"))
    mesh._write_json(output / "resource-summary.json", {"runs": resource_rows})
    write_tap(output / "formal.tap", formal_assertions)
    if not all(passed for passed, _ in formal_assertions):
        return 1

    campaign: dict[str, Any] = {"status": "informational", "affects_formal_readiness": False, "scenes": {}}
    for scene in sorted(protocol["scenes"]):
        try:
            code, result = official_compat.run_official_compat(
                bundle, bundle_lock, scene, fixtures / scene, output / "official-compat" / scene,
                ["real_acceptance.py", "official-compat", scene],
            )
            campaign["scenes"][scene] = {"status": "completed" if code == 0 else "not_completed",
                                          "result": result if code == 0 else None,
                                          "error": None if code == 0 else result.get("error")}
        except Exception as exc:  # campaign is deliberately failure-independent
            campaign["scenes"][scene] = {"status": "not_completed", "result": None, "error": str(exc)}
    mesh._write_json(output / "official-compatibility-campaign.json", campaign)
    after = protected_report(root)
    mesh._write_json(output / "protected-after.json", after)
    if after != before:
        raise mesh.EvaluatorFailure("protected point-cloud authority changed during acceptance")
    mesh._write_json(output / "acceptance-candidate.json", {
        "status": "formal_pass", "formal_tap": "formal.tap",
        "official_campaign_affects_formal_readiness": False,
        "fixture_manifest_sha256": mesh.sha256_file(fixtures / "fixture_manifest.json"),
        "protected_before_sha256": mesh.sha256_file(output / "protected-before.json"),
        "protected_after_sha256": mesh.sha256_file(output / "protected-after.json"),
    })
    return 0


def _run_owned(bundle: Path, bundle_lock: Path, output: Path) -> int:
    # Preparation failures are deliberately outside the terminal-write scope.
    prepare_output(output, Path(__file__).resolve().parents[3])
    code = 1
    error: dict[str, Any] | None = None
    try:
        code = _run_attempt(bundle, bundle_lock, output)
    except Exception as exc:
        error = {"status": "failed", "error": type(exc).__name__, "detail": str(exc)}
        print(json.dumps({"status": "failed", "error": type(exc).__name__, "detail": str(exc)}), file=sys.stderr)
    finally:
        if output.exists() and output.is_dir() and not output.is_symlink():
            root = Path(__file__).resolve().parents[3]
            after = protected_report(root)
            mesh._write_json(output / "protected-after.json", after)
            before_path = output / "protected-before.json"
            if before_path.is_file() and mesh._strict_json(before_path, "PROTECTED_REPORT_INVALID") != after:
                error = {"status": "failed", "error": "EvaluatorFailure",
                         "detail": "protected point-cloud authority changed during acceptance"}
                code = 1
            if code != 0:
                mesh._write_json(output / "acceptance-error.json", error or {
                    "status": "failed", "error": "AcceptanceAssertionFailure",
                    "detail": "one or more frozen acceptance assertions failed",
                })
            elif (output / "acceptance-error.json").exists():
                (output / "acceptance-error.json").unlink()
            state_path = output / "execution-state.json"
            state = mesh._strict_json(state_path, "ACCEPTANCE_STATE_INVALID") if state_path.is_file() else {}
            state.update({"schema_version": "usegeo-mesh-acceptance-state-1.0",
                          "status": "completed" if code == 0 else "failed",
                          "protected_after_sha256": mesh.sha256_file(output / "protected-after.json")})
            mesh._write_json(state_path, state)
            if code == 0:
                try:
                    candidate = output / "acceptance-candidate.json"
                    summary = mesh._strict_json(candidate, "ACCEPTANCE_CANDIDATE_INVALID")
                    if (summary.get("status") != "formal_pass"
                            or summary.get("protected_after_sha256") != state["protected_after_sha256"]):
                        raise mesh.EvaluatorFailure("acceptance candidate does not match terminal protection evidence")
                    os.replace(candidate, output / "acceptance-summary.json")
                except Exception as exc:
                    code = 1
                    error = {"status": "failed", "error": type(exc).__name__, "detail": str(exc)}
                    print(json.dumps(error), file=sys.stderr)
                    state["status"] = "failed"
                    # If storage is wholly unavailable, main still fails on stderr.
                    mesh._write_json(output / "acceptance-error.json", error)
                    mesh._write_json(state_path, state)
    return code


def run(bundle: Path, bundle_lock: Path, output: Path) -> int:
    with output_ownership(output):
        return _run_owned(bundle, bundle_lock, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        return run(arguments.bundle, arguments.bundle_lock, arguments.output)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": type(exc).__name__, "detail": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
