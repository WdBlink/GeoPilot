#!/usr/bin/env python3
"""Independent bounded repair of the published UseGeo mesh comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from usegeo_mesh_benchmark import benchmark as mesh


SCHEMA = "usegeo-mesh-official-compat-1.0"
RUN_SCHEMA = "usegeo-mesh-official-compat-run-1.0"


def stable_best90(distances: np.ndarray) -> tuple[np.ndarray, int]:
    """Select exactly ceil(90% N), breaking equal distances by source index."""
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise mesh.EvaluatorFailure("OFFICIAL_NONFINITE_DISTANCE")
    count = math.ceil(0.9 * len(values))
    order = np.lexsort((np.arange(len(values)), values))
    return values[order[:count]], count


def repaired_pass_loop(
    triangle_count: int,
    pass_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *, max_passes: int = 10000, deadline: float = math.inf,
) -> tuple[np.ndarray, int]:
    """Accumulate only explicitly computed finite triangle distances."""
    if triangle_count < 1:
        raise mesh.InvalidInput("EMPTY_GEOMETRY")
    distances = np.empty(triangle_count, dtype=np.float64)
    computed = np.zeros(triangle_count, dtype=bool)
    iterations = 0
    while not computed.all():
        if iterations >= max_passes:
            raise mesh.EvaluatorFailure("OFFICIAL_PASS_LIMIT")
        if time.monotonic() >= deadline:
            raise mesh.EvaluatorFailure("OFFICIAL_TIME_LIMIT")
        remaining = np.flatnonzero(~computed)
        indices, values = pass_fn(remaining)
        indices = np.asarray(indices)
        values = np.asarray(values, dtype=np.float64)
        if indices.ndim != 1 or values.ndim != 1 or len(indices) != len(values):
            raise mesh.EvaluatorFailure("OFFICIAL_INVALID_PRIMITIVE_ID")
        if len(indices) == 0:
            raise mesh.EvaluatorFailure("OFFICIAL_NO_PROGRESS")
        if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= triangle_count):
            raise mesh.EvaluatorFailure("OFFICIAL_INVALID_PRIMITIVE_ID")
        if len(np.unique(indices)) != len(indices) or np.any(computed[indices]):
            raise mesh.EvaluatorFailure("OFFICIAL_INVALID_PRIMITIVE_ID")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise mesh.EvaluatorFailure("OFFICIAL_NONFINITE_DISTANCE")
        distances[indices] = values
        computed[indices] = True
        iterations += 1
    if not computed.all() or not np.isfinite(distances).all():
        raise mesh.EvaluatorFailure("OFFICIAL_INCOMPLETE")
    return distances, iterations


def _open3d_pass_fn(
    vertices: np.ndarray, triangles: np.ndarray, full_lidar: np.ndarray,
    origin: np.ndarray, deadline: float, check: Callable[[], None],
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def run(remaining: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        check()
        if time.monotonic() >= deadline:
            raise mesh.EvaluatorFailure("OFFICIAL_TIME_LIMIT")
        minima = np.full(len(remaining), np.inf, dtype=np.float64)
        observed = np.zeros(len(remaining), dtype=bool)
        for start in range(0, len(full_lidar), 100000):
            check()
            if time.monotonic() >= deadline:
                raise mesh.EvaluatorFailure("OFFICIAL_TIME_LIMIT")
            distance, primitive = mesh.exact_triangle_queries(
                vertices, triangles[remaining], full_lidar[start:start + 100000], origin,
                check=lambda: (_ for _ in ()).throw(mesh.EvaluatorFailure("OFFICIAL_TIME_LIMIT"))
                if time.monotonic() >= deadline else None,
            )
            if np.any(primitive < 0) or np.any(primitive >= len(remaining)):
                raise mesh.EvaluatorFailure("OFFICIAL_INVALID_PRIMITIVE_ID")
            if not np.isfinite(distance).all():
                raise mesh.EvaluatorFailure("OFFICIAL_NONFINITE_DISTANCE")
            np.minimum.at(minima, primitive, distance)
            observed[primitive] = True
        local = np.flatnonzero(observed)
        return remaining[local], minima[local]

    return run


def compatibility_from_arrays(
    vertices: np.ndarray, triangles: np.ndarray, full_lidar: np.ndarray,
    refined_lidar: np.ndarray, *, timeout_seconds: float = 7200,
    max_passes: int = 10000, check: Callable[[], None] = lambda: None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run only the repaired upstream triangle correspondence procedure."""
    deadline = time.monotonic() + timeout_seconds
    origin = mesh.reference_origin(refined_lidar)
    distances, iterations = repaired_pass_loop(
        len(triangles), _open3d_pass_fn(vertices, triangles, full_lidar, origin, deadline, check),
        max_passes=max_passes, deadline=deadline,
    )
    selected, selected_count = stable_best90(distances)
    reverse = mesh.raycast_distances(
        vertices, triangles, refined_lidar, origin,
        check=lambda: (check(), (_ for _ in ()).throw(mesh.EvaluatorFailure("OFFICIAL_TIME_LIMIT"))
                       if time.monotonic() >= deadline else None)[-1],
    )
    return {
        "triangle_accuracy_l1_m": float(np.mean(selected)),
        "triangle_accuracy_rmse_m": float(np.sqrt(np.mean(np.square(selected)))),
        "triangle_accuracy_best90_count": selected_count,
        "completeness_0_20": float(np.mean(reverse <= 0.20)),
        "iterations": iterations,
        "all_triangle_distances_computed": True,
    }, origin


def _worker(order_path: Path) -> int:
    order = mesh._strict_json(order_path, "WORK_ORDER_INVALID")
    token = os.environ.get("USEGEO_OFFICIAL_TOKEN")
    if not token or order.get("token") != token or order.get("parent_pid") != os.getppid():
        raise mesh.EvaluatorFailure("official worker requires parent-issued work order")
    deadline = time.monotonic() + float(order["timeout_seconds"])
    cancel = Path(order["cancel"])
    check = lambda: mesh._check_worker_limits(deadline, cancel)
    vertices, triangles = mesh.read_ply(Path(order["mesh"]))
    reference = Path(order["bundle"]) / "evaluator-only" / order["scene"]
    full = mesh._las_points(reference / "full_lidar.las", check)
    refined = mesh._las_points(reference / "refined_lidar.las", check)
    metrics, origin = compatibility_from_arrays(
        vertices, triangles, full, refined,
        timeout_seconds=float(order["timeout_seconds"]), max_passes=int(order["max_passes"]), check=check,
    )
    mesh._write_json(Path(order["result"]), {
        "counts": {"vertices": len(vertices), "triangles": len(triangles),
                   "full_lidar_points": len(full), "refined_lidar_points": len(refined)},
        "metrics": metrics, "raycast_origin_m": origin.tolist(),
    })
    return 0


def run_official_compat(
    bundle: Path, bundle_lock: Path, scene: str, submission: Path, output: Path,
    cli_arguments: list[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Snapshot, bound, watchdog, and immutably publish compatibility evidence."""
    protocol = mesh.load_protocol()
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch: Path | None = None
    process: subprocess.Popen[str] | None = None
    try:
        mesh.runtime_versions()
        official_source = mesh.HERE.parents[1] / "out/usegeo_validation/official_evaluator/eval_mesh.py"
        if (not official_source.is_file() or official_source.is_symlink()
                or mesh.sha256_file(official_source) != protocol["upstream"]["official_eval_mesh_sha256"]):
            raise mesh.EvaluatorFailure("protected official evaluator source differs")
        mesh.verify_bundle(bundle, bundle_lock)
        scratch = Path(tempfile.mkdtemp(prefix=".usegeo-official-compat-", dir=output.parent))
        snap = mesh.snapshot_submission(submission, scratch / "submission", scene)
        vertices, triangles = mesh.read_ply(scratch / "submission" / "mesh.ply")
        token = hashlib.sha256(os.urandom(32)).hexdigest()
        order = {
            "token": token, "parent_pid": os.getpid(), "bundle": str(bundle.resolve()), "scene": scene,
            "mesh": str((scratch / "submission" / "mesh.ply").resolve()),
            "result": str((scratch / "result.json").resolve()),
            "cancel": str((scratch / "cancel.json").resolve()),
            "timeout_seconds": protocol["limits"]["official_timeout_seconds"],
            "max_passes": protocol["limits"]["official_max_passes"],
        }
        mesh._write_json(scratch / "order.json", order)
        environment = os.environ.copy()
        environment["USEGEO_OFFICIAL_TOKEN"] = token
        process = subprocess.Popen(
            [sys.executable, "-m", "usegeo_mesh_benchmark.official_compat", "official-worker",
             "--work-order", str(scratch / "order.json")], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=environment,
        )
        peak, stdout, stderr = mesh._watch_process(
            process, started, float(protocol["limits"]["official_timeout_seconds"]),
            int(protocol["limits"]["max_peak_rss_bytes"]),
            float(protocol["limits"]["watchdog_period_seconds"]), scratch / "cancel.json",
        )
        if process.returncode != 0:
            raise mesh.EvaluatorFailure(stderr.strip() or stdout.strip() or "OFFICIAL_WORKER_FAILED")
        worker = mesh._strict_json(scratch / "result.json", "OFFICIAL_RESULT_INVALID")
        bindings = mesh._bindings(bundle, bundle_lock, scene, snap)
        bindings.update({
            "official_eval_mesh_sha256": protocol["upstream"]["official_eval_mesh_sha256"],
            "upstream_commit": protocol["upstream"]["commit"],
        })
        result = {
            "schema_version": SCHEMA, "protocol_version": protocol["protocol_version"],
            "status": "completed", "track": "rgb-oriented", "scene_id": scene,
            "configuration_id": snap["contract"]["configuration_id"], "counts": worker["counts"],
            "official_compatibility_metrics": worker["metrics"], "bindings": bindings,
        }
        run = {
            "schema_version": RUN_SCHEMA, "status": "completed", "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": time.monotonic() - started, "peak_worker_rss_bytes": peak,
            "cli_arguments": list(cli_arguments or []), "bindings": bindings,
            "output_sha256": {"official_compat.json": hashlib.sha256(mesh._json_bytes(result)).hexdigest()},
        }
        if output.exists():
            existing = mesh._strict_json(output / "official_compat.json", "OFFICIAL_RESULT_INVALID")
            existing_run = mesh._strict_json(output / "run_manifest.json", "OFFICIAL_RESULT_INVALID")
            if existing != result or existing_run.get("bindings") != bindings or existing_run.get(
                "output_sha256", {}
            ).get("official_compat.json") != mesh.sha256_file(output / "official_compat.json"):
                raise mesh.EvaluatorFailure(f"existing output differs: {output}")
            return 0, existing
        mesh._publish_directory(output, {"official_compat.json": result, "run_manifest.json": run})
        return 0, result
    except mesh.InvalidInput as exc:
        return _publish_failure(output, protocol, scene, started_at, started, "invalid", exc.reason, 2)
    except (mesh.EvaluatorFailure, OSError, ValueError) as exc:
        return _publish_failure(output, protocol, scene, started_at, started, "not_completed", str(exc), 1)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def _publish_failure(
    output: Path, protocol: dict[str, Any], scene: str, started_at: str,
    started: float, status: str, reason: str, code: int,
) -> tuple[int, dict[str, Any]]:
    result = {
        "schema_version": SCHEMA, "protocol_version": protocol["protocol_version"],
        "status": status, "track": "rgb-oriented", "scene_id": scene,
        "configuration_id": None, "error": reason,
        "bindings": {"protocol_sha256": mesh.sha256_file(mesh.PROTOCOL_PATH)},
    }
    run = {
        "schema_version": RUN_SCHEMA, "status": status, "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": time.monotonic() - started, "peak_worker_rss_bytes": 0,
        "cli_arguments": [], "bindings": result["bindings"],
        "output_sha256": {"official_compat.json": hashlib.sha256(mesh._json_bytes(result)).hexdigest()},
    }
    mesh._publish_directory(output, {"official_compat.json": result, "run_manifest.json": run})
    return code, result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent repaired UseGeo official mesh comparator")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("official-compat")
    command.add_argument("--bundle", type=Path, required=True)
    command.add_argument("--bundle-lock", type=Path, required=True)
    command.add_argument("--scene", choices=["Dataset-1", "Dataset-2", "Dataset-3"], required=True)
    command.add_argument("--submission", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("official-worker", help=argparse.SUPPRESS)
    worker.add_argument("--work-order", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "official-worker":
            return _worker(arguments.work_order)
        code, result = run_official_compat(
            arguments.bundle, arguments.bundle_lock, arguments.scene, arguments.submission,
            arguments.output, list(sys.argv if argv is None else argv),
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return code
    except mesh.InvalidInput as exc:
        print(json.dumps({"status": "invalid", "invalid_reasons": [{"code": exc.reason, "detail": exc.detail}]}), file=sys.stderr)
        return 2
    except (mesh.EvaluatorFailure, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "detail": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
