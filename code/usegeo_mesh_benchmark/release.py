"""Fail-closed validation of a portable mesh acceptance evidence manifest.

This validates evidence identity and the fixed gate set, not scientific claims
inside a report. Those require the separately identified independent reviewer.
All referenced files must be ordinary files below the manifest directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


SCHEMA_VERSION = "usegeo-mesh-release-1.0"
CHECK_REQUIREMENTS = {
    "A01": "REQ-004", "A02": "REQ-004", "A03": "REQ-005",
    "A04": "REQ-006", "A05": "REQ-006", "A06": "REQ-007",
    "A07": "REQ-007", "A08": "REQ-007", "A09": "REQ-008",
    "A10": "REQ-008", "A11": "REQ-009", "A12": "REQ-010",
    "A13": "REQ-010", "A14": "REQ-011", "A15": "REQ-012",
    "A16": "REQ-013", "A17": "REQ-014", "A18": "REQ-015",
}
REQUIRED_FIELDS = {
    "schema_version": str, "release_id": str, "created_at": str,
    "scope": str, "source_files": dict, "protocol": dict,
    "criteria": dict, "data_lock": dict, "runtime": dict,
    "hardware": dict, "commands": list, "checks": list,
    "campaigns": list, "review": dict, "readiness": str,
}
SOURCE_SUFFIXES = {
    'usegeo_mesh_benchmark/benchmark.py', 'usegeo_mesh_benchmark/official_compat.py',
    'usegeo_mesh_benchmark/release.py', 'usegeo_mesh_benchmark/protocol_v1.json',
    'usegeo_mesh_benchmark/requirements.txt', 'usegeo_benchmark/benchmark.py',
    'usegeo_benchmark/protocol_v1.json',
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def text(value: object) -> bool:
    return type(value) is str and bool(value.strip())


def read_json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    data = json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {s}")))
    require(type(data) is dict, "manifest must be an object")
    return data


def evidence(root: Path, ref: dict) -> Path:
    require(isinstance(ref, dict), "evidence reference must be an object")
    name, digest = ref.get("path"), ref.get("sha256")
    require(text(name), "missing evidence path")
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "evidence must stay below its manifest directory")
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "invalid SHA256")
    path = root
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), "symlink evidence is forbidden")
    require(path.is_file(), f"missing ordinary evidence file: {name}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    require(hasher.hexdigest() == digest, f"evidence changed: {name}")
    return path


def validate(path: Path) -> str:
    require(not path.is_symlink(), "manifest cannot be a symlink")
    root = path.parent.resolve()
    data = read_json(path)
    for key, kind in REQUIRED_FIELDS.items():
        require(type(data.get(key)) is kind and bool(data[key]), f"missing/empty {key}")
    require(data["schema_version"] == SCHEMA_VERSION, "unknown schema")
    require(data["scope"] in {"evaluator_release", "paper_results"}, "unknown scope")
    require(text(data["release_id"]), "release ID missing")
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", data["created_at"]) is not None,
            "created_at must be an ISO UTC timestamp")
    datetime.fromisoformat(data["created_at"][:-1] + "+00:00")
    for name, digest in data["source_files"].items():
        evidence(root, {"path": name, "sha256": digest})
    for suffix in SOURCE_SUFFIXES:
        matches = [name for name in data['source_files'] if name == suffix or name.endswith('/' + suffix)]
        require(len(matches) == 1, f"required source snapshot missing/ambiguous: {suffix}")
        if suffix == 'usegeo_mesh_benchmark/protocol_v1.json':
            require(data['source_files'][matches[0]] == data['protocol'].get('sha256'), "source/protocol hash mismatch")
    for key in ("protocol", "criteria", "data_lock"):
        evidence(root, data[key])
    require(isinstance(data["runtime"].get("versions"), dict)
            and bool(data["runtime"]["versions"]), "runtime versions missing")
    require(all(text(k) and text(v) for k, v in data["runtime"]["versions"].items()), "runtime versions must be strings")
    evidence(root, data["runtime"].get("lock"))
    for key in ("cpu", "os"):
        require(text(data["hardware"].get(key)), f"hardware {key} missing")
    memory = data["hardware"].get("memory_bytes")
    require(type(memory) is int and memory > 0, "hardware memory_bytes must be positive integer")
    for command in data["commands"]:
        require(type(command) is dict, "command must be an object")
        require(text(command.get("cwd")), "cwd missing")
        require(isinstance(command.get("argv"), list) and bool(command["argv"])
                and all(type(x) is str for x in command["argv"]) and text(command["argv"][0]), "argv missing")
        require(type(command.get("exit_code")) is int, "exit code missing")
        evidence(root, command.get("log"))
    required = set(list(CHECK_REQUIREMENTS)[:14 if data["scope"] == "evaluator_release" else 18])
    seen = set()
    passed = True
    for check in data["checks"]:
        require(type(check) is dict, "check must be an object")
        key = check.get("check_id")
        require(text(key), "check ID missing")
        require(key in required and key not in seen, "unknown/duplicate check")
        seen.add(key)
        require(check.get("req_id") == CHECK_REQUIREMENTS[key], "requirement mismatch")
        require(type(check.get("status")) is str and check["status"] in {"PASS", "FAIL", "BLOCKED"}, "invalid check status")
        require(text(check.get("expected")) and text(check.get("observed")), "assertion must be nonempty text")
        require(isinstance(check.get("evidence"), list) and bool(check["evidence"]), "check evidence missing")
        for ref in check["evidence"]:
            evidence(root, ref)
        passed = passed and check["status"] == "PASS"
    require(seen == required, "required acceptance checks are missing")
    evaluator = None
    if data["scope"] == "paper_results":
        evaluator_path = evidence(root, data.get("evaluator_release"))
        evaluator = read_json(evaluator_path)
        require(evaluator.get("scope") == "evaluator_release", "paper must bind a separate evaluator release")
        require(validate(evaluator_path) == "READY", "bound evaluator is not READY")
        for key in ("protocol", "criteria", "data_lock"):
            require(data[key]["sha256"] == evaluator[key]["sha256"], f"paper/evaluator {key} mismatch")
        require(data['source_files'] == evaluator['source_files'], "paper/evaluator source snapshot mismatch")
        require(data['runtime']['versions'] == evaluator['runtime']['versions']
                and data['runtime']['lock']['sha256'] == evaluator['runtime']['lock']['sha256'],
                "paper/evaluator runtime mismatch")
    runs = set()
    successful_fixtures, official_attempts, method_scenes = set(), set(), set()
    method_identity = None
    for run in data["campaigns"]:
        require(type(run) is dict, "campaign must be an object")
        require(type(run.get("scene")) is str and run["scene"] in {"Dataset-1", "Dataset-2", "Dataset-3"}, "unknown scene")
        require(text(run.get("run_id")), "run ID missing")
        require(run["run_id"] not in runs, "duplicate run ID")
        runs.add(run["run_id"])
        require(type(run.get("status")) is str and run["status"] in {"valid", "failed", "timeout", "cancelled"}, "invalid run status")
        require(type(run.get("kind")) is str and run["kind"] in {"formal_fixture", "official_compat", "method"}, "run kind missing")
        evidence(root, run.get("output"))
        if run["kind"] == "formal_fixture":
            require(type(run.get("round")) is int and run["round"] in {1, 2}, "fixture round missing")
            pair = (run["scene"], run["round"])
            if run["status"] == "valid":
                require(pair not in successful_fixtures, "duplicate valid fixture scene/round")
                successful_fixtures.add(pair)
        elif run["kind"] == "official_compat":
            official_attempts.add(run["scene"])
        elif data["scope"] == "paper_results":
            identity = validate_experiment(root, run, data, evaluator)
            if run["status"] == "valid":
                require(run["scene"] not in method_scenes, "duplicate valid method scene")
                require(method_identity is None or identity == method_identity, "method configuration differs across scenes")
                method_identity = identity
                method_scenes.add(run["scene"])
    review = data["review"]
    require(text(review.get("reviewer")) and text(review.get("implementer"))
            and review["reviewer"].strip() != review["implementer"].strip(), "independent reviewer missing")
    require(type(review.get("status")) is str and review["status"] in {"PASS", "FAIL", "BLOCKED"}, "review status missing")
    evidence(root, review.get("report"))
    scenes = {"Dataset-1", "Dataset-2", "Dataset-3"}
    complete = successful_fixtures == {(scene, repeat) for scene in scenes for repeat in (1, 2)}
    complete = complete and official_attempts == scenes
    if data["scope"] == "paper_results":
        complete = complete and method_scenes == scenes
    readiness = "READY" if passed and complete and review["status"] == "PASS" else "NOT_READY"
    require(data["readiness"] == readiness, "readiness is not derived from checks/review")
    return readiness


def validate_experiment(root: Path, run: dict, paper: dict, evaluator: dict) -> tuple:
    """Check artifact identity, not whether an author's scientific claim is true."""
    path = evidence(root, run.get("experiment"))
    exp = read_json(path)
    require(exp.get("schema_version") == "usegeo-mesh-experiment-1.0", "unknown experiment schema")
    for key in ("experiment_id", "method_id", "version", "configuration_id"):
        require(text(exp.get(key)), f"experiment {key} missing")
    require(type(exp.get("seed")) is int, "experiment seed missing")
    require(exp.get("track") == "rgb-oriented" and exp.get("scene_id") == run["scene"], "experiment scene/track mismatch")
    require(exp.get("run_id") == run["run_id"] and exp.get("status") == run["status"], "experiment run/status mismatch")
    require(exp.get("release_id") == evaluator["release_id"], "experiment evaluator ID mismatch")
    require(exp.get("release_lock_sha256") == paper["evaluator_release"]["sha256"], "experiment evaluator lock mismatch")
    require(exp.get("data_lock_sha256") == paper["data_lock"]["sha256"], "experiment data lock mismatch")
    # References are always relative to the release evidence root, even if the
    # experiment manifest itself is nested, avoiding two resolution conventions.
    for key in ("preregistration", "allowed_inputs", "run_manifest"):
        evidence(root, exp.get(key))
    for key in ("method_cost", "evaluator_resources"):
        require(type(exp.get(key)) is dict and bool(exp[key]), f"experiment {key} missing")
    if run["status"] == "valid":
        for key in ("mesh", "score"):
            evidence(root, exp.get(key))
        require(exp["score"] == run["output"], "experiment score/output binding mismatch")
        from usegeo_mesh_benchmark import benchmark
        score_path, run_path = root / exp['score']['path'], root / exp['run_manifest']['path']
        require(score_path.name == 'score.json' and run_path.name == 'run_manifest.json'
                and score_path.parent == run_path.parent, "score and run manifest must be siblings")
        try:
            score, recorded = benchmark._load_result(score_path.parent)
        except (benchmark.InvalidInput, benchmark.EvaluatorFailure) as exc:
            raise ValueError(f"invalid bound scorer result: {exc}") from exc
        require(score.get('scene_id') == run['scene'] and score.get('track') == exp['track']
                and score.get('configuration_id') == exp['configuration_id'], "score identity differs from experiment")
        bindings = score.get('bindings')
        require(type(bindings) is dict and recorded.get('bindings') == bindings, "score/run bindings differ")
        for name, expected in {'mesh_sha256':exp['mesh']['sha256'],
                'input_manifest_sha256':exp['allowed_inputs']['sha256'],
                'protocol_sha256':paper['protocol']['sha256'], 'bundle_lock_sha256':paper['data_lock']['sha256']}.items():
            require(bindings.get(name) == expected, f"score {name} differs from experiment")
        require(type(recorded.get('output_sha256')) is dict
                and recorded['output_sha256'].get('score.json') == exp['score']['sha256'], "run score hash mismatch")
        require(recorded['packages'] == paper['runtime']['versions'], "score runtime differs from release")
        for key in ('method_id', 'version', 'seed'):
            require(type(recorded['method_metadata'].get(key)) is type(exp[key])
                    and recorded['method_metadata'][key] == exp[key], f"recorded method {key} mismatch")
        require(exp['evaluator_resources'] == {key:recorded[key] for key in
                ('elapsed_seconds','peak_worker_rss_bytes')}, "evaluator resources differ from run manifest")
    else:
        require(exp["run_manifest"] == run["output"], "failed experiment output must be its run manifest")
    return (exp["method_id"], exp["version"], exp["configuration_id"], exp["seed"], exp["preregistration"]["sha256"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        readiness = validate(args.manifest)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}))
        return 1
    print(json.dumps({"status": "valid", "readiness": readiness}))
    return 0 if readiness == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
