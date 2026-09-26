import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import laspy
import numpy as np
import pytest
from PIL import Image


MODULE_PATH = Path(__file__).parents[1] / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("usegeo_benchmark", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def write_las(path: Path, points: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = points.T
    cloud.write(path)


def install_synthetic_scoring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Install a tiny internal fixture without weakening the production CLI."""
    protocol = benchmark.load_protocol()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    bundle_lock = tmp_path / "bundle.lock.json"
    bundle_lock.write_text("{}", encoding="utf-8")
    manifest = {
        "scenes": {
            scene: {
                "input_manifest_sha256": {
                    "rgb-local": hashlib.sha256(f"local/{scene}".encode()).hexdigest(),
                    "rgb-oriented": hashlib.sha256(
                        f"oriented/{scene}".encode()
                    ).hexdigest(),
                },
                "reference_manifest_sha256": hashlib.sha256(
                    f"reference/{scene}".encode()
                ).hexdigest(),
            }
            for scene in protocol["scenes"]
        }
    }
    observed_snapshot_points: list[np.ndarray] = []

    monkeypatch.setattr(benchmark, "verify_bundle", lambda _bundle, _lock: manifest)

    def validate_snapshot(
        _bundle: Path,
        track: str,
        scene: str,
        snapshot: Path,
        current_protocol: dict,
        current_manifest: dict,
        contract: dict,
        contract_sha: str,
    ) -> dict:
        assert track == "rgb-oriented"
        assert current_protocol is protocol or current_protocol == protocol
        assert current_manifest is manifest
        geometry = snapshot / "pointcloud.las"
        point_count = benchmark._validate_las(geometry, current_protocol)
        scene_record = manifest["scenes"][scene]
        return {
            "protocol": current_protocol,
            "contract": contract,
            "configuration_id": contract["configuration_id"],
            "contract_sha256": contract_sha,
            "camera_centers_sha256": None,
            "geometry": geometry,
            "geometry_sha256": benchmark.sha256_file(geometry),
            "point_count": point_count,
            "input_manifest_sha256": scene_record["input_manifest_sha256"][track],
            "reference_manifest_sha256": scene_record[
                "reference_manifest_sha256"
            ],
            "reference_manifest": {},
            "reference_root": snapshot,
            "alignment": {
                "kind": "identity",
                "matrix_4x4": np.eye(4).tolist(),
            },
        }

    def production_metrics(validated: dict, _scratch: Path) -> dict:
        with laspy.open(validated["geometry"]) as reader:
            cloud = reader.read()
            points = np.column_stack((cloud.x, cloud.y, cloud.z))
        observed_snapshot_points.append(points)
        return {
            "submitted_point_count": int(len(points)),
            "refined_reference_point_count": 3,
            "accuracy_l1_m": 0.0,
            "accuracy_rmse_m": 0.0,
            "completeness_0_20": 1.0,
        }

    monkeypatch.setattr(benchmark, "_validate_submission_snapshot", validate_snapshot)
    monkeypatch.setattr(benchmark, "_production_metrics", production_metrics)
    return bundle, bundle_lock, manifest, observed_snapshot_points


def make_submission(root: Path, scene: str, configuration_id: str) -> Path:
    root.mkdir()
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    write_las(root / "pointcloud.las", points)
    benchmark._write_json(
        root / "submission.json",
        {
            "schema_version": "usegeo-submission-1.1",
            "protocol_version": "usegeo-pointcloud-v1",
            "track": "rgb-oriented",
            "scene_id": scene,
            "product": "pointcloud",
            "geometry": "pointcloud.las",
            "configuration_id": configuration_id,
        },
    )
    return root


def tiny_archive(tmp_path: Path, *, unsafe: bool = False) -> tuple[Path, dict]:
    source = tmp_path / "source"
    source.mkdir()
    image = source / "frame.jpg"
    Image.new("RGB", (4, 3), "white").save(image)
    lidar = source / "lidar.las"
    mvs = source / "mvs.las"
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    write_las(lidar, points)
    write_las(mvs, points)
    archive = tmp_path / "Tiny.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as target:
        target.write(image, "Tiny/Camera_Inputs/Metashape_outputs_images/frame.jpg")
        target.write(image, "Tiny/Camera_Inputs/Metashape_outputs_images/unmatched.jpg")
        target.write(lidar, "Tiny/LiDAR.las")
        target.write(mvs, "Tiny/MVS.las")
        target.writestr(
            "Tiny/orientations.xyz",
            "#label X0 Y0 Z0 omega phi kappa c x0 y0 a3 a4 a5 a6 rho0\n"
            "frame.jpg 0 0 1 0 0 0 1 0 0 0 0 0 0 0\n",
        )
        target.writestr("Tiny/Depth_resized/forbidden.tiff", b"not extracted")
        if unsafe:
            target.writestr("../escape", b"no")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    protocol = {
        "archive_sha256": {"Tiny.zip": digest},
        "completeness_threshold_m": 0.2,
        "corrections": [],
        "evaluator_name": "fixture",
        "input_image_dimensions": [4, 3],
        "limits": {
            "free_space_required_bytes": 0,
            "las_query_chunk_points": 2,
            "max_peak_rss_bytes": 64424509440,
            "max_submission_bytes": 1000000,
            "max_submission_points": 100,
            "query_workers": 1,
        },
        "protocol_version": "fixture-v1",
        "reference_support": {
            "kind": "publisher-MVS-derived 5 m XY support",
            "xy_threshold_m": 5.0,
        },
        "scenes": {
            "Tiny": {
                "archive": "Tiny.zip",
                "full_lidar_member": "Tiny/LiDAR.las",
                "image_count": 1,
                "image_prefix": "Tiny/Camera_Inputs/Metashape_outputs_images/",
                "orientation_member": "Tiny/orientations.xyz",
                "publisher_mvs_member": "Tiny/MVS.las",
            }
        },
        "submission_schema_version": "fixture-submission",
        "tracks": {
            "rgb-local": {},
            "rgb-oriented": {},
        },
        "trim_fraction": 0.9,
        "unsupported_products": ["depth", "mesh"],
        "upstream": {"commit": "fixture"},
    }
    return archive, protocol


def test_metric_identity_ties_and_controlled_degradation() -> None:
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.7, 0.0], [2.3, 2.0, 0.0]]
    )
    identity = benchmark.metric_arrays(reference, reference, reference)
    assert identity["accuracy_l1_m"] == 0.0
    assert identity["accuracy_rmse_m"] == 0.0
    assert identity["completeness_0_20"] == 1.0

    ties = np.zeros((10, 3), dtype=np.float64)
    tied = benchmark.metric_arrays(ties, ties[:1], ties[:1])
    assert tied["accuracy_l1_m"] == tied["accuracy_rmse_m"] == 0.0

    shifted = reference + [0.0, 0.0, 0.25]
    degraded = benchmark.metric_arrays(shifted, reference, reference)
    assert degraded["accuracy_l1_m"] == pytest.approx(0.25)
    assert degraded["accuracy_rmse_m"] == pytest.approx(0.25)
    assert degraded["completeness_0_20"] == 0.0


def test_exact_count_matches_upstream_formula_on_non_tied_fixture() -> None:
    estimate = np.array([[0.01 * i, 0.0, float(i)] for i in range(10)])
    reference = np.array([[0.0, 0.0, float(i)] for i in range(10)])
    result = benchmark.metric_arrays(estimate, reference, reference)
    distances = np.linalg.norm(estimate - reference, axis=1)
    upstream_selected = distances[distances <= np.sort(distances)[8]]
    assert len(upstream_selected) == 9
    assert result["accuracy_l1_m"] == pytest.approx(upstream_selected.mean(), abs=1e-12)
    assert result["accuracy_rmse_m"] == pytest.approx(
        np.sqrt(np.mean(upstream_selected**2)), abs=1e-12
    )


def test_similarity_recovers_proper_sim3_for_planar_centers() -> None:
    target = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    angle = np.deg2rad(31.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    source = ((target - [5.0, -2.0, 8.0]) @ rotation) / 2.5
    matrix, rmse = benchmark.compute_similarity(source, target)
    aligned = source @ matrix[:3, :3].T + matrix[:3, 3]
    assert np.linalg.det(matrix[:3, :3]) > 0
    assert rmse < 1e-12
    assert np.allclose(aligned, target, atol=1e-12)


def test_similarity_rejects_rank_deficiency_reflection_and_nonfinite() -> None:
    with pytest.raises(benchmark.InvalidInput, match="CAMERA_CENTERS_RANK_DEFICIENT"):
        benchmark.compute_similarity(np.zeros((3, 3)), np.zeros((3, 3)))
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    target = source.copy()
    target[:, 0] *= -1
    with pytest.raises(benchmark.InvalidInput, match="REFLECTION_ALIGNMENT"):
        benchmark.compute_similarity(source, target)
    source[0, 0] = np.nan
    with pytest.raises(benchmark.InvalidInput, match="CAMERA_CENTERS_NONFINITE"):
        benchmark.compute_similarity(source, target)


def test_camera_center_contract_reason_codes(tmp_path: Path) -> None:
    expected = ["a.jpg", "b.jpg", "c.jpg"]
    path = tmp_path / "centers.csv"
    path.write_text("image_id,x,y,z\na.jpg,0,0,0\na.jpg,1,0,0\n", encoding="utf-8")
    with pytest.raises(benchmark.InvalidInput, match="CAMERA_CENTERS_DUPLICATE_ID"):
        benchmark._read_camera_centers(path, expected)
    path.write_text("image_id,x,y,z\na.jpg,nan,0,0\nb.jpg,0,1,0\nc.jpg,1,0,0\n")
    with pytest.raises(benchmark.InvalidInput, match="CAMERA_CENTERS_NONFINITE"):
        benchmark._read_camera_centers(path, expected)
    path.write_text("image_id,x,y,z\na.jpg,0,0,0\nb.jpg,0,1,0\n")
    with pytest.raises(benchmark.InvalidInput, match="CAMERA_CENTERS_MISSING_ID"):
        benchmark._read_camera_centers(path, expected)
    path.write_text(
        "image_id,x,y,z\na.jpg,0,0,0\nb.jpg,0,1,0\nc.jpg,1,0,0\nd.jpg,1,1,0\n"
    )
    with pytest.raises(benchmark.InvalidInput, match="CAMERA_CENTERS_EXTRA_ID"):
        benchmark._read_camera_centers(path, expected)


def test_synthetic_preparation_is_safe_complete_and_idempotent(tmp_path: Path) -> None:
    archive, protocol = tiny_archive(tmp_path)
    bundle_parent = tmp_path / "prepared"
    first = benchmark._prepare_archives(
        archive.parent, bundle_parent, protocol, min_free_bytes=0
    )
    second = benchmark._prepare_archives(
        archive.parent, bundle_parent, protocol, min_free_bytes=0
    )
    assert first == second
    bundle = bundle_parent / "v1"
    benchmark._verify_bundle_manifest(
        bundle,
        protocol,
        trusted_manifest_sha256=benchmark.sha256_file(bundle / "manifest.json"),
        expected_protocol_sha=hashlib.sha256(benchmark._json_bytes(protocol)).hexdigest(),
    )
    assert (bundle / "inputs/rgb-local/Tiny/images/frame.jpg").is_file()
    assert not (bundle / "inputs/rgb-local/Tiny/images/unmatched.jpg").exists()
    assert not list((bundle / "inputs").rglob("*Depth*"))
    assert first["scenes"]["Tiny"]["image_count"] == 1
    assert first["forbidden_member_scan"]["Tiny"] == [
        "Tiny/Depth_resized/forbidden.tiff"
    ]
    assert first["scenes"]["Tiny"]["refined_lidar_point_count"] == 3


def test_synthetic_preparation_rejects_traversal(tmp_path: Path) -> None:
    archive, protocol = tiny_archive(tmp_path, unsafe=True)
    with pytest.raises(benchmark.EvaluatorFailure, match="unsafe ZIP member"):
        benchmark._prepare_archives(
            archive.parent, tmp_path / "prepared", protocol, min_free_bytes=0
        )


def test_synthetic_preparation_rejects_selected_member_crc_failure(tmp_path: Path) -> None:
    archive, protocol = tiny_archive(tmp_path)
    jpeg = (tmp_path / "source/frame.jpg").read_bytes()
    raw = bytearray(archive.read_bytes())
    offset = raw.find(jpeg)
    assert offset >= 0
    raw[offset + len(jpeg) // 2] ^= 1
    archive.write_bytes(raw)
    protocol["archive_sha256"]["Tiny.zip"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises((benchmark.EvaluatorFailure, zipfile.BadZipFile), match="CRC"):
        benchmark._prepare_archives(
            archive.parent, tmp_path / "prepared", protocol, min_free_bytes=0
        )


def test_invalid_geometry_paths_and_json_are_strict(tmp_path: Path) -> None:
    empty = tmp_path / "empty.las"
    empty.write_bytes(b"")
    protocol = benchmark.load_protocol()
    with pytest.raises(benchmark.InvalidInput, match="MALFORMED_LAS"):
        benchmark._validate_las(empty, protocol)
    empty_valid = tmp_path / "empty-valid.las"
    write_las(empty_valid, np.empty((0, 3)))
    with pytest.raises(benchmark.InvalidInput, match="EMPTY_GEOMETRY"):
        benchmark._validate_las(empty_valid, protocol)
    real = tmp_path / "real.las"
    write_las(real, np.array([[0.0, 0.0, 0.0]]))
    tiny_byte_limit = json.loads(json.dumps(protocol))
    tiny_byte_limit["limits"]["max_submission_bytes"] = 1
    with pytest.raises(benchmark.InvalidInput, match="GEOMETRY_TOO_LARGE"):
        benchmark._validate_las(real, tiny_byte_limit)
    no_point_limit = json.loads(json.dumps(protocol))
    no_point_limit["limits"]["max_submission_points"] = 0
    with pytest.raises(benchmark.InvalidInput, match="POINT_LIMIT_EXCEEDED"):
        benchmark._validate_las(real, no_point_limit)
    linked = tmp_path / "linked.las"
    linked.symlink_to(real)
    with pytest.raises(benchmark.InvalidInput, match="GEOMETRY_NOT_REGULAR"):
        benchmark._validate_las(linked, protocol)
    payload = {"status": "invalid", "invalid_reasons": [{"code": "EMPTY_GEOMETRY"}]}
    serialized = benchmark._json_bytes(payload)
    assert b"NaN" not in serialized and b"Infinity" not in serialized and b"metrics" not in serialized


def test_method_metadata_cannot_change_metrics() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    agent = {"method_metadata": {"family": "Agent"}}
    conventional = {"method_metadata": {"family": "non-Agent"}}
    agent_hash = hashlib.sha256(benchmark._json_bytes(agent)).hexdigest()
    conventional_hash = hashlib.sha256(benchmark._json_bytes(conventional)).hexdigest()
    first = benchmark.metric_arrays(points, points, points)
    second = benchmark.metric_arrays(points, points, points)
    assert agent_hash != conventional_hash
    assert first == second


def test_bounded_production_metric_phases_on_tiny_las(tmp_path: Path) -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.7, 0.0], [2.3, 2.0, 0.0]]
    )
    reference_root = tmp_path / "references"
    reference_root.mkdir()
    full = reference_root / "full.las"
    refined = reference_root / "refined.las"
    geometry = tmp_path / "submission.las"
    for path in (full, refined, geometry):
        write_las(path, points)
    protocol = benchmark.load_protocol()
    protocol["limits"]["las_query_chunk_points"] = 2
    protocol["limits"]["query_workers"] = 1
    validated = {
        "protocol": protocol,
        "alignment": {"matrix_4x4": np.eye(4).tolist()},
        "reference_manifest": {
            "files": {
                "full_lidar": {"path": full.name},
                "refined_lidar": {"path": refined.name},
            }
        },
        "reference_root": reference_root,
        "geometry": geometry,
        "point_count": len(points),
    }
    metrics = benchmark._production_metrics(validated, tmp_path)
    assert metrics["accuracy_l1_m"] == metrics["accuracy_rmse_m"] == 0.0
    assert metrics["completeness_0_20"] == 1.0
    assert not (tmp_path / "accuracy-distances.float64").exists()


def test_submission_contract_scope_and_unsupported_product() -> None:
    protocol = benchmark.load_protocol()
    assert set(protocol["unsupported_products"]) == {"depth", "mesh"}
    base = {
        "schema_version": protocol["submission_schema_version"],
        "protocol_version": protocol["protocol_version"],
        "track": "rgb-oriented",
        "scene_id": "Dataset-1",
        "product": "pointcloud",
        "geometry": "pointcloud.las",
        "configuration_id": "fixture-config",
    }
    benchmark._validate_submission_contract(base, protocol, "rgb-oriented", "Dataset-1")
    for field, value, reason in (
        ("product", "depth", "UNSUPPORTED_PRODUCT"),
        ("product", "mesh", "UNSUPPORTED_PRODUCT"),
        ("geometry", "../pointcloud.las", "GEOMETRY_MISMATCH"),
        ("camera_centers", "camera_centers.csv", "CAMERA_CENTERS_NOT_ALLOWED"),
        ("hidden_scoring_knob", 1, "UNKNOWN_SUBMISSION_FIELD"),
    ):
        contract = {**base, field: value}
        with pytest.raises(benchmark.InvalidInput, match=reason):
            benchmark._validate_submission_contract(
                contract, protocol, "rgb-oriented", "Dataset-1"
            )
    local = {**base, "track": "rgb-local"}
    with pytest.raises(benchmark.InvalidInput, match="CAMERA_CENTERS_REQUIRED"):
        benchmark._validate_submission_contract(local, protocol, "rgb-local", "Dataset-1")
    result = {"status": "invalid", "invalid_reasons": [{"code": "UNSUPPORTED_PRODUCT"}]}
    assert "metrics" not in result
    json.dumps(result, allow_nan=False)


def test_cli_invalid_is_strict_json_and_exit_two(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "verify-bundle",
            "--bundle",
            str(tmp_path / "missing"),
            "--bundle-lock",
            str(tmp_path / "missing.lock.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "invalid"
    assert "metrics" not in payload
    assert "NaN" not in completed.stdout and "Infinity" not in completed.stdout


def test_score_invalid_atomically_writes_no_metrics(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    submission_json = {
        "schema_version": "usegeo-submission-1.1",
        "protocol_version": "usegeo-pointcloud-v1",
        "track": "rgb-oriented",
        "scene_id": "Dataset-1",
        "product": "mesh",
        "geometry": "pointcloud.las",
        "configuration_id": "fixture-config",
    }
    (submission / "submission.json").write_text(json.dumps(submission_json))
    result_dir = tmp_path / "result"
    code, payload = benchmark.score_submission(
        tmp_path / "missing-bundle",
        tmp_path / "missing.lock.json",
        "rgb-oriented",
        "Dataset-1",
        submission,
        result_dir,
        ["benchmark.py", "score"],
    )
    assert code == 2 and payload["status"] == "invalid"
    assert payload["product"] == "mesh"
    assert "metrics" not in payload
    assert payload["submission_contract_sha256"] == hashlib.sha256(
        (submission / "submission.json").read_bytes()
    ).hexdigest()
    assert json.loads((result_dir / "score.json").read_text()) == payload
    run = json.loads((result_dir / "run_manifest.json").read_text())
    assert run["status"] == "invalid" and run["output_sha256"]["score.json"]


def test_score_uses_private_snapshot_and_refuses_partial_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, bundle_lock, _, observed = install_synthetic_scoring(monkeypatch, tmp_path)
    submission = make_submission(tmp_path / "submission", "Dataset-1", "fixed-a")
    original_hash = benchmark.sha256_file(submission / "pointcloud.las")
    original_snapshot = benchmark._snapshot_submission

    def snapshot_then_mutate(*args, **kwargs):
        value = original_snapshot(*args, **kwargs)
        write_las(
            submission / "pointcloud.las",
            np.array([[0.0, 0.0, 20.0], [1.0, 0.0, 20.0], [0.0, 1.0, 20.0]]),
        )
        return value

    monkeypatch.setattr(benchmark, "_snapshot_submission", snapshot_then_mutate)
    result = tmp_path / "result"
    code, score = benchmark.score_submission(
        bundle,
        bundle_lock,
        "rgb-oriented",
        "Dataset-1",
        submission,
        result,
        ["benchmark.py", "score"],
    )
    assert code == 0 and score["status"] == "valid"
    assert score["submission_geometry_sha256"] == original_hash
    assert benchmark.sha256_file(submission / "pointcloud.las") != original_hash
    assert np.allclose(observed[0][:, 2], 0.0)

    (result / "run_manifest.json").unlink()
    with pytest.raises(benchmark.InvalidInput, match="RESULT_BUNDLE_FILE_SET_MISMATCH"):
        benchmark.score_submission(
            bundle,
            bundle_lock,
            "rgb-oriented",
            "Dataset-1",
            submission,
            result,
            ["benchmark.py", "score"],
        )


def test_aggregate_valid_series_rejects_mixed_and_forged_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, bundle_lock, _, _ = install_synthetic_scoring(monkeypatch, tmp_path)
    results = []
    for scene in ("Dataset-1", "Dataset-2", "Dataset-3"):
        submission = make_submission(tmp_path / f"submission-{scene}", scene, "fixed-a")
        result = tmp_path / f"result-{scene}"
        code, score = benchmark.score_submission(
            bundle,
            bundle_lock,
            "rgb-oriented",
            scene,
            submission,
            result,
            ["benchmark.py", "score"],
        )
        assert code == 0 and score["status"] == "valid"
        results.append(result)

    aggregate = benchmark.aggregate_scores(
        results, bundle, bundle_lock, "rgb-oriented", "fixed-a"
    )
    assert aggregate["status"] == "valid"
    assert aggregate["configuration_id"] == "fixed-a"
    assert set(aggregate["submission_contract_sha256"]) == {
        "Dataset-1",
        "Dataset-2",
        "Dataset-3",
    }

    mixed = tmp_path / "mixed-result"
    mixed.mkdir()
    mixed_score = json.loads((results[2] / "score.json").read_text())
    mixed_run = json.loads((results[2] / "run_manifest.json").read_text())
    mixed_score["configuration_id"] = "fixed-b"
    mixed_run["configuration_id"] = "fixed-b"
    benchmark._write_json(mixed / "score.json", mixed_score)
    mixed_run["output_sha256"] = {
        "score.json": benchmark.sha256_file(mixed / "score.json")
    }
    benchmark._write_json(mixed / "run_manifest.json", mixed_run)
    with pytest.raises(benchmark.InvalidInput, match="AGGREGATE_CONFIGURATION_MISMATCH"):
        benchmark.aggregate_scores(
            [results[0], results[1], mixed],
            bundle,
            bundle_lock,
            "rgb-oriented",
            "fixed-a",
        )

    forged = tmp_path / "forged-result"
    forged.mkdir()
    forged_score = json.loads((results[2] / "score.json").read_text())
    forged_run = json.loads((results[2] / "run_manifest.json").read_text())
    forged_score["protocol_version"] = "foreign-protocol"
    benchmark._write_json(forged / "score.json", forged_score)
    forged_run["output_sha256"] = {
        "score.json": benchmark.sha256_file(forged / "score.json")
    }
    benchmark._write_json(forged / "run_manifest.json", forged_run)
    with pytest.raises(benchmark.InvalidInput, match="RESULT_SCORE_RELATION_MISMATCH"):
        benchmark.aggregate_scores(
            [results[0], results[1], forged],
            bundle,
            bundle_lock,
            "rgb-oriented",
            "fixed-a",
        )


def test_external_bundle_lock_and_manifest_relations_fail_closed(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    protocol = benchmark.load_protocol()
    lock = {
        "schema_version": benchmark.BUNDLE_LOCK_SCHEMA_VERSION,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": benchmark.sha256_file(benchmark.PROTOCOL_PATH),
        "bundle_manifest_sha256": benchmark.sha256_file(bundle / "manifest.json"),
        "archive_sha256": protocol["archive_sha256"],
        "upstream_files_sha256": protocol["upstream"]["files_sha256"],
        "scenes": {},
    }
    lock_path = tmp_path / "bundle.lock.json"
    benchmark._write_json(lock_path, lock)
    assert benchmark._load_bundle_lock(bundle, lock_path, protocol) == lock

    lock["upstream_files_sha256"] = {}
    benchmark._write_json(lock_path, lock)
    with pytest.raises(benchmark.InvalidInput, match="BUNDLE_LOCK_AUTHORITY_MISMATCH"):
        benchmark._load_bundle_lock(bundle, lock_path, protocol)

    tiny_root = tmp_path / "tiny"
    tiny_root.mkdir()
    archive, tiny_protocol = tiny_archive(tiny_root)
    prepared = tmp_path / "prepared"
    benchmark._prepare_archives(archive.parent, prepared, tiny_protocol, min_free_bytes=0)
    tiny_bundle = prepared / "v1"
    manifest_path = tiny_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scenes"]["Tiny"]["image_count"] = 0
    benchmark._write_json(manifest_path, manifest)
    with pytest.raises(benchmark.InvalidInput, match="SCENE_MANIFEST_RELATION_MISMATCH"):
        benchmark._verify_bundle_manifest(
            tiny_bundle,
            tiny_protocol,
            trusted_manifest_sha256=benchmark.sha256_file(manifest_path),
            expected_protocol_sha=hashlib.sha256(
                benchmark._json_bytes(tiny_protocol)
            ).hexdigest(),
        )


def test_vendored_sources_match_protocol_pins() -> None:
    benchmark.verify_vendor_sources(benchmark.load_protocol())
