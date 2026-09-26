"""Deterministic regression suite for the isolated mesh benchmark."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from usegeo_mesh_benchmark import benchmark as mesh
from usegeo_mesh_benchmark import official_compat as official
from usegeo_mesh_benchmark.tests import real_acceptance as acceptance


def square(z: float = 0.0, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array([
        [offset, 0, z], [offset + 1, 0, z], [offset + 1, 1, z], [offset, 1, z]
    ], dtype=np.float64)
    return vertices, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def grid_reference(z: float = 0.0, count: int = 21, offset: float = 0.0) -> np.ndarray:
    axis = np.linspace(0, 1, count)
    x, y = np.meshgrid(axis + offset, axis)
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


class PlyTests(unittest.TestCase):
    def test_round_trip_and_structure(self) -> None:
        vertices, triangles = square()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mesh.ply"
            mesh.write_ply(path, vertices, triangles)
            actual_vertices, actual_triangles = mesh.read_ply(path)
        np.testing.assert_array_equal(actual_vertices, vertices)
        np.testing.assert_array_equal(actual_triangles, triangles)
        diagnostics = mesh.structure_diagnostics(vertices, triangles)
        self.assertEqual(diagnostics["boundary_edge_count"], 4)
        self.assertEqual(diagnostics["connected_component_count"], 1)
        self.assertAlmostEqual(diagnostics["boundary_length_m"], 4.0)

    def test_open_multicomponent_is_valid(self) -> None:
        left, faces = square()
        right, _ = square(offset=3)
        vertices = np.vstack((left, right))
        triangles = np.vstack((faces, faces + 4))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mesh.ply"
            mesh.write_ply(path, vertices, triangles)
            mesh.read_ply(path)
        diagnostics = mesh.structure_diagnostics(vertices, triangles)
        self.assertEqual(diagnostics["connected_component_count"], 2)
        self.assertEqual(diagnostics["boundary_edge_count"], 8)

    def test_invalid_empty_nonfinite_index_and_degenerate(self) -> None:
        cases: list[tuple[np.ndarray, np.ndarray, str]] = [
            (np.empty((0, 3)), np.empty((0, 3), dtype=int), "EMPTY_GEOMETRY"),
            (np.array([[0., 0., 0.], [1., 0., 0.], [np.nan, 1., 0.]]), np.array([[0, 1, 2]]), "NONFINITE_GEOMETRY"),
            (np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), np.array([[0, 1, 3]]), "INDEX_OUT_OF_RANGE"),
            (np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), np.array([[0, 1, 1]]), "DEGENERATE_TRIANGLE"),
            (np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), np.array([[0, 1, 2]]), "DEGENERATE_TRIANGLE"),
        ]
        for vertices, triangles, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "mesh.ply"
                mesh.write_ply(path, vertices, triangles)
                with self.assertRaises(mesh.InvalidInput) as caught:
                    mesh.read_ply(path)
                self.assertEqual(caught.exception.reason, reason)

    def test_malformed_header_payload_and_trailing_bytes(self) -> None:
        vertices, triangles = square()
        with tempfile.TemporaryDirectory() as temporary:
            good = Path(temporary) / "good.ply"
            mesh.write_ply(good, vertices, triangles)
            for name, data, reason in (
                ("header.ply", good.read_bytes().replace(b"binary_little_endian", b"ascii               ", 1), "PLY_FORMAT_UNSUPPORTED"),
                ("short.ply", good.read_bytes()[:-1], "PLY_SIZE_MISMATCH"),
                ("long.ply", good.read_bytes() + b"x", "PLY_SIZE_MISMATCH"),
            ):
                path = Path(temporary) / name
                path.write_bytes(data)
                with self.assertRaises(mesh.InvalidInput) as caught:
                    mesh.read_ply(path)
                self.assertEqual(caught.exception.reason, reason)


class MetricTests(unittest.TestCase):
    def test_perfect_plane_and_discrete_accuracy(self) -> None:
        vertices, triangles = square()
        reference = grid_reference(count=11)
        metrics, origin = mesh.metrics_from_arrays(
            vertices, triangles, reference, reference, sample_count=5000
        )
        self.assertEqual(metrics["completeness_0_20"], 1.0)
        self.assertGreater(metrics["accuracy_l1_m"], 0.0)
        self.assertGreater(metrics["accuracy_rmse_m"], 0.0)
        direct = mesh.raycast_distances(vertices, triangles, reference, origin)
        self.assertLessEqual(float(direct.max()), 1e-6)

        compat, _ = official.compatibility_from_arrays(
            vertices, triangles, np.array([[0.75, 0.25, 0.0], [0.25, 0.75, 0.0]]),
            reference, timeout_seconds=30,
        )
        self.assertEqual(compat["completeness_0_20"], 1.0)
        self.assertTrue(compat["all_triangle_distances_computed"])
        self.assertAlmostEqual(compat["triangle_accuracy_l1_m"], 0.0, places=6)

    def test_offset_worsens_both_independent_paths(self) -> None:
        vertices, triangles = square()
        shifted = vertices.copy(); shifted[:, 2] += 0.25
        reference = grid_reference(count=11)
        base, _ = mesh.metrics_from_arrays(vertices, triangles, reference, reference, sample_count=4000)
        offset, _ = mesh.metrics_from_arrays(shifted, triangles, reference, reference, sample_count=4000)
        self.assertEqual(offset["completeness_0_20"], 0.0)
        self.assertGreater(offset["accuracy_l1_m"], base["accuracy_l1_m"])
        official_base, _ = official.compatibility_from_arrays(
            vertices, triangles, np.array([[.75, .25, 0], [.25, .75, 0]]), reference, timeout_seconds=30)
        official_offset, _ = official.compatibility_from_arrays(
            shifted, triangles, np.array([[.75, .25, 0], [.25, .75, 0]]), reference, timeout_seconds=30)
        self.assertGreater(official_offset["triangle_accuracy_l1_m"], official_base["triangle_accuracy_l1_m"])
        self.assertEqual(official_offset["completeness_0_20"], 0.0)

    def test_missing_patch_is_completeness_not_precision(self) -> None:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [.5, 0, 0], [.5, 1, 0]], float)
        triangles = np.array([[0, 4, 5], [0, 5, 3]], int)  # left half only
        refined = np.array([[.25, .25, 0], [.25, .75, 0], [.75, .25, 0], [.75, .75, 0]])
        full = grid_reference(count=41)
        metrics, _ = mesh.metrics_from_arrays(vertices, triangles, full, refined, sample_count=5000)
        self.assertEqual(metrics["completeness_0_20"], 0.5)
        self.assertEqual(metrics["precision_0_20"], 1.0)

    def test_floating_wrong_area_hurts_untrimmed_metrics(self) -> None:
        base_vertices, base_faces = square()
        wrong_vertices = np.array([[0, 0, 1], [2, 0, 1], [2, 1, 1], [0, 1, 1]], float)
        vertices = np.vstack((base_vertices, wrong_vertices))
        triangles = np.vstack((base_faces, base_faces + 4))
        full = grid_reference(count=31)
        refined = grid_reference(count=11)
        metrics, _ = mesh.metrics_from_arrays(vertices, triangles, full, refined, sample_count=20000)
        self.assertGreater(metrics["accuracy_l1_m"], 0.5)
        self.assertLess(metrics["precision_0_20"], 0.5)
        self.assertIn("accuracy_best90_l1_m", metrics)
        self.assertEqual(metrics["accuracy_best90_count"], 18000)

    def test_retriangulation_and_large_utm_precision(self) -> None:
        vertices, first = square(offset=500000.0)
        second = np.array([[0, 1, 3], [1, 2, 3]], int)
        reference = grid_reference(count=17, offset=500000.0)
        one, origin = mesh.metrics_from_arrays(vertices, first, reference, reference)
        two, other_origin = mesh.metrics_from_arrays(vertices, second, reference, reference)
        np.testing.assert_array_equal(origin, other_origin)
        self.assertLessEqual(abs(one["accuracy_l1_m"] - two["accuracy_l1_m"]), 0.001)
        self.assertLessEqual(abs(one["accuracy_rmse_m"] - two["accuracy_rmse_m"]), 0.001)
        self.assertLessEqual(abs(one["precision_0_20"] - two["precision_0_20"]), 0.002)
        self.assertLessEqual(abs(one["completeness_0_20"] - two["completeness_0_20"]), 1e-6)
        distances = mesh.raycast_distances(vertices, first, reference, origin)
        self.assertLessEqual(float(distances.max()), 1e-5)

    def test_exact_triangle_query_full_envelope_boundaries_and_threshold(self) -> None:
        base = np.array([498500.0, 4379000.0, -100.0])
        narrow = base + np.array([[699.0, 449.0, 399.0], [700.0, 449.0, 399.003],
                                  [699.0, 449.02, 399.001]])
        wide = base + np.array([[0.0, 0.0, 0.0], [700.0, 0.0, 2.1], [0.0, 450.0, 0.9]])
        normal = np.cross(narrow[1] - narrow[0], narrow[2] - narrow[0])
        normal /= np.linalg.norm(normal)
        competitor = narrow + normal * .00002
        vertices = np.vstack((wide, narrow, competitor))
        triangles = np.arange(9).reshape(3, 3)
        interior = .2 * narrow[0] + .3 * narrow[1] + .5 * narrow[2]
        wide_interior = .2 * wide[0] + .3 * wide[1] + .5 * wide[2]
        queries = np.vstack((narrow, (narrow[0] + narrow[1]) / 2, interior, wide_interior,
                             interior + normal * .19999, interior + normal * .20003))
        expected = np.array([0, 0, 0, 0, 0, 0, .19997, .20001])
        actual = mesh.raycast_distances(vertices, triangles, queries, base)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
        self.assertLessEqual(actual[6], .20)
        self.assertGreater(actual[7], .20)
        rng = np.random.Generator(np.random.PCG64(20260916))
        bary = rng.dirichlet(np.ones(3), size=10000)
        surface = np.vstack((bary @ narrow, bary @ wide))
        oracle = mesh.raycast_distances(vertices, triangles, surface, base)
        self.assertLessEqual(float(oracle.max()), 1e-5)

    def test_exact_vertex_survives_utm_broad_phase_rounding(self) -> None:
        rng = np.random.Generator(np.random.PCG64(917))
        base = np.array([498500.0, 4379000.0, -100.0])
        faces = []
        for _ in range(32):
            anchor = base + rng.uniform([0, 0, 0], [700, 450, 400])
            faces.append(anchor + rng.normal(size=(3, 3)) * rng.uniform(.03, 700))
        vertices = np.vstack(faces)
        triangles = np.arange(len(vertices)).reshape(-1, 3)
        query = vertices[40:41]
        distance, primitive = mesh.exact_triangle_queries(vertices, triangles, query, base)
        self.assertEqual(int(primitive[0]), 13)
        self.assertLessEqual(float(distance[0]), 1e-5)


class OfficialRepairTests(unittest.TestCase):
    def test_all_equal_exact_ceil_and_index_stable(self) -> None:
        selected, count = official.stable_best90(np.zeros(11))
        self.assertEqual(count, 10)
        np.testing.assert_array_equal(selected, np.zeros(10))

    def test_boolean_status_is_independent_of_numeric_zero(self) -> None:
        calls = 0
        def pass_fn(remaining: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nonlocal calls
            calls += 1
            return remaining[:1], np.zeros(1)
        distances, iterations = official.repaired_pass_loop(3, pass_fn)
        np.testing.assert_array_equal(distances, np.zeros(3))
        self.assertEqual(iterations, 3)
        self.assertEqual(calls, 3)

    def test_invalid_no_progress_pass_and_time_bounds(self) -> None:
        with self.assertRaisesRegex(mesh.EvaluatorFailure, "OFFICIAL_NO_PROGRESS"):
            official.repaired_pass_loop(1, lambda _: (np.array([], int), np.array([])))
        with self.assertRaisesRegex(mesh.EvaluatorFailure, "OFFICIAL_INVALID_PRIMITIVE_ID"):
            official.repaired_pass_loop(1, lambda _: (np.array([3]), np.array([0.])))
        with self.assertRaisesRegex(mesh.EvaluatorFailure, "OFFICIAL_NONFINITE_DISTANCE"):
            official.repaired_pass_loop(1, lambda _: (np.array([0]), np.array([np.inf])))
        with self.assertRaisesRegex(mesh.EvaluatorFailure, "OFFICIAL_PASS_LIMIT"):
            official.repaired_pass_loop(2, lambda r: (r[:1], np.zeros(1)), max_passes=1)
        with self.assertRaisesRegex(mesh.EvaluatorFailure, "OFFICIAL_TIME_LIMIT"):
            official.repaired_pass_loop(1, lambda r: (r, np.zeros(1)), deadline=time.monotonic() - 1)


class IsolationAndAggregationTests(unittest.TestCase):
    def _submission(self, root: Path, metadata: dict[str, str]) -> Path:
        root.mkdir()
        vertices, triangles = square()
        mesh.write_ply(root / "mesh.ply", vertices, triangles)
        (root / "submission.json").write_text(json.dumps({
            "schema_version": "usegeo-mesh-submission-1.0",
            "protocol_version": "usegeo-mesh-rgb-oriented-v1", "track": "rgb-oriented",
            "scene_id": "Dataset-1", "product": "mesh", "geometry": "mesh.ply",
            "configuration_id": "same-v1", "method_metadata": metadata,
        }), encoding="utf-8")
        return root

    def test_label_neutral_snapshot_and_symlink_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._submission(root / "first", {"family": "Agent", "name": "a"})
            second = self._submission(root / "second", {"family": "non-Agent", "name": "b"})
            snap_a = mesh.snapshot_submission(first, root / "snap-a", "Dataset-1")
            snap_b = mesh.snapshot_submission(second, root / "snap-b", "Dataset-1")
            self.assertEqual(snap_a["mesh_sha256"], snap_b["mesh_sha256"])
            original = mesh.read_ply(root / "snap-a" / "mesh.ply")
            (first / "mesh.ply").write_bytes(b"mutated")
            frozen = mesh.read_ply(root / "snap-a" / "mesh.ply")
            np.testing.assert_array_equal(original[0], frozen[0])
            unsafe = root / "unsafe"; unsafe.mkdir()
            (unsafe / "submission.json").symlink_to(second / "submission.json")
            (unsafe / "mesh.ply").symlink_to(second / "mesh.ply")
            with self.assertRaises(mesh.InvalidInput):
                mesh.snapshot_submission(unsafe, root / "unsafe-snap", "Dataset-1")

    def _result(
        self, root: Path, scene: str, config: str = "same-v1", finite: bool = True,
        authority: dict[str, str] | None = None,
    ) -> Path:
        root.mkdir()
        value = 1.0 if finite else float("inf")
        bindings = {**(authority or {"protocol_sha256": "a" * 64, "bundle_lock_sha256": "b" * 64,
                    "bundle_manifest_sha256": "c" * 64, "input_manifest_sha256": "f" * 64,
                    "reference_manifest_sha256": "1" * 64}), "submission_contract_sha256": "d" * 64,
                    "mesh_sha256": "e" * 64,
                    "runtime_implementation_id": mesh.load_protocol()["runtime"]["implementation_id"]}
        score = {
            "schema_version": mesh.SCORE_SCHEMA, "protocol_version": mesh.load_protocol()["protocol_version"],
            "status": "valid", "scene_id": scene, "track": "rgb-oriented", "configuration_id": config,
            "alignment": {"kind": "identity", "raycast_translation_kind": mesh.load_protocol()["raycast"]["origin_rule"],
                          "raycast_origin_m": [0.0, 0.0, 0.0]},
            "counts": {"vertices": 4, "triangles": 2, "surface_samples": 1000000,
                       "full_lidar_points": 4, "refined_lidar_points": 4},
            "surface_metrics": {**{name: value for name in mesh.METRIC_NAMES},
                                "accuracy_best90_count": 900000},
            "structure_diagnostics": {"surface_area_m2": 1.0, "degenerate_triangle_count": 0,
                                      "unreferenced_vertex_count": 0, "boundary_edge_count": 4,
                                      "boundary_length_m": 4.0, "nonmanifold_edge_count": 0,
                                      "connected_component_count": 1}, "bindings": bindings,
        }
        mesh._write_json(root / "score.json", score)
        protocol = mesh.load_protocol()
        run = {"schema_version": mesh.RUN_SCHEMA, "status": "valid",
               "protocol_version": score["protocol_version"], "started_at": "2026-09-16T00:00:00Z",
               "finished_at": "2026-09-16T00:00:01Z", "elapsed_seconds": 1.0,
               "peak_worker_rss_bytes": 1, "watchdog": {
                   "period_seconds": protocol["limits"]["watchdog_period_seconds"],
                   "timeout_seconds": protocol["limits"]["scene_timeout_seconds"],
                   "rss_limit_bytes": protocol["limits"]["max_peak_rss_bytes"], "exit_code": 0},
               "sampling": protocol["sampling"], "python": "test", "platform": "test",
               "packages": mesh.runtime_versions(), "cli_arguments": [], "method_metadata": {},
               "bindings": bindings,
               "output_sha256": {"score.json": mesh.sha256_file(root / "score.json")}}
        mesh._write_json(root / "run_manifest.json", run)
        return root

    @mock.patch.object(mesh, "verify_bundle", return_value={})
    def test_aggregate_happy_and_rejections(self, _verify: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = ("Dataset-1", "Dataset-2", "Dataset-3")
            authority = {"scenes": {scene: {"input_manifest_sha256": {"rgb-oriented": "f" * 64},
                                             "reference_manifest_sha256": "1" * 64} for scene in scenes}}
            mesh._write_json(root / "manifest.json", authority)
            (root / "lock").write_text("lock", encoding="utf-8")
            common = {"protocol_sha256": mesh.sha256_file(mesh.PROTOCOL_PATH),
                      "bundle_lock_sha256": mesh.sha256_file(root / "lock"),
                      "bundle_manifest_sha256": mesh.sha256_file(root / "manifest.json"),
                      "input_manifest_sha256": "f" * 64, "reference_manifest_sha256": "1" * 64}
            paths = [self._result(root / scene, scene, authority=common) for scene in scenes]
            _verify.return_value = authority
            aggregate = mesh.aggregate_scores(root, root / "lock", "same-v1", paths, root / "aggregate.json")
            self.assertEqual(aggregate["scene_ids"], ["Dataset-1", "Dataset-2", "Dataset-3"])
            self.assertEqual(set(aggregate["macro_mean"]), set(mesh.METRIC_NAMES))
            with self.assertRaises(mesh.InvalidInput):
                mesh.aggregate_scores(root, root / "lock", "same-v1", paths[:2] + [paths[1]], root / "x.json")
            changed = self._result(root / "changed", "Dataset-3", config="other-v1", authority=common)
            with self.assertRaises(mesh.InvalidInput):
                mesh.aggregate_scores(root, root / "lock", "same-v1", paths[:2] + [changed], root / "y.json")
            impossible = self._result(root / "impossible", "Dataset-3", authority=common)
            bad_score = mesh._strict_json(impossible / "score.json", "test")
            bad_score["surface_metrics"]["precision_0_20"] = -0.25
            mesh._write_json(impossible / "score.json", bad_score)
            bad_run = mesh._strict_json(impossible / "run_manifest.json", "test")
            bad_run["output_sha256"]["score.json"] = mesh.sha256_file(impossible / "score.json")
            mesh._write_json(impossible / "run_manifest.json", bad_run)
            with self.assertRaises(mesh.InvalidInput):
                mesh.aggregate_scores(root, root / "lock", "same-v1", paths[:2] + [impossible], root / "bad.json")
            forged = self._result(root / "forged", "Dataset-3")
            with self.assertRaises(mesh.InvalidInput):
                mesh.aggregate_scores(root, root / "lock", "same-v1", paths[:2] + [forged], root / "forged.json")
            (paths[0] / "score.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(mesh.InvalidInput):
                mesh.aggregate_scores(root, root / "lock", "same-v1", paths, root / "z.json")

    def test_runtime_cancel_rss_and_failure_artifacts(self) -> None:
        with mock.patch.object(mesh.importlib.metadata, "version", side_effect=lambda name: "0" if name == "numpy" else mesh.load_protocol()["runtime"][name]):
            with self.assertRaisesRegex(mesh.EvaluatorFailure, "frozen runtime mismatch"):
                mesh.runtime_versions()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cancel = root / "cancel.json"
            mesh._write_json(cancel, {"reason": "rss"})
            with self.assertRaisesRegex(mesh.EvaluatorFailure, "cancelled"):
                mesh._check_worker_limits(math.inf, cancel)

            class Process:
                pid = 1
                returncode = None
                def poll(self) -> None: return None
                def wait(self, timeout: float | None = None) -> int:
                    self.returncode = 1
                    return 1
                def communicate(self) -> tuple[str, str]: return "", ""
                def kill(self) -> None: self.returncode = -9

            unavailable = root / "unavailable.json"
            with mock.patch.object(mesh, "_peak_rss_bytes", return_value=None), mock.patch.object(mesh.time, "sleep"):
                with self.assertRaisesRegex(mesh.EvaluatorFailure, "measurement unavailable"):
                    mesh._watch_process(Process(), time.monotonic(), 30, 100, 0.01, unavailable)
            self.assertTrue(unavailable.is_file())

            with mock.patch.object(mesh, "verify_bundle", side_effect=mesh.EvaluatorFailure("forced")):
                code, result = mesh.score_submission(root, root / "lock", "Dataset-1", root, root / "formal-error")
            self.assertEqual((code, result["status"]), (1, "error"))
            self.assertEqual(set((root / "formal-error").iterdir()), {
                root / "formal-error" / "score.json", root / "formal-error" / "run_manifest.json"})

            with mock.patch.object(mesh, "verify_bundle", side_effect=mesh.EvaluatorFailure("forced")):
                code, result = official.run_official_compat(
                    root, root / "lock", "Dataset-1", root, root / "official-error")
            self.assertEqual((code, result["status"]), (1, "not_completed"))

    @mock.patch.object(mesh, "verify_bundle", return_value={})
    def test_empty_fixture_reuse_rejected(self, _verify: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "fixtures"; output.mkdir()
            mesh._write_json(output / "fixture_manifest.json", {"scenes": {}})
            with self.assertRaises(mesh.InvalidInput):
                mesh.generate_real_fixtures(root, root / "lock", output,
                                            mesh.load_protocol()["fixture"]["configuration_id"])

    @mock.patch.object(mesh, "verify_bundle", return_value={})
    def test_fixture_reuse_rederives_source_not_adjacent_hashes(self, _verify: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "lock").write_text("lock", encoding="utf-8")
            for scene in mesh.load_protocol()["scenes"]:
                directory = root / "evaluator-only" / scene
                directory.mkdir(parents=True)
                mesh._write_json(directory / "reference_manifest.json", {"files": {"publisher_mvs": {"sha256": "a" * 64}}})
            output = root / "fixtures"
            vertices, triangles = square()
            stats = {"publisher_mvs_point_count": 4, "xy_bounds": [[0., 0.], [1., 1.]],
                     "grid_origin_m": [0., 0.], "nonempty_cell_count": 4}
            with mock.patch.object(mesh, "_fixture_scene", return_value=(vertices, triangles, stats)):
                mesh.generate_real_fixtures(root, root / "lock", output, mesh.load_protocol()["fixture"]["configuration_id"])
                altered = vertices.copy(); altered[:, 2] = 9
                (output / "Dataset-1" / "mesh.ply").unlink()
                mesh.write_ply(output / "Dataset-1" / "mesh.ply", altered, triangles)
                manifest = mesh._strict_json(output / "fixture_manifest.json", "test")
                manifest["scenes"]["Dataset-1"]["mesh_sha256"] = mesh.sha256_file(output / "Dataset-1" / "mesh.ply")
                mesh._write_json(output / "fixture_manifest.json", manifest)
                with self.assertRaisesRegex(mesh.InvalidInput, "FIXTURE_DERIVATION_MISMATCH"):
                    mesh.generate_real_fixtures(root, root / "lock", output,
                                                mesh.load_protocol()["fixture"]["configuration_id"])

    def test_failed_acceptance_output_is_retryable_without_deleting_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acceptance"
            acceptance.prepare_output(output)
            completed = output / "formal" / "pass-1" / "Dataset-1" / "score.json"
            completed.parent.mkdir(parents=True)
            completed.write_text("preserved", encoding="utf-8")
            for name in ("acceptance-error.json", "protected-after.json", "execution-state.json"):
                mesh._write_json(output / name, {"status": "failed"})
            acceptance.prepare_output(output)
            self.assertEqual(completed.read_text(encoding="utf-8"), "preserved")
            archived = output / "attempt-history" / "attempt-0001"
            self.assertEqual({item.name for item in archived.iterdir()}, {
                "acceptance-error.json", "protected-after.json", "execution-state.json"})

    def test_interrupted_owner_is_recovered_without_stealing_live_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acceptance"; output.mkdir()
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                start = acceptance._process_start_token(child.pid)
                self.assertIsNotNone(start)
                mesh._write_json(output / "execution-state.json", {"status": "running", "owner": {"pid": child.pid, "start": start}})
                with self.assertRaises(mesh.EvaluatorFailure):
                    acceptance.prepare_output(output, Path.cwd())
                child.kill(); child.wait()
                completed = output / "formal/pass-1/Dataset-1/score.json"
                completed.parent.mkdir(parents=True); completed.write_text("preserved", encoding="utf-8")
                acceptance.prepare_output(output, Path.cwd())
                self.assertEqual(completed.read_text(encoding="utf-8"), "preserved")
                self.assertTrue((output / "attempt-history/attempt-0001/protected-after.json").is_file())
            finally:
                if child.poll() is None:
                    child.kill(); child.wait()

    def test_real_cli_rejection_preserves_every_output_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "acceptance"
            output.mkdir()
            mesh._write_json(output / "execution-state.json", {
                "status": "running", "owner": {
                    "pid": os.getpid(),
                    "start": acceptance._process_start_token(os.getpid()),
                },
            })
            (output / "protected-after.json").write_bytes(b"existing evidence")
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            command = [sys.executable, str(Path(acceptance.__file__).resolve()),
                       "--bundle", temporary, "--bundle-lock", str(Path(temporary) / "missing"),
                       "--output", str(output)]
            result = subprocess.run(command, capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"live or unverified owner", result.stderr)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
            with acceptance.output_ownership(output):
                result = subprocess.run(command, capture_output=True, timeout=15)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"owned by another process", result.stderr)
                self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_unknown_or_reused_live_pid_never_proves_interruption(self) -> None:
        state = {"owner": {"pid": os.getpid(), "start": "different start"}}
        with mock.patch.object(acceptance, "_process_start_token", return_value=None):
            self.assertFalse(acceptance._owner_interrupted(state))
        self.assertFalse(acceptance._owner_interrupted(state))
        for pid in (False, 0, -999999):
            self.assertFalse(acceptance._owner_interrupted({"owner": {"pid": pid, "start": "x"}}))

    def test_resumed_score_must_bind_current_submission(self) -> None:
        score = {"scene_id": "Dataset-1", "configuration_id": "fixed", "bindings": {"mesh_sha256": "old"}}
        with mock.patch.object(mesh, "_load_result", return_value=(score, {})), \
                mock.patch.object(mesh, "sha256_file", return_value="new"), \
                mock.patch.object(mesh, "_bindings", return_value={"mesh_sha256": "new"}):
            with self.assertRaisesRegex(mesh.EvaluatorFailure, "does not match"):
                acceptance.load_bound_result(Path("result"), Path("bundle"), Path("lock"),
                                             "Dataset-1", Path("submission"), "fixed")

    def test_all_public_clis_reject_missing_authority_without_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            common = ["--bundle", str(root), "--bundle-lock", str(root / "absent.lock")]
            scene = ["--scene", "Dataset-1", "--submission", str(root / "submission")]
            for command in ("validate-submission", "score", "official-compat", "aggregate", "generate-real-fixtures"):
                with self.subTest(command=command):
                    output = root / command
                    module = "official_compat" if command == "official-compat" else "benchmark"
                    args = [sys.executable, "-m", f"usegeo_mesh_benchmark.{module}", command, *common]
                    if command in ("validate-submission", "score", "official-compat"):
                        args += scene
                    else:
                        args += ["--configuration-id", "fixed"]
                    if command != "validate-submission":
                        args += ["--output", str(output)]
                    if command == "aggregate":
                        args += ["--results", str(root / "a"), str(root / "b"), str(root / "c")]
                    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
                    self.assertNotEqual(result.returncode, 0)
                    payload = json.loads((result.stdout or result.stderr).strip().splitlines()[-1])
                    self.assertNotIn(payload["status"], ("valid", "formal_pass"))
                    self.assertNotIn("surface_metrics", payload)
                    if output.exists():
                        files = list(output.rglob("*.json")) if output.is_dir() else [output]
                        for file in files:
                            value = json.loads(file.read_text())
                            self.assertNotIn("surface_metrics", value)
                            self.assertNotEqual(value.get("status"), "valid")

    def test_summary_is_published_only_after_terminal_protection(self) -> None:
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary).resolve() / "acceptance"
                before = {"status": "valid", "observed": {"example": "original"}}

                def candidate(*_args):
                    mesh._write_json(output / "execution-state.json", {
                        "status": "running", "owner": {
                            "pid": os.getpid(), "start": acceptance._process_start_token(os.getpid())}})
                    mesh._write_json(output / "protected-before.json", before)
                    mesh._write_json(output / "protected-after.json", before)
                    mesh._write_json(output / "acceptance-candidate.json", {
                        "status": "formal_pass",
                        "protected_after_sha256": mesh.sha256_file(output / "protected-after.json")})
                    return 0

                after = {"status": "invalid"} if changed else before
                with mock.patch.object(acceptance, "_run_attempt", side_effect=candidate), \
                        mock.patch.object(acceptance, "protected_report", return_value=after):
                    code = acceptance.run(Path(temporary), Path(temporary) / "lock", output)
                self.assertEqual(code, 1 if changed else 0)
                self.assertEqual((output / "acceptance-summary.json").exists(), not changed)
                state = json.loads((output / "execution-state.json").read_text())
                self.assertEqual(state["status"], "failed" if changed else "completed")


if __name__ == "__main__":
    unittest.main()
