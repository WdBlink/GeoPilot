"""Independent controlled acceptance; no full-scene data or production oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout, redirect_stderr
import io

import numpy as np

from usegeo_mesh_benchmark import benchmark as mesh
from usegeo_mesh_benchmark import official_compat as official
from usegeo_mesh_benchmark.tests import test_benchmark as fixtures


MEASUREMENTS = {}
N = 1_000_000
SEED = 20260916
FREQUENCY_BOUND = math.sqrt(math.log(2 / 1e-6) / (2 * N))


def oracle_distance(p, triangle):
    """Scalar Voronoi-region closest point, independent of query implementation."""
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        closest = a
    else:
        bp = p - b
        d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
        if d3 >= 0 and d4 <= d3:
            closest = b
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0 and d1 >= 0 and d3 <= 0:
                closest = a + d1 / (d1 - d3) * ab
            else:
                cp = p - c
                d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
                if d6 >= 0 and d5 <= d6:
                    closest = c
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0 and d2 >= 0 and d6 <= 0:
                        closest = a + d2 / (d2 - d6) * ac
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
                            closest = b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)
                        else:
                            closest = a + (vb * ab + vc * ac) / (va + vb + vc)
    return float(np.linalg.norm(p - closest))


def square(shift=(0., 0., 0.)):
    return (np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]) + shift,
            np.array([[0, 1, 2], [0, 2, 3]]))


def reference():
    x, y = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def score(v, f, ref):
    return mesh.metrics_from_arrays(v, f, ref, ref, sample_count=N, seed=SEED)[0]


class IndependentAcceptance(unittest.TestCase):
    def test_08_strict_contract_and_point_only_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v, _ = square()
            mesh.write_ply(root / 'points.ply', v, np.empty((0,3), int))
            with self.assertRaisesRegex(mesh.InvalidInput, 'EMPTY_GEOMETRY'):
                mesh.read_ply(root / 'points.ply')
            source = fixtures.IsolationAndAggregationTests()._submission(root / 'submission', {})
            contract = json.loads((source / 'submission.json').read_text())
            mutations = [('extra', True, 'UNKNOWN_FIELD'), ('geometry','../mesh.ply','UNSAFE_PATH')]
            for index, (key, value, expected) in enumerate(mutations):
                changed = dict(contract); changed[key] = value
                (source / 'submission.json').write_text(json.dumps(changed))
                with self.assertRaisesRegex(mesh.InvalidInput, expected):
                    mesh.snapshot_submission(source, root / f'snapshot-{index}', 'Dataset-1')
        MEASUREMENTS['A02-strict-supplement'] = {'point_only':'rejected', 'extra_field':'rejected', 'path_escape':'rejected'}

    def test_09_command_dispatch_failure_injections(self):
        observed = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for reason, patch_name in [('frozen runtime mismatch','runtime_versions'),
                                       ('reference hash mismatch','verify_bundle')]:
                for command in ('validate-submission','score','official-compat','aggregate','generate-real-fixtures'):
                    output = root / f'{patch_name}-{command}'
                    args = [command, '--bundle',str(root),'--bundle-lock',str(root/'lock')]
                    if command in ('validate-submission','score','official-compat'):
                        args += ['--scene','Dataset-1','--submission',str(root/'input')]
                    else:
                        args += ['--configuration-id','fixed']
                    if command != 'validate-submission':
                        args += ['--output',str(output)]
                    if command == 'aggregate':
                        args += ['--results',str(root/'a'),str(root/'b'),str(root/'c')]
                    stdout, stderr = io.StringIO(), io.StringIO()
                    entry = official.main if command == 'official-compat' else mesh.main
                    with mock.patch.object(mesh, patch_name, side_effect=mesh.EvaluatorFailure(reason)), \
                         redirect_stdout(stdout), redirect_stderr(stderr):
                        code = entry(args)
                    self.assertNotEqual(code, 0)
                    payload = json.loads((stdout.getvalue() or stderr.getvalue()).strip().splitlines()[-1])
                    self.assertNotEqual(payload['status'], 'valid')
                    self.assertNotIn('surface_metrics', payload)
                    artifacts = list(output.rglob('*.json')) if output.is_dir() else ([output] if output.exists() else [])
                    for path in artifacts:
                        value = json.loads(path.read_text())
                        self.assertNotEqual(value.get('status'), 'valid')
                        self.assertNotIn('surface_metrics', value)
                    observed.append({'command':command, 'injection':patch_name,'exit_code':code})
        MEASUREMENTS['A09-dispatch-injections'] = observed

    def test_10_independent_macro_and_invalid_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = ['Dataset-1','Dataset-2','Dataset-3']
            authority = {'scenes':{scene:{'input_manifest_sha256':{'rgb-oriented':'f'*64},
                                         'reference_manifest_sha256':'1'*64} for scene in scenes}}
            (root/'manifest.json').write_text(json.dumps(authority))
            (root/'lock').write_text('synthetic lock')
            common = {'protocol_sha256':mesh.sha256_file(mesh.PROTOCOL_PATH),
                'bundle_lock_sha256':mesh.sha256_file(root/'lock'),
                'bundle_manifest_sha256':mesh.sha256_file(root/'manifest.json'),
                'input_manifest_sha256':'f'*64,'reference_manifest_sha256':'1'*64}
            paths = [fixtures.IsolationAndAggregationTests()._result(root/scene,scene,authority=common) for scene in scenes]
            scores, runs = [], []
            for i, path in enumerate(paths):
                s, r = json.loads((path/'score.json').read_text()), json.loads((path/'run_manifest.json').read_text())
                for name in mesh.METRIC_NAMES:
                    s['surface_metrics'][name] = [1.,2.,4.][i] if name.startswith('accuracy_') else [.2,.4,.6][i]
                (path/'score.json').write_text(json.dumps(s))
                r['output_sha256']['score.json'] = mesh.sha256_file(path/'score.json')
                (path/'run_manifest.json').write_text(json.dumps(r))
                scores.append(s); runs.append(r)
            with mock.patch.object(mesh,'verify_bundle',return_value=authority):
                result = mesh.aggregate_scores(root,root/'lock','same-v1',paths,root/'aggregate.json')
                for key, actual in result['macro_mean'].items():
                    self.assertAlmostEqual(actual, 7/3 if key.startswith('accuracy_') else .4, places=12)
                for index, label in enumerate(('missing_scene','failed_scene','nonfinite','runtime_drift')):
                    s, r = json.loads(json.dumps(scores[0])), json.loads(json.dumps(runs[0]))
                    chosen = paths
                    if label == 'missing_scene': chosen = paths[:2]
                    if label == 'failed_scene': s['status'] = 'failed'
                    if label == 'nonfinite': s['surface_metrics']['accuracy_l1_m'] = float('nan')
                    if label == 'runtime_drift': r['packages']['numpy'] = 'wrong'
                    (paths[0]/'score.json').write_text(json.dumps(s))
                    r['output_sha256']['score.json'] = mesh.sha256_file(paths[0]/'score.json')
                    (paths[0]/'run_manifest.json').write_text(json.dumps(r))
                    target = root/f'invalid-{index}.json'
                    with self.assertRaises((mesh.InvalidInput, mesh.EvaluatorFailure)):
                        mesh.aggregate_scores(root,root/'lock','same-v1',chosen,target)
                    self.assertFalse(target.exists())
        MEASUREMENTS['A10-independent-macro'] = {'accuracy_mean':7/3, 'fraction_mean':.4,
            'rejected':['missing_scene','failed_scene','nonfinite','runtime_drift']}

    def test_01_exhaustive_independent_oracle(self):
        base = np.array([498500., 4379000., -100.])
        rng = np.random.default_rng(917)
        faces = []
        for _ in range(32):
            anchor = base + rng.uniform([0, 0, 0], [700, 450, 400])
            faces.append(anchor + rng.normal(size=(3, 3)) * rng.uniform(.03, 700))
        seed_vertices = np.vstack(faces)
        # Radius bins on both sides of a power-of-two boundary.
        boundary_faces = []
        for radius in [1. - 1e-8, 1., 1. + 1e-8, 2., 8.]:
            side = math.sqrt(2) * radius
            boundary_faces.append(base + np.array([[0, 0, 0], [side, 0, 0], [0, side, 0]]))
        faces.extend(boundary_faces)
        bounds = np.asarray(boundary_faces)
        actual_radii = np.linalg.norm(bounds.max(axis=1)-bounds.min(axis=1), axis=1)/2
        self.assertLess(actual_radii[0], 1.)
        self.assertGreater(actual_radii[2], 1.)
        wide = base + np.array([[0, 0, 0], [20000, 0, 0], [0, 20, 0]])
        small = base + np.array([[1, 1, .3], [2, 1, .3], [1, 2, .3]])
        faces.extend([wide, small])
        vertices = np.vstack(faces)
        triangles = np.arange(len(vertices)).reshape(-1, 3)
        queries = np.vstack((seed_vertices, base + rng.uniform(-100, 900, (100, 3)),
                             base + [[1.1, 1.1, .01], [20001, 0, 0], [0, 0, 100]]))
        expected = np.array([min(oracle_distance(p, face) for face in faces) for p in queries])
        errors = []
        for ordering in [triangles, triangles[::-1]]:
            actual = mesh.raycast_distances(vertices, ordering, queries, base, chunk=17)
            errors.append(float(np.max(np.abs(actual - expected))))
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
        self.assertLessEqual(mesh.raycast_distances(vertices, triangles, seed_vertices[40:41], base)[0], 1e-5)
        # Official pass entry over a single triangle has an analytic exhaustive minimum.
        off, _ = official.compatibility_from_arrays(wide, np.array([[0, 1, 2]]), queries, queries)
        off_expected = min(oracle_distance(p, wide) for p in queries)
        self.assertAlmostEqual(off['triangle_accuracy_l1_m'], off_expected, delta=1e-5)
        MEASUREMENTS['A06-oracle'] = dict(queries=len(queries), triangles=len(faces), max_error_m=max(errors),
                                        radius_boundary_actual_m=actual_radii.tolist())

    def test_02_planes_offset_vertical(self):
        v, f = square()
        r = reference()
        results = {}
        for orientation, perm in [('horizontal', [0, 1, 2]), ('vertical', [2, 0, 1])]:
            base = v[:, perm]
            ref = r[:, perm]
            normal = np.array([0., 0., 1.])[perm]
            good, bad = score(base, f, ref), score(base + .25 * normal, f, ref)
            self.assertEqual(good['completeness_0_20'], 1.)
            self.assertEqual(bad['completeness_0_20'], 0.)
            self.assertLessEqual(good['accuracy_rmse_m'], math.sqrt(2) * .025)
            self.assertGreaterEqual(bad['accuracy_l1_m'], .25 - 1e-12)
            self.assertGreater(bad['accuracy_l1_m'], good['accuracy_l1_m'])
            compat_good, _ = official.compatibility_from_arrays(base, f, ref, ref)
            compat_bad, _ = official.compatibility_from_arrays(base + .25 * normal, f, ref, ref)
            self.assertAlmostEqual(compat_good['triangle_accuracy_l1_m'], 0., delta=1e-6)
            self.assertAlmostEqual(compat_bad['triangle_accuracy_l1_m'], .25, delta=1e-6)
            self.assertEqual(compat_bad['completeness_0_20'], 0.)
            results[orientation] = dict(formal=good, offset=bad, official=compat_good, official_offset=compat_bad)
        MEASUREMENTS['A07-plane-A15-vertical'] = results

    def test_03_missing_and_floating(self):
        v, f = square()
        r = reference()
        two_ref = np.vstack((r, r + [3, 0, 0]))
        missing = score(v, f, two_ref)
        self.assertEqual(missing['completeness_0_20'], .5)
        good = score(v, f, r)
        floating = score(np.vstack((v, v + [0, 0, 1])), np.vstack((f, f + 4)), r)
        self.assertLessEqual(abs(floating['precision_0_20'] - .5), FREQUENCY_BOUND)
        for key in ['accuracy_l1_m', 'accuracy_rmse_m']:
            self.assertGreater(floating[key], good[key])
        MEASUREMENTS['A07-degradation'] = dict(missing=missing, floating=floating, frequency_bound=FREQUENCY_BOUND)

    def test_04_full_budget_retriangulation(self):
        v, f = square()
        r = reference()
        baseline = score(v, f, r)
        # Off-centre fan yields deliberately unequal triangle areas.
        other_v = np.vstack((v, [.013, .721, 0]))
        other_f = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        alternate = score(other_v, other_f, r)
        deltas = {k: abs(baseline[k] - alternate[k]) for k in baseline}
        for key in ['accuracy_l1_m', 'accuracy_rmse_m']:
            self.assertLessEqual(deltas[key], .001)
        self.assertLessEqual(deltas['precision_0_20'], .002)
        self.assertLessEqual(deltas['completeness_0_20'], 1e-6)
        MEASUREMENTS['A07-retriangulation'] = dict(sample_count=N, seed=SEED, deltas=deltas)

    def test_05_area_frequency_and_duplicate_semantics(self):
        v, f = square()
        large = v.copy(); large[:, 0] *= 3; large[:, 2] = 2
        samples = mesh.sample_surface(np.vstack((v, large)), np.vstack((f, f + 4)), N, SEED)
        observed = float(np.mean(samples[:, 2] < 1))
        self.assertLessEqual(abs(observed - .25), FREQUENCY_BOUND)
        repeated = mesh.structure_diagnostics(v, np.vstack((f, f)))
        self.assertAlmostEqual(repeated['surface_area_m2'], 2.)
        MEASUREMENTS['A07-area'] = dict(expected=.25, observed=observed, bound=FREQUENCY_BOUND,
                                       duplicate_semantics='Each submitted face contributes area; no deduplication.')

    def test_06_analytic_structure(self):
        v, f = square()
        open_d = mesh.structure_diagnostics(np.vstack((v, [9, 9, 9])), f)
        self.assertEqual(open_d, dict(surface_area_m2=1., degenerate_triangle_count=0,
            unreferenced_vertex_count=1, boundary_edge_count=4, boundary_length_m=4.,
            nonmanifold_edge_count=0, connected_component_count=1))
        two = mesh.structure_diagnostics(np.vstack((v, v + [3, 0, 0])), np.vstack((f, f + 4)))
        self.assertEqual(two['connected_component_count'], 2)
        self.assertEqual(two['boundary_edge_count'], 8)
        self.assertEqual(two['surface_area_m2'], 2.)
        nv = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [0,-1,0]], float)
        nf = np.array([[0,1,2], [0,1,3], [0,1,4]])
        nonmanifold = mesh.structure_diagnostics(nv, nf)
        self.assertEqual(nonmanifold['nonmanifold_edge_count'], 1)
        self.assertEqual(nonmanifold['boundary_edge_count'], 6)
        self.assertAlmostEqual(nonmanifold['boundary_length_m'], 3*(1+math.sqrt(2)))
        self.assertEqual(nonmanifold['connected_component_count'], 1)
        self.assertEqual(nonmanifold['surface_area_m2'], 1.5)
        invalid_cases = [
            (np.empty((0, 3)), np.empty((0, 3), int), 'EMPTY_GEOMETRY'),
            (np.array([[0.,0,0], [1,0,0], [2,0,0]]), np.array([[0,1,2]]), 'DEGENERATE_TRIANGLE'),
            (v, np.array([[0,1,9]]), 'INDEX_OUT_OF_RANGE'),
            (np.full(v.shape, np.inf), f, 'NONFINITE_GEOMETRY'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, (bad_v, bad_f, reason) in enumerate(invalid_cases):
                path = Path(directory) / f'{index}.ply'
                mesh.write_ply(path, bad_v, bad_f)
                with self.assertRaises(mesh.InvalidInput) as error:
                    mesh.read_ply(path)
                self.assertEqual(error.exception.reason, reason)
        MEASUREMENTS['A08-structure'] = dict(open=open_d, two_components=two, nonmanifold=nonmanifold,
                                            invalid_mesh_rejections=[x[2] for x in invalid_cases])

    def test_07_official_guards_and_threshold(self):
        for values in [np.zeros(11), np.array([2., 1., 1., 1., 3., 1., 1., 1., 1., 1., 1.])]:
            selected, count = official.stable_best90(values)
            self.assertEqual(count, 10)
            np.testing.assert_array_equal(selected, np.sort(values, kind='stable')[:10])
        for invalid in [np.array([]), np.array([np.nan]), np.array([np.inf]), np.array([-1.])]:
            with self.assertRaises(mesh.EvaluatorFailure):
                official.stable_best90(invalid)
        distance, iterations = official.repaired_pass_loop(3, lambda ids: (ids[:1], np.zeros(1)))
        np.testing.assert_array_equal(distance, np.zeros(3))
        self.assertEqual(iterations, 3)
        cases = [
            ('OFFICIAL_NO_PROGRESS', lambda _: (np.array([], int), np.array([])), {}),
            ('OFFICIAL_PASS_LIMIT', lambda ids: (ids[:1], np.zeros(1)), {'max_passes':1}),
            ('OFFICIAL_TIME_LIMIT', lambda ids: (ids, np.zeros(len(ids))), {'deadline':time.monotonic()-1}),
            ('OFFICIAL_INVALID_PRIMITIVE_ID', lambda _: (np.array([99]), np.zeros(1)), {}),
            ('OFFICIAL_NONFINITE_DISTANCE', lambda _: (np.array([0]), np.array([np.nan])), {}),
        ]
        for reason, callback, kw in cases:
            with self.assertRaisesRegex(mesh.EvaluatorFailure, reason):
                official.repaired_pass_loop(3, callback, **kw)
        v, f = square()
        residuals = []
        for height, expected in [(.2-1e-8, 1.), (.2, 1.), (.2+1e-8, 0.)]:
            ref = np.array([[.5, .5, height]])
            distances = mesh.raycast_distances(v, f, ref, np.zeros(3))
            self.assertEqual(float(distances[0] <= .2), expected)
            # These completeness checks exercise both scoring entrances.
            formal, _ = mesh.metrics_from_arrays(v, f, ref, ref, sample_count=10, seed=SEED)
            compat, _ = official.compatibility_from_arrays(v, f, ref, ref)
            self.assertEqual(formal['completeness_0_20'], expected)
            self.assertEqual(compat['completeness_0_20'], expected)
            # Isolate only sampling to place a point at the exact threshold;
            # the production forward nearest-neighbour/reducer remain real.
            with mock.patch.object(mesh, 'sample_surface', return_value=np.array([[.5,.5,0.]])):
                precision_check, _ = mesh.metrics_from_arrays(v,f,ref,ref,sample_count=1,seed=SEED)
            self.assertEqual(precision_check['precision_0_20'], expected)
            residuals.append(float(distances[0]-height))
        MEASUREMENTS['A06-threshold-A11-guards'] = dict(residuals_m=residuals, termination_cases=[c[0] for c in cases])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    runtime = mesh.runtime_versions()
    assert mesh.load_protocol()['sampling']['count'] == N
    assert mesh.load_protocol()['sampling']['seed'] == SEED
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentAcceptance)
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    files = [Path(__file__), Path(mesh.__file__), Path(official.__file__), Path(mesh.__file__).with_name('protocol_v1.json')]
    report = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun, runtime=runtime,
        failures=[(str(t), s) for t,s in result.failures + result.errors], seconds=time.monotonic()-started,
        measurements=MEASUREMENTS, source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope='Controlled A02/A06/A07/A08/A09/A10/A11; full-scene attempts and release sign-off are separate. Fault injections exercise actual dispatch in-process, not independent OS processes.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
