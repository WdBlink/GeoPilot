"""Fixed, read-only OPALS/RGB-track diagnostic; never optimize or read reference data."""
import argparse
from collections import Counter, defaultdict
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ENTRY = Path(__file__).resolve()
HELPER = ENTRY.with_name('shared_camera_ba_diagnostic.py')
spec = importlib.util.spec_from_file_location('projection_ba_helpers', HELPER)
ba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ba)  # Exact file import also works under the existing file allowlist.
np, pc = ba.np, ba.pc
BASE = ba.BASE
V1_ENTRY = ENTRY.with_name('publisher_projection_diagnostic.py')
V1_PLAN = BASE / 'publisher-projection-plan.md'
PLAN = BASE / 'publisher-projection-r2-plan.md'
REPORT = BASE / 'publisher-projection-conventions.md'
OUTPUT = BASE / 'publisher-projection-r2-run'
TABLE = ba.INPUTS / 'Image_orientations_dataset1.xyz'
ISOLATION = ba.ROOT / 'code/geopilot_rsih/isolation.mjs'
NODE = Path(shutil.which('node') or '/opt/homebrew/bin/node').resolve()
LIMITS = {'timeout_seconds': 1800, 'max_rss_bytes': 8 * 1024**3,
          'chunk_pairs': 100000, 'min_free_bytes': 2 * 1024**3}
PARAMETERS = {'deltas_px': [0.0, 0.5], 'model': 'SIMPLE_PINHOLE',
              'rotation': 'Rcw=diag(1,-1,-1)@(Rx(omega)@Ry(phi)@Rz(kappa)).T',
              'intrinsics': '[c,x0+delta,-y0+delta]', 'distortion': 'zero',
              'distance': 'sqrt-Sampson-pixels', 'minimum_denominator': 0.0,
              'camera_tuple': '[Rcw,C,K]', 'relative_translation': 'Rj@(Ci-Cj)',
              'order': 'point_id, image_id_1, image_id_2',
              'multiple_xy': 'bitwise-identical float64 xy collapse; differing xy invalidates all incident pairs'}
PINS = {str(V1_ENTRY): '9f6c4bada7a05a267d65db20966a57ff1d96d14e689c12b90b18800452fa4af3',
        str(V1_PLAN): 'ee2bec88736ae422230651c7c6ea212ac38cc67bdbc7ce896148632cb28b6d6c',
        str(HELPER): '07d85983a366f8b928e104a47884d9ddaea343554efd3b0c0b0210db9c5f71ac',
        str(REPORT): 'f5c6cd3c87fa86043acb21f16cb7ade6674059b376523a5cf79f2402b0d93958',
        str(ISOLATION): 'b71d7693a414fe779a0bc9430e9458833e4ea87a47f188f5d2d80e30cd4631b9'}
STATUS = {'valid': 0, 'ambiguous_xy': 1, 'nonfinite_input': 2,
          'degenerate_denominator': 3, 'nonfinite_calculation': 4}
RAW = np.dtype([('point_id', '<u8'), ('image1', '<u4'), ('image2', '<u4'),
                ('xy1', '<f8', (2,)), ('xy2', '<f8', (2,)),
                ('distance_px', '<f8', (2,)), ('status', 'u1', (2,))])


def camera(row, delta):
    """The one predeclared OPALS rotation and pixel-origin interpretation."""
    center = np.asarray(row[:3], dtype=float)
    w, p, k = np.deg2rad(row[3:6])
    cw, cp, ck, sw, sp, sk = np.cos(w), np.cos(p), np.cos(k), np.sin(w), np.sin(p), np.sin(k)
    rx = np.array([[1, 0, 0], [0, cw, -sw], [0, sw, cw]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[ck, -sk, 0], [sk, ck, 0], [0, 0, 1]])
    rotation = np.diag([1., -1., -1.]) @ (rx @ ry @ rz).T
    intrinsic = np.array([[row[6], 0, row[7] + delta],
                          [0, row[6], -row[8] + delta], [0, 0, 1.]])
    return rotation, center, intrinsic


def fundamental(first, second):
    r1, c1, k1 = first
    r2, c2, k2 = second
    relative = r2 @ r1.T
    # Difference the stored centers before rotation; never cancel large world translations.
    x, y, z = r2 @ (c1 - c2)
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    value = np.linalg.inv(k2).T @ cross @ relative @ np.linalg.inv(k1)
    norm = np.linalg.norm(value)
    return value / norm if norm > 0 and np.isfinite(norm) else value


def sampson(matrices, xy1, xy2):
    """Return sqrt Sampson distance and explicit per-record status, without filtering."""
    p1 = np.column_stack((xy1, np.ones(len(xy1))))
    p2 = np.column_stack((xy2, np.ones(len(xy2))))
    l2 = np.einsum('nij,nj->ni', matrices, p1)
    l1 = np.einsum('nji,nj->ni', matrices, p2)
    denominator = np.sum(l2[:, :2]**2 + l1[:, :2]**2, axis=1)
    numerator = np.abs(np.sum(p2 * l2, axis=1))
    status = np.zeros(len(xy1), dtype=np.uint8)
    finite_input = np.isfinite(p1).all(axis=1) & np.isfinite(p2).all(axis=1)
    status[~finite_input] = STATUS['nonfinite_input']
    status[finite_input & (~np.isfinite(denominator) | ~np.isfinite(numerator))] = STATUS['nonfinite_calculation']
    status[finite_input & np.isfinite(denominator) & np.isfinite(numerator)
           & (denominator <= 0)] = STATUS['degenerate_denominator']
    distance = np.full(len(xy1), np.nan)
    valid = status == 0
    distance[valid] = numerator[valid] / np.sqrt(denominator[valid])
    status[valid & ~np.isfinite(distance)] = STATUS['nonfinite_calculation']
    distance[status != 0] = np.nan
    return distance, status


def self_test():
    """Independent scalar collinearity, native camera projection and epipolar oracles."""
    row = [10., 20., 30., 17., -9., 23., 500., 320., -240., 0., 0., 0., 0., 0.]
    w, p, k = np.deg2rad(row[3:6])
    a, b, c, d, e, f = np.cos(w), np.cos(p), np.cos(k), np.sin(w), np.sin(p), np.sin(k)
    scalar_r = ((b*c, -b*f, e), (a*f+d*e*c, a*c-d*e*f, -d*b),
                (d*f-a*e*c, d*c+a*e*f, a*b))
    r, center, _ = camera(row, 0)
    t = -r @ center  # Projection-only check, not the relative-translation implementation.
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-14, rtol=0)
    assert abs(np.linalg.det(r) - 1) < 1e-14
    assert np.allclose(-r.T @ t, row[:3], atol=1e-12, rtol=0)
    for offset in ([1., 2., -10.], [-2., 3., -15.], [4., -3., -20.]):
        world = np.asarray(row[:3]) + offset
        q = r @ world + t
        assert q[2] > 0
        u = [sum(scalar_r[j][i] * offset[j] for j in range(3)) for i in range(3)]
        expected = np.array([row[7]-row[6]*u[0]/u[2], -row[8]+row[6]*u[1]/u[2]])
        projections = []
        for delta in PARAMETERS['deltas_px']:
            _, _, intrinsic = camera(row, delta)
            cam = pc.Camera(model='SIMPLE_PINHOLE', width=640, height=480,
                            params=[row[6], intrinsic[0, 2], intrinsic[1, 2]])
            projections.append(cam.img_from_cam(q))
            assert np.allclose(projections[-1], expected + delta, atol=1e-10, rtol=0)
            assert cam.img_from_cam(-q) is None
        assert np.allclose(projections[1]-projections[0], [.5, .5], atol=1e-12, rtol=0)
    # Independently construct two translated identity cameras, rather than deriving observations from F.
    k = np.array([[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]])
    F = fundamental((np.eye(3), np.zeros(3), k), (np.eye(3), np.array([2., 0, 0]), k))
    x1 = np.array([[370., 340.]])  # X=(1,2,10), C1=(0,0,0)
    x2 = np.array([[270., 340.]])  # C2=(2,0,0)
    distances, status = sampson(F[None], x1, x2)
    assert status[0] == 0 and distances[0] < 1e-12
    distances, status = sampson(F[None], x1, x2 + [0., 2.])
    assert status[0] == 0 and abs(distances[0]-np.sqrt(2)) < 1e-12
    distances, status = sampson(np.zeros((1, 3, 3)), x1, x2)
    assert status[0] == STATUS['degenerate_denominator'] and np.isnan(distances[0])
    distances, status = sampson(F[None], x1 * np.nan, x2)
    assert status[0] == STATUS['nonfinite_input'] and np.isnan(distances[0])
    # UTM-sized centers: coincident cameras must remain exactly degenerate despite different R.
    large = np.array([500000., 4400000., 200.])
    first_row = [*large, 17., -9., 23., 500., 320., -240., 0., 0., 0., 0., 0.]
    second_row = [*large, -3., 11., -5., 500., 320., -240., 0., 0., 0., 0., 0.]
    for delta in PARAMETERS['deltas_px']:
        first = camera(first_row, delta)
        coincident = camera(second_row, delta)
        zero = fundamental(first, coincident)
        assert np.array_equal(zero, np.zeros((3, 3)))
        distances, status = sampson(zero[None], x1, x2)
        assert status[0] == STATUS['degenerate_denominator'] and np.isnan(distances[0])
        # A representable 1 micrometre-scale baseline is legal, without a chosen cutoff.
        separated_row = list(second_row)
        separated_row[:3] = large + np.array([1.e-6, -2.e-6, .25e-6])
        second = camera(separated_row, delta)
        baseline = second[1] - first[1]
        assert 0 < np.linalg.norm(baseline) < 3.e-6
        F = fundamental(first, second)
        local_first = (first[0], np.zeros(3), first[2])
        local_second = (second[0], baseline, second[2])
        assert np.array_equal(F, fundamental(local_first, local_second))
        # Correspondence generated by local 3D offsets, not by F or large translation subtraction.
        local_point = np.array([3., -2., -40.])
        q1, q2 = first[0] @ local_point, second[0] @ (local_point - baseline)
        assert q1[2] > 0 and q2[2] > 0
        h1, h2 = first[2] @ q1, second[2] @ q2
        xy1, xy2 = (h1[:2] / h1[2])[None], (h2[:2] / h2[2])[None]
        distances, status = sampson(F[None], xy1, xy2)
        assert status[0] == 0 and distances[0] < 1.e-9
        perturbed, status = sampson(F[None], xy1, xy2 + [0., 2.])
        assert status[0] == 0 and perturbed[0] > 0
    # Identical geometry may collapse; different pixels cannot acquire a representative.
    assert identical_xy([np.array([1., 2.]), np.array([1., 2.])])
    assert not identical_xy([np.array([1., 2.]), np.array([1., 3.])])
    from io import StringIO
    from types import SimpleNamespace as Obj
    images = {i: Obj(points2D=[Obj(xy=np.array(x), point3D_id=7) for x in xy])
              for i, xy in {1: [[1., 2.], [1., 2.]], 2: [[3., 4.], [3., 5.]],
                            3: [[6., 7.]]}.items()}
    elements = [Obj(image_id=i, point2D_idx=j) for i in images
                for j in range(len(images[i].points2D))]
    toy = Obj(images=images, points3D={7: Obj(track=Obj(elements=elements))})
    counts, sidecar = Counter(), StringIO()
    [(pid, groups)] = observations(toy, counts, sidecar)
    assert pid == 7 and counts['all_pairs'] == 3 and counts['ambiguous_pairs'] == 2
    assert counts['identical_xy_groups'] == counts['ambiguous_xy_groups'] == 1
    assert np.isnan(groups[1][1]).all() and groups[1][2]
    assert len(sidecar.getvalue().splitlines()) == 2
    return 'PASS: scalar/native projection, SO3, centers, positive depth, delta, Sampson, invalids, xy dedup, large-center zero/micro baseline'


def identical_xy(values):
    return len({np.asarray(x, dtype='<f8').tobytes() for x in values}) == 1


def load_inputs():
    manifest = ba.read_json(ba.INPUTS / 'input_manifest.json')
    rows = {}
    for line in TABLE.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 15 or fields[0] in rows:
            raise ValueError('Malformed/duplicate publisher row')
        rows[fields[0]] = np.asarray(fields[1:], dtype=float)
    if (len(rows) != 224 or set(rows) != {x['image_id'] for x in manifest['images']}
            or not np.isfinite(list(rows.values())).all()
            or any(list(x[6:9]) != [4636.912, 3970.288, -2601.560]
                   or np.any(x[9:] != 0) for x in rows.values())):
        raise ValueError('Unexpected fixed publisher calibration/coverage')
    model = pc.Reconstruction()
    model.read_binary(str(ba.MODEL))
    if (model.num_images() != 224 or model.num_reg_images() != 224
            or {x.name for x in model.images.values()} != set(rows)
            or any((c.width, c.height) != (7952, 5304) for c in model.cameras.values())):
        raise ValueError('P2 image coverage/dimensions changed')
    return model, rows


def observations(model, counts, sidecar=None):
    """Read track IDs and original pixels only; never access point.xyz or fitted poses."""
    for point_id in sorted(model.points3D):
        groups = defaultdict(set)
        elements = model.points3D[point_id].track.elements
        counts['raw_track_elements'] += len(elements)
        for element in elements:
            groups[element.image_id].add(element.point2D_idx)
        counts['exact_duplicate_track_elements'] += len(elements)-sum(map(len, groups.values()))
        result = []
        for image_id, indices in sorted(groups.items()):
            image = model.images[image_id]
            pixels = [image.points2D[i].xy for i in sorted(indices)]
            if any(image.points2D[i].point3D_id != point_id for i in indices):
                raise ValueError('Track-to-image backlink mismatch')
            ambiguous = not identical_xy(pixels)
            if len(indices) > 1:
                counts['multiple_observation_groups'] += 1
                counts['ambiguous_xy_groups' if ambiguous else 'identical_xy_groups'] += 1
                if sidecar is not None:
                    sidecar.write(json.dumps({'point_id': point_id, 'image_id': image_id,
                        'status': 'ambiguous' if ambiguous else 'identical_xy',
                        'observations': [{'point2D_idx': i,
                            'xy': [float(v) if np.isfinite(v) else str(v) for v in image.points2D[i].xy],
                            'xy_float64_le_hex': np.asarray(image.points2D[i].xy, dtype='<f8').tobytes().hex()}
                                         for i in sorted(indices)]}, allow_nan=False) + '\n')
            # A differing-xy group has no selected pixel. Its pairs retain NaN + explicit invalid status.
            xy = np.full(2, np.nan) if ambiguous else pixels[0]
            result.append((image_id, xy, ambiguous))
        counts['points'] += 1
        counts['unique_point_image_observations'] += len(result)
        counts['max_unique_track_length'] = max(counts['max_unique_track_length'], len(result))
        counts['all_pairs'] += len(result)*(len(result)-1)//2
        n = sum(not x[2] for x in result)
        counts['ambiguous_pairs'] += len(result)*(len(result)-1)//2 - n*(n-1)//2
        yield point_id, result


def preflight():
    ba.verify(PINS)
    bound = ba.bindings()
    bound.update(PINS)  # Bind both frozen v1 files as provenance, without modifying them.
    launch = ba.read_json(ba.SOURCE / 'launch.json')
    # Original RGB bytes are verified by the parent only; workers consume frozen pixels in images.bin.
    bound.update(launch['input_hashes'])
    for path in (ENTRY, PLAN, REPORT, ISOLATION, NODE, HELPER):
        bound[str(path)] = ba.sha(path)
    ba.verify(bound)
    worker = {p: h for p, h in bound.items() if p not in launch['input_hashes']
              or p in (str(TABLE), str(ba.INPUTS / 'input_manifest.json'))}
    # The worker does not read the historical launch or unrelated old BA planning documents.
    for path in (ba.SOURCE / 'launch.json', ba.PLAN, ba.DIAGNOSIS):
        worker.pop(str(path), None)
    if OUTPUT.exists():
        raise FileExistsError('Fresh output required: ' + str(OUTPUT))
    if shutil.disk_usage(BASE).free < LIMITS['min_free_bytes']:
        raise RuntimeError('Insufficient free disk for raw pairs and logs')
    return {'schema': 'publisher-projection-diagnostic/2', 'formal_candidate': False,
            'official_conversion_certified': False, 'parameters': PARAMETERS, 'limits': LIMITS,
            'bindings': bound, 'worker_bindings': worker, 'runtime': launch['runtime'],
            'original_input_hashes': launch['input_hashes'], 'status_codes': STATUS,
            'raw_dtype': RAW.descr, 'points_xyz_used': False,
            'source_selection': 'all original P2 RGB-SfM tracks; already filtered by that reconstruction'}


def isolated(case, manifest):
    allowed = [p for p in manifest['worker_bindings'] if not any(
        Path(p).is_relative_to(manifest['runtime'][k]) for k in ('prefix', 'base_prefix'))]
    allowed.append(str(case / 'inputs.json'))
    reference = ba.ROOT / 'out/usegeo_benchmark/prepared/v1/evaluator-only/Dataset-1'
    forbidden = [reference / f for f in ('full_lidar.las', 'refined_lidar.las', 'publisher_mvs.las')]
    # Use a known real historical score only for a denied-open probe, never read its bytes.
    forbidden.append(ba.ROOT / 'out/usegeo_mesh_benchmark/paper-readiness-20260918/full-score-pass-1/Dataset-1/score.json')
    if not all(p.is_file() for p in forbidden):
        raise ValueError('Missing forbidden probe target')
    ba.write(case / 'isolation-request.json', {'output': str(case), 'python': str(ba.RUNTIME / 'bin/python'),
        'allowed': allowed, 'forbidden': list(map(str, forbidden))})
    script = """import {readFileSync,writeFileSync} from 'node:fs';
const {makeProfile,probeIsolation}=await import(process.argv[2]);
const s=JSON.parse(readFileSync(process.argv[1],'utf8')),start=performance.now();
const profile=makeProfile(s.output,s.python,s.allowed);
const proof=probeIsolation(profile,s.output,s.python,s.allowed,s.forbidden,true);
writeFileSync(s.output+'/isolation.json',JSON.stringify({...proof,profile,probe_seconds:(performance.now()-start)/1000}),{flag:'wx'});"""
    with (case / 'isolation.stdout').open('xb') as out, (case / 'isolation.stderr').open('xb') as err:
        subprocess.run([str(NODE), '--input-type=module', '-e', script,
                        str(case / 'isolation-request.json'), ISOLATION.as_uri()],
                       stdout=out, stderr=err, timeout=120, check=True)
    return ba.read_json(case / 'isolation.json')


def worker(case, manifest, check_only):
    if manifest['parameters'] != PARAMETERS or manifest['limits'] != LIMITS:
        raise ValueError('Worker protocol drift')
    ba.verify(manifest['worker_bindings'])
    proof = self_test()
    model, rows = load_inputs()
    counts = Counter()
    if check_only:
        for _ in observations(model, counts):
            pass
        ba.verify(manifest['worker_bindings'])
        ba.write(case / 'worker-result.json', {'status': 'checked_without_real_residuals', 'self_test': proof,
            'counts': dict(counts), 'raw_bytes_estimate': counts['all_pairs'] * RAW.itemsize,
            'chunk_pairs': LIMITS['chunk_pairs'], 'source_bytes': sum((ba.MODEL / n).stat().st_size for n in ba.MODEL_SHA)})
        return
    ids = sorted(model.images)
    positions = {image_id: index for index, image_id in enumerate(ids)}
    n = len(ids)
    matrices = np.zeros((2, n*n, 3, 3))
    for variant, delta in enumerate(PARAMETERS['deltas_px']):
        cameras = {i: camera(rows[model.images[i].name], delta) for i in ids}
        for i, j in combinations(ids, 2):
            matrices[variant, positions[i]*n+positions[j]] = fundamental(cameras[i], cameras[j])
    total = np.zeros(n*n, dtype=np.int64)
    valid = np.zeros((2, n*n), dtype=np.int64)
    sums, squares = np.zeros((2, n*n)), np.zeros((2, n*n))
    invalid = np.zeros((2, len(STATUS)), dtype=np.int64)
    raw_files, buf, used = [], np.empty(LIMITS['chunk_pairs'], dtype=RAW), 0

    def flush(size):
        batch = buf[:size]
        keys = np.array([positions[int(i)]*n+positions[int(j)] for i, j in zip(batch['image1'], batch['image2'])])
        total[:] += np.bincount(keys, minlength=n*n)
        ambiguous = batch['status'][:, 0] == STATUS['ambiguous_xy']
        for variant in range(2):
            distance = np.full(size, np.nan)
            status = np.full(size, STATUS['ambiguous_xy'], dtype=np.uint8)
            candidates = ~ambiguous
            distance[candidates], status[candidates] = sampson(
                matrices[variant, keys[candidates]], batch['xy1'][candidates], batch['xy2'][candidates])
            batch['distance_px'][:, variant], batch['status'][:, variant] = distance, status
            ok = status == 0
            valid[variant] += np.bincount(keys[ok], minlength=n*n)
            sums[variant] += np.bincount(keys[ok], weights=distance[ok], minlength=n*n)
            squares[variant] += np.bincount(keys[ok], weights=distance[ok]**2, minlength=n*n)
            invalid[variant] += np.bincount(status, minlength=len(STATUS))
        path = case / f'pairs-{len(raw_files):05d}.npy'
        with path.open('xb') as stream:
            np.save(stream, batch, allow_pickle=False)
        raw_files.append({'path': str(path), 'count': size, 'sha256': ba.sha(path)})
        print(json.dumps({'chunk': len(raw_files), 'pairs_written': int(total.sum())}), flush=True)

    with (case / 'multiple-observations.jsonl').open('x') as sidecar:
        for point_id, groups in observations(model, counts, sidecar):
            for (i, xi, ai), (j, xj, aj) in combinations(groups, 2):
                buf[used] = (point_id, i, j, xi, xj, [np.nan, np.nan], [1 if ai or aj else 0]*2)
                used += 1
                if used == len(buf):
                    flush(used)
                    used = 0
        if used:
            flush(used)
    if (int(total.sum()) != counts['all_pairs']
            or any(int(invalid[v].sum()) != counts['all_pairs']
                   or int(invalid[v, 0]) != int(valid[v].sum()) for v in range(2))):
        raise ValueError('Raw pair denominator changed')
    summaries = []
    for i, j in combinations(ids, 2):
        key = positions[i]*n+positions[j]
        if not total[key]:
            continue
        summaries.append({'image1': i, 'image2': j, 'all_pairs': int(total[key]),
            'variants': [{'delta_px': delta, 'valid_pairs': int(valid[v, key]),
                'invalid_pairs': int(total[key]-valid[v, key]),
                'mean_px': float(sums[v, key]/valid[v, key]) if valid[v, key] else None,
                'rms_px': float(np.sqrt(squares[v, key]/valid[v, key])) if valid[v, key] else None}
                for v, delta in enumerate(PARAMETERS['deltas_px'])]})
    ba.write(case / 'pair-summary.json', summaries)
    ba.verify(manifest['worker_bindings'])
    ba.write(case / 'worker-result.json', {'status': 'completed', 'formal_candidate': False,
        'official_conversion_certified': False, 'counts': dict(counts), 'raw_files': raw_files,
        'image_ids': {i: model.images[i].name for i in ids}, 'status_codes': STATUS,
        'variants': [{'delta_px': delta, 'all_pairs': int(total.sum()),
            'status_counts': {name: int(invalid[v, code]) for name, code in STATUS.items()},
            'valid_pairs': int(valid[v].sum()),
            'mean_valid_px': float(sums[v].sum()/valid[v].sum()) if valid[v].sum() else None,
            'rms_valid_px': float(np.sqrt(squares[v].sum()/valid[v].sum())) if valid[v].sum() else None}
            for v, delta in enumerate(PARAMETERS['deltas_px'])]})


def main():
    if not __debug__ or Path(sys.prefix).resolve() != ba.RUNTIME.resolve() or pc.__version__ != '3.12.6':
        raise RuntimeError('Use frozen baseline-runtime, -B, and never -O')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('self-test', 'check', 'run', '_check_worker', '_worker'))
    parser.add_argument('manifest', nargs='?')
    args = parser.parse_args()
    if args.action == 'self-test':
        print(self_test())
        return
    if args.action.startswith('_'):
        source = Path(args.manifest).resolve()
        worker(source.parent, ba.read_json(source), args.action == '_check_worker')
        return
    manifest = preflight()
    case = Path(tempfile.mkdtemp(prefix='publisher-projection-r2-check-')).resolve() if args.action == 'check' else OUTPUT
    if args.action == 'run':
        case.mkdir(exist_ok=False)
    ba.write(case / 'inputs.json', manifest)
    started = time.monotonic()
    try:
        isolation = isolated(case, manifest)
        command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin:/usr/sbin:/sbin',
            'PYTHONDONTWRITEBYTECODE=1', 'TMPDIR=' + str(case), 'OPENSSL_CONF=/dev/null',
            'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1',
            '/usr/bin/sandbox-exec', '-f', isolation['profile'], str(ba.RUNTIME / 'bin/python'),
            '-B', str(ENTRY), '_check_worker' if args.action == 'check' else '_worker', str(case / 'inputs.json')]
        ba.write(case / 'command.json', {'argv': command, 'limits': LIMITS,
            'inputs_sha256': ba.sha(case / 'inputs.json'), 'profile_sha256': ba.sha(isolation['profile'])})
        status = ba.supervise(command, case, timeout=LIMITS['timeout_seconds'], max_rss=LIMITS['max_rss_bytes'])
        status['final_log_hashes'] = {str(case / name): ba.sha(case / name) for name in ('runtime.stdout', 'runtime.stderr')}
        ba.write(case / 'supervision.json', status)
        ba.verify(manifest['bindings'])
        if status['returncode'] or status['reason'] is not None:
            raise RuntimeError('Worker failed; retain all partial outputs: ' + str(status))
        result = ba.read_json(case / 'worker-result.json')
        result.update(inputs_sha256=ba.sha(case / 'inputs.json'), wall_seconds=time.monotonic()-started,
            supervision=status, isolation=isolation,
            output_hashes={str(p): ba.sha(p) for p in sorted(case.iterdir()) if p.is_file()})
        ba.write(case / 'result.json', result)
        print(json.dumps({'status': result['status'], 'case': str(case), 'counts': result['counts'],
                          'supervision': status, 'raw_bytes_estimate': result.get('raw_bytes_estimate')}))
    except BaseException as exc:
        ba.write(case / 'failure.json', {'status': 'failed', 'error': repr(exc), 'wall_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
