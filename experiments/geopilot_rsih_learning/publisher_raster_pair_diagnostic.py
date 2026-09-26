"""Fixed M/U raster compatibility; check matches synthetic descriptors only, never real images."""
import argparse
from collections import Counter
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
R2 = ENTRY.with_name('publisher_projection_diagnostic_r2.py')
spec = importlib.util.spec_from_file_location('raster_projection', R2)
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)
ba, np, pc = projection.ba, projection.np, projection.pc
from strict_json import finite, unique
PLAN = ba.BASE / 'publisher-raster-pair-plan.md'
DESIGN = ba.BASE / 'publisher-raster-experiment-design.md'
AUDIT = ba.BASE / 'rgb-export-version-audit.json'
U = ba.BASE / 'publisher-raster-input'
OUTPUT = ba.BASE / 'publisher-raster-pair-run'
ROOTS = {'M': ba.INPUTS, 'U': U}
DIMENSIONS = {'M': [7952, 5304], 'U': [7953, 5279]}
K = [4636.912, 3970.788, 2602.060, 0.0]
LIMITS = {'timeout_seconds_per_arm': 600, 'max_rss_bytes': 8 * 1024**3,
          'startup_free_bytes': 8 * 1024**3, 'num_threads': 1}
PARAMETERS = {'images': 16, 'selection': 'lexicographically first 16 names', 'pairs': 120,
    'pair_order': 'lexicographic image-name combinations', 'arm_order': ['M', 'U'],
    'seed': 917, 'delta_px': .5, 'camera_model': 'SIMPLE_RADIAL', 'K': K,
    'camera_mode': 'PER_IMAGE', 'require_prior_focal_length': True,
    'guided_matching': False, 'native_min_num_inliers': 1,
    'native_geometry_role': 'auxiliary min1 output; not original min15 geometry control',
    'publisher_camera': 'fixed R2 OPALS; f=c; k=0; no pixel rescaling',
    'rgb_F_source': 'all raw descriptor matches; TwoViewGeometryOptions.ransac; seed917 per pair',
    'statistics': ['count', 'valid', 'invalid', 'mean', 'rms', 'median', 'p90', 'p95', 'min', 'max']}


def plain(value):
    if isinstance(value, pc.Normalization):
        return str(value)
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {'nonfinite': str(value)}
    return value


def options():
    old = ba.legacy.sparse.options()
    old['reader'].camera_params = ','.join(format(v, '.17g') for v in K)
    old['matching'].guided_matching = False
    if old['verification'].min_num_inliers != 15:
        raise ValueError('Original native retention threshold changed')
    # COLMAP otherwise clears descriptor matches with fewer than 15 rows before DB write.
    # Min1 retains every nonempty descriptor set; independent RGB F uses unchanged RANSAC.
    old['verification'].min_num_inliers = PARAMETERS['native_min_num_inliers']
    return {name: old[name] for name in ('reader', 'extraction', 'matching', 'verification')} | {
        'exhaustive': pc.ExhaustiveMatchingOptions(), 'rgb_ransac': old['verification'].ransac}


def settings(opts):
    return {key: plain(value.todict()) for key, value in opts.items()} | {
        'device': 'cpu', 'camera_mode': 'PER_IMAGE', 'pycolmap_version': pc.__version__,
        'api_docs': {name: getattr(pc, name).__doc__ for name in (
            'extract_features', 'match_exhaustive', 'estimate_fundamental_matrix')},
        'raw_matches_api': pc.Database.read_matches.__doc__,
        'native_verified_matches_api': pc.Database.read_two_view_geometry.__doc__}


def rgb_estimate(xy1, xy2, ransac):
    """Fit only independent RGB geometry; never receive the publisher F or residuals."""
    count = len(xy1)
    empty = np.zeros(count, dtype=bool)
    pc.set_random_seed(PARAMETERS['seed'])
    if not count:
        return {'status': 'zero_matches', 'num_inliers': 0, 'F': None}, empty
    if not np.isfinite(xy1).all() or not np.isfinite(xy2).all():
        return {'status': 'nonfinite_input_not_estimated', 'num_inliers': 0, 'F': None}, empty
    if count < 7:
        return {'status': 'insufficient_matches_lt7', 'num_inliers': 0, 'F': None}, empty
    try:
        result = pc.estimate_fundamental_matrix(xy1, xy2, estimation_options=ransac)
    except (ValueError, RuntimeError) as exc:
        return {'status': 'estimator_exception', 'num_inliers': 0, 'F': None, 'error': repr(exc)}, empty
    if result is None:
        return {'status': 'estimator_returned_none', 'num_inliers': 0, 'F': None}, empty
    if set(result) != {'F', 'num_inliers', 'inlier_mask'}:
        raise ValueError('Unexpected native fundamental estimator result schema')
    F, mask = np.asarray(result['F']), np.asarray(result['inlier_mask'])
    if F.shape != (3, 3) or mask.shape != (count,) or mask.dtype != np.bool_:
        raise ValueError('Invalid native fundamental estimator result shape/type')
    if int(result['num_inliers']) != int(mask.sum()):
        raise ValueError('Native inlier count and mask disagree')
    if not np.isfinite(F).all() or np.linalg.norm(F) == 0:
        return {'status': 'invalid_estimated_F', 'num_inliers': 0, 'F': None,
                'reported_native_num_inliers': int(mask.sum())}, empty
    return {'status': 'estimated', 'num_inliers': int(mask.sum()), 'F': F.tolist()}, mask


def distribution(distance, status):
    valid = status == 0
    values = distance[valid]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite distance marked valid')
    return {'count': len(distance), 'valid': int(valid.sum()), 'invalid': int((~valid).sum()),
        'status_counts': {name: int((status == code).sum()) for name, code in projection.STATUS.items()},
        'mean': float(values.mean()) if len(values) else None,
        'rms': float(np.sqrt(np.mean(values**2))) if len(values) else None,
        'median': float(np.median(values)) if len(values) else None,
        'p90': float(np.quantile(values, .90)) if len(values) else None,
        'p95': float(np.quantile(values, .95)) if len(values) else None,
        'min': float(values.min()) if len(values) else None, 'max': float(values.max()) if len(values) else None}


def read_pair(db, first, second):
    raw_exists, native_exists = db.exists_matches(first, second), db.exists_inlier_matches(first, second)
    raw = db.read_matches(first, second) if raw_exists else np.empty((0, 2), dtype=np.uint32)
    native = db.read_two_view_geometry(first, second) if native_exists else None
    verified = native.inlier_matches if native is not None else np.empty((0, 2), dtype=np.uint32)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.dtype != np.uint32:
        raise ValueError('Invalid raw descriptor-match table')
    if not set(map(tuple, verified)).issubset(set(map(tuple, raw))):
        raise ValueError('Native verified matches not a subset of raw descriptors with guided_matching=False')
    metadata = plain(native.todict()) if native is not None else None
    if metadata is not None:
        metadata.pop('inlier_matches')  # Exact array is saved separately, without JSON expansion.
    return raw, verified, {'raw_row_present': raw_exists, 'native_row_present': native_exists,
                           'native_two_view': metadata}


def camera_records(db, names, dimensions):
    images = sorted(db.read_all_images(), key=lambda i: i.name)
    cameras = {c.camera_id: c for c in db.read_all_cameras()}
    if (len(images) != 16 or [i.name for i in images] != names or len(cameras) != 16
            or len({i.camera_id for i in images}) != 16):
        raise ValueError('Fresh PER_IMAGE database lost or added images/cameras')
    records = []
    for image in images:
        camera = cameras[image.camera_id]
        if (camera.model != pc.CameraModelId.SIMPLE_RADIAL or [camera.width, camera.height] != dimensions
                or not np.array_equal(camera.params, K) or not camera.has_prior_focal_length):
            raise ValueError('Actual camera initialization differs from fixed K/dimensions/prior semantics')
        records.append({'image_id': image.image_id, 'name': image.name, 'camera_id': image.camera_id,
            'model': str(camera.model), 'width': camera.width, 'height': camera.height,
            'params': camera.params.tolist(), 'has_prior_focal_length': camera.has_prior_focal_length})
    return records


def self_test():
    projection.self_test()
    opts = options()
    assert settings(opts) == json.loads(json.dumps(settings(opts), allow_nan=False))
    assert opts['matching'].guided_matching is False
    assert opts['verification'].min_num_inliers == 1
    assert opts['rgb_ransac'].todict() == opts['verification'].ransac.todict()
    assert opts['rgb_ransac'].todict() == {'max_error': 4., 'min_inlier_ratio': .25, 'confidence': .999,
        'dyn_num_trials_multiplier': 3., 'min_num_trials': 100, 'max_num_trials': 10000}
    rng = np.random.default_rng(917)
    xyz = np.column_stack((rng.uniform(-2, 2, 50), rng.uniform(-1, 1, 50), rng.uniform(4, 10, 50)))
    moved = xyz - [1., .2, .1]
    xy1, xy2 = 500*xyz[:, :2]/xyz[:, 2, None]+[320, 240], 500*moved[:, :2]/moved[:, 2, None]+[320, 240]
    estimated, mask = rgb_estimate(xy1, xy2, opts['rgb_ransac'])
    assert estimated['status'] == 'estimated' and mask.all()
    intrinsic = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    F = projection.fundamental((np.eye(3), np.zeros(3), intrinsic), (np.eye(3), np.array([1., .2, .1]), intrinsic))
    distance, status = projection.sampson(np.broadcast_to(F, (50, 3, 3)), xy1, xy2)
    assert distribution(distance, status)['max'] < 1.e-9
    for count, expected in ((0, 'zero_matches'), (6, 'insufficient_matches_lt7'), (10, 'estimator_returned_none')):
        result, mask = rgb_estimate(np.zeros((count, 2)), np.zeros((count, 2)), opts['rgb_ransac'])
        assert result['status'] == expected and not mask.any()
    empty = distribution(np.empty(0), np.empty(0, dtype=np.uint8))
    assert empty['count'] == 0 and empty['mean'] is None
    distance, status = projection.sampson(np.zeros((1, 3, 3)), xy1[:1], xy2[:1])
    assert distribution(distance, status)['invalid'] == 1
    # Actual native matching, synthetic descriptors only: catches pre-write row clearing.
    for count in (2, 8):
        for threshold in (15, 1):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'retention.db'
                db = pc.Database(str(path))
                try:
                    xy = np.column_stack((np.arange(count)*3+10, np.arange(count)**2+10)).astype(np.float32)
                    descriptors = np.zeros((count, 128), dtype=np.uint8)
                    descriptors[np.arange(count), np.arange(count)] = 255
                    for ident in (1, 2):
                        db.write_camera(pc.Camera(camera_id=ident, model='SIMPLE_RADIAL', width=100, height=100,
                            params=[80., 50., 50., 0.], has_prior_focal_length=True), use_camera_id=True)
                        db.write_image(pc.Image(image_id=ident, name=f'{ident}.jpg', camera_id=ident), use_image_id=True)
                        db.write_keypoints(ident, xy)
                        db.write_descriptors(ident, descriptors)
                finally:
                    db.close()
                native = options()
                native['verification'].min_num_inliers = threshold
                pc.set_random_seed(917)
                pc.match_exhaustive(str(path), sift_options=native['matching'], matching_options=native['exhaustive'],
                                    verification_options=native['verification'], device=pc.Device.cpu)
                db = pc.Database(str(path))
                try:
                    raw = db.read_matches(1, 2)
                    expected = np.column_stack((np.arange(count), np.arange(count))).astype(np.uint32)
                    assert len(raw) == (0 if threshold == 15 else count)
                    if threshold == 1:
                        assert np.array_equal(raw, expected)
                finally:
                    db.close()
    # Purely synthetic DB: no image loading, extraction, matching or two-view estimation.
    with tempfile.TemporaryDirectory() as temporary:
        db = pc.Database(str(Path(temporary) / 'toy.db'))
        try:
            names = [f'{i:02d}.jpg' for i in range(16)]
            for ident, name in enumerate(names, 1):
                db.write_camera(pc.Camera(camera_id=ident, model='SIMPLE_RADIAL', width=7952, height=5304,
                    params=K, has_prior_focal_length=True), use_camera_id=True)
                db.write_image(pc.Image(image_id=ident, name=name, camera_id=ident), use_image_id=True)
                db.write_keypoints(ident, np.array([[1., 2.], [3., 4.]], dtype=np.float32))
                db.write_descriptors(ident, np.zeros((2, 128), dtype=np.uint8))
            raw = np.array([[0, 0], [1, 1]], dtype=np.uint32)
            db.write_matches(1, 2, raw)
            native = pc.TwoViewGeometry(config=pc.TwoViewGeometryConfiguration.CALIBRATED,
                                        inlier_matches=raw[:1])
            db.write_two_view_geometry(1, 2, native)
            observed, verified, _ = read_pair(db, 1, 2)
            assert np.array_equal(raw, observed) and len(verified) == 1
            assert read_pair(db, 1, 3)[0].shape == (0, 2)
            pairs = pc.ExhaustivePairGenerator(opts['exhaustive'], db).all_pairs()
            assert {tuple(sorted(p)) for p in pairs} == set(combinations(range(1, 17), 2))
            assert len(camera_records(db, names, DIMENSIONS['M'])) == 16
        finally:
            db.close()
    return 'PASS: R2, fixed RANSAC/schema, toy F/Sampson/empty/degenerate, native descriptor retention 2/8 at min15/min1, raw-versus-native DB, 120 pairs'


def preflight():
    pins = {str(R2): '83714c483084e38277c7dec8d71bacf2288d85591fc0344e0c3404eec1edfb9f',
        str(AUDIT): 'a00457ff19ffb3b9041b5510e46dd0d16d84a5a01ea73dda4aff98fce43bd84b',
        str(ba.INPUTS / 'input_manifest.json'): '7eecd1e27e645c1f790d81cec663c042aa1320a7c542f08f91e004a2cf09a626',
        str(U / 'input_manifest.json'): '0327334e7607a77ab81fee4ffd6dcca4a90d76a6d45221956f78307b3ebd4ecc',
        str(ba.SOURCE / 'launch.json'): '2a5023086874bd1d138164634f109540fcf1ce57f9ab6a64a66c3d3955145e88'}
    ba.verify(pins)
    manifests = {arm: ba.read_json(root / 'input_manifest.json') for arm, root in ROOTS.items()}
    image_hashes = {arm: {x['image_id']: x['sha256'] for x in m['images']} for arm, m in manifests.items()}
    if set(image_hashes['M']) != set(image_hashes['U']) or len(image_hashes['M']) != 224:
        raise ValueError('Full source manifest name sets changed')
    names = sorted(image_hashes['M'])[:16]
    # The pinned audit also has unused filesystem nanosecond/inode integers >2**53.
    # Preserve duplicate-key rejection globally and strict numeric checks on consumed records.
    audit_records = finite(json.loads(AUDIT.read_text(), object_pairs_hook=unique)['records'])
    audit = {r['image_id']: r for r in audit_records}
    receipt = ba.read_json(U / 'import-receipt.json')
    if receipt['status'] != 'complete':
        raise ValueError('U input import incomplete')
    launch = ba.read_json(ba.SOURCE / 'launch.json')
    bound, common = dict(pins), dict(projection.PINS)
    for path, digest in launch['bindings'].items():
        if any(Path(path).is_relative_to(launch['runtime'][k]) for k in ('prefix', 'base_prefix')):
            common[path] = digest
    for name in ('code/geopilot_rsih/tools.py', 'code/geopilot_rsih/strict_json.py', 'code/geopilot_rsih/supervise.py',
                 'code/usegeo_mesh_baseline/runner.py', 'experiments/geopilot_rsi/run.py'):
        common[str(ba.ROOT / name)] = launch['bindings'][str(ba.ROOT / name)]
    for path in (ENTRY, R2, PLAN, DESIGN, projection.PLAN, projection.NODE):
        common[str(path)] = ba.sha(path)
    bound.update(common)
    for path in (U / 'import-receipt.json',):
        bound[str(path)] = ba.sha(path)
    subsets = {}
    for arm, root in ROOTS.items():
        table = root / 'Image_orientations_dataset1.xyz'
        if ba.sha(table) != manifests[arm]['orientation_sha256'] or ba.sha(table) != '523ac45c41e097b7396a66239c1be932f4c0e05c6e354667cfcfa5d57a783b8b':
            raise ValueError('Publisher table identity changed')
        bound[str(table)] = ba.sha(table)
        rows = {}
        for line in table.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) != 15 or fields[0] in rows:
                raise ValueError('Malformed publisher rows')
            rows[fields[0]] = list(map(float, fields[1:]))
        if set(rows) != set(image_hashes[arm]):
            raise ValueError('Publisher table names changed')
        chosen = {}
        for name in names:
            record = audit[name]['metashape' if arm == 'M' else 'undistorted_full']
            path = root / 'images' / name
            digest = image_hashes[arm][name]
            if digest != record['sha256'] or record['dimensions'] != DIMENSIONS[arm] or ba.sha(path) != digest:
                raise ValueError('Subset raster differs from audited/manifest member')
            if arm == 'U' and receipt['outputs'].get(str(path)) != digest:
                raise ValueError('U receipt identity mismatch')
            if not np.isfinite(rows[name]).all() or rows[name][6:9] != [4636.912, 3970.288, -2601.56] or any(rows[name][9:]):
                raise ValueError('Publisher fixed calibration changed')
            chosen[name] = {'path': str(path), 'sha256': digest, 'archive_member': record['archive_member']}
            bound[str(path)] = digest
        subsets[arm] = {'images': chosen, 'rows': {n: rows[n] for n in names}, 'dimensions': DIMENSIONS[arm],
                        'source_manifest_sha256': pins[str(root / 'input_manifest.json')], 'table_sha256': bound[str(table)]}
    ba.verify(bound)
    if OUTPUT.exists() or shutil.disk_usage(ba.BASE).free < LIMITS['startup_free_bytes']:
        raise RuntimeError('Fresh output and 8 GiB free required')
    return {'schema': 'publisher-raster-pair/1', 'formal_candidate': False, 'parameters': PARAMETERS,
        'limits': LIMITS, 'names': names, 'pairs': list(map(list, combinations(names, 2))), 'subsets': subsets,
        'bindings': bound, 'worker_common_bindings': common, 'runtime': launch['runtime'], 'settings': settings(options())}


def isolation(case, manifest):
    allowed = [p for p in manifest['worker_bindings'] if not any(
        Path(p).is_relative_to(manifest['runtime'][k]) for k in ('prefix', 'base_prefix'))] + [str(case / 'inputs.json')]
    reference = ba.ROOT / 'out/usegeo_benchmark/prepared/v1/evaluator-only/Dataset-1'
    # Real denied-open witnesses; unselected pixels and all old geometry stay outside the allowlist.
    all_names = sorted(x['image_id'] for x in ba.read_json(U / 'input_manifest.json')['images'])
    forbidden = [root / 'images' / all_names[16] for root in ROOTS.values()]
    forbidden += [ba.SFM / 'database.db', *[ba.MODEL / f for f in ba.MODEL_SHA],
                  *[reference / f for f in ('full_lidar.las', 'refined_lidar.las', 'publisher_mvs.las')],
                  ba.ROOT / 'out/usegeo_mesh_benchmark/paper-readiness-20260918/full-score-pass-1/Dataset-1/score.json']
    ba.write(case / 'isolation-request.json', {'output': str(case), 'python': str(ba.RUNTIME / 'bin/python'),
        'allowed': allowed, 'forbidden': list(map(str, forbidden))})
    script = """import {readFileSync,writeFileSync} from 'node:fs';
const {makeProfile,probeIsolation}=await import(process.argv[2]);
const s=JSON.parse(readFileSync(process.argv[1],'utf8')),start=performance.now();
const profile=makeProfile(s.output,s.python,s.allowed);
const proof=probeIsolation(profile,s.output,s.python,s.allowed,s.forbidden,true);
writeFileSync(s.output+'/isolation.json',JSON.stringify({...proof,profile,probe_seconds:(performance.now()-start)/1000}),{flag:'wx'});"""
    with (case / 'isolation.stdout').open('xb') as out, (case / 'isolation.stderr').open('xb') as err:
        subprocess.run([str(projection.NODE), '--input-type=module', '-e', script,
            str(case / 'isolation-request.json'), projection.ISOLATION.as_uri()], stdout=out, stderr=err, timeout=120, check=True)
    return ba.read_json(case / 'isolation.json')


def worker(case, manifest, check_only):
    if manifest['parameters'] != PARAMETERS or manifest['limits'] != LIMITS:
        raise ValueError('Protocol drift')
    ba.verify(manifest['worker_bindings'])
    proof, opts = self_test(), options()
    if settings(opts) != manifest['settings']:
        raise ValueError('Actual native options drift')
    ba.write(case / 'actual-options.json', settings(opts))
    names, pairs = manifest['names'], manifest['pairs']
    if len(names) != 16 or pairs != list(map(list, combinations(names, 2))) or len(pairs) != 120:
        raise ValueError('Fixed subset/pair denominator changed')
    if check_only:
        ba.verify(manifest['worker_bindings'])
        ba.write(case / 'worker-result.json', {'status': 'checked_without_real_extract_match_or_residuals',
            'arm': manifest['arm'], 'self_test': proof, 'images': len(names), 'all_pairs': len(pairs)})
        return
    database = case / 'database.db'
    if database.exists():
        raise FileExistsError(database)
    pc.set_random_seed(917)
    pc.extract_features(str(database), str(case / 'images'), image_names=names,
        camera_mode=pc.CameraMode.PER_IMAGE, camera_model='SIMPLE_RADIAL',
        reader_options=opts['reader'], sift_options=opts['extraction'], device=pc.Device.cpu)
    db = pc.Database(str(database))
    try:
        before = camera_records(db, names, manifest['subset']['dimensions'])
        generated = pc.ExhaustivePairGenerator(opts['exhaustive'], db).all_pairs()
        lookup = {r['image_id']: r['name'] for r in before}
        actual_pairs = {tuple(sorted((lookup[i], lookup[j]))) for i, j in generated}
        if actual_pairs != set(map(tuple, pairs)) or len(generated) != 120:
            raise ValueError('Exhaustive native generator differs from declared 120 pairs')
    finally:
        db.close()
    ba.write(case / 'cameras-before-matching.json', before)
    pc.set_random_seed(917)
    pc.match_exhaustive(str(database), sift_options=opts['matching'], matching_options=opts['exhaustive'],
                        verification_options=opts['verification'], device=pc.Device.cpu)
    db = pc.Database(str(database))
    summaries, distances, statuses, masks = [], [], [], []
    try:
        after = camera_records(db, names, manifest['subset']['dimensions'])
        ba.write(case / 'cameras-after-matching.json', after)
        if before != after:
            raise ValueError('Matching changed camera identities or calibration')
        ids = {r['name']: r['image_id'] for r in after}
        keypoints = {name: db.read_keypoints(ids[name]) for name in names}
        ba.write(case / 'feature-counts.json', {n: {'image_id': ids[n], 'keypoints': len(k),
            'descriptor_rows': len(db.read_descriptors(ids[n]))} for n, k in keypoints.items()})
        with (case / 'pair-summary.jsonl').open('x') as ledger:
            for index, (first, second) in enumerate(pairs):
                raw, verified, native = read_pair(db, ids[first], ids[second])
                if len(raw) and (raw[:, 0].max() >= len(keypoints[first]) or raw[:, 1].max() >= len(keypoints[second])):
                    raise ValueError('Raw match index outside fresh feature observations')
                xy1 = keypoints[first][raw[:, 0], :2].astype(float)
                xy2 = keypoints[second][raw[:, 1], :2].astype(float)
                independent, mask = rgb_estimate(xy1, xy2, opts['rgb_ransac'])
                F = projection.fundamental(projection.camera(manifest['subset']['rows'][first], .5),
                                           projection.camera(manifest['subset']['rows'][second], .5))
                distance, status = projection.sampson(np.broadcast_to(F, (len(raw), 3, 3)), xy1, xy2)
                path = case / f'pair-{index:03d}.npz'
                with path.open('xb') as stream:
                    np.savez(stream, raw_matches=raw, xy1=xy1, xy2=xy2, native_inlier_matches=verified,
                        rgb_F_inlier_mask=mask, publisher_F=F, publisher_distance_px=distance, publisher_status=status)
                record = {'pair_index': index, 'names': [first, second], 'image_ids': [ids[first], ids[second]],
                    'raw_match_count': len(raw), 'native_verified_count': len(verified), **native,
                    'rgb_estimator': independent, 'rgb_seed': 917, 'rgb_seed_order_index': index,
                    'all_raw': distribution(distance, status), 'rgb_estimator_subset': distribution(distance[mask], status[mask]),
                    'arrays': {'path': str(path), 'sha256': ba.sha(path)}}
                ledger.write(json.dumps(record, allow_nan=False)+'\n')
                ledger.flush()
                summaries.append(record)
                distances.append(distance)
                statuses.append(status)
                masks.append(mask)
    finally:
        db.close()
    distance, status, mask = np.concatenate(distances), np.concatenate(statuses), np.concatenate(masks)
    ba.verify(manifest['worker_bindings'])
    ba.write(case / 'worker-result.json', {'status': 'completed', 'arm': manifest['arm'], 'formal_candidate': False,
        'all_pairs': 120, 'images': 16, 'zero_match_pairs': sum(r['raw_match_count'] == 0 for r in summaries),
        'rgb_estimator_status_counts': dict(Counter(r['rgb_estimator']['status'] for r in summaries)),
        'all_raw': distribution(distance, status), 'rgb_estimator_subset': distribution(distance[mask], status[mask]),
        'database_sha256': ba.sha(database), 'global_weighting': 'raw-match weighted; not independent samples',
        'status_codes': projection.STATUS})


def main():
    if not __debug__ or Path(sys.prefix).resolve() != ba.RUNTIME.resolve() or pc.__version__ != '3.12.6':
        raise RuntimeError('Use fixed baseline-runtime -B, never -O')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('self-test', 'check', 'run', '_check_worker', '_worker'))
    parser.add_argument('manifest', nargs='?')
    args = parser.parse_args()
    if args.action == 'self-test':
        print(self_test())
        return
    if args.action.startswith('_'):
        path = Path(args.manifest).resolve()
        worker(path.parent, ba.read_json(path), args.action == '_check_worker')
        return
    campaign = preflight()
    root = Path(tempfile.mkdtemp(prefix='publisher-raster-pair-check-')).resolve() if args.action == 'check' else OUTPUT
    if args.action == 'run':
        root.mkdir(exist_ok=False)
    ba.write(root / 'inputs.json', campaign)
    started, results = time.monotonic(), {}
    try:
        for arm in ('M', 'U'):
            case = root / arm
            (case / 'images').mkdir(parents=True)
            copied = {}
            for name, original in campaign['subsets'][arm]['images'].items():
                path = case / 'images' / name
                shutil.copyfile(original['path'], path)
                path.chmod(0o444)
                copied[str(path)] = original['sha256']
            ba.verify(copied)
            manifest = {k: campaign[k] for k in ('parameters', 'limits', 'names', 'pairs', 'runtime', 'settings')}
            manifest.update(arm=arm, subset=campaign['subsets'][arm], copied_input_hashes=copied,
                campaign_inputs_sha256=ba.sha(root / 'inputs.json'),
                worker_bindings={**campaign['worker_common_bindings'], **copied})
            ba.write(case / 'inputs.json', manifest)
            probe = isolation(case, manifest)
            command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONDONTWRITEBYTECODE=1',
                'TMPDIR='+str(case), 'OPENSSL_CONF=/dev/null', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
                'MKL_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1', '/usr/bin/sandbox-exec', '-f', probe['profile'],
                str(ba.RUNTIME / 'bin/python'), '-B', str(ENTRY),
                '_check_worker' if args.action == 'check' else '_worker', str(case / 'inputs.json')]
            ba.write(case / 'command.json', {'argv': command, 'limits': LIMITS,
                'inputs_sha256': ba.sha(case / 'inputs.json'), 'profile_sha256': ba.sha(probe['profile'])})
            supervision = ba.supervise(command, case, timeout=600, max_rss=LIMITS['max_rss_bytes'])
            supervision['final_log_hashes'] = {str(case / n): ba.sha(case / n) for n in ('runtime.stdout', 'runtime.stderr')}
            ba.write(case / 'supervision.json', supervision)
            ba.verify(campaign['bindings'])
            ba.verify(copied)
            if supervision['returncode'] or supervision['reason'] is not None:
                raise RuntimeError(f'{arm} failed; preserve outputs: {supervision}')
            result = ba.read_json(case / 'worker-result.json')
            result.update(supervision=supervision, isolation=probe, input_manifest_sha256=ba.sha(case / 'inputs.json'),
                output_hashes={str(p): ba.sha(p) for p in sorted(case.rglob('*')) if p.is_file()})
            ba.write(case / 'result.json', result)
            results[arm] = {'path': str(case / 'result.json'), 'sha256': ba.sha(case / 'result.json'), 'status': result['status']}
        ba.write(root / 'result.json', {'status': 'checked' if args.action == 'check' else 'completed',
            'formal_candidate': False, 'arms': results, 'elapsed_seconds': time.monotonic()-started})
        print(json.dumps({'root': str(root), 'arms': results}))
    except BaseException as exc:
        ba.write(root / 'failure.json', {'status': 'failed', 'error': repr(exc), 'completed_arms': results,
                                       'elapsed_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
