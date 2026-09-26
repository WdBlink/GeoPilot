"""Fixed OPALS metadata cameras and cached RGB matches; sparse triangulation only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ENTRY = Path(__file__).resolve()
PROJECTION = ENTRY.with_name('publisher_projection_diagnostic_r2.py')
spec = importlib.util.spec_from_file_location('fixed_projection', PROJECTION)
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)
ba, np, pc = projection.ba, projection.np, projection.pc
PLAN = ba.BASE / 'fixed-publisher-camera-plan.md'
API = ba.BASE / 'fixed-metadata-baseline-api.md'
OUTPUT = ba.BASE / 'fixed-publisher-camera-run'
DATABASE = ba.SFM / 'database.db'
DB_SHA = 'bb107084e6edee4d9d6960bb6bb51ffef932c34110d97fa5208e5e01325a7d6e'
LIMITS = {'timeout_seconds': 1800, 'max_rss_bytes': 8 * 1024**3,
          'min_free_bytes': 4 * 1024**3, 'num_threads': 1}
PARAMETERS = {'delta_px': .5, 'camera_model': 'SIMPLE_RADIAL', 'k': 0.0,
              'origin': 'publisher center of lexicographically first image', 'random_seed': 917,
              'clear_points': True, 'refine_intrinsics': False,
              'max_K_difference': 0.0, 'max_R_element_difference': 1.e-12,
              'max_center_difference_m': 1.e-9}


def plain(value):
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def options(names):
    o = pc.IncrementalPipelineOptions()
    o.image_names = sorted(names)
    o.num_threads = o.mapper.num_threads = 1
    o.fix_existing_frames = o.mapper.fix_existing_frames = True
    o.use_prior_position = o.ba_use_gpu = False
    o.ba_refine_focal_length = o.ba_refine_principal_point = o.ba_refine_extra_params = False
    o.ba_refine_sensor_from_rig = False
    o.mapper.abs_pose_refine_focal_length = o.mapper.abs_pose_refine_extra_params = False
    if not o.check():
        raise ValueError('Invalid fixed pipeline options')
    return o


def settings(o):
    return plain({'pipeline': o.todict(), 'effective_mapper': o.get_mapper().todict(),
                  'effective_triangulation': o.get_triangulation().todict(),
                  'global_bundle_adjustment': o.get_global_bundle_adjustment().todict(),
                  'local_bundle_adjustment': o.get_local_bundle_adjustment().todict(),
                  'triangulate_kwargs': {'clear_points': True, 'refine_intrinsics': False},
                  'pose_freeze': 'all existing frames are constant via fix_existing_frames; '
                                 'BA getter refine_rig_from_world=True does not unfix those frames'})


def publisher_rows():
    manifest = ba.read_json(ba.INPUTS / 'input_manifest.json')
    rows = {}
    for line in projection.TABLE.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 15 or fields[0] in rows:
            raise ValueError('Malformed publisher row')
        rows[fields[0]] = np.asarray(fields[1:], dtype=float)
    if (manifest['track'] != 'rgb-oriented' or 'publisher orientation table' not in manifest['permissions']
            or len(rows) != 224 or set(rows) != {x['image_id'] for x in manifest['images']}
            or not np.isfinite(list(rows.values())).all()
            or any(list(r[6:9]) != [4636.912, 3970.288, -2601.560]
                   or np.any(r[9:] != 0) for r in rows.values())):
        raise ValueError('Publisher input identity/calibration mismatch')
    return rows


def database_audit(path, rows):
    """Only the copied database: validate cached observations and pair identity, not old geometry."""
    expected = ba.read_json(ba.SFM / 'pairs.json')
    inputs = ba.read_json(ba.SFM / 'inputs.json')
    manifest = ba.read_json(ba.INPUTS / 'input_manifest.json')
    if (inputs['input_manifest_sha256'] != ba.sha(ba.INPUTS / 'input_manifest.json')
            or inputs['orientation_sha256'] != ba.sha(projection.TABLE)
            or inputs['images'] != {x['image_id']: x['sha256'] for x in manifest['images']}):
        raise ValueError('Cached feature inputs differ from allowed RGB/table')
    connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        image_names = dict(connection.execute('SELECT image_id,name FROM images'))
        if ({str(i): n for i, n in image_names.items()} != expected['image_names']
                or set(image_names.values()) != set(rows) or len(image_names) != 224):
            raise ValueError('Database image IDs/names mismatch')
        keypoints = {}
        for ident, count, columns, data in connection.execute('SELECT image_id,rows,cols,data FROM keypoints'):
            xy = np.frombuffer(data, dtype='<f4')
            if (columns not in (2, 4, 6) or xy.size != count*columns
                    or not np.isfinite(xy).all() or count <= 0):
                raise ValueError('Invalid cached keypoints')
            keypoints[ident] = count
        if set(keypoints) != set(image_names) or sum(keypoints.values()) != 2634249:
            raise ValueError('Keypoint count/coverage changed')
        for ident, count, columns, size in connection.execute('SELECT image_id,rows,cols,length(data) FROM descriptors'):
            if count != keypoints[ident] or columns != 128 or size != count*columns:
                raise ValueError('Descriptor/keypoint identity mismatch')
        expected_pairs = {tuple(p) for p in expected['union']}
        counts, verified_support = {}, {i: 0 for i in image_names}
        for table in ('matches', 'two_view_geometries'):
            pairs, nonzero, total = set(), 0, 0
            for pair_id, count, columns, data in connection.execute(f'SELECT pair_id,rows,cols,data FROM {table}'):
                i, j = divmod(pair_id, 2147483647)
                pairs.add((i, j))
                matches = np.frombuffer(data or b'', dtype='<u4')
                if columns != 2 or matches.size != 2*count or i >= j or i not in keypoints or j not in keypoints:
                    raise ValueError('Invalid pair/match array')
                if count and (np.max(matches[::2]) >= keypoints[i] or np.max(matches[1::2]) >= keypoints[j]):
                    raise ValueError('Match index outside source keypoints')
                total += count
                nonzero += count > 0
                if table == 'two_view_geometries' and count >= 15:
                    verified_support[i] += count
                    verified_support[j] += count
            if pairs != expected_pairs or len(pairs) != 1766:
                raise ValueError('Pair set mismatch')
            counts[table] = {'pairs': len(pairs), 'nonzero_pairs': nonzero, 'correspondences': total}
        if counts['two_view_geometries']['nonzero_pairs'] != 1750 or min(verified_support.values()) == 0:
            raise ValueError('Verified correspondence coverage changed')
        return {'image_names': image_names, 'keypoints': keypoints, 'keypoints_total': sum(keypoints.values()),
                'pair_counts': counts, 'verified_support_at_min_15': verified_support,
                'pose_prior_rows_not_used': connection.execute('SELECT count(*) FROM pose_priors').fetchone()[0]}
    finally:
        connection.close()


def database_identity(path):
    """Hash each ordered SQLite table including schema; retain header changes separately."""
    wal = Path(str(path) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Pending WAL prevents a complete immutable database identity')
    connection = sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True)
    identity = {}
    try:
        for name, schema in connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name"):
            digest, count = hashlib.sha256(schema.encode()), 0
            quoted = '"' + name.replace('"', '""') + '"'
            for row in connection.execute(f'SELECT * FROM {quoted} ORDER BY rowid'):
                count += 1
                for value in row:
                    data = value if isinstance(value, bytes) else repr(value).encode()
                    digest.update(type(value).__name__.encode()+b'\0'+len(data).to_bytes(8, 'little')+data)
            identity[name] = {'rows': count, 'sha256': digest.hexdigest()}
        return identity
    finally:
        connection.close()


def add_camera(model, ident, name, row, origin):
    r, center, k = projection.camera(row, PARAMETERS['delta_px'])
    camera = pc.Camera(camera_id=ident, model='SIMPLE_RADIAL', width=7952, height=5304,
                       params=[k[0, 0], k[0, 2], k[1, 2], 0.], has_prior_focal_length=True)
    model.add_camera(camera)
    rig = pc.Rig(rig_id=ident)
    rig.add_ref_sensor(camera.sensor_id)
    model.add_rig(rig)
    image = pc.Image(name=name, camera_id=ident, image_id=ident)
    image.frame_id = ident
    frame = pc.Frame(frame_id=ident, rig_id=ident,
                     rig_from_world=pc.Rigid3d(np.column_stack((r, -r @ (center-origin)))))
    frame.add_data_id(image.data_id)
    model.add_frame(frame)
    model.add_image(image)
    model.register_frame(ident)


def snapshot(model):
    values = []
    for ident in sorted(model.images):
        image = model.images[ident]
        camera = model.cameras[image.camera_id]
        frame, rig = model.frames[image.frame_id], model.rigs[model.frames[image.frame_id].rig_id]
        values.append({'id': ident, 'name': image.name, 'camera_id': image.camera_id,
            'frame_id': image.frame_id, 'rig_id': frame.rig_id, 'registered': ident in model.reg_image_ids(),
            'model': str(camera.model), 'width': camera.width, 'height': camera.height,
            'sensor': [str(rig.ref_sensor_id.type), rig.ref_sensor_id.id],
            'K': camera.params.tolist(), 'R': image.cam_from_world().rotation.matrix().tolist(),
            'C': image.projection_center().tolist()})
    if any(not np.isfinite(x[k]).all() for x in values for k in ('K', 'R', 'C')):
        raise ValueError('Nonfinite camera')
    return {'cameras': values, 'counts': {'cameras': model.num_cameras(), 'rigs': model.num_rigs(),
            'frames': model.num_frames(), 'images': model.num_images(), 'registered': model.num_reg_images(),
            'points3D': model.num_points3D(), 'observations': model.compute_num_observations()}}


def compare(before, after):
    maxima = {'K': 0., 'R': 0., 'C': 0.}
    identity = len(before['cameras']) == len(after['cameras'])
    for x, y in zip(before['cameras'], after['cameras']):
        identity &= {k: v for k, v in x.items() if k not in maxima} == {k: v for k, v in y.items() if k not in maxima}
        for key in maxima:
            delta = np.asarray(x[key])-y[key]
            maxima[key] = max(maxima[key], float(np.linalg.norm(delta) if key == 'C' else np.max(np.abs(delta))))
    identity &= all(before['counts'][k] == after['counts'][k] for k in ('images', 'registered', 'cameras', 'rigs', 'frames'))
    passed = (identity and maxima['K'] <= PARAMETERS['max_K_difference']
              and maxima['R'] <= PARAMETERS['max_R_element_difference']
              and maxima['C'] <= PARAMETERS['max_center_difference_m'])
    return {'passed': bool(passed), 'identity_equal': bool(identity), 'max_K_difference': maxima['K'],
            'max_R_element_difference': maxima['R'], 'max_center_difference_m': maxima['C']}


def build_model(db, rows):
    images = {i.image_id: i for i in db.read_all_images()}
    cameras = {c.camera_id: c for c in db.read_all_cameras()}
    rigs = {r.rig_id: r for r in db.read_all_rigs()}
    frames = {f.frame_id: f for f in db.read_all_frames()}
    if len(images) != 224 or not set(images) == set(cameras) == set(rigs) == set(frames):
        raise ValueError('Expected 224 independent camera/rig/frame IDs')
    origin_name = min(rows)
    origin = rows[origin_name][:3].copy()
    model = pc.Reconstruction()
    for ident, image in sorted(images.items()):
        cam, rig, frame = cameras[ident], rigs[ident], frames[ident]
        if (image.camera_id != ident or cam.model != pc.CameraModelId.SIMPLE_RADIAL
                or (cam.width, cam.height) != (7952, 5304) or rig.num_sensors() != 1
                or (rig.ref_sensor_id.type, rig.ref_sensor_id.id) != (cam.sensor_id.type, cam.sensor_id.id)
                or frame.rig_id != ident
                or frame.num_data_ids() != 1
                or {(d.sensor_id.type, d.sensor_id.id, d.id) for d in frame.data_ids}
                   != {(cam.sensor_id.type, ident, ident)}):
            raise ValueError('Database camera/rig/frame relation mismatch')
        add_camera(model, ident, image.name, rows[image.name], origin)
    if model.num_points3D() != 0 or model.compute_num_observations() != 0:
        raise ValueError('New metadata model is not empty')
    return model, {'origin_image': origin_name, 'C0': origin.tolist(),
                   'local_to_world': np.column_stack((np.eye(3), origin)).tolist()}


def self_test():
    projection.self_test()
    model = pc.Reconstruction()
    origin = np.array([500000., 4400000., 200.])
    for i in (1, 2):
        row = [*(origin + [i, 0., 0.]), 17., -9., 23., 4636.912, 3970.288, -2601.56, 0., 0., 0., 0., 0.]
        add_camera(model, i, f'{i}.jpg', row, origin)
    before = snapshot(model)
    assert all(x['point_observations'] == 0 for x in image_support(model))
    assert before['counts']['points3D'] == 0 and before['counts']['registered'] == 2
    with tempfile.TemporaryDirectory() as path:
        model.write_binary(path)
        after = snapshot(pc.Reconstruction(path))
    assert compare(before, after)['passed']
    changed = plain(after)
    changed['cameras'][0]['K'][0] += 1.e-6
    assert not compare(before, changed)['passed']
    changed = plain(after)
    changed['cameras'][0]['C'][0] += 1.e-6
    assert not compare(before, changed)['passed']
    return 'PASS: R2 algebra/boundaries, zero-point native roundtrip, K/center-change rejection'


def image_support(model):
    return [{'id': i, 'name': model.images[i].name, 'keypoints': len(model.images[i].points2D),
             'point_observations': model.images[i].num_points3D} for i in sorted(model.images)]


def preflight():
    if not __debug__ or Path(sys.prefix).resolve() != ba.RUNTIME.resolve() or pc.__version__ != '3.12.6':
        raise RuntimeError('Use frozen runtime with -B, never -O')
    pins = {str(PROJECTION): '83714c483084e38277c7dec8d71bacf2288d85591fc0344e0c3404eec1edfb9f',
            str(DATABASE): DB_SHA, str(ba.SOURCE / 'launch.json'): '2a5023086874bd1d138164634f109540fcf1ce57f9ab6a64a66c3d3955145e88'}
    ba.verify(pins)
    launch = ba.read_json(ba.SOURCE / 'launch.json')
    bound = {**pins, **launch['input_hashes']}
    # Deliberately do not call ba.bindings(): that helper binds old poses/3D, which are not inputs here.
    for path, digest in launch['bindings'].items():
        if any(Path(path).is_relative_to(launch['runtime'][k]) for k in ('prefix', 'base_prefix')):
            bound[path] = digest
    for name in ('code/geopilot_rsih/tools.py', 'code/geopilot_rsih/strict_json.py',
                 'code/geopilot_rsih/supervise.py', 'code/usegeo_mesh_baseline/runner.py', 'experiments/geopilot_rsi/run.py'):
        bound[str(ba.ROOT / name)] = launch['bindings'][str(ba.ROOT / name)]
    bound.update(projection.PINS)
    for path in (ENTRY, PLAN, API, projection.PLAN, projection.NODE, ba.SFM / 'inputs.json', ba.SFM / 'pairs.json'):
        bound[str(path)] = ba.sha(path)
    ba.verify(bound)
    rows = publisher_rows()
    if OUTPUT.exists() or DATABASE.stat().st_size != 446148608:
        raise ValueError('Fresh output required or database size changed')
    if shutil.disk_usage(ba.BASE).free < LIMITS['min_free_bytes']:
        raise RuntimeError('Less than 4 GiB free')
    worker = {p: h for p, h in bound.items() if p not in (str(DATABASE), str(ba.SOURCE / 'launch.json'))}
    return {'schema': 'fixed-publisher-camera/1', 'formal_candidate': False, 'official_conversion_certified': False,
            'baseline': 'fixed-OPALS-publisher-camera-delta-0.5', 'parameters': PARAMETERS, 'limits': LIMITS,
            'bindings': bound, 'worker_bindings': worker, 'runtime': launch['runtime'], 'settings': settings(options(rows)),
            'database_provenance': 'current frozen RGB feature/match snapshot; no historical DB hash claim'}


def isolate(case, manifest):
    allowed = [p for p in manifest['worker_bindings'] if not any(
        Path(p).is_relative_to(manifest['runtime'][k]) for k in ('prefix', 'base_prefix'))]
    allowed.append(str(case / 'inputs.json'))
    reference = ba.ROOT / 'out/usegeo_benchmark/prepared/v1/evaluator-only/Dataset-1'
    forbidden = [reference / f for f in ('full_lidar.las', 'refined_lidar.las', 'publisher_mvs.las')]
    forbidden += [DATABASE, *[ba.MODEL / f for f in ba.MODEL_SHA],
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
        raise ValueError('Worker protocol changed')
    ba.verify(manifest['worker_bindings'])
    proof, rows = self_test(), publisher_rows()
    o = options(rows)
    if settings(o) != manifest['settings']:
        raise ValueError('Actual options differ from declared options')
    ba.write(case / 'settings.json', settings(o))
    work_db = case / 'work/database.db'
    if ba.sha(work_db) != DB_SHA:
        raise ValueError('Working DB initial bytes differ from frozen snapshot')
    identity_before = database_identity(work_db)
    ba.write(case / 'database-identity-before.json', identity_before)
    audit = database_audit(work_db, rows)
    ba.write(case / 'database-audit.json', audit)
    db = pc.Database(str(work_db))
    try:
        model, transform = build_model(db, rows)
        before = snapshot(model)
        expected = plain(before)
        for camera in expected['cameras']:
            r, center, k = projection.camera(rows[camera['name']], PARAMETERS['delta_px'])
            camera.update(K=[k[0, 0], k[0, 2], k[1, 2], 0.], R=r.tolist(),
                          C=(center-np.asarray(transform['C0'])).tolist())
        initialization = compare(expected, before)
        ba.write(case / 'publisher-initialization-comparison.json', initialization)
        if not initialization['passed']:
            raise ValueError('Native initialization differs from declared publisher cameras')
        ba.write(case / 'before-load.json', before)
        ba.write(case / 'transform.json', transform)
        cache = pc.DatabaseCache.create(db, o.min_num_matches, o.ignore_watermarks, set(rows))
        if cache.num_images() != 224 or {i.name for i in cache.images.values()} != set(rows):
            raise ValueError('Native cache lost allowed images')
        model.load(cache)
    finally:
        db.close()
    loaded = snapshot(model)
    load_check = compare(before, loaded)
    ba.write(case / 'after-load.json', loaded)
    ba.write(case / 'loaded-image-support.json', image_support(model))
    ba.write(case / 'load-comparison.json', load_check)
    if not load_check['passed'] or model.num_points3D() != 0 or model.num_reg_images() != 224:
        raise ValueError('Native load changed fixed cameras or introduced points')
    initial_path = case / 'initialized-model'
    initial_path.mkdir()
    model.write_binary(str(initial_path))
    reread = pc.Reconstruction(str(initial_path))
    roundtrip = compare(before, snapshot(reread))
    ba.write(case / 'roundtrip-comparison.json', roundtrip)
    if not roundtrip['passed'] or reread.num_points3D() != 0:
        raise ValueError('Native binary roundtrip changed cameras')
    result = {'status': 'checked_without_triangulation', 'formal_candidate': False, 'self_test': proof,
              'counts': loaded['counts'], 'load_comparison': load_check, 'roundtrip_comparison': roundtrip,
              'initialization_comparison': initialization, 'database_snapshot_sha256': DB_SHA, 'transform': transform}
    if not check_only:
        pc.set_random_seed(PARAMETERS['random_seed'])
        output_model = case / 'model'
        output_model.mkdir()
        started = time.monotonic()
        model = pc.triangulate_points(model, str(work_db), str(ba.INPUTS / 'images'), str(output_model),
                                      clear_points=True, options=o, refine_intrinsics=False)
        elapsed = time.monotonic()-started
        model.update_point_3d_errors()
        after = snapshot(model)
        difference = compare(before, after)
        ba.write(case / 'after-triangulation.json', after)
        ba.write(case / 'camera-comparison.json', difference)
        if (not difference['passed'] or model.num_points3D() == 0 or model.compute_num_observations() == 0
                or model.num_reg_images() != 224
                or any(not np.isfinite(p.xyz).all() or not np.isfinite(p.error) for p in model.points3D.values())
                or any(c.params[0] <= 0 for c in model.cameras.values())):
            raise ValueError('Triangulation failed finite nonempty/fixed-camera gate')
        model.write_binary(str(output_model))
        if not compare(before, snapshot(pc.Reconstruction(str(output_model))))['passed']:
            raise ValueError('Final native roundtrip changed cameras')
        support = image_support(model)
        ba.write(case / 'image-support.json', support)
        result.update(status='completed', counts=after['counts'], triangulation_seconds=elapsed,
                      termination='API returned; postconditions passed', camera_comparison=difference,
                      refreshed_mean_reprojection_error_px=model.compute_mean_reprojection_error(),
                      model_path=str(output_model), images_without_points=sum(x['point_observations'] == 0 for x in support))
    identity_after = database_identity(work_db)
    ba.write(case / 'database-identity-after.json', identity_after)
    database_record = {'snapshot_sha256': DB_SHA, 'working_before_sha256': DB_SHA,
        'working_after_sha256': ba.sha(work_db), 'table_identity_unchanged': identity_before == identity_after,
        'native_database_may_update_sqlite_header': True}
    ba.write(case / 'database-provenance.json', database_record)
    if identity_before != identity_after:
        raise ValueError('Native API changed cached table contents')
    result['database_provenance'] = database_record
    ba.verify(manifest['worker_bindings'])
    ba.write(case / 'worker-result.json', result)


def main():
    if not __debug__ or Path(sys.prefix).resolve() != ba.RUNTIME.resolve() or pc.__version__ != '3.12.6':
        raise RuntimeError('Use frozen baseline-runtime with -B, never -O')
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
    manifest = preflight()
    case = Path(tempfile.mkdtemp(prefix='fixed-publisher-camera-check-')).resolve() if args.action == 'check' else OUTPUT
    if args.action == 'run':
        case.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        (case / 'input').mkdir()
        copied = case / 'input/database.db'
        shutil.copyfile(DATABASE, copied)
        manifest['copied_input_hashes'] = {str(copied): DB_SHA}
        ba.verify(manifest['copied_input_hashes'])
        copied.chmod(0o444)
        (case / 'work').mkdir()
        working = case / 'work/database.db'
        shutil.copyfile(copied, working)
        manifest['working_database'] = {'path': str(working), 'initial_sha256': DB_SHA,
            'may_write': True, 'require_same_table_identity_after': True}
        manifest['worker_bindings'].update(manifest['copied_input_hashes'])
        ba.write(case / 'inputs.json', manifest)
        isolation = isolate(case, manifest)
        command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONDONTWRITEBYTECODE=1',
            'TMPDIR=' + str(case), 'OPENSSL_CONF=/dev/null', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
            'VECLIB_MAXIMUM_THREADS=1', '/usr/bin/sandbox-exec', '-f', isolation['profile'],
            str(ba.RUNTIME / 'bin/python'), '-B', str(ENTRY),
            '_check_worker' if args.action == 'check' else '_worker', str(case / 'inputs.json')]
        ba.write(case / 'command.json', {'argv': command, 'limits': LIMITS, 'inputs_sha256': ba.sha(case / 'inputs.json'),
                                       'profile_sha256': ba.sha(isolation['profile'])})
        status = ba.supervise(command, case, timeout=LIMITS['timeout_seconds'], max_rss=LIMITS['max_rss_bytes'])
        status['final_log_hashes'] = {str(case / n): ba.sha(case / n) for n in ('runtime.stdout', 'runtime.stderr')}
        ba.write(case / 'supervision.json', status)
        ba.verify(manifest['bindings'])
        ba.verify(manifest['copied_input_hashes'])
        if status['returncode'] or status['reason'] is not None:
            raise RuntimeError('Worker failed; retain all outputs: ' + str(status))
        result = ba.read_json(case / 'worker-result.json')
        result.update(inputs_sha256=ba.sha(case / 'inputs.json'), supervision=status, isolation=isolation,
            wall_seconds=time.monotonic()-started, output_hashes={str(p): ba.sha(p) for p in sorted(case.rglob('*')) if p.is_file()})
        ba.write(case / 'result.json', result)
        print(json.dumps({'case': str(case), 'status': result['status'], 'counts': result['counts'], 'supervision': status}))
    except BaseException as exc:
        ba.write(case / 'failure.json', {'status': 'failed', 'error': repr(exc), 'wall_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
