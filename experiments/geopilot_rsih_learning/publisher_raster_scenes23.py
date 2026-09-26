"""Explicit-scene U2/U3 checks and common-prefix F/S sparse controls only."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ENTRY = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location('scenes23_sparse_helpers', ENTRY.with_name('publisher_raster_full_sparse.py'))
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)
tri, fixed, ba, pc, np = fs.tri, fs.fixed, fs.ba, fs.pc, fs.np
BASE = ba.BASE
PLAN = BASE / 'scenes23-counterfactual-plan.md'
CHECKS = BASE / 'scenes23-entry-check'
RUNS = BASE / 'publisher-raster-scenes23-sparse-run'
PRIOR = BASE / 'publisher-raster-full-sparse-run/inputs.json'
PRIOR_SHA = '6cb1ae12fb0da1eb0b917dbe358a19ffd0e3ba90dafb94b06cc093c6fffe0526'
AUDIT = BASE / 'publisher_raster_scenes23_input_audit.py'
spec = importlib.util.spec_from_file_location('scenes23_jpeg_header', AUDIT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
SCENES = {
    'Dataset-2': {'count': 327, 'dimensions': [7954, 5279], 'K': [4640.301, 3969.820, 2601.328, 0.],
                  'C0': [498483.116, 4379370.545, 197.655],
                  'manifest_sha256': 'f462e1b486189392ff46c1722c9b48acbeb1edaf2197404e82dd8dc17b7697b9',
                  'table_sha256': 'b346a979677eb36777eb148133c003c6eafe27985f52ee0724659fb70b722b1c',
                  'receipt_sha256': 'b8481176a510798ca2e997c64306bb934d80ca406a2ed0d311f62fb98975d3f2'},
    'Dataset-3': {'count': 277, 'dimensions': [7955, 5279], 'K': [4637.852, 3971.688, 2600.188, 0.],
                  'C0': [498436.821, 4379060.558, 178.320],
                  'manifest_sha256': 'ea25a2b1193add52148899d4f54ad6e8b2cbff33652132dadfcbf924726b80d4',
                  'table_sha256': 'f227cc2645c435db3cb46dee50e74e7cd2ab55cf1af470e5ae5720e9ae292be6',
                  'receipt_sha256': '92a74a256e2a3fb35bdce2ee7735c097fbffbc3fb638bb9a416f4013175a5e09'}}
LIMITS = {'features': [3600, 8 * 1024**3], 'matches': [3600, 8 * 1024**3],
          'F-U': [3600, 12 * 1024**3], 'S-U': [7200, 12 * 1024**3],
          'probe': [120, 8 * 1024**3], 'min_free_bytes': 30 * 1024**3}
STAGES = ('features', 'matches', 'F-U', 'S-U')
ACTIVE_RECONSTRUCTION = re.compile(
    r'DensifyPointCloud|ReconstructMesh|RefineMesh|InterfaceCOLMAP|/evaluator/|'
    r'publisher_raster_scenes23\.py\s+(?:_worker|sparse|check\b(?=[^\n]*--full\b))|'
    r'publisher_raster_full_sparse\.py\s+(?:_worker|run|check)\b|'
    r'geopilot_rsih/(?:(?:tools|supervise)\.py|launch\.mjs)|native_launch\.mjs.*--action[ =]+run')


def scene_record(name):
    """Only the two frozen U identities; never infer D1 defaults."""
    if name not in SCENES:
        raise ValueError('Only Dataset-2 and Dataset-3 are registered here')
    return {'scene_id': name, **copy.deepcopy(SCENES[name])}


def input_root(scene):
    return BASE / 'publisher-raster-input-scenes23' / scene['scene_id']


def orientation_path(scene):
    return input_root(scene) / f"Image_orientations_dataset{scene['scene_id'][-1]}.xyz"


def options(scene, names):
    """Retain U1 algorithmic options; replace only this returned object's K."""
    value = fs.options(names)
    value['reader'].camera_params = ','.join(format(v, '.17g') for v in scene['K'])
    return value


def settings(scene, names):
    value = options(scene, names)
    return {'shared_and_S': fixed.plain({k: v.todict() for k, v in value.items()}),
            'F-U': fixed.settings(fixed.options(names)), 'pycolmap': pc.__version__,
            'numpy': np.__version__, 'python': sys.version,
            'S_entry': 'incremental_mapping without input_path; XYZ only for pairs and final Sim3'}


def validate_metadata(scene, manifest, rows):
    """Check scene, count, names and calibration together, before file access."""
    if scene != scene_record(scene['scene_id']):
        raise ValueError('Scene specification drift')
    entries = manifest['images']
    names = sorted(e['image_id'] for e in entries)
    if (manifest['scene_id'] != scene['scene_id'] or manifest['image_count'] != scene['count']
            or len(names) != scene['count'] or len(set(names)) != len(names)
            or manifest['image_dimensions'] != scene['dimensions']
            or manifest['track'] != 'manual-publisher-undistorted'
            or manifest['formal_candidate'] is not False or manifest['benchmark_eligible'] is not False
            or 'publisher orientation table' not in manifest['permissions']
            or any(Path(n).name != n or not n.endswith('.jpg') for n in names)
            or any(not re.fullmatch('[0-9a-f]{64}', e['sha256']) for e in entries)):
        raise ValueError('Scene/count/raster/name/development identity mismatch')
    if (set(rows) != set(names) or any(len(r) != 14 for r in rows.values())
            or not np.isfinite(list(rows.values())).all()
            or any([r[6], r[7] + .5, -r[8] + .5, 0.] != scene['K'] or any(r[9:])
                   for r in rows.values()) or rows[names[0]][:3] != scene['C0']):
        raise ValueError('Publisher rows/calibration/C0 mismatch')
    return names


def camera_identity(db, scene, names):
    images = sorted(db.read_all_images(), key=lambda i: i.name)
    cameras = {c.camera_id: c for c in db.read_all_cameras()}
    rigs = {r.rig_id: r for r in db.read_all_rigs()}
    frames = {f.frame_id: f for f in db.read_all_frames()}
    ids = {i.image_id for i in images}
    if (len(images) != scene['count'] or [i.name for i in images] != names
            or ids != set(cameras) or ids != set(rigs) or ids != set(frames)):
        raise ValueError('Expected N independent image/camera/rig/frame IDs')
    result = []
    for image in images:
        i = image.image_id
        c, r, f = cameras[i], rigs[i], frames[i]
        # Database image reads do not hydrate frame_id; frame.data_ids binds it.
        if (image.camera_id != i or c.model != pc.CameraModelId.SIMPLE_RADIAL
                or [c.width, c.height] != scene['dimensions'] or not np.array_equal(c.params, scene['K'])
                or not c.has_prior_focal_length or r.num_sensors() != 1 or f.rig_id != i
                or (r.ref_sensor_id.type, r.ref_sensor_id.id) != (c.sensor_id.type, i)
                or f.num_data_ids() != 1
                or {(d.sensor_id.type, d.sensor_id.id, d.id) for d in f.data_ids} != {(c.sensor_id.type, i, i)}):
            raise ValueError('Native camera/calibration/frame identity mismatch')
        result.append({'id': i, 'name': image.name, 'K': c.params.tolist(), 'dimensions': scene['dimensions']})
    return result


def toy_database(path, scene, names, centers):
    db = pc.Database(str(path))
    for i, name in enumerate(names, 1):
        camera = pc.Camera(camera_id=i, model='SIMPLE_RADIAL', width=scene['dimensions'][0],
                           height=scene['dimensions'][1], params=scene['K'], has_prior_focal_length=True)
        db.write_camera(camera, use_camera_id=True)
        rig = pc.Rig(rig_id=i)
        rig.add_ref_sensor(camera.sensor_id)
        db.write_rig(rig, use_rig_id=True)
        image = pc.Image(image_id=i, name=name, camera_id=i)
        db.write_image(image, use_image_id=True)
        frame = pc.Frame(frame_id=i, rig_id=i)
        frame.add_data_id(image.data_id)
        db.write_frame(frame, use_frame_id=True)
        db.write_pose_prior(i, pc.PosePrior(np.asarray(centers[name]), pc.PosePriorCoordinateSystem.CARTESIAN))
        db.write_keypoints(i, np.empty((0, 2), np.float32))
    return db


def fixed_model(db, scene, rows, names):
    model, origin = pc.Reconstruction(), np.asarray(rows[names[0]][:3])
    if origin.tolist() != scene['C0']:
        raise ValueError('Wrong scene origin')
    for camera in camera_identity(db, scene, names):
        tri.add_camera(model, camera['id'], camera['name'], rows[camera['name']], origin, scene['dimensions'])
    return model, {'origin_image': names[0], 'C0': origin.tolist(),
                   'local_to_world': np.column_stack((np.eye(3), origin)).tolist()}


def inherited_bindings():
    ba.verify({str(PRIOR): PRIOR_SHA})
    prior = ba.read_json(PRIOR)
    runtime = prior['runtime']
    source_files = [Path(fs.__file__), Path(tri.__file__), Path(fixed.__file__),
                    Path(fixed.projection.__file__), Path(ba.__file__),
                    ba.ROOT / 'code/usegeo_mesh_baseline/runner.py', ba.ROOT / 'experiments/geopilot_rsi/run.py',
                    *[ba.ROOT / 'code/geopilot_rsih' / name for name in
                      ('tools.py', 'strict_json.py', 'supervise.py', 'isolation.mjs')], fixed.projection.NODE]
    common = {str(p): prior['worker_common_bindings'][str(p)] for p in source_files}
    # Explicit supervisor-only revision; all numerical helpers remain pinned to U1.
    supervisor = str(ba.ROOT / 'code/geopilot_rsih/supervise.py')
    if common[supervisor] != 'ba40d43efcf6b006b52fe55204c39a1244b44a07016a57a92a2a16baaffa59de':
        raise ValueError('Unexpected inherited supervision version')
    common[supervisor] = 'a35bbda70c027342994fe8e6e49ba4b1f94e9bf4f64ea400215abf2a949fd796'
    ba.verify(common)
    runtime_bindings = {p: h for p, h in prior['worker_common_bindings'].items()
                        if any(Path(p).is_relative_to(runtime[k]) for k in ('prefix', 'base_prefix'))}
    common.update({str(p): ba.sha(p) for p in (ENTRY, AUDIT)})
    return runtime, common, runtime_bindings


def metadata(scene):
    root = input_root(scene)
    paths = {str(root / 'input_manifest.json'): scene['manifest_sha256'],
             str(root / 'import-receipt.json'): scene['receipt_sha256'],
             str(orientation_path(scene)): scene['table_sha256']}
    if root.is_symlink() or (root / 'images').is_symlink():
        raise ValueError('Input directory symlink forbidden')
    for path in paths:
        ba.legacy.sparse.regular(Path(path), root)
    ba.verify(paths)
    manifest, receipt = ba.read_json(root / 'input_manifest.json'), ba.read_json(root / 'import-receipt.json')
    rows = {}
    for line in orientation_path(scene).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 15 or fields[0] in rows:
            raise ValueError('Malformed/duplicate orientation row')
        rows[fields[0]] = list(map(float, fields[1:]))
    names = validate_metadata(scene, manifest, rows)
    if (receipt['status'] != 'complete' or manifest['orientation_sha256'] != scene['table_sha256']
            or receipt['outputs'].get(str(orientation_path(scene))) != scene['table_sha256']
            or receipt['outputs'].get(str(root / 'input_manifest.json')) != scene['manifest_sha256']
            or set(p.name for p in (root / 'images').iterdir()) != set(names)):
        raise ValueError('Import receipt/directory mismatch')
    images = {str(root / 'images' / e['image_id']): e['sha256'] for e in manifest['images']}
    headers = {}
    for path, expected in images.items():
        ba.legacy.sparse.regular(Path(path), root)
        if receipt['outputs'].get(path) != expected:
            raise ValueError('Image receipt mismatch')
        with Path(path).open('rb') as stream:
            headers[path] = audit.jpeg_sof(stream)
        if headers[path]['dimensions'] != scene['dimensions']:
            raise ValueError('JPEG SOF dimension mismatch')
    runtime, common, runtime_bindings = inherited_bindings()
    return {'schema': 'publisher-raster-scenes23-sparse/1', 'scene': scene, 'formal_candidate': False,
            'benchmark_eligible': False, 'names': names, 'rows': rows,
            'centers': {n: rows[n][:3] for n in names}, 'images': images, 'headers': headers,
            'runtime': runtime, 'worker_common_bindings': common, 'runtime_bindings': runtime_bindings,
            'bindings': {**paths, str(PRIOR): PRIOR_SHA, str(PLAN): ba.sha(PLAN), **common},
            'limits': LIMITS, 'settings': settings(scene, names)}


def stage_manifest(campaign, root, case, stage, probe_only=False):
    if stage not in STAGES:
        raise ValueError('Unknown stage')
    m = {k: campaign[k] for k in ('scene', 'names', 'centers', 'runtime', 'settings', 'pairs', 'limits')}
    m.update(stage=stage, probe_only=probe_only, campaign_inputs_sha256=ba.sha(root / 'inputs.json'),
             worker_bindings={**campaign['worker_common_bindings'], **campaign['runtime_bindings'], **campaign['images']})
    if stage == 'F-U':
        m['rows'] = campaign['rows']
    reference = ba.ROOT / 'out/usegeo_benchmark/prepared/v1/evaluator-only'
    forbidden = [reference / f'Dataset-{n}' / filename for n in (1, 2, 3)
                 for filename in ('full_lidar.las', 'refined_lidar.las', 'publisher_mvs.las')]
    forbidden += [ba.MODEL / 'cameras.bin', fs.U / 'Image_orientations_dataset1.xyz']
    if stage != 'F-U':
        forbidden += [orientation_path(campaign['scene']), root / 'inputs.json', root / 'F-U/model/cameras.bin']
    m['denied_inputs'] = [str(p) for p in forbidden if p.is_file()]
    m['denied_roots'] = [str(reference), str(ba.SOURCE), str(fs.OUTPUT)]
    if stage != 'F-U':
        m['denied_roots'].append(str(root / 'F-U'))
    if set(m['worker_bindings']) & set(m['denied_inputs']):
        raise ValueError('Allowed/forbidden input overlap')
    return m


def isolate(case, m):
    allowed = [p for p in m['worker_bindings'] if not any(
        Path(p).is_relative_to(m['runtime'][k]) for k in ('prefix', 'base_prefix'))]
    allowed.append(str(case / 'inputs.json'))
    ba.write(case / 'isolation-request.json', {'output': str(case), 'python': str(ba.RUNTIME / 'bin/python'),
             'allowed': allowed, 'forbidden': m['denied_inputs'], 'denied_roots': m['denied_roots']})
    script = """const fs=await import('node:fs');
const {makeProfile,probeIsolation}=await import(process.argv[2]);
const s=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const profile=makeProfile(s.output,s.python,s.allowed);
fs.appendFileSync(profile,'\\n(deny file-read* file-write* '+s.denied_roots.map(p=>'(subpath '+JSON.stringify(p)+')').join(' ')+')\\n');
const proof=probeIsolation(profile,s.output,s.python,s.allowed,s.forbidden,true);
fs.writeFileSync(s.output+'/isolation.json',JSON.stringify({...proof,profile}),{flag:'wx'});"""
    with (case / 'isolation.stdout').open('xb') as out, (case / 'isolation.stderr').open('xb') as err:
        subprocess.run([str(fixed.projection.NODE), '--input-type=module', '-e', script,
                        str(case / 'isolation-request.json'), fixed.projection.ISOLATION.as_uri()],
                       stdout=out, stderr=err, timeout=120, check=True)
    return ba.read_json(case / 'isolation.json')


def independent_database(source, case):
    """Never share an inode between the immutable common prefix and branch DBs."""
    original = Path(source['path'])
    (case / 'input').mkdir()
    frozen, work = case / 'input/database.db', case / 'work/database.db'
    shutil.copyfile(original, frozen)
    frozen.chmod(0o444)
    shutil.copyfile(frozen, work)
    if len({(p.stat().st_dev, p.stat().st_ino) for p in (original, frozen, work)}) != 3:
        raise ValueError('Shared database inode')
    ba.verify({str(p): source['sha256'] for p in (original, frozen, work)})
    return frozen


def rank_models(results):
    return sorted(results, key=lambda key: (-results[key]['registered'], -results[key]['points3D'], int(key)))


class NumericalStageFailure(RuntimeError):
    """A supervised F/S numerical call failed, with its retained evidence."""

    def __init__(self, stage, status, isolation, call):
        super().__init__(f'{stage} numerical call failed: {status}')
        self.status, self.isolation, self.call = status, isolation, call


def numerical_call(case, name, function, *args, **kwargs):
    """Distinguish an active native call from Python input/invariant failures."""
    record = {'api': name, 'state': 'started'}
    ba.write(case / 'numerical-call.json', record)
    try:
        value = function(*args, **kwargs)
    except Exception as exc:
        record.update(state='numerical_exception' if isinstance(exc, RuntimeError) else 'unexpected_exception',
                      error=repr(exc))
        ba.write(case / 'numerical-return.json', record)
        raise
    ba.write(case / 'numerical-return.json', {**record, 'state': 'returned'})
    return value


def is_numerical_failure(stage, status, call):
    expected = {'F-U': 'triangulate_points', 'S-U': 'incremental_mapping'}
    if call.get('api') != expected.get(stage) or stage not in expected:
        return False
    if status['reason'] is None:
        return ((status['returncode'] != 0 and call.get('state') == 'numerical_exception')
                or (status['returncode'] < 0 and call.get('state') == 'started'))
    return (call.get('state') == 'started'
            and status['reason'] in ('RuntimeError: TIME_LIMIT', 'RuntimeError: RSS_LIMIT'))


def fixed_branch(case, db, m):
    scene, names, o = m['scene'], m['names'], fixed.options(m['names'])
    model, transform = fixed_model(db, scene, m['rows'], names)
    before = fixed.snapshot(model)
    cache = pc.DatabaseCache.create(db, o.min_num_matches, o.ignore_watermarks, set(names))
    support = fs.cache_report(db, cache, m['pairs'], case / 'work/database.db')
    ba.write(case / 'cache-support.json', support)
    model.load(cache)
    if not fixed.compare(before, fixed.snapshot(model))['passed'] or model.num_points3D():
        raise ValueError('Zero model cache load changed cameras')
    initial, target = case / 'initialized-model', case / 'model'
    initial.mkdir(); target.mkdir()
    model.write_binary(str(initial))
    if not fixed.compare(before, fixed.snapshot(pc.Reconstruction(str(initial))))['passed']:
        raise ValueError('Fixed initialization roundtrip drift')
    ba.write(case / 'initialization.json', {'snapshot': before, 'transform': transform})
    db.close()
    called = support['cache_images'] > 0
    if called:
        pc.set_random_seed(917)
        model = numerical_call(case, 'triangulate_points', pc.triangulate_points, model,
                                      str(case / 'work/database.db'), str(input_root(scene) / 'images'),
                                      str(target), clear_points=True, options=o, refine_intrinsics=False)
    summary = fs.model_report(model, names)
    comparison = fixed.compare(before, fixed.snapshot(model))
    if not comparison['passed']:
        raise ValueError('Fixed publisher camera drift')
    model.write_binary(str(target))
    if not fixed.compare(before, fixed.snapshot(pc.Reconstruction(str(target))))['passed']:
        raise ValueError('Fixed output roundtrip drift')
    ba.write(case / 'model-report.json', summary)
    record = {'path': str(target), 'report_path': str(case / 'model-report.json'),
              'registered': summary['registered'], 'points3D': summary['points3D'],
              'output_hashes': {str(p): ba.sha(p) for p in target.iterdir() if p.is_file()}}
    return {'outcome': 'observed_points' if summary['points3D'] else 'zero_points_no_retry',
            'triangulation_api_called': called, 'all_models': {'fixed': record}, 'model_ranking': ['fixed'],
            'selected_model': {'id': 'fixed', 'path': str(target), 'transform': transform['local_to_world']}
            if summary['points3D'] else None, 'transform': transform, 'camera_comparison': comparison}


def sfm_branch(case, db, m, o):
    support = fs.cache_report(db, pc.DatabaseCache.create(db, o.min_num_matches, o.ignore_watermarks,
                                                       set(m['names'])), m['pairs'], case / 'work/database.db')
    ba.write(case / 'cache-support.json', support)
    db.close()
    target = case / 'models'
    target.mkdir()
    pc.set_random_seed(917)
    models = numerical_call(case, 'incremental_mapping', pc.incremental_mapping,
                                    str(case / 'work/database.db'), str(input_root(m['scene']) / 'images'),
                                    str(target), options=o) if support['cache_images'] else {}
    results = {}
    for ident, model in sorted(models.items()):
        summary = fs.model_report(model, m['names'])
        if any(c['K'][1:3] != m['scene']['K'][1:3] or c['dimensions'] != m['scene']['dimensions']
               for c in summary['cameras']):
            raise ValueError('S fixed principal point/raster drift')
        folder = target / str(ident)
        folder.mkdir(exist_ok=True)
        model.write_binary(str(folder))
        ids = sorted(model.reg_image_ids(), key=lambda i: model.images[i].name)
        try:
            transform, residual = ba.legacy.sparse.fit_centers(
                [model.images[i].projection_center() for i in ids],
                [m['centers'][model.images[i].name] for i in ids])
            alignment = {'status': 'available', 'matrix': transform.matrix().tolist(), 'scale': transform.scale,
                         'residuals': [{'name': model.images[i].name, 'meters': float(r)} for i, r in zip(ids, residual)]}
        except ValueError as exc:
            alignment = {'status': 'unavailable', 'reason': str(exc)}
        ba.write(folder / 'report.json', {**summary, 'allowed_XYZ_alignment': alignment})
        results[str(ident)] = {'path': str(folder), 'registered': summary['registered'],
            'points3D': summary['points3D'], 'alignment_status': alignment['status'], 'transform': alignment.get('matrix'),
            'report_path': str(folder / 'report.json'),
            'output_hashes': {str(p): ba.sha(p) for p in folder.iterdir() if p.is_file()}}
    if not models:
        ba.write(case / 'no-model-report.json', fs.model_report(pc.Reconstruction(), m['names']))
    ranking = rank_models(results)
    selected = {'id': ranking[0], 'path': results[ranking[0]]['path'],
                'transform': results[ranking[0]]['transform']} if ranking else None
    return {'outcome': 'models_returned' if models else 'no_models_no_retry', 'all_models': results,
            'model_ranking': ranking, 'selected_model': selected, 'mapper_api_called': bool(support['cache_images'])}


def worker(case, m, scene):
    if (m['scene'] != scene or m['limits'] != LIMITS or settings(scene, m['names']) != m['settings']
            or m['stage'] not in STAGES or len(m['names']) != scene['count']):
        raise ValueError('Worker scene/settings drift')
    if m['stage'] != 'F-U' and 'rows' in m:
        raise ValueError('Non-F worker must not receive OPK rows')
    ba.verify(m['worker_bindings'])
    for path in m['denied_inputs']:
        try:
            with Path(path).open('rb'):
                raise ValueError('Worker can read prohibited input: ' + path)
        except PermissionError:
            pass
    ba.write(case / 'denied-inputs.json', {'status': 'PASS', 'files': m['denied_inputs']})
    if m['probe_only']:
        ba.write(case / 'worker-result.json', {'status': 'permission_probe_only', 'scene_id': scene['scene_id']})
        return
    stage, names, opts = m['stage'], m['names'], options(scene, m['names'])
    work = case / 'work/database.db'
    result = {'status': 'completed', 'scene_id': scene['scene_id'], 'stage': stage,
              'input_image_count': scene['count'], 'formal_candidate': False, 'benchmark_eligible': False}
    ba.write(case / 'actual-options.json', settings(scene, names))
    if stage == 'features':
        pc.set_random_seed(917)
        pc.extract_features(str(work), str(input_root(scene) / 'images'), image_names=names,
            camera_mode=pc.CameraMode.PER_IMAGE, camera_model='SIMPLE_RADIAL', reader_options=opts['reader'],
            sift_options=opts['extraction'], device=pc.Device.cpu)
        db = pc.Database(str(work))
        try:
            cameras = camera_identity(db, scene, names)
            for image in db.read_all_images():
                db.write_pose_prior(image.image_id, pc.PosePrior(np.asarray(m['centers'][image.name]),
                                                               pc.PosePriorCoordinateSystem.CARTESIAN))
            if fs.name_pairs(db, opts) != m['pairs']:
                raise ValueError('Feature IDs changed declared name pairs')
            ba.write(case / 'cameras.json', cameras)
            ba.write(case / 'feature-counts.json', {i.name: len(db.read_keypoints(i.image_id)) for i in db.read_all_images()})
        finally:
            db.close()
    else:
        ba.verify({str(work): m['database_sha256']})
        before = fixed.database_identity(work)
        ba.write(case / 'database-before.json', before)
        db = pc.Database(str(work))
        camera_identity(db, scene, names)
        if fs.name_pairs(db, opts) != m['pairs']:
            raise ValueError('Copied database pair denominator drift')
        if stage == 'matches':
            db.close()
            pc.set_random_seed(917)
            pc.match_sequential(str(work), sift_options=opts['matching'], matching_options=opts['sequential'],
                                verification_options=opts['verification'], device=pc.Device.cpu)
            pc.match_spatial(str(work), sift_options=opts['matching'], matching_options=opts['spatial'],
                             verification_options=opts['verification'], device=pc.Device.cpu)
            db = pc.Database(str(work))
            try:
                camera_identity(db, scene, names)
                ba.write(case / 'cache-support.json', fs.cache_report(
                    db, pc.DatabaseCache.create(db, 15, False, set(names)), m['pairs'], work))
            finally:
                db.close()
        else:
            result.update(fixed_branch(case, db, m) if stage == 'F-U' else sfm_branch(case, db, m, opts['mapper']))
        after = fixed.database_identity(work)
        ba.write(case / 'database-after.json', after)
        allowed = {'matches', 'two_view_geometries'} if stage == 'matches' else set()
        unchanged = set(before) == set(after) and all(before[k] == after[k] for k in set(before) - allowed)
        ba.write(case / 'database-provenance.json', {'before_sha256': m['database_sha256'],
                  'after_sha256': ba.sha(work), 'protected_tables_unchanged': unchanged,
                  'allowed_table_changes': sorted(allowed), 'native_header_changes_recorded': True})
        if not unchanged:
            raise ValueError('Protected database contents changed')
    result['database_sha256'] = ba.sha(work)
    ba.verify(m['worker_bindings'])
    ba.write(case / 'worker-result.json', result)


def active_processes(listing, own_pid):
    """Ignore this process only; another scene's identical parent stays active."""
    active = []
    for line in listing.splitlines():
        fields = line.split(None, 1)
        if not fields:
            continue
        if len(fields) != 2 or not fields[0].isdigit():
            raise ValueError('Malformed process listing')
        if int(fields[0]) != own_pid and ACTIVE_RECONSTRUCTION.search(fields[1]):
            active.append(line.strip())
    return active


def ensure_idle():
    result = subprocess.run(['/bin/ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True)
    active = active_processes(result.stdout, os.getpid())
    if active:
        raise RuntimeError('Other reconstruction/scoring active: ' + repr(active))
    if shutil.disk_usage(BASE).free < LIMITS['min_free_bytes']:
        raise RuntimeError('At least 30 GiB free disk required')


def launch_worker(case, m):
    ba.write(case / 'inputs.json', m)
    probe = isolate(case, m)
    command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONDONTWRITEBYTECODE=1',
        'TMPDIR=' + str(case), 'OPENSSL_CONF=/dev/null', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
        'MKL_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1', '/usr/bin/sandbox-exec', '-f', probe['profile'],
        str(ba.RUNTIME / 'bin/python'), '-B', str(ENTRY), '_worker', '--scene', m['scene']['scene_id'],
        '--manifest', str(case / 'inputs.json')]
    limits = LIMITS['probe' if m['probe_only'] else m['stage']]
    ba.write(case / 'command.json', {'argv': command, 'inputs_sha256': ba.sha(case / 'inputs.json'),
              'profile_sha256': ba.sha(probe['profile']), 'isolation_sha256': ba.sha(case / 'isolation.json'),
              'limits': limits})
    status = ba.supervise(command, case, timeout=limits[0], max_rss=limits[1])
    status['final_log_hashes'] = {str(case / n): ba.sha(case / n) for n in ('runtime.stdout', 'runtime.stderr')}
    ba.write(case / 'supervision.json', status)
    if status['returncode'] or status['reason'] is not None:
        call_path = case / 'numerical-return.json'
        if not call_path.is_file():
            call_path = case / 'numerical-call.json'
        call = ba.read_json(call_path) if call_path.is_file() else {}
        if not m['probe_only'] and is_numerical_failure(m['stage'], status, call):
            raise NumericalStageFailure(m['stage'], status, probe, call)
        raise RuntimeError(f"{m['stage']} failed outside a classified numerical call; preserve output: {status}")
    return ba.read_json(case / 'worker-result.json'), status, probe



def failure_flow_check(root):
    """Exercise actual orchestration using tiny files and explicit worker stubs."""
    from unittest.mock import patch
    active = {'api': 'triangulate_points', 'state': 'started'}
    assert is_numerical_failure('F-U', {'returncode': -11, 'reason': None}, active)
    assert is_numerical_failure('F-U', {'returncode': -15, 'reason': 'RuntimeError: TIME_LIMIT'}, active)
    assert is_numerical_failure('F-U', {'returncode': -15, 'reason': 'RuntimeError: RSS_LIMIT'}, active)
    for status, call in [({'returncode': 1, 'reason': None}, active),
                         ({'returncode': -15, 'reason': 'KeyboardInterrupt: Cancelled'}, active),
                         ({'returncode': -11, 'reason': None}, {**active, 'state': 'returned'}),
                         ({'returncode': 1, 'reason': None}, {**active, 'state': 'unexpected_exception'})]:
        assert not is_numerical_failure('F-U', status, call)
    scene = scene_record('Dataset-2')
    canonical = {'scene': scene, 'rows': {'a.jpg': [1., 2., 3., 4.]}, 'centers': {'a.jpg': [1., 2., 3.]},
                 'names': ['a.jpg'], 'images': {'a.jpg': '0' * 64}, 'limits': LIMITS, 'runtime_bindings': {}}
    prepared = {**copy.deepcopy(canonical), 'pairs': {}}
    receipt = {'status': 'full_preflight_passed', 'scene_id': scene['scene_id'], 'entry_sha256': ba.sha(ENTRY),
               'image_full_bytes_hashed': True, 'runtime_full_hashes_checked': True,
               'permission_checks': {stage: {'sandbox_executed': True,
                    'result': {'status': 'permission_probe_only', 'scene_id': scene['scene_id']},
                    'supervision': {'returncode': 0, 'reason': None}} for stage in ('F-U', 'S-U')}}
    validate_preflight(scene, receipt, prepared, canonical)
    tampered = []
    for key in ('rows', 'centers', 'names', 'images', 'limits', 'runtime_bindings'):
        bad = copy.deepcopy(prepared); bad[key] = {'changed': True}; tampered.append((receipt, bad))
    for key, value in (('permission_checks', {}), ('runtime_full_hashes_checked', False)):
        bad = copy.deepcopy(receipt); bad[key] = value; tampered.append((bad, prepared))
    for changed_receipt, changed_inputs in tampered:
        try:
            validate_preflight(scene, changed_receipt, changed_inputs, canonical)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed prepared metadata accepted')
    outcomes = {}
    for scenario in ('arm_failure', 'prefix_failure', 'contract_failure', 'source_changed'):
        with tempfile.TemporaryDirectory(dir=root) as folder:
            case_root = Path(folder)
            small_input = case_root / 'small-image-fixture'; small_input.write_bytes(b'fixture')
            campaign = {'scene': scene_record('Dataset-2'), 'names': [], 'centers': {}, 'rows': {},
                        'runtime': {}, 'settings': {}, 'pairs': {}, 'limits': LIMITS,
                        'worker_common_bindings': {}, 'runtime_bindings': {},
                        'images': {str(small_input): ba.sha(small_input)}, 'bindings': {}}
            ba.write(case_root / 'inputs.json', campaign)
            calls = []

            def fake_launch(case, m):
                stage = m['stage']; calls.append(stage)
                ba.write(case / 'inputs.json', m)
                profile = case / 'profile'; profile.write_text('synthetic fixture only')
                isolation = {'profile': str(profile)}
                ba.write(case / 'isolation.json', isolation)
                ba.write(case / 'command.json', {'inputs_sha256': ba.sha(case / 'inputs.json'),
                         'profile_sha256': ba.sha(profile), 'isolation_sha256': ba.sha(case / 'isolation.json')})
                if stage == 'matches' and scenario == 'prefix_failure':
                    raise RuntimeError('synthetic common prefix failure')
                if stage in ('F-U', 'S-U'):
                    assert (case / 'work/database.db').read_bytes() == b'matches'
                    assert Path(m['source_database']['path']).read_bytes() == b'matches'
                if stage == 'F-U':
                    if scenario == 'contract_failure':
                        raise ValueError('synthetic fixed camera drift')
                    if scenario == 'source_changed':
                        small_input.write_bytes(b'changed')
                    (case / 'work/database.db').write_bytes(b'failed F work, never reused')
                    raise NumericalStageFailure(stage, {'returncode': 1, 'reason': None}, isolation,
                                                {**active, 'state': 'numerical_exception'})
                work = case / 'work/database.db'
                if stage in ('features', 'matches'):
                    work.write_bytes(stage.encode())
                return {'status': 'completed', 'database_sha256': ba.sha(work)}, {}, isolation

            # Test doubles are confined to this entry; no imported module/global changes.
            with patch.dict(globals(), ensure_idle=lambda: None, launch_worker=fake_launch):
                if scenario == 'arm_failure':
                    run_stages(campaign, case_root)
                    assert calls == list(STAGES)
                    assert ba.read_json(case_root / 'result.json')['status'] == 'completed_with_arm_failure'
                    assert ba.read_json(case_root / 'F-U/failure.json')['status'] == 'numerical_failure'
                else:
                    try:
                        run_stages(campaign, case_root)
                    except (ValueError, RuntimeError):
                        pass
                    else:
                        raise AssertionError('Unsafe continuation accepted: ' + scenario)
                    assert 'S-U' not in calls
                    assert ba.read_json(case_root / 'failure.json')['status'] == 'blocked'
            outcomes[scenario] = calls
    return {'scenarios': outcomes, 'prepared_artifact_tampering_rejected': len(tampered)}


def self_check(root):
    """Small zero-feature fixtures exercise scene mistakes and branch permissions."""
    checked = []
    for name in SCENES:
        scene = scene_record(name)
        names = [f'{i:04d}.jpg' for i in range(scene['count'])]
        offsets = np.random.default_rng(917).uniform(-100, 100, (len(names), 3))
        offsets[0] = 0
        centers = {n: (np.asarray(scene['C0']) + delta).tolist() for n, delta in zip(names, offsets)}
        rows = {n: [*centers[n], 0., 0., 0., scene['K'][0], scene['K'][1] - .5,
                    .5 - scene['K'][2], 0., 0., 0., 0., 0.] for n in names}
        manifest = {'scene_id': name, 'image_count': scene['count'], 'image_dimensions': scene['dimensions'],
                    'track': 'manual-publisher-undistorted', 'formal_candidate': False, 'benchmark_eligible': False,
                    'permissions': ['publisher orientation table'],
                    'images': [{'image_id': n, 'sha256': '0' * 64} for n in names]}
        assert validate_metadata(scene, manifest, rows) == names
        bad_inputs = []
        for field, value in [('scene_id', 'Dataset-1'), ('image_count', 224), ('image_dimensions', [7953, 5279])]:
            bad = copy.deepcopy(manifest); bad[field] = value; bad_inputs.append((scene, bad, rows))
        bad = copy.deepcopy(manifest); bad['images'][0]['image_id'] = '../escape.jpg'; bad_inputs.append((scene, bad, rows))
        bad = copy.deepcopy(manifest); bad['images'][0] = bad['images'][1]; bad_inputs.append((scene, bad, rows))
        bad = copy.deepcopy(rows); bad[names[0]][0] = float('nan'); bad_inputs.append((scene, manifest, bad))
        bad = copy.deepcopy(rows); bad[names[0]][6] = 4636.912; bad_inputs.append((scene, manifest, bad))
        bad = copy.deepcopy(scene); bad['K'] = fs.K; bad_inputs.append((bad, manifest, rows))
        for args in bad_inputs:
            try:
                validate_metadata(*args)
            except ValueError:
                continue
            raise AssertionError('Malformed fixture accepted')
        with tempfile.TemporaryDirectory(dir=root) as folder:
            path = Path(folder)
            db = toy_database(path / 'fixture.db', scene, names, centers)
            try:
                assert len(camera_identity(db, scene, names)) == scene['count']
                pairs = fs.name_pairs(db, options(scene, names))
                sequential = {tuple((names[i], names[j])) for i in range(len(names))
                              for j in range(i + 1, min(i + 6, len(names)))}
                assert {tuple(p) for p in pairs['sequential']} == sequential
                coords = np.asarray([centers[n] for n in names])
                coords = (coords - coords.mean(axis=0)).astype(np.float32)
                distances = np.sum((coords[:, None] - coords[None, :]) ** 2, axis=2)
                spatial = {tuple(sorted((names[i], names[j]))) for i in range(len(names))
                           for j in np.argsort(distances[i])[:11] if i != j}
                assert {tuple(p) for p in pairs['spatial']} == spatial
                assert {tuple(p) for p in pairs['union']} == sequential | {tuple(p) for p in pairs['spatial']}
                model, transform = fixed_model(db, scene, rows, names)
                before = fixed.snapshot(model)
                model.load(pc.DatabaseCache.create(db, 15, False, set(names)))
                assert fixed.compare(before, fixed.snapshot(model))['passed'] and not model.num_points3D()
                target = path / 'zero-model'; target.mkdir(); model.write_binary(str(target))
                assert fixed.compare(before, fixed.snapshot(pc.Reconstruction(str(target))))['passed']
                assert transform['C0'] == scene['C0']
                wrong = copy.deepcopy(scene); wrong['K'] = fs.K
                try:
                    camera_identity(db, wrong, names)
                except ValueError:
                    pass
                else:
                    raise AssertionError('D1 calibration accepted')
                with sqlite3.connect(path / 'fixture.db') as connection:
                    connection.execute('UPDATE images SET camera_id=1 WHERE image_id=2')
                try:
                    camera_identity(db, scene, names)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Shared camera ID accepted')
            finally:
                db.close()
            case = path / 'branch'; (case / 'work').mkdir(parents=True)
            independent_database({'path': str(path / 'fixture.db'), 'sha256': ba.sha(path / 'fixture.db')}, case)
        checked.append({'scene': name, 'fixture_images': len(names), 'invalid_cases_rejected': len(bad_inputs),
                        'pair_counts': {k: len(v) for k, v in pairs.items()}, 'zero_model_roundtrip': True,
                        'shared_camera_id_rejected': True, 'independent_branch_database_copy': True,
                        'spatial_nearest10_independent_reconstruction': True})
    assert rank_models({'2': {'registered': 5, 'points3D': 9}, '1': {'registered': 5, 'points3D': 9},
                        '0': {'registered': 4, 'points3D': 99}}) == ['1', '2', '0']
    assert ACTIVE_RECONSTRUCTION.search('node native_launch.mjs --action run')
    assert ACTIVE_RECONSTRUCTION.search('python /code/geopilot_rsih/supervise.py campaign')
    assert ACTIVE_RECONSTRUCTION.search('node /code/geopilot_rsih/launch.mjs --program P0 --scope development')
    assert ACTIVE_RECONSTRUCTION.search('python publisher_raster_scenes23.py sparse --scene Dataset-2')
    assert ACTIVE_RECONSTRUCTION.search('python publisher_raster_scenes23.py check --scene Dataset-3 --full')
    assert not ACTIVE_RECONSTRUCTION.search('python publisher_raster_scenes23.py check --scene Dataset-3')
    parent = 'python publisher_raster_scenes23.py sparse --scene Dataset-2'
    assert active_processes('100 ' + parent + '\n101 ' + parent, 100) == ['101 ' + parent]
    return {'status': 'PASS', 'fixtures': checked, 'model_ranking': 'PASS',
            'campaign_concurrency_recognition': 'PASS', 'failure_flow': failure_flow_check(root),
            'forbidden_numerical_calls_executed': False}


def check(scene, root, full):
    root.mkdir(parents=True, exist_ok=False)
    proof = self_check(root)
    campaign = metadata(scene)
    db = toy_database(root / 'pair-selection.db', scene, campaign['names'], campaign['centers'])
    try:
        camera_identity(db, scene, campaign['names'])
        campaign['pairs'] = fs.name_pairs(db, options(scene, campaign['names']))
        model, _ = fixed_model(db, scene, campaign['rows'], campaign['names'])
        target = root / 'F-U/model'; target.mkdir(parents=True); model.write_binary(str(target))
    finally:
        db.close()
    ba.write(root / 'inputs.json', campaign)
    ba.write(root / 'candidate-pairs.json', campaign['pairs'])
    permission_checks = {}
    for stage in ('F-U', 'S-U'):
        case = root / (stage + '-probe'); (case / 'work').mkdir(parents=True)
        m = stage_manifest(campaign, root, case, stage, probe_only=True)
        assert ('rows' in m) == (stage == 'F-U')
        if stage == 'S-U':
            assert {str(orientation_path(scene)), str(root / 'inputs.json'), str(target / 'cameras.bin')} <= set(m['denied_inputs'])
        ba.write(case / 'planned-inputs.json', m)
        permission_checks[stage] = {'manifest_boundary': 'PASS', 'sandbox_executed': False}
    if full:
        ensure_idle()
        ba.verify(campaign['runtime_bindings'])
        ba.verify(campaign['images'])
        for stage in ('F-U', 'S-U'):
            case = root / (stage + '-probe')
            m = ba.read_json(case / 'planned-inputs.json')
            result, status, isolation = launch_worker(case, m)
            permission_checks[stage].update(sandbox_executed=True, result=result, supervision=status, isolation=isolation)
    ba.verify(campaign['bindings'])
    result = {'status': 'full_preflight_passed' if full else 'metadata_and_synthetic_only', 'scene_id': scene['scene_id'],
              'self_check': proof, 'pair_counts': {k: len(v) for k, v in campaign['pairs'].items()},
              'image_prefix_bytes_returned': sum(h['prefix_bytes_returned_through_sof'] for h in campaign['headers'].values()),
              'image_full_bytes_hashed': full, 'runtime_full_hashes_checked': full,
              'permission_checks': permission_checks, 'numerical_experiment_executed': False,
              'inputs_sha256': ba.sha(root / 'inputs.json'), 'entry_sha256': ba.sha(ENTRY)}
    ba.write(root / 'result.json', result)
    print(json.dumps({'root': str(root), 'status': result['status'], 'pair_counts': result['pair_counts']}))



def validate_preflight(scene, receipt, campaign, canonical):
    """A prepared artifact cannot authorize changed publisher metadata."""
    permissions = receipt.get('permission_checks', {})
    if (receipt.get('status') != 'full_preflight_passed' or receipt.get('scene_id') != scene['scene_id']
            or receipt.get('entry_sha256') != ba.sha(ENTRY) or receipt.get('image_full_bytes_hashed') is not True
            or receipt.get('runtime_full_hashes_checked') is not True or set(permissions) != {'F-U', 'S-U'}
            or any(p.get('sandbox_executed') is not True
                   or p.get('result') != {'status': 'permission_probe_only', 'scene_id': scene['scene_id']}
                   or p.get('supervision', {}).get('returncode') != 0
                   or p.get('supervision', {}).get('reason', 'missing') is not None for p in permissions.values())):
        raise ValueError('Full scene-matched preflight required')
    if (set(campaign) != set(canonical) | {'pairs'}
            or any(campaign[key] != value for key, value in canonical.items())):
        raise ValueError('Prepared metadata differs from fixed publisher sources')


def sparse(scene, checked):
    ensure_idle()
    receipt = ba.read_json(checked / 'result.json')
    ba.verify({str(checked / 'inputs.json'): receipt['inputs_sha256']})
    campaign = ba.read_json(checked / 'inputs.json')
    validate_preflight(scene, receipt, campaign, metadata(scene))
    ba.verify({**campaign['bindings'], **campaign['runtime_bindings'], **campaign['images']})
    root = RUNS / scene['scene_id']; root.mkdir(parents=True, exist_ok=False)
    ba.write(root / 'inputs.json', campaign)
    ba.write(root / 'candidate-pairs.json', campaign['pairs'])
    run_stages(campaign, root)


def run_stages(campaign, root):
    """Run one common prefix and both arms; only classified arm failures continue."""
    scene = campaign['scene']
    results, databases, started = {}, {}, time.monotonic()
    try:
        for stage in STAGES:
            ensure_idle()
            case = root / stage; (case / 'work').mkdir(parents=True)
            m = stage_manifest(campaign, root, case, stage)
            if stage != 'features':
                source = databases['features' if stage == 'matches' else 'matches']
                frozen = independent_database(source, case)
                m.update(database_sha256=source['sha256'], source_database=source)
                m['worker_bindings'][str(frozen)] = source['sha256']
            try:
                result, status, isolation = launch_worker(case, m)
            except NumericalStageFailure as exc:
                if stage not in ('F-U', 'S-U'):
                    raise
                # Prefix, sandbox and non-numerical errors never reach this handler.
                result = {'status': 'numerical_failure', 'stage': stage, 'error': str(exc),
                          'supervision': exc.status, 'isolation': exc.isolation, 'numerical_call': exc.call,
                          'inputs_sha256': ba.sha(case / 'inputs.json')}
                ba.write(case / 'failure.json', result)
                results[stage] = {'path': str(case / 'failure.json'), 'sha256': ba.sha(case / 'failure.json'),
                                  'status': result['status']}
                command = ba.read_json(case / 'command.json')
                ba.verify({**campaign['bindings'], **campaign['runtime_bindings'], **campaign['images'],
                           databases['matches']['path']: databases['matches']['sha256'],
                           str(case / 'inputs.json'): command['inputs_sha256'],
                           exc.isolation['profile']: command['profile_sha256'],
                           str(case / 'isolation.json'): command['isolation_sha256']})
                continue
            ba.verify(campaign['bindings'])
            if stage in ('features', 'matches'):
                target = case / 'snapshot.db'; shutil.copyfile(case / 'work/database.db', target); target.chmod(0o444)
                ba.verify({str(target): result['database_sha256']})
                databases[stage] = {'path': str(target), 'sha256': result['database_sha256']}
                result['frozen_database'] = databases[stage]
            result.update(supervision=status, isolation=isolation, inputs_sha256=ba.sha(case / 'inputs.json'),
                          output_hashes={str(p): ba.sha(p) for p in sorted(case.rglob('*')) if p.is_file()})
            ba.write(case / 'result.json', result)
            results[stage] = {'path': str(case / 'result.json'), 'sha256': ba.sha(case / 'result.json'), 'status': result['status']}
        ba.verify(campaign['images'])
        state = 'completed_with_arm_failure' if any(r['status'] == 'numerical_failure' for r in results.values()) else 'completed'
        ba.write(root / 'result.json', {'status': state, 'scene_id': scene['scene_id'],
                 'formal_candidate': False, 'benchmark_eligible': False, 'stages': results,
                 'elapsed_seconds': time.monotonic() - started})
    except BaseException as exc:
        ba.write(root / 'failure.json', {'status': 'blocked', 'error': repr(exc), 'recorded_stages': results,
                                         'elapsed_seconds': time.monotonic() - started})
        raise
    print(json.dumps({'root': str(root), 'stages': results}))


def main():
    if not __debug__ or Path(sys.prefix).resolve() != ba.RUNTIME.resolve() or pc.__version__ != '3.12.6':
        raise RuntimeError('Use the frozen numerical Python with -B and without -O')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'sparse', '_worker'))
    parser.add_argument('--scene', choices=tuple(SCENES), required=True)
    parser.add_argument('--full', action='store_true', help='check: also hash all inputs/runtime and execute real isolation probes')
    parser.add_argument('--output', type=Path, help='check output; must be fresh under scenes23-entry-check')
    parser.add_argument('--checked', type=Path, help='sparse: previously completed full check directory')
    parser.add_argument('--manifest', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    scene = scene_record(args.scene)
    if args.action == '_worker':
        worker(args.manifest.resolve().parent, ba.read_json(args.manifest), scene)
    elif args.action == 'check':
        root = (args.output or CHECKS / (args.scene + ('-full' if args.full else '-metadata'))).resolve()
        if not root.is_relative_to(CHECKS.resolve()) or root == CHECKS.resolve():
            raise ValueError('Check outputs must stay below scenes23-entry-check')
        check(scene, root, args.full)
    else:
        if args.checked is None or args.full or args.output is not None:
            parser.error('sparse requires --checked only')
        sparse(scene, args.checked.resolve())


if __name__ == '__main__':
    main()
