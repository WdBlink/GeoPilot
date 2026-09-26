"""U224 shared fresh features/matches, fixed-publisher and conventional-SfM sparse controls."""
import argparse
import importlib.util
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ENTRY = Path(__file__).resolve()
HELPER = ENTRY.with_name('publisher_raster_triangulation_diagnostic.py')
spec = importlib.util.spec_from_file_location('full_sparse_helpers', HELPER)
tri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tri)
fixed, ba, pc, np = tri.fixed, tri.ba, tri.pc, tri.np
U = ba.BASE/'publisher-raster-input'
PLAN = ba.BASE/'publisher-raster-full-sparse-plan.md'
DESIGN = ba.BASE/'publisher-raster-experiment-design.md'
OUTPUT = ba.BASE/'publisher-raster-full-sparse-run'
K = [4636.912, 3970.788, 2602.060, 0.]
DIMENSIONS = [7953, 5279]
LIMITS = {'features': [1800, 8*1024**3], 'matches': [1800, 8*1024**3],
          'F-U': [1800, 12*1024**3], 'S-U': [3600, 12*1024**3],
          'check': [600, 8*1024**3], 'S-access': [600, 8*1024**3],
          'min_free_bytes': 8*1024**3, 'threads': 1}
PARAMETERS = {'images': 224, 'dimensions': DIMENSIONS, 'K': K, 'seed': 917,
    'camera_model': 'SIMPLE_RADIAL', 'camera_mode': 'PER_IMAGE', 'native_min_num_inliers': 15,
    'matching': 'sequential overlap5 nonquadratic no-loop UNION XYZ spatial nearest10 max1e9m',
    'pair_counts': {'sequential': 1105, 'spatial': 1262, 'union': 1766},
    'stage_order': ['features', 'matches', 'F-U', 'S-U'],
    'model_selection': 'registered descending, points3D descending, native integer ID ascending; retain all',
    'F-U': fixed.PARAMETERS, 'S-U': 'fresh mapper; f/k free, PP fixed, no pose prior; per-model XYZ Sim3'}


def options(names):
    value = ba.legacy.sparse.options()
    value['reader'].camera_params = ','.join(format(v, '.17g') for v in K)
    value['matching'].guided_matching = False
    value['mapper'].image_names = names
    value['mapper'].mapper.num_threads = 1
    o = value['mapper']
    o.use_prior_position = o.ba_use_gpu = o.fix_existing_frames = o.mapper.fix_existing_frames = False
    o.ba_refine_focal_length = o.ba_refine_extra_params = True
    o.ba_refine_principal_point = o.ba_refine_sensor_from_rig = False
    o.mapper.abs_pose_refine_focal_length = o.mapper.abs_pose_refine_extra_params = True
    if value['verification'].min_num_inliers != 15 or not o.check():
        raise ValueError('Legacy min15/options changed')
    return value


def settings(names):
    free = fixed.settings(options(names)['mapper'])
    del free['triangulate_kwargs'], free['pose_freeze']
    free['entry'] = 'incremental_mapping without input_path; no existing reconstruction'
    return {'shared_and_S': fixed.plain({k: v.todict() for k, v in options(names).items()}),
        'F-U': fixed.settings(fixed.options(names)), 'S-U': free,
        'api': {k: getattr(pc, k).__doc__ for k in ('extract_features', 'match_sequential', 'match_spatial',
                                                  'triangulate_points', 'incremental_mapping')},
        'pycolmap': pc.__version__, 'numpy': np.__version__, 'python': sys.version}


def camera_identity(db, names):
    images = sorted(db.read_all_images(), key=lambda i: i.name)
    cameras = {c.camera_id: c for c in db.read_all_cameras()}
    rigs, frames = {r.rig_id: r for r in db.read_all_rigs()}, {f.frame_id: f for f in db.read_all_frames()}
    ids = {i.image_id for i in images}
    if len(images) != 224 or [i.name for i in images] != names or not ids == set(cameras) == set(rigs) == set(frames):
        raise ValueError('Expected 224 independent image/camera/rig/frame IDs')
    result = []
    for image in images:
        i = image.image_id
        c, r, f = cameras[i], rigs[i], frames[i]
        if (image.camera_id != i or c.model != pc.CameraModelId.SIMPLE_RADIAL or [c.width, c.height] != DIMENSIONS
                or not np.array_equal(c.params, K) or not c.has_prior_focal_length or r.num_sensors() != 1
                or (r.ref_sensor_id.type, r.ref_sensor_id.id) != (c.sensor_id.type, i)
                or f.rig_id != i or f.num_data_ids() != 1
                or {(d.sensor_id.type, d.sensor_id.id, d.id) for d in f.data_ids} != {(c.sensor_id.type, i, i)}):
            raise ValueError('Actual U camera/calibration/frame differs from fixed initialization')
        result.append({'id': i, 'name': image.name, 'K': c.params.tolist(), 'dimensions': [c.width, c.height]})
    return result


def name_pairs(db, opts):
    names = {i.image_id: i.name for i in db.read_all_images()}
    return {k: sorted([sorted((names[i], names[j])) for i, j in pairs])
            for k, pairs in ba.legacy.sparse.native_pairs(db, opts).items()}


def toy_database(path, names, centers):
    db = pc.Database(str(path))
    for i, name in enumerate(names, 1):
        c = pc.Camera(camera_id=i, model='SIMPLE_RADIAL', width=7953, height=5279,
                      params=K, has_prior_focal_length=True)
        db.write_camera(c, use_camera_id=True)
        r = pc.Rig(rig_id=i)
        r.add_ref_sensor(c.sensor_id)
        db.write_rig(r, use_rig_id=True)
        image = pc.Image(image_id=i, name=name, camera_id=i)
        db.write_image(image, use_image_id=True)
        frame = pc.Frame(frame_id=i, rig_id=i)
        frame.add_data_id(image.data_id)
        db.write_frame(frame, use_frame_id=True)
        db.write_pose_prior(i, pc.PosePrior(np.asarray(centers[name]), pc.PosePriorCoordinateSystem.CARTESIAN))
        db.write_keypoints(i, np.empty((0, 2), np.float32))
    return db


def fixed_model(db, rows, names):
    cameras = camera_identity(db, names)
    origin, model = np.asarray(rows[names[0]][:3]), pc.Reconstruction()
    for c in cameras:
        tri.add_camera(model, c['id'], c['name'], rows[c['name']], origin, DIMENSIONS)
    return model, {'origin_image': names[0], 'C0': origin.tolist(),
                   'local_to_world': np.column_stack((np.eye(3), origin)).tolist()}


def cache_report(db, cache, pairs, path):
    images = {i.name: i for i in db.read_all_images()}
    included, graph = set(cache.images), cache.correspondence_graph
    expected = {tuple(sorted((images[a].image_id, images[b].image_id))) for a, b in pairs['union']}
    connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        for table in ('matches', 'two_view_geometries'):
            actual = {divmod(row[0], 2147483647) for row in connection.execute(f'SELECT pair_id FROM {table}')}
            if actual != expected:
                raise ValueError('Actual database pair table differs from declared candidate union')
    finally:
        connection.close()
    records = []
    for first, second in pairs['union']:
        i, j = images[first].image_id, images[second].image_id
        if not db.exists_inlier_matches(i, j) or not db.exists_matches(i, j):
            raise ValueError('Missing declared candidate pair row')
        g = db.read_two_view_geometry(i, j)
        records.append({'names': [first, second], 'ids': [i, j], 'raw_retained_by_min15': len(db.read_matches(i, j)),
            'native_inliers': len(g.inlier_matches), 'config': int(g.config),
            'cache_correspondences': graph.num_correspondences_between_images(i, j) if i in included and j in included else 0})
    return {'candidate_pairs': len(records), 'cache_images': cache.num_images(), 'cache_pairs': graph.num_image_pairs(),
        'pairs': records, 'images': [{'name': name, 'id': image.image_id, 'in_cache': image.image_id in included,
            'keypoints': len(db.read_keypoints(image.image_id)),
            'cache_observations': graph.num_observations_for_image(image.image_id) if image.image_id in included else 0,
            'cache_correspondences': graph.num_correspondences_for_image(image.image_id) if image.image_id in included else 0}
            for name, image in sorted(images.items())]}


def model_report(model, names):
    model.update_point_3d_errors()
    registered = set(model.reg_image_ids())
    by_name = {model.images[i].name: model.images[i] for i in registered}
    support, cameras = [], []
    for name in names:
        image = by_name.get(name)
        if image is None:
            support.append({'name': name, 'registered': False, 'point_observations': 0, 'depth': tri.depth_counts(np.empty(0))})
            continue
        pose, c = image.cam_from_world(), model.cameras[image.camera_id]
        ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
        xyz = np.asarray([model.points3D[i].xyz for i in ids]).reshape(-1, 3)
        support.append({'name': name, 'id': image.image_id, 'registered': True, 'keypoints': len(image.points2D),
                        'point_observations': len(ids), 'depth': tri.depth_counts(xyz @ pose.rotation.matrix()[2]+pose.translation[2])})
        cameras.append({'name': name, 'id': image.image_id, 'camera_id': c.camera_id, 'K': c.params.tolist(),
                        'dimensions': [c.width, c.height], 'R': pose.rotation.matrix().tolist(), 'C': image.projection_center().tolist()})
    depth = {k: sum(x['depth'][k] for x in support) for k in tri.depth_counts(np.empty(0))}
    if (depth['total'] != model.compute_num_observations() or depth['nonfinite']
            or any(not np.isfinite(p.xyz).all() or not np.isfinite(p.error) for p in model.points3D.values())):
        raise ValueError('Invalid model observation/finite geometry identity')
    return {'registered': len(registered), 'points3D': model.num_points3D(), 'observations': depth['total'],
        'mean_reprojection_error_px': model.compute_mean_reprojection_error() if model.num_points3D() else None,
        'depth': depth, 'images': support, 'cameras': cameras}


def self_test():
    tri.self_test()
    names = [f'{i:03d}.jpg' for i in range(224)]
    centers = {n: [float(i), float(i % 7), float(i % 3)] for i, n in enumerate(names)}
    with tempfile.TemporaryDirectory() as folder:
        db = toy_database(Path(folder)/'toy.db', names, centers)
        try:
            assert len(camera_identity(db, names)) == 224
            generated = name_pairs(db, options(names))
            assert len(generated['sequential']) == 1105 and len(generated['union']) < 24976
        finally:
            db.close()
    empty = model_report(pc.Reconstruction(), names)
    assert empty['registered'] == empty['points3D'] == empty['observations'] == 0 and len(empty['images']) == 224
    a = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    _, residual = ba.legacy.sparse.fit_centers(a, 2*a+[1,2,3])
    assert max(residual) < 1e-12
    assert options(names)['mapper'].use_prior_position is False
    assert fixed.options(names).fix_existing_frames is True
    return 'PASS: inherited fixed-camera checks, U224 native IDs/pair generation, zero-model support, toy XYZ Sim3'


def preflight():
    pair = ba.BASE/'publisher-raster-pair-run/inputs.json'
    pins = {str(HELPER): '4872c4bf0e27284d0c1cc54a4f785d22f1ba4353eede160d13b57e725bf554a4',
        str(tri.HELPER): '823e67cd9e963dd7e438eea537c7bdd1469a9cf27444b59d4ce19534ba0c88b2',
        str(fixed.PROJECTION): '83714c483084e38277c7dec8d71bacf2288d85591fc0344e0c3404eec1edfb9f',
        str(pair): 'e42c6aeb9089882ac4a3f2b011409a3d8c05136c0384403dbcf287d15fe15aae',
        str(U/'input_manifest.json'): '0327334e7607a77ab81fee4ffd6dcca4a90d76a6d45221956f78307b3ebd4ecc',
        str(U/'Image_orientations_dataset1.xyz'): '523ac45c41e097b7396a66239c1be932f4c0e05c6e354667cfcfa5d57a783b8b'}
    ba.verify(pins)
    original = ba.read_json(pair)
    common = {**original['worker_common_bindings'], **{str(p): ba.sha(p) for p in (ENTRY, PLAN, DESIGN, HELPER, tri.HELPER)}}
    manifest, receipt = ba.read_json(U/'input_manifest.json'), ba.read_json(U/'import-receipt.json')
    names = sorted(e['image_id'] for e in manifest['images'])
    images = {str(U/'images'/e['image_id']): e['sha256'] for e in manifest['images']}
    if (len(names) != 224 or len(set(names)) != 224 or manifest['image_dimensions'] != DIMENSIONS
            or receipt['status'] != 'complete' or manifest['benchmark_eligible'] or manifest['formal_candidate']):
        raise ValueError('Frozen U224 development input identity changed')
    if any(receipt['outputs'].get(p) != h or not Path(p).is_file() or Path(p).is_symlink() for p, h in images.items()):
        raise ValueError('U224 image receipt/path identity mismatch')
    rows = {}
    for line in (U/'Image_orientations_dataset1.xyz').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 15 or fields[0] in rows:
            raise ValueError('Malformed publisher rows')
        rows[fields[0]] = list(map(float, fields[1:]))
    if (set(rows) != set(names) or not np.isfinite(list(rows.values())).all()
            or any(r[6:9] != [4636.912,3970.288,-2601.56] or any(r[9:]) for r in rows.values())):
        raise ValueError('Publisher calibration/name mismatch')
    bound = {**pins, **common, str(U/'import-receipt.json'): ba.sha(U/'import-receipt.json')}
    ba.verify(bound)
    if OUTPUT.exists() or shutil.disk_usage(ba.BASE).free < LIMITS['min_free_bytes']:
        raise RuntimeError('Fresh output and 8 GiB free required')
    return {'schema': 'publisher-raster-full-sparse/1', 'formal_candidate': False, 'parameters': PARAMETERS,
        'limits': LIMITS, 'runtime': original['runtime'], 'bindings': bound, 'worker_common_bindings': common,
        'names': names, 'rows': rows, 'centers': {n: rows[n][:3] for n in names}, 'images': images,
        'settings': settings(names)}


def fixed_branch(case, db, m, o):
    model, transform = fixed_model(db, m['rows'], m['names'])
    before = fixed.snapshot(model)
    expected = fixed.plain(before)
    for camera in expected['cameras']:
        r, center, k = fixed.projection.camera(m['rows'][camera['name']], .5)
        camera.update(R=r.tolist(), C=(center-transform['C0']).tolist(), K=[k[0,0],k[0,2],k[1,2],0.])
    init = fixed.compare(expected, before)
    cache = pc.DatabaseCache.create(db, o.min_num_matches, o.ignore_watermarks, set(m['names']))
    report = cache_report(db, cache, m['pairs'], case/'work/database.db')
    ba.write(case/'cache-support.json', report)
    model.load(cache)
    if not init['passed'] or not fixed.compare(before, fixed.snapshot(model))['passed'] or model.num_points3D():
        raise ValueError('Publisher zero-model initialization/load changed cameras')
    ba.write(case/'initialization.json', {'transform': transform, 'comparison': init, 'snapshot': before})
    initial = case/'initialized-model'
    initial.mkdir()
    model.write_binary(str(initial))
    if not fixed.compare(before, fixed.snapshot(pc.Reconstruction(str(initial))))['passed']:
        raise ValueError('Zero model roundtrip drift')
    db.close()
    target = case/'model'
    target.mkdir()
    called = report['cache_images'] > 0
    if called:
        pc.set_random_seed(917)
        model = pc.triangulate_points(model, str(case/'work/database.db'), str(U/'images'), str(target),
                                      clear_points=True, options=o, refine_intrinsics=False)
    summary = model_report(model, m['names'])
    comparison = fixed.compare(before, fixed.snapshot(model))
    if not comparison['passed']:
        raise ValueError('Fixed publisher camera drift')
    model.write_binary(str(target))
    if not fixed.compare(before, fixed.snapshot(pc.Reconstruction(str(target))))['passed']:
        raise ValueError('Final fixed-camera roundtrip drift')
    ba.write(case/'model-report.json', summary)
    record = {'path': str(target), 'report_path': str(case/'model-report.json'),
        'registered': summary['registered'], 'points3D': summary['points3D'],
        'output_hashes': {str(p): ba.sha(p) for p in sorted(target.iterdir()) if p.is_file()}}
    return {'outcome': 'observed_points' if summary['points3D'] else 'zero_points_no_retry',
        'triangulation_api_called': called, 'model_path': str(target), 'transform': transform,
        'all_models': {'fixed': record}, 'model_ranking': ['fixed'],
        'selected_model': {'id': 'fixed', 'path': str(target), 'transform': transform['local_to_world']} if summary['points3D'] else None,
        'camera_comparison': comparison, 'registered': summary['registered'], 'points3D': summary['points3D']}


def sfm_branch(case, db, m, o):
    cache = pc.DatabaseCache.create(db, o.min_num_matches, o.ignore_watermarks, set(m['names']))
    cache_data = cache_report(db, cache, m['pairs'], case/'work/database.db')
    ba.write(case/'cache-support.json', cache_data)
    db.close()
    target = case/'models'
    target.mkdir()
    pc.set_random_seed(917)
    models = pc.incremental_mapping(str(case/'work/database.db'), str(U/'images'), str(target), options=o) if cache_data['cache_images'] else {}
    results = {}
    for ident, model in sorted(models.items()):
        summary = model_report(model, m['names'])
        if any(c['K'][1:3] != K[1:3] or c['dimensions'] != DIMENSIONS for c in summary['cameras']):
            raise ValueError('S-U fixed principal point or raster dimensions drifted')
        folder = target/str(ident)
        folder.mkdir(exist_ok=True)
        model.write_binary(str(folder))
        camera_ids = sorted(model.reg_image_ids(), key=lambda i: model.images[i].name)
        try:
            transform, residual = ba.legacy.sparse.fit_centers(
                [model.images[i].projection_center() for i in camera_ids],
                [m['centers'][model.images[i].name] for i in camera_ids])
            alignment = {'status': 'available', 'matrix': transform.matrix().tolist(), 'scale': transform.scale,
                'residuals': [{'name': model.images[i].name, 'meters': float(r)} for i, r in zip(camera_ids, residual)]}
        except ValueError as exc:
            alignment = {'status': 'unavailable', 'reason': str(exc)}
        ba.write(folder/'report.json', {**summary, 'allowed_XYZ_alignment': alignment})
        results[str(ident)] = {'path': str(folder), 'report_path': str(folder/'report.json'),
            'report_sha256': ba.sha(folder/'report.json'), 'registered': summary['registered'],
            'points3D': summary['points3D'], 'alignment_status': alignment['status'], 'transform': alignment.get('matrix'),
            'output_hashes': {str(p): ba.sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    if not models:
        ba.write(case/'no-model-report.json', model_report(pc.Reconstruction(), m['names']))
    ranking = sorted(results, key=lambda i: (-results[i]['registered'], -results[i]['points3D'], int(i)))
    selected = {'id': ranking[0], 'path': results[ranking[0]]['path'],
                'transform': results[ranking[0]]['transform']} if ranking else None
    return {'outcome': 'models_returned' if models else 'no_models_no_retry', 'all_models': results,
            'model_ranking': ranking, 'selected_model': selected, 'mapper_api_called': bool(cache_data['cache_images']),
            'mapper_output_directory': str(target)}


def worker(case, m):
    if m['parameters'] != PARAMETERS or m['limits'] != LIMITS or settings(m['names']) != m['settings']:
        raise ValueError('Protocol/settings drift')
    ba.verify(m['worker_bindings'])
    stage, names, opts = m['stage'], m['names'], options(m['names'])
    ba.write(case/'actual-options.json', settings(names))
    work = case/'work/database.db'
    result = {'status': 'completed', 'stage': stage, 'formal_candidate': False}
    if stage in ('S-U', 'S-access'):
        if 'rows' in m:
            raise ValueError('S-U must not receive OPK rows')
        for path in m['denied_inputs']:
            try:
                with Path(path).open('rb'):
                    raise ValueError('S worker can read prohibited input: '+path)
            except PermissionError:
                pass
        ba.write(case/'S-denied-inputs.json', {'status':'PASS', 'files':m['denied_inputs']})
        if stage == 'S-access':
            ba.write(case/'worker-result.json', {**result, 'status':'checked_S_permission_boundary'})
            return
    if stage == 'check':
        proof = self_test()
        db = toy_database(work, names, m['centers'])
        try:
            cameras = camera_identity(db, names)
            pairs = name_pairs(db, opts)
            model, transform = fixed_model(db, m['rows'], names)
            before = fixed.snapshot(model)
            cache = pc.DatabaseCache.create(db, 15, False, set(names))
            model.load(cache)
            if not fixed.compare(before, fixed.snapshot(model))['passed'] or model.num_points3D():
                raise ValueError('Zero-point cache changed fixed cameras')
            folder = case/'zero-model'
            folder.mkdir()
            model.write_binary(str(folder))
            if not fixed.compare(before, fixed.snapshot(pc.Reconstruction(str(folder))))['passed']:
                raise ValueError('U224 zero-model roundtrip failed')
        finally:
            db.close()
        ba.write(case/'candidate-pairs.json', pairs)
        result.update(status='checked_without_real_sparse', self_test=proof, counts=before['counts'],
            pair_counts={k: len(v) for k, v in pairs.items()}, transform=transform)
    elif stage == 'features':
        pc.set_random_seed(917)
        pc.extract_features(str(work), str(U/'images'), image_names=names, camera_mode=pc.CameraMode.PER_IMAGE,
            camera_model='SIMPLE_RADIAL', reader_options=opts['reader'], sift_options=opts['extraction'], device=pc.Device.cpu)
        db = pc.Database(str(work))
        try:
            cameras = camera_identity(db, names)
            for image in db.read_all_images():
                db.write_pose_prior(image.image_id, pc.PosePrior(np.asarray(m['centers'][image.name]),
                                                               pc.PosePriorCoordinateSystem.CARTESIAN))
            pairs = name_pairs(db, opts)
            if pairs != m['pairs']:
                raise ValueError('Actual feature IDs produce different declared name pairs')
            ba.write(case/'feature-counts.json', {i.name: len(db.read_keypoints(i.image_id)) for i in db.read_all_images()})
            ba.write(case/'cameras.json', cameras)
        finally:
            db.close()
        result['database_sha256'] = ba.sha(work)
    else:
        if ba.sha(work) != m['database_sha256']:
            raise ValueError('Working DB bytes differ from frozen snapshot')
        identity = fixed.database_identity(work)
        ba.write(case/'database-before.json', identity)
        db = pc.Database(str(work))
        camera_identity(db, names)
        if name_pairs(db, opts) != m['pairs']:
            raise ValueError('Copied DB pair denominator drift')
        if stage == 'matches':
            db.close()
            pc.set_random_seed(917)
            pc.match_sequential(str(work), sift_options=opts['matching'], matching_options=opts['sequential'],
                                verification_options=opts['verification'], device=pc.Device.cpu)
            pc.match_spatial(str(work), sift_options=opts['matching'], matching_options=opts['spatial'],
                             verification_options=opts['verification'], device=pc.Device.cpu)
            db = pc.Database(str(work))
            try:
                camera_identity(db, names)
                cache = pc.DatabaseCache.create(db, 15, False, set(names))
                ba.write(case/'cache-support.json', cache_report(db, cache, m['pairs'], case/'work/database.db'))
            finally:
                db.close()
            result['database_sha256'] = ba.sha(work)
        elif stage == 'F-U':
            result.update(fixed_branch(case, db, m, fixed.options(names)))
        elif stage == 'S-U':
            if 'rows' in m:
                raise ValueError('S-U must not receive OPK rows')
            result.update(sfm_branch(case, db, m, opts['mapper']))
        after = fixed.database_identity(work)
        ba.write(case/'database-after.json', after)
        immutable_tables = set(identity)-{'matches','two_view_geometries'} if stage == 'matches' else set(identity)
        unchanged = set(identity) == set(after) and all(identity[k] == after[k] for k in immutable_tables)
        ba.write(case/'database-provenance.json', {'before_sha256': m['database_sha256'],
            'after_sha256': ba.sha(work), 'protected_tables_unchanged': unchanged,
            'allowed_table_changes': ['matches','two_view_geometries'] if stage == 'matches' else [],
            'native_header_changes_recorded': True})
        if not unchanged:
            raise ValueError('Native API changed protected database table contents')
    ba.verify(m['worker_bindings'])
    ba.write(case/'worker-result.json', result)


def main():
    if not __debug__ or Path(sys.prefix).resolve() != ba.RUNTIME.resolve() or pc.__version__ != '3.12.6':
        raise RuntimeError('Use fixed runtime -B, never -O')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('self-test','check','run','_worker'))
    parser.add_argument('manifest', nargs='?')
    args = parser.parse_args()
    if args.action == 'self-test':
        print(self_test()); return
    if args.action == '_worker':
        path = Path(args.manifest).resolve()
        worker(path.parent, ba.read_json(path)); return
    campaign = preflight()
    root = Path(tempfile.mkdtemp(prefix='publisher-raster-full-sparse-check-')).resolve() if args.action == 'check' else OUTPUT
    if args.action == 'run': root.mkdir(exist_ok=False)
    # Selection uses only names/allowed XYZ, no descriptors, poses or 3D; runs before real numerics.
    with (root/'pair-selection.log').open('xb') as log:
        saved = os.dup(2)
        try:
            os.dup2(log.fileno(), 2)
            db = toy_database(root/'pair-selection.db', campaign['names'], campaign['centers'])
            try: campaign['pairs'] = name_pairs(db, options(campaign['names']))
            finally: db.close()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
    if {k:len(v) for k,v in campaign['pairs'].items()} != PARAMETERS['pair_counts']:
        raise ValueError('Predeclared U224 candidate-pair counts changed')
    ba.write(root/'inputs.json', campaign)
    ba.write(root/'candidate-pairs.json', campaign['pairs'])
    started, results, databases = time.monotonic(), {}, {}
    try:
        for stage in (['check','S-access'] if args.action == 'check' else PARAMETERS['stage_order']):
            case = root/stage
            (case/'work').mkdir(parents=True)
            m = {k: campaign[k] for k in ('parameters','limits','runtime','names','centers','settings','pairs')}
            m.update(stage=stage, campaign_inputs_sha256=ba.sha(root/'inputs.json'),
                     worker_bindings={**campaign['worker_common_bindings'], **campaign['images']})
            if stage in ('F-U','check'): m['rows'] = campaign['rows']
            if stage in ('S-U','S-access'):
                f_model = root/('check/zero-model' if stage == 'S-access' else 'F-U/model')
                m['denied_inputs'] = list(map(str, [U/'Image_orientations_dataset1.xyz', root/'inputs.json', f_model/'cameras.bin']))
            if stage in ('matches','F-U','S-U'):
                source = databases['features' if stage == 'matches' else 'matches']
                (case/'input').mkdir()
                frozen = case/'input/database.db'
                shutil.copyfile(source['path'], frozen)
                frozen.chmod(0o444)
                shutil.copyfile(frozen, case/'work/database.db')
                m['database_sha256'] = source['sha256']
                m['worker_bindings'][str(frozen)] = source['sha256']
                m['source_database'] = source
                fixed.DATABASE = Path(source['path'])
            else:
                fixed.DATABASE = ba.SFM/'database.db'
            ba.write(case/'inputs.json', m)
            probe = fixed.isolate(case, m)
            command = ['/usr/bin/env','-i','PATH=/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE=1',
                'TMPDIR='+str(case),'OPENSSL_CONF=/dev/null','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
                'MKL_NUM_THREADS=1','VECLIB_MAXIMUM_THREADS=1','/usr/bin/sandbox-exec','-f',probe['profile'],
                str(ba.RUNTIME/'bin/python'),'-B',str(ENTRY),'_worker',str(case/'inputs.json')]
            ba.write(case/'command.json', {'argv': command, 'inputs_sha256': ba.sha(case/'inputs.json'),
                'profile_sha256': ba.sha(probe['profile']), 'limits': LIMITS[stage]})
            status = ba.supervise(command, case, timeout=LIMITS[stage][0], max_rss=LIMITS[stage][1])
            status['final_log_hashes'] = {str(case/n): ba.sha(case/n) for n in ('runtime.stdout','runtime.stderr')}
            ba.write(case/'supervision.json', status)
            ba.verify(campaign['bindings'])
            if status['returncode'] or status['reason'] is not None:
                raise RuntimeError(f'{stage} failed; retain all outputs: {status}')
            result = ba.read_json(case/'worker-result.json')
            if stage in ('features','matches'):
                snapshot = case/'snapshot.db'
                shutil.copyfile(case/'work/database.db', snapshot)
                snapshot.chmod(0o444)
                databases[stage] = {'path': str(snapshot), 'sha256': result['database_sha256']}
                ba.verify({str(snapshot): result['database_sha256']})
                result['frozen_database'] = databases[stage]
            result.update(supervision=status, isolation=probe, inputs_sha256=ba.sha(case/'inputs.json'),
                output_hashes={str(p): ba.sha(p) for p in sorted(case.rglob('*')) if p.is_file()})
            ba.write(case/'result.json', result)
            results[stage] = {'path': str(case/'result.json'), 'sha256': ba.sha(case/'result.json'), 'status': result['status']}
        ba.verify(campaign['images'])
        ba.write(root/'result.json', {'status': 'checked' if args.action == 'check' else 'completed',
            'formal_candidate': False, 'stages': results, 'elapsed_seconds': time.monotonic()-started})
        print(json.dumps({'root': str(root), 'stages': results, 'pair_counts': {k:len(v) for k,v in campaign['pairs'].items()}}))
    except BaseException as exc:
        ba.write(root/'failure.json', {'status':'failed','error':repr(exc),'completed_stages':results,
                                      'elapsed_seconds':time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
