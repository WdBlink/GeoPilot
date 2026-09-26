"""Fixed publisher cameras on the two frozen 16-image raster matching databases."""
import argparse
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ENTRY = Path(__file__).resolve()
HELPER = ENTRY.with_name('fixed_publisher_camera_diagnostic.py')
spec = importlib.util.spec_from_file_location('raster_fixed_camera', HELPER)
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
ba, np, pc, projection = fixed.ba, fixed.np, fixed.pc, fixed.projection
SOURCE = ba.BASE / 'publisher-raster-pair-run'
OUTPUT = ba.BASE / 'publisher-raster-triangulation-run'
PLAN = ba.BASE / 'publisher-raster-triangulation-plan.md'
LIMITS = {'timeout_seconds': 600, 'max_rss_bytes': 8*1024**3,
          'min_free_bytes': 2*1024**3, 'num_threads': 1}
PARAMETERS = {**fixed.PARAMETERS, 'images': 16, 'arm_order': ['M', 'U'],
    'min_num_matches': 15, 'source_native_geometry_min_num_inliers': 1,
    'zero_support': 'retain completed zero-point result; do not select or retry',
    'image_access': 'read-only exact 16 frozen pair-run JPEGs per arm'}
DIMENSIONS = {'M': [7952, 5304], 'U': [7953, 5279]}


def add_camera(model, ident, name, row, origin, dimensions):
    r, center, k = projection.camera(row, .5)
    camera = pc.Camera(camera_id=ident, model='SIMPLE_RADIAL', width=dimensions[0], height=dimensions[1],
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


def build_model(db, rows, dimensions, expected_cameras):
    images, cameras = {i.image_id: i for i in db.read_all_images()}, {c.camera_id: c for c in db.read_all_cameras()}
    rigs, frames = {r.rig_id: r for r in db.read_all_rigs()}, {f.frame_id: f for f in db.read_all_frames()}
    if (len(images) != 16 or set(i.name for i in images.values()) != set(rows)
            or not set(images) == set(cameras) == set(rigs) == set(frames)):
        raise ValueError('Expected 16 independent camera/image/rig/frame IDs')
    actual = []
    model, origin = pc.Reconstruction(), np.asarray(rows[min(rows)][:3])
    for ident, image in sorted(images.items()):
        camera, rig, frame = cameras[ident], rigs[ident], frames[ident]
        if (image.camera_id != ident or frame.rig_id != ident or rig.num_sensors() != 1
                or (rig.ref_sensor_id.type, rig.ref_sensor_id.id) != (camera.sensor_id.type, ident)
                or frame.num_data_ids() != 1
                or {(d.sensor_id.type, d.sensor_id.id, d.id) for d in frame.data_ids}
                   != {(camera.sensor_id.type, ident, ident)}):
            raise ValueError('Independent native frame/sensor identity changed')
        actual.append({'image_id': ident, 'name': image.name, 'camera_id': image.camera_id,
            'model': str(camera.model), 'width': camera.width, 'height': camera.height,
            'params': camera.params.tolist(), 'has_prior_focal_length': camera.has_prior_focal_length})
        if [camera.width, camera.height] != dimensions:
            raise ValueError('Native raster dimensions changed')
        add_camera(model, ident, image.name, rows[image.name], origin, dimensions)
    if sorted(actual, key=lambda x: x['name']) != expected_cameras:
        raise ValueError('Database differs from frozen pair-run camera records')
    return model, {'origin_image': min(rows), 'C0': origin.tolist(),
                  'local_to_world': np.column_stack((np.eye(3), origin)).tolist()}


def cache_audit(db, cache, names):
    """Report every pair and image before/after native cache filtering, without residuals."""
    images = sorted(db.read_all_images(), key=lambda i: i.name)
    graph, included = cache.correspondence_graph, set(cache.images)
    if not {i.name for i in cache.images.values()}.issubset(names):
        raise ValueError('Unexpected image entered cache')
    pairs = []
    for first, second in combinations(images, 2):
        i, j = first.image_id, second.image_id
        raw_present, native_present = db.exists_matches(i, j), db.exists_inlier_matches(i, j)
        if not raw_present or not native_present:
            raise ValueError('Frozen exhaustive database lacks a pair row')
        geometry = db.read_two_view_geometry(i, j)
        pairs.append({'names': [first.name, second.name], 'ids': [i, j],
            'raw_matches': len(db.read_matches(i, j)), 'native_min1_inliers': len(geometry.inlier_matches),
            'native_config': int(geometry.config),
            'cache_correspondences': graph.num_correspondences_between_images(i, j) if i in included and j in included else 0})
    if len(pairs) != 120:
        raise ValueError('Pair denominator changed')
    return {'all_pairs': 120, 'cache_images': cache.num_images(), 'cache_image_pairs': graph.num_image_pairs(),
        'pairs': pairs, 'images': [{'id': i.image_id, 'name': i.name,
            'database_keypoints': len(db.read_keypoints(i.image_id)), 'in_cache': i.image_id in included,
            'cache_observations': graph.num_observations_for_image(i.image_id) if i.image_id in included else 0,
            'cache_correspondences': graph.num_correspondences_for_image(i.image_id) if i.image_id in included else 0}
            for i in images]}


def depth_counts(z):
    finite = np.isfinite(z)
    return {'total': len(z), 'positive': int((finite & (z > 0)).sum()),
            'negative': int((finite & (z < 0)).sum()), 'zero': int((finite & (z == 0)).sum()),
            'nonfinite': int((~finite).sum())}


def observation_support(model):
    output = []
    for support in fixed.image_support(model):
        image = model.images[support['id']]
        ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
        pose = image.cam_from_world()
        xyz = np.asarray([model.points3D[i].xyz for i in ids]).reshape(-1, 3)
        z = xyz @ pose.rotation.matrix()[2] + pose.translation[2]
        output.append({**support, 'depth': depth_counts(z)})
    total = {k: sum(r['depth'][k] for r in output) for k in depth_counts(np.empty(0))}
    if total['total'] != model.compute_num_observations():
        raise ValueError('Track/image observation denominator mismatch')
    return output, total


def self_test():
    fixed.self_test()
    assert depth_counts(np.array([-1., 0., 1., np.nan, np.inf, -np.inf])) == {
        'total': 6, 'positive': 1, 'negative': 1, 'zero': 1, 'nonfinite': 3}
    for dimensions in DIMENSIONS.values():
        model = pc.Reconstruction()
        origin = np.array([500000., 4400000., 200.])
        row = [*origin, 17., -9., 23., 4636.912, 3970.288, -2601.56, 0., 0., 0., 0., 0.]
        add_camera(model, 1, 'toy.jpg', row, origin, dimensions)
        before = fixed.snapshot(model)
        with tempfile.TemporaryDirectory() as path:
            model.write_binary(path)
            assert fixed.compare(before, fixed.snapshot(pc.Reconstruction(path)))['passed']
            db = pc.Database(str(Path(path)/'empty.db'))
            try:
                cache = pc.DatabaseCache.create(db, 15, False, {'toy.jpg'})
                model.load(cache)
            finally:
                db.close()
        assert fixed.compare(before, fixed.snapshot(model))['passed']
        support, depth = observation_support(model)
        assert depth['total'] == 0 and len(support) == 1
        assert [before['cameras'][0]['width'], before['cameras'][0]['height']] == dimensions
    return 'PASS: fixed/R2 checks, M/U native dimensions and empty-cache roundtrip, depth-sign/empty boundaries'


def preflight():
    pins = {str(HELPER): '823e67cd9e963dd7e438eea537c7bdd1469a9cf27444b59d4ce19534ba0c88b2',
        str(fixed.PROJECTION): '83714c483084e38277c7dec8d71bacf2288d85591fc0344e0c3404eec1edfb9f',
        str(SOURCE/'result.json'): 'e58c7496b9868f3af2825b09ae6e934f6aaa6b403e00d9135bc3fdeec21e2f03',
        str(SOURCE/'inputs.json'): 'e42c6aeb9089882ac4a3f2b011409a3d8c05136c0384403dbcf287d15fe15aae'}
    ba.verify(pins)
    campaign, completion = ba.read_json(SOURCE/'inputs.json'), ba.read_json(SOURCE/'result.json')
    if completion['status'] != 'completed' or campaign['parameters']['native_min_num_inliers'] != 1:
        raise ValueError('Completed min1 matching campaign required')
    common = {**campaign['worker_common_bindings'], str(HELPER): pins[str(HELPER)],
              str(ENTRY): ba.sha(ENTRY), str(PLAN): ba.sha(PLAN)}
    bound, arms = {**campaign['bindings'], **common, **pins}, {}
    names = campaign['names']
    if len(names) != 16 or names != sorted(names):
        raise ValueError('Fixed name selection changed')
    for arm in ('M', 'U'):
        case = SOURCE/arm
        result, inputs = ba.read_json(case/'result.json'), ba.read_json(case/'inputs.json')
        if (completion['arms'][arm]['sha256'] != ba.sha(case/'result.json')
                or result['status'] != 'completed' or result['arm'] != arm
                or result['input_manifest_sha256'] != ba.sha(case/'inputs.json')
                or inputs['campaign_inputs_sha256'] != pins[str(SOURCE/'inputs.json')]
                or inputs['names'] != names or inputs['subset'] != campaign['subsets'][arm]
                or inputs['subset']['dimensions'] != DIMENSIONS[arm]):
            raise ValueError('Source completion/input chain changed')
        for path in (case/'result.json', case/'inputs.json'):
            bound[str(path)] = ba.sha(path)
        selected = {str(case/'database.db'): result['database_sha256'], **inputs['copied_input_hashes']}
        for name in ('cameras-before-matching.json', 'cameras-after-matching.json', 'actual-options.json'):
            selected[str(case/name)] = result['output_hashes'][str(case/name)]
        if any(result['output_hashes'].get(p) != h for p, h in selected.items()):
            raise ValueError('Selected DB/images/options disagree with completed output bindings')
        ba.verify(selected)
        before, after = (ba.read_json(case/n) for n in ('cameras-before-matching.json', 'cameras-after-matching.json'))
        if before != after or len(before) != 16:
            raise ValueError('Source cameras changed during matching')
        bound.update(selected)
        arms[arm] = {'rows': inputs['subset']['rows'], 'names': names, 'dimensions': DIMENSIONS[arm],
            'expected_cameras': after, 'images': inputs['copied_input_hashes'], 'image_path': str(case/'images'),
            'database': str(case/'database.db'), 'database_sha256': result['database_sha256']}
    if arms['M']['rows'] != arms['U']['rows']:
        raise ValueError('The two arms must share publisher rows and C0')
    ba.verify(bound)
    o = fixed.options(names)
    if o.min_num_matches != 15:
        raise ValueError('Original cache threshold changed')
    if OUTPUT.exists() or shutil.disk_usage(ba.BASE).free < LIMITS['min_free_bytes']:
        raise RuntimeError('Fresh output and 2 GiB free required')
    return {'schema': 'publisher-raster-triangulation/1', 'formal_candidate': False,
        'parameters': PARAMETERS, 'limits': LIMITS, 'bindings': bound, 'worker_common_bindings': common,
        'runtime': campaign['runtime'], 'settings': fixed.settings(o), 'arms': arms}


def worker(case, manifest, check_only):
    if manifest['parameters'] != PARAMETERS or manifest['limits'] != LIMITS:
        raise ValueError('Worker protocol drift')
    ba.verify(manifest['worker_bindings'])
    proof, source = self_test(), manifest['source']
    o = fixed.options(source['names'])
    if fixed.settings(o) != manifest['settings'] or o.min_num_matches != 15:
        raise ValueError('Native options drift')
    ba.write(case/'settings.json', fixed.settings(o))
    work = case/'work/database.db'
    if ba.sha(work) != source['database_sha256']:
        raise ValueError('Working database differs from snapshot')
    identity_before = fixed.database_identity(work)
    ba.write(case/'database-identity-before.json', identity_before)
    db = pc.Database(str(work))
    try:
        model, transform = build_model(db, source['rows'], source['dimensions'], source['expected_cameras'])
        before = fixed.snapshot(model)
        expected = fixed.plain(before)
        for camera in expected['cameras']:
            r, center, k = projection.camera(source['rows'][camera['name']], .5)
            camera.update(K=[k[0, 0], k[0, 2], k[1, 2], 0.], R=r.tolist(), C=(center-transform['C0']).tolist())
        initialization = fixed.compare(expected, before)
        if not initialization['passed'] or before['counts']['points3D'] or before['counts']['observations']:
            raise ValueError('Nonempty or incorrect publisher initialization')
        ba.write(case/'before-load.json', before)
        ba.write(case/'transform.json', transform)
        cache = pc.DatabaseCache.create(db, o.min_num_matches, o.ignore_watermarks, set(source['names']))
        cache_record = cache_audit(db, cache, set(source['names']))
        ba.write(case/'cache-support.json', cache_record)
        model.load(cache)
    finally:
        db.close()
    loaded = fixed.snapshot(model)
    load_check = fixed.compare(before, loaded)
    ba.write(case/'after-load.json', loaded)
    if not load_check['passed'] or model.num_points3D() or model.num_reg_images() != 16:
        raise ValueError('Cache load changed cameras or introduced points')
    initial = case/'initialized-model'
    initial.mkdir()
    model.write_binary(str(initial))
    roundtrip = fixed.compare(before, fixed.snapshot(pc.Reconstruction(str(initial))))
    if not roundtrip['passed']:
        raise ValueError('Initialization roundtrip changed cameras')
    result = {'status': 'checked_without_triangulation', 'arm': manifest['arm'], 'formal_candidate': False,
        'self_test': proof, 'initialization_comparison': initialization, 'load_comparison': load_check,
        'roundtrip_comparison': roundtrip, 'counts': loaded['counts'], 'transform': transform}
    if not check_only:
        target = case/'model'
        target.mkdir()
        started = time.monotonic()
        called = cache_record['cache_images'] > 0
        if called:
            pc.set_random_seed(917)
            model = pc.triangulate_points(model, str(work), source['image_path'], str(target),
                clear_points=True, options=o, refine_intrinsics=False)
        model.update_point_3d_errors()
        after = fixed.snapshot(model)
        comparison = fixed.compare(before, after)
        ba.write(case/'after-triangulation.json', after)
        ba.write(case/'camera-comparison.json', comparison)
        support, depth = observation_support(model)
        ba.write(case/'image-support.json', support)
        if (not comparison['passed'] or depth['nonfinite']
                or any(not np.isfinite(p.xyz).all() or not np.isfinite(p.error) for p in model.points3D.values())):
            raise ValueError('Camera drift or nonfinite sparse geometry')
        model.write_binary(str(target))
        if not fixed.compare(before, fixed.snapshot(pc.Reconstruction(str(target))))['passed']:
            raise ValueError('Final roundtrip changed fixed cameras')
        result.update(status='completed', counts=after['counts'], camera_comparison=comparison,
            triangulation_seconds=time.monotonic()-started, triangulation_api_called=called,
            outcome='observed_points' if model.num_points3D() else 'zero_points_no_retry', depth=depth,
            refreshed_mean_reprojection_error_px=model.compute_mean_reprojection_error() if model.num_points3D() else None,
            model_path=str(target), images_without_points=sum(r['point_observations'] == 0 for r in support))
    else:
        ba.write(case/'loaded-image-support.json', fixed.image_support(model))
    identity_after = fixed.database_identity(work)
    ba.write(case/'database-identity-after.json', identity_after)
    provenance = {'snapshot_sha256': source['database_sha256'], 'working_before_sha256': source['database_sha256'],
        'working_after_sha256': ba.sha(work), 'table_identity_unchanged': identity_before == identity_after,
        'native_database_may_update_sqlite_header': True}
    ba.write(case/'database-provenance.json', provenance)
    if identity_before != identity_after:
        raise ValueError('Native API changed frozen descriptor/geometry table contents')
    result['database_provenance'] = provenance
    ba.verify(manifest['worker_bindings'])
    ba.write(case/'worker-result.json', result)


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
    root = Path(tempfile.mkdtemp(prefix='publisher-raster-triangulation-check-')).resolve() if args.action == 'check' else OUTPUT
    if args.action == 'run':
        root.mkdir(exist_ok=False)
    ba.write(root/'inputs.json', campaign)
    started, results = time.monotonic(), {}
    try:
        for arm in ('M', 'U'):
            case, source = root/arm, campaign['arms'][arm]
            (case/'input').mkdir(parents=True)
            (case/'work').mkdir()
            frozen, working = case/'input/database.db', case/'work/database.db'
            shutil.copyfile(source['database'], frozen)
            frozen.chmod(0o444)
            shutil.copyfile(frozen, working)
            copied = {str(frozen): source['database_sha256']}
            ba.verify(copied)
            manifest = {k: campaign[k] for k in ('parameters', 'limits', 'runtime', 'settings')}
            manifest.update(arm=arm, source=source, copied_input_hashes=copied,
                campaign_inputs_sha256=ba.sha(root/'inputs.json'),
                worker_bindings={**campaign['worker_common_bindings'], **source['images'], **copied})
            ba.write(case/'inputs.json', manifest)
            fixed.DATABASE = Path(source['database'])  # Reused probe must deny the actual frozen source DB.
            probe = fixed.isolate(case, manifest)
            command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONDONTWRITEBYTECODE=1',
                'TMPDIR='+str(case), 'OPENSSL_CONF=/dev/null', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
                'MKL_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1', '/usr/bin/sandbox-exec', '-f', probe['profile'],
                str(ba.RUNTIME/'bin/python'), '-B', str(ENTRY),
                '_check_worker' if args.action == 'check' else '_worker', str(case/'inputs.json')]
            ba.write(case/'command.json', {'argv': command, 'limits': LIMITS,
                'inputs_sha256': ba.sha(case/'inputs.json'), 'profile_sha256': ba.sha(probe['profile'])})
            status = ba.supervise(command, case, timeout=600, max_rss=LIMITS['max_rss_bytes'])
            status['final_log_hashes'] = {str(case/n): ba.sha(case/n) for n in ('runtime.stdout', 'runtime.stderr')}
            ba.write(case/'supervision.json', status)
            ba.verify(campaign['bindings'])
            ba.verify(copied)
            if status['returncode'] or status['reason'] is not None:
                raise RuntimeError(f'{arm} worker failed; retain outputs: {status}')
            result = ba.read_json(case/'worker-result.json')
            result.update(supervision=status, isolation=probe, inputs_sha256=ba.sha(case/'inputs.json'),
                output_hashes={str(p): ba.sha(p) for p in sorted(case.rglob('*')) if p.is_file()})
            ba.write(case/'result.json', result)
            results[arm] = {'path': str(case/'result.json'), 'sha256': ba.sha(case/'result.json'), 'status': result['status']}
        ba.write(root/'result.json', {'status': 'checked' if args.action == 'check' else 'completed',
            'formal_candidate': False, 'arms': results, 'elapsed_seconds': time.monotonic()-started})
        print(json.dumps({'root': str(root), 'arms': results}))
    except BaseException as exc:
        ba.write(root/'failure.json', {'status': 'failed', 'error': repr(exc), 'completed_arms': results,
            'elapsed_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
