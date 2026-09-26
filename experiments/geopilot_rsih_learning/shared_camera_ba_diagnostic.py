"""Two fixed sparse-only BA branches: matched K, per-image versus shared camera."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'code/geopilot_rsih'))
from strict_json import read_json
from supervise import supervise
from tools import load_legacy_adapter, sha

legacy = load_legacy_adapter()
np, pc = legacy.np, legacy.pc
SOURCE = ROOT / 'out/geopilot-learning-20260923/p2-run'
SFM = SOURCE / 'nodes/prepare/sfm'
MODEL = SFM / 'models/0'
INPUTS = ROOT / 'out/usegeo_benchmark/prepared/v1/inputs/rgb-oriented/Dataset-1'
BASE = ROOT / 'out/geopilot-research-20260925'
OUTPUT = BASE / 'shared-camera-ba-run'
PLAN = BASE / 'shared-camera-ba-plan.md'
DIAGNOSIS = BASE / 'preparation-diagnosis.md'
RUNTIME = ROOT / 'out/usegeo_mesh_benchmark/paper-readiness-20260918/baseline-preflight/baseline-runtime'
INITIAL_K = [4639.021835367615, 3976.0, 2652.0, 0.0]
LIMITS = {'timeout_seconds': 1800, 'max_rss_bytes': 24 * 1024**3, 'num_threads': 1, 'max_iterations': 100}
MODEL_SHA = {
    'cameras.bin': '0b098fe6c88af0eb64a1b216d4500effbafddfcd081968022166eec24af6f65f',
    'frames.bin': '67413130a3f4fa0dcd7cdd2f4e74f34334c849d0bb5232305001aa400ac98b02',
    'images.bin': '5a2d81aab7203dba2149dcdaab073a448e75af0193620378f86a7d999d67937b',
    'points3D.bin': 'a99dc365da490a9ca147784fcf69eba8d35b2af656afba5fe2c4569bc2f2a53d',
    'rigs.bin': 'de35405fbae0b9906ba36f6b1c867d89b2973c3e3274b2edc4cd9519ba686a1c',
}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
        stream.write('\n')


def verify(bindings):
    for name, expected in bindings.items():
        if sha(name) != expected:
            raise ValueError(f'Bound input changed: {name}')


def options():
    """Same explicit solver settings in both branches; retain/export other defaults."""
    value = pc.BundleAdjustmentOptions()
    value.refine_focal_length = True
    value.refine_principal_point = False
    value.refine_extra_params = True
    value.refine_rig_from_world = True
    value.refine_sensor_from_rig = False
    value.use_gpu = False
    value.print_summary = True
    value.loss_function_type = pc.LossFunctionType.TRIVIAL
    value.solver_options.num_threads = LIMITS['num_threads']
    value.solver_options.max_num_iterations = LIMITS['max_iterations']
    value.solver_options.minimizer_progress_to_stdout = True
    return value


def validate_trivial(model):
    """Only migrate independent single-camera, single-image frames; reject real rigs."""
    ids = sorted(model.reg_image_ids())
    if (not ids or len(ids) != model.num_images() or len(ids) != model.num_cameras()
            or len(ids) != model.num_frames() or len(ids) != model.num_rigs()
            or len({model.images[i].camera_id for i in ids}) != len(ids)):
        raise ValueError('Expected one original camera/rig/frame per registered image')
    for i in ids:
        image = model.images[i]
        frame, camera = model.frames[image.frame_id], model.cameras[image.camera_id]
        rig = model.rigs[frame.rig_id]
        if (rig.num_sensors() != 1
                or (rig.ref_sensor_id.type, rig.ref_sensor_id.id) != (camera.sensor_id.type, camera.sensor_id.id)
                or frame.num_data_ids() != 1
                or {(d.sensor_id.type, d.sensor_id.id, d.id) for d in frame.data_ids}
                    != {(image.data_id.sensor_id.type, image.data_id.sensor_id.id, image.data_id.id)}
                or not np.array_equal(frame.rig_from_world.matrix(), image.cam_from_world().matrix())):
            raise ValueError('Nontrivial rig/frame semantics; migration not authorized')


def fingerprint(model):
    """Canonical geometry/observation identities, excluding intentionally changed K."""
    poses, points, observations = (hashlib.sha256() for _ in range(3))
    for image_id in sorted(model.images):
        image = model.images[image_id]
        identity = struct.pack('<II', image_id, image.frame_id) + image.name.encode() + b'\0'
        poses.update(identity + image.cam_from_world().matrix().astype('<f8').tobytes())
        observations.update(identity)
        observations.update(np.asarray([p.xy for p in image.points2D], dtype='<f8').tobytes())
        observations.update(np.asarray([p.point3D_id for p in image.points2D], dtype='<u8').tobytes())
    for point_id in sorted(model.points3D):
        point = model.points3D[point_id]
        points.update(struct.pack('<Q', point_id) + point.xyz.astype('<f8').tobytes() + point.color.tobytes())
        observations.update(struct.pack('<Q', point_id))
        for image_id, index in sorted((e.image_id, e.point2D_idx) for e in point.track.elements):
            observations.update(struct.pack('<II', image_id, index))
    return {'poses_sha256': poses.hexdigest(), 'points_sha256': points.hexdigest(),
            'observations_sha256': observations.hexdigest(), 'images': model.num_images(),
            'registered_images': model.num_reg_images(), 'points3D': model.num_points3D(),
            'observations': model.compute_num_observations()}


def branch_model(source, branch, initial):
    """Use native objects throughout; one shared sensor with independent frame poses."""
    validate_trivial(source)
    if branch == 'A':
        model = pc.Reconstruction(source)
        for camera in model.cameras.values():
            camera.params = np.array(initial)
        return model
    if branch != 'B':
        raise ValueError('Unknown fixed branch')
    model = pc.Reconstruction()
    camera_id, rig_id = min(source.cameras), min(source.rigs)
    original = source.cameras[camera_id]
    camera = pc.Camera(camera_id=camera_id, model=original.model, width=original.width,
                       height=original.height, params=initial,
                       has_prior_focal_length=original.has_prior_focal_length)
    model.add_camera(camera)
    rig = pc.Rig(rig_id=rig_id)
    rig.add_ref_sensor(camera.sensor_id)
    model.add_rig(rig)
    for image_id in sorted(source.images):
        old = source.images[image_id]
        image = pc.Image(name=old.name, points2D=old.points2D, camera_id=camera_id, image_id=image_id)
        image.frame_id = old.frame_id
        frame = pc.Frame(frame_id=old.frame_id, rig_id=rig_id,
                         rig_from_world=old.frame.rig_from_world)
        frame.add_data_id(image.data_id)
        model.add_frame(frame)
        model.add_image(image)
        model.register_frame(frame.frame_id)
    for point_id, point in source.points3D.items():
        model.points3D[point_id] = pc.Point3D(point.todict())
    return model


def initial_cameras(path, model):
    entries = read_json(path)
    if (len(entries) != 224 or {x['id'] for x in entries} != set(model.cameras)
            or any(x['params'] != INITIAL_K or x['model'] != 'CameraModelId.SIMPLE_RADIAL'
                   or (x['width'], x['height']) != (7952, 5304) for x in entries)
            or any(c.model != pc.CameraModelId.SIMPLE_RADIAL or (c.width, c.height) != (7952, 5304)
                   for c in model.cameras.values())
            or len({c.has_prior_focal_length for c in model.cameras.values()}) != 1):
        raise ValueError('Initial camera identity/format mismatch')
    return np.asarray(entries[0]['params'], dtype=float)


def centers(path, names):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) < 4 or fields[0] in result:
            raise ValueError('Malformed/duplicate allowed center')
        result[fields[0]] = np.asarray(fields[1:4], dtype=float)
    if set(result) != set(names) or not np.isfinite(list(result.values())).all():
        raise ValueError('Missing/unexpected/nonfinite allowed centers')
    return result


def snapshot(model, allowed):
    """Refresh cached reprojection errors, then refit all 224 allowed centers."""
    model.update_point_3d_errors()
    ids = sorted(model.reg_image_ids())
    source = np.asarray([model.images[i].projection_center() for i in ids])
    target = np.asarray([allowed[model.images[i].name] for i in ids])
    transform, residuals = legacy.sparse.fit_centers(source, target)
    return {'mean_point_reprojection_error_px': model.compute_mean_reprojection_error(),
            'counts': {'cameras': model.num_cameras(), 'rigs': model.num_rigs(),
                       'frames': model.num_frames(), 'images': model.num_reg_images(),
                       'points3D': model.num_points3D(), 'observations': model.compute_num_observations()},
            'cameras': [{'id': i, 'model': str(model.cameras[i].model),
                         'params': model.cameras[i].params.tolist()} for i in sorted(model.cameras)],
            'image_camera_map': {i: model.images[i].camera_id for i in ids},
            'fit_centers_transform': transform.matrix().tolist(),
            'center_residuals_m': [{'image_id': i, 'image': model.images[i].name, 'residual_m': float(r)}
                                   for i, r in zip(ids, residuals)],
            'center_residual_summary_m': {'mean': float(residuals.mean()),
                'rms': float(np.sqrt(np.square(residuals).mean())), 'median': float(np.median(residuals)),
                'max': float(residuals.max())}}


def bindings():
    """Bind sparse/allowed inputs and the existing numerical environment only."""
    if Path(sys.prefix).resolve() != RUNTIME.resolve() or pc.__version__ != '3.12.6':
        raise RuntimeError('Use the fixed baseline-runtime Python with pycolmap 3.12.6')
    launch_path = SOURCE / 'launch.json'
    if sha(launch_path) != '2a5023086874bd1d138164634f109540fcf1ce57f9ab6a64a66c3d3955145e88':
        raise ValueError('P2 launch identity changed')
    launch = read_json(launch_path)
    result = {str(MODEL / name): digest for name, digest in MODEL_SHA.items()}
    result[str(SFM / 'cameras.json')] = '30203a5a41d5df20da05a37dc3259921793ade6274eb9e990324d919bd00bb55'
    for name in ('input_manifest.json', 'Image_orientations_dataset1.xyz'):
        result[str(INPUTS / name)] = launch['input_hashes'][str(INPUTS / name)]
    for name in ('code/geopilot_rsih/tools.py', 'code/geopilot_rsih/strict_json.py',
                 'code/geopilot_rsih/supervise.py', 'code/usegeo_mesh_baseline/runner.py',
                 'experiments/geopilot_rsi/run.py'):
        result[str(ROOT / name)] = launch['bindings'][str(ROOT / name)]
    for path, digest in launch['bindings'].items():
        if any(Path(path).is_relative_to(launch['runtime'][key]) for key in ('prefix', 'base_prefix')):
            result[path] = digest
    for path in (Path(__file__).resolve(), PLAN, DIAGNOSIS, launch_path):
        result[str(path)] = sha(path)
    verify(result)
    return result


def check_structure(model, initial):
    """Real model conversion/roundtrip check without invoking any optimizer."""
    expected = fingerprint(model)
    for name in ('A', 'B'):
        branch = branch_model(model, name, initial)
        if fingerprint(branch) != expected:
            raise ValueError('Migration changed source poses/points/observations')
        with tempfile.TemporaryDirectory() as temporary:
            branch.write(temporary)
            reread = pc.Reconstruction(temporary)
            if fingerprint(reread) != expected:
                raise ValueError('Native binary roundtrip changed geometry/observations')
            if reread.num_cameras() != (224 if name == 'A' else 1):
                raise ValueError('Wrong camera count after roundtrip')
    return expected


def api_smoke_check():
    """One two-iteration synthetic BA; never reads P2 or any reference data."""
    model = pc.Reconstruction()
    xyz = np.array([[x, y, z] for x in (-.4, .4) for y in (-.3, .3) for z in (4., 5., 6.)])
    for ident, center in enumerate(([-1., 0., 0.], [0., .2, 0.], [1., -.1, 0.]), 1):
        camera = pc.Camera(camera_id=ident, model='SIMPLE_RADIAL', width=100, height=100,
                           params=[100., 50., 50., 0.])
        model.add_camera(camera)
        rig = pc.Rig(rig_id=ident)
        rig.add_ref_sensor(camera.sensor_id)
        model.add_rig(rig)
        image = pc.Image(name=f'{ident}.jpg', keypoints=camera.img_from_cam(xyz - center),
                         camera_id=ident, image_id=ident)
        image.frame_id = ident
        frame = pc.Frame(frame_id=ident, rig_id=ident,
                         rig_from_world=pc.Rigid3d(np.column_stack((np.eye(3), -np.array(center)))))
        frame.add_data_id(image.data_id)
        model.add_frame(frame)
        model.add_image(image)
        model.register_frame(ident)
    for index, point in enumerate(xyz):
        model.add_point3D(point + [.002, -.001, .004],
                         pc.Track([pc.TrackElement(i, index) for i in (1, 2, 3)]))
    shared = branch_model(model, 'B', [100., 50., 50., 0.])
    if fingerprint(shared) != fingerprint(model):
        raise ValueError('Synthetic migration changed geometry')
    value, config = options(), pc.BundleAdjustmentConfig()
    value.solver_options.max_num_iterations = 2
    for ident in (1, 2, 3):
        config.add_image(ident)
    config.fix_gauge(pc.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    settings = {'options': value.todict(), 'config': config.todict()}
    solver = pc.create_default_bundle_adjuster(value, config, shared)
    summary = solver.solve()
    if not summary.IsSolutionUsable():
        raise RuntimeError('Synthetic BA API did not return a usable solution')
    json.dumps({'settings': settings, 'summary': summary.todict()}, allow_nan=False, default=str)
    print(json.dumps({'synthetic_BA_API': 'passed', 'images': 3, 'points': 12,
                      'termination': str(summary.termination_type), 'usable': True}))


def worker(name):
    case = OUTPUT / name
    manifest = read_json(OUTPUT / 'inputs.json')
    verify(manifest['bindings'])
    copied = {}
    model_input = case / 'input/model'
    model_input.mkdir(parents=True)
    pairs = [(MODEL / file, model_input / file) for file in MODEL_SHA]
    pairs += [(SFM / 'cameras.json', case / 'input/cameras.json'),
              (INPUTS / 'Image_orientations_dataset1.xyz', case / 'input/centers.xyz'),
              (INPUTS / 'input_manifest.json', case / 'input/input_manifest.json')]
    for source, target in pairs:
        shutil.copyfile(source, target)
        copied[str(target)] = manifest['bindings'][str(source)]
        target.chmod(0o444)
    verify(copied)
    original = pc.Reconstruction(str(model_input))
    initial = initial_cameras(case / 'input/cameras.json', original)
    allowed = centers(case / 'input/centers.xyz', [i.name for i in original.images.values()])
    expected = fingerprint(original)
    write(case / 'historical-model.json', snapshot(original, allowed))
    model = branch_model(original, name, initial)
    if fingerprint(model) != expected:
        raise ValueError('Pre-BA geometry/observations changed')
    initialized = case / 'initialized-model'
    initialized.mkdir()
    model.write(str(initialized))
    model = pc.Reconstruction(str(initialized))
    if fingerprint(model) != expected:
        raise ValueError('Initialized model roundtrip changed geometry/observations')
    before = snapshot(model, allowed)
    write(case / 'before-ba.json', before)
    config = pc.BundleAdjustmentConfig()
    for image_id in sorted(model.reg_image_ids()):
        config.add_image(image_id)
    config.fix_gauge(pc.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    value = options()
    write(case / 'solver-settings.json', {'options': value.todict(), 'config': config.todict()})
    pc.set_random_seed(917)
    solver = pc.create_default_bundle_adjuster(value, config, model)
    summary = solver.solve()
    write(case / 'solver-summary.json', summary.todict())
    if not summary.IsSolutionUsable():
        raise RuntimeError('BA solver did not return a usable solution')
    after = snapshot(model, allowed)
    observed = fingerprint(model)
    for key in ('observations_sha256', 'images', 'registered_images', 'points3D', 'observations'):
        if observed[key] != expected[key]:
            raise ValueError('BA altered observation/track identity or counts')
    if any(not np.isfinite(c.params).all() or c.focal_length <= 0 or not c.verify_params()
           for c in model.cameras.values()):
        raise ValueError('Invalid optimized intrinsics')
    destination = case / 'optimized-model'
    destination.mkdir()
    model.write(str(destination))
    verify(copied)
    verify(manifest['bindings'])
    write(case / 'after-ba.json', after)
    write(case / 'result.json', {'status': 'completed', 'branch': name,
        'solver_termination': str(summary.termination_type), 'solution_usable': True,
        'formal_candidate': False, 'mesh_quality_evaluated': False,
        'source_fingerprint': expected, 'after_fingerprint': observed, 'copied_input_hashes': copied,
        'output_hashes': {str(p): sha(p) for p in case.rglob('*') if p.is_file()
                         and p.name not in ('runtime.stdout', 'runtime.stderr', 'worker.log')
                         and 'input' not in p.relative_to(case).parts}})


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python is unsupported')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', '_worker'))
    parser.add_argument('branch', choices=('A', 'B'), nargs='?')
    args = parser.parse_args()
    if args.action == '_worker':
        if args.branch is None:
            raise ValueError('Worker branch required')
        worker(args.branch)
        return
    if args.branch is not None:
        raise ValueError('No branch override accepted')
    bound = bindings()
    source = pc.Reconstruction(str(MODEL))
    initial = initial_cameras(SFM / 'cameras.json', source)
    identity = check_structure(source, initial)
    verify(bound)
    if args.action == 'check':
        print(json.dumps({'status': 'checked_without_BA', 'source': identity, 'initial_K': initial.tolist(),
                          'source_bytes': sum((MODEL / p).stat().st_size for p in MODEL_SHA)}))
        return
    if OUTPUT.resolve() != OUTPUT:
        raise ValueError('Output path must not contain symlinks')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write(OUTPUT / 'inputs.json', {'bindings': bound, 'source_fingerprint': identity,
          'initial_K': initial.tolist(), 'limits': LIMITS, 'options': options().todict(),
          'gauge': str(pc.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)})
    for name in ('A', 'B'):
        case = OUTPUT / name
        case.mkdir()
        command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin:/usr/sbin:/sbin',
                   'TMPDIR=' + str(case), 'PYTHONDONTWRITEBYTECODE=1', str(RUNTIME / 'bin/python'),
                   '-B', str(Path(__file__).resolve()), '_worker', name]
        write(case / 'command.json', {'argv': command, 'limits': LIMITS, 'inputs_sha256': sha(OUTPUT / 'inputs.json')})
        status = supervise(command, case, timeout=LIMITS['timeout_seconds'], max_rss=LIMITS['max_rss_bytes'])
        status['final_log_hashes'] = {str(case / log): sha(case / log)
                                     for log in ('runtime.stdout', 'runtime.stderr', 'worker.log')
                                     if (case / log).is_file()}
        write(case / 'supervision.json', status)
        verify(bound)
        if status['returncode'] != 0 or status['reason'] is not None:
            raise RuntimeError(f'{name} failed: stop; retain partial outputs and supervision logs')
    print('A and B sparse-only BA complete; see branch diagnostics and supervision.')


if __name__ == '__main__':
    main()
