"""Rebuild fixed A/B sparse camera controls through fresh undistortion and MVS."""
import argparse
import copy
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
import shared_camera_ba_diagnostic as ba
import tools

BASE = ba.BASE
CONTROL = BASE / 'shared-camera-ba-run'
OUTPUT = BASE / 'camera-mesh-run'
PLAN = BASE / 'camera-mesh-plan.md'
FEASIBILITY = BASE / 'camera-mesh-feasibility.md'
ISOLATION = ba.ROOT / 'code/geopilot_rsih/isolation.mjs'
NODE = Path(shutil.which('node')).resolve()
CASES = ('A', 'B')
LIMITS = {'timeout_seconds': 1800, 'max_rss_bytes': 32 * 1024**3,
          'min_free_bytes': 60 * 1024**3, 'openmvs_threads': 8}
PARAMETERS = {'undistort_max_image_size': 2400, 'densify': {'resolution-level': 1},
              'mesh': {'decimate': .5}}
PINS = {
    str(Path(ba.__file__).resolve()): '07d85983a366f8b928e104a47884d9ddaea343554efd3b0c0b0210db9c5f71ac',
    str(CONTROL / 'inputs.json'): 'd6a292e04684544845919e880f18625a3726e4290f156645eddb3361dc53682c',
    str(CONTROL / 'A/result.json'): 'b06d7d818efecdf28be1b786644bb601e8d9812d86920f7e068e9f3c9d3736e1',
    str(CONTROL / 'B/result.json'): 'b5750c704304ef47df1d9151a9be7f5a397ef8360441f737eafdb862c42d6c24',
    str(CONTROL / 'A/supervision.json'): '558e6cc33c4487b9ba0f4757b3763a21ef842821712ed4273c0bce5eba18ddcf',
    str(CONTROL / 'B/supervision.json'): '02835d6e49ea0ea9017460dca3e361fb02256429c2a810104bd2ef2e08f787ba',
    str(ba.SOURCE / 'nodes/densify/state.json'): '79b385a7b591fd6867091739ac4e7448e0216a1c03da68a8839b2946a2f6982a',
    str(ba.SOURCE / 'events.jsonl'): '06438d3c6dcfaed3cc0efaa2988e20323011eed742f55ad4609116d3fe45878a',
}


def undistort_options():
    return ba.pc.UndistortCameraOptions(max_image_size=PARAMETERS['undistort_max_image_size'])


def model_identity(model, name, images):
    if (model.num_images() != 224 or model.num_reg_images() != 224
            or {image.name for image in model.images.values()} != set(images)
            or model.num_cameras() != (224 if name == 'A' else 1)
            or any(c.model != ba.pc.CameraModelId.SIMPLE_RADIAL or (c.width, c.height) != (7952, 5304)
                   or not ba.np.isfinite(c.params).all() for c in model.cameras.values())):
        raise ValueError('Wrong fixed branch camera model or registered image coverage')


def inputs():
    """Read and bind only allowed inputs, reconstruction tools and existing BA evidence."""
    if (Path(sys.prefix).resolve() != ba.RUNTIME.resolve() or ba.pc.__version__ != '3.12.6'
            or ba.legacy.THREADS != 8):
        raise RuntimeError('Use the frozen numerical runtime and eight-thread OpenMVS adapter')
    ba.verify(PINS)
    prior = ba.read_json(CONTROL / 'inputs.json')
    launch = ba.read_json(ba.SOURCE / 'launch.json')
    historical = ba.read_json(ba.SOURCE / 'nodes/densify/state.json')
    bound = {**prior['bindings'], **PINS, **launch['input_hashes']}
    sources = {}
    for name in CASES:
        case = CONTROL / name
        result, status = [ba.read_json(case / f) for f in ('result.json', 'supervision.json')]
        if (result['status'] != 'completed' or result['branch'] != name or not result['solution_usable']
                or result['solver_termination'] != 'TerminationType.CONVERGENCE'
                or status['returncode'] != 0 or status['reason'] is not None
                or result['source_fingerprint'] != prior['source_fingerprint']
                or result['after_fingerprint']['observations_sha256'] != prior['source_fingerprint']['observations_sha256']
                or ba.read_json(case / 'solver-settings.json')['options'] != prior['options']):
            raise ValueError('Both original BA controls must be completed with identical settings')
        bound.update(result['copied_input_hashes'])
        bound.update(result['output_hashes'])
        bound.update(status['final_log_hashes'])
        model = case / 'optimized-model'
        files = {str(model / f): result['output_hashes'][str(model / f)] for f in ba.MODEL_SHA}
        if set(model.iterdir()) != {Path(p) for p in files}:
            raise ValueError('Unexpected optimized-model files')
        sources[name] = {'model': str(model), 'model_hashes': files,
                         'BA_result': str(case / 'result.json'), 'fingerprint': result['after_fingerprint']}
    if ba.read_json(CONTROL / 'A/solver-settings.json') != ba.read_json(CONTROL / 'B/solver-settings.json'):
        raise ValueError('Original A/B solver configurations differ')
    evidence = [historical['coverage_prepare_path'], *historical['sfm_evidence_files']]
    bound.update({p: historical['artifact_hashes'][p] for p in evidence})
    for name in ('InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'RefineMesh'):
        path = str(ba.legacy.TOOLS / name)
        bound[path] = launch['bindings'][path]
    for name in ('densify-help.txt', 'mesh-help.txt'):
        path = str(ba.ROOT / 'out/geopilot-learning-20260921' / name)
        bound[path] = launch['bindings'][path]
    bound[str(ISOLATION)] = launch['bindings'][str(ISOLATION)]
    for path in (Path(__file__).resolve(), PLAN, FEASIBILITY, NODE, ba.SOURCE / 'supervision.json'):
        bound[str(path)] = ba.sha(path)
    ba.verify(bound)
    manifest, images, _ = ba.legacy.check_inputs(ba.INPUTS)
    coverage = ba.read_json(historical['coverage_prepare_path'])
    if (len(images) != 224 or set(images) != set(launch['input_image_ids'])
            or coverage['input_manifest_sha256'] != launch['input_manifest_sha256']
            or set(coverage['images']) != set(images)
            or any(not all(item[k] for k in ('sfm_ingested', 'matching_candidate', 'registered'))
                   or item['input_sha256'] != images[name] for name, item in coverage['images'].items())):
        raise ValueError('The full original 224-image sparse coverage cannot be established')
    events = [json.loads(line) for line in (ba.SOURCE / 'events.jsonl').read_text().splitlines()]
    timings = {e['node']: e['wall_seconds'] for e in events
               if e['event'] == 'numerical_call' and e['node'] in PARAMETERS}
    # Parent verifies every old BA record; each worker sees only its own model,
    # exact RGB/XYZ files and the source/runtime files needed to rebuild it.
    common = set(launch['input_hashes']) | set(evidence)
    common |= {str(p) for p in (Path(__file__).resolve(), Path(ba.__file__).resolve(), PLAN, FEASIBILITY)}
    common |= {p for p in bound if any(Path(p).is_relative_to(root)
        for root in (ba.RUNTIME, Path(launch['runtime']['base_prefix']), ba.legacy.TOOLS,
                     ba.ROOT / 'code/geopilot_rsih'))}
    common |= {str(ba.ROOT / p) for p in ('code/usegeo_mesh_baseline/runner.py', 'experiments/geopilot_rsi/run.py')}
    return {'schema': 'manual-camera-mesh/1', 'formal_candidate': False, 'benchmark_eligible': False,
            'bindings': bound, 'source_models': sources, 'parameters': PARAMETERS, 'limits': LIMITS,
            'worker_bindings': {n: {p: bound[p] for p in common | set(sources[n]['model_hashes'])} for n in CASES},
            'runtime': launch['runtime'], 'host': platform.platform(), 'input_images': images,
            'input_manifest': manifest, 'coverage_source': historical['coverage_prepare_path'],
            'sfm_evidence_files': historical['sfm_evidence_files'],
            'undistort_options': undistort_options().todict(),
            'undistort_call': {'output_type': 'COLMAP', 'copy_policy': 'copy', 'num_patch_match_src_images': 20},
            'historical_P2_seconds': timings,
            'historical_P2_supervision': ba.read_json(ba.SOURCE / 'supervision.json'),
            'implicit_defaults': 'Bound OpenMVS 2.4.0 binaries/help; fresh workspaces without tool .cfg files'}


def isolation(case, name, manifest, manifest_path):
    """Use the existing tested default-deny profile and real negative probes."""
    allowed = [p for p in manifest['worker_bindings'][name]
        if not any(Path(p).is_relative_to(manifest['runtime'][k]) for k in ('prefix', 'base_prefix'))]
    allowed.append(str(manifest_path))
    reference = ba.ROOT / 'out/usegeo_benchmark/prepared/v1/evaluator-only/Dataset-1'
    forbidden = [reference / f for f in ('reference_manifest.json', 'full_lidar.las', 'refined_lidar.las',
                                       'publisher_mvs.las', 'publisher_camera_centers.csv')]
    forbidden += [ba.ROOT / 'out/usegeo_mesh_benchmark/paper-readiness-20260918/full-score-pass-1/Dataset-1/score.json',
                  CONTROL / ('B' if name == 'A' else 'A') / 'optimized-model/cameras.bin']
    if not all(p.is_file() for p in forbidden):
        raise ValueError('Required real forbidden probe targets are unavailable')
    request = case / 'isolation-request.json'
    ba.write(request, {'output': str(case), 'python': str(ba.RUNTIME / 'bin/python'),
                      'allowed': allowed, 'forbidden': [str(p) for p in forbidden]})
    script = """
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
const {makeProfile,probeIsolation}=await import(process.argv[2]);
const s=JSON.parse(readFileSync(process.argv[1],'utf8')),start=performance.now();
const profile=makeProfile(s.output,s.python,s.allowed),made=performance.now();
const result=probeIsolation(profile,s.output,s.python,s.allowed,s.forbidden,true);
writeFileSync(s.output+'/isolation.json',JSON.stringify({...result,profile,
  profile_sha256:createHash('sha256').update(readFileSync(profile)).digest('hex'),
  make_profile_seconds:(made-start)/1000,probe_seconds:(performance.now()-made)/1000},null,2)+'\\n',{flag:'wx'});
"""
    with (case / 'isolation.stdout').open('xb') as out, (case / 'isolation.stderr').open('xb') as err:
        result = subprocess.run([str(NODE), '--input-type=module', '-e', script, str(request), ISOLATION.as_uri()],
                                stdout=out, stderr=err, timeout=120, check=False)
    if result.returncode:
        raise RuntimeError('Isolation probe failed: ' + (case / 'isolation.stderr').read_text()[-4000:])
    return ba.read_json(case / 'isolation.json')


def coverage_complete(state):
    coverage = ba.read_json(state['coverage_path'])
    if (coverage['depth_image_count'] != 224 or coverage['depth_map_count'] != 224
            or any(len(x['depth_maps']) != 1 or x['omissions'] for x in coverage['images'].values())):
        raise ValueError('Full fresh depth coverage failed; retain this failed arm without retrying')


def worker(name, manifest):
    case, source = OUTPUT / name, manifest['source_models'][name]
    model_dir = case / 'input-model'
    model_dir.mkdir()
    copied = {}
    for original, digest in source['model_hashes'].items():
        target = model_dir / Path(original).name
        shutil.copyfile(original, target)
        target.chmod(0o444)
        copied[str(target)] = digest
    ba.verify(copied)
    model = ba.pc.Reconstruction(str(model_dir))
    model_identity(model, name, manifest['input_images'])
    if ba.fingerprint(model) != source['fingerprint']:
        raise ValueError('Copied optimized model geometry/observations changed')
    allowed = ba.centers(ba.INPUTS / 'Image_orientations_dataset1.xyz', manifest['input_images'])
    fit = ba.snapshot(model, allowed)
    ba.write(case / 'camera-fit.json', fit)
    prepare = case / 'prepare'
    prepare.mkdir()
    undistorted, scene = prepare / 'undistorted', prepare / 'scene.mvs'
    start = time.monotonic()
    ba.write(prepare / 'undistort-command.json', {'input_model': str(model_dir),
        'input_images': str(ba.INPUTS / 'images'), 'output': str(undistorted),
        'options': manifest['undistort_options'], **manifest['undistort_call']})
    ba.pc.undistort_images(str(undistorted), str(model_dir), str(ba.INPUTS / 'images'),
        output_type='COLMAP', copy_policy=ba.pc.CopyType.copy, num_patch_match_src_images=20,
        undistort_options=undistort_options())
    files = {p.name: p for p in (undistorted / 'images').iterdir()}
    if set(files) != set(manifest['input_images']) or any(not p.is_file() or p.is_symlink() for p in files.values()):
        raise ValueError('Fresh undistortion must cover exactly all 224 images')
    coverage = copy.deepcopy(ba.read_json(manifest['coverage_source']))
    coverage.update(policy='manual-reused-BA-sparse-full-input-v1',
                    sparse_evidence='sfm_ingested/matching_candidate reuse bound original P2 evidence; no new SfM',
                    source_BA_result=source['BA_result'], depth_image_count=0, depth_map_count=0)
    for image, item in coverage['images'].items():
        item.update(undistorted_sha256=ba.sha(files[image]), depth_maps=[], omissions=[])
    coverage_path = prepare / 'coverage-prepare.json'
    ba.write(coverage_path, coverage)
    argv = [str(ba.legacy.TOOLS / 'InterfaceCOLMAP'), '-i', str(undistorted), '-o', str(scene),
            '--image-folder', str(undistorted / 'images'), '--archive-type', '2', '--max-threads', '8']
    ba.write(prepare / 'interface-command.json', {'argv': argv, 'cwd': str(prepare)})
    tools.invoke(argv, prepare, prepare)
    with scene.open('rb') as stream:
        if stream.read(12) != bytes.fromhex('4d5653490700000000000000'):
            raise ValueError('Unexpected fresh MVS archive header')
        metadata = stream.read(16 * 1024**2)  # InterfaceCOLMAP's bound MVSI v7 format is uncompressed.
    if any(('undistorted/images/' + image).encode() not in metadata for image in files):
        raise ValueError('Fresh MVS relative image layout cannot be verified')
    diagnostics = {k: tools.diagnostic(v, str(case / 'camera-fit.json'), 'manual-BA-model-v1/' + k)
        for k, v in {'registered_fraction': 1., 'sparse_points': model.num_points3D(),
                     'mean_reprojection_error': fit['mean_point_reprojection_error_px']}.items()}
    state = {'stage': 'sparse', 'diagnostics': diagnostics, 'scene': str(scene), 'mesh': None,
             'transform': fit['fit_centers_transform'], 'mvs_workspace': str(prepare),
             'scene_id': manifest['input_manifest']['scene_id'], 'track': manifest['input_manifest']['track'],
             'input_manifest_sha256': coverage['input_manifest_sha256'], 'input_image_count': 224,
             'registered_image_count': 224, 'undistorted_image_count': 224,
             'coverage_prepare_path': str(coverage_path), 'coverage_path': str(coverage_path),
             'undistorted_files': [str(files[n]) for n in sorted(files)],
             'sfm_evidence_files': manifest['sfm_evidence_files'],
             'artifact_hashes': {str(scene): ba.sha(scene)}}
    ba.write(prepare / 'state.json', state)
    timings = {'undistort_and_interface_seconds': time.monotonic() - start}
    for action in ('densify', 'mesh'):
        if any(prepare.glob('*.cfg')):
            raise ValueError('Unexpected OpenMVS default configuration file')
        output = case / action
        output.mkdir()
        request = {'scope': 'development', 'inputs': str(ba.INPUTS), 'output': str(output),
                   'node': {'id': action, 'kind': 'tool', 'action': action, 'parameters': PARAMETERS[action]}, 'parent': state}
        ba.write(output / 'request.json', request)
        start = time.monotonic()
        state = tools.real_action(request)
        ba.write(output / 'state.json', state)
        timings[action + '_seconds'] = time.monotonic() - start
        coverage_complete(state)
    ba.verify(copied)
    ba.verify(manifest['worker_bindings'][name])
    ba.verify(state['artifact_hashes'])
    ba.verify({state['mesh']['path']: state['mesh']['sha256']})
    ba.write(case / 'worker-result.json', {'status': 'completed', 'branch': name, 'mesh': state['mesh'],
        'formal_candidate': False, 'mesh_quality_evaluated': False, 'copied_input_hashes': copied,
        'source_BA_result': source['BA_result'], 'phase_seconds': timings, 'diagnostics': state['diagnostics']})


def space_check():
    free = shutil.disk_usage(BASE).free
    if free < LIMITS['min_free_bytes']:
        raise RuntimeError('At least 60 GiB free space is required before each arm')
    return free


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python is unsupported')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', '_worker'))
    parser.add_argument('branch', choices=CASES, nargs='?')
    args = parser.parse_args()
    if (args.action == '_worker') != (args.branch is not None):
        parser.error('Only the internal worker accepts a fixed branch; run executes A then B')
    if args.action == '_worker':
        manifest = ba.read_json(OUTPUT / 'inputs.json')
        if manifest['parameters'] != PARAMETERS or manifest['limits'] != LIMITS:
            raise ValueError('Child parameters differ from the fixed declaration')
        ba.verify(manifest['worker_bindings'][args.branch])
        worker(args.branch, manifest)
        return
    manifest = inputs()
    if OUTPUT.exists() or OUTPUT.resolve() != OUTPUT:
        raise ValueError('A fresh output directory without symlinks is required')
    free = space_check()
    for name in CASES:
        model_identity(ba.pc.Reconstruction(manifest['source_models'][name]['model']), name, manifest['input_images'])
    if args.action == 'check':
        for name in CASES:
            with tempfile.TemporaryDirectory(prefix='camera-mesh-isolation-') as temporary:
                case = Path(temporary).resolve()
                ba.write(case / 'inputs.json', manifest)
                isolation(case, name, manifest, case / 'inputs.json')
        ba.verify(manifest['bindings'])
        print(json.dumps({'status': 'checked_without_undistort_or_MVS', 'bound_files': len(manifest['bindings']),
            'input_images': 224, 'free_bytes': free, 'historical_P2_seconds': manifest['historical_P2_seconds'],
            'parameters': PARAMETERS, 'limits': LIMITS, 'A_B_isolation_probes': 'PASS; numerical imports/help only'}))
        return
    OUTPUT.mkdir()
    ba.write(OUTPUT / 'inputs.json', manifest)
    inputs_sha = ba.sha(OUTPUT / 'inputs.json')
    for name in CASES:
        space_check()
        case = OUTPUT / name
        case.mkdir()
        proof = isolation(case, name, manifest, OUTPUT / 'inputs.json')
        command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin',
            'TMPDIR=' + str(case), 'PYTHONDONTWRITEBYTECODE=1', 'OPENSSL_CONF=/dev/null',
            '/usr/bin/sandbox-exec', '-f', proof['profile'], str(ba.RUNTIME / 'bin/python'), '-B',
            str(Path(__file__).resolve()), '_worker', name]
        ba.write(case / 'command.json', {'argv': command, 'limits': LIMITS,
                                       'inputs_sha256': inputs_sha})
        status = ba.supervise(command, case, timeout=LIMITS['timeout_seconds'], max_rss=LIMITS['max_rss_bytes'])
        status['final_log_hashes'] = {str(p): ba.sha(p) for p in case.rglob('*')
            if p.is_file() and (p.suffix in ('.log', '.stdout', '.stderr'))}
        ba.write(case / 'supervision.json', status)
        ba.verify(manifest['bindings'])
        ba.verify({str(OUTPUT / 'inputs.json'): inputs_sha, proof['profile']: proof['profile_sha256']})
        if status['returncode'] != 0 or status['reason'] is not None:
            raise RuntimeError(f'{name} failed; keep partial outputs and stop the A/B campaign')
        result = ba.read_json(case / 'worker-result.json')
        state = ba.read_json(case / 'mesh/state.json')
        if result['status'] != 'completed' or result['branch'] != name or result['mesh'] != state['mesh']:
            raise ValueError('Completed branch result identity mismatch')
        # Bind all files only after the worker and its descendants have stopped.
        result['output_hashes'] = {str(p): ba.sha(p) for p in case.rglob('*')
            if p.is_file() and p.name != 'result.json' and 'input-model' not in p.relative_to(case).parts}
        ba.write(case / 'result.json', result)
    print('A/B reconstruction completed; no geometry evaluation was run.')


if __name__ == '__main__':
    main()
