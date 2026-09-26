"""Publish one sealed real run using the frozen release validator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from strict_json import read_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUNDLE = ROOT / 'out/usegeo_benchmark/prepared/v1'
LOCK = ROOT / 'out/usegeo_benchmark/prepared/v1.lock.json'
RELEASE = ROOT / 'out/usegeo_mesh_benchmark/paper-readiness-20260918/release-v1/source/code'
EVALUATOR = ROOT / 'out/usegeo_mesh_benchmark/paper-readiness-20260918/clean-runtime/bin/python'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def contained(run, path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(run):
        raise ValueError('Unsafe or missing artifact: ' + str(path))
    return path


def verify_bindings(launch, offline_only=False):
    offline = launch['offline_validator']
    roots = tuple(Path(offline[key]) for key in
                  ('release_root', 'evaluator_runtime_root', 'evaluator_base_prefix'))
    bindings = {**launch['bindings'], **launch['input_hashes'],
                **launch.get('effective', {})}
    for path, expected in bindings.items():
        if offline_only and not any(Path(path).is_relative_to(root) for root in roots):
            continue
        if sha(path) != expected:
            raise ValueError('Source, runtime or input drift: ' + path)
    for path, target in launch.get('symlinks', {}).items():
        if offline_only and not any(Path(path).is_relative_to(root) for root in roots):
            continue
        if Path(path).resolve(strict=True) != Path(target):
            raise ValueError('Runtime symlink drift: ' + path)


def export(run, expected_launch_sha256):
    run = Path(run).resolve(strict=True)
    launch_path = contained(run, run / 'launch.json')
    if sha(launch_path) != expected_launch_sha256:
        raise ValueError('Launch provenance drift')
    launch = read_json(launch_path)
    status = read_json(run / 'run.json')
    result = read_json(run / 'result.json')
    if status['status'] != 'sealed' or status['scope'] != 'development' or status['benchmark_eligible']:
        raise ValueError('Run is not a fresh sealed real run')
    if status['returncode'] != 0 or status['cancellation_reason'] is not None or not status['frozen_bindings_unchanged']:
        raise ValueError('Incomplete numerical execution')
    candidate = launch.get('candidate')
    valid_calls = (type(result['calls']) is int and 3 <= result['calls'] <= 16) if candidate else result['calls'] == 3
    if (result['status'] != 'sealed' or result['scope'] != 'development'
            or result['delivery'] not in (('terminal', 'recovered') if candidate else ('terminal',))
            or not valid_calls):
        raise ValueError('Incomplete reconstruction result')
    state = result['state']
    if (state['stage'], state['scene_id'], state['track'], state['input_image_count']) != ('mesh', 'Dataset-1', 'rgb-oriented', 224):
        raise ValueError('Wrong scene, track or coverage')
    if state['depth_map_count'] <= 0 or state['registered_image_count'] <= 0:
        raise ValueError('Missing real dense provenance')
    bundle_lock = read_json(LOCK)
    expected_input = bundle_lock['scenes']['Dataset-1']['input_manifest_sha256']['rgb-oriented']
    if (state['input_manifest_sha256'] != launch['input_manifest_sha256']
            or launch['input_manifest_sha256'] != launch['frozen_input_manifest_sha256']
            or launch['input_manifest_sha256'] != expected_input):
        raise ValueError('Input lineage mismatch')
    program = Path(launch['program_path'])
    if (not candidate and program != HERE / 'p0.json') or sha(program) != launch['program_bytes_sha256'] or result['program_sha256'] != launch['program_content_sha256']:
        raise ValueError('Program lineage mismatch')
    if candidate and (candidate['program_sha256'] != launch['program_bytes_sha256']
            or candidate['parent_program_sha256'] != '619cc3f49b06951fffc7f9ad2f19e45fe7da810596e074c0db58f64130736b8f'
            or not candidate['bindings']
            or any(launch['bindings'].get(path) != expected for path, expected in candidate['bindings'].items())):
        raise ValueError('Candidate provenance mismatch')
    verify_bindings(launch)
    if sha(run / 'worker.sb') != launch.get('isolation_profile_sha256'):
        raise ValueError('Isolation profile drift')
    for path, expected in state['artifact_hashes'].items():
        if sha(contained(run, path)) != expected:
            raise ValueError('Numerical artifact drift: ' + path)
    for key in ('scene', 'dense', 'dense_cloud', 'depth_manifest', 'mesh_scene', 'native_obj',
                'coverage_path', 'coverage_prepare_path'):
        if key not in state or str(contained(run, state[key])) not in state['artifact_hashes']:
            raise ValueError('Missing dense/surface provenance: ' + key)
    coverage = read_json(contained(run, state['coverage_path']))
    if (coverage['policy'] != 'fixed-p0-full-input-v1' or
            coverage['input_manifest_sha256'] != expected_input or
            coverage['input_count'] != 224 or coverage['sfm_input_count'] != 224 or
            sorted(coverage['images']) != launch['input_image_ids']):
        raise ValueError('Full input coverage evidence mismatch')
    if (coverage['registered_count'] != state['registered_image_count'] or
            coverage['undistorted_count'] != state['undistorted_image_count'] or
            coverage['depth_image_count'] != state['depth_image_count'] or
            coverage['depth_map_count'] != state['depth_map_count']):
        raise ValueError('Registration/undistortion/depth count mismatch')
    depth_manifest = read_json(contained(run, state['depth_manifest']))
    depth_records = []
    for name, item in coverage['images'].items():
        if not item['sfm_ingested']:
            raise ValueError('Input absent from SfM: ' + name)
        for entry in item['depth_maps']:
            if (not item['registered'] or not item['undistorted_sha256'] or
                    depth_manifest.get(Path(entry['path']).name) !=
                    {'path': entry['path'], 'sha256': entry['sha256'], 'image_id': name}):
                raise ValueError('Unbound depth image: ' + name)
            depth_records.append(entry['path'])
    if len(depth_records) != coverage['depth_map_count'] or len(set(depth_records)) != len(depth_records):
        raise ValueError('Depth map coverage mismatch')
    if set(depth_manifest) != {Path(path).name for path in depth_records}:
        raise ValueError('Depth manifest file set mismatch')
    mesh = contained(run, state['mesh']['path'])
    if sha(mesh) != state['mesh']['sha256']:
        raise ValueError('Final mesh drift')
    submission = run / 'submission'
    staging = run / '.submission-pending'
    if submission.exists() or staging.exists():
        raise ValueError('Submission already exists')
    staging.mkdir()
    try:
        target = staging / 'mesh.ply'
        # O_NOFOLLOW is the trust boundary; compare both source and copied bytes.
        fd = os.open(mesh, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(fd)
            if not os.path.isfile(mesh) or before.st_size <= 0:
                raise ValueError('Invalid mesh file')
            with os.fdopen(os.dup(fd), 'rb') as source, target.open('xb') as dest:
                shutil.copyfileobj(source, dest, 8 * 1024**2)
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('Mesh changed during copy')
        finally:
            os.close(fd)
        if sha(mesh) != state['mesh']['sha256'] or sha(target) != state['mesh']['sha256']:
            raise ValueError('Mesh copy mismatch')
        contract = {
            'schema_version': 'usegeo-mesh-submission-1.0',
            'protocol_version': 'usegeo-mesh-rgb-oriented-v1',
            'track': 'rgb-oriented', 'scene_id': 'Dataset-1', 'product': 'mesh',
            'geometry': 'mesh.ply',
            'configuration_id': ('geopilot-rsih-candidate-' if candidate else 'geopilot-rsih-p0-') + launch['program_bytes_sha256'][:16],
            'method_metadata': {'name': 'COLMAP-OpenMVS conditional candidate' if candidate else 'COLMAP-OpenMVS fixed dense P0',
                                'program_sha256': launch['program_bytes_sha256'],
                                'source_sha256': launch['bindings'][str(HERE / 'tools.py')],
                                'input_image_count': coverage['input_count'],
                                'registered_image_count': coverage['registered_count'],
                                'undistorted_image_count': coverage['undistorted_count'],
                                'depth_image_count': coverage['depth_image_count'],
                                'coverage_sha256': sha(state['coverage_path'])},
        }
        (staging / 'submission.json').write_text(json.dumps(contract, indent=2) + '\n')
        env = {'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(RELEASE)}
        command = [str(EVALUATOR), '-B', '-m', 'usegeo_mesh_benchmark.benchmark',
                   'validate-submission', '--bundle', str(BUNDLE), '--bundle-lock', str(LOCK),
                   '--scene', 'Dataset-1', '--submission', str(staging)]
        checked = subprocess.run(command, capture_output=True, text=True, env=env)
        verify_bindings(launch, offline_only=True)
        (run / 'validation.json').write_text(json.dumps({
            'argv': command, 'returncode': checked.returncode,
            'stdout': checked.stdout, 'stderr': checked.stderr,
            'offline_validator': launch['offline_validator']}, indent=2))
        if checked.returncode != 0 or read_json_value(checked.stdout).get('status') != 'valid':
            raise ValueError('Official validator rejected submission')
        if sha(contained(run, launch_path)) != expected_launch_sha256:
            raise ValueError('Launch provenance drift')
        staging.rename(submission)
        status.update(benchmark_eligible=True, submission=str(submission),
                      submission_mesh_sha256=sha(submission / 'mesh.ply'),
                      submission_contract_sha256=sha(submission / 'submission.json'),
                      input_coverage={'input': coverage['input_count'],
                                      'matching_candidates': coverage['matching_candidate_count'],
                                      'registered': coverage['registered_count'],
                                      'undistorted': coverage['undistorted_count'],
                                      'depth_images': coverage['depth_image_count'],
                                      'depth_maps': coverage['depth_map_count'],
                                      'manifest_sha256': sha(state['coverage_path'])},
                      evaluator_source_sha256=launch['offline_validator']['release_source_tree_sha256'],
                      evaluator_runtime_sha256=launch['offline_validator']['evaluator_runtime_tree_sha256'])
        temporary = run / '.run-exported.json'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(run / 'run.json')
        return status
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_json_value(value):
    return json.loads(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--expected-launch-sha256', required=True)
    try:
        args = parser.parse_args()
        print(json.dumps(export(args.run, args.expected_launch_sha256)))
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
