"""Development-only U-raster parent for the unchanged six-metric worker."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

ENTRY = Path(__file__).resolve()
ROOT = ENTRY.parents[2]
BASE = ROOT/'out/geopilot-research-20260925'
PLAN = BASE/'publisher-raster-score-plan.md'
SOURCE = BASE/'publisher-raster-mesh-run'
OUTPUT = BASE/'publisher-raster-scores'
U = BASE/'publisher-raster-input/input_manifest.json'
CLEAN = ROOT/'out/usegeo_mesh_benchmark/paper-readiness-20260918/clean-runtime'
SCORER = ENTRY.parent/'evaluator/benchmark.py'
PROTOCOL = SCORER.with_name('protocol_v1.json')
BUNDLE = ROOT/'out/usegeo_benchmark/prepared/v1'
LOCK = BUNDLE.with_name('v1.lock.json')
LAUNCH = ROOT/'out/geopilot-learning-20260923/p2-run/launch.json'
ARMS = ('F-U', 'S-U')
IDENTITY = {'schema_version': 'publisher-undistorted-development-score/1',
            'track': 'manual-publisher-undistorted', 'input_variant': 'U',
            'scope': 'development', 'formal_candidate': False, 'benchmark_eligible': False,
            'scene_id': 'Dataset-1'}
PINS = {ENTRY.parent/'publisher_raster_mesh.py': '973aab47b2ca1ace4d92a643f8f1d4e5c587ea8e4b0c1bda5429a710ae2bc37a',
        BASE/'publisher-raster-mesh-plan.md': 'f77076efe708d0ea0aa82bf64536a34228a0a0bb670708901d7a072ce459416a',
        SCORER: '800c751822049c6a8814ab7d7d388fd38a5b96fa918df4ed97374c43393ea285',
        PROTOCOL: 'a16e8a9dabd4a2a2188a391ea2205ca941717639ab95a626d16b10c06dfab622',
        U: '0327334e7607a77ab81fee4ffd6dcca4a90d76a6d45221956f78307b3ebd4ecc',
        LAUNCH: '2a5023086874bd1d138164634f109540fcf1ce57f9ab6a64a66c3d3955145e88',
        LOCK: 'bdff58a760d90327b5ed40876670cf3b2bcf38e08ed189d56529919dd738fe27',
        BUNDLE/'manifest.json': 'b4a99f53ab7297a159913776dd3ce97fcf8741f364cf713dda9fdd1a35771e40',
        BUNDLE/'evaluator-only/Dataset-1/reference_manifest.json': '56dcb79402d4c3001b7cbc6d34ce5ef7027bc699e5a1c41e4c4aa35b8350f795'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(bindings):
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError('Bound bytes changed: '+str(path))


def load_evaluator():
    verify({str(p): PINS[p] for p in (SCORER, PROTOCOL)})
    spec = importlib.util.spec_from_file_location('unchanged_u_metric_worker', SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.runtime_versions()
    return module


def runtime_bindings():
    verify({str(LAUNCH): PINS[LAUNCH]})
    launch = read(LAUNCH)
    roots = tuple(Path(launch['offline_validator'][k]) for k in
                  ('release_root', 'evaluator_runtime_root', 'evaluator_base_prefix'))
    bindings = {p: h for p, h in launch['bindings'].items()
                if any(Path(p).is_relative_to(r) for r in roots)}
    if not bindings:
        raise ValueError('Missing frozen runtime bindings')
    for p, target in launch.get('symlinks', {}).items():
        if any(Path(p).is_relative_to(r) for r in roots) and str(Path(p).resolve()) != target:
            raise ValueError('Runtime symlink changed: '+p)
    verify(bindings)
    return bindings


def ready():
    """Read completed native producer evidence; missing geometry is not a score."""
    if not (SOURCE/'result.json').is_file():
        return None
    campaign, final = read(SOURCE/'inputs.json'), read(SOURCE/'result.json')
    if final['status'] != 'completed':
        return None
    if (campaign['schema'] != 'publisher-raster-mesh/1' or campaign['formal_candidate'] is not False
            or campaign['readiness'] != 'ready' or set(campaign['sources']) != set(ARMS)
            or campaign['input_manifest_sha256'] != PINS[U] or len(campaign['names']) != 224):
        raise ValueError('Wrong mesh campaign/U identity')
    bindings = {str(SOURCE/p): sha(SOURCE/p) for p in ('inputs.json', 'result.json')}
    bindings.update(campaign['bindings'])
    if bindings.get(str(U)) != PINS[U]:
        raise ValueError('Producer does not bind the frozen U input')
    records = {}
    for arm in ARMS:
        case = SOURCE/arm
        result_path = case/'result.json'
        record = read(result_path)
        if record['status'] != 'completed':
            return None
        if record['branch'] != arm:
            raise ValueError('Wrong producer arm')
        digest = sha(result_path)
        declared = final['arms'][arm]
        if declared['path'] != str(result_path) or declared['sha256'] != digest:
            raise ValueError('Campaign/arm identity mismatch')
        bindings[str(result_path)] = digest
        outputs = record['output_hashes']
        required = [case/p for p in ('mesh/mesh.ply', 'mesh/state.json', 'command.json', 'supervision.json',
                    'inputs.json', 'prepare/state.json', 'densify/state.json', 'densify/request.json',
                    'mesh/request.json', 'export-verification.json')]
        if not all(str(p) in outputs for p in required):
            raise ValueError('Incomplete producer output bindings')
        bindings.update(outputs)
        state, supervision = read(case/'mesh/state.json'), read(case/'supervision.json')
        mesh = case/'mesh/mesh.ply'
        if (record['mesh'] != state['mesh'] or record['mesh']['path'] != str(mesh)
                or outputs[str(mesh)] != record['mesh']['sha256']
                or supervision['returncode'] != 0 or supervision['reason'] is not None):
            raise ValueError('Mesh completion identity mismatch')
        command, worker_input = read(case/'command.json'), read(case/'inputs.json')
        source = campaign['sources'][arm]
        if (command['inputs_sha256'] != sha(case/'inputs.json') or worker_input['source'] != source
                or worker_input['parameters'] != campaign['parameters']
                or record['export_verification'].get('passed') is not True
                or state['input_manifest_sha256'] != PINS[U]):
            raise ValueError('Executed source/settings/export chain mismatch')
        if (campaign['parameters']['pair_order'] != list(ARMS)
                or state['track'] != IDENTITY['track'] or state['stage'] != 'mesh'):
            raise ValueError('Wrong producer track or stage')
        selected = record['source_selected_model']
        if selected != source['selected_model']:
            raise ValueError('Selected native model changed')
        if not selected or selected['transform'] is None or state['transform'] != selected['transform']:
            raise ValueError('Missing selected model/world transform')
        copied = record['copied_input_hashes']
        expected_copies = {str(case/'input-model'/Path(p).name): h for p, h in source['model_hashes'].items()}
        if copied != expected_copies or len(copied) != 5 or any(outputs.get(p) != h for p, h in copied.items()):
            raise ValueError('Incomplete five-file native model provenance')
        coverage = Path(record['coverage_path'])
        if str(coverage) not in outputs or type(record['full_224_coverage']) is not bool:
            raise ValueError('Missing explicit image coverage')
        coverage_value = read(coverage)
        if set(coverage_value['images']) != set(campaign['names']):
            raise ValueError('Coverage does not retain all 224 names')
        actual_full = all(not row['omissions'] and len(row['depth_maps']) == 1
                          for row in coverage_value['images'].values())
        if actual_full != record['full_224_coverage']:
            raise ValueError('Coverage summary mismatch')
        for action, previous in (('densify', 'prepare'), ('mesh', 'densify')):
            request = read(case/action/'request.json')
            if (request['parent'] != read(case/previous/'state.json')
                    or request['node']['parameters'] != campaign['parameters'][action]):
                raise ValueError('Actual MVS parameter/state chain mismatch')
        records[arm] = {'source_mesh': str(mesh), 'source_mesh_sha256': record['mesh']['sha256'],
                        'source_result': str(result_path), 'source_result_sha256': digest,
                        'selected_model': selected, 'full_224_coverage': record['full_224_coverage'],
                        'coverage_path': str(coverage), 'coverage_sha256': outputs[str(coverage)]}
    verify(bindings)
    return {'bindings': bindings, 'common_settings': campaign['parameters'], 'arms': records}


def worker_run(evaluator, case, mesh, bundle, *, synthetic=False):
    """Same unchanged worker and watchdog in production and synthetic checks."""
    protocol = evaluator.load_protocol()
    limits = protocol['limits']
    started = time.monotonic()
    token = hashlib.sha256(os.urandom(32)).hexdigest()
    order = {'token': token, 'parent_pid': os.getpid(), 'bundle': str(bundle),
             'scene': 'Dataset-1', 'mesh': str(mesh), 'result': str(case/'worker-result.json'),
             'progress': str(case/'progress.jsonl'), 'cancel': str(case/'cancel.json'),
             'deadline': started+limits['scene_timeout_seconds']}
    write(case/'work-order.json', order)
    env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONDONTWRITEBYTECODE': '1',
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'VECLIB_MAXIMUM_THREADS': '1', 'USEGEO_MESH_WORKER_TOKEN': token}
    command = [str(CLEAN/'bin/python'), '-B', str(SCORER), 'score-worker', '--work-order', str(case/'work-order.json')]
    write(case/'command.json', {'argv': command, 'cwd': str(ROOT), 'environment': env,
                              'synthetic_only': synthetic, 'parent_pid': os.getpid(),
                              'limits': limits, 'work_order_sha256': sha(case/'work-order.json')})
    receipt = {'started_at': datetime.now(timezone.utc).isoformat(), 'parent_pid': os.getpid(),
               'command_sha256': sha(case/'command.json'), 'synthetic_only': synthetic}
    peak, error = None, None
    with (case/'stdout.log').open('x') as stdout, (case/'stderr.log').open('x') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, text=True)
        receipt['worker_pid'] = process.pid
        write(case/'execution.json', receipt)
        try:
            peak, _, _ = evaluator._watch_process(process, started, limits['scene_timeout_seconds'],
                limits['max_peak_rss_bytes'], limits['watchdog_period_seconds'], case/'cancel.json')
            if process.returncode:
                raise evaluator.EvaluatorFailure('Worker exit '+str(process.returncode))
        except BaseException as exc:
            peak = getattr(exc, 'peak_rss_bytes', peak)
            error = repr(exc)
            raise
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            write(case/'supervision.json', {**receipt, 'finished_at': datetime.now(timezone.utc).isoformat(),
                  'returncode': process.returncode, 'error': error, 'peak_worker_rss_bytes': peak,
                  'elapsed_seconds': time.monotonic()-started, 'limits': limits})
    result = read(case/'worker-result.json')
    if not set(evaluator.METRIC_NAMES).issubset(result['surface_metrics']):
        raise ValueError('Missing one of six metrics')
    for key in evaluator.METRIC_NAMES:
        value = result['surface_metrics'][key]
        if not math.isfinite(value) or value < 0 or (key.endswith('0_20') and value > 1):
            raise ValueError('Invalid metric: '+key)
    if result['counts']['surface_samples'] != 1000000 or result['surface_metrics']['accuracy_best90_count'] != 900000:
        raise ValueError('Wrong sampling denominator')
    if not synthetic and (result['counts']['full_lidar_points'] != 105922149 or
                          result['counts']['refined_lidar_points'] != 88655810):
        raise ValueError('Wrong reference denominator')
    progress = [json.loads(line) for line in (case/'progress.jsonl').read_text().splitlines()]
    if not progress or progress[-1]['stage'] != 'worker_complete':
        raise ValueError('Missing actual worker completion progress')
    return result


def score(evaluator, source):
    OUTPUT.mkdir(exist_ok=False)
    bindings = {str(p): h for p, h in PINS.items()}
    statuses = {arm: {'status': 'not_executed'} for arm in ARMS}
    try:
        verify(bindings)
        bindings.update(runtime_bindings())
        bindings.update(source['bindings'])
        bindings.update({str(ENTRY): sha(ENTRY), str(PLAN): sha(PLAN)})
        if evaluator._peak_rss_bytes(os.getpid()) is None:
            raise evaluator.EvaluatorFailure('Host ps/RSS permission unavailable before score')
        evaluator.verify_bundle(BUNDLE, LOCK)
        ref = BUNDLE/'evaluator-only/Dataset-1'
        for record in read(ref/'reference_manifest.json')['files'].values():
            bindings[str(ref/record['path'])] = record['sha256']
        verify(bindings)
        write(OUTPUT/'inputs.json', {**IDENTITY, 'bindings': bindings, 'source': source,
                                    'runtime': evaluator.runtime_versions(), 'sampling': evaluator.load_protocol()['sampling']})
        for arm in ARMS:
            case = OUTPUT/arm
            case.mkdir()
            original = Path(source['arms'][arm]['source_mesh'])
            original_hash = sha(original)
            try:
                if original_hash != source['arms'][arm]['source_mesh_sha256']:
                    raise ValueError('Source mesh changed')
                snapshot = case/'mesh.ply'
                snap_hash = evaluator._snapshot(original, snapshot,
                    evaluator.load_protocol()['limits']['max_mesh_bytes'], 'MESH_SNAPSHOT_INVALID')
                if snap_hash != original_hash or snapshot.stat().st_ino == original.stat().st_ino:
                    raise ValueError('Mesh snapshot bytes/inode mismatch')
                evaluator.read_ply(snapshot)
                worker = worker_run(evaluator, case, snapshot, BUNDLE)
                verify(bindings)
                if sha(original) != original_hash or sha(snapshot) != original_hash:
                    raise ValueError('Source or snapshot changed during score')
                result = {**IDENTITY, 'status': 'valid', 'arm': arm, 'input_manifest_sha256': PINS[U],
                          'alignment': {'kind': 'identity', 'fitted_degrees_of_freedom': 0, 'geometry_repairs': 0,
                                        'raycast_origin_m': worker['raycast_origin_m']}, **worker}
                write(case/'score.json', result)
                write(case/'run_manifest.json', {**IDENTITY, 'schema_version': 'publisher-undistorted-development-run/1',
                      'status': 'valid', 'arm': arm, 'source': source['arms'][arm], 'bindings': bindings,
                      'source_before_sha256': original_hash, 'source_after_sha256': sha(original),
                      'snapshot_sha256': sha(snapshot), 'common_settings': source['common_settings'],
                      'sampling': evaluator.load_protocol()['sampling'], 'runtime': evaluator.runtime_versions(),
                      'supervision': read(case/'supervision.json'),
                      'output_hashes': {str(p): sha(p) for p in sorted(case.iterdir()) if p.is_file()}})
                statuses[arm] = {'status': 'valid', 'score_sha256': sha(case/'score.json'),
                                 'manifest_sha256': sha(case/'run_manifest.json')}
            except BaseException as exc:
                statuses[arm] = {'status': 'failed', 'error': repr(exc)}
                write(case/'failure.json', {**IDENTITY, **statuses[arm], 'source_before_sha256': original_hash,
                    'source_after_sha256': sha(original) if original.is_file() else None})
                raise
    except BaseException as exc:
        write(OUTPUT/'result.json', {**IDENTITY, 'status': 'failed', 'error': repr(exc), 'arms': statuses})
        raise
    write(OUTPUT/'result.json', {**IDENTITY, 'status': 'completed', 'arms': statuses})


def synthetic_check(evaluator):
    import laspy
    import numpy as np
    case = Path(tempfile.mkdtemp(prefix='publisher-raster-score-check-')).resolve()
    ref = case/'bundle/evaluator-only/Dataset-1'
    ref.mkdir(parents=True)
    vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]], dtype='<f8')
    mesh = case/'mesh.ply'
    mesh.write_bytes(evaluator.PLY_HEADER.replace(b'{vertices}', b'3').replace(b'{triangles}', b'1')+
                     vertices.tobytes()+b'\x03'+np.array([0,1,2], dtype='<i4').tobytes())
    before = sha(mesh)
    for name in ('full_lidar.las', 'refined_lidar.las'):
        las = laspy.LasData(laspy.LasHeader(point_format=3, version='1.2'))
        las.x=[0,.5,0]; las.y=[0,0,.5]; las.z=[0,0,0]
        las.write(ref/name)
    result = worker_run(evaluator, case, mesh, case/'bundle', synthetic=True)
    assert result['counts']['full_lidar_points'] == result['counts']['refined_lidar_points'] == 3
    assert sha(mesh) == before
    write(case/'check.json', {'status': 'PASS', 'synthetic_only': True, 'real_reference_opened': False,
          'entry_sha256': sha(ENTRY), 'plan_sha256': sha(PLAN), 'scorer_sha256': sha(SCORER),
          'worker_counts': result['counts'], 'hashes': {str(p): sha(p) for p in case.rglob('*') if p.is_file()}})
    return {'status': 'PASS', 'synthetic_only': True, 'evidence': str(case/'check.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'ready', 'run'))
    args = parser.parse_args()
    if not __debug__ or Path(sys.prefix).resolve() != CLEAN.resolve():
        raise RuntimeError('Use frozen clean-runtime Python -B, never -O')
    evaluator = load_evaluator()
    if args.action == 'check':
        print(json.dumps(synthetic_check(evaluator)))
        return
    source = ready()
    if source is None:
        print(json.dumps({**IDENTITY, 'status': 'not_ready', 'reason': 'Complete F-U/S-U producer meshes required'}))
        return
    if args.action == 'ready':
        print(json.dumps({**IDENTITY, 'status': 'ready', 'arms': source['arms']}))
    else:
        score(evaluator, source)


if __name__ == '__main__':
    main()
