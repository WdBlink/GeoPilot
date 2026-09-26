"""One fixed A/A-repeat/B forward diagnostic, never a six-metric formal score."""
import json
import os
import platform
import resource
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
from sampling_diagnostic import METRICS, SOURCE, check, evaluator, np, summarize

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / 'out/geopilot-research-20260925/fixed-prefix-control'
OUTPUT = ROOT / 'out/geopilot-research-20260925/prefix-forward-run'
PLAN = ROOT / 'out/geopilot-research-20260925/prefix-forward-plan.md'
CLEAN = ROOT / 'out/usegeo_mesh_benchmark/paper-readiness-20260918/clean-runtime'
CASES = {'A': .25, 'A-repeat': .25, 'B': .5}
PREFLIGHT_SHA = 'bd116b90a5001fea0c481e0a08eb51af0773225510dc33d9748416b4042e9f2c'
SAMPLING_SHA = '3a462aa2dc500bbf817212a61bdcf959cceb39af0b544c09e893a425e2b6c119'
EVALUATOR_SHA = '800c751822049c6a8814ab7d7d388fd38a5b96fa918df4ed97374c43393ea285'
PROTOCOL_SHA = 'a16e8a9dabd4a2a2188a391ea2205ca941717639ab95a626d16b10c06dfab622'
REFERENCE_SHA = '56dcb79402d4c3001b7cbc6d34ce5ef7027bc699e5a1c41e4c4aa35b8350f795'
SEED, SAMPLES, CHUNK = 20260916, 1000000, 100000


def cases_ready(bind):
    """Reject incomplete or inconsistent controls before reading any mesh geometry."""
    bind(CAMPAIGN / 'preflight.json', PREFLIGHT_SHA)
    preflight = evaluator._strict_json(CAMPAIGN / 'preflight.json', 'INVALID_PREFLIGHT')
    source_path = Path(preflight['source']) / 'nodes/densify/state.json'
    bind(source_path, preflight['bindings'][str(source_path)])
    source = evaluator._strict_json(source_path, 'INVALID_DENSE_STATE')
    meshes = {}
    for name, value in CASES.items():
        case = CAMPAIGN / name
        records, digests = {}, {}
        for filename in ('status.json', 'request.json', 'command.json',
                         'prefix-bindings.json', 'mesh/state.json'):
            path = case / filename
            digests[str(path)] = bind(path)
            records[filename] = evaluator._strict_json(path, 'INVALID_CASE_RECORD')
        status, request = records['status.json'], records['request.json']
        bindings, state = records['prefix-bindings.json'], records['mesh/state.json']
        for path, digest in digests.items():
            if path != str(case / 'status.json') and status['output_hashes'][path] != digest:
                raise ValueError(f'Control record changed: {path}')
        if (status['status'] != 'completed' or status['case'] != name
                or status['decimate'] != value or state['stage'] != 'mesh'
                or request['node']['id'] != name
                or request['node']['action'] != 'mesh'
                or request['node']['parameters'] != {'decimate': value}
                or records['command.json']['preflight_sha256'] != PREFLIGHT_SHA
                or bindings['original_hashes'] != source['artifact_hashes']
                or request['parent']['artifact_hashes'] != bindings['executed_hashes']
                or state['transform'] != preflight['transform']
                or request['parent']['transform'] != preflight['transform']):
            raise ValueError(f'Fixed-prefix identity mismatch: {name}')
        dense = case / 'prefix/nodes/densify/densify.mvs'
        dense_sha = source['artifact_hashes'][source['dense']]
        if (state['dense'] != str(dense) or request['parent']['dense'] != str(dense)
                or bindings['original_to_copy'][source['dense']] != str(dense)
                or bindings['executed_hashes'][str(dense)] != dense_sha):
            raise ValueError(f'Dense identity mismatch: {name}')
        bind(dense, dense_sha)
        mesh = case / 'mesh/mesh.ply'
        if (state['mesh']['path'] != str(mesh) or status['mesh'] != state['mesh']
                or status['output_hashes'][str(mesh)] != state['mesh']['sha256']):
            raise ValueError(f'Mesh identity mismatch: {name}')
        bind(mesh, state['mesh']['sha256'])
        meshes[name] = mesh
    return meshes


def self_check():
    """Tiny real-file gate check; no PLY parsing, LiDAR read or scoring."""
    import tempfile
    from unittest.mock import patch

    if not __debug__:
        raise RuntimeError('Self-check requires assertions')
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        campaign, source = root / 'controls', root / 'source'
        transform = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        dense_source = str(source / 'nodes/densify/densify.mvs')

        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))

        def bind(path, expected=None):
            digest = evaluator.sha256_file(path)
            if expected is not None and digest != expected:
                raise ValueError('Synthetic input changed')
            return digest

        dense_path = Path(dense_source)
        dense_path.parent.mkdir(parents=True)
        dense_path.write_bytes(b'dense fixture')
        dense_sha = bind(dense_path)
        original = {dense_source: dense_sha}
        source_state = source / 'nodes/densify/state.json'
        write(source_state, {'dense': dense_source, 'artifact_hashes': original})
        write(campaign / 'preflight.json', {'source': str(source), 'transform': transform,
              'bindings': {str(source_state): bind(source_state)}})
        preflight_sha = bind(campaign / 'preflight.json')
        for name, value in CASES.items():
            case = campaign / name
            dense = case / 'prefix/nodes/densify/densify.mvs'
            mesh = case / 'mesh/mesh.ply'
            for path, content in ((dense, b'dense fixture'), (mesh, b'mesh fixture')):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            mesh_record = {'path': str(mesh), 'sha256': bind(mesh)}
            executed = {str(dense): dense_sha}
            records = {
                'request.json': {'node': {'id': name, 'action': 'mesh', 'parameters': {'decimate': value}},
                                 'parent': {'dense': str(dense), 'transform': transform, 'artifact_hashes': executed}},
                'command.json': {'preflight_sha256': preflight_sha},
                'prefix-bindings.json': {'original_hashes': original, 'executed_hashes': executed,
                                        'original_to_copy': {dense_source: str(dense)}},
                'mesh/state.json': {'stage': 'mesh', 'dense': str(dense), 'transform': transform, 'mesh': mesh_record},
            }
            for filename, record in records.items():
                write(case / filename, record)
            write(case / 'status.json', {'status': 'completed', 'case': name, 'decimate': value,
                  'mesh': mesh_record, 'output_hashes': {str(p): bind(p) for p in
                  [case / filename for filename in records] + [mesh]}})
        with patch(__name__ + '.CAMPAIGN', campaign), patch(__name__ + '.PREFLIGHT_SHA', preflight_sha):
            assert len(cases_ready(bind)) == 3
            for path, replacement in (
                (campaign / 'A/status.json', (campaign / 'A/status.json').read_bytes().replace(b'completed', b'failed')),
                (campaign / 'A/prefix/nodes/densify/densify.mvs', b'changed dense'),
                (campaign / 'B/mesh/mesh.ply', b'changed mesh'),
            ):
                saved = path.read_bytes()
                path.write_bytes(replacement)
                try:
                    cases_ready(bind)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'Invalid control accepted: {path.name}')
                finally:
                    path.write_bytes(saved)
    print('Binding gates passed: completed controls accepted; failed case, changed dense and changed mesh rejected.')


def main():
    """Run the one preregistered diagnostic using the existing clean runtime."""
    if not __debug__:
        raise RuntimeError('Optimized Python is unsupported: validation assertions must remain active')
    if len(sys.argv) != 1:
        raise ValueError('This fixed diagnostic accepts no arguments')
    if platform.system() != 'Darwin' or Path(sys.prefix).resolve() != CLEAN.resolve():
        raise RuntimeError('Use the fixed Darwin clean-runtime Python; RSS units are bytes')
    if OUTPUT.resolve() != OUTPUT:
        raise ValueError('Output path must not contain symlinks')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = {'status': 'running', 'formal_score': False, 'completeness_evaluated': False,
              'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'python_executable': sys.executable,
              'seed': SEED, 'surface_samples': SAMPLES, 'query_chunk': CHUNK,
              'query_workers': 1, 'records': [], 'input_hashes': {}}

    def save():
        result['elapsed_seconds'] = time.monotonic() - start
        result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = OUTPUT / 'result.json.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(OUTPUT / 'result.json')

    def bind(path, expected=None):
        path = Path(path)
        if path.resolve() != path.absolute() or not path.is_file():
            raise ValueError(f'Missing or linked input: {path}')
        digest = evaluator.sha256_file(path)
        result['input_hashes'][str(path)] = digest
        if expected is not None and digest != expected:
            raise ValueError(f'Input identity mismatch: {path}')
        return digest

    def alarm(*_):
        raise TimeoutError('30 minute diagnostic protection')

    try:
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(1800)
        result['versions'] = evaluator.runtime_versions()
        bind(Path(__file__).resolve())
        bind(PLAN)
        bind(Path(__file__).with_name('sampling_diagnostic.py').resolve(), SAMPLING_SHA)
        bind(SOURCE.resolve(), EVALUATOR_SHA)
        bind(SOURCE.with_name('protocol_v1.json').resolve(), PROTOCOL_SHA)
        meshes = cases_ready(bind)
        ref = ROOT / 'out/usegeo_benchmark/prepared/v1/evaluator-only/Dataset-1'
        bind(ref / 'reference_manifest.json', REFERENCE_SHA)
        reference = evaluator._strict_json(ref / 'reference_manifest.json', 'INVALID_REFERENCE')
        bind(ref / 'full_lidar.las', reference['files']['full_lidar']['sha256'])
        save()
        full = evaluator._las_points(ref / 'full_lidar.las', check)
        assert len(full) == 105922149, 'Full LiDAR point count changed'
        result['full_lidar_points'] = len(full)
        print('Building one fixed full-LiDAR tree', flush=True)
        tree = evaluator.KDTree(full)
        check()
        for name, path in meshes.items():
            case_start = time.monotonic()
            vertices, triangles = evaluator.read_ply(path)
            check()
            samples = evaluator.sample_surface(vertices, triangles, SAMPLES, SEED, check)
            distances = np.empty(SAMPLES, dtype=np.float64)
            for offset in range(0, SAMPLES, CHUNK):
                check()
                distances[offset:offset + CHUNK] = tree.query(
                    samples[offset:offset + CHUNK], workers=1)[0]
            metrics = summarize(distances)
            distance_path = OUTPUT / f'{name}-{SEED}-distances.npy'
            np.save(distance_path, distances)
            result['records'].append({
                'case': name, 'decimate': CASES[name], 'mesh_sha256': result['input_hashes'][str(path)],
                'vertices': len(vertices), 'faces': len(triangles), 'metrics': metrics,
                'distance_path': str(distance_path), 'distance_sha256': evaluator.sha256_file(distance_path),
                'elapsed_seconds_excluding_shared_tree': time.monotonic() - case_start,
                'process_peak_rss_bytes_after_case': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            })
            save()
            print(f'{name} forward diagnostic complete', flush=True)
            del vertices, triangles, samples, distances
        metrics = {item['case']: item['metrics'] for item in result['records']}
        result['differences'] = {f'{name}_minus_A':
            {key: metrics[name][key] - metrics['A'][key] for key in METRICS}
            for name in ('A-repeat', 'B')}
        for path, expected in list(result['input_hashes'].items()):
            bind(path, expected)
        check()
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.alarm(0)
        save()


if __name__ == '__main__':
    main()
