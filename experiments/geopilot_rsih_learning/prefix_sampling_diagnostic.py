"""Exploratory A sampling follow-up; reuse byte-identical B/P2's five fixed draws."""
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
import prefix_forward_diagnostic as fixed
from sampling_diagnostic import INPUTS, METRICS, SEEDS, SOURCE, check, evaluator, np, summarize

ROOT = fixed.ROOT
BASE = ROOT / 'out/geopilot-research-20260925'
OUTPUT = BASE / 'prefix-sampling-run'
PLAN = BASE / 'prefix-sampling-plan.md'
EVIDENCE = BASE / 'sampling-evidence.json'
PRIOR = BASE / 'sampling-run/result.json'
EVIDENCE_SHA = '9709922fb99d5e993a2dbafa44f22ae4a3a752e70fc9323de46c5ee53998adc4'
FORWARD_SOURCE_SHA = '2c4aec5f15dea69012db6a494e6cb42c9cb40cc3e42d326357d43f971b91b5ba'
OBSERVED_RESULT_SHA = '7ee7f2b1516ca5b512f3b1e7c791623aa8e7935ad3d7ae1b9e288e3b0133b055'
SAMPLES, CHUNK = 1000000, 100000


def reused_metrics(path, expected, record, bind):
    """Verify the original array bytes and reproduce every stored metric."""
    bind(path, expected)
    distances = np.load(path, allow_pickle=False)
    if distances.shape != (SAMPLES,) or distances.dtype != np.dtype('float64'):
        raise ValueError('Unexpected reused distance array')
    metrics = summarize(distances)
    if (record['surface_samples'] != SAMPLES or set(record['metrics']) != set(METRICS)
            or max(abs(metrics[k] - record['metrics'][k]) for k in METRICS) > 1e-12):
        raise ValueError('Reused metrics do not match the bound array')
    return metrics


def reuse_ready(bind, meshes, versions):
    """Require all five existing P2 draws, the same reference and identical B bytes."""
    bind(EVIDENCE, EVIDENCE_SHA)
    evidence = evaluator._strict_json(EVIDENCE, 'INVALID_SAMPLING_EVIDENCE')
    digests = {entry['path']: entry['sha256'] for entry in evidence['files']}
    if len(digests) != len(evidence['files']):
        raise ValueError('Duplicate evidence identities')
    bind(PRIOR, digests[str(PRIOR)])
    prior = evaluator._strict_json(PRIOR, 'INVALID_SAMPLING_RESULT')
    if prior['status'] != 'complete' or prior['versions'] != versions:
        raise ValueError('Prior sampling incomplete or runtime changed')
    for path, expected in (
        (Path(__file__).with_name('sampling_diagnostic.py'), fixed.SAMPLING_SHA),
        (SOURCE, fixed.EVALUATOR_SHA), (SOURCE.with_name('protocol_v1.json'), fixed.PROTOCOL_SHA),
    ):
        if prior['input_hashes'][str(path)] != expected:
            raise ValueError('Prior evaluator/protocol identity mismatch')
        bind(path, expected)
    ref = ROOT / 'out/usegeo_benchmark/prepared/v1/evaluator-only/Dataset-1'
    manifest = ref / 'reference_manifest.json'
    if prior['input_hashes'][str(manifest)] != fixed.REFERENCE_SHA:
        raise ValueError('Prior reference manifest mismatch')
    bind(manifest, fixed.REFERENCE_SHA)
    reference = evaluator._strict_json(manifest, 'INVALID_REFERENCE')
    full = ref / 'full_lidar.las'
    if prior['input_hashes'][str(full)] != reference['files']['full_lidar']['sha256']:
        raise ValueError('Prior full LiDAR mismatch')
    bind(full, prior['input_hashes'][str(full)])
    p2 = ROOT / INPUTS['P2'][0]
    bind(p2, prior['input_hashes'][str(p2)])
    bind(meshes['B'], prior['input_hashes'][str(p2)])
    records = [r for r in prior['records'] if r['program'] == 'P2' and r['purpose'] == 'sampling_repeat']
    if len(records) != 5 or sorted(r['seed'] for r in records) != list(SEEDS):
        raise ValueError('Must reuse exactly the five pre-existing P2 seeds')
    records = {r['seed']: r for r in records}
    reused = {}
    for seed in SEEDS:
        path = PRIOR.parent / f'P2-{seed}-distances.npy'
        reused[seed] = {'metrics': reused_metrics(path, digests[str(path)], records[seed], bind),
                        'distance_path': str(path), 'distance_sha256': digests[str(path)]}
    return full, reused


def self_check():
    """Small array check for byte integrity and all-metric replay; no geometry."""
    import tempfile
    from unittest.mock import patch

    if not __debug__:
        raise RuntimeError('Self-check requires assertions')
    with tempfile.TemporaryDirectory() as temporary, patch(__name__ + '.SAMPLES', 10):
        path = Path(temporary) / 'distances.npy'
        distances = np.linspace(0, 1, SAMPLES)
        np.save(path, distances)
        expected = evaluator.sha256_file(path)
        record = {'surface_samples': SAMPLES, 'metrics': summarize(distances)}

        def bind(path, digest):
            if evaluator.sha256_file(path) != digest:
                raise ValueError('Changed raw array')

        assert reused_metrics(path, expected, record, bind) == record['metrics']
        for change in ('bytes', 'metric'):
            np.save(path, distances + (1 if change == 'bytes' else 0))
            altered = json.loads(json.dumps(record))
            if change == 'metric':
                altered['metrics'][METRICS[-1]] += 1
            try:
                reused_metrics(path, expected, altered, bind)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Invalid reused {change} accepted')
    print('Reuse check passed: original array accepted; changed bytes or metrics rejected.')


def main():
    """Compute only the five A draws; reuse every existing B/P2 draw unchanged."""
    if not __debug__:
        raise RuntimeError('Optimized Python is unsupported: validation assertions must remain active')
    if len(sys.argv) != 1:
        raise ValueError('This fixed diagnostic accepts no arguments')
    if platform.system() != 'Darwin' or Path(sys.prefix).resolve() != fixed.CLEAN.resolve():
        raise RuntimeError('Use the fixed Darwin clean-runtime Python')
    if OUTPUT.resolve() != OUTPUT:
        raise ValueError('Output path must not contain symlinks')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = {'status': 'running', 'exploratory_after_observed_results': True,
              'formal_score': False, 'completeness_evaluated': False, 'pid': os.getpid(),
              'started_at': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
              'python_executable': sys.executable, 'seeds': SEEDS, 'surface_samples': SAMPLES,
              'query_chunk': CHUNK, 'query_workers': 1, 'records': [], 'input_hashes': {}}

    def bind(path, expected=None):
        path = Path(path)
        if path.resolve() != path.absolute() or not path.is_file():
            raise ValueError(f'Missing or linked input: {path}')
        digest = evaluator.sha256_file(path)
        result['input_hashes'][str(path)] = digest
        if expected is not None and digest != expected:
            raise ValueError(f'Input identity mismatch: {path}')
        return digest

    def save():
        result['elapsed_seconds'] = time.monotonic() - start
        result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = OUTPUT / 'result.json.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(OUTPUT / 'result.json')

    def alarm(*_):
        raise TimeoutError('30 minute diagnostic protection')

    try:
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(1800)
        result['versions'] = evaluator.runtime_versions()
        bind(Path(__file__).resolve())
        bind(PLAN)
        bind(Path(fixed.__file__).resolve(), FORWARD_SOURCE_SHA)
        bind(BASE / 'prefix-forward-run/result.json', OBSERVED_RESULT_SHA)
        meshes = fixed.cases_ready(bind)
        full_path, reused = reuse_ready(bind, meshes, result['versions'])
        result['reused_B_mesh_sha256'] = result['input_hashes'][str(meshes['B'])]
        save()
        full = evaluator._las_points(full_path, check)
        assert len(full) == 105922149, 'Full LiDAR point count changed'
        result['full_lidar_points'] = len(full)
        print('Building one fixed full-LiDAR tree for A', flush=True)
        tree = evaluator.KDTree(full)
        check()
        vertices, triangles = evaluator.read_ply(meshes['A'])
        result['A_mesh'] = {'sha256': result['input_hashes'][str(meshes['A'])],
                            'vertices': len(vertices), 'faces': len(triangles)}
        check()
        for seed in SEEDS:
            seed_start = time.monotonic()
            samples = evaluator.sample_surface(vertices, triangles, SAMPLES, seed, check)
            distances = np.empty(SAMPLES, dtype=np.float64)
            for offset in range(0, SAMPLES, CHUNK):
                check()
                distances[offset:offset + CHUNK] = tree.query(samples[offset:offset + CHUNK], workers=1)[0]
            a = summarize(distances)
            path = OUTPUT / f'A-{seed}-distances.npy'
            np.save(path, distances)
            result['records'].append({'seed': seed, 'A_metrics': a, 'B_reused': reused[seed],
                'A_distance_path': str(path), 'A_distance_sha256': evaluator.sha256_file(path),
                'B_minus_A_same_seed_not_spatially_paired': {k: reused[seed]['metrics'][k] - a[k] for k in METRICS},
                'elapsed_seconds_excluding_shared_tree': time.monotonic() - seed_start})
            save()
            print(f'A seed {seed} complete; B/P2 distances reused', flush=True)
            del samples, distances
        arrays = {'A': np.array([[r['A_metrics'][k] for k in METRICS] for r in result['records']]),
                  'B_reused': np.array([[r['B_reused']['metrics'][k] for k in METRICS] for r in result['records']])}
        arrays['B_minus_A_same_seed_not_spatially_paired'] = arrays['B_reused'] - arrays['A']
        result['summary'] = {}
        for name, array in arrays.items():
            assert array.shape == (5, 5)
            result['summary'][name] = {key: dict(zip(('mean', 'sample_sd', 'min', 'max'),
                map(float, (col.mean(), col.std(ddof=1), col.min(), col.max()))))
                for key, col in zip(METRICS, array.T)}
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
