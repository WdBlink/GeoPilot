"""Fixed-mesh forward-sampling sensitivity; never an official benchmark score."""
import argparse
import importlib.util
import json
import platform
import resource
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).parent / 'evaluator/benchmark.py'
spec = importlib.util.spec_from_file_location('sampling_evaluator', SOURCE)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
METRICS = ('accuracy_l1_m', 'accuracy_rmse_m', 'accuracy_best90_l1_m',
           'accuracy_best90_rmse_m', 'precision_0_20')
SEEDS = tuple(range(20260925, 20260930))
INPUTS = {
    'P0': ('out/geopilot-rsih-p0-mu9geojt-r3/submission/mesh.ply',
           'out/geopilot-learning-20260921/evaluator-r3-score/score.json'),
    'P2': ('out/geopilot-learning-20260923/p2-run/submission/mesh.ply',
           'out/geopilot-learning-20260923/p2-score-r2/score.json'),
}


def summarize(distances):
    """Match the unchanged evaluator's five forward metrics and stable trim."""
    assert len(distances) > 0 and np.isfinite(distances).all()
    count = int(np.ceil(0.9 * len(distances)))
    best = distances[np.lexsort((np.arange(len(distances)), distances))[:count]]
    return dict(zip(METRICS, map(float, (
        distances.mean(), np.sqrt(np.square(distances).mean()), best.mean(),
        np.sqrt(np.square(best).mean()), (distances <= 0.20).mean()))))


def check():
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 12 * 1024**3:
        raise MemoryError('12 GiB diagnostic peak RSS protection')


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python is unsupported: validation assertions must remain active')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    result = {'status': 'running', 'formal_score': False, 'pid': __import__('os').getpid(),
              'started_at': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'python_executable': sys.executable,
              'records': [], 'input_hashes': {}}

    def save():
        result['elapsed_seconds'] = time.monotonic() - begin
        result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = args.output / 'result.json.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output / 'result.json')

    def alarm(*_):
        raise TimeoutError('30 minute diagnostic protection')

    def bind(path, expected=None):
        digest = evaluator.sha256_file(path)
        result['input_hashes'][str(path.resolve())] = digest
        if expected is not None and digest != expected:
            raise ValueError(f'Input identity mismatch: {path}')
        return digest

    try:
        if platform.system() != 'Darwin':
            raise RuntimeError('RSS units and environment are specified for Darwin only')
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(1800)
        result['versions'] = evaluator.runtime_versions()
        bind(Path(__file__))
        bind(args.plan)
        bind(SOURCE, '800c751822049c6a8814ab7d7d388fd38a5b96fa918df4ed97374c43393ea285')
        protocol_hash = bind(SOURCE.with_name('protocol_v1.json'))
        ref = ROOT / 'out/usegeo_benchmark/prepared/v1/evaluator-only/Dataset-1'
        reference_hash = bind(ref / 'reference_manifest.json')
        reference = json.loads((ref / 'reference_manifest.json').read_text())
        bind(ref / 'full_lidar.las', reference['files']['full_lidar']['sha256'])
        meshes, scores = {}, {}
        for name, (mesh, score_path) in INPUTS.items():
            bind(ROOT / score_path)
            score = json.loads((ROOT / score_path).read_text())
            assert score['status'] == 'valid' and score['scene_id'] == 'Dataset-1'
            assert score['bindings']['protocol_sha256'] == protocol_hash
            assert score['bindings']['reference_manifest_sha256'] == reference_hash
            bind(ROOT / mesh, score['bindings']['mesh_sha256'])
            meshes[name] = evaluator.read_ply(ROOT / mesh)
            scores[name] = score
        save()
        full = evaluator._las_points(ref / 'full_lidar.las', check)
        assert all(len(full) == s['counts']['full_lidar_points'] for s in scores.values())
        print('Building fixed full-LiDAR tree', flush=True)
        tree = evaluator.KDTree(full)
        check()
        for seed in (20260916,) + SEEDS:
            for name, (vertices, triangles) in meshes.items():
                samples = evaluator.sample_surface(vertices, triangles, 1000000, seed, check)
                distances = np.empty(len(samples), dtype=np.float64)
                for start in range(0, len(samples), 100000):
                    check()
                    distances[start:start + 100000] = tree.query(
                        samples[start:start + 100000], workers=1)[0]
                metrics = summarize(distances)
                np.save(args.output / f'{name}-{seed}-distances.npy', distances)
                record = {'program': name, 'seed': seed, 'surface_samples': len(samples),
                          'purpose': 'replay' if seed == 20260916 else 'sampling_repeat',
                          'metrics': metrics}
                if seed == 20260916:
                    record['max_absolute_replay_error'] = max(
                        abs(metrics[k] - scores[name]['surface_metrics'][k]) for k in METRICS)
                    result['records'].append(record)
                    save()
                    assert record['max_absolute_replay_error'] <= 1e-12, 'Replay mismatch'
                else:
                    result['records'].append(record)
                    save()
                print(f'{name} seed {seed} complete', flush=True)
        result['summary'] = {}
        arrays = {}
        for name in INPUTS:
            arrays[name] = np.array([[r['metrics'][k] for k in METRICS] for r in
                                     result['records'] if r['program'] == name and
                                     r['purpose'] == 'sampling_repeat'])
        arrays['P2_minus_P0_same_seed_not_spatially_paired'] = arrays['P2'] - arrays['P0']
        for name, array in arrays.items():
            assert array.shape == (5, 5)
            result['summary'][name] = {key: dict(zip(('mean', 'sample_sd', 'min', 'max'),
                map(float, (col.mean(), col.std(ddof=1), col.min(), col.max()))))
                for key, col in zip(METRICS, array.T)}
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.alarm(0)
        save()


if __name__ == '__main__':
    main()
