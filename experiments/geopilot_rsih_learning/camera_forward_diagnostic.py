"""Fixed five-seed forward screen for completed manual camera-to-mesh A/B controls."""
import argparse
import json
import platform
import resource
import signal
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
from sampling_diagnostic import METRICS, SEEDS, SOURCE, check, evaluator, np, summarize

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'out/geopilot-research-20260925'
CAMPAIGN, OUTPUT = BASE / 'camera-mesh-r2-run', BASE / 'camera-forward-run'
PLAN = BASE / 'camera-forward-plan.md'
CLEAN = ROOT / 'out/usegeo_mesh_benchmark/paper-readiness-20260918/clean-runtime'
PARAMETERS = {'undistort_max_image_size': 2400, 'densify': {'resolution-level': 1}, 'mesh': {'decimate': .5}}


def aggregate(records):
    """Resource triage only; these samples are neither spatial pairs nor reconstructions."""
    if len(records) != 10 or {(r['case'], r['seed']) for r in records} != {(c, s) for c in ('A', 'B') for s in SEEDS}:
        raise ValueError('Exactly five fixed draws from each complete arm are required')
    arrays = {c: np.array([[next(r for r in records if r['case'] == c and r['seed'] == s)['metrics'][k]
                           for k in METRICS] for s in SEEDS]) for c in ('A', 'B')}
    arrays['B_minus_A_same_seed_not_spatial_pairs'] = delta = arrays['B'] - arrays['A']
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite metrics')
    summary = {name: {k: dict(zip(('mean', 'sample_sd', 'min', 'max'), map(float,
                (v.mean(), v.std(ddof=1), v.min(), v.max())))) for k, v in zip(METRICS, a.T)}
               for name, a in arrays.items()}
    return summary, bool(np.all(delta[:, 0] < 0) and np.all(delta.mean(axis=0)[1:4] <= 0)
                         and delta.mean(axis=0)[4] >= 0)


def ready(bind):
    campaign_hash = bind(CAMPAIGN / 'inputs.json')
    manifest = json.loads((CAMPAIGN / 'inputs.json').read_text())
    if manifest['schema'] != 'manual-camera-mesh/1' or manifest['parameters'] != PARAMETERS or set(manifest['source_models']) != {'A', 'B'}:
        raise ValueError('Unexpected reconstruction campaign')
    for path, digest in manifest['bindings'].items(): bind(Path(path), digest, allow_symlink=True)
    for path in (ROOT / 'experiments/geopilot_rsih_learning/camera_mesh_diagnostic.py',
                 ROOT / 'experiments/geopilot_rsih_learning/camera_mesh_diagnostic_r2.py',
                 BASE / 'camera-mesh-plan.md', BASE / 'camera-mesh-r2-plan.md'):
        bind(path, manifest['bindings'][str(path)])
    meshes = {}
    for name in ('A', 'B'):
        case = CAMPAIGN / name
        result = json.loads((case / 'result.json').read_text())
        bind(case / 'result.json')
        required = ('command.json', 'supervision.json', 'camera-fit.json', 'prepare/state.json',
                    'densify/request.json', 'densify/state.json', 'mesh/request.json', 'mesh/state.json', 'mesh/mesh.ply')
        if not {str(case / p) for p in required}.issubset(result['output_hashes']):
            raise ValueError('Critical output hash missing')
        for p, digest in result['output_hashes'].items():
            bind(Path(p), digest)
        status = json.loads((case / 'supervision.json').read_text())
        state = json.loads((case / 'mesh/state.json').read_text())
        fit = json.loads((case / 'camera-fit.json').read_text())
        command = json.loads((case / 'command.json').read_text())
        if command['inputs_sha256'] != campaign_hash:
            raise ValueError('Executed arm belongs to another campaign')
        source = manifest['source_models'][name]
        expected_copies = {str(case / 'input-model' / Path(p).name): digest for p, digest in source['model_hashes'].items()}
        if result['source_BA_result'] != source['BA_result'] or result['copied_input_hashes'] != expected_copies:
            raise ValueError('Sparse source or copied-model identity mismatch')
        for p, digest in expected_copies.items(): bind(Path(p), digest)
        for action, parent_stage in (('densify', 'prepare'), ('mesh', 'densify')):
            request = json.loads((case / action / 'request.json').read_text())
            parent = json.loads((case / parent_stage / 'state.json').read_text())
            if (request['node']['action'] != action or request['node']['parameters'] != PARAMETERS[action]
                    or request['parent'] != parent or parent['transform'] != fit['fit_centers_transform']):
                raise ValueError('Actual parameter or transform chain differs from the plan')
        if (result['status'] != 'completed' or result['branch'] != name
                or status['returncode'] != 0 or status['reason'] is not None
                or state['stage'] != 'mesh' or state['mesh'] != result['mesh']
                or state['transform'] != fit['fit_centers_transform']):
            raise ValueError('Incomplete or inconsistent arm: ' + name)
        mesh = case / 'mesh/mesh.ply'
        if state['mesh']['path'] != str(mesh):
            raise ValueError('Unexpected mesh path')
        bind(mesh, state['mesh']['sha256'])
        meshes[name] = mesh
    return meshes


def self_check():
    def rows(difference):
        return [dict(case=c, seed=s, metrics={k: 2.0 + (difference[j] if c == 'B' else 0)
                 for j, k in enumerate(METRICS)}) for c in ('A', 'B') for s in SEEDS]
    assert aggregate(rows([-.1, -.1, -.1, -.1, .1]))[1]
    assert not aggregate(rows([0, -.1, -.1, -.1, .1]))[1]
    assert not aggregate(rows([-.1, .1, -.1, -.1, .1]))[1]
    assert not aggregate(rows([-.1, -.1, -.1, -.1, -.1]))[1]
    bad = rows([-.1] * 5); bad[-1]['seed'] = SEEDS[0]
    try:
        aggregate(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate draw accepted')
    print('Ten-draw identity and resource-triage direction checks passed')


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python is unsupported')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', 'self-check'))
    action = parser.parse_args().action
    if action == 'self-check':
        self_check(); return
    if platform.system() != 'Darwin' or Path(sys.prefix).resolve() != CLEAN.resolve():
        raise RuntimeError('Use the fixed Darwin clean-runtime')
    bindings, resolved_paths = {}, {}
    def bind(path, expected=None, *, allow_symlink=False):
        path = Path(path)
        resolved = str(path.resolve())
        if not path.is_file() or (not allow_symlink and path.resolve() != path.absolute()):
            raise ValueError('Input missing or unexpectedly linked: ' + str(path))
        if str(path) in resolved_paths and resolved_paths[str(path)] != resolved:
            raise ValueError('Bound path target changed: ' + str(path))
        resolved_paths[str(path)] = resolved
        digest = evaluator.sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError('Bound input changed: ' + str(path))
        bindings[str(path)] = digest
        return digest
    versions = evaluator.runtime_versions()
    bind(SOURCE.resolve(), '800c751822049c6a8814ab7d7d388fd38a5b96fa918df4ed97374c43393ea285')
    bind(SOURCE.with_name('protocol_v1.json').resolve(), 'a16e8a9dabd4a2a2188a391ea2205ca941717639ab95a626d16b10c06dfab622')
    bind(Path(__file__).with_name('sampling_diagnostic.py').resolve(), '3a462aa2dc500bbf817212a61bdcf959cceb39af0b544c09e893a425e2b6c119')
    bind(Path(__file__).resolve()); bind(PLAN)
    meshes = ready(bind)
    ref = ROOT / 'out/usegeo_benchmark/prepared/v1/evaluator-only/Dataset-1'
    bind(ref / 'reference_manifest.json', '56dcb79402d4c3001b7cbc6d34ce5ef7027bc699e5a1c41e4c4aa35b8350f795')
    bind(ref / 'full_lidar.las', '2552c67d19d433b6cbe65af6238e2bcca6825d4711791982432d843a0940c998')
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if action == 'check':
        print(json.dumps({'status': 'bound_without_scoring', 'files': len(bindings)})); return
    OUTPUT.mkdir()
    started = time.monotonic()
    result = dict(status='running', formal_score=False, completeness_evaluated=False, input_hashes=bindings, input_resolved_paths=resolved_paths,
                  versions=versions, platform=platform.platform(), samples_per_draw=1000000,
                  chunk=100000, workers=1, seeds=SEEDS, records=[])
    def save():
        result.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temp = OUTPUT / 'result.json.tmp'; temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n'); temp.replace(OUTPUT / 'result.json')
    def alarm(*_):
        raise TimeoutError('30 minute forward diagnostic protection')
    try:
        signal.signal(signal.SIGALRM, alarm); signal.alarm(1800)
        full = evaluator._las_points(ref / 'full_lidar.las', check)
        if len(full) != 105922149: raise ValueError('Reference count changed')
        tree = evaluator.KDTree(full); check()
        for name, path in meshes.items():
            vertices, triangles = evaluator.read_ply(path)
            for seed in SEEDS:
                samples = evaluator.sample_surface(vertices, triangles, 1000000, seed, check)
                distances = np.empty(1000000, dtype=np.float64)
                for start in range(0, 1000000, 100000):
                    check(); distances[start:start+100000] = tree.query(samples[start:start+100000], workers=1)[0]
                array = OUTPUT / f'{name}-{seed}-distances.npy'; np.save(array, distances)
                result['records'].append(dict(case=name, seed=seed, metrics=summarize(distances),
                    mesh_sha256=bindings[str(path)], distance_path=str(array), distance_sha256=evaluator.sha256_file(array)))
                save(); print(f'{name} seed {seed} complete', flush=True)
            del vertices, triangles, samples, distances
        result['summary'], result['proceed_to_full_evaluation_by_resource_rule'] = aggregate(result['records'])
        for p, digest in list(bindings.items()): bind(Path(p), digest, allow_symlink=True)
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        signal.alarm(0); save()


if __name__ == '__main__':
    main()
