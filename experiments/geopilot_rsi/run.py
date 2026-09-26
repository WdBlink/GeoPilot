"""Execute a bounded reconstruction program without reference or score access."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'code/usegeo_mesh_baseline'))
import runner as sparse
import numpy as np
import pycolmap as pc

TOOLS = ROOT / 'out/geopilot_rsi_dependencies/openmvs-2.4.0'
THREADS = 8
OPTIONS = {
    'densify': {'resolution-level': (int, 0, 2), 'number-views': (int, 2, 20),
                'number-views-fuse': (int, 2, 5), 'iters': (int, 1, 6),
                'geometric-iters': (int, 1, 4),
                'fusion-depth-diff-threshold': (float, .001, .05)},
    'mesh': {'min-point-distance': (float, 0, 5), 'free-space-support': (int, 0, 1),
             'remove-spurious': (float, 0, 40), 'close-holes': (int, 0, 60),
             'smooth': (int, 0, 5), 'decimate': (float, .25, 1)},
    'refine': {'resolution-level': (int, 0, 2), 'max-face-area': (int, 16, 128)},
    'stop': {},
}
DIAGNOSTICS = {'registered_fraction', 'sparse_points', 'mean_reprojection_error',
               'dense_points', 'mesh_vertices', 'mesh_faces'}


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=sparse.unique_json,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate(program):
    """Validate data-only instructions; neither expressions nor paths are accepted."""
    if not isinstance(program, dict) or set(program) != {'schema', 'steps'}:
        raise ValueError('Invalid program fields')
    if program['schema'] != 'geopilot-reconstruction-program/1':
        raise ValueError('Unknown program schema')
    if not isinstance(program['steps'], list) or not program['steps']:
        raise ValueError('Empty program')
    for step in program['steps']:
        if not isinstance(step, dict) or not {'action', 'parameters'} <= set(step) <= {'action', 'parameters', 'when'}:
            raise ValueError('Invalid step')
        action, params = step['action'], step['parameters']
        if not isinstance(action, str) or action not in OPTIONS or not isinstance(params, dict):
            raise ValueError('Unknown action or parameters')
        for key, value in params.items():
            if key not in OPTIONS[action]:
                raise ValueError('Unregistered parameter')
            kind, low, high = OPTIONS[action][key]
            if type(value) not in ((int,) if kind is int else (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Parameter outside contract')
        if 'when' in step:
            condition = step['when']
            if not isinstance(condition, dict) or set(condition) != {'metric', 'op', 'value'}:
                raise ValueError('Invalid condition')
            if condition['metric'] not in DIAGNOSTICS or condition['op'] not in {'lt', 'le', 'gt', 'ge'}:
                raise ValueError('Unknown diagnostic or operator')
            if type(condition['value']) not in (int, float) or not math.isfinite(condition['value']):
                raise ValueError('Invalid threshold')
    if program['steps'][-1] != {'action': 'stop', 'parameters': {}}:
        raise ValueError('Program must end with an unconditional stop')


def applies(step, diagnostics):
    if 'when' not in step:
        return True
    c = step['when']
    value = diagnostics[c['metric']]  # Missing measurement is an error, not false.
    if not math.isfinite(value):
        raise ValueError('Nonfinite diagnostic')
    return {'lt': value < c['value'], 'le': value <= c['value'],
            'gt': value > c['value'], 'ge': value >= c['value']}[c['op']]


def read_obj(path):
    """Read native OpenMVS triangle OBJ, rejecting invalid or ambiguous geometry."""
    vertices, faces = [], []
    with Path(path).open() as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith('#'):
                continue
            if fields[0] == 'v':
                if len(fields) != 4:
                    raise ValueError('Unexpected vertex format')
                vertices.append([float(x) for x in fields[1:]])
            elif fields[0] == 'f':
                if len(fields) != 4:
                    raise ValueError('Nontriangle face')
                face = [int(x.split('/')[0]) for x in fields[1:]]
                if min(face) <= 0:
                    raise ValueError('Relative OBJ indices not supported')
                faces.append([x - 1 for x in face])
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Empty/malformed mesh')
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid mesh coordinates/indices')
    return vertices, faces


def event(out, value):
    with (out / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                 **value}, allow_nan=False) + '\n')


def archive_depths(out, index):
    """Retain depth evidence without letting a later parameter change reuse it."""
    archive = out / f'depth-history-{index:03d}'
    archive.mkdir()
    files = {}
    for path in sorted(out.glob('depth*.dmap')):
        sparse.regular(path, out)
        files[path.name] = sparse.sha(path)
        path.rename(archive / path.name)
    sparse.dump(archive / 'manifest.json', files)
    return {'directory': archive.name, 'manifest_sha256': sparse.sha(archive / 'manifest.json'),
            'depth_map_count': len(files)}


def invoke(out, command, parent):
    started = time.monotonic()
    event(out, {'event': 'tool_started', 'argv': command, 'parent': parent})
    result = subprocess.run(command, cwd=out, check=False)
    event(out, {'event': 'tool_finished', 'argv': command, 'returncode': result.returncode,
                'wallclock_s': time.monotonic() - started, 'parent': parent})
    if result.returncode:
        raise RuntimeError(f'Tool failed: {Path(command[0]).name}, exit {result.returncode}')


def check_inputs(inputs):
    manifest = read_json(sparse.regular(inputs / 'input_manifest.json', inputs))
    if manifest['track'] != 'rgb-oriented' or manifest['scene_id'] not in {'Dataset-1', 'Dataset-2', 'Dataset-3'}:
        raise ValueError('Unregistered input track/scene')
    images = {}
    for entry in manifest['images']:
        name = entry['image_id']
        if Path(name).name != name or name in images:
            raise ValueError('Invalid/duplicate image identity')
        path = sparse.regular(inputs / 'images' / name, inputs)
        images[name] = sparse.sha(path)
        if images[name] != entry['sha256']:
            raise ValueError('Image hash mismatch')
    if len(images) != manifest['image_count']:
        raise ValueError('Image count mismatch')
    orientation = inputs / f"Image_orientations_dataset{manifest['scene_id'][-1]}.xyz"
    sparse.regular(orientation, inputs)
    if sparse.sha(orientation) != manifest['orientation_sha256']:
        raise ValueError('Orientation hash mismatch')
    centers = {}
    for line in orientation.read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            values = line.split()
            if len(values) < 4 or values[0] in centers:
                raise ValueError('Malformed/duplicate orientation')
            centers[values[0]] = [float(x) for x in values[1:4]]
    if any(name not in centers or not np.isfinite(centers[name]).all() for name in images):
        raise ValueError('Missing/nonfinite centers')
    return manifest, images, centers


def worker(inputs, out, program_path, cached):
    program = read_json(program_path)
    validate(program)
    sparse.dump(out / 'program.json', program)
    manifest, images, centers = check_inputs(inputs)
    if cached:
        # Cache is a method-only staged copy; no old mesh, scores or reference paths.
        cache = read_json(cached / 'cache.json')
        if cache['images'] != images or cache['input_manifest_sha256'] != sparse.sha(inputs / 'input_manifest.json'):
            raise ValueError('Sparse cache input mismatch')
        if cache['sparse_source_sha256'] != sparse.sha(sparse.__file__):
            raise ValueError('Sparse cache source mismatch')
        for name, expected in cache['model_files'].items():
            if Path(name).name != name or sparse.sha(sparse.regular(cached / 'model' / name, cached)) != expected:
                raise ValueError('Sparse model changed')
        model_path = cached / 'model'
        sparse.dump(out / 'cache-reuse.json', cache)
    else:
        sfm = out / 'sfm'
        sfm.mkdir()
        sparse.worker(inputs, sfm)
        result = read_json(sfm / 'method-result.json')
        model_path = sfm / 'models' / str(result['model_id'])
    model = pc.Reconstruction(str(model_path))
    registered = sorted(model.reg_image_ids())
    if any(model.images[i].name not in images for i in registered):
        raise ValueError('Unexpected reconstructed image')
    transform, residuals = sparse.fit_centers(
        [model.images[i].projection_center() for i in registered],
        [centers[model.images[i].name] for i in registered])
    diagnostics = {'registered_fraction': len(registered) / len(images),
                   'sparse_points': len(model.points3D),
                   'mean_reprojection_error': model.compute_mean_reprojection_error()}
    sparse.dump(out / 'input-lineage.json', {'manifest_sha256': sparse.sha(inputs / 'input_manifest.json'),
                'images': images, 'registered_images': [model.images[i].name for i in registered],
                'diagnostics': diagnostics, 'sim3': transform.matrix().tolist(),
                'center_residuals_m': residuals.tolist()})
    undistorted = out / 'undistorted'
    start = time.monotonic()
    event(out, {'event': 'undistort_started'})
    pc.undistort_images(str(undistorted), str(model_path), str(inputs / 'images'),
                       undistort_options=pc.UndistortCameraOptions(max_image_size=3200))
    event(out, {'event': 'undistort_finished', 'wallclock_s': time.monotonic() - start})
    scene = out / 'scene.mvs'
    invoke(out, [str(TOOLS / 'InterfaceCOLMAP'), '-i', str(undistorted), '-o', str(scene),
                 '--image-folder', str(undistorted / 'images'), '--archive-type', '2',
                 '--max-threads', str(THREADS)], 'sparse')
    dense = None
    mesh = None
    for index, step in enumerate(program['steps']):
        before = dict(diagnostics)
        active = applies(step, diagnostics)
        event(out, {'event': 'decision', 'step': index, 'instruction': step,
                    'diagnostics': before, 'applies': active})
        if not active:
            continue
        action = step['action']
        if action == 'stop':
            break
        if (action == 'mesh' and dense is None) or (action == 'refine' and mesh is None):
            raise ValueError('Stage precondition failed')
        parent = scene if action == 'densify' else dense if action == 'mesh' else mesh
        result = out / f'{index:03d}_{action}.mvs'
        tool = {'densify': 'DensifyPointCloud', 'mesh': 'ReconstructMesh', 'refine': 'RefineMesh'}[action]
        command = [str(TOOLS / tool), '-i', str(parent), '-o', str(result),
                   '--max-threads', str(THREADS), '--archive-type', '2']
        if action == 'densify':
            command += ['--estimate-roi', '0', '--crop-to-roi', '0', '--tower-mode', '0']
        else:
            command += ['--export-type', 'obj']
            if action == 'mesh':
                command += ['--crop-to-roi', '0']
        for key, value in sorted(step['parameters'].items()):
            command += ['--' + key, str(value)]
        invoke(out, command, sparse.sha(parent))
        if action == 'densify':
            dense, mesh = result, None
            diagnostics.pop('mesh_vertices', None)
            diagnostics.pop('mesh_faces', None)
            with result.with_suffix('.ply').open('rb') as stream:
                for _ in range(100):
                    line = stream.readline().decode('ascii').strip()
                    if line.startswith('element vertex '):
                        diagnostics['dense_points'] = int(line.split()[-1])
                        break
                else:
                    raise ValueError('Missing dense point count')
            if diagnostics['dense_points'] <= 0:
                raise ValueError('Empty dense reconstruction')
            event(out, {'event': 'depth_maps_archived', 'step': index,
                        **archive_depths(out, index)})
        else:
            mesh = result
            vertices, faces = read_obj(result.with_suffix('.obj'))
            diagnostics.update(mesh_vertices=len(vertices), mesh_faces=len(faces))
        event(out, {'event': 'artifact', 'step': index, 'path': result.name,
                    'sha256': sparse.sha(result), 'diagnostics': diagnostics})
    if mesh is None:
        raise ValueError('Program did not produce a final dense-derived mesh')
    vertices, faces = read_obj(mesh.with_suffix('.obj'))
    vertices = transform * vertices
    areas = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                    vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1) / 2
    if not np.isfinite(areas).all() or np.any(areas <= 1e-12):
        raise ValueError('Invalid/degenerate final mesh; no silent export repair')
    sparse.write_mesh(out / 'mesh.ply', vertices, faces)
    identity = sparse.sha(program_path)
    sparse.dump(out / 'submission.json', {'schema_version': 'usegeo-mesh-submission-1.0',
        'protocol_version': 'usegeo-mesh-rgb-oriented-v1', 'track': 'rgb-oriented',
        'scene_id': manifest['scene_id'], 'product': 'mesh', 'geometry': 'mesh.ply',
        'configuration_id': 'geopilot-rsi-' + identity[:16],
        'method_metadata': {'name': 'COLMAP-OpenMVS conditional program', 'program_sha256': identity}})
    sparse.dump(out / 'sealed.json', {'program_sha256': identity, 'diagnostics': diagnostics,
        'mesh_sha256': sparse.sha(out / 'mesh.ply'),
        'submission_sha256': sparse.sha(out / 'submission.json'), 'online_tokens': 0,
        'scope': 'development, no unseen-scene or learning claim'})


def profile(inputs, out, program, cached):
    runtime = Path(sys.prefix).resolve()
    interpreter = Path(sys.base_prefix).resolve()
    read_roots = ['/System/Library', '/usr/lib', str(runtime), str(interpreter),
                  str(HERE), str(Path(sparse.__file__).parent), str(TOOLS), str(inputs)]
    if cached:
        read_roots.append(str(cached))
    return '\n'.join(['(version 1)', '(deny default)',
        '(import "/System/Library/Sandbox/Profiles/dyld-support.sb")',
        '(allow process-exec process-fork sysctl-read process-info* signal)',
        '(allow file-read-metadata)',
        '(allow file-read* file-map-executable ' + ' '.join('(subpath ' + json.dumps(p) + ')' for p in read_roots) + ')',
        '(allow file-read* (literal ' + json.dumps(str(program)) + ') (literal "/dev/random") (literal "/dev/urandom"))',
        '(allow file-read* file-write* file-lock (literal "/dev/null") (subpath ' + json.dumps(str(out)) + '))'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--program', type=Path, required=True)
    parser.add_argument('--sfm-cache', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    inputs, out, program = args.inputs.resolve(), args.output.resolve(), args.program.resolve()
    cached = args.sfm_cache.resolve() if args.sfm_cache else None
    validate(read_json(program))
    if args.worker:
        worker(inputs, out, program, cached)
        return
    if any(out.is_relative_to(p) or p.is_relative_to(out) for p in [inputs, HERE, TOOLS] + ([cached] if cached else [])):
        raise ValueError('Output overlaps protected roots')
    out.mkdir(parents=True, exist_ok=False)
    sandbox = out / 'worker.sb'
    sandbox.write_text(profile(inputs, out, program, cached))
    protected = {str(p): sparse.sha(p) for p in [program, Path(__file__), Path(sparse.__file__)]}
    protected.update({str(p): sparse.sha(p) for p in TOOLS.iterdir() if p.is_file()})
    sparse.dump(out / 'launch.json', {'bindings': protected, 'inputs': str(inputs), 'pid': os.getpid(),
                'threads': THREADS, 'memory_protection_bytes': 48 * 1024**3,
                'research_time_cap': None, 'python': sys.version, 'pycolmap': pc.__version__,
                'sandbox_sha256': sparse.sha(sandbox)})
    probe = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(sandbox), sys.executable, '-B', '-c',
        'from pathlib import Path; Path(' + repr(str(ROOT / 'code/usegeo_mesh_benchmark/protocol_v1.json')) + ').read_bytes()'],
        capture_output=True, text=True)
    if probe.returncode == 0 or 'Operation not permitted' not in probe.stderr:
        raise RuntimeError('Reference-side read isolation probe failed')
    command = ['/usr/bin/sandbox-exec', '-f', str(sandbox), sys.executable, '-B', str(Path(__file__).resolve()),
               '--inputs', str(inputs), '--output', str(out), '--program', str(program), '--worker']
    if cached:
        command += ['--sfm-cache', str(cached)]
    os.environ['TMPDIR'] = str(out)
    metrics = sparse.supervise(command, out, timeout=float('inf'), max_rss=48 * 1024**3)
    unchanged = all(sparse.sha(path) == expected for path, expected in protected.items())
    success = metrics['returncode'] == 0 and metrics['reason'] is None and unchanged and (out / 'sealed.json').is_file()
    sparse.dump(out / 'run.json', {'status': 'sealed' if success else 'failed',
                                 'frozen_bindings_unchanged': unchanged, **metrics})
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
