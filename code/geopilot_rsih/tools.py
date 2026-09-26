"""Fixed numerical actions; invoked only by the bound GeoPilot runtime extension."""
import importlib.util
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from strict_json import read_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_legacy_adapter():
    """Load the approved adapter and its sparse dependency from exact files."""
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load {path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load('runner', ROOT / 'code/usegeo_mesh_baseline/runner.py')
    return load('geopilot_rsi_run', ROOT / 'experiments/geopilot_rsi/run.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def diagnostic(value, artifact, definition):
    return {'value': value, 'status': 'valid', 'definition_id': definition,
            'artifact_id': artifact, 'region': 'scene', 'reason': None,
            'cost': {'included_in': 'numerical_call', 'node_id': Path(artifact).parent.name
                     if Path(artifact).suffix else Path(artifact).name}}


def invoke(argv, output, workdir=None):
    with (output / 'numerical.log').open('a') as log:
        subprocess.run(argv, cwd=workdir or output, stdout=log, stderr=subprocess.STDOUT, check=True)


def depth_image_id(path):
    """Read OpenMVS .dmap's image name without loading its depth payload."""
    with Path(path).open('rb') as stream:
        head = stream.read(30)
        if len(head) != 30 or head[:2] != b'DR':
            raise ValueError('Invalid OpenMVS depth header')
        length = int.from_bytes(head[28:30], 'little')
        if not 1 <= length <= 255:
            raise ValueError('Invalid OpenMVS depth image length')
        name = stream.read(length).decode('utf-8')
    prefix = 'undistorted/images/'
    if not name.startswith(prefix) or Path(name).name != name[len(prefix):]:
        raise ValueError('Invalid OpenMVS depth image path')
    return name[len(prefix):]


def real_action(request):
    # Only numerical functions are reused; the old program validator/executor is not called.
    legacy = load_legacy_adapter()
    import numpy as np
    import pycolmap as pc
    node, parent, out = request['node'], request['parent'], Path(request['output'])
    action = node['action']
    state = dict(parent)
    state['diagnostics'] = dict(parent['diagnostics'])
    if action == 'prepare':
        inputs = Path(request['inputs'])
        manifest, images, centers = legacy.check_inputs(inputs)
        sfm = out / 'sfm'
        sfm.mkdir()
        legacy.sparse.worker(inputs, sfm)
        method = read_json(sfm / 'method-result.json')
        model_id = method['model_id']
        model_path = sfm / 'models' / str(model_id)
        model = pc.Reconstruction(str(model_path))
        registered = sorted(model.reg_image_ids())
        if not registered or any(model.images[i].name not in images for i in registered):
            raise ValueError('Invalid registered images')
        names = sorted(images)
        registered_names = {model.images[i].name for i in registered}
        sfm_inputs = read_json(sfm / 'inputs.json')
        sfm_pairs = read_json(sfm / 'pairs.json')
        pair_names = {int(i): name for i, name in sfm_pairs['image_names'].items()}
        if (method['input_count'] != len(names) or set(method['registered']) != registered_names
                or sfm_inputs['images'] != images or set(pair_names.values()) != set(names)):
            raise ValueError('Sparse stage did not ingest the frozen full input')
        matched_ids = {int(i) for a, b in sfm_pairs['union'] for i in (a, b)}
        matching_names = {pair_names[i] for i in matched_ids}
        transform, residuals = legacy.sparse.fit_centers(
            [model.images[i].projection_center() for i in registered],
            [centers[model.images[i].name] for i in registered])
        undistorted = out / 'undistorted'
        pc.undistort_images(str(undistorted), str(model_path), str(inputs / 'images'),
                           undistort_options=pc.UndistortCameraOptions(max_image_size=2400))
        undistorted_files = {p.name: p for p in (undistorted / 'images').iterdir()
                             if p.is_file() and not p.is_symlink()}
        if not undistorted_files or any(name not in images for name in undistorted_files):
            raise ValueError('Invalid undistorted image set')
        coverage = {
            'policy': 'fixed-p0-full-input-v1', 'input_manifest_sha256': sha(inputs / 'input_manifest.json'),
            'input_count': len(names), 'sfm_input_count': len(sfm_inputs['images']),
            'matching_candidate_count': len(matching_names),
            'registered_count': len(registered_names), 'undistorted_count': len(undistorted_files),
            'depth_image_count': 0, 'depth_map_count': 0,
            'images': {name: {'input_sha256': images[name], 'sfm_ingested': True,
                              'matching_candidate': name in matching_names,
                              'registered': name in registered_names,
                              'undistorted_sha256': sha(undistorted_files[name])
                              if name in undistorted_files else None,
                              'depth_maps': [],
                              'omissions': (["not_selected_for_matching"] if name not in matching_names else []) +
                              (["not_registered_by_sfm"] if name not in registered_names else []) +
                              (["not_undistorted"] if name not in undistorted_files else [])}
                       for name in names}}
        coverage_path = out / 'coverage-prepare.json'
        coverage_path.write_text(json.dumps(coverage, sort_keys=True))
        scene = out / 'scene.mvs'
        invoke([str(legacy.TOOLS / 'InterfaceCOLMAP'), '-i', str(undistorted), '-o', str(scene),
                '--image-folder', str(undistorted / 'images'), '--archive-type', '2',
                '--max-threads', str(legacy.THREADS)], out)
        state.update(stage='sparse', scene=str(scene), transform=transform.matrix().tolist(),
                     input_manifest_sha256=sha(inputs / 'input_manifest.json'),
                     input_image_count=len(images), registered_image_count=len(registered),
                     undistorted_image_count=len(undistorted_files),
                     scene_id=manifest['scene_id'], track=manifest['track'], mesh=None,
                     mvs_workspace=str(out), coverage_prepare_path=str(coverage_path),
                     coverage_path=str(coverage_path),
                     undistorted_files=[str(undistorted_files[name]) for name in sorted(undistorted_files)],
                     sfm_evidence_files=[str(sfm / name) for name in
                                         ('inputs.json', 'pairs.json', 'options.json', 'method-result.json')])
        values = {'registered_fraction': len(registered) / len(images),
                  'sparse_points': len(model.points3D),
                  'mean_reprojection_error': model.compute_mean_reprojection_error()}
    else:
        source = {'densify': 'scene', 'mesh': 'dense', 'refine': 'mesh_scene'}[action]
        source_path = Path(parent[source])
        expected = parent['artifact_hashes'][str(source_path)]
        if sha(source_path) != expected:
            raise ValueError('Parent artifact changed')
        result = out / (action + '.mvs')
        tool = {'densify': 'DensifyPointCloud', 'mesh': 'ReconstructMesh', 'refine': 'RefineMesh'}[action]
        argv = [str(legacy.TOOLS / tool), '-i', str(source_path), '-o', str(result),
                '--max-threads', str(legacy.THREADS), '--archive-type', '2']
        if action == 'densify':
            argv += ['--estimate-roi', '0', '--crop-to-roi', '0', '--tower-mode', '0']
        else:
            argv += ['--export-type', 'obj']
            if action == 'mesh':
                argv += ['--crop-to-roi', '0']
        for key, value in sorted(node['parameters'].items()):
            argv += ['--' + key, str(value)]
        workspace = Path(parent['mvs_workspace'])
        invoke(argv, out, workspace)
        if action == 'densify':
            cloud = result.with_suffix('.ply')
            with cloud.open('rb') as stream:
                for _ in range(100):
                    line = stream.readline().decode('ascii').strip()
                    if line.startswith('element vertex '):
                        count = int(line.split()[-1])
                        break
                else:
                    raise ValueError('Missing dense point count')
            if count <= 0:
                raise ValueError('Empty dense output')
            depth_files = sorted(set(workspace.glob('depth*.dmap')) |
                                 set(out.glob('depth*.dmap')))
            if not depth_files:
                raise ValueError('Missing depth maps')
            if len({p.name for p in depth_files}) != len(depth_files):
                raise ValueError('Ambiguous depth map names')
            coverage = json.loads(Path(parent['coverage_path']).read_text())
            depth_by_image = {}
            for path in depth_files:
                image_id = depth_image_id(path)
                if image_id not in coverage['images'] or not coverage['images'][image_id]['registered'] or not coverage['images'][image_id]['undistorted_sha256']:
                    raise ValueError('Depth map cannot be bound to registered undistorted input')
                depth_by_image.setdefault(image_id, []).append({'path': str(path), 'sha256': sha(path)})
            for name, item in coverage['images'].items():
                item['depth_maps'] = depth_by_image.get(name, [])
                if not item['depth_maps']:
                    item['omissions'].append('not_selected_for_depth')
            coverage['depth_image_count'] = len(depth_by_image)
            coverage['depth_map_count'] = len(depth_files)
            coverage_path = out / 'coverage-dense.json'
            coverage_path.write_text(json.dumps(coverage, sort_keys=True))
            depth_manifest = out / 'depth-manifest.json'
            depth_manifest.write_text(json.dumps({p.name: {'path': str(p), 'sha256': sha(p),
                                                           'image_id': depth_image_id(p)}
                                                  for p in depth_files}, sort_keys=True))
            state.update(stage='dense', dense=str(result), dense_cloud=str(cloud),
                         depth_manifest=str(depth_manifest), depth_map_count=len(depth_files),
                         depth_image_count=len(depth_by_image), coverage_path=str(coverage_path))
            values = {'dense_points': count}
            for metric in ('mesh_vertices', 'mesh_faces'):
                state['diagnostics'].pop(metric, None)
        else:
            obj = result.with_suffix('.obj')
            vertices, faces = legacy.read_obj(obj)
            transform = np.asarray(parent['transform'])
            vertices = vertices @ transform[:, :3].T + transform[:, 3]
            area = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                          vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1) / 2
            if not np.isfinite(area).all() or np.any(area <= 1e-12):
                raise ValueError('Degenerate/nonfinite mesh')
            mesh = out / 'mesh.ply'
            legacy.sparse.write_mesh(mesh, vertices, faces)
            state.update(stage='mesh', mesh_scene=str(result), native_obj=str(obj),
                         mesh={'path': str(mesh), 'sha256': sha(mesh)})
            values = {'mesh_vertices': len(vertices), 'mesh_faces': len(faces)}
    files = [state[key] for key in ('scene', 'dense', 'dense_cloud', 'depth_manifest',
                                   'mesh_scene', 'native_obj', 'coverage_prepare_path',
                                   'coverage_path') if key in state]
    files.extend(state.get('sfm_evidence_files', []))
    files.extend(state.get('undistorted_files', []))
    if 'depth_manifest' in state:
        files.extend(item['path'] for item in
                     json.loads(Path(state['depth_manifest']).read_text()).values())
    state['artifact_hashes'] = {path: sha(path) for path in files}
    for key, value in values.items():
        state['diagnostics'][key] = diagnostic(value, str(out), 'colmap-openmvs-v1/' + key)
    return state


def fixture_action(request):
    """Explicit non-benchmark tetrahedron fixture; never accepted as a real submission."""
    state = dict(request['parent'])
    action = request['node']['action']
    state['stage'] = {'prepare': 'sparse', 'densify': 'dense', 'mesh': 'mesh', 'refine': 'mesh'}[action]
    if action in ('mesh', 'refine'):
        path = Path(request['output']) / 'fixture.ply'
        path.write_text('ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 4\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 2 1\n3 0 1 3\n3 1 2 3\n3 2 0 3\n')
        state['mesh'] = {'path': str(path), 'sha256': sha(path)}
        state['diagnostics'] = {'mesh_faces': diagnostic(4, str(path), 'fixture/v1')}
    return state


if __name__ == '__main__':
    request = read_json(sys.argv[1])
    if request['scope'] == 'controlled_fixture':
        result = fixture_action(request)
    elif request['scope'] == 'development':
        result = real_action(request)
    else:
        raise ValueError('Unknown execution scope')
    with (Path(request['output']) / 'state.json').open('x') as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
