"""One manual P2 dense-prefix control: A=.25, A-repeat=.25, B=.5.

This is not a GeoPilot candidate launcher. No prepare, densify, provider, export,
or evaluator is invoked. Run ``check`` before running the three cases in order.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'code/geopilot_rsih'))
from strict_json import read_json
from supervise import supervise
from tools import sha

SOURCE = ROOT / 'out/geopilot-learning-20260923/p2-run'
OUTPUT = ROOT / 'out/geopilot-research-20260925/fixed-prefix-control'
PLAN = ROOT / 'out/geopilot-research-20260925/prefix-plan.md'
DEFAULTS_HELP = ROOT / 'out/geopilot-learning-20260921/mesh-help.txt'
CASES = {'A': .25, 'A-repeat': .25, 'B': .5}
LIMITS = {'timeout_seconds': 1200, 'max_rss_bytes': 32 * 1024**3,
          'min_free_bytes': 30 * 1024**3, 'threads': 8}
PINNED = {
    'nodes/densify/state.json': '79b385a7b591fd6867091739ac4e7448e0216a1c03da68a8839b2946a2f6982a',
    'nodes/densify/request.json': '39b0ee3ea051d5f2ed493cae8a4d99bd06dfdb1df85fd2c1000e58148785fdd4',
    'nodes/mesh/request.json': '867c13dbc811ea2f9b68368e0ee062a691dcb421f02c9f74a36bd1a2e8442196',
    'request.json': '846f84b42c301261b0197f1ee1606d96db2f67ae28aaed9a96aba0887e2ca6ec',
    'launch.json': '2a5023086874bd1d138164634f109540fcf1ce57f9ab6a64a66c3d3955145e88',
}
TOOLS = ROOT / 'out/geopilot_rsi_dependencies/openmvs-2.4.0'
NUMERICAL_SOURCES = (
    'code/geopilot_rsih/tools.py', 'code/geopilot_rsih/strict_json.py',
    'code/geopilot_rsih/supervise.py', 'experiments/geopilot_rsi/run.py',
    'code/usegeo_mesh_baseline/runner.py',
    'out/geopilot_rsi_dependencies/openmvs-2.4.0/ReconstructMesh',
)


def write_json(path: Path, value: object) -> None:
    """Never overwrite a previous case or its evidence."""
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def verify(bindings: dict[str, str]) -> None:
    """Reject missing or changed bound files, including copied inputs."""
    for name, expected in bindings.items():
        if not Path(name).is_file() or sha(name) != expected:
            raise ValueError(f'Bound file changed or missing: {name}')


def regular_under(path: Path, root: Path) -> Path:
    """Require an actual file below root, with no symlink path components."""
    path.relative_to(root)
    if path.resolve() != path or not path.is_file():
        raise ValueError(f'Not a regular, unlinked input: {path}')
    return path


def check_source() -> tuple[dict, dict]:
    """Bind the fixed historical artifacts, original inputs and numerical runtime."""
    pinned = {str(SOURCE / name): value for name, value in PINNED.items()}
    verify(pinned)
    parent = read_json(SOURCE / 'nodes/densify/state.json')
    dense_request = read_json(SOURCE / 'nodes/densify/request.json')
    mesh_request = read_json(SOURCE / 'nodes/mesh/request.json')
    launch = read_json(SOURCE / 'launch.json')
    runtime = launch['runtime']
    if (parent['stage'] != 'dense' or mesh_request['parent'] != parent
            or dense_request['node']['parameters'] != {'resolution-level': 1}
            or mesh_request['node']['parameters'] != {'decimate': .5}
            or parent['input_manifest_sha256'] != launch['input_manifest_sha256']
            or dense_request['inputs'] != mesh_request['inputs']
            or runtime['versions'] != ['1.26.4', '1.11.1', '3.12.6']):
        raise ValueError('Unexpected P2 contract')
    bindings = dict(pinned)
    for name in parent['artifact_hashes']:
        regular_under(Path(name), SOURCE)
    bindings.update(parent['artifact_hashes'])
    bindings.update(launch['input_hashes'])
    for name in NUMERICAL_SOURCES:
        path = str(ROOT / name)
        bindings[path] = launch['bindings'][path]
    bindings[str(DEFAULTS_HELP)] = launch['bindings'][str(DEFAULTS_HELP)]
    # Bind the actual numerical environment, not unrelated upstream npm/evaluator trees.
    for path, digest in launch['bindings'].items():
        if any(Path(path).is_relative_to(runtime[key]) for key in ('prefix', 'base_prefix')):
            bindings[path] = digest
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    bindings[str(PLAN)] = sha(PLAN)
    verify(bindings)
    # P2's pinned archive is OpenMVS v1, zlib archive-type 2. Its image table is
    # near the beginning; bound decompression avoids reading the point payload.
    with Path(parent['dense']).open('rb') as stream:
        if stream.read(20) != bytes.fromhex('4d56530001000000020000000000000000000000'):
            raise ValueError('Unexpected P2 MVS archive header')
        metadata = zlib.decompressobj().decompress(stream.read(1024**2), 16 * 1024**2)
    workspace = Path(parent['mvs_workspace'])
    for name in parent['undistorted_files']:
        relative = Path(name).relative_to(workspace).as_posix()
        if not relative.startswith('undistorted/images/') or relative.encode() not in metadata:
            raise ValueError('P2 archive image layout cannot be verified')
    log = SOURCE / 'nodes/mesh/numerical.log'
    snapshot = {
        'schema': 'manual-fixed-prefix-control/1', 'formal_candidate': False,
        'benchmark_eligible': False, 'provider_calls': 0, 'cases': CASES,
        'source': str(SOURCE), 'inputs': mesh_request['inputs'],
        'transform': parent['transform'], 'runtime': runtime,
        'bindings': bindings, 'limits': LIMITS,
        'historical_mesh_log_sha256': sha(log),
        'historical_openmvs_version': log.read_text().splitlines()[:2],
        'implicit_defaults_source': {'path': str(DEFAULTS_HELP),
                                     'sha256': bindings[str(DEFAULTS_HELP)],
                                     'config_file': 'ReconstructMesh.cfg',
                                     'new_workspace_config_policy': 'must_be_absent'},
        'source_artifact_count': len(parent['artifact_hashes']),
        'copy_bytes_per_case': sum(Path(p).stat().st_size for p in parent['artifact_hashes']),
        'mvs_relative_images_verified': len(parent['undistorted_files']),
    }
    return parent, snapshot


def stage_prefix(parent: dict, case: Path, source: Path = SOURCE) -> dict:
    """Copy all prefix artifacts; keep the MVS-relative image/depth layout intact."""
    staged = copy.deepcopy(parent)
    mapping = {}
    for name, digest in parent['artifact_hashes'].items():
        original = regular_under(Path(name), source)
        target = case / 'prefix' / original.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)  # Independent inodes; never hardlink history.
        if sha(target) != digest:
            raise ValueError(f'Copy hash mismatch: {name}')
        mapping[name] = str(target)
    for key in ('scene', 'dense', 'dense_cloud', 'depth_manifest',
                'coverage_prepare_path', 'coverage_path'):
        staged[key] = mapping[parent[key]]
    for key in ('sfm_evidence_files', 'undistorted_files'):
        staged[key] = [mapping[name] for name in parent[key]]
    staged['mvs_workspace'] = str(case / 'prefix' / Path(parent['mvs_workspace']).relative_to(source))
    # real_action rehashes every depth record after meshing, so rebase this one
    # consumed JSON manifest. Other copied provenance JSON remains byte-identical.
    manifest_path = Path(staged['depth_manifest'])
    manifest = read_json(manifest_path)
    for item in manifest.values():
        item['path'] = mapping[item['path']]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, allow_nan=False))
    staged['artifact_hashes'] = {target: sha(target) for target in mapping.values()}
    for name in staged['artifact_hashes']:
        Path(name).chmod(0o444)
    write_json(case / 'prefix-bindings.json', {
        'original_hashes': parent['artifact_hashes'], 'original_to_copy': mapping,
        'executed_hashes': staged['artifact_hashes'],
        'rewritten_manifest': str(manifest_path),
    })
    return staged


def make_request(label: str, parent: dict, case: Path, inputs: str) -> dict:
    """Expose exactly the three preregistered mesh calls; no arbitrary parameters."""
    if label not in CASES:
        raise ValueError('Unknown fixed control case')
    return {'node': {'id': label, 'kind': 'tool', 'action': 'mesh',
                     'parameters': {'decimate': CASES[label]}},
            'parent': parent, 'output': str(case / 'mesh'), 'inputs': inputs,
            'scope': 'development'}


def run_case(label: str, parent: dict, snapshot: dict) -> dict:
    """Run once under existing process-group limits, retaining failures and logs."""
    previous = list(CASES)[:list(CASES).index(label)]
    for name in previous:
        if read_json(OUTPUT / name / 'status.json')['status'] != 'completed':
            raise ValueError('Previous case failed; stop this campaign')
    if shutil.disk_usage(OUTPUT).free < LIMITS['min_free_bytes']:
        raise RuntimeError('At least 30 GiB free space required before each case')
    case = OUTPUT / label
    case.mkdir()  # Refuse resume/overwrite, including incomplete previous runs.
    started = time.monotonic()
    report = {'case': label, 'decimate': CASES[label], 'status': 'failed',
              'formal_candidate': False, 'benchmark_eligible': False}
    try:
        staged = stage_prefix(parent, case)
        if (Path(staged['mvs_workspace']) / 'ReconstructMesh.cfg').exists():
            raise ValueError('Unexpected default OpenMVS configuration file')
        (case / 'mesh').mkdir()
        request = make_request(label, staged, case, snapshot['inputs'])
        write_json(case / 'request.json', request)
        command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin',
                   'TMPDIR=' + str(case),
                   'PYTHONDONTWRITEBYTECODE=1', 'OPENSSL_CONF=/dev/null',
                   snapshot['runtime']['python'], '-B',
                   str(ROOT / 'code/geopilot_rsih/tools.py'), str(case / 'request.json')]
        argv = [str(TOOLS / 'ReconstructMesh'), '-i', staged['dense'], '-o',
                str(case / 'mesh/mesh.mvs'), '--max-threads', '8', '--archive-type', '2',
                '--export-type', 'obj', '--crop-to-roi', '0', '--decimate', str(CASES[label])]
        write_json(case / 'command.json', {'worker_argv': command,
                   'expected_numerical_argv': argv, 'numerical_cwd': staged['mvs_workspace'],
                   'limits': LIMITS, 'preflight_sha256': sha(OUTPUT / 'preflight.json')})
        measured = supervise(command, case, timeout=LIMITS['timeout_seconds'],
                             max_rss=LIMITS['max_rss_bytes'])
        write_json(case / 'supervision.json', measured)
        report['supervision'] = measured
        verify(snapshot['bindings'])
        verify(staged['artifact_hashes'])
        if measured['returncode'] != 0 or measured['reason'] is not None:
            raise RuntimeError('Numerical worker failed or exceeded limits; see supervision.json')
        state = read_json(case / 'mesh/state.json')
        if state['stage'] != 'mesh' or state['transform'] != parent['transform']:
            raise ValueError('Unexpected output state')
        verify(state['artifact_hashes'])
        verify({state['mesh']['path']: state['mesh']['sha256']})
        report.update(status='completed', mesh=state['mesh'],
                      mesh_vertices=state['diagnostics']['mesh_vertices']['value'],
                      mesh_faces=state['diagnostics']['mesh_faces']['value'])
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds_including_copy_and_checks'] = time.monotonic() - started
        # Include partial outputs and OpenMVS logs on failure, too.
        report['output_hashes'] = {str(p): sha(p) for p in case.rglob('*')
                                   if p.is_file() and 'prefix' not in p.relative_to(case).parts}
        report['workspace_log_hashes'] = {str(p): sha(p) for p in (case / 'prefix').rglob('*.log')}
        write_json(case / 'status.json', report)
    return report


def main() -> None:
    """CLI intentionally accepts neither a source override nor numeric options."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check', help='Hash verification only; no copying or reconstruction')
    commands.add_parser('run', help='Run one mesh case once').add_argument('case', choices=CASES)
    args = parser.parse_args()
    parent, snapshot = check_source()
    if OUTPUT.resolve() != OUTPUT:
        raise ValueError('Output path must not contain symlinks')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = OUTPUT / 'preflight.json'
    if frozen.exists():
        if read_json(frozen) != snapshot:
            raise ValueError('Preflight snapshot changed; do not continue this campaign')
    elif args.command == 'check':
        write_json(frozen, snapshot)
    else:
        raise ValueError('Run check first')
    if args.command == 'check':
        print(json.dumps({k: snapshot[k] for k in ('schema', 'source_artifact_count',
                         'copy_bytes_per_case', 'mvs_relative_images_verified', 'cases', 'limits')}))
        return
    lock = OUTPUT / '.active'
    with lock.open('x'):
        pass
    try:
        result = run_case(args.case, parent, snapshot)
        print(json.dumps({k: result[k] for k in ('case', 'status', 'mesh_vertices', 'mesh_faces')}))
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
