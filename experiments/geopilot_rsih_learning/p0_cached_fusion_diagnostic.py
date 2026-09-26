"""One isolated cached-depth diagnostic; never resumes or replaces failed P0."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/geopilot_rsih'))
from supervise import supervise

PILOT = ROOT / 'out/geopilot-research-20260925/native-post-prepare-pilot'
SOURCE = PILOT / 'p0-run'
PREPARE = SOURCE / 'nodes/prepare'
OUTPUT = PILOT / 'p0-cache-diagnostic'
WORKSPACE = OUTPUT / 'workspace'
BINARY = ROOT / 'out/geopilot_rsi_dependencies/openmvs-2.4.0/DensifyPointCloud'
BINARY_SHA = '8ea970b0349270754d23f31a11dea87417c06d1001fa1ef437ee69b6146a1ce0'
ISOLATION = ROOT / 'code/geopilot_rsih/isolation.mjs'
LIMITS = {'timeout_seconds': 900, 'max_rss_bytes': 16 * 1024**3,
          'min_free_bytes': 20 * 1024**3}


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(name, value):
    with (OUTPUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def snapshot(root):
    """Reject links/special entries and bind every regular file, including inode."""
    result = {}
    for path in [root, *sorted(root.rglob('*'))]:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError(f'Nonregular input: {path}')
        if path.is_file():
            result[str(path.relative_to(root))] = {
                'sha256': sha(path), 'size': info.st_size, 'device': info.st_dev,
                'inode': info.st_ino, 'mtime_ns': info.st_mtime_ns}
    return result


def content(files):
    return {p: (v['size'], v['sha256']) for p, v in files.items()}


def independent_copy(source, copied):
    if content(source) != content(copied):
        raise ValueError('Copy content differs')
    if any((v['device'], v['inode']) == (copied[p]['device'], copied[p]['inode'])
           for p, v in source.items()):
        raise ValueError('Hardlinked copy rejected')


def self_check():
    """Small runnable regression check for the immutable-copy boundary."""
    with tempfile.TemporaryDirectory(dir=OUTPUT) as folder:
        base = Path(folder)
        a, b = base / 'a', base / 'b'
        a.mkdir(); b.mkdir()
        (a / 'file').write_bytes(b'original')
        shutil.copyfile(a / 'file', b / 'file')
        independent_copy(snapshot(a), snapshot(b))
        for kind in ('hardlink', 'symlink', 'changed'):
            (b / 'file').unlink()
            if kind == 'hardlink':
                os.link(a / 'file', b / 'file')
            elif kind == 'symlink':
                (b / 'file').symlink_to(a / 'file')
            else:
                (b / 'file').write_bytes(b'changed')
            try:
                independent_copy(snapshot(a), snapshot(b))
            except ValueError:
                continue
            raise AssertionError(f'{kind} was accepted')


def resources(name):
    """Record host observations; they are not OpenMVS available-cache bytes."""
    observed = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                'disk_free_bytes': shutil.disk_usage(OUTPUT).free,
                'memory_caveat': 'memory_pressure/vm_stat are host observations, not exact available OpenMVS cache memory'}
    for key, argv in {'memory_pressure': ['/usr/bin/memory_pressure'],
                      'vm_stat': ['/usr/bin/vm_stat']}.items():
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        observed[key] = {'argv': argv, 'returncode': result.returncode,
                         'stdout': result.stdout, 'stderr': result.stderr}
    result = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,rss=,command='],
                            capture_output=True, text=True, check=True)
    pattern = re.compile(r'DensifyPointCloud|ReconstructMesh|RefineMesh|InterfaceCOLMAP|'
                         r'/evaluator/|(?:^|/)(?:score[^ /]*|benchmark)\.py(?: |$)|'
                         r'(?:native_launch|geopilot_rsih/launch)\.mjs.*--run|'
                         r'colmap (?:mapper|patch_match_stereo|stereo_fusion)')
    observed['other_reconstruction_or_scoring_processes'] = [
        line.strip() for line in result.stdout.splitlines() if pattern.search(line)]
    write(name, observed)
    return observed


def prepare():
    """Clone and verify all inputs, then exercise the actual isolation profile."""
    OUTPUT.mkdir()
    self_check()
    if sha(BINARY) != BINARY_SHA:
        raise ValueError('Binary identity mismatch')
    host = resources('resources-prepare.json')
    if host['disk_free_bytes'] < LIMITS['min_free_bytes'] or host['other_reconstruction_or_scoring_processes']:
        raise ValueError('Disk/concurrency preflight failed')
    before = snapshot(SOURCE)
    original = {p.removeprefix('nodes/prepare/'): v for p, v in before.items()
                if p.startswith('nodes/prepare/')}
    subprocess.run(['/bin/cp', '-cRp', str(PREPARE), str(WORKSPACE)], check=True)
    copied = snapshot(WORKSPACE)
    independent_copy(original, copied)
    depth = sorted(p for p in copied if re.fullmatch(r'depth\d+\.dmap', p))
    images = sorted(p for p in copied if p.startswith('undistorted/images/'))
    if len(depth) != 224 or len(images) != 224:
        raise ValueError('Expected exactly 224 cached depths and undistorted images')
    if content(before) != content(snapshot(SOURCE)):
        raise ValueError('Original P0 changed during copy')
    python = json.loads((SOURCE / 'request.json').read_text())['python']
    node = str(Path(shutil.which('node')).resolve())
    forbidden = [str(PREPARE / 'scene.mvs'), str(PREPARE / depth[0]),
                 str(PREPARE / images[0]), str(SOURCE / 'run.json'),
                 str(ROOT / 'out/usegeo_benchmark/prepared/v1/evaluator-only/Dataset-1/reference_manifest.json'),
                 str(ROOT / 'experiments/geopilot_rsih_learning/evaluator/benchmark.py'),
                 *[str(PILOT / branch / 'raw-response.txt')
                   for branch in ('import-a', 'import-b', 'import-a-r2', 'import-b-r2')]]
    # Existence only: prohibited reference/raw files are never read by this parent.
    if not all(Path(p).is_file() for p in forbidden):
        raise ValueError('A required negative-probe path is missing')
    request = {'output': str(OUTPUT), 'python': python,
               'allowed': [str(WORKSPACE / p) for p in ['scene.mvs', *images, *depth]],
               'forbidden': forbidden, 'source': str(SOURCE)}
    write('isolation-request.json', request)
    javascript = """
const fs=await import('node:fs');
const {spawnSync}=await import('node:child_process');
const {makeProfile,probeIsolation}=await import(process.argv[1]);
const s=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const profile=makeProfile(s.output,s.python,[]);
fs.appendFileSync(profile,'\\n(deny file-read* file-write* (subpath '+JSON.stringify(s.source)+'))\\n');
const result=probeIsolation(profile,s.output,s.python,s.allowed,s.forbidden,false);
const code=`import os,sys,json\nfor p in json.loads(sys.argv[1]):\n try:\n  fd=os.open(p,os.O_WRONLY);os.close(fd);sys.exit(3)\n except PermissionError: pass\nprint('PASS')`;
const writeProbe=spawnSync('/usr/bin/sandbox-exec',['-f',profile,s.python,'-B','-c',code,JSON.stringify(s.forbidden)],{encoding:'utf8',cwd:s.output});
if(writeProbe.status!==0||writeProbe.stdout.trim()!=='PASS')throw Error('Write denial failed: '+writeProbe.stderr);
result.python_numerical_imports=false;
result.numerical_imports_requested=false;
result.forbidden_write_open_denials=true;
console.log(JSON.stringify(result));
"""
    result = subprocess.run([node, '--input-type=module', '-e', javascript,
                             ISOLATION.as_uri(), str(OUTPUT / 'isolation-request.json')],
                            capture_output=True, text=True, timeout=120)
    (OUTPUT / 'isolation.stdout').write_text(result.stdout)
    (OUTPUT / 'isolation.stderr').write_text(result.stderr)
    result.check_returncode()
    write('isolation.json', json.loads(result.stdout))
    (OUTPUT / 'fusion').mkdir()
    argv = [str(BINARY), '-i', 'scene.mvs', '-o', '../fusion/densify.mvs',
            '--max-threads', '8', '--archive-type', '2', '--estimate-roi', '0',
            '--crop-to-roi', '0', '--tower-mode', '0', '--resolution-level', '1',
            '--fusion-mode', '0', '--geometric-iters', '0', '--remove-dmaps', '0']
    environment = {'PATH': '/usr/bin:/bin', 'HOME': str(WORKSPACE), 'TMPDIR': str(WORKSPACE),
                   'OPENSSL_CONF': '/dev/null'}
    wrapper = 'import os,sys,json; p=json.loads(sys.argv[1]); os.chdir(p["cwd"]); os.execve(p["argv"][0],p["argv"],p["environment"])'
    payload = {'cwd': str(WORKSPACE), 'argv': argv, 'environment': environment}
    command = ['/usr/bin/sandbox-exec', '-f', str(OUTPUT / 'worker.sb'),
               python, '-B', '-c', wrapper, json.dumps(payload)]
    write('command.json', {**payload, 'supervised_command': command, 'limits': LIMITS,
                          'numerical_invocations_permitted': 1})
    write('inputs.json', {'source': str(SOURCE), 'source_before': before,
                          'workspace': str(WORKSPACE), 'workspace_before': copied,
                          'depth_files': depth, 'image_files': images,
                          'copy_method': 'APFS clone via cp -cRp; every file has a distinct inode',
                          'bindings': {str(p): sha(p) for p in [Path(__file__), BINARY, ISOLATION,
                                      ROOT / 'code/geopilot_rsih/supervise.py', Path(python).resolve(), Path(node),
                                      OUTPUT / 'worker.sb', OUTPUT / 'command.json', OUTPUT / 'isolation.json']},
                          'self_check': 'PASS: ordinary copy accepted; hardlink, symlink and changed bytes rejected'})
    write('ready.json', {'status': 'ready_for_parent_review', 'numerical_run_started': False,
                        'inputs_sha256': sha(OUTPUT / 'inputs.json'),
                        'command_sha256': sha(OUTPUT / 'command.json')})
    print(json.dumps({'status': 'ready_for_parent_review', 'output': str(OUTPUT)}), flush=True)


def run():
    """Execute only after parent review; an exclusive receipt prevents a retry."""
    inputs = json.loads((OUTPUT / 'inputs.json').read_text())
    ready = json.loads((OUTPUT / 'ready.json').read_text())
    if ready['numerical_run_started'] or sha(OUTPUT / 'inputs.json') != ready['inputs_sha256']:
        raise ValueError('Prepared input manifest changed')
    for path, expected in inputs['bindings'].items():
        if sha(path) != expected:
            raise ValueError(f'Prepared binding changed: {path}')
    source_now, workspace_now = snapshot(SOURCE), snapshot(WORKSPACE)
    if content(source_now) != content(inputs['source_before']):
        raise ValueError('Original P0 changed since prepare')
    if content(workspace_now) != content(inputs['workspace_before']):
        raise ValueError('Workspace changed since prepare')
    independent_copy({p.removeprefix('nodes/prepare/'): v for p, v in source_now.items()
                      if p.startswith('nodes/prepare/')}, workspace_now)
    host = resources('resources-before.json')
    if host['disk_free_bytes'] < LIMITS['min_free_bytes'] or host['other_reconstruction_or_scoring_processes']:
        raise ValueError('Disk/concurrency run gate failed')
    write('started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
                           'numerical_invocation': 1})
    command = json.loads((OUTPUT / 'command.json').read_text())
    measured = supervise(command['supervised_command'], OUTPUT,
                         timeout=LIMITS['timeout_seconds'], max_rss=LIMITS['max_rss_bytes'])
    write('supervision.json', measured)
    resources('resources-after.json')
    original_after, after = snapshot(SOURCE), snapshot(WORKSPACE)
    write('original-after.json', original_after)
    write('workspace-after.json', after)
    changed_depths = [p for p in inputs['depth_files'] if p not in after or
                      content({p: after[p]}) != content({p: inputs['workspace_before'][p]})]
    extra_depths = [p for p in after if p.endswith('.dmap') and p not in inputs['depth_files']]
    fixed = not changed_depths and not extra_depths
    immutable_inputs = ['scene.mvs', *inputs['image_files']]
    scene_images_unchanged = all(p in after and content({p: after[p]}) ==
                                 content({p: inputs['workspace_before'][p]}) for p in immutable_inputs)
    logs = (OUTPUT / 'runtime.stdout').read_text(errors='replace') + (OUTPUT / 'runtime.stderr').read_text(errors='replace')
    traces = [line for line in logs.splitlines() if re.search(r'Estimated|estimating|estimation|depth-map|Fus|fus|memory|Memory', line)]
    products = {str(p.relative_to(OUTPUT)): {'size': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted((OUTPUT / 'fusion').rglob('*')) if p.is_file()}
    report = {'status': 'completed' if measured['returncode'] == 0 and measured['reason'] is None else 'failed',
              'scope': 'One cached-depth diagnostic, not a replacement P0 or valid mesh experiment',
              'supervision': measured, 'fixed_depth_cache_bytes': fixed,
              'depth_count_before': len(inputs['depth_files']), 'changed_depths': changed_depths,
              'extra_depths': extra_depths, 'scene_and_images_unchanged': scene_images_unchanged,
              'original_p0_bytes_unchanged': content(original_after) == content(inputs['source_before']),
              'dmap_mtime_changed_count': sum(p in after and after[p]['mtime_ns'] != inputs['workspace_before'][p]['mtime_ns'] for p in inputs['depth_files']),
              'outputs': products, 'trace_tail': traces[-40:],
              'code_sha256': sha(Path(__file__)), 'binary_sha256': sha(BINARY),
              'evidence_sha256': {name: sha(OUTPUT / name) for name in ['inputs.json', 'command.json',
                                  'worker.sb', 'isolation.json', 'supervision.json', 'runtime.stdout',
                                  'runtime.stderr', 'workspace-after.json', 'original-after.json']},
              'limitations': ['Exact v2.4.0 source-to-binary build provenance is unproven.',
                              'Estimated depth-maps progress may include cache loading; no direct EstimateDepthMap call instrumentation was used.',
                              'Unchanged depth bytes establish a fixed file-cache observation, not proof that no internal computation occurred.',
                              'geometric-iters=0 changes a stage option; this cannot repair the original failed P0 record.',
                              'Host memory_pressure/vm_stat cannot establish exact OpenMVS available-cache bytes.',
                              'No ground truth, scoring, mesh reconstruction, external network or model calls were used.']}
    write('report.json', report)
    (OUTPUT / 'report.md').write_text(
        '# P0 cached-depth diagnostic\n\n' +
        f"Status: **{report['status']}**. Exit {measured['returncode']}; reason {measured['reason']}; " +
        f"wall {measured['wall_seconds']:.3f} s; sampled peak RSS {measured['peak_worker_process_group_rss_bytes']} bytes.\n\n" +
        f"224 depth files byte-identical: {fixed}; original P0 byte-identical: {report['original_p0_bytes_unchanged']}; " +
        f"scene/images unchanged: {scene_images_unchanged}.\n\n" +
        '\n'.join('- ' + line for line in report['limitations']) + '\n\n' +
        'Commands, complete input identities, isolation proof, supervision, raw logs and output SHA-256 values are retained beside this report.\n')
    print(json.dumps({key: report[key] for key in ['status', 'supervision', 'fixed_depth_cache_bytes',
                                                  'original_p0_bytes_unchanged', 'outputs']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run}[args.action]()
