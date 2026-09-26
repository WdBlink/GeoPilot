"""Exact-file bootstrap repair for the frozen A/B camera-to-mesh diagnostic."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ENTRY = Path(__file__).resolve()
ROOT = ENTRY.parents[2]
BASE = ROOT / 'out/geopilot-research-20260925'
HELPER = ENTRY.with_name('shared_camera_ba_diagnostic.py')
DRIVER = ENTRY.with_name('camera_mesh_diagnostic.py')
SOURCE_SHA = {
    str(HELPER): '07d85983a366f8b928e104a47884d9ddaea343554efd3b0c0b0210db9c5f71ac',
    str(DRIVER): 'd7783ee8fd12fda25af73f2722b7d389826abc48bde02c2a2febff0ed77581b2',
}


def load_bound(name, path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA[str(path)]:
        raise ValueError('Frozen bootstrap source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ba = load_bound('shared_camera_ba_diagnostic', HELPER)
driver = load_bound('camera_mesh_diagnostic_frozen', DRIVER)
ORIGINAL_INPUTS, ORIGINAL_ISOLATION = driver.inputs, driver.isolation
driver.OUTPUT = BASE / 'camera-mesh-r2-run'
driver.PLAN = BASE / 'camera-mesh-r2-plan.md'
# Frozen main constructs its child argv from __file__; map that entry to this
# bootstrap while preserving and separately binding the actual implementation.
driver.__file__ = str(ENTRY)


def inputs():
    manifest = ORIGINAL_INPUTS()
    manifest['bindings'].update(SOURCE_SHA)
    manifest['bindings'][str(BASE / 'camera-mesh-plan.md')] = '4295d4308d1011084071bc8bf3bd5fc66d738df86d9b1e43884980c21e9c7711'
    for bound in manifest['worker_bindings'].values():
        bound.update(SOURCE_SHA)
    manifest['bootstrap'] = {
        'entrypoint': str(ENTRY), 'implementation': str(DRIVER), 'source_hashes': SOURCE_SHA,
        'module_file_mapping': {'camera_mesh_diagnostic_frozen.__file__': str(ENTRY)},
        'reason': 'Exact file imports avoid Python FileFinder directory listing under the unchanged profile',
        'failed_campaign_preserved': str(BASE / 'camera-mesh-run'),
    }
    ba.verify(manifest['bindings'])
    return manifest


def isolation(case, name, manifest, manifest_path):
    proof = ORIGINAL_ISOLATION(case, name, manifest, manifest_path)
    command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin',
        'TMPDIR=' + str(case), 'PYTHONDONTWRITEBYTECODE=1', 'OPENSSL_CONF=/dev/null',
        '/usr/bin/sandbox-exec', '-f', proof['profile'], str(ba.RUNTIME / 'bin/python'),
        '-B', str(ENTRY), '_import_check']
    started = time.monotonic()
    with (case / 'bootstrap-probe.stdout').open('xb') as out, (case / 'bootstrap-probe.stderr').open('xb') as err:
        result = subprocess.run(command, cwd=case, stdout=out, stderr=err, timeout=15, check=False)
    if result.returncode or (case / 'bootstrap-probe.stdout').read_text().strip() != 'BOOTSTRAP_IMPORT_PASS':
        raise RuntimeError('Actual worker bootstrap import failed: ' + (case / 'bootstrap-probe.stderr').read_text()[-4000:])
    ba.write(case / 'bootstrap-probe.json', {'status': 'PASS', 'argv': command,
        'wall_seconds': time.monotonic() - started, 'source_hashes': SOURCE_SHA,
        'profile_sha256': proof['profile_sha256'], 'numerical_execution': False})
    print('A/B branch ' + name + ': BOOTSTRAP_IMPORT_PASS under the actual worker profile')
    return proof


driver.inputs, driver.isolation = inputs, isolation

if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('Optimized Python is unsupported')
    if sys.argv[1:] == ['_import_check']:
        # Imports and overrides above are identical to _worker; stop before
        # manifest/model reads, copying, undistortion or numerical reconstruction.
        print('BOOTSTRAP_IMPORT_PASS')
    else:
        driver.main()
