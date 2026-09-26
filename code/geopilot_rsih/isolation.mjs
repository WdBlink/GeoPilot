import { spawnSync } from 'node:child_process';
import { writeFileSync, realpathSync, existsSync, readFileSync, statSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const mvs = join(root, 'out/geopilot_rsi_dependencies/openmvs-2.4.0');
const binaries = ['InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'RefineMesh']
  .map(name => join(mvs, name));

function sharedLibraries(executable, seen = new Set()) {
  const file = realpathSync(executable);
  if (seen.has(file)) return seen;
  seen.add(file);
  const result = spawnSync('/usr/bin/otool', ['-L', file], { encoding: 'utf8' });
  if (result.status !== 0) throw new Error('Cannot inspect runtime libraries: ' + file);
  for (const line of result.stdout.split('\n').slice(1)) {
    const library = line.trim().split(' (')[0].replace('@loader_path/', dirname(file) + '/');
    if (library.startsWith('/opt/homebrew/') && existsSync(library)) sharedLibraries(library, seen);
  }
  return seen;
}

/** Default-deny process profile; input data is an exact file allowlist. */
export function makeProfile(output, python, inputFiles = []) {
  if (process.platform !== 'darwin') throw new Error('No accepted isolation backend for this platform');
  const result = spawnSync(python, ['-B', '-c', 'import sys,json; print(json.dumps([sys.prefix,sys.base_prefix]))'], { encoding: 'utf8' });
  if (result.status !== 0) throw new Error('Python runtime discovery failed');
  const runtimeRoots = JSON.parse(result.stdout).map(path => realpathSync(path));
  const roots = ['/System/Library', '/usr/lib', '/Library/Apple',
    dirname(dirname(realpathSync(process.execPath))), ...runtimeRoots, here,
    ...(python === '/usr/bin/python3' ? ['/Library/Developer/CommandLineTools'] : [])];
  const files = [join(root, 'experiments/geopilot_rsi/run.py'),
    join(root, 'code/usegeo_mesh_baseline/runner.py'), ...binaries, ...inputFiles];
  for (const file of files) {
    if (!statSync(file).isFile()) throw new Error('Nonregular allowed file: ' + file);
  }
  const libraries = [...new Set([process.execPath, python, ...binaries]
    .flatMap(path => [...sharedLibraries(path)]))];
  const libraryDirectories = [...new Set(libraries.filter(p => p.includes('/lib/')).map(dirname))];
  const profile = ['(version 1)', '(deny default)',
    '(import "/System/Library/Sandbox/Profiles/dyld-support.sb")',
    '(allow process-exec process-fork sysctl-read process-info* signal)',
    '(allow file-read-metadata)',
    '(allow file-read* file-map-executable ' + roots.map(p => `(subpath ${JSON.stringify(p)})`).join(' ') + ')',
    '(allow file-read* file-map-executable ' + files.map(p => `(literal ${JSON.stringify(p)})`).join(' ') + ')',
    '(allow file-read* file-map-executable ' + libraries.map(p => `(literal ${JSON.stringify(p)})`).join(' ') + ')',
    '(allow file-read* file-map-executable ' + libraryDirectories.map(p => `(subpath ${JSON.stringify(p)})`).join(' ') + ')',
    '(allow file-read* (literal "/dev/random") (literal "/dev/urandom"))',
    `(allow file-read* file-write* file-lock (literal "/dev/null") (subpath ${JSON.stringify(output)}))`,
  ].join('\n');
  const path = join(output, 'worker.sb');
  writeFileSync(path, profile, { flag: 'wx' });
  return path;
}

function sandbox(profile, output, command, args) {
  return spawnSync('/usr/bin/sandbox-exec', ['-f', profile, command, ...args],
    { encoding: 'utf8', cwd: output, timeout: 15000,
      env: { PATH: '/usr/bin:/bin', HOME: output, TMPDIR: output, OPENSSL_CONF: '/dev/null' } });
}

/** Probe allowed data, prohibited data/network and actual numerical loaders. */
export function probeIsolation(profile, output, python, allowedFiles, forbiddenFiles, numerical = true) {
  const marker = join(output, 'isolation-positive.txt');
  const probe = (command, args, label) => {
    const result = sandbox(profile, output, command, args);
    if (result.status !== 0 || result.stdout.trim() !== 'PASS') {
      throw new Error(`${label} probe failed: ${result.stderr || result.stdout || result.error?.message}`);
    }
  };
  const payload = JSON.stringify({ marker, allowedFiles, forbiddenFiles });
  const node = `const fs=require('node:fs'),net=require('node:net'),p=${payload};
    fs.writeFileSync(p.marker,'allowed');
    if(fs.readFileSync(p.marker,'utf8')!=='allowed')process.exit(2);
    for(const f of p.allowedFiles){const fd=fs.openSync(f,'r');fs.closeSync(fd)}
    for(const f of p.forbiddenFiles){try{fs.readFileSync(f);process.exit(3)}catch(e){if(!['EPERM','EACCES'].includes(e.code))process.exit(4)}}
    const s=net.createConnection({host:'127.0.0.1',port:9});
    s.on('connect',()=>process.exit(5));s.on('error',e=>{if(!['EPERM','EACCES'].includes(e.code))process.exit(6);console.log('PASS')});
    setTimeout(()=>process.exit(7),3000).unref();`;
  probe(process.execPath, ['-e', node], 'Node');
  const py = `import json,os,socket,sys\np=json.loads(sys.argv[1])\nopen(p['marker'],'rb').close()\nfor f in p['allowedFiles']: open(f,'rb').close()\nfor f in p['forbiddenFiles']:\n try: open(f,'rb').close(); sys.exit(3)\n except PermissionError: pass\ns=socket.socket()\ntry: s.connect(('127.0.0.1',9)); sys.exit(5)\nexcept PermissionError: pass\n${numerical ? `sys.path.insert(0,${JSON.stringify(here)})\nfrom tools import load_legacy_adapter\nload_legacy_adapter()` : ''}\nprint('PASS')`;
  probe(python, ['-B', '-c', py, payload], 'Python/pycolmap');
  for (const binary of binaries) {
    const result = sandbox(profile, output, binary, ['--help']);
    if (result.error || result.signal || ![0, 1].includes(result.status) ||
        !((result.stdout || '') + (result.stderr || '')).trim()) {
      throw new Error('OpenMVS loader/help probe failed: ' + binary + ': ' + (result.stderr || result.error?.message));
    }
  }
  return { status: 'PASS', allowed_files: allowedFiles.length, forbidden_files: forbiddenFiles.length,
    node_denials: true, python_denials: true, python_numerical_imports: true, openmvs_help: binaries.length };
}
