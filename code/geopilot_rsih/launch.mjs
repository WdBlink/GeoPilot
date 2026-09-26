import { readFileSync, writeFileSync, mkdirSync, realpathSync, existsSync, readdirSync,
  lstatSync, openSync, readSync, closeSync } from 'node:fs';
import { dirname, basename, join, resolve, relative, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { makeGenome, digest, validateProgram, validateLearningProgram, applyProgramPatch, getProgram } from './program.mjs';
import { makeProfile, probeIsolation } from './isolation.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const mvs = join(root, 'out/geopilot_rsi_dependencies/openmvs-2.4.0');
const bundle = join(root, 'out/usegeo_benchmark/prepared/v1');
const bundleLockPath = join(root, 'out/usegeo_benchmark/prepared/v1.lock.json');
const releaseRoot = join(root, 'out/usegeo_mesh_benchmark/paper-readiness-20260918/release-v1');
const evaluatorRuntime = join(root, 'out/usegeo_mesh_benchmark/paper-readiness-20260918/clean-runtime');
const evaluatorPython = join(evaluatorRuntime, 'bin/python');
const PINNED_P0_SHA256 = '9cb15276ff543299f56c1ed54de71b18b664f329794ac7783de4098784360c3e';
const PINNED_BUNDLE_LOCK_SHA256 = 'bdff58a760d90327b5ed40876670cf3b2bcf38e08ed189d56529919dd738fe27';
const PINNED_RELEASE_MANIFEST_SHA256 = 'adcad9a3f7eee2373e165f94f20127a4323239392a98b5e924c0d8db72f77ba6';
const fileHash = path => {
  const h = createHash('sha256'), fd = openSync(path, 'r'), block = Buffer.allocUnsafe(8 * 1024 * 1024);
  try { for (let n; (n = readSync(fd, block, 0, block.length, null)) > 0;) h.update(block.subarray(0, n)); }
  finally { closeSync(fd); }
  return h.digest('hex');
};
const inside = (path, base) => { const r = relative(base, path); return r === '' || (!r.startsWith('..' + '/') && r !== '..' && !isAbsolute(r)); };
const write = (path, value) => writeFileSync(path, JSON.stringify(value, null, 2), { flag: 'wx' });
export function launchRecordUnchanged(path, expectedHash, original) {
  try {
    const current = lstatSync(path);
    return current.isFile() && current.dev === original.dev && current.ino === original.ino &&
      fileHash(path) === expectedHash;
  } catch { return false; }
}
function filesUnder(directory, links = {}, allowedExternal = []) {
  const base = realpathSync(directory);
  function visit(path) {
    return readdirSync(path, { withFileTypes: true }).flatMap(entry => {
      const child = join(path, entry.name);
      if (entry.isDirectory()) return visit(child);
      if (entry.isFile()) return [child];
      if (!entry.isSymbolicLink()) throw new Error('Unsupported runtime entry: ' + child);
      const target = realpathSync(child);
      if (![base, ...allowedExternal].some(root => inside(target, root)) || !lstatSync(target).isFile()) {
        throw new Error('Runtime symlink escapes installed roots: ' + child);
      }
      links[child] = target;
      return [child, target];
    });
  }
  return visit(directory);
}

export function assertRegisteredP0(path) {
  if (fileHash(path) !== PINNED_P0_SHA256) throw new Error('Registered fixed P0 changed');
}

/** Parent-side provenance check; offline evidence is never copied into the worker. */
export function verifyCandidate(candidate, program, python) {
  const directory = dirname(resolve(candidate));
  const files = { candidate, program, proposal: join(directory, 'proposal.json'),
    request: join(directory, 'request.json'), response: join(directory, 'response.json'),
    parent_genome: join(directory, 'parent-genome.json'), context: join(directory, 'context.json') };
  for (const path of Object.values(files)) {
    if (lstatSync(path).isSymbolicLink() || !lstatSync(path).isFile() ||
        lstatSync(path).size > 4 * 1024 * 1024) throw new Error('Unsafe candidate evidence');
  }
  const manifest = strictRead(candidate, python);
  if (!['geopilot-candidate/1', 'geopilot-candidate/2'].includes(manifest.schema) ||
      manifest.parent_program_sha256 !== '619cc3f49b06951fffc7f9ad2f19e45fe7da810596e074c0db58f64130736b8f') {
    throw new Error('Unregistered candidate parent');
  }
  for (const [key, path] of Object.entries(files)) {
    if (key !== 'candidate' && manifest[key + '_sha256'] !== fileHash(path)) {
      throw new Error('Candidate evidence hash mismatch: ' + key);
    }
  }
  assertRegisteredP0(join(here, 'p0.json'));
  const parent = strictRead(files.parent_genome, python), p = strictRead(program, python);
  validateLearningProgram(p, { requireCondition: manifest.schema !== 'geopilot-candidate/2' });
  if (digest(getProgram(parent)) !== digest(strictRead(join(here, 'p0.json'), python))) {
    throw new Error('Candidate parent program mismatch');
  }
  const response = strictRead(files.response, python);
  if (response.status !== 'ok' || digest(response.decision?.program) !== digest(p)) {
    throw new Error('Candidate differs from actual model response');
  }
  const request = strictRead(files.request, python), evidence = request.context?.evidence_bindings;
  const verifier = manifest.schema === 'geopilot-candidate/2' ? 'continuation.py' : 'learning.py';
  const verified = spawnSync(python, ['-B', join(root, 'experiments/geopilot_rsih_learning', verifier),
    '--verify-request', files.request, '--verify-response', files.response], { encoding: 'utf8' });
  if (verified.status !== 0) throw new Error('Offline evidence verification failed: ' + verified.stderr);
  if (!evidence || !Object.keys(evidence).length ||
      !/^[a-f0-9]{64}$/.test(request.provider_payload_sha256 ?? '') ||
      response.request_sha256 !== request.provider_payload_sha256 ||
      response.model !== 'MiniMax-M3' || response.response_model !== 'MiniMax-M3' ||
      !['prompt_tokens', 'completion_tokens', 'total_tokens'].every(k =>
        Number.isSafeInteger(response.usage?.[k]) && response.usage[k] >= 0)) {
    throw new Error('Unbound provider request or usage');
  }
  for (const [path, expected] of Object.entries(evidence)) {
    if (!isAbsolute(path) || lstatSync(path).isSymbolicLink() || !lstatSync(path).isFile() ||
        fileHash(path) !== expected) throw new Error('Candidate source evidence drift: ' + path);
  }
  const applied = applyProgramPatch(parent, strictRead(files.proposal, python), strictRead(files.context, python));
  if (digest(getProgram(applied.genome)) !== digest(p)) throw new Error('Candidate patch mismatch');
  return { manifest_sha256: fileHash(candidate), program_sha256: fileHash(program),
    parent_program_sha256: manifest.parent_program_sha256,
    bindings: { ...evidence,
      ...Object.fromEntries(Object.values(files).map(path => [resolve(path), fileHash(path)])) } };
}

export function assertFrozenManifest(path, expected) {
  if (fileHash(path) !== expected) throw new Error('Frozen Dataset-1 input manifest mismatch');
}

export function strictRead(path, python) {
  const result = spawnSync(python, ['-B', join(here, 'strict_json.py'), path], { encoding: 'utf8' });
  if (result.status !== 0) throw new Error(result.stderr || 'Strict JSON reader failed');
  return JSON.parse(result.stdout);
}

function regular(path, base) {
  if (!inside(path, base) || lstatSync(path).isSymbolicLink() || !lstatSync(path).isFile() ||
      !inside(realpathSync(path), base)) throw new Error('Unsafe input file: ' + path);
  return path;
}

function inputBindings(inputs, python) {
  if (inputs !== realpathSync(join(bundle, 'inputs/rgb-oriented/Dataset-1'))) {
    throw new Error('Development inputs must be the frozen Dataset-1 bundle directory');
  }
  if (fileHash(bundleLockPath) !== PINNED_BUNDLE_LOCK_SHA256) {
    throw new Error('Frozen bundle lock changed');
  }
  const lock = strictRead(bundleLockPath, python);
  const bundleManifest = strictRead(join(bundle, 'manifest.json'), python);
  const expected = lock.scenes?.['Dataset-1']?.input_manifest_sha256?.['rgb-oriented'];
  if (!/^[a-f0-9]{64}$/.test(expected ?? '') ||
      lock.bundle_manifest_sha256 !== fileHash(join(bundle, 'manifest.json')) ||
      bundleManifest.scenes?.['Dataset-1']?.input_manifest_sha256?.['rgb-oriented'] !== expected) {
    throw new Error('Frozen bundle lock or input identity mismatch');
  }
  const manifestPath = regular(join(inputs, 'input_manifest.json'), inputs);
  assertFrozenManifest(manifestPath, expected);
  const manifest = strictRead(manifestPath, python);
  if (manifest.scene_id !== 'Dataset-1' || manifest.track !== 'rgb-oriented' ||
      manifest.image_count !== 224 || !Array.isArray(manifest.images) || manifest.images.length !== 224) {
    throw new Error('Expected Dataset-1 RGB-oriented 224-image manifest');
  }
  const orientation = regular(join(inputs, 'Image_orientations_dataset1.xyz'), inputs);
  if (fileHash(orientation) !== manifest.orientation_sha256) throw new Error('Orientation hash mismatch');
  if (lstatSync(join(inputs, 'images')).isSymbolicLink()) throw new Error('Symlink image directory');
  const files = [manifestPath, orientation], seen = new Set();
  for (const item of manifest.images) {
    if (typeof item.image_id !== 'string' || basename(item.image_id) !== item.image_id ||
        seen.has(item.image_id) || !/^[a-f0-9]{64}$/.test(item.sha256)) throw new Error('Invalid image entry');
    seen.add(item.image_id);
    const path = regular(join(inputs, 'images', item.image_id), inputs);
    if (fileHash(path) !== item.sha256) throw new Error('Image hash mismatch: ' + item.image_id);
    files.push(path);
  }
  return { files, hashes: Object.fromEntries(files.map(path => [path, fileHash(path)])),
    manifest_sha256: expected, frozen_manifest_sha256: expected };
}

export function runtimeBindings(python, program, cli, numerical) {
  const symlinks = {};
  const paths = [program, cli, join(root, 'experiments/geopilot_rsi/run.py'),
    join(root, 'code/usegeo_mesh_baseline/runner.py'),
    join(root, 'code/usegeo_mesh_benchmark/protocol_v1.json'),
    join(bundle, 'manifest.json'), bundleLockPath,
    join(here, 'upstream/package-lock.json'),
    ...filesUnder(join(here, 'upstream/dist'), symlinks),
    ...filesUnder(join(here, 'upstream/node_modules'), symlinks),
    ...['InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'RefineMesh'].map(n => join(mvs, n)),
    ...readdirSync(here).filter(n => /\.(mjs|ts|py|json)$/.test(n)).map(n => join(here, n))];
  const discovery = spawnSync(python, ['-B', '-c', numerical ?
    'import json,sys,numpy,scipy,pycolmap; print(json.dumps({"python":sys.executable,"prefix":sys.prefix,"base_prefix":sys.base_prefix,"numpy":numpy.__file__,"scipy":scipy.__file__,"pycolmap":pycolmap.__file__,"versions":[numpy.__version__,scipy.__version__,pycolmap.__version__]}))' :
    'import json,sys; print(json.dumps({"python":sys.executable,"prefix":sys.prefix,"base_prefix":sys.base_prefix}))'],
    { encoding: 'utf8' });
  if (discovery.status !== 0) throw new Error('Numerical runtime discovery failed: ' + discovery.stderr);
  const modules = JSON.parse(discovery.stdout);
  if (numerical && modules.versions.join('/') !== '1.26.4/1.11.1/3.12.6') throw new Error('Numerical runtime version mismatch');
  paths.push(realpathSync(python));
  if (numerical) {
    const runtimePrefix = realpathSync(modules.prefix);
    const basePrefix = realpathSync(modules.base_prefix);
    paths.push(...filesUnder(runtimePrefix, symlinks, [basePrefix]), ...filesUnder(basePrefix, symlinks));
    paths.push(modules.numpy, modules.scipy, modules.pycolmap);
    for (const module of [modules.numpy, modules.scipy, modules.pycolmap]) {
      paths.push(...filesUnder(dirname(module), symlinks));
    }
  }
  let offline = null;
  if (numerical) {
    const releaseManifestPath = join(releaseRoot, 'release_manifest.json');
    if (fileHash(releaseManifestPath) !== PINNED_RELEASE_MANIFEST_SHA256) {
      throw new Error('Frozen release manifest changed');
    }
    const releaseManifest = strictRead(releaseManifestPath, python);
    const releaseFiles = Object.entries(releaseManifest.source_files).map(([name, expected]) => {
      const path = resolve(releaseRoot, name);
      if (!inside(path, join(releaseRoot, 'source')) || fileHash(path) !== expected) {
        throw new Error('Frozen release source changed: ' + name);
      }
      return path;
    });
    const actualReleaseFiles = filesUnder(join(releaseRoot, 'source'), symlinks)
      .filter(path => !inside(path, join(releaseRoot, 'source/out/usegeo_benchmark/prepared')));
    if (new Set(releaseFiles).size !== actualReleaseFiles.length ||
        actualReleaseFiles.some(path => !releaseFiles.includes(path))) {
      throw new Error('Frozen release source file set changed');
    }
    const evaluator = spawnSync(evaluatorPython, ['-B', '-c',
      'import json,sys,numpy,scipy,laspy,open3d; print(json.dumps({"prefix":sys.prefix,"base_prefix":sys.base_prefix,"versions":[sys.version.split()[0],numpy.__version__,scipy.__version__,laspy.__version__,open3d.__version__]}))'],
      { encoding: 'utf8', env: { PATH: '/usr/bin:/bin', HOME: '/nonexistent' } });
    if (evaluator.status !== 0) throw new Error('Frozen evaluator runtime discovery failed: ' + evaluator.stderr);
    const info = JSON.parse(evaluator.stdout);
    if (info.versions.join('/') !== '3.10.20/1.25.2/1.11.1/2.7.0/0.17.0' ||
        realpathSync(info.prefix) !== realpathSync(evaluatorRuntime)) {
      throw new Error('Frozen evaluator runtime version or path mismatch');
    }
    const basePrefix = realpathSync(info.base_prefix);
    const runtimeFiles = [...filesUnder(evaluatorRuntime, symlinks, [basePrefix]),
      ...filesUnder(basePrefix, symlinks)];
    paths.push(releaseManifestPath, ...releaseFiles, ...runtimeFiles);
    offline = { release_manifest_sha256: PINNED_RELEASE_MANIFEST_SHA256,
      release_file_count: releaseFiles.length, evaluator_runtime_file_count: runtimeFiles.length,
      evaluator_versions: info.versions, evaluator_python: evaluatorPython,
      release_root: releaseRoot, evaluator_runtime_root: evaluatorRuntime,
      evaluator_base_prefix: basePrefix };
  }
  const bindings = Object.fromEntries([...new Set(paths)].map(path => [path, fileHash(path)]));
  if (offline) {
    const treeHash = base => createHash('sha256').update(JSON.stringify(Object.entries(bindings)
      .filter(([path]) => inside(path, base)).sort(([a], [b]) => a.localeCompare(b)))).digest('hex');
    offline.release_source_tree_sha256 = treeHash(releaseRoot);
    offline.evaluator_runtime_tree_sha256 = createHash('sha256').update(
      treeHash(evaluatorRuntime) + treeHash(offline.evaluator_base_prefix)).digest('hex');
  }
  return { bindings, symlinks, modules, offline };
}

export function launch({ program, output, inputs, python, scope = 'development', preflightOnly = false, candidate = null }) {
  if (!['development', 'controlled_fixture'].includes(scope)) throw new Error('Unknown scope');
  python = resolve(python);
  if (!existsSync(python)) throw new Error('Python executable missing');
  program = realpathSync(program);
  output = join(realpathSync(dirname(resolve(output))), basename(output));
  inputs = inputs ? realpathSync(inputs) : null;
  if (scope === 'development' && (!inputs || (!candidate && program !== join(here, 'p0.json')))) {
    throw new Error('Development requires registered P0 or an explicit candidate, and Dataset-1 inputs');
  }
  if (candidate && scope !== 'development') throw new Error('Candidates require development scope');
  for (const protectedPath of [here, program, inputs].filter(Boolean)) {
    if (inside(output, protectedPath) || inside(protectedPath, output)) throw new Error('Output overlaps protected input');
  }
  if (existsSync(output)) throw new Error('Output must be new');
  const p = validateProgram(strictRead(program, python));
  const candidateProof = candidate ? verifyCandidate(resolve(candidate), program, python) : null;
  if (scope === 'development' && !candidateProof) assertRegisteredP0(program);
  const lock = strictRead(join(here, 'upstream-lock.json'), python);
  for (const [name, expected] of Object.entries(lock.files_sha256)) {
    if (fileHash(join(here, 'upstream', name)) !== expected) throw new Error('Upstream source drift: ' + name);
  }
  const cli = join(here, 'upstream/dist/src/cli.js');
  if (!existsSync(cli)) throw new Error('Build upstream first');
  const sources = runtimeBindings(python, program, cli, scope === 'development');
  if (candidateProof) Object.assign(sources.bindings, candidateProof.bindings);
  const input = scope === 'development' ? inputBindings(inputs, python) : null;
  mkdirSync(output);
  output = realpathSync(output);
  const genome = makeGenome(p);
  genome.model = { profile: 'geopilot-driver', id: 'no-model-calls' };
  write(join(output, 'genome.json'), genome);
  write(join(output, 'program.json'), p);
  const config = join(output, 'providers.json');
  write(config, { 'geopilot-driver': { kind: 'mock', model: 'no-model-calls', mock_responses: [] } });
  const request = { genome_path: join(output, 'genome.json'), genome_sha256: digest(genome),
    output, inputs, python, scope };
  write(join(output, 'request.json'), request);
  const effective = Object.fromEntries(['genome.json', 'program.json', 'providers.json', 'request.json']
    .map(name => [join(output, name), fileHash(join(output, name))]));
  const environment = { PATH: '/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin', HOME: output,
    TMPDIR: output, GEOPILOT_REQUEST: join(output, 'request.json'), GEOPILOT_PYTHON: python,
    RSIH_CODING_AGENT_DIR: join(output, 'agent'), PI_CODING_AGENT_DIR: join(output, 'agent'),
    PI_OFFLINE: '1', PI_SKIP_VERSION_CHECK: '1', OPENSSL_CONF: '/dev/null' };
  const argv = [cli, '--genome', request.genome_path, '--config', config, '--cwd', output,
    '--run-id', 'geopilot', '--no-tools', '--tools', 'geopilot_execute',
    '--no-extensions', '--no-skills', '--no-prompt-templates', '--no-themes',
    '-e', join(here, 'extension.ts'), '-p', '/geopilot-run'];
  let isolation, profile;
  try {
    profile = makeProfile(output, python, input?.files ?? []);
    const forbidden = [join(bundle, 'evaluator-only/Dataset-1/reference_manifest.json'),
      join(root, 'code/usegeo_mesh_benchmark/protocol_v1.json'),
      join(releaseRoot, 'source/code/usegeo_mesh_benchmark/benchmark.py'),
      join(evaluatorRuntime, 'lib/python3.10/site-packages/open3d/__init__.py'),
      join(root, 'experiments/geopilot_rsi/p0.json'),
      join(process.env.HOME ?? '/nonexistent', '.codex/config.toml'),
      ...Object.keys(candidateProof?.bindings ?? {}).filter(path => path !== program)].filter(existsSync);
    if (!forbidden.length) throw new Error('No real forbidden targets available for preflight');
    isolation = probeIsolation(profile, output, python,
      scope === 'development' ? [input.files[0], input.files[1], input.files[2]] : [], forbidden,
      scope === 'development');
    write(join(output, 'isolation.json'), isolation);
    const launchPath = join(output, 'launch.json');
    write(launchPath, { scope, bindings: sources.bindings,
      effective, symlinks: sources.symlinks, offline_validator: sources.offline,
      input_hashes: input?.hashes ?? null, input_manifest_sha256: input?.manifest_sha256 ?? null,
      frozen_input_manifest_sha256: input?.frozen_manifest_sha256 ?? null,
      input_image_ids: input?.files.slice(2).map(path => basename(path)).sort() ?? null,
      runtime: sources.modules, upstream_commit: lock.commit, program_path: program,
      program_bytes_sha256: fileHash(program), program_content_sha256: digest(p),
      genome_sha256: request.genome_sha256, argv, environment, online_tokens: 0,
      model_usage: { mode: 'disabled_by_entry', measured: false }, candidate: candidateProof,
      isolation_profile_sha256: fileHash(profile) });
    const launchHash = fileHash(launchPath), launchStat = lstatSync(launchPath);
    if (preflightOnly) {
      const run = { status: 'preflight_passed', scope, benchmark_eligible: false,
        isolation_status: isolation.status };
      write(join(output, 'run.json'), run);
      return run;
    }
    const supervised = spawnSync(python, ['-B', join(here, 'supervise.py'), output,
      '/usr/bin/sandbox-exec', '-f', profile, process.execPath, ...argv],
      { cwd: output, encoding: 'utf8', maxBuffer: 1024 * 1024, env: environment });
    const costs = existsSync(join(output, 'supervision.json')) ? strictRead(join(output, 'supervision.json'), python) : {};
    const unchanged = launchRecordUnchanged(launchPath, launchHash, launchStat) &&
      Object.entries({ ...sources.bindings, ...(input?.hashes ?? {}), ...effective,
      [profile]: fileHash(profile) }).every(([path, hash]) => fileHash(path) === hash) &&
      Object.entries(sources.symlinks).every(([path, target]) => realpathSync(path) === target);
    const execution = existsSync(join(output, 'result.json')) ? strictRead(join(output, 'result.json'), python) : null;
    const status = supervised.status === 0 && costs.returncode === 0 && costs.reason === null &&
      unchanged && execution?.status === 'sealed' && execution.program_sha256 === digest(p) ? 'sealed' : 'failed';
    const run = { status, scope, wall_seconds: costs.wall_seconds ?? null,
      peak_worker_process_group_rss_bytes: costs.peak_worker_process_group_rss_bytes ?? null,
      returncode: costs.returncode ?? supervised.status, cancellation_reason: costs.reason ?? null,
      error: supervised.error?.message ?? (supervised.status ? supervised.stderr : null),
      frozen_bindings_unchanged: unchanged, online_tokens: 0, benchmark_eligible: false };
    write(join(output, 'run.json'), run);
    if (status === 'sealed' && scope === 'development') {
      const exported = spawnSync(python, ['-B', join(here, 'export.py'), output,
        '--expected-launch-sha256', launchHash],
        { encoding: 'utf8', maxBuffer: 1024 * 1024 });
      if (exported.status !== 0) throw new Error('Export failed: ' + exported.stderr);
      return strictRead(join(output, 'run.json'), python);
    }
    return run;
  } catch (error) {
    if (!existsSync(join(output, 'run.json'))) write(join(output, 'run.json'), {
      status: 'blocked', scope, reason: error.message, benchmark_eligible: false });
    throw error;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: { program: { type: 'string' }, output: { type: 'string' },
    inputs: { type: 'string' }, python: { type: 'string' }, candidate: { type: 'string' },
    scope: { type: 'string', default: 'development' },
    'preflight-only': { type: 'boolean', default: false } } });
  try {
    if (!values.program || !values.output || !values.python) throw new Error('--program, --output and --python required');
    const result = launch({ ...values, preflightOnly: values['preflight-only'] });
    console.log(JSON.stringify(result));
    if (!['sealed', 'preflight_passed'].includes(result.status)) process.exitCode = 1;
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
