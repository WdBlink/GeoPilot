import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, symlinkSync, existsSync, realpathSync, lstatSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { makeProfile, probeIsolation } from './isolation.mjs';
import { assertRegisteredP0, assertFrozenManifest, runtimeBindings, launchRecordUnchanged, verifyCandidate } from './launch.mjs';
import { makeGenome, digest, CONTRACT_ID, SKILL } from './program.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const python = join(root, 'out/usegeo_mesh_benchmark/paper-readiness-20260918/baseline-preflight/baseline-runtime/bin/python');
const sha = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const write = (path, value) => writeFileSync(path, JSON.stringify(value));
const bundle = join(root, 'out/usegeo_benchmark/prepared/v1');
const manifest = JSON.parse(readFileSync(join(bundle, 'inputs/rgb-oriented/Dataset-1/input_manifest.json')));
const frozenInputHash = sha(join(bundle, 'inputs/rgb-oriented/Dataset-1/input_manifest.json'));
const binaryPly = () => {
  const header = Buffer.from('ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty double x\nproperty double y\nproperty double z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n');
  const vertices = Buffer.alloc(72), face = Buffer.alloc(13);
  [[0, 0, 0], [1, 0, 0], [0, 1, 0]].flat().forEach((x, i) => vertices.writeDoubleLE(x, i * 8));
  face.writeUInt8(3, 0); [0, 1, 2].forEach((x, i) => face.writeInt32LE(x, 1 + i * 4));
  return Buffer.concat([header, vertices, face]);
};

function syntheticRun(home) {
  const run = join(home, 'synthetic'); mkdirSync(run, { recursive: true });
  const dense = join(run, 'dense.mvs'), cloud = join(run, 'dense.ply'),
    depth = join(run, 'depth000.dmap'), depthManifest = join(run, 'depth-manifest.json'),
    scene = join(run, 'scene.mvs'), surface = join(run, 'mesh.mvs'), obj = join(run, 'mesh.obj'),
    mesh = join(run, 'mesh.ply');
  for (const path of [dense, cloud, depth, scene, surface, obj]) writeFileSync(path, 'synthetic test artifact');
  write(depthManifest, { 'depth000.dmap': sha(depth) });
  writeFileSync(mesh, binaryPly());
  const profile = join(run, 'worker.sb'); writeFileSync(profile, 'synthetic test profile');
  const imageId = manifest.images[0].image_id;
  write(depthManifest, { 'depth000.dmap': { path: depth, sha256: sha(depth), image_id: imageId } });
  const coveragePrepare = join(run, 'coverage-prepare.json'), coverage = join(run, 'coverage-dense.json');
  const images = Object.fromEntries(manifest.images.map(entry => [entry.image_id, {
    input_sha256: entry.sha256, sfm_ingested: true, matching_candidate: true,
    registered: entry.image_id === imageId,
    undistorted_sha256: entry.image_id === imageId ? 'synthetic' : null,
    depth_maps: entry.image_id === imageId ? [{ path: depth, sha256: sha(depth) }] : [],
    omissions: entry.image_id === imageId ? [] : ['not_registered_by_sfm', 'not_undistorted', 'not_selected_for_depth'],
  }]));
  const coverageValue = { policy: 'fixed-p0-full-input-v1', input_manifest_sha256: frozenInputHash,
    input_count: 224, sfm_input_count: 224, matching_candidate_count: 224,
    registered_count: 1, undistorted_count: 1, depth_image_count: 1, depth_map_count: 1, images };
  write(coveragePrepare, coverageValue); write(coverage, coverageValue);
  const hashes = Object.fromEntries([dense, cloud, depth, depthManifest, scene, surface, obj].map(path => [path, sha(path)]));
  hashes[coveragePrepare] = sha(coveragePrepare); hashes[coverage] = sha(coverage);
  const p0 = join(here, 'p0.json'), tools = join(here, 'tools.py');
  const release = join(root, 'out/usegeo_mesh_benchmark/paper-readiness-20260918/release-v1');
  const evaluatorRoot = join(root, 'out/usegeo_mesh_benchmark/paper-readiness-20260918/clean-runtime');
  const evaluator = join(evaluatorRoot, 'bin/python');
  const releaseSource = join(release, 'source/code/usegeo_mesh_benchmark/benchmark.py');
  write(join(run, 'launch.json'), { program_path: p0, program_bytes_sha256: sha(p0),
    program_content_sha256: 'synthetic-content-hash', input_manifest_sha256: frozenInputHash,
    frozen_input_manifest_sha256: frozenInputHash,
    input_image_ids: manifest.images.map(entry => entry.image_id).sort(),
    isolation_profile_sha256: sha(profile),
    bindings: { [p0]: sha(p0), [tools]: sha(tools), [releaseSource]: sha(releaseSource),
      [evaluator]: sha(evaluator) }, input_hashes: {}, symlinks: {},
    offline_validator: { release_root: release, evaluator_runtime_root: evaluatorRoot,
      evaluator_base_prefix: '/nonexistent', release_source_tree_sha256: 'synthetic-release',
      evaluator_runtime_tree_sha256: 'synthetic-runtime' } });
  write(join(run, 'run.json'), { status: 'sealed', scope: 'development', returncode: 0,
    cancellation_reason: null, frozen_bindings_unchanged: true, benchmark_eligible: false });
  write(join(run, 'result.json'), { status: 'sealed', scope: 'development', delivery: 'terminal',
    calls: 3, program_sha256: 'synthetic-content-hash', state: {
      stage: 'mesh', scene_id: 'Dataset-1', track: 'rgb-oriented', input_image_count: 224,
      registered_image_count: 1, undistorted_image_count: 1,
      input_manifest_sha256: frozenInputHash, depth_map_count: 1, depth_image_count: 1,
      scene, dense, dense_cloud: cloud, depth_manifest: depthManifest,
      mesh_scene: surface, native_obj: obj, coverage_path: coverage,
      coverage_prepare_path: coveragePrepare, artifact_hashes: hashes,
      mesh: { path: mesh, sha256: sha(mesh) } } });
  return { run, mesh };
}

function exportRun(run, expectedHash) {
  return spawnSync(python, ['-B', join(here, 'export.py'), run,
    '--expected-launch-sha256', expectedHash], { encoding: 'utf8' });
}

test('external supervisor kills children left by a successful leader', () => {
  const home = mkdtempSync(join(tmpdir(), 'geopilot-supervise-'));
  try {
    const script = `import subprocess,sys\np=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\nopen(sys.argv[1],'w').write(str(p.pid))`;
    const run = spawnSync(python, ['-B', join(here, 'supervise.py'), home,
      python, '-B', '-c', script, join(home, 'child.pid')], { encoding: 'utf8', timeout: 10000 });
    assert.equal(run.status, 1, run.stderr);
    assert.match(JSON.parse(readFileSync(join(home, 'supervision.json'))).reason, /CHILD_SURVIVED_LEADER/);
    const pid = Number(readFileSync(join(home, 'child.pid')));
    const listing = spawnSync('/bin/ps', ['-p', String(pid), '-o', 'state='], { encoding: 'utf8' });
    assert.ok(!listing.stdout.trim() || listing.stdout.trim().startsWith('Z'), 'child survived');
  } finally { rmSync(home, { recursive: true, force: true }); }
});

test('registered P0 rejects an extra numerical option', () => {
  const home = mkdtempSync(join(tmpdir(), 'geopilot-p0-'));
  try {
    assert.doesNotThrow(() => assertRegisteredP0(join(here, 'p0.json')));
    const altered = JSON.parse(readFileSync(join(here, 'p0.json')));
    altered.nodes[1].parameters.iters = 6;
    const path = join(home, 'p0.json'); write(path, altered);
    assert.throws(() => assertRegisteredP0(path), /Registered fixed P0 changed/);
  } finally { rmSync(home, { recursive: true, force: true }); }
});

test('candidate rejects incomplete offline provenance; evidence stays outside sandbox', () => {
  // Synthetic plumbing evidence only, never an actual model proposal or research result.
  const home = mkdtempSync(join(tmpdir(), 'geopilot-candidate-'));
  try {
    const p0 = JSON.parse(readFileSync(join(here, 'p0.json'))), parent = makeGenome(p0);
    const p1 = structuredClone(p0); p1.program_id = 'p1'; p1.parent_id = 'p0';
    p1.nodes[2].parameters.decimate = 0.5;
    p1.nodes[1].success = 'check';
    p1.nodes.push({ id: 'check', kind: 'check', metric: 'dense_points', op: 'gt', value: 1000,
      success: 'mesh', failure: 'alternate', unknown: 'mesh' },
    { ...structuredClone(p0.nodes[2]), id: 'alternate' });
    const context = { parent_sha256: digest(parent), signal_id: 'test', group_ids: ['test'], evidence_ids: ['test'] };
    const proposal = { contract_id: CONTRACT_ID, parent_sha256: digest(parent), signal_id: 'test',
      group_id: 'test', declared_surface: SKILL, patch: { parent_genome_id: parent.genome_id,
        evidence_refs: ['test'], hypothesis: 'test only', expected_effect: 'test only', risks: ['test only'],
        operations: [{ op: 'upsert_skill', name: SKILL, value: {
          description: parent.skills[0].description, content: JSON.stringify(p1) } }] } };
    const values = { 'parent-genome.json': parent, 'context.json': context, 'p1.json': p1,
      'proposal.json': proposal, 'request.json': { test_only: true, provider_payload_sha256: '1'.repeat(64),
        context: { evidence_bindings: { [join(here, 'p0.json')]: sha(join(here, 'p0.json')) } } },
      'response.json': { status: 'ok', decision: { program: p1 }, model: 'MiniMax-M3', response_model: 'MiniMax-M3',
        request_sha256: '1'.repeat(64), usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } } };
    for (const [name, value] of Object.entries(values)) write(join(home, name), value);
    const manifest = { schema: 'geopilot-candidate/1',
      parent_program_sha256: '619cc3f49b06951fffc7f9ad2f19e45fe7da810596e074c0db58f64130736b8f' };
    for (const [key, name] of Object.entries({ program: 'p1.json', parent_genome: 'parent-genome.json',
      context: 'context.json', proposal: 'proposal.json', request: 'request.json', response: 'response.json' })) {
      manifest[key + '_sha256'] = sha(join(home, name));
    }
    const path = join(home, 'candidate.json'); write(path, manifest);
    assert.throws(() => verifyCandidate(path, join(home, 'p1.json'), python), /Offline evidence verification failed/);
    const worker = join(home, 'online'); mkdirSync(worker);
    const profile = makeProfile(worker, python);
    const denied = spawnSync('/usr/bin/sandbox-exec', ['-f', profile, python, '-B', '-c',
      'import sys\ntry: open(sys.argv[1]).read(); sys.exit(1)\nexcept PermissionError: pass', join(home, 'request.json')]);
    assert.equal(denied.status, 0);
    const response = { status: 'ok', decision: { program: p0 } }; write(join(home, 'response.json'), response);
    assert.throws(() => verifyCandidate(path, join(home, 'p1.json'), python), /hash mismatch/);
    manifest.response_sha256 = sha(join(home, 'response.json')); write(path, manifest);
    assert.throws(() => verifyCandidate(path, join(home, 'p1.json'), python), /actual model response/);
  } finally { rmSync(home, { recursive: true, force: true }); }
});

test('frozen bundle lock rejects a self-consistent substitute input manifest', () => {
  const home = mkdtempSync(join(tmpdir(), 'geopilot-input-'));
  try {
    const lock = JSON.parse(readFileSync(join(root, 'out/usegeo_benchmark/prepared/v1.lock.json')));
    const expected = lock.scenes['Dataset-1'].input_manifest_sha256['rgb-oriented'];
    assert.equal(expected, frozenInputHash);
    assert.doesNotThrow(() => assertFrozenManifest(join(bundle, 'inputs/rgb-oriented/Dataset-1/input_manifest.json'), expected));
    const substitute = structuredClone(manifest);
    substitute.images[0].sha256 = '0'.repeat(64);
    const path = join(home, 'input_manifest.json'); write(path, substitute);
    assert.throws(() => assertFrozenManifest(path, expected), /Frozen Dataset-1/);
  } finally { rmSync(home, { recursive: true, force: true }); }
});

test('numerical startup and base runtime are frozen with the worker dependencies', () => {
  const cli = join(here, 'upstream/dist/src/cli.js');
  const sources = runtimeBindings(python, join(here, 'p0.json'), cli, true);
  const info = JSON.parse(spawnSync(python, ['-B', '-c',
    'import _virtualenv,json,sys; print(json.dumps({"prefix":sys.prefix,"base_prefix":sys.base_prefix,"startup":_virtualenv.__file__}))'],
    { encoding: 'utf8' }).stdout);
  const startupPth = join(info.prefix, 'lib/python3.10/site-packages/_virtualenv.pth');
  assert.equal(sources.bindings[startupPth], sha(startupPth));
  assert.equal(sources.bindings[info.startup], sha(info.startup));
  assert.ok(Object.keys(sources.bindings).some(path => path.startsWith(info.base_prefix + '/')));
});

test('OpenMVS depth header binds one licensed image without loading the map', () => {
  const home = mkdtempSync(join(tmpdir(), 'geopilot-depth-'));
  try {
    const path = join(home, 'depth0000.dmap');
    const name = Buffer.from('undistorted/images/' + manifest.images[0].image_id);
    const header = Buffer.alloc(30); header.write('DR', 0); header.writeUInt16LE(name.length, 28);
    writeFileSync(path, Buffer.concat([header, name, Buffer.from([1, 2, 3])]));
    const script = `import sys;sys.path.insert(0,${JSON.stringify(here)});from tools import depth_image_id;print(depth_image_id(sys.argv[1]))`;
    const result = spawnSync(python, ['-B', '-c', script, path], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout.trim(), manifest.images[0].image_id);
  } finally { rmSync(home, { recursive: true, force: true }); }
});

test('poisoned home and foreign session/config remain outside the sandbox', () => {
  const home = mkdtempSync(join(tmpdir(), 'geopilot-isolation-'));
  try {
    const output = join(realpathSync(home), 'output'), foreign = join(realpathSync(home), 'foreign');
    mkdirSync(output); mkdirSync(foreign);
    const allowed = join(foreign, 'allowed.jpg'), reference = join(foreign, 'reference.json'),
      config = join(foreign, 'config.toml'), session = join(foreign, 'session.jsonl');
    for (const file of [allowed, reference, config, session]) writeFileSync(file, 'private');
    const profile = makeProfile(output, '/usr/bin/python3', [allowed]);
    assert.ok(!readFileSync(profile, 'utf8').includes(`(subpath "${foreign}")`));
    const previous = process.env.HOME;
    process.env.HOME = foreign;
    try {
      assert.equal(probeIsolation(profile, output, '/usr/bin/python3', [allowed],
        [reference, config, session], false).status, 'PASS');
    } finally { process.env.HOME = previous; }
  } finally { rmSync(home, { recursive: true, force: true }); }
});

test('exact-file adapter import succeeds while private files remain denied', () => {
  const home = mkdtempSync(join(tmpdir(), 'geopilot-adapter-'));
  try {
    const output = join(realpathSync(home), 'output'), privateDir = join(realpathSync(home), 'private');
    mkdirSync(output); mkdirSync(privateDir);
    const reference = join(privateDir, 'reference.json'), personal = join(privateDir, 'personal.json');
    writeFileSync(reference, 'reference'); writeFileSync(personal, 'personal');
    const profile = makeProfile(output, python);
    assert.equal(probeIsolation(profile, output, python, [], [reference, personal]).status, 'PASS');
  } finally { rmSync(home, { recursive: true, force: true }); }
});

test('post-run exporter passes frozen validator only for a sealed regular binary mesh', () => {
  const home = mkdtempSync(join(tmpdir(), 'geopilot-export-'));
  try {
    let { run } = syntheticRun(join(home, 'positive'));
    const good = exportRun(run, sha(join(run, 'launch.json')));
    assert.equal(good.status, 0, good.stderr);
    assert.equal(JSON.parse(readFileSync(join(run, 'run.json'))).benchmark_eligible, true);
    assert.equal(JSON.parse(readFileSync(join(run, 'validation.json'))).returncode, 0);
    assert.deepEqual(new Set(['mesh.ply', 'submission.json']),
      new Set(['mesh.ply', 'submission.json'].filter(name => existsSync(join(run, 'submission', name)))));
    // Synthetic exporter coverage; actual model provenance is tested at verifyCandidate.
    const candidateRun = syntheticRun(join(home, 'candidate')).run;
    const candidateLaunchPath = join(candidateRun, 'launch.json');
    const candidateLaunch = JSON.parse(readFileSync(candidateLaunchPath));
    const candidateProgram = join(candidateRun, 'p1.json');
    write(candidateProgram, { synthetic: true });
    candidateLaunch.program_path = candidateProgram;
    candidateLaunch.program_bytes_sha256 = sha(candidateProgram);
    candidateLaunch.bindings[candidateProgram] = sha(candidateProgram);
    candidateLaunch.candidate = { program_sha256: sha(candidateProgram),
      parent_program_sha256: '619cc3f49b06951fffc7f9ad2f19e45fe7da810596e074c0db58f64130736b8f',
      bindings: { [candidateProgram]: sha(candidateProgram) } };
    write(candidateLaunchPath, candidateLaunch);
    const candidateResultPath = join(candidateRun, 'result.json');
    write(candidateResultPath, { ...JSON.parse(readFileSync(candidateResultPath)), calls: 4, delivery: 'recovered' });
    const candidateExport = exportRun(candidateRun, sha(candidateLaunchPath));
    assert.equal(candidateExport.status, 0, candidateExport.stderr);
    assert.match(JSON.parse(readFileSync(join(candidateRun, 'submission/submission.json'))).configuration_id,
      /^geopilot-rsih-candidate-/);
    const cases = [
      ['failed', ({ run }) => { const path = join(run, 'run.json'), value = JSON.parse(readFileSync(path)); value.status = 'failed'; writeFileSync(path, JSON.stringify(value)); }],
      ['missing', ({ mesh }) => rmSync(mesh)],
      ['symlink', ({ mesh, run }) => { rmSync(mesh); symlinkSync(join(run, 'dense.ply'), mesh); }],
      ['changed', ({ mesh }) => writeFileSync(mesh, Buffer.concat([binaryPly(), Buffer.from('x')]))],
      ['scene', ({ run }) => { const path = join(run, 'result.json'), value = JSON.parse(readFileSync(path)); value.state.scene_id = 'Dataset-2'; writeFileSync(path, JSON.stringify(value)); }],
      ['track', ({ run }) => { const path = join(run, 'result.json'), value = JSON.parse(readFileSync(path)); value.state.track = 'rgb'; writeFileSync(path, JSON.stringify(value)); }],
      ['source-drift', ({ run }) => { const source = join(run, 'source.txt'); writeFileSync(source, 'before'); const path = join(run, 'launch.json'), value = JSON.parse(readFileSync(path)); value.bindings[source] = sha(source); writeFileSync(path, JSON.stringify(value)); writeFileSync(source, 'after'); }],
      ['numerical-startup-drift', ({ run }) => { const source = join(run, '_virtualenv.pth'); writeFileSync(source, 'before'); const path = join(run, 'launch.json'), value = JSON.parse(readFileSync(path)); value.bindings[source] = sha(source); writeFileSync(path, JSON.stringify(value)); writeFileSync(source, 'after'); }],
      ['evaluator-drift', ({ run }) => { const path = join(run, 'launch.json'), value = JSON.parse(readFileSync(path)); const source = join(value.offline_validator.release_root, 'source/code/usegeo_mesh_benchmark/benchmark.py'); value.bindings[source] = '0'.repeat(64); writeFileSync(path, JSON.stringify(value)); }],
      ['coverage-missing', ({ run }) => { const path = join(run, 'coverage-dense.json'), value = JSON.parse(readFileSync(path)); delete value.images[manifest.images[1].image_id]; writeFileSync(path, JSON.stringify(value)); const resultPath = join(run, 'result.json'), result = JSON.parse(readFileSync(resultPath)); result.state.artifact_hashes[path] = sha(path); writeFileSync(resultPath, JSON.stringify(result)); }],
      ['malformed', ({ mesh, run }) => { writeFileSync(mesh, 'ply\n'); const path = join(run, 'result.json'), value = JSON.parse(readFileSync(path)); value.state.mesh.sha256 = sha(mesh); writeFileSync(path, JSON.stringify(value)); }],
    ];
    for (const [name, mutate] of cases) {
      const directory = join(home, name); mkdirSync(directory);
      const sample = syntheticRun(directory); mutate(sample);
      const result = exportRun(sample.run, sha(join(sample.run, 'launch.json')));
      assert.notEqual(result.status, 0, name);
      assert.equal(JSON.parse(readFileSync(join(sample.run, 'run.json'))).benchmark_eligible, false, name);
      assert.equal(existsSync(join(sample.run, 'submission')), false, name);
    }
  } finally { rmSync(home, { recursive: true, force: true }); }
});

test('sandbox launch provenance tampering fails parent pin and standalone export', () => {
  const home = mkdtempSync(join(tmpdir(), 'geopilot-launch-pin-'));
  try {
    const { run } = syntheticRun(realpathSync(home));
    rmSync(join(run, 'worker.sb'));
    const profile = makeProfile(run, '/usr/bin/python3');
    const launchPath = join(run, 'launch.json');
    const original = JSON.parse(readFileSync(launchPath));
    original.isolation_profile_sha256 = sha(profile);
    write(launchPath, original);
    const trustedBytes = readFileSync(launchPath);
    const missing = spawnSync(python, ['-B', join(here, 'export.py'), run], { encoding: 'utf8' });
    assert.notEqual(missing.status, 0);
    assert.equal(JSON.parse(readFileSync(join(run, 'run.json'))).benchmark_eligible, false);

    for (const [name, script] of [
      ['bindings', "const v=JSON.parse(fs.readFileSync(p));delete v.bindings[Object.keys(v.bindings)[0]];fs.writeFileSync(p,JSON.stringify(v))"],
      ['input-ids', "const v=JSON.parse(fs.readFileSync(p));v.input_image_ids=['wrong'];fs.writeFileSync(p,JSON.stringify(v))"],
      ['replacement', "fs.renameSync(p,p+'.old');fs.writeFileSync(p,fs.readFileSync(p+'.old'))"],
      ['deletion', 'fs.unlinkSync(p)'],
    ]) {
      writeFileSync(launchPath, trustedBytes);
      const expected = sha(launchPath), originalStat = lstatSync(launchPath);
      const worker = spawnSync('/usr/bin/sandbox-exec', ['-f', profile, process.execPath,
        '-e', `const fs=require('node:fs'),p=process.argv[1];${script}`, launchPath],
        { cwd: run, encoding: 'utf8', env: { PATH: '/usr/bin:/bin', HOME: run,
          TMPDIR: run, OPENSSL_CONF: '/dev/null' } });
      assert.equal(worker.status, 0, `${name}: ${worker.stderr}`);
      assert.equal(launchRecordUnchanged(launchPath, expected, originalStat), false, name);
      if (name !== 'replacement') {
        const exported = exportRun(run, expected);
        assert.notEqual(exported.status, 0, name);
      }
      assert.equal(JSON.parse(readFileSync(join(run, 'run.json'))).benchmark_eligible, false, name);
      assert.equal(existsSync(join(run, 'submission')), false, name);
    }
  } finally { rmSync(home, { recursive: true, force: true }); }
});
