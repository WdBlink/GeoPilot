// All receipts, prompts, model labels and decisions below are SYNTHETIC fixtures.
// No agent, API, numerical worker, launcher or evaluator is invoked.
import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, readFileSync, writeFileSync, rmSync, existsSync, symlinkSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { importNativeProposal, verifyNativeCandidate, NATIVE_SCHEMA } from './native_proposal.mjs';
import { makeGenome, digest, REGISTRY, DIAGNOSTICS, LIMITS, CONTRACT_ID } from '../../code/geopilot_rsih/program.mjs';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const read = path => JSON.parse(readFileSync(path, 'utf8'));
const write = (path, value) => writeFileSync(path, JSON.stringify(value, null, 2) + '\n');
const p0 = read(new URL('../../code/geopilot_rsih/p0.json', import.meta.url));
function fixture(t, mode = 'single_mesh_parameter') {
  const dir = mkdtempSync(join(tmpdir(), 'geopilot-native-synthetic-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const paths = Object.fromEntries(['prompt', 'pack', 'response', 'receipt', 'parentProgram', 'parentGenome', 'editContract']
    .map(role => [role, join(dir, role + '.json')]));
  const program = structuredClone(p0); program.program_id = 'native_fixture_1'; program.parent_id = 'p0';
  program.nodes[2].parameters.decimate = 0.5;
  if (mode === 'conditional_post_prepare') {
    program.nodes[1].success = 'diagnose';
    program.nodes.push({ id: 'diagnose', kind: 'check', metric: 'dense_points', op: 'gt', value: 1000,
      success: 'mesh', failure: 'alternate', unknown: 'mesh' },
    { ...structuredClone(p0.nodes[2]), id: 'alternate' });
  }
  writeFileSync(paths.prompt, 'SYNTHETIC fixture. No native call took place. Return the specified decision.\n');
  write(paths.pack, { parent: p0, registry: REGISTRY, allowed_diagnostics: DIAGNOSTICS, limits: LIMITS,
    evidence: { 'synthetic-evidence': { kind: 'synthetic_fixture', content: 'Invented diagnostic, not a measurement.' } } });
  write(paths.response, { hypothesis: 'Synthetic hypothesis', expected_effect: 'Synthetic expectation; no result',
    risks: ['Fixture risk'], competing_explanations: ['Fixture alternative'], evidence_refs: ['synthetic-evidence'], program });
  write(paths.parentProgram, p0);
  const genome = makeGenome(p0); genome.model = { profile: 'geopilot-driver', id: 'no-model-calls' };
  write(paths.parentGenome, genome);
  write(paths.editContract, { contract_id: CONTRACT_ID, parent_program_sha256: digest(p0),
    candidate_id: program.program_id, mode });
  const receipt = { provider: 'native_codex', evidence_kind: 'synthetic_fixture', batch_id: 'synthetic-batch',
    agent_id: 'synthetic-agent-not-a-real-call', thread_id: null, turn_id: null,
    started_at: '2026-09-25T00:00:00Z', completed_at: '2026-09-25T00:00:01Z',
    visible_config: { model: 'synthetic-model-label', reasoning_effort: null },
    backend_model: null, token_usage: null, seed: null, temperature: null, source_sha256: {} };
  const seal = () => {
    receipt.source_sha256 = Object.fromEntries(Object.entries(paths).filter(([role]) => role !== 'receipt')
      .map(([role, path]) => [role, sha(readFileSync(path))]));
    write(paths.receipt, receipt);
  };
  seal();
  const options = { ...paths, output: join(dir, 'imported') };
  return { dir, paths, receipt, seal, options, program,
    mutate: (role, fn) => { const v = read(paths[role]); fn(v); write(paths[role], v); } };
}
function rejected(f, pattern) {
  const source = readFileSync(f.paths.response);
  assert.throws(() => importNativeProposal(f.options), pattern);
  assert.equal(existsSync(join(f.options.output, 'candidate.json')), false);
  const record = read(join(f.options.output, 'rejected.json'));
  assert.equal(record.status, 'rejected'); assert.equal(record.execution_status, 'not_launched');
  assert.deepEqual(readFileSync(join(f.options.output, 'raw-response.txt')), source);
  assert.deepEqual(readFileSync(f.paths.response), source);
}

for (const mode of ['single_mesh_parameter', 'conditional_post_prepare']) {
  test(`synthetic ${mode}: exact response, upstream patch, native receipt and deterministic replay`, t => {
    const f = fixture(t, mode), before = Object.fromEntries(Object.entries(f.paths).map(([r, p]) => [r, sha(readFileSync(p))]));
    const result = importNativeProposal(f.options);
    assert.equal(result.candidate.schema, NATIVE_SCHEMA);
    assert.equal(result.candidate.evidence_kind, 'synthetic_fixture');
    assert.equal(result.candidate.execution_status, 'not_launched');
    assert.equal(result.candidate.benchmark_eligible, false);
    assert.deepEqual(read(join(f.options.output, 'program.json')), f.program);
    assert.deepEqual(JSON.parse(read(join(f.options.output, 'proposal.json')).patch.operations[0].value.content), f.program);
    assert.deepEqual(readFileSync(join(f.options.output, 'raw-response.txt')), readFileSync(f.paths.response));
    const source = read(join(f.options.output, 'source.json'));
    assert.equal(source.token_usage, null); assert.equal(source.backend_model, null);
    assert.equal(source.authenticity, 'caller_supplied_receipt_not_platform_authenticated');
    assert.equal(existsSync(join(f.options.output, 'request.json')), false);
    assert.deepEqual(verifyNativeCandidate(f.options.output, result.candidate_sha256), result.candidate);
    const repeat = importNativeProposal({ ...f.options, output: join(f.dir, 'second-import') });
    assert.equal(repeat.candidate_sha256, result.candidate_sha256);
    for (const [role, path] of Object.entries(f.paths)) assert.equal(sha(readFileSync(path)), before[role]);
  });
}

test('explicit evidence catalogs bind nested IDs, reject ambiguity and ignore program node IDs', t => {
  for (const refs of [['K1'], ['K1', 'P0']]) {
    const f = fixture(t);
    f.mutate('pack', v => {
      v.evidence['domain-knowledge'] = [{ id: 'K1', content: 'Synthetic knowledge' }];
      v.evidence['program-history'] = { cases: [{ id: 'P0', program: p0 }] };
    });
    f.mutate('response', v => { v.evidence_refs = refs; }); f.seal();
    const result = importNativeProposal(f.options);
    assert.deepEqual(read(join(f.options.output, 'context.json')).evidence_ids,
      ['synthetic-evidence', 'domain-knowledge', 'program-history', 'K1', 'P0']);
    assert.deepEqual(verifyNativeCandidate(f.options.output, result.candidate_sha256), result.candidate);
  }
  for (const [catalog, refs, pattern] of [
    [[{ id: 'K1' }, { id: 'K1' }], ['K1'], /evidence IDs/],
    [[{ id: 'synthetic-evidence' }], ['synthetic-evidence'], /evidence IDs/],
    [[{ id: '' }], ['synthetic-evidence'], /evidence IDs/],
    [[{}], ['synthetic-evidence'], /evidence IDs/],
    [{ id: 'K1' }, ['K1'], /Invalid evidence catalog/],
    [[{ id: 'K1', program: p0 }], ['prepare'], /Unbound evidence/],
  ]) {
    const f = fixture(t);
    f.mutate('pack', v => { v.evidence['domain-knowledge'] = catalog; });
    f.mutate('response', v => { v.evidence_refs = refs; }); f.seal(); rejected(f, pattern);
  }
});

test('CLI performs an import only, using clearly synthetic files', t => {
  const f = fixture(t);
  const args = Object.entries(f.options).flatMap(([key, value]) => ['--' + key.replace(/[A-Z]/g, c => '-' + c.toLowerCase()), value]);
  const result = spawnSync(process.execPath, [new URL('./native_proposal.mjs', import.meta.url).pathname, ...args], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).evidence_kind, 'synthetic_fixture');
});

test('unexposed inherited native model configuration stays null without invented identifiers', t => {
  const f = fixture(t);
  f.receipt.visible_config = { model: null, reasoning_effort: null }; f.seal();
  const result = importNativeProposal(f.options);
  assert.deepEqual(read(join(f.options.output, 'receipt.json')).visible_config,
    { model: null, reasoning_effort: null });
  assert.equal(verifyNativeCandidate(f.options.output, result.candidate_sha256).evidence_kind, 'synthetic_fixture');
});

for (const [name, body] of [
  ['duplicate key', '{"a":1,"a":2}'], ['escaped duplicate key', '{"a":1,"\\u0061":2}'],
  ['overflow', '{"a":1e400}'], ['NaN', '{"a":NaN}'], ['infinity', '{"a":Infinity}'],
  ['unsafe integer', '{"a":9007199254740993}'], ['unsafe exponent integer', '{"a":1e16}'],
  ['fenced JSON', '```json\n{}\n```'], ['trailing prose', '{}\nexplanation'],
  ['truncated JSON', '{"program":'], ['invalid UTF-8', Buffer.from([0xff, 0xfe])],
]) {
  test(`raw response rejects ${name} without repair and preserves original bytes`, t => {
    const f = fixture(t); writeFileSync(f.paths.response, body); f.seal(); rejected(f, /Strict JSON|Nonfinite|unsafe|encoded data/i);
  });
}

for (const [name, role, mutate, pattern] of [
  ['changed parent', 'parentProgram', v => { v.nodes[2].parameters.decimate = 0.5; }, /registered P0/],
  ['hidden parent capability', 'parentGenome', v => { v.tools = ['shell']; }, /genome envelope/],
  ['new registry', 'pack', v => { v.registry.prepare.parameters.properties.binding = { type: 'string' }; }, /registry drift/],
  ['wrong contract', 'editContract', v => { v.contract_id = 'future-unregistered'; }, /contract\/parent/],
  ['unknown scope', 'editContract', v => { v.mode = 'shared_intrinsics'; }, /edit scope/],
  ['wrong candidate identity', 'response', v => { v.program.program_id = 'wrong_id'; }, /identity/],
  ['wrong parent identity', 'response', v => { v.program.parent_id = 'p2'; }, /identity/],
  ['no executable change', 'response', v => { v.program.nodes[2].parameters.decimate = 0.25; }, /Exactly one/],
  ['two mesh edits', 'response', v => { v.program.nodes[2].parameters.smooth = 3; }, /Exactly one/],
  ['non-mesh edit', 'response', v => { v.program.nodes[1].parameters.iters = 2; }, /Only a mesh/],
  ['prepare edit', 'response', v => { v.program.nodes[0].parameters.binding = 'shared'; }, /./],
  ['graph edit', 'response', v => { v.program.nodes[2].failure = 'mesh'; }, /Cycle/],
  ['unbound evidence', 'response', v => { v.evidence_refs = ['not-supplied']; }, /Unbound evidence/],
  ['undeclared response field', 'response', v => { v.repair = true; }, /unexpected\/missing/],
]) {
  test(`rejects ${name}`, t => { const f = fixture(t); f.mutate(role, mutate); f.seal(); rejected(f, pattern); });
}

test('conditional scope rejects ceremonial branches and prepare failure edits', t => {
  const f = fixture(t, 'conditional_post_prepare');
  f.mutate('response', v => { v.program.nodes.at(-1).parameters.decimate = 0.5; }); f.seal();
  rejected(f, /different executable branches/);
  const g = fixture(t, 'conditional_post_prepare');
  g.mutate('response', v => {
    v.program.nodes[0].failure = 'other_stop'; v.program.nodes.push({ id: 'other_stop', kind: 'stop' });
  }); g.seal(); rejected(g, /Prepare mutation/);
});

test('conditional scope retains the repeated-densify prohibition', t => {
  const f = fixture(t, 'conditional_post_prepare');
  f.mutate('response', v => {
    v.program.nodes[1].success = 'again';
    v.program.nodes.push({ id: 'again', kind: 'tool', action: 'densify', parameters: {}, success: 'diagnose', failure: 'end' });
  }); f.seal(); rejected(f, /Repeated densify/);
});

for (const [name, change] of [
  ['API provider', r => { r.provider = 'MiniMax-M3'; }], ['invented zero usage', r => { r.token_usage = 0; }],
  ['invented usage object', r => { r.token_usage = { total_tokens: 0 }; }],
  ['invented backend', r => { r.backend_model = 'MiniMax-M3'; }], ['invented seed', r => { r.seed = 917; }],
  ['API request ID', r => { r.request_id = 'fake'; }], ['missing agent', r => { r.agent_id = ''; }],
  ['reversed time', r => { r.completed_at = '2026-09-24T00:00:00Z'; }],
  ['nonexistent calendar date', r => { r.started_at = '2026-02-30T00:00:00Z'; }],
]) {
  test(`receipt rejects ${name}`, t => { const f = fixture(t); change(f.receipt); f.seal(); rejected(f, /./); });
}

test('receipt binds all original files; drift is not silently resealed', t => {
  const f = fixture(t); writeFileSync(f.paths.prompt, 'Changed after the receipt was recorded');
  rejected(f, /Source hash drift: prompt/);
});

test('duplicate keys in context/receipt/embedded parent content are also rejected', t => {
  for (const role of ['pack', 'receipt', 'parentGenome']) {
    const f = fixture(t);
    if (role === 'parentGenome') f.mutate(role, v => { v.skills[0].content = v.skills[0].content.replace('"entry":', '"entry":"bad","entry":'); });
    else writeFileSync(f.paths[role], '{"x":1,"x":2}');
    if (role !== 'receipt') f.seal();
    rejected(f, /Duplicate JSON key/);
  }
});

test('no overwrite, output/source overlap, symlink and nonregular file safety', t => {
  const f = fixture(t), first = importNativeProposal(f.options);
  assert.throws(() => importNativeProposal(f.options), /EEXIST/);
  assert.equal(sha(readFileSync(join(f.options.output, 'candidate.json'))), first.candidate_sha256);
  assert.throws(() => importNativeProposal({ ...f.options, output: f.dir }), /overlaps/);
  assert.throws(() => importNativeProposal({ ...f.options, output: f.paths.response }), /overlaps/);
  const link = join(f.dir, 'response-link'); symlinkSync(f.paths.response, link);
  assert.throws(() => importNativeProposal({ ...f.options, response: link, output: join(f.dir, 'new') }), /symlink/);
  mkdirSync(join(f.dir, 'not-a-file'));
  assert.throws(() => importNativeProposal({ ...f.options, response: join(f.dir, 'not-a-file'), output: join(f.dir, 'bad-dir') }), /Unsafe/);
  const outputLink = join(f.dir, 'output-link'); symlinkSync(f.options.output, outputLink);
  assert.throws(() => importNativeProposal({ ...f.options, output: outputLink }), /EEXIST/);
  assert.throws(() => verifyNativeCandidate(outputLink, first.candidate_sha256), /symlink/);
});

test('oversize source is refused before parsing', t => {
  const f = fixture(t); writeFileSync(f.paths.response, Buffer.alloc(4 * 1024 * 1024 + 1, 32));
  assert.throws(() => importNativeProposal(f.options), /oversize/);
  assert.equal(read(join(f.options.output, 'rejected.json')).status, 'rejected');
});

test('verification catches artifact, original-source and manifest drift', t => {
  for (const target of ['program.json', 'raw-response.txt', 'original', 'candidate.json']) {
    const f = fixture(t), result = importNativeProposal(f.options);
    const path = target === 'original' ? f.paths.prompt : join(f.options.output, target);
    writeFileSync(path, readFileSync(path, 'utf8') + '\n');
    assert.throws(() => verifyNativeCandidate(f.options.output, result.candidate_sha256), /drift/);
  }
  const f = fixture(t), result = importNativeProposal(f.options);
  assert.throws(() => verifyNativeCandidate(f.options.output, null), /previously recorded/);
  const path = join(f.options.output, 'program.json'), bytes = readFileSync(path);
  rmSync(path); writeFileSync(join(f.dir, 'alias'), bytes); symlinkSync(join(f.dir, 'alias'), path);
  assert.throws(() => verifyNativeCandidate(f.options.output, result.candidate_sha256), /ELOOP/);
});

// Broader post-prepare scope does not require ceremonial conditional nodes.
test('post_prepare_program accepts an unconditional real change and rejects prepare mutation', t => {
  const f = fixture(t, 'post_prepare_program');
  const imported = importNativeProposal(f.options);
  assert.equal(verifyNativeCandidate(f.options.output, imported.candidate_sha256).status, 'imported');
  assert.equal(read(join(f.options.output, 'program.json')).nodes.some(n => n.kind === 'check'), false);
  assert.equal(read(join(f.options.output, 'context.json')).experiment_mode, 'post_prepare_program');
  const g = fixture(t, 'post_prepare_program');
  g.mutate('response', v => { v.program.nodes[0].failure = 'densify'; });
  g.seal();
  rejected(g, /Incompatible stage|Prepare mutation/);
});
