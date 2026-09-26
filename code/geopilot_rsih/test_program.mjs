import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { SCHEMA, CONTRACT_ID, SKILL, validateProgram, validateLearningProgram, makeGenome, getProgram,
  applyProgramPatch, executeProgram, digest } from './program.mjs';
import { launch, strictRead } from './launch.mjs';

export const p0 = () => ({ schema: SCHEMA, contract_id: CONTRACT_ID, program_id: 'p0', parent_id: null, entry: 'prepare', nodes: [
  { id: 'prepare', kind: 'tool', action: 'prepare', parameters: {}, success: 'dense', failure: 'end' },
  { id: 'dense', kind: 'tool', action: 'densify', parameters: {}, success: 'mesh', failure: 'end' },
  { id: 'mesh', kind: 'tool', action: 'mesh', parameters: {}, success: 'end', failure: 'end' },
  { id: 'end', kind: 'stop' },
] });

test('closed schemas, all branch types, DAG and finite numbers', () => {
  validateProgram(p0());
  const changes = [
    p => { p.nodes[0].success = 'absent'; },
    p => { p.nodes[0].failure = 'mesh'; },
    p => { p.nodes[1].success = 'prepare'; },
    p => { p.nodes[1].parameters = { shell: 'echo' }; },
    p => { p.nodes[1].parameters = { iters: true }; },
    p => { p.nodes[1].parameters = { iters: NaN }; },
    p => { p.nodes[1].action = '__proto__'; },
    p => { p.nodes[1].action = 'unregistered_sfm'; },
    p => { p.nodes[1].id = p.nodes[0].id; },
    p => { p.nodes.push({ id: 'orphan', kind: 'stop' }); },
  ];
  for (const change of changes) { const p = p0(); change(p); assert.throws(() => validateProgram(p)); }
});

test('frozen node/call limits and stable upstream content identity', () => {
  const p = p0();
  p.entry = 'c0';
  p.nodes = Array.from({ length: 31 }, (_, i) => ({ id: 'c' + i, kind: 'check',
    metric: 'mesh_faces', op: 'gt', value: 0, success: i === 30 ? 'end' : 'c' + (i + 1),
    failure: 'end', unknown: 'end' })).concat([{ id: 'end', kind: 'stop' }]);
  validateProgram(p);
  p.nodes.push({ id: 'excess', kind: 'stop' });
  assert.throws(() => validateProgram(p));
  const calls = p0(); calls.nodes = [calls.nodes[0], ...Array.from({ length: 16 }, (_, i) => ({
    id: i === 0 ? 'dense' : 'd' + i, kind: 'tool', action: 'densify', parameters: {},
    success: i === 15 ? 'end' : 'd' + (i + 1), failure: 'end',
  })), { id: 'end', kind: 'stop' }];
  assert.throws(() => validateProgram(calls), /Call limit/);
  assert.equal(digest({ a: 1, b: 2 }), digest({ b: 2, a: 1 }));
});

test('learning candidates need meaningful branches and cannot repeat shared densify', () => {
  const p = p0();
  p.nodes[1].success = 'check';
  p.nodes.push({ id: 'check', kind: 'check', metric: 'dense_points', op: 'gt', value: 1000,
    success: 'mesh', failure: 'alternate', unknown: 'mesh' },
  { ...structuredClone(p.nodes[2]), id: 'alternate' });
  assert.throws(() => validateLearningProgram(p), /different executable branches/);
  p.nodes.at(-1).parameters = { decimate: 0.5 };
  validateLearningProgram(p);
  p.nodes[1].success = 'again';
  p.nodes.push({ id: 'again', kind: 'tool', action: 'densify', parameters: {}, success: 'check', failure: 'end' });
  assert.throws(() => validateLearningProgram(p), /Repeated densify/);
});

test('actual upstream skill patch, parent immutability, evidence and permissions', () => {
  const parent = makeGenome(p0()), original = JSON.stringify(parent), p = p0();
  p.program_id = 'p1'; p.parent_id = 'p0';
  p.nodes[2].parameters = { decimate: 0.5 };
  const context = { parent_sha256: digest(parent), signal_id: 's1', group_ids: ['g1'], evidence_ids: ['e1'] };
  const proposal = { contract_id: CONTRACT_ID, parent_sha256: digest(parent), signal_id: 's1',
    group_id: 'g1', declared_surface: SKILL, patch: { parent_genome_id: parent.genome_id,
      evidence_refs: ['e1'], hypothesis: 'Smaller mesh may reduce evaluation cost',
      expected_effect: 'Fewer faces, geometry quality unproven', risks: ['Surface detail loss'],
      operations: [{ op: 'upsert_skill', name: SKILL, value: {
        description: parent.skills[0].description, content: JSON.stringify(p),
      } }] } };
  const result = applyProgramPatch(parent, proposal, context);
  assert.deepEqual(getProgram(result.genome), p);
  assert.equal(JSON.stringify(parent), original);
  assert.equal(result.genome.parent_id, parent.genome_id);
  for (const mutate of [
    x => { x.parent_sha256 = 'bad'; }, x => { x.signal_id = 'old'; },
    x => { x.patch.operations[0].name = 'other'; },
    x => { x.patch.operations[0].op = 'set_model'; },
    x => { x.patch.operations[0].value.extra = 1; },
    x => { x.patch.evidence_refs = ['missing']; },
    x => { x.patch.operations[0].value.content = x.patch.operations[0].value.content.replace('"entry":', '"entry":"bad","entry":'); },
    x => { const bad = p0(); bad.nodes[0].action = 'mesh'; x.patch.operations[0].value.content = JSON.stringify(bad); },
  ]) {
    const bad = structuredClone(proposal); mutate(bad);
    assert.throws(() => applyProgramPatch(parent, bad, context));
  }
});

test('unknown diagnostic branch, terminal selection and failure recovery', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'geopilot-graph-'));
  try {
    const path = join(directory, 'mesh.ply'); writeFileSync(path, 'immutable geometric fixture');
    const mesh = { path, sha256: createHash('sha256').update(readFileSync(path)).digest('hex') };
    const program = p0(); program.nodes[2].success = 'check';
    program.nodes.splice(3, 0,
      { id: 'check', kind: 'check', metric: 'mesh_faces', op: 'gt', value: 10,
        success: 'refine', failure: 'end', unknown: 'end' },
      { id: 'refine', kind: 'tool', action: 'refine', parameters: {}, success: 'end', failure: 'end' });
    for (const mode of ['unknown', 'low', 'high']) {
      const events = [];
      const result = await executeProgram(program, async (n, state) => {
        if (n.action === 'refine') { state.mesh = null; throw new Error('Numerical failure'); }
        return { stage: { prepare: 'sparse', densify: 'dense', mesh: 'mesh' }[n.action],
          mesh: n.action === 'mesh' ? mesh : null,
          diagnostics: mode === 'unknown' ? {} : { mesh_faces: { status: 'valid', value: mode === 'high' ? 100 : 2 } } };
      }, e => events.push(e));
      assert.equal(result.delivery, mode === 'high' ? 'recovered' : 'terminal');
      assert.deepEqual(result.state.mesh, mesh);
      assert.equal(events.find(e => e.event === 'decision').branch,
        { unknown: 'unknown', low: 'failure', high: 'success' }[mode]);
    }
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('strict file boundary rejects duplicate keys and overflowing numbers', () => {
  const directory = mkdtempSync(join(tmpdir(), 'geopilot-json-'));
  try {
    for (const payload of ['{"x":1,"x":2}', '{"x":1e400}', '{"x":NaN}', '{"x":9007199254740993}']) {
      const file = join(directory, 'input.json'); writeFileSync(file, payload);
      assert.throws(() => strictRead(file, '/usr/bin/python3'));
    }
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('GeoPilot entry uses actual upstream CLI, native extension and registered domain tool', () => {
  const directory = mkdtempSync(join(tmpdir(), 'geopilot-native-'));
  try {
    const file = join(directory, 'p0.json'); writeFileSync(file, JSON.stringify(p0()));
    const output = join(directory, 'run');
    const result = launch({ program: file, output, python: '/usr/bin/python3', scope: 'controlled_fixture' });
    assert.equal(result.status, 'sealed', readFileSync(join(output, 'runtime.stderr'), 'utf8') +
      (result.status === 'failed' ? readFileSync(join(output, 'events.jsonl'), 'utf8') : ''));
    assert.equal(result.benchmark_eligible, false);
    const events = readFileSync(join(output, 'events.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(events.filter(e => e.event === 'numerical_call').length, 3);
    const started = JSON.parse(readFileSync(join(output, 'execution-started.json')));
    assert.equal(started.runtime, 'RSI-Harness/Pi');
    assert.throws(() => launch({ program: file, output, python: '/usr/bin/python3', scope: 'controlled_fixture' }));
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
