import { createHash } from 'node:crypto';
import { readFileSync, lstatSync, realpathSync, openSync, readSync, closeSync } from 'node:fs';
import { relative, isAbsolute, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import {
  assertJsonSchema, createHarnessGenome, createHarnessPatchSchema,
  validateHarnessPatch, applyHarnessPatch, assertHarnessComplexity, contentHash,
} from './upstream/dist/src/index.js';

const object = properties => ({ type: 'object', properties,
  required: Object.keys(properties), additionalProperties: false });
const text = { type: 'string', minLength: 1 };
const number = { type: 'number' };
const bounded = (type, minimum, maximum) => ({ type, minimum, maximum });
const options = properties => ({ type: 'object', properties, additionalProperties: false });
export const REGISTRY = Object.freeze({
  prepare: { input: ['input'], output: 'sparse', parameters: options({}) },
  densify: { input: ['sparse', 'dense', 'mesh'], output: 'dense', parameters: options({
    'resolution-level': bounded('integer', 0, 2), 'number-views': bounded('integer', 2, 20),
    'number-views-fuse': bounded('integer', 2, 5), 'iters': bounded('integer', 1, 6),
    'geometric-iters': bounded('integer', 1, 4),
    'fusion-depth-diff-threshold': bounded('number', 0.001, 0.05),
  }) },
  mesh: { input: ['dense', 'mesh'], output: 'mesh', parameters: options({
    'min-point-distance': bounded('number', 0, 5), 'free-space-support': bounded('integer', 0, 1),
    'remove-spurious': bounded('number', 0, 40), 'close-holes': bounded('integer', 0, 60),
    'smooth': bounded('integer', 0, 5), 'decimate': bounded('number', 0.25, 1),
  }) },
  refine: { input: ['mesh'], output: 'mesh', parameters: options({
    'resolution-level': bounded('integer', 0, 2), 'max-face-area': bounded('integer', 16, 128),
  }) },
});
export const DIAGNOSTICS = Object.freeze(['registered_fraction', 'sparse_points',
  'mean_reprojection_error', 'dense_points', 'mesh_vertices', 'mesh_faces']);
export const LIMITS = Object.freeze({ max_nodes: 32, max_calls: 16 });
export const SCHEMA = 'geopilot-condition-program/2';
export const SKILL = 'geopilot-reconstruction-program';

/** Reject lossy JSON before passing it through upstream cloning/hashing. */
export function safeJson(value) {
  if (typeof value === 'number' && (!Number.isFinite(value) ||
      (Number.isInteger(value) && !Number.isSafeInteger(value)))) throw new Error('Nonfinite/unsafe number');
  if (value !== null && typeof value === 'object') {
    if (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype) throw new Error('Non-JSON object');
    for (const child of Object.values(value)) safeJson(child);
  } else if (!['string', 'number', 'boolean'].includes(typeof value) && value !== null) {
    throw new Error('Non-JSON value');
  }
  return value;
}
export function digest(value) {
  safeJson(value);
  return contentHash(value);
}
export const CONTRACT_ID = digest({ schema: SCHEMA, registry: REGISTRY, diagnostics: DIAGNOSTICS, limits: LIMITS });

/** Domain semantics not provided by Genome: strict DAG and stage types on every branch. */
export function validateProgram(p) {
  safeJson(p);
  assertJsonSchema(p, object({ schema: { enum: [SCHEMA] }, contract_id: { enum: [CONTRACT_ID] },
    program_id: text, parent_id: {},
    entry: text, nodes: { type: 'array', minItems: 1, maxItems: LIMITS.max_nodes,
      items: { type: 'object' } } }));
  if (!/^[a-z][a-z0-9_-]{0,63}$/.test(p.program_id) ||
      !(p.parent_id === null || (typeof p.parent_id === 'string' && /^[a-z][a-z0-9_-]{0,63}$/.test(p.parent_id)))) {
    throw new Error('Invalid program identity');
  }
  const nodes = new Map();
  for (const n of p.nodes) {
    if (!/^[a-z][a-z0-9_]{0,63}$/.test(n.id) || nodes.has(n.id)) throw new Error('Invalid/duplicate node ID');
    const base = { id: text, kind: { enum: ['tool', 'check', 'stop'] } };
    if (n.kind === 'tool') {
      if (!Object.hasOwn(REGISTRY, n.action)) throw new Error('unsupported_action');
      assertJsonSchema(n, object({ ...base, action: { enum: Object.keys(REGISTRY) },
        parameters: REGISTRY[n.action].parameters, success: text, failure: text }));
    } else if (n.kind === 'check') {
      assertJsonSchema(n, object({ ...base, metric: { enum: DIAGNOSTICS },
        op: { enum: ['lt', 'le', 'gt', 'ge'] }, value: number, success: text, failure: text, unknown: text }));
    } else if (n.kind === 'stop') assertJsonSchema(n, object(base));
    else throw new Error('Unknown node kind');
    nodes.set(n.id, n);
  }
  const visited = new Set(), states = new Set();
  function visit(id, stage, path = new Set(), calls = 0) {
    if (!nodes.has(id)) throw new Error('Missing successor');
    if (path.has(id)) throw new Error('Cycle');
    const key = `${id}:${stage}:${calls}`;
    if (states.has(key)) return;
    states.add(key);
    const n = nodes.get(id); visited.add(id);
    if (n.kind === 'stop') return;
    if (n.kind === 'tool' && !REGISTRY[n.action].input.includes(stage)) throw new Error('Incompatible stage');
    const nextCalls = calls + (n.kind === 'tool' ? 1 : 0);
    if (nextCalls > LIMITS.max_calls) throw new Error('Call limit');
    const nextPath = new Set(path).add(id);
    visit(n.success, n.kind === 'tool' ? REGISTRY[n.action].output : stage, nextPath, nextCalls);
    visit(n.failure, stage, nextPath, nextCalls);
    if (n.kind === 'check') visit(n.unknown, stage, nextPath, nextCalls);
  }
  visit(p.entry, 'input');
  if (visited.size !== nodes.size) throw new Error('Unreachable node');
  return p;
}

export function makeGenome(p) {
  validateProgram(p);
  return createHarnessGenome({ genome_id: `geopilot:${digest(p)}`,
    skills: [{ name: SKILL, description: 'Frozen conditional reconstruction program', content: JSON.stringify(p) }],
    resources: { isolate: true } });
}

/** First learning round: shared depth workspace permits one densify per path. */
export function validateLearningProgram(p, { requireCondition = true } = {}) {
  validateProgram(p);
  const nodes = new Map(p.nodes.map(n => [n.id, n])), visited = new Set();
  function visit(id, dense) {
    const key = `${id}:${dense}`;
    if (visited.has(key)) return;
    visited.add(key);
    const n = nodes.get(id);
    if (n.action === 'densify' && dense) throw new Error('Repeated densify shares immutable depth artifacts');
    if (n.kind === 'stop') return;
    const next = dense || n.action === 'densify';
    for (const successor of new Set([n.success, n.failure, n.unknown].filter(Boolean))) visit(successor, next);
  }
  visit(p.entry, false);
  // Compare behavior, not node labels: duplicated identical branches are ceremonial.
  const cache = new Map();
  function behavior(id) {
    if (cache.has(id)) return cache.get(id);
    const { id: ignored, success, failure, unknown, ...n } = nodes.get(id);
    const value = digest({ ...n, ...(success ? { success: behavior(success), failure: behavior(failure) } : {}),
      ...(unknown ? { unknown: behavior(unknown) } : {}) });
    cache.set(id, value); return value;
  }
  if (requireCondition && !p.nodes.some(n => n.kind === 'check' && behavior(n.success) !== behavior(n.failure))) {
    throw new Error('No condition with different executable branches');
  }
  return p;
}
function readProgramContent(content) {
  const helper = join(dirname(fileURLToPath(import.meta.url)), 'strict_json.py');
  const result = spawnSync(process.env.GEOPILOT_PYTHON ?? '/usr/bin/python3', ['-B', helper, '-'],
    { input: content, encoding: 'utf8', maxBuffer: 1024 * 1024 });
  if (result.status !== 0) throw new Error('Invalid program JSON: ' + result.stderr);
  return validateProgram(JSON.parse(result.stdout));
}
export function getProgram(genome) {
  const skills = genome.skills?.filter(s => s.name === SKILL) ?? [];
  if (skills.length !== 1) throw new Error('Expected one reconstruction program');
  return readProgramContent(skills[0].content);
}

/** Use upstream upsert_skill, restricting it to the one program-bearing skill. */
export function applyProgramPatch(parent, proposal, context) {
  safeJson(parent); safeJson(proposal); safeJson(context);
  getProgram(parent);
  const schema = object({ contract_id: { enum: [CONTRACT_ID] }, parent_sha256: text,
    signal_id: text, group_id: text, declared_surface: { enum: [SKILL] },
    patch: createHarnessPatchSchema(['upsert_skill']) });
  assertJsonSchema(proposal, schema);
  if (proposal.parent_sha256 !== digest(parent) || context.parent_sha256 !== digest(parent) ||
      context.signal_id !== proposal.signal_id || !context.group_ids.includes(proposal.group_id)) {
    throw new Error('Parent or current signal mismatch');
  }
  const patch = proposal.patch;
  if (patch.parent_genome_id !== parent.genome_id || !patch.evidence_refs?.length ||
      patch.evidence_refs.some(id => !context.evidence_ids.includes(id))) throw new Error('Unbound evidence');
  if (!patch.hypothesis.trim() || !patch.expected_effect.trim() || patch.operations.length !== 1) throw new Error('Invalid proposal');
  const op = patch.operations[0];
  if (op.name !== SKILL) throw new Error('Forbidden skill');
  assertJsonSchema(op.value, object({ description: text, content: text }));
  if (op.value.description !== parent.skills.find(s => s.name === SKILL).description) throw new Error('Undeclared description change');
  const currentProgram = getProgram(parent), proposedProgram = readProgramContent(op.value.content);
  if (proposedProgram.parent_id !== currentProgram.program_id || proposedProgram.program_id === currentProgram.program_id) {
    throw new Error('Program parent/version mismatch');
  }
  validateHarnessPatch(patch, { allowedOperations: ['upsert_skill'] });
  const next = applyHarnessPatch(parent, patch);
  assertHarnessComplexity(next);
  const behavior = p => ({ entry: p.entry, nodes: p.nodes });
  if (digest(behavior(currentProgram)) === digest(behavior(proposedProgram))) throw new Error('No executable change');
  // Upstream materializes absent empty collections. They are deterministic framework bookkeeping.
  const expected = applyHarnessPatch(parent, { ...patch, operations: [{ ...op,
    value: { ...op.value, content: parent.skills.find(s => s.name === SKILL).content } }] });
  const unchanged = g => Object.fromEntries(Object.entries(g).filter(([k]) => !['skills', 'genome_id'].includes(k)));
  if (digest(unchanged(next)) !== digest(unchanged(expected))) throw new Error('Undeclared Genome change');
  return { genome: next, parent_sha256: digest(parent), genome_sha256: digest(next),
    program_sha256: digest(getProgram(next)), proposal_sha256: digest(proposal) };
}

/** An executor receives immutable parent state; only successful stages advance it. */
export async function executeProgram(p, executeTool, record = () => {}, artifactRoot = null) {
  validateProgram(p);
  const fileHash = path => {
    const h = createHash('sha256'), fd = openSync(path, 'r'), block = Buffer.allocUnsafe(8 * 1024 * 1024);
    try { for (let n; (n = readSync(fd, block, 0, block.length, null)) > 0;) h.update(block.subarray(0, n)); }
    finally { closeSync(fd); }
    return h.digest('hex');
  };
  function verifyArtifacts(state) {
    const hashes = { ...(state.artifact_hashes ?? {}) };
    if (state.mesh) hashes[state.mesh.path] = state.mesh.sha256;
    for (const [path, hash] of Object.entries(hashes)) {
      if (!lstatSync(path).isFile() || lstatSync(path).isSymbolicLink()) throw new Error('Unsafe artifact');
      if (artifactRoot) {
        const r = relative(realpathSync(artifactRoot), realpathSync(path));
        if (r.startsWith('..') || isAbsolute(r)) throw new Error('Artifact outside run');
      }
      if (fileHash(path) !== hash) throw new Error('Artifact hash mismatch');
    }
  }
  const nodes = new Map(p.nodes.map(n => [n.id, n]));
  let id = p.entry, state = { stage: 'input', diagnostics: {}, mesh: null }, failed = false;
  let calls = 0;
  while (true) {
    const n = nodes.get(id);
    if (n.kind === 'stop') {
      verifyArtifacts(state);
      const result = { status: state.mesh ? 'sealed' : 'failed',
        delivery: state.mesh ? (failed ? 'recovered' : 'terminal') : 'failed',
        node: id, calls, state, program_sha256: digest(p) };
      await record({ event: 'terminal', ...result });
      return result;
    }
    if (n.kind === 'check') {
      const d = state.diagnostics[n.metric];
      const v = d?.value;
      let branch = 'unknown';
      if (d?.status === 'valid' && typeof v === 'number' && Number.isFinite(v)) {
        const pass = { lt: v < n.value, le: v <= n.value, gt: v > n.value, ge: v >= n.value }[n.op];
        branch = pass ? 'success' : 'failure';
      }
      await record({ event: 'decision', node: id, branch, successor: n[branch],
        program_sha256: digest(p), rule: { metric: n.metric, op: n.op, value: n.value },
        diagnostic: d ?? { status: 'missing', value: null } });
      id = n[branch]; continue;
    }
    const parent = structuredClone(state), parentHash = digest(parent);
    verifyArtifacts(parent);
    await record({ event: 'tool_started', node: id, action: n.action, parameters: n.parameters, parent_sha256: parentHash });
    calls++;
    let outcome;
    try {
      outcome = await executeTool(structuredClone(n), structuredClone(parent));
      safeJson(outcome);
      if (outcome.stage !== REGISTRY[n.action].output || !outcome.diagnostics ||
          !Object.hasOwn(outcome, 'mesh')) throw new Error('Invalid tool state');
      if (REGISTRY[n.action].output === 'mesh' && !outcome.mesh) throw new Error('Missing mesh');
      verifyArtifacts(outcome);
    } catch (error) {
      failed = true;
      verifyArtifacts(parent);
      await record({ event: 'tool_failed', node: id, error: String(error.message), parent_sha256: parentHash });
      id = n.failure; continue;
    }
    verifyArtifacts(parent);
    state = structuredClone(outcome); failed = false;
    await record({ event: 'tool_finished', node: id, state_sha256: digest(state) });
    id = n.success;
  }
}
