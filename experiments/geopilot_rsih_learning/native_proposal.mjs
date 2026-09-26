/** Offline, deterministic native-Codex import. Never calls a model or the launcher.
 * CLI: node native_proposal.mjs --prompt FILE --pack FILE --response FILE
 *   --receipt FILE --parent-program FILE --parent-genome FILE --edit-contract FILE --output NEW_DIR
 * Input object examples (including the complete receipt/pack/edit shapes) are in
 * test_native_proposal.mjs. Raw response must be one JSON decision, without fences.
 * Receipt authenticity is caller-supplied; hashes do not authenticate a platform.
 */
import { constants, openSync, closeSync, fstatSync, readFileSync, writeFileSync,
  mkdirSync, realpathSync, lstatSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, basename, resolve, relative, isAbsolute, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { parseArgs, isDeepStrictEqual } from 'node:util';
import { safeJson, digest, getProgram, makeGenome, applyProgramPatch, validateLearningProgram,
  REGISTRY, DIAGNOSTICS, LIMITS, CONTRACT_ID, SKILL } from '../../code/geopilot_rsih/program.mjs';
import { assertRegisteredP0 } from '../../code/geopilot_rsih/launch.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const runtime = resolve(here, '../../code/geopilot_rsih');
const files = Object.freeze({ response: 'raw-response.txt', prompt: 'prompt.txt', pack: 'context-pack.json',
  receipt: 'receipt.json', parentProgram: 'parent-program.json', parentGenome: 'parent-genome.json',
  editContract: 'edit-contract.json' });
const derivedFiles = ['program.json', 'proposal.json', 'context.json', 'source.json'];
const maxBytes = 4 * 1024 * 1024;
export const NATIVE_SCHEMA = 'geopilot-native-candidate/1';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const encode = value => Buffer.from(JSON.stringify(value, null, 2) + '\n');
const require = (ok, message) => { if (!ok) throw new Error(message); };
const equal = (a, b, message) => require(isDeepStrictEqual(a, b), message);
const inside = (path, base) => { const r = relative(base, path); return r === '' || (r !== '..' && !r.startsWith('../') && !isAbsolute(r)); };
const text = v => typeof v === 'string' && v.trim().length > 0;
function keys(value, names, label) {
  require(value && typeof value === 'object' && !Array.isArray(value), `${label}: expected object`);
  equal(Object.keys(value).sort(), [...names].sort(), `${label}: unexpected/missing fields`);
}
function ids(value, label) {
  require(Array.isArray(value) && value.length > 0 && value.every(text) &&
    new Set(value).size === value.length, `${label}: expected unique nonempty strings`);
}
function bytesAt(path) {
  const fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const before = fstatSync(fd);
    require(before.isFile() && before.size > 0 && before.size <= maxBytes, 'Unsafe/empty/oversize source: ' + path);
    const bytes = readFileSync(fd), after = fstatSync(fd);
    require(bytes.length === before.size && before.size === after.size &&
      before.mtimeMs === after.mtimeMs && before.ctimeMs === after.ctimeMs, 'Source changed while reading: ' + path);
    return bytes;
  } finally { closeSync(fd); }
}
function strict(bytes) {
  // Reuse the repository duplicate-key/nonfinite/unsafe-integer parser on the captured bytes.
  const input = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  const result = spawnSync('/usr/bin/python3', ['-B', join(runtime, 'strict_json.py'), '-'],
    { input, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024, timeout: 10000 });
  require(result.status === 0, 'Strict JSON rejected: ' + (result.stderr || result.error?.message || 'parse failure'));
  return safeJson(JSON.parse(result.stdout));
}
function canonicalSource(path) {
  require(text(path), 'Missing source path');
  const absolute = resolve(path);
  // Canonicalize directory aliases (e.g. macOS /tmp), but never follow a file symlink.
  require(!lstatSync(absolute).isSymbolicLink(), 'Source symlink forbidden: ' + path);
  return join(realpathSync(dirname(absolute)), basename(absolute));
}
function checkScope(parent, proposed, edit) {
  require(proposed.program_id === edit.candidate_id && proposed.parent_id === 'p0', 'Wrong program identity/parent');
  validateLearningProgram(proposed, { requireCondition: edit.mode === 'conditional_post_prepare' });
  if (edit.mode === 'single_mesh_parameter') {
    const a = structuredClone(parent), b = structuredClone(proposed);
    delete a.program_id; delete a.parent_id; delete b.program_id; delete b.parent_id;
    require(a.nodes.length === b.nodes.length, 'Graph mutation forbidden');
    let changes = 0;
    for (let i = 0; i < a.nodes.length; i++) {
      if (a.nodes[i].action !== 'mesh') continue;
      const old = a.nodes[i].parameters, next = b.nodes[i].parameters ?? {};
      for (const key of new Set([...Object.keys(old), ...Object.keys(next)])) {
        if (!isDeepStrictEqual(old[key], next[key])) changes++;
      }
      b.nodes[i].parameters = structuredClone(old);
    }
    equal(a, b, 'Only a mesh parameter may change');
    require(changes === 1, 'Exactly one mesh parameter must change');
  } else {
    const prepares = proposed.nodes.filter(n => n.action === 'prepare');
    require(proposed.entry === parent.entry && prepares.length === 1, 'Prepare/entry mutation forbidden');
    const before = { ...parent.nodes[0] }, after = { ...prepares[0] };
    delete before.success; delete after.success;
    equal(before, after, 'Prepare mutation forbidden');
  }
}
function compile(captured) {
  const parsed = Object.fromEntries(Object.entries(captured).filter(([key]) => key !== 'prompt')
    .map(([key, source]) => [key, strict(source.bytes)]));
  require(text(new TextDecoder('utf-8', { fatal: true }).decode(captured.prompt.bytes)), 'Empty prompt');
  const { receipt, pack, parentProgram: parent, parentGenome: genome, editContract: edit, response: decision } = parsed;
  keys(receipt, ['provider', 'evidence_kind', 'batch_id', 'agent_id', 'thread_id', 'turn_id', 'started_at',
    'completed_at', 'visible_config', 'backend_model', 'token_usage', 'seed', 'temperature', 'source_sha256'], 'receipt');
  require(receipt.provider === 'native_codex', 'Only native_codex receipts accepted');
  require(['native_call', 'synthetic_fixture'].includes(receipt.evidence_kind), 'Unknown evidence kind');
  require(text(receipt.batch_id) && text(receipt.agent_id), 'Missing native batch/agent identity');
  for (const key of ['thread_id', 'turn_id']) require(receipt[key] === null || text(receipt[key]), 'Invalid visible identity');
  for (const key of ['started_at', 'completed_at']) {
    const value = receipt[key];
    require(typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(value) &&
      Number.isFinite(Date.parse(value)) && new Date(value).toISOString() ===
        (value.includes('.') ? value : value.replace('Z', '.000Z')), 'Invalid UTC receipt time');
  }
  require(Date.parse(receipt.completed_at) >= Date.parse(receipt.started_at), 'Receipt time order invalid');
  keys(receipt.visible_config, ['model', 'reasoning_effort'], 'visible_config');
  // Record only publicly exposed values; inherited host settings may be unknown.
  for (const value of Object.values(receipt.visible_config)) {
    require(value === null || text(value), 'Invalid visible model config');
  }
  for (const key of ['backend_model', 'token_usage', 'seed', 'temperature']) {
    require(receipt[key] === null, 'Unexposed backend/usage/seed/temperature must be null');
  }
  const sourceRoles = Object.keys(files).filter(key => key !== 'receipt');
  keys(receipt.source_sha256, sourceRoles, 'source_sha256');
  for (const role of sourceRoles) equal(receipt.source_sha256[role], hash(captured[role].bytes), 'Source hash drift: ' + role);

  assertRegisteredP0(join(runtime, 'p0.json'));
  const p0 = strict(bytesAt(join(runtime, 'p0.json')));
  equal(parent, p0, 'Only the registered P0 parent is accepted');
  equal(getProgram(genome), p0, 'Parent genome/program mismatch');
  const expectedGenome = makeGenome(p0), normalized = structuredClone(genome);
  if (Object.hasOwn(genome, 'model')) expectedGenome.model = { profile: 'geopilot-driver', id: 'no-model-calls' };
  normalized.skills[0].content = expectedGenome.skills[0].content;
  equal(normalized, expectedGenome, 'Unregistered parent genome envelope');

  keys(edit, ['contract_id', 'parent_program_sha256', 'candidate_id', 'mode'], 'edit contract');
  require(edit.contract_id === CONTRACT_ID && edit.parent_program_sha256 === digest(p0), 'Edit contract/parent mismatch');
  require(text(edit.candidate_id) && edit.candidate_id !== 'p0', 'Invalid candidate identity');
  require(['single_mesh_parameter', 'conditional_post_prepare', 'post_prepare_program'].includes(edit.mode), 'Unregistered edit scope');
  keys(pack, ['parent', 'registry', 'allowed_diagnostics', 'limits', 'evidence'], 'context pack');
  equal(pack.parent, p0, 'Pack parent mismatch'); equal(pack.registry, REGISTRY, 'Pack registry drift');
  equal(pack.allowed_diagnostics, DIAGNOSTICS, 'Pack diagnostics drift'); equal(pack.limits, LIMITS, 'Pack limits drift');
  require(pack.evidence && typeof pack.evidence === 'object' && !Array.isArray(pack.evidence), 'Invalid inline evidence');
  const evidenceIds = Object.keys(pack.evidence);
  // Only these explicit source catalogs declare nested evidence IDs; program node IDs do not.
  for (const [section, field] of [['domain-knowledge', null], ['program-history', 'cases']]) {
    if (!Object.hasOwn(pack.evidence, section)) continue;
    const entries = field === null ? pack.evidence[section] : pack.evidence[section]?.[field];
    require(Array.isArray(entries), `Invalid evidence catalog: ${section}`);
    evidenceIds.push(...entries.map(entry => entry?.id));
  }
  ids(evidenceIds, 'evidence IDs');
  // Evidence is inline JSON: paths mentioned inside it are data, never files to execute/read.
  keys(decision, ['hypothesis', 'expected_effect', 'risks', 'competing_explanations', 'evidence_refs', 'program'], 'decision');
  require(text(decision.hypothesis) && text(decision.expected_effect), 'Empty hypothesis/effect');
  for (const key of ['risks', 'competing_explanations', 'evidence_refs']) ids(decision[key], key);
  require(decision.evidence_refs.every(id => evidenceIds.includes(id)), 'Unbound evidence reference');
  checkScope(p0, decision.program, edit);
  const context = { parent_sha256: digest(genome), signal_id: receipt.batch_id,
    group_ids: [receipt.batch_id], evidence_ids: evidenceIds,
    experiment_mode: edit.mode === 'single_mesh_parameter' ? 'single_parameter' :
      (edit.mode === 'post_prepare_program' ? 'post_prepare_program' : 'conditional') };
  const proposal = { contract_id: CONTRACT_ID, parent_sha256: context.parent_sha256,
    signal_id: context.signal_id, group_id: receipt.batch_id, declared_surface: SKILL,
    patch: { parent_genome_id: genome.genome_id, evidence_refs: decision.evidence_refs,
      hypothesis: decision.hypothesis, expected_effect: decision.expected_effect, risks: decision.risks,
      operations: [{ op: 'upsert_skill', name: SKILL, value: { description: genome.skills[0].description,
        content: JSON.stringify(decision.program) } }] } };
  const applied = applyProgramPatch(genome, proposal, context);
  equal(getProgram(applied.genome), decision.program, 'Patch changed raw response program');
  const source = { provider: 'native_codex', evidence_kind: receipt.evidence_kind,
    authenticity: 'caller_supplied_receipt_not_platform_authenticated',
    full_host_context: 'not_exposed', backend_model: null, token_usage: null, seed: null, temperature: null,
    implementation: Object.fromEntries([fileURLToPath(import.meta.url), join(runtime, 'program.mjs'),
      join(runtime, 'strict_json.py'), join(runtime, 'p0.json'), join(runtime, 'upstream-lock.json')]
      .map(path => [path, hash(bytesAt(path))])),
    original_inputs: Object.fromEntries(Object.entries(captured).map(([role, v]) =>
      [role, { path: v.path, sha256: hash(v.bytes) }])) };
  return { 'program.json': decision.program, 'proposal.json': proposal, 'context.json': context, 'source.json': source };
}

/** Safety/path rejection leaves existing outputs untouched; content rejection retains snapshots. */
export function importNativeProposal(options) {
  keys(options, [...Object.keys(files), 'output'], 'import options');
  const paths = Object.fromEntries(Object.keys(files).map(role => [role, canonicalSource(options[role])]));
  const output = join(realpathSync(dirname(resolve(options.output))), basename(resolve(options.output)));
  for (const protectedPath of [runtime, here, ...Object.values(paths)]) {
    require(!inside(output, protectedPath) && !inside(protectedPath, output), 'Output overlaps protected source');
  }
  mkdirSync(output); // Deliberately no recursive/exist_ok: never overwrite an earlier attempt.
  const captured = {};
  try {
    for (const [role, path] of Object.entries(paths)) {
      const bytes = bytesAt(path); captured[role] = { path, bytes };
      writeFileSync(join(output, files[role]), bytes, { flag: 'wx' });
    }
    const derived = compile(captured);
    for (const [role, v] of Object.entries(captured)) {
      equal(bytesAt(canonicalSource(v.path)), v.bytes, 'Source changed during import: ' + role);
    }
    for (const [name, value] of Object.entries(derived)) writeFileSync(join(output, name), encode(value), { flag: 'wx' });
    const artifactNames = [...Object.values(files), ...derivedFiles];
    const receipt = strict(captured.receipt.bytes);
    const candidate = { schema: NATIVE_SCHEMA, status: 'imported', benchmark_eligible: false,
      execution_status: 'not_launched', evidence_kind: receipt.evidence_kind, contract_id: CONTRACT_ID,
      program_sha256: digest(derived['program.json']),
      artifacts: Object.fromEntries(artifactNames.map(name => [name, hash(bytesAt(join(output, name)))])) };
    writeFileSync(join(output, 'candidate.json'), encode(candidate), { flag: 'wx' });
    return { output, candidate, candidate_sha256: hash(bytesAt(join(output, 'candidate.json'))) };
  } catch (error) {
    writeFileSync(join(output, 'rejected.json'), encode({ schema: NATIVE_SCHEMA, status: 'rejected',
      execution_status: 'not_launched', error: String(error.message),
      preserved_sources: Object.keys(captured).map(role => files[role]) }), { flag: 'wx' });
    throw error;
  }
}

/** Recheck using a candidate hash recorded by the caller, not a self-certified manifest. */
export function verifyNativeCandidate(directory, expectedSha256) {
  require(/^[a-f0-9]{64}$/.test(expectedSha256), 'A previously recorded candidate SHA-256 is required');
  require(!lstatSync(directory).isSymbolicLink(), 'Candidate directory symlink forbidden');
  const output = realpathSync(directory), raw = bytesAt(join(output, 'candidate.json'));
  equal(hash(raw), expectedSha256, 'Candidate hash drift');
  const candidate = strict(raw);
  keys(candidate, ['schema', 'status', 'benchmark_eligible', 'execution_status', 'evidence_kind',
    'contract_id', 'program_sha256', 'artifacts'], 'candidate');
  require(candidate.schema === NATIVE_SCHEMA && candidate.status === 'imported' &&
    candidate.benchmark_eligible === false && candidate.execution_status === 'not_launched' &&
    candidate.contract_id === CONTRACT_ID, 'Invalid native candidate');
  keys(candidate.artifacts, [...Object.values(files), ...derivedFiles], 'candidate artifacts');
  const artifacts = {};
  for (const [name, expected] of Object.entries(candidate.artifacts)) {
    artifacts[name] = bytesAt(join(output, name)); equal(hash(artifacts[name]), expected, 'Artifact hash drift: ' + name);
  }
  const source = strict(artifacts['source.json']);
  const captured = Object.fromEntries(Object.entries(files).map(([role, name]) => {
    const bound = source.original_inputs[role], path = canonicalSource(bound.path);
    equal(bytesAt(path), artifacts[name], 'Original source drift: ' + role);
    equal(hash(artifacts[name]), bound.sha256, 'Original source hash drift: ' + role);
    return [role, { path, bytes: artifacts[name] }];
  }));
  const derived = compile(captured);
  for (const name of derivedFiles) equal(strict(artifacts[name]), derived[name], 'Derived output mismatch: ' + name);
  equal(candidate.program_sha256, digest(derived['program.json']), 'Program digest mismatch');
  equal(candidate.evidence_kind, strict(captured.receipt.bytes).evidence_kind, 'Evidence kind mismatch');
  return candidate;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const names = ['prompt', 'pack', 'response', 'receipt', 'parent-program', 'parent-genome', 'edit-contract', 'output'];
    const { values } = parseArgs({ options: Object.fromEntries(names.map(name => [name, { type: 'string' }])), strict: true });
    const options = Object.fromEntries(names.map(name => [name.replace(/-([a-z])/g, (_, c) => c.toUpperCase()), values[name]]));
    const result = importNativeProposal(options);
    console.log(JSON.stringify({ status: 'imported', output: result.output, candidate_sha256: result.candidate_sha256,
      execution_status: 'not_launched', evidence_kind: result.candidate.evidence_kind }));
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
