// Offline adoption check: exact upstream sources and the built Pi/Genome session flow.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createHarnessGenome } from './upstream/dist/src/index.js';

const root = dirname(fileURLToPath(import.meta.url));
const lock = JSON.parse(readFileSync(join(root, 'upstream-lock.json'), 'utf8'));
for (const [name, expected] of Object.entries(lock.files_sha256)) {
  const actual = createHash('sha256').update(readFileSync(join(root, 'upstream', name))).digest('hex');
  assert.equal(actual, expected, `Upstream source drift: ${name}`);
}
const directory = mkdtempSync(join(tmpdir(), 'geopilot-rsih-check-'));
try {
  const config = join(directory, 'providers.json');
  const genome = join(directory, 'genome.json');
  writeFileSync(config, JSON.stringify({
    mock: { kind: 'mock', model: 'mock-adoption', mock_responses: ['runtime ok'] },
  }));
  writeFileSync(genome, JSON.stringify(createHarnessGenome({
    genome_id: 'harness:geopilot-adoption-check',
    model: { profile: 'mock', id: 'mock-adoption' },
    resources: { isolate: true },
  })));
  const agentDirectory = join(directory, 'agent');
  for (const [index, prompt] of ['First.', 'Second.'].entries()) {
    const result = spawnSync(process.execPath, [
      join(root, 'upstream', 'dist', 'src', 'cli.js'),
      '--config', config, '--cwd', directory, '--run-id', 'adoption-check',
      '--no-tools', '--no-extensions', '--no-skills', '--no-prompt-templates', '--no-themes',
      ...(index === 0 ? ['--genome', genome] : []), '-p', prompt,
    ], {
      cwd: directory, encoding: 'utf8', input: '', timeout: 30000,
      env: {
        PATH: process.env.PATH,
        RSIH_CODING_AGENT_DIR: agentDirectory, PI_CODING_AGENT_DIR: agentDirectory,
        PI_OFFLINE: '1', PI_SKIP_VERSION_CHECK: '1',
      },
    });
    assert.ifError(result.error);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout.trim(), 'runtime ok');
  }
  const sessions = readdirSync(join(agentDirectory, 'sessions'), { recursive: true })
    .filter(name => name.endsWith('.jsonl'));
  assert.equal(sessions.length, 1);
  const entries = readFileSync(join(agentDirectory, 'sessions', sessions[0]), 'utf8')
    .trim().split('\n').map(line => JSON.parse(line));
  assert.deepEqual(entries.filter(e => e.type === 'message' && e.message.role === 'user')
    .map(e => e.message.content[0].text), ['First.', 'Second.']);
  const genomes = entries.filter(e => e.type === 'custom' && e.customType === 'rsih.genome');
  assert.equal(genomes.length, 1);
  assert.equal(genomes[0].data.genome_id, 'harness:geopilot-adoption-check');
  console.log(JSON.stringify({ status: 'PASS', commit: lock.commit,
    source_files: Object.keys(lock.files_sha256).length,
    built_cli_turns: 2, restored_genome: true, model: 'mock',
    scope: 'upstream runtime adoption; not GeoPilot domain acceptance' }, null, 2));
} finally {
  rmSync(directory, { recursive: true, force: true });
}
