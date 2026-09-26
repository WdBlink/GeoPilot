import { appendFileSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { executeProgram, getProgram, digest } from './program.mjs';

const exec = promisify(execFile);
const here = dirname(fileURLToPath(import.meta.url));

export default function geopilot(pi) {
  // Both command and model tool routes share the same registered, single-use handler.
  const tool = {
    name: 'geopilot_execute', label: 'Execute frozen GeoPilot program',
    description: 'Execute the already-bound reconstruction program. No paths or code accepted.',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
    executionMode: 'sequential' as const,
    async execute(_id, params, signal) {
      if (Object.keys(params).length) throw new Error('Unexpected tool arguments');
      const request = JSON.parse(readFileSync(process.env.GEOPILOT_REQUEST!, 'utf8'));
      const genome = JSON.parse(readFileSync(request.genome_path, 'utf8'));
      if (digest(genome) !== request.genome_sha256) throw new Error('Genome identity mismatch');
      writeFileSync(join(request.output, 'execution-started.json'), JSON.stringify({
        runtime: 'RSI-Harness/Pi', genome_sha256: request.genome_sha256,
      }), { flag: 'wx' });
      const record = event => appendFileSync(join(request.output, 'events.jsonl'),
        JSON.stringify({ ...event, utc: new Date().toISOString() }) + '\n');
      const result = await executeProgram(getProgram(genome), async (node, parent) => {
        const directory = join(request.output, 'nodes', node.id);
        mkdirSync(directory, { recursive: true });
        const input = join(directory, 'request.json');
        writeFileSync(input, JSON.stringify({ node, parent, inputs: request.inputs,
          output: directory, scope: request.scope }), { flag: 'wx' });
        const started = Date.now();
        const argv = ['-B', join(here, 'tools.py'), input];
        try {
          const response = await exec(request.python, argv, {
            cwd: directory, signal, maxBuffer: 1024 * 1024,
            env: { PATH: process.env.PATH!, HOME: request.output, TMPDIR: directory,
              OPENSSL_CONF: '/dev/null' },
          });
          writeFileSync(join(directory, 'tool.stdout'), response.stdout, { flag: 'wx' });
          writeFileSync(join(directory, 'tool.stderr'), response.stderr, { flag: 'wx' });
          return JSON.parse(readFileSync(join(directory, 'state.json'), 'utf8'));
        } catch (error) {
          writeFileSync(join(directory, 'tool-error.json'), JSON.stringify({
            code: error.code ?? null, stdout: error.stdout ?? '', stderr: error.stderr ?? '',
          }), { flag: 'wx' });
          throw error;
        } finally {
          record({ event: 'numerical_call', node: node.id, argv: [request.python, ...argv],
            wall_seconds: (Date.now() - started) / 1000 });
        }
      }, record, request.output);
      result.scope = request.scope;
      writeFileSync(join(request.output, 'result.json'), JSON.stringify(result, null, 2), { flag: 'wx' });
      return { content: [{ type: 'text', text: JSON.stringify(result) }], details: result };
    },
  };
  pi.registerTool(tool);
  pi.registerCommand('geopilot-run', {
    description: 'Run the frozen conditional program without a model decision.',
    handler: async (args) => {
      if (args.trim()) throw new Error('Command accepts no arguments');
      const result = await tool.execute('geopilot-command', {}, undefined);
      console.log(result.content[0].text);
      if (result.details.status !== 'sealed') process.exitCode = 1;
    },
  });
}
