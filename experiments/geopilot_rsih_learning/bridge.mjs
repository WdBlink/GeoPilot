import { readFileSync, writeFileSync } from 'node:fs';
import { applyProgramPatch, getProgram, validateLearningProgram, digest, REGISTRY, DIAGNOSTICS, LIMITS } from '../../code/geopilot_rsih/program.mjs';

const [parentPath, proposalPath, contextPath, outputPath] = process.argv.slice(2);
try {
  const parent = JSON.parse(readFileSync(parentPath));
  if (proposalPath === '--inspect') {
    const program = getProgram(parent);
    console.log(JSON.stringify({ program, program_hash: digest(program), genome_hash: digest(parent),
      registry: REGISTRY, diagnostics: DIAGNOSTICS, limits: LIMITS })); process.exit();
  }
  if (proposalPath === '--hash') { console.log(digest(parent)); process.exit(); }
  const proposal = JSON.parse(readFileSync(proposalPath));
  const context = JSON.parse(readFileSync(contextPath));
  const result = applyProgramPatch(parent, proposal, context);
  validateLearningProgram(getProgram(result.genome), { requireCondition: context.experiment_mode !== 'single_parameter' });
  writeFileSync(outputPath, JSON.stringify({ program: getProgram(result.genome), ...result }, null, 2) + '\n', { flag: 'wx' });
} catch (error) {
  console.error(error.message); process.exitCode = 1;
}
