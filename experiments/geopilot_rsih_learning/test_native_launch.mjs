/** Explicit synthetic/mock tests only. No real native receipt, reconstruction or score. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,readFileSync,existsSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {importNativeProposal} from './native_proposal.mjs';
import {nativeLaunch,strictReadManifest,validateAdmission,parentIdentity,validateLocations,probeNativeBoundary,INPUTS} from './native_launch.mjs';
import {makeGenome,digest,REGISTRY,DIAGNOSTICS,LIMITS,CONTRACT_ID} from '../../code/geopilot_rsih/program.mjs';
const ROOT=resolve(new URL('../..',import.meta.url).pathname);
const PY=join(ROOT,'out/usegeo_mesh_benchmark/paper-readiness-20260918/baseline-preflight/baseline-runtime/bin/python');
const write=(p,v)=>writeFileSync(p,JSON.stringify(v,null,2)+'\n');
const sha=p=>createHash('sha256').update(readFileSync(p)).digest('hex');
function fixture(t){
  const root=mkdtempSync(join(tmpdir(),'native-launch-synthetic-'));
  t.after(()=>rmSync(root,{recursive:true,force:true}));
  const names=['prompt','pack','response','receipt','parentProgram','parentGenome','editContract'];
  const paths=Object.fromEntries(names.map(n=>[n,join(root,n+'.json')]));
  const p0=JSON.parse(readFileSync(join(ROOT,'code/geopilot_rsih/p0.json')));
  const proposed=structuredClone(p0);proposed.program_id='synthetic_native_test';proposed.parent_id='p0';proposed.nodes[2].parameters.decimate=.5;
  writeFileSync(paths.prompt,'SYNTHETIC ONLY: no native call occurred.');
  write(paths.pack,{parent:p0,registry:REGISTRY,allowed_diagnostics:DIAGNOSTICS,limits:LIMITS,evidence:{toy:{synthetic:true}}});
  write(paths.response,{hypothesis:'Synthetic',expected_effect:'None measured',risks:['synthetic'],competing_explanations:['synthetic'],evidence_refs:['toy'],program:proposed});
  write(paths.parentProgram,p0);write(paths.parentGenome,makeGenome(p0));
  write(paths.editContract,{contract_id:CONTRACT_ID,parent_program_sha256:digest(p0),candidate_id:proposed.program_id,mode:'post_prepare_program'});
  write(paths.receipt,{provider:'native_codex',evidence_kind:'synthetic_fixture',batch_id:'synthetic-only',agent_id:'not-real',thread_id:null,turn_id:null,
    started_at:'2026-09-25T00:00:00Z',completed_at:'2026-09-25T00:00:01Z',visible_config:{model:null,reasoning_effort:null},
    backend_model:null,token_usage:null,seed:null,temperature:null,source_sha256:Object.fromEntries(names.filter(n=>n!=='receipt').map(n=>[n,sha(paths[n])]))});
  const directory=join(root,'candidate');const imported=importNativeProposal({...paths,output:directory});
  return {root,paths,directory,imported,options:{candidateDirectory:directory,expectedCandidateSha256:imported.candidate_sha256,
    output:join(root,'must-not-execute'),inputs:INPUTS,python:PY,preflightOnly:false}};
}

test('admission predicate: explicitly mocked shape, only native post-prepare accepted',()=>{
  validateAdmission({evidence_kind:'native_call'},{mode:'post_prepare_program'});
  for(const kind of ['synthetic_fixture','MiniMax-M3',undefined])assert.throws(()=>validateAdmission({evidence_kind:kind},{mode:'post_prepare_program'}),/native_call/);
  for(const mode of ['conditional_post_prepare','single_mesh_parameter'])assert.throws(()=>validateAdmission({evidence_kind:'native_call'},{mode}),/post_prepare/);
});
test('genuine synthetic import cannot reach preflight or numerical execution',t=>{
  const f=fixture(t);assert.throws(()=>nativeLaunch(f.options),/native_call/);assert.equal(existsSync(f.options.output),false);
});
test('caller expected SHA and source-byte replay reject drift before execution',t=>{
  const f=fixture(t);assert.throws(()=>nativeLaunch({...f.options,expectedCandidateSha256:'0'.repeat(64)}),/hash drift/);
  writeFileSync(f.paths.prompt,'Changed raw source');assert.throws(()=>nativeLaunch(f.options),/source drift/i);assert.equal(existsSync(f.options.output),false);
});
test('old MiniMax manifest is not reinterpreted as native provenance',t=>{
  const f=fixture(t);write(join(f.directory,'candidate.json'),{schema:'geopilot-candidate/1',model:'MiniMax-M3'});
  assert.throws(()=>nativeLaunch({...f.options,expectedCandidateSha256:sha(join(f.directory,'candidate.json'))}),/unexpected|missing|Invalid/);
  assert.equal(existsSync(f.options.output),false);
});
test('U input and output/candidate overlap rejected without launcher execution',t=>{
  const f=fixture(t);assert.throws(()=>nativeLaunch({...f.options,inputs:f.root}),/old M/);
  assert.throws(()=>validateLocations({...f.options,output:join(f.directory,'child')}),/overlaps/);
});
test('actual default-deny profile rejects inline evidence and scorer, imports only numerical runtime',t=>{
  const f=fixture(t), output=join(f.root,'boundary');mkdirSync(output);
  const allowed=join(f.root,'allowed.txt');writeFileSync(allowed,'synthetic allowlist marker');
  const paths=[...Object.values(f.paths),join(f.directory,'candidate.json'),join(f.directory,'program.json')];
  const proof=probeNativeBoundary(output,PY,[allowed],paths);
  assert.equal(proof.isolation.status,'PASS');assert.equal(existsSync(join(output,'execution-started.json')),false);
});

test('parent proof binds actual historical bytes separately from registered semantic identity',t=>{
  const f=fixture(t), snapshot=join(ROOT,'out/geopilot-rsih-p0-mu9geojt-r3/program.json');
  const parent=JSON.parse(readFileSync(snapshot)), semantic=digest(parent), bytes=sha(snapshot);
  assert.notEqual(bytes,semantic);
  writeFileSync(f.paths.parentProgram,readFileSync(snapshot));
  const source={original_inputs:{parentProgram:{path:f.paths.parentProgram,sha256:bytes}}};
  assert.deepEqual(parentIdentity(source,{parent_program_sha256:semantic},PY),
    {parent_program_sha256:bytes,parent_program_content_sha256:semantic});
  assert.throws(()=>parentIdentity(source,{parent_program_sha256:bytes},PY),/semantic identity/);
  assert.throws(()=>parentIdentity({original_inputs:{parentProgram:{path:f.paths.parentProgram,sha256:semantic}}},
    {parent_program_sha256:semantic},PY),/byte hash drift/);
  writeFileSync(f.paths.parentProgram,Buffer.concat([readFileSync(snapshot),Buffer.from('\n')]));
  assert.deepEqual(JSON.parse(readFileSync(f.paths.parentProgram)),parent);
  source.original_inputs.parentProgram.sha256=sha(f.paths.parentProgram);
  assert.throws(()=>parentIdentity(source,{parent_program_sha256:semantic},PY),/historical P0 snapshot/);
});

test('large metadata uses bounded strict parsing, including duplicate/nonfinite/unsafe rejection',t=>{
  const root=mkdtempSync(join(tmpdir(),'native-large-manifest-synthetic-'));
  t.after(()=>rmSync(root,{recursive:true,force:true}));
  const path=join(root,'launch.json'), padding='x'.repeat(11*1024*1024);
  write(path,{synthetic_only:true,padding,bindings:{example:'a'.repeat(64)}});
  const parsed=strictReadManifest(path,PY);
  assert.equal(parsed.synthetic_only,true);assert.equal(parsed.padding,padding);
  for(const text of ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e400}', '{"x":9007199254740993}']) {
    writeFileSync(path,text);assert.throws(()=>strictReadManifest(path,PY),/Strict launch manifest rejected/);
  }
  writeFileSync(path,'x'.repeat(32*1024*1024+1));
  assert.throws(()=>strictReadManifest(path,PY),/oversize launch manifest/);
});
