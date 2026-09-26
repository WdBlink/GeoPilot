/** Native-only parent entry. Old P0/M registry, launcher, numerical tools stay unchanged. */
import { readFileSync, writeFileSync, mkdirSync, existsSync, realpathSync, lstatSync, openSync, readSync, closeSync } from 'node:fs';
import { dirname, basename, join, resolve, relative, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { parseArgs } from 'node:util';
import { verifyNativeCandidate } from './native_proposal.mjs';
import { launch, runtimeBindings, strictRead, launchRecordUnchanged, assertRegisteredP0 } from '../../code/geopilot_rsih/launch.mjs';
import { makeGenome, digest } from '../../code/geopilot_rsih/program.mjs';
import { makeProfile, probeIsolation } from '../../code/geopilot_rsih/isolation.mjs';

const ENTRY = fileURLToPath(import.meta.url), HERE = dirname(ENTRY), ROOT = resolve(HERE, '../..');
const CODE = join(ROOT, 'code/geopilot_rsih');
export const INPUTS = join(ROOT, 'out/usegeo_benchmark/prepared/v1/inputs/rgb-oriented/Dataset-1');
const hash = path => {
  const h=createHash('sha256'), fd=openSync(path,'r'), block=Buffer.allocUnsafe(8*1024*1024);
  try { for(let n;(n=readSync(fd,block,0,block.length,null))>0;)h.update(block.subarray(0,n)); }
  finally {closeSync(fd);} return h.digest('hex');
};
const write=(path,value)=>writeFileSync(path,JSON.stringify(value,null,2)+'\n',{flag:'wx'});
const inside=(p,base)=>{const r=relative(base,p);return r===''||(!r.startsWith('../')&&r!=='..'&&!isAbsolute(r));};
const assert=(ok,message)=>{if(!ok)throw new Error(message);};
const verify=bindings=>{for(const [p,h]of Object.entries(bindings))assert(hash(p)===h,'Bound bytes changed: '+p);};

/** Large launch metadata still uses the frozen duplicate/nonfinite/unsafe-number parser. */
export function strictReadManifest(path, python) {
  const stat=lstatSync(path);
  assert(stat.isFile()&&!stat.isSymbolicLink()&&stat.size<=32*1024*1024,'Unsafe/oversize launch manifest');
  const parsed=spawnSync(python,['-B',join(CODE,'strict_json.py'),path],
    {encoding:'utf8',maxBuffer:64*1024*1024,timeout:30000});
  assert(parsed.status===0&&!parsed.error,'Strict launch manifest rejected: '+(parsed.stderr||parsed.error?.message||'parser failure'));
  return JSON.parse(parsed.stdout);
}

export function validateAdmission(candidate, edit) {
  assert(candidate.evidence_kind==='native_call','Real native_call required; synthetic receipts cannot launch');
  assert(edit.mode==='post_prepare_program','This batch requires post_prepare_program mode');
}

export function parentIdentity(source, edit, python) {
  const parent=source.original_inputs.parentProgram, bytes=hash(parent.path);
  assert(bytes===parent.sha256,'Original parent byte hash drift');
  assert(bytes==='619cc3f49b06951fffc7f9ad2f19e45fe7da810596e074c0db58f64130736b8f',
    'Parent bytes must match the historical P0 snapshot, not its semantic digest');
  assertRegisteredP0(join(CODE,'p0.json'));
  const content=digest(strictRead(parent.path,python));
  assert(content===edit.parent_program_sha256&&content===digest(strictRead(join(CODE,'p0.json'),python)),
    'Parent semantic identity differs from registered P0');
  return {parent_program_sha256:bytes,parent_program_content_sha256:content};
}

export function validateLocations({candidateDirectory,output,inputs,python}) {
  assert(realpathSync(inputs)===realpathSync(INPUTS),'Only frozen old M Dataset-1 inputs accepted');
  assert(!lstatSync(candidateDirectory).isSymbolicLink(),'Candidate directory symlink forbidden');
  const directory=realpathSync(candidateDirectory);
  const destination=join(realpathSync(dirname(resolve(output))),basename(output));
  assert(!existsSync(destination),'Output must be fresh');
  for(const p of [directory,CODE,realpathSync(inputs),resolve(python)]) {
    assert(!inside(destination,p)&&!inside(p,destination),'Output overlaps protected source');
  }
  assert(!inside(directory,CODE),'Offline candidate must be outside worker code allowlist');
  return {directory,destination};
}

/** The real profile must deny all offline files, including original prompt/response. */
export function probeNativeBoundary(output,python,inputFiles,offlineFiles) {
  output=realpathSync(output); inputFiles=inputFiles.map(p=>realpathSync(p)); offlineFiles=offlineFiles.map(p=>realpathSync(p));
  const profile=makeProfile(output,python,inputFiles);
  const forbidden=[...new Set([...offlineFiles,
    join(ROOT,'out/usegeo_benchmark/prepared/v1/evaluator-only/Dataset-1/reference_manifest.json'),
    join(ROOT,'experiments/geopilot_rsih_learning/evaluator/benchmark.py')])];
  return {profile,isolation:probeIsolation(profile,output,python,inputFiles.slice(0,3),forbidden,true)};
}

export function nativeLaunch({candidateDirectory,expectedCandidateSha256,output,inputs,python,preflightOnly=true}) {
  assert(typeof preflightOnly==='boolean','Invalid preflight flag');
  const {directory,destination}=validateLocations({candidateDirectory,output,inputs,python});
  python=resolve(python); inputs=realpathSync(inputs);
  // Full deterministic replay verifies raw bytes and original sources before any launch activity.
  const candidate=verifyNativeCandidate(directory,expectedCandidateSha256);
  const edit=strictRead(join(directory,'edit-contract.json'),python);
  validateAdmission(candidate,edit);
  const programPath=join(directory,'program.json'), p=strictRead(programPath,python);
  const source=strictRead(join(directory,'source.json'),python);
  const offline=Object.fromEntries(Object.keys(candidate.artifacts).map(n=>[join(directory,n),hash(join(directory,n))]));
  offline[join(directory,'candidate.json')]=expectedCandidateSha256;
  for(const entry of Object.values(source.original_inputs)) offline[entry.path]=entry.sha256;
  const nativeProof={schema:'geopilot-native-execution-proof/1',provider:'native_codex',
    evidence_kind:'native_call',authenticity:source.authenticity,edit_mode:edit.mode,
    manifest_sha256:expectedCandidateSha256,program_sha256:hash(programPath),
    program_content_sha256:digest(p),...parentIdentity(source,edit,python),bindings:offline};
  // Reuse the old launcher's complete private input validator without bypassing its admission gate.
  // This is a separate P0 PREFLIGHT receipt, never a P0 numerical run or native receipt.
  mkdirSync(destination);
  try {
    const proofDir=join(destination,'p0-input-preflight');
    launch({program:join(CODE,'p0.json'),output:proofDir,inputs,python,scope:'development',preflightOnly:true});
    const proofPath=join(proofDir,'launch.json'), proof=strictReadManifest(proofPath,python);
    verify({...proof.bindings,...proof.input_hashes,...proof.effective});
    const cli=join(CODE,'upstream/dist/src/cli.js');
    const sources=runtimeBindings(python,programPath,cli,true);
    Object.assign(sources.bindings,offline,{[ENTRY]:hash(ENTRY),[join(HERE,'native_proposal.mjs')]:hash(join(HERE,'native_proposal.mjs')),
      [proofPath]:hash(proofPath)});
    const genome=makeGenome(p);genome.model={profile:'geopilot-driver',id:'no-model-calls'};
    write(join(destination,'genome.json'),genome);write(join(destination,'program.json'),p);
    const config=join(destination,'providers.json');
    write(config,{'geopilot-driver':{kind:'mock',model:'no-model-calls',mock_responses:[]}});
    const request={genome_path:join(destination,'genome.json'),genome_sha256:digest(genome),output:destination,inputs,python,scope:'development'};
    write(join(destination,'request.json'),request);
    const effective=Object.fromEntries(['genome.json','program.json','providers.json','request.json'].map(n=>[join(destination,n),hash(join(destination,n))]));
    const inputFiles=[join(inputs,'input_manifest.json'),join(inputs,'Image_orientations_dataset1.xyz'),
      ...proof.input_image_ids.map(n=>join(inputs,'images',n))];
    const {profile,isolation}=probeNativeBoundary(destination,python,inputFiles,Object.keys(offline));
    write(join(destination,'isolation.json'),isolation);
    const environment={PATH:'/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin',HOME:destination,TMPDIR:destination,
      GEOPILOT_REQUEST:join(destination,'request.json'),GEOPILOT_PYTHON:python,
      RSIH_CODING_AGENT_DIR:join(destination,'agent'),PI_CODING_AGENT_DIR:join(destination,'agent'),
      PI_OFFLINE:'1',PI_SKIP_VERSION_CHECK:'1',OPENSSL_CONF:'/dev/null'};
    const argv=[cli,'--genome',request.genome_path,'--config',config,'--cwd',destination,'--run-id','geopilot',
      '--no-tools','--tools','geopilot_execute','--no-extensions','--no-skills','--no-prompt-templates','--no-themes',
      '-e',join(CODE,'extension.ts'),'-p','/geopilot-run'];
    const launchPath=join(destination,'launch.json');
    write(launchPath,{scope:'development',entry_kind:'native_post_prepare',formal_candidate:false,
      bindings:sources.bindings,effective,symlinks:sources.symlinks,offline_validator:sources.offline,
      input_hashes:proof.input_hashes,input_manifest_sha256:proof.input_manifest_sha256,
      frozen_input_manifest_sha256:proof.frozen_input_manifest_sha256,input_image_ids:proof.input_image_ids,
      runtime:sources.modules,upstream_commit:proof.upstream_commit,program_path:programPath,
      program_bytes_sha256:hash(programPath),program_content_sha256:digest(p),genome_sha256:digest(genome),
      argv,environment,candidate:nativeProof,isolation_profile_sha256:hash(profile),
      model_usage:{mode:'disabled_by_entry',measured:false},offline_native_token_usage:null});
    const launchHash=hash(launchPath),stat=lstatSync(launchPath),profileHash=hash(profile);
    const recheck=()=>{
      verifyNativeCandidate(directory,expectedCandidateSha256);
      verify({...sources.bindings,...proof.input_hashes,...effective,[profile]:profileHash});
      for(const [path,target]of Object.entries(sources.symlinks))assert(realpathSync(path)===target,'Runtime symlink changed');
      assert(launchRecordUnchanged(launchPath,launchHash,stat),'Launch record replaced');
    };
    recheck();
    if(preflightOnly){const result={status:'preflight_passed',scope:'development',benchmark_eligible:false,
      formal_candidate:false,execution_status:'not_launched',candidate_sha256:expectedCandidateSha256};
      write(join(destination,'run.json'),result);return result;}
    const supervised=spawnSync(python,['-B',join(CODE,'supervise.py'),destination,'/usr/bin/sandbox-exec','-f',profile,process.execPath,...argv],
      {cwd:destination,env:environment,encoding:'utf8',maxBuffer:1024*1024});
    const costs=existsSync(join(destination,'supervision.json'))?strictRead(join(destination,'supervision.json'),python):{};
    let unchanged=true,drift=null;try{recheck();}catch(error){unchanged=false;drift=error.message;}
    const result=existsSync(join(destination,'result.json'))?strictRead(join(destination,'result.json'),python):null;
    const sealed=supervised.status===0&&costs.returncode===0&&costs.reason===null&&unchanged&&
      result?.status==='sealed'&&result.program_sha256===digest(p);
    const run={status:sealed?'sealed':'failed',scope:'development',benchmark_eligible:false,formal_candidate:false,
      wall_seconds:costs.wall_seconds??null,peak_worker_process_group_rss_bytes:costs.peak_worker_process_group_rss_bytes??null,
      returncode:costs.returncode??supervised.status,cancellation_reason:costs.reason??null,
      frozen_bindings_unchanged:unchanged,error:drift??supervised.error?.message??null};
    write(join(destination,'run.json'),run);
    // Export/validation and independent scoring remain explicit parent operations, never admission by this launcher.
    return run;
  }catch(error){if(!existsSync(join(destination,'run.json')))write(join(destination,'run.json'),{
    status:'blocked',scope:'development',benchmark_eligible:false,formal_candidate:false,error:String(error.message)});throw error;}
}

if(process.argv[1]&&resolve(process.argv[1])===ENTRY){
  try{
    const {values}=parseArgs({options:Object.fromEntries(['candidate-directory','expected-candidate-sha256','output','inputs','python','action'].map(k=>[k,{type:'string'}])),strict:true});
    assert(['preflight','run'].includes(values.action),'Explicit --action preflight|run required');
    const options=Object.fromEntries(Object.entries(values).filter(([k])=>k!=='action').map(([k,v])=>[k.replace(/-([a-z])/g,(_,c)=>c.toUpperCase()),v]));
    const result=nativeLaunch({...options,preflightOnly:values.action==='preflight'});
    console.log(JSON.stringify(result));if(!['sealed','preflight_passed'].includes(result.status))process.exitCode=1;
  }catch(error){console.error(error.message);process.exitCode=1;}
}
