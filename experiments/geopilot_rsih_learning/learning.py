"""One offline, evidence-bound conditional proposal; no automatic release."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code/geopilot_rsih'))
from strict_json import read_json as load

METRICS = ('accuracy_l1_m', 'accuracy_rmse_m', 'accuracy_best90_l1_m',
           'accuracy_best90_rmse_m', 'precision_0_20', 'completeness_0_20')
SYSTEM = '''You are the OFFLINE GeoPilot reconstruction-program improver.
Use the supplied real development geometry feedback and legal execution diagnostics,
with the supplied domain knowledge. Return exactly one JSON object with keys:
hypothesis (string), expected_effect (string), risks (nonempty string list),
competing_explanations (nonempty string list), evidence_refs (nonempty supplied IDs),
program (complete executable program, not a patch or Markdown).
The program preserves schema and contract_id, has program_id p1 and parent_id p0.
Retain the exact prepare node action and parameters and entry=prepare; only its
success successor may change. All other tool actions are densify, mesh, refine.
Use the actual provided registry parameter ranges, diagnostics, and graph limits.
Nodes are tool {id,kind,action,parameters,success,failure}, check
{id,kind,metric,op,value,success,failure,unknown}, or stop {id,kind}.
Checks use only allowed CURRENT diagnostics, op lt/le/gt/ge and finite threshold.
All successor IDs must exist, all nodes reachable, DAG, and stage types legal on
ALL paths including tool failure and unknown diagnostics. Node IDs are lowercase
letters/digits/underscore, start with a letter. Tools have no reference access.
Initial stage=input with no diagnostics. prepare produces sparse diagnostics;
densify produces dense_points and invalidates any mesh; mesh/refine produce mesh
counts. Tool failure retains parent state. Stop without a mesh means failure.
Make a modest, scientifically motivated conditional update: an actual diagnostic
must choose between DIFFERENT tool/parameter sequences. Do not add a ceremonial
condition whose branches have the same behavior. Explain the rule and threshold
in hypothesis as a testable hypothesis, not a proven cause. Use broad interpretable
thresholds, not a scene identifier or a threshold just above/below one observed count.
Diagnosing causes from global metrics alone is uncertain; preserve alternatives.
Do not alter prepare parameters, inputs, permission, exporter, evaluator, or metric.
Do not optimize evaluator runtime via geometry simplification. The objective is
geometric quality. Timeouts are evaluation failures, never geometric failure evidence.
The numeric adapter currently shares depth-map workspace: do NOT repeat densify
within a path; choose its parameters before its first execution, or change mesh/
refine after dense diagnostics. Stop leaves a dense-derived legal mesh if possible.
Choose a small executable intervention supported by the supplied knowledge; do not
claim its improvement in advance. No hidden reasoning required: concise hypotheses,
risks and structured program are sufficient. No runtime LLM call is being added.
'''


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def provider_request_sha(request):
    # Match the existing provider.ask no-image wire payload; do not add another client.
    payload = {'model': 'MiniMax-M3', 'messages': [
        {'role': 'system', 'content': request['system']},
        {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(
            request['context'], ensure_ascii=False, allow_nan=False)}]}],
        'max_tokens': request['model']['max_tokens'], 'temperature': 0.2, 'reasoning_split': True}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def verify_response(request, response):
    require(response.get('status') == 'ok', 'provider rejected proposal')
    require(response.get('request_sha256') == request.get('provider_payload_sha256') == provider_request_sha(request)
            and response.get('model') == response.get('response_model') == 'MiniMax-M3',
            'provider request/model identity mismatch')
    usage = response.get('usage') or {}
    require(all(type(usage.get(k)) is int and usage[k] >= 0
                for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')),
            'missing measured provider usage')


def verify_request(request_path, response_path):
    """Reuse the same real-score importer at the final candidate launch boundary."""
    request, response = load(request_path), load(response_path)
    require(request.get('schema') == 'geopilot-offline-request/1', 'unknown offline request')
    paths = request['evidence_paths']
    require(set(paths) == {'run', 'score', 'manifest'}, 'missing evidence roles')
    run, score_path, manifest_path = (Path(paths[k]).resolve() for k in ('run', 'score', 'manifest'))
    _, result, score = checked(run, score_path, manifest_path)
    knowledge_path = HERE / 'knowledge.json'
    expected = {str(p.resolve()): sha(p) for p in [run / 'run.json', run / 'result.json',
        run / 'program.json', run / 'genome.json', score_path, manifest_path, knowledge_path]}
    context = request['context']
    require(context['evidence_bindings'] == expected, 'missing or changed source evidence')
    require(context['parent'] == load(run / 'program.json') and
            context['diagnostics'] == result['state']['diagnostics'] and
            context['metrics'] == score['surface_metrics'] and
            context['structure'] == score['structure_diagnostics'] and
            context['knowledge'] == load(knowledge_path), 'model context differs from source evidence')
    verify_response(request, response)


def inspect_genome(path, node='node'):
    result = subprocess.run([node, str(HERE / 'bridge.mjs'), str(path), '--inspect'],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def checked(run_dir, score_path, manifest_path):
    """Import existing valid results; do not redefine the submission validator."""
    run_dir = Path(run_dir).resolve()
    run, result, score, manifest = (load(run_dir / 'run.json'), load(run_dir / 'result.json'),
                                    load(score_path), load(manifest_path))
    require(run.get('status') == 'sealed' and result.get('status') == 'sealed'
            and run.get('frozen_bindings_unchanged') is True
            and run.get('benchmark_eligible') is True and run.get('returncode') == 0,
            'unsealed or ineligible method')
    require(score.get('status') == 'valid' and manifest.get('status') == 'valid', 'invalid score')
    bindings = score.get('bindings', {})
    require(bindings == manifest.get('bindings'), 'score identity drift')
    require(manifest.get('output_sha256', {}).get('score.json') == sha(score_path), 'score output drift')
    for key, filename, run_key in [('mesh_sha256', 'mesh.ply', 'submission_mesh_sha256'),
                                   ('submission_contract_sha256', 'submission.json', 'submission_contract_sha256')]:
        require(bindings.get(key) == run.get(run_key) == sha(run_dir / 'submission' / filename),
                'actual submission identity drift: ' + key)
    launch = load(run_dir / 'launch.json')
    validation = load(run_dir / 'validation.json')
    accepted = json.loads(validation['stdout'])
    require(validation['returncode'] == 0 and accepted['status'] == 'valid', 'missing original validator acceptance')
    for key in ('mesh_sha256', 'submission_contract_sha256', 'input_manifest_sha256',
                'bundle_lock_sha256', 'bundle_manifest_sha256', 'reference_manifest_sha256'):
        require(bindings.get(key) == accepted['bindings'].get(key), 'validator binding drift: ' + key)
    require(score.get('scene_id') == accepted['scene_id'] == 'Dataset-1'
            and score.get('track') == accepted['track'] == 'rgb-oriented'
            and score.get('configuration_id') == accepted['configuration_id'], 'score task identity drift')
    for key in ('release_source_tree_sha256', 'evaluator_runtime_tree_sha256'):
        require(launch['offline_validator'][key] == validation['offline_validator'][key], 'validator runtime drift')
    # launch records source bytes; run/program.json is separately pretty-serialized.
    source_program = Path(launch['program_path'])
    require(sha(source_program) == launch['program_bytes_sha256'], 'program source drift')
    require(load(source_program) == load(run_dir / 'program.json'), 'program semantic drift')
    require(launch['effective'].get(str(run_dir / 'program.json')) == sha(run_dir / 'program.json'), 'effective program drift')
    require(launch['effective'].get(str(run_dir / 'genome.json')) == sha(run_dir / 'genome.json'), 'effective Genome drift')
    inspected = inspect_genome(run_dir / 'genome.json')
    require(inspected['program'] == load(run_dir / 'program.json') and
            inspected['program_hash'] == result['program_sha256'] == launch['program_content_sha256'] and
            inspected['genome_hash'] == launch['genome_sha256'], 'Genome/program identity drift')
    protocol = HERE / 'evaluator/protocol_v1.json'
    require(bindings.get('protocol_sha256') == sha(protocol) and
            bindings.get('runtime_implementation_id') == load(protocol)['runtime']['implementation_id'],
            'unregistered evaluator implementation')
    metrics = score.get('surface_metrics', {})
    for name in METRICS:
        require(type(metrics.get(name)) in (int, float) and math.isfinite(metrics[name]) and metrics[name] >= 0,
                'missing or nonfinite metric: ' + name)
    for name in ('precision_0_20', 'completeness_0_20'):
        require(metrics[name] <= 1, 'invalid fraction')
    return run, result, score


def proposal(parent, context, decision):
    keys = {'hypothesis', 'expected_effect', 'risks', 'competing_explanations', 'evidence_refs', 'program'}
    require(isinstance(decision, dict) and set(decision) == keys, 'invalid model decision')
    require(all(isinstance(decision[k], str) and decision[k].strip()
                for k in ('hypothesis', 'expected_effect')), 'empty model claim')
    require(all(isinstance(decision[k], list) and decision[k] and
                all(isinstance(x, str) and x.strip() for x in decision[k])
                for k in ('risks', 'competing_explanations', 'evidence_refs')), 'invalid model lists')
    require(set(decision['evidence_refs']) <= set(context['evidence_ids']), 'unbound evidence')
    p = decision['program']
    current = json.loads(next(s['content'] for s in parent['skills'] if s['name'] == 'geopilot-reconstruction-program'))
    require(p['parent_id'] == 'p0' and p['program_id'] == 'p1' and p['entry'] == 'prepare', 'candidate identity mismatch')
    prepares = [n for n in p['nodes'] if n.get('action') == 'prepare']
    require(len(prepares) == 1 and
            {k: v for k, v in prepares[0].items() if k != 'success'} ==
            {k: v for k, v in current['nodes'][0].items() if k != 'success'}, 'prepare mutation forbidden')
    checks = [n for n in p['nodes'] if n['kind'] == 'check']
    require(checks and any(n['success'] != n['failure'] for n in checks), 'missing conditional intervention')
    return {'contract_id': p['contract_id'], 'parent_sha256': context['parent_sha256'],
            'signal_id': context['signal_id'], 'group_id': context['group_ids'][0],
            'declared_surface': 'geopilot-reconstruction-program', 'patch': {
                'parent_genome_id': parent['genome_id'], 'evidence_refs': decision['evidence_refs'],
                'hypothesis': decision['hypothesis'], 'expected_effect': decision['expected_effect'],
                'risks': decision['risks'], 'operations': [{'op': 'upsert_skill',
                    'name': 'geopilot-reconstruction-program', 'value': {
                        'description': parent['skills'][0]['description'],
                        'content': json.dumps(p, separators=(',', ':'))}}]}}


def propose(args, ask):
    run, result, score = checked(args.run, args.score, args.manifest)
    current = load(args.run / 'program.json')
    require(current['program_id'] == 'p0', 'first round requires P0')
    parent = load(args.run / 'genome.json')
    inspected = inspect_genome(args.run / 'genome.json', args.node)
    knowledge_path = HERE / 'knowledge.json'
    knowledge = load(knowledge_path)
    for entry in knowledge['sources']:
        require(sha(ROOT / entry['path']) == entry['sha256'], 'domain source changed')
    context = {'parent_sha256': inspected['genome_hash'], 'signal_id': 'development-quality',
               'group_ids': ['dataset1-development'], 'evidence_ids': ['r3-run', 'r3-score', 'domain-knowledge']}
    request = {'schema': 'geopilot-offline-request/1',
               'evidence_paths': {k: str(p.resolve()) for k, p in
                   [('run', args.run), ('score', args.score), ('manifest', args.manifest)]},
               'system': SYSTEM, 'model': {'provider': 'existing project provider', 'model': 'MiniMax-M3',
               'temperature': 0.2, 'max_tokens': 16384}, 'context': {
        'parent': current, 'registry': inspected['registry'], 'allowed_diagnostics': inspected['diagnostics'],
        'limits': inspected['limits'], 'diagnostics': result['state']['diagnostics'],
        'metrics': score['surface_metrics'], 'structure': score['structure_diagnostics'],
        'cost': {'method_wall_seconds': run['wall_seconds'], 'online_mode': 'disabled_by_entry'},
        'knowledge': knowledge, 'evidence_ids': context['evidence_ids'],
        'evidence_bindings': {str(p.resolve()): sha(p) for p in
            [args.run / 'run.json', args.run / 'result.json', args.run / 'program.json',
             args.run / 'genome.json', args.score, args.manifest, knowledge_path]}}}
    request['provider_payload_sha256'] = provider_request_sha(request)
    out = args.output
    out.mkdir()
    shutil.copyfile(args.run / 'genome.json', out / 'parent-genome.json')
    write(out / 'request.json', request)
    write(out / 'context.json', context)
    try:
        response = ask(request)
        write(out / 'response.json', response)
        verify_response(request, response)
        p = proposal(parent, context, response.get('decision'))
        write(out / 'proposal.json', p)
        subprocess.run([args.node, str(HERE / 'bridge.mjs'), str(out / 'parent-genome.json'),
                        str(out / 'proposal.json'), str(out / 'context.json'), str(out / 'bridge.json')], check=True)
        bridged = load(out / 'bridge.json')
        require(bridged['program'] == response['decision']['program'], 'bridge/model program mismatch')
        write(out / 'p1.json', bridged['program'])
        candidate = {'schema': 'geopilot-candidate/1', 'parent_program_sha256': sha(args.run / 'program.json')}
        for key, name in {'program': 'p1.json', 'proposal': 'proposal.json', 'request': 'request.json',
                          'response': 'response.json', 'context': 'context.json', 'parent_genome': 'parent-genome.json'}.items():
            candidate[key + '_sha256'] = sha(out / name)
        candidate['knowledge_sha256'] = sha(knowledge_path)
        write(out / 'candidate.json', candidate)
    except Exception as exc:
        write(out / 'rejected.json', {'status': 'rejected', 'error': str(exc), 'released_program': 'P0'})
        raise


def compare(p0_run, p0_score, p0_manifest, p1_run, p1_score, p1_manifest, output):
    a = checked(p0_run, p0_score, p0_manifest)[2]
    b = checked(p1_run, p1_score, p1_manifest)[2]
    for key in ('protocol_sha256', 'runtime_implementation_id', 'input_manifest_sha256',
                'reference_manifest_sha256', 'bundle_manifest_sha256', 'bundle_lock_sha256'):
        require(a['bindings'][key] == b['bindings'][key], 'unpaired evaluator/input: ' + key)
    p1 = load(p1_run / 'program.json')
    require(p1['parent_id'] == load(p0_run / 'program.json')['program_id'], 'program pair mismatch')
    events = [json.loads(s) for s in (p1_run / 'events.jsonl').read_text().splitlines()]
    decisions = [e for e in events if e['event'] == 'decision']
    calls = lambda es: [(e['action'], e['parameters']) for e in es if e['event'] == 'tool_started']
    old_events = [json.loads(s) for s in (p0_run / 'events.jsonl').read_text().splitlines()]
    candidate = load(p1_run / 'launch.json')['candidate']
    response_path = next(Path(path) for path in candidate['bindings'] if Path(path).name == 'response.json')
    response = load(response_path)
    write(output, {'status': 'unresolved', 'released_program': 'P0',
        'reason': 'missing nine independent real-geometry calibration runs',
        'metrics': {k: {'p0': a['surface_metrics'][k], 'p1': b['surface_metrics'][k],
                        'p1_minus_p0': b['surface_metrics'][k] - a['surface_metrics'][k]} for k in METRICS},
        'conditional_execution': {'decisions': decisions, 'calls_differ_from_p0': calls(events) != calls(old_events)},
        'cost': {'p0_wall_seconds': load(p0_run / 'run.json')['wall_seconds'],
                 'p1_wall_seconds': load(p1_run / 'run.json')['wall_seconds'],
                 'p0_evaluator_seconds': load(p0_manifest)['elapsed_seconds'],
                 'p1_evaluator_seconds': load(p1_manifest)['elapsed_seconds'],
                 'offline_model_usage': response['usage'],
                 'offline_model_seconds': response['wallclock_s'],
                 'online_model_mode': 'disabled_by_entry; zero is not a measured usage record'},
        'bindings': {str(p.resolve()): sha(p) for p in
                     [p0_score, p0_manifest, p1_score, p1_manifest, response_path]}})


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--verify-request' and sys.argv[3] == '--verify-response':
        verify_request(Path(sys.argv[2]), Path(sys.argv[4]))
        raise SystemExit(0)
    parser = argparse.ArgumentParser()
    for name in ('run', 'score', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--node', default='node')
    for name in ('p1-run', 'p1-score', 'p1-manifest'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.p1_run:
        compare(args.run, args.score, args.manifest, args.p1_run, args.p1_score, args.p1_manifest, args.output)
    else:
        sys.path.insert(0, str(ROOT / 'experiments/geopilot_v1'))
        from provider import ask
        propose(args, lambda request: ask(request['system'], request['context'], [], 180,
                                          max_tokens=request['model']['max_tokens']))
