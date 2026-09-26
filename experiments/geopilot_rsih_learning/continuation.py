"""Bounded history-guided exploration from frozen P0; no automatic promotion."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import learning as old

SYSTEM = '''You are the OFFLINE GeoPilot reconstruction-program improver.
The stable parent and unsuccessful historical attempts are DIFFERENT objects.
Use the verified history and tool knowledge to propose ONE testable intervention.
Return only JSON with hypothesis, expected_effect, risks (nonempty list),
competing_explanations (nonempty list), evidence_refs (nonempty supplied IDs),
history_update (object with evidence_ids nonempty list, changed_reasoning string,
unresolved string), test_prediction (object with observed_if_supported string,
observed_if_not_supported string), and program (complete executable program).
Do not claim causality from a combined intervention or increased face counts.
Observed failures are evidence, not proof that every action in them is harmful.
Retain alternatives; the current narrow experiment does not resolve all causes.
Use the exact parent graph/node IDs/edges and all other parameters unchanged.
Change exactly ONE mesh-node parameter allowed by the registry; do not add a
condition, tool, or RefineMesh. Use plan.candidate_id and parent.program_id.
The unit of this experiment is a parameter intervention, not a learned selector.
Explain how the history changed your proposal, without private reasoning traces.
State observable expectations and what remains unknown before execution.
Unknown effective defaults must remain unknown; help defaults alone do not
prove historical config-file contents. Do not invent a result or repair a mesh.
No hidden reference access online. Benchmark, model, tools and preparation stay fixed.
If no defensible testable intervention exists, return an error object instead of
fabricating an improvement; it will be preserved as a rejected proposal.
'''


def sources(run, score, manifest):
    return [run / name for name in ('run.json', 'result.json', 'program.json', 'genome.json',
            'launch.json', 'validation.json', 'events.jsonl')] + [score, manifest]


def calls(run):
    return [{k: e[k] for k in ('node', 'action', 'parameters')} for e in
            map(json.loads, (run / 'events.jsonl').read_text().splitlines()) if e['event'] == 'tool_started']


def compile_context(plan_path, ancestry=()):
    """Recompute shared facts from authoritative files, not a hand-written failure summary."""
    plan_path = Path(plan_path).resolve()
    old.require(plan_path not in ancestry and len(ancestry) < 8, 'cyclic or excessive history ancestry')
    ancestry = (*ancestry, plan_path)
    plan = old.load(plan_path)
    old.require(plan['schema'] == 'geopilot-exploration-plan/1' and
                plan['mode'] == 'single_parameter' and plan['candidate_count'] == 1, 'unknown exploration plan')
    old.require(plan['allowed_action'] == 'mesh', 'unsupported exploration scope')
    paths = {k: Path(v).resolve() for k, v in plan['stable'].items()}
    old.require(set(paths) == {'run', 'score', 'manifest'}, 'invalid stable sources')
    stable_run, result, score = old.checked(paths['run'], paths['score'], paths['manifest'])
    parent = old.load(paths['run'] / 'program.json')
    old.require(parent['program_id'] == 'p0', 'this exploration keeps registered P0 stable')
    old.require(plan['candidate_id'] != 'p0' and plan['candidate_id'] not in ('p1', ''), 'candidate ID reused')
    files = [plan_path] + sources(**paths)
    knowledge_path = Path(plan['knowledge']).resolve()
    knowledge = old.load(knowledge_path)
    files.append(knowledge_path)
    for source in knowledge['sources']:
        path = (old.ROOT / source['path']).resolve()
        old.require(old.sha(path) == source['sha256'], 'knowledge source changed')
        files.append(path)
    history, seen = [], set()
    old.require(0 < len(plan['history']) <= 8, 'history must contain 1..8 attempts')
    for entry in plan['history']:
        h = {k: Path(v).resolve() for k, v in entry.items()}
        old.require(set(h) == {'run', 'score', 'manifest', 'request', 'response'}, 'invalid history sources')
        _, observed, measured = old.checked(h['run'], h['score'], h['manifest'])
        if old.load(h['request'])['schema'] == 'geopilot-offline-request/2':
            verify_request(h['request'], h['response'], ancestry)
        else:
            old.verify_request(h['request'], h['response'])  # original P1 retains v1 knowledge
        raw = old.load(h['response'])['decision']
        program = old.load(h['run'] / 'program.json')
        old.require(program == raw['program'] and program['parent_id'] == parent['program_id'],
                    'history program or parent mismatch')
        hist_request = old.load(h['request'])
        old.require(hist_request['context']['parent'] == parent, 'history baseline differs')
        launch = old.load(h['run'] / 'launch.json')
        bindings = launch['candidate']['bindings']
        old.require(bindings.get(str(h['response'])) == old.sha(h['response']) and
                    bindings.get(str(h['request'])) == old.sha(h['request']), 'unbound history model call')
        for key in ('protocol_sha256', 'runtime_implementation_id', 'input_manifest_sha256',
                    'reference_manifest_sha256', 'bundle_manifest_sha256', 'bundle_lock_sha256'):
            old.require(measured['bindings'][key] == score['bindings'][key], 'history comparison mismatch')
        identity = old.sha(h['run'] / 'program.json')
        old.require(identity not in seen and program['program_id'] != plan['candidate_id'], 'duplicate history or candidate ID')
        seen.add(identity)
        id = 'attempt-' + program['program_id']
        history.append({'id': id, 'role': 'attempt_not_stable_parent', 'program': program,
            'hypothesis': raw['hypothesis'], 'expected_effect': raw['expected_effect'],
            'risks': raw['risks'], 'competing_explanations': raw['competing_explanations'],
            'observed_calls': calls(h['run']), 'diagnostics': observed['state']['diagnostics'],
            'metrics': measured['surface_metrics'], 'structure': measured['structure_diagnostics'],
            'delta_from_stable': {k: measured['surface_metrics'][k] - score['surface_metrics'][k] for k in old.METRICS},
            'causal_status': 'combined interventions and prefix variation; individual causes unresolved',
            'release': 'not_promoted; stable P0 retained'})
        files += sources(h['run'], h['score'], h['manifest']) + [h['request'], h['response']]
        files += [Path(p) for p in hist_request['context']['evidence_bindings']]
    inspected = old.inspect_genome(paths['run'] / 'genome.json')
    return {'parent': parent, 'registry': inspected['registry'], 'allowed_diagnostics': inspected['diagnostics'],
        'limits': inspected['limits'], 'diagnostics': result['state']['diagnostics'],
        'metrics': score['surface_metrics'], 'structure': score['structure_diagnostics'],
        'history': history, 'knowledge': knowledge,
        'plan': {k: plan[k] for k in ('candidate_id', 'mode', 'allowed_action', 'candidate_count', 'question', 'observation_rule')},
        'evidence_ids': ['stable-run', 'stable-score', 'domain-knowledge'] + [e['id'] for e in history],
        'evidence_bindings': {str(p): old.sha(p) for p in files}}


def validate_decision(context, decision):
    keys = {'hypothesis', 'expected_effect', 'risks', 'competing_explanations', 'evidence_refs',
            'history_update', 'test_prediction', 'program'}
    echoes = {'candidate_id', 'parent_program_id', 'mode', 'intervention'}
    old.require(isinstance(decision, dict) and keys <= set(decision) <= keys | echoes,
                'invalid continuation response')
    for field in ('hypothesis', 'expected_effect'):
        old.require(isinstance(decision[field], str) and decision[field].strip(), 'empty model claim')
    for field in ('risks', 'competing_explanations', 'evidence_refs'):
        old.require(isinstance(decision[field], list) and decision[field] and
                    all(isinstance(v, str) and v.strip() for v in decision[field]), 'empty model list')
    old.require(set(decision['evidence_refs']) <= set(context['evidence_ids']), 'unknown evidence')
    update = decision['history_update']
    old.require(isinstance(update, dict) and set(update) == {'evidence_ids', 'changed_reasoning', 'unresolved'},
                'missing history response')
    old.require(isinstance(update['evidence_ids'], list) and update['evidence_ids'] and
                bool(set(update['evidence_ids']) & {e['id'] for e in context['history']}) and
                set(update['evidence_ids']) <= set(context['evidence_ids']) and
                set(update['evidence_ids']) <= set(decision['evidence_refs']), 'history citation missing')
    for field in ('changed_reasoning', 'unresolved'):
        old.require(isinstance(update[field], str) and update[field].strip(), 'empty history interpretation')
    prediction = decision['test_prediction']
    old.require(isinstance(prediction, dict) and set(prediction) == {'observed_if_supported', 'observed_if_not_supported'}
                and all(isinstance(v, str) and v.strip() for v in prediction.values()), 'missing test prediction')
    parent, p = context['parent'], decision['program']
    old.require(p['program_id'] == context['plan']['candidate_id'] and p['parent_id'] == parent['program_id'],
                'wrong candidate identity')
    a, b = copy.deepcopy(parent), copy.deepcopy(p)
    a.pop('program_id'); a.pop('parent_id'); b.pop('program_id'); b.pop('parent_id')
    old.require(len(a['nodes']) == len(b['nodes']), 'graph mutation forbidden')
    changes = []
    for before, after in zip(a['nodes'], b['nodes']):
        if before.get('action') == context['plan']['allowed_action']:
            parameters = before['parameters']; other = after.get('parameters', {})
            for key in sorted(parameters.keys() | other.keys()):
                if parameters.get(key) != other.get(key):
                    changes.append({'node': before['id'], 'parameter': key,
                                    'before_explicit': parameters.get(key), 'after_explicit': other.get(key)})
            after['parameters'] = copy.deepcopy(parameters)
    old.require(a == b and len(changes) == 1, 'exactly one registered mesh parameter may change')
    expected_echoes = {'candidate_id': p['program_id'], 'parent_program_id': p['parent_id'],
        'mode': context['plan']['mode'], 'intervention': {'node_id': changes[0]['node'],
        'parameter': changes[0]['parameter'], 'from': changes[0]['before_explicit'], 'to': changes[0]['after_explicit']}}
    old.require(all(decision[k] == v for k, v in expected_echoes.items() if k in decision),
                'contradictory response metadata')
    return changes


def verify_request(request_path, response_path, ancestry=()):
    request, response = old.load(request_path), old.load(response_path)
    old.require(request['schema'] == 'geopilot-offline-request/2' and request['system'] == SYSTEM,
                'unregistered continuation request')
    context = compile_context(request['plan_path'], ancestry)
    old.require(context == request['context'], 'continuation evidence changed')
    old.verify_response(request, response)
    validate_decision(context, response['decision'])


def propose(plan_path, output, ask):
    context = compile_context(plan_path)
    plan = old.load(plan_path)
    output = Path(output).resolve(); output.mkdir()
    request = {'schema': 'geopilot-offline-request/2', 'plan_path': str(Path(plan_path).resolve()),
               'system': SYSTEM, 'context': context,
               'model': {'model': 'MiniMax-M3', 'temperature': 0.2, 'max_tokens': 16384}}
    request['provider_payload_sha256'] = old.provider_request_sha(request)
    old.write(output / 'request.json', request)
    parent_path = Path(plan['stable']['run']) / 'genome.json'
    shutil.copyfile(parent_path, output / 'parent-genome.json')
    parent = old.load(parent_path)
    bridge_context = {'parent_sha256': old.inspect_genome(parent_path)['genome_hash'],
        'signal_id': 'history-' + context['plan']['candidate_id'], 'group_ids': ['dataset1-development'],
        'evidence_ids': context['evidence_ids'], 'experiment_mode': 'single_parameter'}
    old.write(output / 'context.json', bridge_context)
    try:
        response = ask(request); old.write(output / 'response.json', response)
        old.verify_response(request, response)
        changes = validate_decision(context, response['decision'])
        d = response['decision']; p = d['program']
        proposal = {'contract_id': p['contract_id'], 'parent_sha256': bridge_context['parent_sha256'],
            'signal_id': bridge_context['signal_id'], 'group_id': bridge_context['group_ids'][0],
            'declared_surface': 'geopilot-reconstruction-program', 'patch': {
            'parent_genome_id': parent['genome_id'], 'evidence_refs': d['evidence_refs'],
            'hypothesis': d['hypothesis'], 'expected_effect': d['expected_effect'], 'risks': d['risks'],
            'operations': [{'op': 'upsert_skill', 'name': 'geopilot-reconstruction-program', 'value': {
                'description': parent['skills'][0]['description'], 'content': json.dumps(p, separators=(',', ':'))}}]}}
        old.write(output / 'proposal.json', proposal)
        subprocess.run(['node', str(old.HERE / 'bridge.mjs'), str(output / 'parent-genome.json'),
            str(output / 'proposal.json'), str(output / 'context.json'), str(output / 'bridge.json')], check=True)
        old.require(old.load(output / 'bridge.json')['program'] == p, 'bridge changed model program')
        old.write(output / 'program.json', p)
        old.write(output / 'change.json', {'changes': changes, 'history_update': d['history_update'],
                                           'test_prediction': d['test_prediction']})
        manifest = {'schema': 'geopilot-candidate/2', 'parent_program_sha256': old.sha(Path(plan['stable']['run']) / 'program.json')}
        for key, name in {'program': 'program.json', 'request': 'request.json', 'response': 'response.json',
                'context': 'context.json', 'proposal': 'proposal.json', 'parent_genome': 'parent-genome.json'}.items():
            manifest[key + '_sha256'] = old.sha(output / name)
        old.write(output / 'candidate.json', manifest)
    except Exception as exc:
        old.write(output / 'rejected.json', {'status': 'rejected', 'error': str(exc), 'released_program': 'P0'})
        raise


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--verify-request' and sys.argv[3] == '--verify-response':
        verify_request(Path(sys.argv[2]), Path(sys.argv[4])); raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(old.ROOT / 'experiments/geopilot_v1'))
    from provider import ask
    propose(args.plan, args.output, lambda request: ask(request['system'], request['context'], [],
                                                      timeout=180, max_tokens=16384))
