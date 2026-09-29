"""Offline checks for the four-arm host dispatcher.

No model API, no numerical chain, no scorer, no reference read, no network:
every boundary below is a stub, and the network and process entry points are
explicitly disabled for the duration of the run so a regression cannot slip a
real call past a test. Run with the project research Python and ``-B``.
"""
from __future__ import annotations

import copy
import io
import json
import socket
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import outer_agent_dispatch as host
import outer_agent_receipt as receipts
from outer_agent_receipt import digest, require_sendable

METRICS = {name: 0.1 for name in receipts.METRICS}
REGISTERED = {
    'cfg-mesh-dec-0.25': {'mesh': {'decimate': 0.25}},
    'cfg-mesh-dec-0.50': {'mesh': {'decimate': 0.50}},
    'cfg-dense-l2': {'densify': {'resolution-level': 2}},
}
EXPECTED_MODEL = 'stub-model-v1'


def registry():
    """A three-entry registered set; membership is a pure function of the id."""

    def require(value):
        name = value if isinstance(value, str) else None
        if name not in REGISTERED:
            raise ValueError('unregistered_config')

    def resolve(value):
        require(value)
        base = {'undistort_max_image_size': 2400, 'densify': {'resolution-level': 1},
                'mesh': {'decimate': 0.5}, 'undistort_call': {'num_patch_match_src_images': 20}}
        base.update(REGISTERED[value])
        return base

    return host.ConfigRegistry(ids=tuple(sorted(REGISTERED)), require=require, resolve=resolve,
                               identify=lambda value: str(value), membership_sha256='fixture')


class StubClient:
    """Returns fixed transcript bodies; never opens a socket."""

    def __init__(self, transcripts, model=EXPECTED_MODEL):
        self.transcripts = list(transcripts)
        self.model = model
        self.payloads = []

    def settings(self):
        return {'endpoint': 'stub://offline', 'model': self.model, 'temperature': 0.2,
                'max_tokens': 4096, 'reasoning_split': True, 'timeout_s': 1.0,
                'max_response_bytes': 65536, 'mode': 'injected'}

    def complete(self, payload, *, clock=None):
        self.payloads.append(json.loads(payload))
        answer = self.transcripts.pop(0) if self.transcripts else {'tool': 'stop', 'reason': 'done'}
        body = json.dumps({'model': self.model, 'id': 'stub-response-1', 'usage': {
            'prompt_tokens': 120, 'completion_tokens': 40, 'total_tokens': 160},
            'choices': [{'finish_reason': 'stop',
                         'message': {'content': json.dumps(answer)}}]}).encode()
        return {'status': 'ok', 'error': None, 'body': body, 'http_status': 200,
                'request_id': 'stub-request-1',
                'timing': {'transport_started_monotonic_s': 0.0, 'http_started_monotonic_s': 0.0,
                           'http_ended_monotonic_s': 0.01, 'transport_elapsed_s': 0.01}}


def numerical_stub(scene_id, branch, checked, output, guards, *, preflight_only, config):
    """Record the exact call the host made, then write a synthetic result chain."""
    CALLS.append({'scene_id': scene_id, 'branch': branch, 'checked': checked,
                  'output': output, 'guards': guards, 'preflight_only': preflight_only,
                  'config': config})
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    for stage in ('F-U', 'delivery'):
        (root / stage).mkdir(exist_ok=True)
        (root / stage / 'command.json').write_text(
            json.dumps({'argv': ['/usr/bin/env', 'RefineMesh', '--stub', stage]}))
    (root / 'delivery/supervision.json').write_text(
        json.dumps({'returncode': 0, 'peak_worker_rss_bytes': 12345678}))
    (root / 'result.json').write_text(json.dumps({'status': 'completed', 'parameters': config}))
    return {'status': 'completed', 'result': str(root / 'result.json'),
            'sha256': digest({'stages': ['F-U', 'delivery'], 'config': config})}


CALLS: list = []
SCORES: list = []


def score_stub(scene_id, output, guards, *, numerical_result, expected_sha256,
               preflight_only, config):
    """Write a deliberately leaky score receipt; the host must project it."""
    SCORES.append({'scene_id': scene_id, 'output': output, 'numerical_result': numerical_result,
                   'expected_sha256': expected_sha256, 'preflight_only': preflight_only,
                   'config': config})
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    metrics = dict(METRICS)
    metrics['accuracy_best90_count'] = 900000
    payload = {'schema': 'geopilot-outer-development-score/1', 'status': 'valid',
               'surface_metrics': metrics, 'counts': {'surface_samples': 1000000},
               'raycast_origin_m': [1.0, 2.0, 3.0], 'alignment': {'kind': 'identity'},
               'reference_manifest_sha256': 'a' * 64, 'default_label': 'p0'}
    (root / 'result.json').write_text(json.dumps(payload))
    return {'status': 'valid', 'result': str(root / 'result.json'), 'sha256': 'c' * 64}


RULES = [{'id': 'R1', 'after': 'start', 'config': 'cfg-mesh-dec-0.25',
          'hypothesis': 'decimate 0.25 raises completeness', 'competing_explanation': 'no change',
          'expected_observations': ['face count', 'completeness']},
         {'id': 'R2', 'after': 'score', 'config': 'cfg-dense-l2',
          'hypothesis': 'resolution 2 raises accuracy', 'competing_explanation': 'regression',
          'expected_observations': ['accuracy_l1_m']},
         {'id': 'R3', 'after': 'score', 'config': 'cfg-mesh-dec-0.50',
          'hypothesis': 'half decimate trades accuracy', 'competing_explanation': 'no trade',
          'expected_observations': ['accuracy_l1_m', 'mesh faces']}]


def experience(source_path):
    return {'GE': {'source': {'path': str(source_path), 'sha256': receipts.sha256_file(source_path)},
                   'entries': [{'id': 'E-P', 'trigger': 'dense under-sampling',
                                'lesson': 'raise resolution level', 'source_id': 'history-E-P'}]}}


def task_for(tmp):
    manifest = tmp / 'input-manifest.json'
    manifest.write_text('{"scene_id": "Dataset-2", "synthetic": true}\n')
    help_text = tmp / 'mesh-help.txt'
    help_text.write_text('--decimate <float>\n')
    common = {'task': 'deliver a mesh or an honest failure', 'scene': 'Dataset-2',
              'role': 'development', 'feedback': host.FEEDBACK_INTERFACE}
    return {'id': 'task-D2-fixture', 'scene_id': 'Dataset-2', 'branch': 'F-U',
            'checked': str(tmp / 'checked'), 'common': common,
            'information_sources': {
                'input_manifest': {'path': str(manifest), 'sha256': receipts.sha256_file(manifest)},
                'mesh_help': {'path': str(help_text), 'sha256': receipts.sha256_file(help_text)}}}


class FakeClock:
    """Deterministic monotonic clock so a fixed transcript yields a fixed receipt."""

    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.25
        return self.value


def dispatch(**kwargs):
    """Call the host with frozen stubs, a fixed clock and the four-arm rule table."""
    defaults = {'clock': FakeClock(), 'utc': lambda: '2026-09-29T00:00:00Z', 'rules': RULES}
    defaults.update(kwargs)
    return host.dispatch(**defaults)


def run_proposal(config, hypothesis='raise completeness'):
    return {'tool': 'run_numerical', 'config': config, 'hypothesis': hypothesis,
            'competing_explanation': 'the rival explanation',
            'expected_observations': ['accuracy_l1_m']}


def latest_feedback(state):
    """Resolve the model-facing reference without altering historical facts."""
    reference = state['previous_feedback']
    return None if reference is None else state['history'][reference['history_index']]['feedback']


def claim_row(*, status='not_tested', observations=(), results=()):
    return {'claim_id': 'C1', 'observation_refs': list(observations),
            'alternatives': 'The parameter may have no effect.',
            'proposed_action': 'cfg-dense-l2',
            'predicted_observation': 'The declared common measurement changes.',
            'observed_result_refs': list(results), 'status': status}


def update_state(row, *, memories=()):
    return {'tool': 'update_decision_state', 'claims': [row], 'experience': list(memories)}


class DispatcherTest(unittest.TestCase):
    """Every check runs offline; the network and process entry points are disabled."""

    def setUp(self):
        CALLS.clear()
        SCORES.clear()
        self._net = patch.object(urllib.request, 'urlopen',
                                 side_effect=AssertionError('network call attempted'))
        self._net.start()
        self._sock = patch.object(socket, 'socket', side_effect=AssertionError('socket opened'))
        self._sock.start()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._net.stop()
        self._sock.stop()
        self._tmp.cleanup()

    def boundaries(self):
        return (host.NumericalBoundary(numerical_stub), host.ScoreBoundary(score_stub))

    def test_default_policy_does_not_cap_model_calls(self):
        task = task_for(self.tmp)
        client = StubClient([run_proposal('cfg-not-registered')] * 17 +
                            [{'tool': 'stop', 'reason': 'no supported intervention'}])
        receipt = dispatch(arm_id='L', task=task, output=self.tmp / 'uncapped',
                           registry=registry(), boundaries=self.boundaries(),
                           client=client, policy=host.Policy(),
                           experience=experience(self.tmp / 'input-manifest.json'))
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['cost']['model_calls'], 18)
        self.assertTrue(all(v is None for v in receipt['capability']['budget'].values()))
        self.assertEqual(len(CALLS), 0)

    def test_token_limit_is_omitted_unless_explicit(self):
        client = host.HttpModelClient(endpoint='https://example.invalid',
                                      api_key='synthetic', model='MiniMax-M3.1-Flash-Preview')
        self.assertNotIn('max_tokens', json.loads(host._payload(client, 'test', {})))
        client = host.HttpModelClient(endpoint='https://example.invalid', api_key='synthetic',
                                      model='MiniMax-M3.1-Flash-Preview', max_tokens=1234)
        self.assertEqual(json.loads(host._payload(client, 'test', {}))['max_tokens'], 1234)

    def test_malformed_choice_still_writes_failure_receipt(self):
        task = task_for(self.tmp)
        for i, choice in enumerate((None, 'bad', 17, [])):
            client = StubClient([])
            reply = client.complete(b'{}')
            raw = json.loads(reply['body'])
            raw['choices'] = [choice]
            reply['body'] = json.dumps(raw).encode()
            with patch.object(client, 'complete', return_value=reply):
                out = self.tmp / ('malformed-' + str(i))
                receipt = dispatch(arm_id='L', task=task, output=out, registry=registry(),
                                   boundaries=self.boundaries(), client=client, policy=host.Policy(),
                                   experience=experience(self.tmp / 'input-manifest.json'))
            self.assertEqual(receipt['error'], 'response_fault')
            self.assertTrue((out / 'receipt.json').is_file())
        self.assertEqual(len(CALLS), 0)

    def test_utc_start_is_captured_before_model_call(self):
        task = task_for(self.tmp)
        stamp = ['2026-09-29T00:00:00Z']
        client = StubClient([{'tool': 'stop', 'reason': 'done'}])
        original = client.complete
        def complete(payload, *, clock=None):
            stamp[0] = '2026-09-29T00:01:00Z'
            return original(payload, clock=clock)
        with patch.object(client, 'complete', side_effect=complete):
            receipt = dispatch(arm_id='L', task=task, output=self.tmp / 'start',
                               registry=registry(), boundaries=self.boundaries(),
                               client=client, policy=host.Policy(), utc=lambda: stamp[0],
                               experience=experience(self.tmp / 'input-manifest.json'))
        self.assertEqual(receipt['timing']['started_at_utc'], '2026-09-29T00:00:00Z')

    def test_arms_differ_only_in_declared_inputs(self):
        task = task_for(self.tmp)
        reg, policy = registry(), host.Policy(resource_guards={'timeout_seconds': 60,
                                                               'max_rss_bytes': 1024})
        snapshot = experience(self.tmp / 'input-manifest.json')
        arms = host.arm_definitions(host.geopilot_procedure())
        records = {a.arm: host.capabilities(a, task, policy, reg) for a in arms}
        self.assertEqual(host.require_equal_capability(records), 'PASS')
        equal = host.EQUALITY_KEYS
        for key in host.CONTRAST_KEYS:
            values = {name: record[key] for name, record in records.items()}
            if key == 'decision_state_enforcement':
                self.assertEqual(set(values.values()), {'disabled'})
            else:
                self.assertGreater(len(set(map(repr, values.values()))), 1, key)
        for name, record in records.items():
            for key in equal:
                self.assertEqual(record[key], records['R'][key], name + '/' + key)
        # A doctored capability set must stop the dispatch, not merely be noticed.
        doctored = copy.deepcopy(records)
        doctored['GE']['permissions'] = records['GE']['permissions'] + ' extra reach'
        with self.assertRaises(ValueError) as caught:
            host.require_equal_capability(doctored)
        self.assertIn('permissions', str(caught.exception))
        doctored = copy.deepcopy(records)
        del doctored['L']['config_set']
        with self.assertRaises(ValueError):
            host.require_equal_capability(doctored)
        # G0 and GE must carry byte-identical procedure text.
        self.assertEqual(arms[2].procedure, arms[3].procedure)
        with self.assertRaises(ValueError):
            host.require_arm_parity(arms, {'G0': {'entries': [{'id': 'x'}]}, 'GE':
                                           snapshot['GE']}, RULES)

    def test_unregistered_config_is_rejected_and_recorded(self):
        task = task_for(self.tmp)
        client = StubClient([run_proposal('cfg-not-registered'),
                             run_proposal('cfg-mesh-dec-0.25'),
                             {'tool': 'stop', 'reason': 'table exhausted'}])
        out = self.tmp / 'ge'
        receipt = dispatch(arm_id='GE', task=task, output=out, registry=registry(),
                           boundaries=self.boundaries(), client=client,
                           policy=host.Policy(resource_guards={'timeout_seconds': 60}),
                           experience=experience(self.tmp / 'input-manifest.json'))
        self.assertEqual([r['reason'] for r in receipt['rejected_actions']],
                         ['unregistered_config'])
        self.assertEqual(receipt['rejected_actions'][0]['proposal']['config'],
                         'cfg-not-registered')
        self.assertEqual(receipt['turns'][0]['outcome'], 'rejected')
        self.assertEqual(receipt['tool_calls'][0]['exit_status'], 'rejected')
        self.assertEqual(len(CALLS), 1, 'the rejected proposal must not reach the chain')
        self.assertEqual(CALLS[0]['config'], 'cfg-mesh-dec-0.25')
        # The refusal crosses the boundary without the proposed configuration.
        refusal = receipt['turns'][1]['request']['context_sources']
        self.assertEqual(set(refusal), {'input_manifest', 'mesh_help'})
        self.assertEqual(receipt['feedback_history'][0]['feedback']['reason'],
                         'unregistered_config')
        self.assertEqual(receipt['feedback_history'][1]['config_id'], 'cfg-mesh-dec-0.25')
        second = json.loads(client.payloads[1]['messages'][1]['content'])
        final = json.loads(client.payloads[2]['messages'][1]['content'])
        self.assertEqual(second['previous_feedback'], {'history_index': 0})
        self.assertEqual(latest_feedback(second), receipt['feedback_history'][0]['feedback'])
        self.assertEqual(final['previous_feedback'], {'history_index': 1})
        self.assertEqual(latest_feedback(final), receipt['feedback_history'][1]['feedback'])
        self.assertEqual(final['history'], receipt['feedback_history'])

    def test_feedback_projection_cannot_leak(self):
        leaky = {'status': 'valid', 'surface_metrics': {**METRICS, 'reference_fitted_l1_m': 0.0,
                                                        'accuracy_best90_count': 900000},
                 'counts': {'surface_samples': 1000000}, 'raycast_origin_m': [1.0, 2.0, 3.0],
                 'alignment': {'kind': 'identity'}, 'reference_manifest_sha256': 'a' * 64,
                 'default_label': 'p0'}
        projected = receipts.project_feedback(leaky)
        self.assertEqual(set(projected), {'status', 'surface_metrics'})
        self.assertLessEqual(set(projected), receipts.FEEDBACK_WHITELIST)
        self.assertNotIn('reference_fitted_l1_m', projected['surface_metrics'])
        self.assertEqual(set(projected['surface_metrics']), set(receipts.METRICS))
        dropped = receipts.dropped_keys(leaky)
        for name in ('counts', 'raycast_origin_m', 'alignment', 'reference_manifest_sha256',
                     'default_label', 'surface_metrics.reference_fitted_l1_m',
                     'surface_metrics.accuracy_best90_count'):
            self.assertIn(name, dropped)
            self.assertNotIn(name, projected)
        require_sendable(projected)
        with self.assertRaises(ValueError):
            receipts.project_feedback({'status': 'valid', 'surface_metrics': {**METRICS,
                                                                             'accuracy_l1_m': -1.0}})
        # End to end: what the model actually receives is the projection, not the receipt.
        task = task_for(self.tmp)
        client = StubClient([run_proposal('cfg-dense-l2'), {'tool': 'stop', 'reason': 'done'}])
        receipt = dispatch(arm_id='L', task=task, output=self.tmp / 'leak', registry=registry(),
                           boundaries=self.boundaries(), client=client,
                           policy=host.Policy(resource_guards={'timeout_seconds': 60}),
                           experience=experience(self.tmp / 'input-manifest.json'))
        wire = json.dumps(client.payloads[1]['messages'], ensure_ascii=False)
        for leaked in ('reference_manifest_sha256', 'raycast_origin_m', 'accuracy_best90_count',
                       'default_label', 'surface_samples'):
            self.assertNotIn(leaked, wire)
        self.assertIn('accuracy_l1_m', wire)

    def test_receipt_is_byte_reproducible(self):
        task = task_for(self.tmp)
        args = dict(task=task, registry=registry(), boundaries=self.boundaries(),
                    policy=host.Policy(resource_guards={'timeout_seconds': 60}),
                    experience=experience(self.tmp / 'input-manifest.json'))
        transcript = [run_proposal('cfg-mesh-dec-0.25'), run_proposal('cfg-dense-l2'),
                      {'tool': 'stop', 'reason': 'exhausted'}]
        first = dispatch(arm_id='GE', output=self.tmp / 'one', client=StubClient(transcript), **args)
        CALLS.clear()
        SCORES.clear()
        second = dispatch(arm_id='GE', output=self.tmp / 'two', client=StubClient(transcript), **args)

        def normalised(receipt, root):
            """Replace the run's own absolute output path; nothing else may differ."""
            return json.dumps(receipt, sort_keys=True).replace(str(root), '<OUT>')

        left = (self.tmp / 'one/receipt.json').read_text()
        right = (self.tmp / 'two/receipt.json').read_text()
        self.assertEqual(normalised(first, self.tmp / 'one'),
                         normalised(second, self.tmp / 'two'))
        self.assertEqual(left.replace(str(self.tmp / 'one'), '<OUT>'),
                         right.replace(str(self.tmp / 'two'), '<OUT>'))
        self.assertEqual(first['turn_count'], 3)
        self.assertEqual(len(first['executed_actions']), 2)
        self.assertEqual(first['status'], 'completed')
        self.assertEqual(first['stop_reason'], 'agent_stop')
        self.assertEqual(first['cost']['model_calls'], 3)
        self.assertEqual(first['cost']['input_tokens'], 360)
        self.assertEqual(first['cost']['peak_rss_bytes'], 12345678)
        self.assertIsNone(first['cost']['gpu_seconds'])
        self.assertEqual(first['executed_actions'][0]['config_id'], 'cfg-mesh-dec-0.25')
        self.assertIn('F-U', first['executed_actions'][0]['argv'])
        self.assertEqual(first['executed_actions'][0]['numerical']['entry'],
                         'outer_numerical.execute')
        self.assertEqual(first['executed_actions'][0]['score']['receipt_sha256'], 'c' * 64)
        self.assertTrue((self.tmp / 'one/raw/step-0000.response.body').is_file())

    def test_model_identity_mismatch_is_terminal(self):
        task = task_for(self.tmp)
        for reported, expected_error in (('a-different-backend', 'model_identity_mismatch'),
                                         (None, 'model_identity_missing')):
            client = StubClient([run_proposal('cfg-dense-l2')])
            body = json.loads(json.dumps({'choices': []}))
            client.transcripts = None

            def reply(payload, *, clock=None, reported=reported):
                answer = {'tool': 'run_numerical', 'config': 'cfg-dense-l2',
                          'hypothesis': 'h', 'competing_explanation': 'c',
                          'expected_observations': ['o']}
                raw = {'id': 'r1', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
                       'choices': [{'finish_reason': 'stop',
                                    'message': {'content': json.dumps(answer)}}]}
                if reported is not None:
                    raw['model'] = reported
                return {'status': 'ok', 'error': None, 'body': json.dumps(raw).encode(),
                        'http_status': 200, 'request_id': 'r',
                        'timing': {'transport_started_monotonic_s': 0.0,
                                   'http_started_monotonic_s': 0.0,
                                   'http_ended_monotonic_s': 0.0, 'transport_elapsed_s': 0.0}}

            client.complete = reply
            with self.assertRaises(ValueError):
                host.require_task_parity({**task, 'unexpected': 1})
            receipt = dispatch(arm_id='G0', task=task, output=self.tmp / ('id-' + str(reported)),
                               registry=registry(), boundaries=self.boundaries(),
                               client=client,
                               policy=host.Policy(resource_guards={'timeout_seconds': 60}),
                               experience=experience(self.tmp / 'input-manifest.json'))
            self.assertEqual(receipt['status'], 'invalid')
            self.assertEqual(receipt['error'], expected_error)
            self.assertEqual(receipt['stop_reason'], expected_error)
            self.assertEqual(receipt['model']['response_model'], reported)
            self.assertFalse(receipt['model']['model_match'])
            self.assertEqual(CALLS, [], 'no arm may execute after an identity mismatch')
            self.assertEqual(receipt['executed_actions'], [])

    def test_rule_arm_makes_no_model_call(self):
        task = task_for(self.tmp)
        receipt = dispatch(arm_id='R', task=task, output=self.tmp / 'rules', registry=registry(),
                           boundaries=self.boundaries(), client=None,
                           policy=host.Policy(resource_guards={'timeout_seconds': 60}),
                           experience=experience(self.tmp / 'input-manifest.json'),
                           rules=RULES)
        self.assertEqual(receipt['cost']['model_calls'], 0)
        self.assertIsNone(receipt['model']['expected'])
        self.assertEqual(receipt['stop_reason'], 'rule_table_exhausted')
        self.assertEqual([a['config_id'] for a in receipt['executed_actions']],
                         ['cfg-mesh-dec-0.25', 'cfg-dense-l2', 'cfg-mesh-dec-0.50'])
        self.assertEqual([t['outcome'] for t in receipt['turns']],
                         ['rule_table_step', 'rule_table_step', 'rule_table_step', 'stopped'])
        self.assertEqual(receipt['instruction']['rule_table_sha256'], digest(RULES))
        self.assertEqual([c['name'] for c in receipt['tool_calls']],
                         ['run_numerical'] * 3 + ['stop'])
        self.assertEqual(receipt['tool_calls'][-1]['exit_status'], 'ok')
        for call in receipt['tool_calls']:
            self.assertEqual(call['exit_status'], 'ok')

    def test_send_gate_and_source_drift(self):
        with self.assertRaises(ValueError):
            require_sendable({'note': 'out/usegeo_benchmark/prepared/v1/evaluator-only/Dataset-2'})
        with self.assertRaises(ValueError):
            require_sendable({'api_key': 'x'})
        with self.assertRaises(ValueError):
            require_sendable({'note': 'arm-GE/step-0000'}, forbidden=['arm-GE'])
        require_sendable({'note': 'arm-R/step-0000'}, forbidden=['arm-GE'])
        task = task_for(self.tmp)
        receipt_sources = task['information_sources']
        task['information_sources'] = {name: {**entry, 'sha256': 'f' * 64}
                                       for name, entry in receipt_sources.items()}
        with self.assertRaises(ValueError):
            dispatch(arm_id='L', task=task, output=self.tmp / 'drift', registry=registry(),
                     boundaries=self.boundaries(), client=StubClient([]),
                     policy=host.Policy(resource_guards={'timeout_seconds': 60}))

    def test_budget_and_failure_terminals(self):
        task = task_for(self.tmp)
        policy = host.Policy(max_steps=2, max_rejections=1,
                             resource_guards={'timeout_seconds': 60})
        client = StubClient([run_proposal('cfg-nope'), run_proposal('cfg-nope')])
        receipt = dispatch(arm_id='L', task=task, output=self.tmp / 'reject', registry=registry(),
                           boundaries=self.boundaries(), client=client, policy=policy,
                           experience=experience(self.tmp / 'input-manifest.json'), rules=RULES)
        self.assertEqual(receipt['stop_reason'], 'rejection_budget')
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(len(receipt['rejected_actions']), 1)
        self.assertEqual(CALLS, [])

        client = StubClient([run_proposal('cfg-dense-l2'), run_proposal('cfg-dense-l2'),
                             {'tool': 'stop', 'reason': 'unused'}])
        receipt = dispatch(arm_id='L', task=task, output=self.tmp / 'budget',
                           registry=registry(), boundaries=self.boundaries(), client=client,
                           policy=host.Policy(max_numerical_runs=1,
                                              resource_guards={'timeout_seconds': 60}),
                           experience=experience(self.tmp / 'input-manifest.json'), rules=RULES)
        self.assertEqual(receipt['stop_reason'], 'run_budget')
        self.assertEqual(len(receipt['executed_actions']), 1)

        def broken(*args, **kwargs):
            raise RuntimeError('synthetic numerical failure')

        client = StubClient([run_proposal('cfg-dense-l2')])
        receipt = dispatch(arm_id='L', task=task, output=self.tmp / 'fail', registry=registry(),
                           boundaries=(host.NumericalBoundary(broken),
                                       host.ScoreBoundary(score_stub)), client=client,
                           policy=host.Policy(resource_guards={'timeout_seconds': 60}),
                           experience=experience(self.tmp / 'input-manifest.json'), rules=RULES)
        self.assertEqual(receipt['error'], 'numerical_failed')
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(len(receipt['executed_actions']), 1)
        self.assertEqual(receipt['executed_actions'][0]['exit_status'], 'failed')
        self.assertIsNone(receipt['executed_actions'][0]['score'])
        self.assertEqual(receipt['feedback_history'], [])

    def test_wall_clock_budget_binds(self):
        """The receipt advertises wall_clock_s as arm-equal, so it must bind.

        Regression: the loop checked max_steps, max_numerical_runs,
        max_model_calls and max_rejections but never the clock, so a stalled
        arm ran unbounded while another was cut off at max_steps.
        """
        task = task_for(self.tmp)
        client = StubClient([run_proposal('cfg-dense-l2')] * 4)
        receipt = dispatch(arm_id='L', task=task, output=self.tmp / 'clock',
                           registry=registry(), boundaries=self.boundaries(), client=client,
                           policy=host.Policy(wall_clock_s=0.5, max_steps=8,
                                              resource_guards={'timeout_seconds': 60}),
                           experience=experience(self.tmp / 'input-manifest.json'), rules=RULES)
        self.assertEqual(receipt['stop_reason'], 'time_budget')
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['error'], 'time_budget')
        self.assertLess(len(receipt['executed_actions']), 4)
        # The limit that fired must still be the one declared to be arm-equal.
        self.assertEqual(receipt['capability']['budget']['wall_clock_s'], 0.5)

    def test_raised_numerical_call_keeps_cost_and_nested_evidence(self):
        def broken(scene_id, branch, checked, output, guards, **kwargs):
            root = Path(output)
            (root / 'delivery/mesh').mkdir(parents=True)
            (root / 'failure.json').write_text('{"status":"failed"}')
            (root / 'delivery/mesh/request.json').write_text('{"parameters":{"decimate":0.25}}')
            (root / 'delivery/mesh/numerical.log').write_text('synthetic tool failure\n')
            (root / 'delivery/mesh/command.json').write_text(
                json.dumps({'argv': ['ReconstructMesh', '--decimate', '0.25']}))
            (root / 'delivery/supervision.json').write_text(
                json.dumps({'peak_worker_process_group_rss_bytes': 12345}))
            raise RuntimeError('synthetic failure after work')

        task = task_for(self.tmp)
        out = self.tmp / 'numerical-raised'
        receipt = dispatch(arm_id='L', task=task, output=out, registry=registry(),
                           boundaries=(host.NumericalBoundary(broken), host.ScoreBoundary(score_stub)),
                           client=StubClient([run_proposal('cfg-dense-l2')]), policy=host.Policy(),
                           experience=experience(self.tmp / 'input-manifest.json'))
        cost = receipt['cost']
        self.assertEqual((cost['numerical_runs'], cost['score_calls']), (1, 0))
        self.assertGreater(cost['numerical_wall_clock_s'], 0)
        self.assertEqual(cost['score_wall_clock_s'], 0)
        self.assertEqual(cost['peak_rss_bytes'], 12345)
        action = receipt['executed_actions'][0]
        artifacts = action['numerical']['artifacts']
        self.assertEqual(artifacts['failure.json']['sha256'], receipts.sha256_file(
            out / 'step-0000/numerical/failure.json'))
        self.assertIn('delivery/mesh/request.json', artifacts)
        self.assertIn('delivery/mesh/numerical.log', artifacts)
        self.assertEqual(action['argv']['delivery/mesh'], ['ReconstructMesh', '--decimate', '0.25'])
        self.assertIsNone(action['score'])
        self.assertEqual(receipt['error'], 'numerical_failed')

    def test_raised_score_call_keeps_cost_and_failure_binding(self):
        def broken(scene_id, output, guards, **kwargs):
            root = Path(output)
            root.mkdir()
            (root / 'failure.json').write_text('{"status":"failed","reason":"fixture"}')
            (root / 'supervision.json').write_text(
                json.dumps({'peak_worker_rss_bytes': 30000000}))
            raise RuntimeError('synthetic scoring timeout')

        task = task_for(self.tmp)
        out = self.tmp / 'score-raised'
        receipt = dispatch(arm_id='L', task=task, output=out, registry=registry(),
                           boundaries=(host.NumericalBoundary(numerical_stub), host.ScoreBoundary(broken)),
                           client=StubClient([run_proposal('cfg-dense-l2')]), policy=host.Policy(),
                           experience=experience(self.tmp / 'input-manifest.json'))
        cost = receipt['cost']
        self.assertEqual((cost['numerical_runs'], cost['score_calls']), (1, 1))
        self.assertGreater(cost['numerical_wall_clock_s'], 0)
        self.assertGreater(cost['score_wall_clock_s'], 0)
        self.assertEqual(cost['peak_rss_bytes'], 30000000)
        score = receipt['executed_actions'][0]['score']
        self.assertEqual(score['status'], 'failed')
        self.assertEqual(score['artifacts']['failure.json']['sha256'], receipts.sha256_file(
            out / 'step-0000/score/failure.json'))
        self.assertEqual(receipt['error'], 'scoring_failed')

    def test_rss_is_max_observed_across_stages_and_calls(self):
        def numerical(*args, **kwargs):
            result = numerical_stub(*args, **kwargs)
            root = Path(result['result']).parent
            (root / 'F-U/supervision.json').write_text(
                json.dumps({'peak_worker_process_group_rss_bytes': 40000000}))
            (root / 'delivery/supervision.json').write_text(
                json.dumps({'peak_worker_process_group_rss_bytes': 20000000}))
            return result

        def score(*args, **kwargs):
            result = score_stub(*args, **kwargs)
            root = Path(result['result']).parent
            (root / 'supervision.json').write_text(json.dumps({
                'peak_worker_rss_bytes': 50000000 if len(SCORES) == 1 else 10000000}))
            return result

        task = task_for(self.tmp)
        out = self.tmp / 'peaks'
        receipt = dispatch(arm_id='L', task=task, output=out, registry=registry(),
                           boundaries=(host.NumericalBoundary(numerical), host.ScoreBoundary(score)),
                           client=StubClient([run_proposal('cfg-dense-l2')] * 2), policy=host.Policy(),
                           experience=experience(self.tmp / 'input-manifest.json'))
        self.assertEqual((receipt['cost']['numerical_runs'], receipt['cost']['score_calls']), (2, 2))
        self.assertEqual(receipt['cost']['peak_rss_bytes'], 50000000)
        self.assertEqual(receipt['cost']['peak_rss_source'],
                         str(out / 'step-0000/score/supervision.json') + '#peak_worker_rss_bytes')

    def test_unobserved_rss_stays_null(self):
        def numerical(*args, **kwargs):
            result = numerical_stub(*args, **kwargs)
            root = Path(result['result']).parent
            (root / 'delivery/supervision.json').write_text('{"peak_worker_process_group_rss_bytes":null}')
            (root / 'F-U/supervision.json').write_text('{"peak_worker_rss_bytes":true}')
            return result

        task = task_for(self.tmp)
        receipt = dispatch(arm_id='L', task=task, output=self.tmp / 'unknown-rss', registry=registry(),
                           boundaries=(host.NumericalBoundary(numerical), host.ScoreBoundary(score_stub)),
                           client=StubClient([run_proposal('cfg-dense-l2')]), policy=host.Policy(),
                           experience=experience(self.tmp / 'input-manifest.json'))
        self.assertIsNone(receipt['cost']['peak_rss_bytes'])
        self.assertIsNone(receipt['cost']['peak_rss_source'])

    def test_wall_clock_budget_is_arm_equal(self):
        task = task_for(self.tmp)
        policy = host.Policy(wall_clock_s=14400.0)
        records = {a.arm: host.capabilities(a, task, policy, registry())
                   for a in host.arm_definitions(host.geopilot_procedure())}
        self.assertEqual(host.require_equal_capability(records), 'PASS')
        self.assertEqual({r['budget']['wall_clock_s'] for r in records.values()}, {14400.0})

    def test_unreadable_score_result_is_recorded_not_raised(self):
        """A missing or corrupt score result must not cost the whole receipt.

        Regression: reading the scorer's result sat outside the delivery guard,
        so FileNotFoundError / JSONDecodeError escaped dispatch() and no receipt
        was written at all.
        """
        def scoring_to_nowhere(scene_id, out, guards, **kwargs):
            Path(out).mkdir(parents=True, exist_ok=True)
            return {'status': 'valid', 'result': str(Path(out) / 'absent.json'),
                    'sha256': 'c' * 64}

        task = task_for(self.tmp)
        client = StubClient([run_proposal('cfg-dense-l2')])
        receipt = dispatch(arm_id='L', task=task, output=self.tmp / 'delivery',
                           registry=registry(),
                           boundaries=(host.NumericalBoundary(numerical_stub),
                                       host.ScoreBoundary(scoring_to_nowhere)),
                           client=client,
                           policy=host.Policy(resource_guards={'timeout_seconds': 60}),
                           experience=experience(self.tmp / 'input-manifest.json'), rules=RULES)
        self.assertEqual(receipt['error'], 'delivery_invalid')
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['executed_actions'][0]['exit_status'], 'failed')
        # The receipt is still written, which is the whole point of recording.
        self.assertTrue((self.tmp / 'delivery' / 'receipt.json').is_file())

    def test_usage_absences_stay_null(self):
        self.assertEqual(receipts.normalise_usage(None)['source'], 'unobserved')
        cost = receipts.new_cost()
        receipts.add_usage(cost, receipts.normalise_usage(None))
        self.assertIsNone(cost['output_tokens'])
        self.assertEqual(cost['calls_without_usage'], 1)
        cost = receipts.new_cost()
        receipts.add_usage(cost, receipts.normalise_usage(
            [{'input_tokens': 5, 'output_tokens': 2, 'output_token_details': {'reasoning': 3}}]))
        self.assertEqual((cost['input_tokens'], cost['output_tokens'], cost['total_tokens']), (5, 2, 7))
        self.assertEqual(cost['reasoning_tokens'], 3)
        self.assertEqual(cost['usage_field_map']['total_tokens'], 'derived_input_plus_output')
        receipt = {'schema': receipts.RECEIPT_SCHEMA}
        with self.assertRaises(ValueError):
            receipts.require_receipt(receipt)

    def test_nothing_real_is_imported_or_called(self):
        self.assertNotIn('outer_numerical', sys.modules)
        self.assertNotIn('outer_score', sys.modules)
        self.assertNotIn('publisher_raster_scenes23', sys.modules)
        for boundary in (host.NumericalBoundary, host.ScoreBoundary):
            self.assertFalse(hasattr(boundary, 'execute'))
        client = host.HttpModelClient(endpoint='https://example.invalid/v1', api_key='k',
                                      model='stub-model-v1')
        self.assertEqual(client.settings(), {
            'endpoint': 'https://example.invalid/v1/chat/completions', 'model': 'stub-model-v1',
            'temperature': 0.2, 'max_tokens': None, 'reasoning_split': True,
            'timeout_s': 120.0, 'max_response_bytes': 2097152, 'mode': 'live'})
        self.assertNotIn('api_key', json.dumps(client.settings()))
        for kwargs in ({'endpoint': 'http://plain.invalid', 'api_key': 'k', 'model': 'm'},
                       {'endpoint': 'https://x.invalid', 'api_key': '', 'model': 'm'},
                       {'endpoint': 'https://x.invalid', 'api_key': 'k', 'model': 'bad model'},
                       {'endpoint': 'https://x.invalid', 'api_key': 'k', 'model': 'm',
                        'max_tokens': 0}):
            with self.assertRaises(ValueError):
                host.HttpModelClient(**kwargs)

    def test_registry_adapter_is_duck_typed(self):
        table = {'cfg-a': {'mesh': {'decimate': 0.5}}, 'cfg-b': {'mesh': {'decimate': 0.25}}}

        def key(value):
            return value if isinstance(value, str) else value['id']

        class FakeConfig:
            REGISTERED_CONFIGS = ({'id': 'cfg-a'}, {'id': 'cfg-b'})

            @staticmethod
            def require_registered(value):
                if key(value) not in table:
                    raise ValueError('unregistered_config')
                return key(value)

            @staticmethod
            def resolve(value):
                return dict(table[key(value)])

        reg = host.outer_config_registry(FakeConfig)
        self.assertEqual(reg.ids, ('cfg-a', 'cfg-b'))
        self.assertEqual(reg.identify('cfg-b'), 'cfg-b')
        self.assertEqual(reg.identify({'id': 'cfg-b'}), 'cfg-b')
        self.assertEqual(reg.effective('cfg-b')['mesh'], {'decimate': 0.25})
        self.assertEqual(len(reg.membership_sha256), 64)
        with self.assertRaises(ValueError):
            reg.effective('cfg-c')
        with self.assertRaises(ValueError):
            reg.identify({'id': 'cfg-c'})

    def test_binds_to_the_real_registered_set_when_present(self):
        try:
            import outer_config
        except ImportError:
            self.skipTest('outer_config is written by a parallel task')
        reg = host.outer_config_registry(outer_config)
        self.assertEqual(tuple(reg.ids), tuple(e['id'] for e in outer_config.REGISTERED_CONFIGS))
        self.assertIsNotNone(reg.audit)
        for entry in outer_config.REGISTERED_CONFIGS:
            self.assertEqual(reg.identify(entry['id']), entry['id'])
            self.assertEqual(reg.effective(entry['id']), outer_config.resolve(entry['id']))
        with self.assertRaises(ValueError):
            reg.identify('mesh-decimate-099-v1')
        with self.assertRaises(ValueError):
            reg.identify({'mesh': {'decimate': 0.99}})
        task = task_for(self.tmp)
        client = StubClient([run_proposal('mesh-decimate-025-v1'),
                             {'tool': 'stop', 'reason': 'end'}])
        receipt = dispatch(arm_id='G0', task=task, output=self.tmp / 'real', registry=reg,
                           boundaries=self.boundaries(), client=client,
                           policy=host.Policy(resource_guards={'timeout_seconds': 60}),
                           experience=experience(self.tmp / 'input-manifest.json'))
        self.assertEqual(receipt['executed_actions'][0]['config_id'], 'mesh-decimate-025-v1')
        self.assertEqual(CALLS[0]['config'], 'mesh-decimate-025-v1')

    def test_dead_knob_audit_failure_is_a_host_refusal(self):
        reg = registry()

        def audit(value):
            if value == 'cfg-dense-l2':
                raise ValueError('knob does not reach the numerical chain')

        reg = host.ConfigRegistry(ids=reg.ids, require=reg.require, resolve=reg.resolve,
                                  identify=reg.identify,
                                  membership_sha256=reg.membership_sha256, audit=audit)
        task = task_for(self.tmp)
        client = StubClient([run_proposal('cfg-dense-l2'),
                             run_proposal('cfg-mesh-dec-0.25'),
                             {'tool': 'stop', 'reason': 'end'}])
        receipt = dispatch(arm_id='L', task=task, output=self.tmp / 'audit', registry=reg,
                           boundaries=self.boundaries(), client=client,
                           policy=host.Policy(resource_guards={'timeout_seconds': 60}),
                           experience=experience(self.tmp / 'input-manifest.json'))
        self.assertEqual(len(receipt['rejected_actions']), 1)
        self.assertEqual([c['config'] for c in CALLS], ['cfg-mesh-dec-0.25'])

    def test_proposal_parsing_is_strict(self):
        good = json.dumps(run_proposal('cfg-dense-l2'))
        self.assertEqual(host.parse_proposal('```json\n' + good + '\n```')['tool'], 'run_numerical')
        for bad in ('', 'prose then ' + good, json.dumps({'tool': 'run_numerical'}),
                    json.dumps({'tool': 'run_numerical', 'config': 'a', 'hypothesis': 'h',
                                'competing_explanation': 'c', 'expected_observations': []}),
                    json.dumps({'tool': 'stop'}),
                    json.dumps({'tool': 'invoke', 'config': 'a'}),
                    json.dumps({'tool': 'run_numerical', 'config': 'a', 'hypothesis': 'h',
                                'competing_explanation': 'c', 'expected_observations': ['o'],
                                'extra': 1})):
            with self.assertRaises(ValueError):
                host.parse_proposal(bad)

    def diagnostic_episode(self, transcripts=(), *, arm='L', task=None, diagnostics=None,
                           rules=None, read_only=True, output='diagnostic', policy=None,
                           structured_state=False):
        task = task or task_for(self.tmp)
        diagnostics = diagnostics if diagnostics is not None else {
            'camera.summary': {'description': 'Read image and camera group facts.',
                               'source_ids': ['input_manifest'],
                               'run': lambda: {'image_count': 327, 'camera_groups': 1}}}
        client = StubClient(transcripts)
        receipt = dispatch(arm_id=arm, task=task, output=self.tmp / output,
                           registry=registry(), boundaries=self.boundaries(), client=client,
                           policy=policy or host.Policy(), diagnostics=diagnostics,
                           read_only=read_only, rules=rules or RULES,
                           structured_state=structured_state,
                           experience=experience(self.tmp / 'input-manifest.json'))
        return receipt, client

    def test_diagnostic_menu_and_facts_are_identical_capabilities_for_four_arms(self):
        task = task_for(self.tmp)
        proposal = {'tool': 'inspect', 'name': 'camera.summary', 'reason': 'check grouping'}
        decision = json.dumps({'next_action': 'check camera estimation', 'is_mesh': False})
        stop = {'tool': 'stop', 'reason': decision}
        rules = [{'id': 'D1', 'after': 'start', **proposal},
                 {'id': 'D2', 'after': 'diagnostic', **stop}]
        records, menus = {}, []
        for arm in host.ARM_IDS:
            with self.subTest(arm=arm):
                receipt, client = self.diagnostic_episode(
                    [proposal, stop], task=task, arm=arm, rules=rules, output='diagnostic-' + arm)
                records[arm] = receipt['capability']
                self.assertEqual(receipt['status'], 'completed')
                self.assertEqual(receipt['tool_calls'][-1]['arguments']['reason'], decision)
                self.assertEqual(receipt['cost']['diagnostic_calls'], 1)
                self.assertGreater(receipt['cost']['diagnostic_wall_clock_s'], 0)
                self.assertEqual(receipt['cost']['numerical_runs'], 0)
                self.assertEqual(receipt['cost']['score_calls'], 0)
                self.assertEqual(receipt['cost']['model_calls'], 0 if arm == 'R' else 2)
                call = receipt['tool_calls'][0]
                self.assertIsNone(call['score'])
                self.assertIsNone(call['numerical'])
                self.assertEqual(call['feedback']['kind'], 'diagnostic')
                self.assertEqual(call['feedback']['facts']['image_count'], 327)
                detail = receipt['executed_actions'][0]['diagnostic']
                self.assertEqual(detail['raw_result_sha256'], digest(call['feedback']['facts']))
                self.assertEqual(detail['result_sha256'], receipts.sha256_file(
                    Path(detail['result_path'])))
                source = detail['source_bindings']['input_manifest']
                self.assertEqual(source['sha256'], source['observed_before_sha256'])
                self.assertEqual(source['sha256'], source['observed_after_sha256'])
                self.assertEqual(receipt['turns'][0]['timing']['diagnostic_elapsed_s'],
                                 receipt['cost']['diagnostic_wall_clock_s'])
                receipts.require_receipt(receipt)
                if arm != 'R':
                    first = json.loads(client.payloads[0]['messages'][1]['content'])
                    second = json.loads(client.payloads[1]['messages'][1]['content'])
                    menus.append(first['tools'])
                    self.assertEqual(set(first['tools']), {'inspect', 'stop'})
                    self.assertIsNone(first['previous_feedback'])
                    self.assertEqual(second['previous_feedback'], {'history_index': 0})
                    self.assertEqual(latest_feedback(second), call['feedback'])
                    self.assertEqual(second['history'], receipt['feedback_history'])
                    self.assertNotIn(str(self.tmp), client.payloads[1]['messages'][1]['content'])
        self.assertEqual(host.require_equal_capability(records), 'PASS')
        self.assertTrue(all(menu == menus[0] for menu in menus))
        self.assertEqual(CALLS, [])
        self.assertEqual(SCORES, [])

    def test_feedback_body_is_sent_once_and_all_history_can_be_restored(self):
        marker = 'unique-permitted-camera-fact-' + 'x' * 10000
        facts = [{'camera': {'description': marker, 'dimensions': [6205, 4136]},
                  'all_ids': ['image-1', 'image-2']},
                 {'camera': {'description': 'second observation', 'dimensions': [6203, 4134]},
                  'all_ids': ['image-3']}]
        pending = iter(copy.deepcopy(facts))
        diagnostics = {'camera.full': {'description': 'Read complete camera facts.',
                                       'source_ids': ['input_manifest'],
                                       'run': lambda: next(pending)}}
        proposal = {'tool': 'inspect', 'name': 'camera.full', 'reason': 'check camera facts'}
        receipt, client = self.diagnostic_episode([proposal, proposal], diagnostics=diagnostics)
        states = [json.loads(payload['messages'][1]['content']) for payload in client.payloads]
        self.assertEqual(states[0]['history'], [])
        self.assertIsNone(latest_feedback(states[0]))
        for step in (1, 2):
            with self.subTest(step=step):
                self.assertEqual(states[step]['history'], receipt['feedback_history'][:step])
                self.assertEqual(states[step]['previous_feedback'], {'history_index': step - 1})
                self.assertEqual(latest_feedback(states[step])['facts'], facts[step - 1])
                self.assertEqual(json.dumps(client.payloads[step]).count(marker), 1)
                self.assertIn('history[history_index].feedback',
                              client.payloads[step]['messages'][0]['content'])
        self.assertEqual([h['feedback']['facts'] for h in states[-1]['history']], facts)
        self.assertEqual([h['step_index'] for h in states[-1]['history']], [0, 1])
        self.assertEqual(receipt['cost']['diagnostic_calls'], 2)

    def test_latest_feedback_without_history_is_not_silently_discarded(self):
        arm = host.arm_definitions(host.geopilot_procedure())[1]
        with self.assertRaisesRegex(ValueError, 'Latest feedback missing from history'):
            host._agent_state(arm, task_for(self.tmp), 1, host.Policy(), registry(), {},
                              [], {'status': 'rejected'}, None, [])

    def test_structured_state_tool_has_equal_capabilities_and_real_persistence(self):
        task = task_for(self.tmp)
        inspect = {'tool': 'inspect', 'name': 'camera.summary', 'reason': 'observe grouping'}
        stop = {'tool': 'stop', 'reason': 'No numerical outcome is claimed.'}
        records = {}
        for arm in host.ARM_IDS:
            memories = [{'entry_id': 'E-P', 'applicability_refs': [], 'status': 'unresolved'}] \
                if arm == 'GE' else []
            first = update_state(claim_row(), memories=memories)
            last = update_state(claim_row(observations=['step-1']), memories=memories)
            rules = [{'id': 'S1', **first}, {'id': 'D1', 'after': 'decision_state', **inspect},
                     {'id': 'S2', 'after': 'diagnostic', **last},
                     {'id': 'done', 'after': 'decision_state', **stop}]
            with self.subTest(arm=arm):
                receipt, client = self.diagnostic_episode(
                    [first, inspect, last, stop], task=task, arm=arm, rules=rules,
                    structured_state=True, output='state-' + arm)
                records[arm] = receipt['capability']
                self.assertEqual(receipt['status'], 'completed')
                self.assertEqual(receipt['cost']['decision_state_calls'], 2)
                self.assertEqual(receipt['cost']['diagnostic_calls'], 1)
                self.assertEqual(receipt['cost']['numerical_runs'], 0)
                self.assertEqual(receipt['cost']['model_calls'], 0 if arm == 'R' else 4)
                binding = receipt['instruction']['structured_state']
                self.assertEqual(binding['enforcement'], 'required' if arm in ('G0', 'GE')
                                 else 'optional')
                for event in binding['state_events']:
                    self.assertEqual(receipts.sha256_file(Path(event['path'])), event['sha256'])
                final = json.loads(Path(binding['final_state']['path']).read_text())
                self.assertEqual(set(final['state']['observations']), {'step-1'})
                self.assertFalse(final['state']['claims']['C1']['needs_update'])
                self.assertEqual(final['frozen_experience'],
                                 experience(self.tmp / 'input-manifest.json')['GE'] if arm == 'GE' else {})
                self.assertTrue(all(receipts.sha256_file(Path(p)) == sha
                                    for p, sha in binding['source_bindings'].items()))
                if arm != 'R':
                    states = [json.loads(p['messages'][1]['content']) for p in client.payloads]
                    self.assertTrue(states[2]['decision_state']['claims']['C1']['needs_update'])
                    self.assertFalse(states[3]['decision_state']['claims']['C1']['needs_update'])
                    self.assertNotIn('step-0', states[3]['decision_state']['observations'])
                    self.assertIsNone(states[0]['budget_remaining']['steps'])
                receipts.require_receipt(receipt)
        self.assertEqual(host.require_equal_capability(records), 'PASS')

    def test_required_state_gates_actions_and_stop_until_real_observation_is_updated(self):
        inspect = {'tool': 'inspect', 'name': 'camera.summary', 'reason': 'check input'}
        stop = {'tool': 'stop', 'reason': 'Record the supported result.'}
        transcript = [run_proposal('cfg-dense-l2'), stop, update_state(claim_row()), inspect,
                      run_proposal('cfg-dense-l2'), stop,
                      update_state(claim_row(observations=['step-3'])),
                      run_proposal('cfg-dense-l2'), stop,
                      update_state(claim_row(status='supported', observations=['step-3'],
                                             results=['step-7'])), stop]
        receipt, client = self.diagnostic_episode(transcript, arm='G0', read_only=False,
                                                  structured_state=True)
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['cost']['numerical_runs'], 1)
        self.assertEqual(len(CALLS), 1)
        self.assertEqual([r['reason'] for r in receipt['rejected_actions']],
                         ['decision_claim_required', 'decision_claim_required',
                          'decision_update_required', 'decision_update_required',
                          'decision_update_required'])
        final = json.loads(client.payloads[-1]['messages'][1]['content'])
        self.assertEqual(final['decision_action_bindings'], {'C1': ['step-7']})
        self.assertEqual(set(final['decision_state']['observations']), {'step-3', 'step-7'})
        self.assertEqual(final['decision_state']['claims']['C1']['observed_result_refs'], ['step-7'])
        self.assertFalse(final['decision_state']['claims']['C1']['needs_update'])

    def test_optional_state_does_not_impose_geopilot_policy_on_general_arm(self):
        receipt, client = self.diagnostic_episode(
            [run_proposal('cfg-dense-l2'), {'tool': 'stop', 'reason': 'done'}], arm='L',
            read_only=False, structured_state=True)
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['cost']['decision_state_calls'], 0)
        self.assertEqual(receipt['cost']['numerical_runs'], 1)
        final = json.loads(client.payloads[-1]['messages'][1]['content'])
        self.assertEqual(set(final['decision_state']['observations']), {'step-0'})
        self.assertEqual(final['decision_state']['claims'], {})

    def test_structured_state_is_explicit_opt_in_and_rejects_unmatched_action(self):
        receipt, client = self.diagnostic_episode(
            [{'tool': 'stop', 'reason': 'historical behaviour'}], output='without-state')
        initial = json.loads(client.payloads[0]['messages'][1]['content'])
        self.assertNotIn('decision_state', initial)
        self.assertNotIn('update_decision_state', initial['tools'])
        self.assertNotIn('decision_state_calls', receipt['cost'])
        self.assertFalse((self.tmp / 'without-state' / 'decision-state').exists())
        self.assertEqual(receipt['capability']['decision_state_enforcement'], 'disabled')
        receipt, _ = self.diagnostic_episode(
            [update_state(claim_row()), run_proposal('cfg-mesh-dec-0.25'),
             {'tool': 'stop', 'reason': 'The available prediction was not executed.'}],
            arm='G0', read_only=False, structured_state=True, output='wrong-action')
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['rejected_actions'][0]['reason'], 'untested_action_prediction_required')
        self.assertEqual(CALLS, [])

    def test_source_drift_requires_update_and_does_not_let_stale_refs_through(self):
        task = task_for(self.tmp)
        original_file = Path(task['information_sources']['input_manifest']['path'])
        inspect = {'tool': 'inspect', 'name': 'camera.summary', 'reason': 'read source'}
        before = claim_row(observations=['input_manifest'])
        client = StubClient([inspect, update_state(before),
                             {'tool': 'stop', 'reason': 'must be blocked after drift'},
                             update_state(before), update_state(claim_row(status='unresolved')),
                             {'tool': 'stop', 'reason': 'The inspected source has changed.'}])
        original_complete = client.complete
        def complete(payload, *, clock=None):
            reply = original_complete(payload, clock=clock)
            if len(client.payloads) == 3:
                original_file.write_text('{"changed_input": true}')
            return reply
        # Mutate between the last valid observation and the next host state refresh.
        def inspect_source():
            return {'image_count': 327}
        with patch.object(client, 'complete', side_effect=complete):
            receipt = dispatch(
                arm_id='G0', task=task, output=self.tmp / 'source-change', registry=registry(),
                boundaries=self.boundaries(), client=client, policy=host.Policy(),
                structured_state=True, experience=experience(original_file),
                diagnostics={'camera.summary': {'description': 'Read permitted input facts.',
                                                'source_ids': ['input_manifest'], 'run': inspect_source}})
        self.assertEqual(receipt['status'], 'completed')
        final = json.loads(Path(receipt['instruction']['structured_state']['final_state']['path']).read_text())
        self.assertEqual(final['state']['stale_observation_ids'], ['step-0'])
        self.assertEqual(final['state']['claims']['C1']['status'], 'unresolved')
        self.assertIn('unknown_or_stale_observation_reference',
                      [r['reason'] for r in receipt['rejected_actions']])

    def test_state_update_cannot_promote_or_replace_frozen_experience_text(self):
        memories = [{'entry_id': 'E-P', 'applicability_refs': [], 'status': 'unresolved'}]
        bad = [{**memories[0], 'conditions': 'Rewritten as universally true.'}]
        receipt, client = self.diagnostic_episode(
            [update_state(claim_row(), memories=bad),
             update_state(claim_row(), memories=memories),
             {'tool': 'stop', 'reason': 'Applicability remains unresolved.'}],
            arm='GE', structured_state=True)
        self.assertEqual(receipt['rejected_actions'][0]['reason'], 'invalid_experience_row')
        final = json.loads(client.payloads[-1]['messages'][1]['content'])
        self.assertIsNone(final['decision_state']['experience']['E-P']['conditions'])
        self.assertEqual(final['cross_task_experience']['entries'][0]['lesson'], 'raise resolution level')
        self.assertEqual(final['decision_state']['observations'], {})

    def test_code_drift_during_model_request_invalidates_state_and_prevents_execution(self):
        task = task_for(self.tmp)
        client = StubClient([update_state(claim_row()), run_proposal('cfg-dense-l2')])
        drifted = [False]
        original_complete, original_sha = client.complete, host.sha256_file
        state_source = host.ENTRY.with_name('decision_state.py')
        def complete(payload, *, clock=None):
            reply = original_complete(payload, clock=clock)
            if len(client.payloads) == 2:
                drifted[0] = True
            return reply
        def observed_sha(path):
            return 'f' * 64 if drifted[0] and Path(path) == state_source else original_sha(path)
        with patch.object(client, 'complete', side_effect=complete), \
                patch.object(host, 'sha256_file', side_effect=observed_sha):
            receipt = dispatch(
                arm_id='G0', task=task, output=self.tmp / 'code-change', registry=registry(),
                boundaries=self.boundaries(), client=client, policy=host.Policy(),
                structured_state=True, experience=experience(self.tmp / 'input-manifest.json'))
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['stop_reason'], 'code_source_drift')
        self.assertEqual(receipt['error'], 'invalid_schema')
        self.assertEqual(receipt['cost']['model_calls'], 2)
        self.assertEqual(receipt['cost']['numerical_runs'], 0)
        self.assertEqual(CALLS, [])
        record = receipt['instruction']['structured_state']
        self.assertEqual(record['code_drift']['observed'][str(state_source)], 'f' * 64)
        self.assertEqual(record['code_drift']['expected'][str(state_source)], original_sha(state_source))
        final = json.loads(Path(record['final_state']['path']).read_text())
        self.assertEqual(final['trigger'], 'code_source_drift')
        self.assertTrue(final['state']['claims']['C1']['needs_update'])
        self.assertEqual(final['state']['claims']['C1']['status'], 'unresolved')
        self.assertEqual(receipt['tool_calls'][-1]['exit_status'], 'not_executed')
        receipts.require_receipt(receipt)

    def test_unknown_state_and_fabricated_observations_are_rejected_and_recoverable(self):
        invalid = [update_state({**claim_row(), 'status': 'verified_forever'}),
                   update_state(claim_row(status='supported', results=['step-0'])),
                   update_state(claim_row(observations=['input_manifest'])),
                   {**update_state(claim_row()), 'path': '/tmp/invented'},
                   update_state(claim_row(), memories=[{'entry_id': [],
                                                       'applicability_refs': [], 'status': 'unresolved'}])]
        receipt, client = self.diagnostic_episode(
            [*invalid, update_state(claim_row(status='unresolved')),
             {'tool': 'stop', 'reason': 'Evidence is not yet available.'}],
            arm='G0', structured_state=True)
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(len(receipt['rejected_actions']), len(invalid))
        self.assertEqual(receipt['cost']['decision_state_calls'], len(invalid) + 1)
        final = json.loads(client.payloads[-1]['messages'][1]['content'])
        self.assertEqual(final['decision_state']['observations'], {})
        self.assertEqual(final['decision_state']['claims']['C1']['status'], 'unresolved')
        self.assertEqual(CALLS, [])

    def test_run_backfill_requires_run_reference_and_preserves_predeclared_prediction(self):
        inspect = {'tool': 'inspect', 'name': 'camera.summary', 'reason': 'check source'}
        before = claim_row(observations=['input_manifest'])
        changed = {**before, 'predicted_observation': 'A prediction changed after execution.',
                   'status': 'supported', 'observed_result_refs': ['step-2']}
        old_source_only = {**before, 'status': 'supported', 'observed_result_refs': ['input_manifest']}
        after = {**before, 'status': 'unresolved', 'observed_result_refs': ['step-2']}
        receipt, client = self.diagnostic_episode(
            [inspect, update_state(before), run_proposal('cfg-dense-l2'), update_state(changed),
             update_state(old_source_only), update_state(before), update_state(after),
             {'tool': 'stop', 'reason': 'The actual feedback does not settle the hypothesis.'}],
            arm='G0', structured_state=True, read_only=False)
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual([r['reason'] for r in receipt['rejected_actions']],
                         ['executed_prediction_changed', 'executed_result_reference_required',
                          'executed_claim_cannot_be_not_tested'])
        final = json.loads(client.payloads[-1]['messages'][1]['content'])
        self.assertEqual(final['decision_state']['claims']['C1']['predicted_observation'],
                         before['predicted_observation'])
        self.assertEqual(final['decision_state']['claims']['C1']['status'], 'unresolved')

    def test_failed_numerical_attempt_leaves_actual_pending_state_and_receipt(self):
        task = task_for(self.tmp)
        client = StubClient([update_state(claim_row()), run_proposal('cfg-dense-l2')])
        def fail(*args, **kwargs):
            raise RuntimeError('fixture numerical fault')
        receipt = dispatch(arm_id='G0', task=task, output=self.tmp / 'failed-state',
                           registry=registry(), client=client, policy=host.Policy(),
                           boundaries=(host.NumericalBoundary(fail), host.ScoreBoundary(score_stub)),
                           structured_state=True,
                           experience=experience(self.tmp / 'input-manifest.json'))
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['error'], 'numerical_failed')
        self.assertEqual(receipt['cost']['numerical_runs'], 1)
        self.assertEqual(receipt['cost']['score_calls'], 0)
        final = json.loads(Path(receipt['instruction']['structured_state']['final_state']['path']).read_text())
        self.assertEqual(set(final['state']['observations']), {'step-1'})
        self.assertTrue(final['state']['claims']['C1']['needs_update'])
        self.assertEqual(final['action_bindings'], {'C1': ['step-1']})
        self.assertEqual(receipt['tool_calls'][-1]['feedback']['status'], 'failed')

    def test_diagnostic_then_numerical_is_available_to_model_and_rule_arms(self):
        task = task_for(self.tmp)
        proposal = {'tool': 'inspect', 'name': 'camera.summary', 'reason': 'check grouping'}
        rules = [{'id': 'D1', **proposal}, {'id': 'N1', 'after': 'diagnostic',
                                         **run_proposal('cfg-dense-l2')}]
        for arm in ('R', 'G0'):
            receipt, client = self.diagnostic_episode(
                [proposal, run_proposal('cfg-dense-l2')], task=task, arm=arm,
                rules=rules, read_only=False, output='full-' + arm)
            self.assertEqual(receipt['cost']['diagnostic_calls'], 1)
            self.assertEqual(receipt['cost']['numerical_runs'], 1)
            self.assertEqual(receipt['cost']['score_calls'], 1)
            self.assertEqual(receipt['feedback_history'][0]['feedback']['kind'], 'diagnostic')
            self.assertIn('surface_metrics', receipt['feedback_history'][1]['feedback'])
            if arm != 'R':
                final = json.loads(client.payloads[-1]['messages'][1]['content'])
                self.assertEqual(final['previous_feedback'], {'history_index': 1})
                self.assertEqual(latest_feedback(final), receipt['feedback_history'][1]['feedback'])
        self.assertEqual(len(CALLS), 2)
        self.assertEqual(len(SCORES), 2)

    def test_read_only_refuses_numerical_proposals_for_model_and_rules(self):
        task = task_for(self.tmp)
        for arm in ('R', 'L'):
            receipt, _ = self.diagnostic_episode(
                [run_proposal('cfg-dense-l2')], task=task, arm=arm,
                output='read-only-' + arm, policy=host.Policy(max_numerical_runs=0))
            self.assertEqual(receipt['tool_calls'][0]['exit_status'], 'rejected')
            self.assertEqual(receipt['rejected_actions'][0]['reason'], 'unknown_tool')
            self.assertEqual(receipt['cost']['numerical_runs'], 0)
            self.assertEqual(receipt['cost']['score_calls'], 0)
            self.assertEqual(receipt['cost']['diagnostic_calls'], 0)
        self.assertEqual(CALLS, [])
        self.assertEqual(SCORES, [])

    def test_unknown_diagnostic_name_is_rejected_without_callback(self):
        receipt, client = self.diagnostic_episode([
            {'tool': 'inspect', 'name': '../../input-manifest.json', 'reason': 'read this'}])
        self.assertEqual(receipt['tool_calls'][0]['exit_status'], 'rejected')
        self.assertEqual(receipt['rejected_actions'][0]['reason'], 'invalid_arguments')
        self.assertEqual(receipt['cost']['diagnostic_calls'], 0)
        self.assertEqual(receipt['executed_actions'], [])
        state = json.loads(client.payloads[1]['messages'][1]['content'])
        self.assertEqual(latest_feedback(state)['status'], 'rejected')
        self.assertNotIn('../../', json.dumps(state))

    def test_diagnostic_proposals_reject_extra_paths_code_and_invalid_fields(self):
        valid = {'tool': 'inspect', 'name': 'camera.summary', 'reason': 'check grouping'}
        with self.assertRaises(ValueError):
            host.parse_proposal(json.dumps(valid))
        self.assertEqual(host.parse_proposal(json.dumps(valid), allow_inspect=True), valid)
        for bad in ({**valid, 'path': '/tmp/x'}, {**valid, 'code': 'read()'},
                    {**valid, 'name': ['camera.summary']}, {**valid, 'reason': ''},
                    {'tool': 'inspect', 'name': 'camera.summary'}):
            with self.subTest(proposal=bad), self.assertRaises(ValueError):
                host.parse_proposal(json.dumps(bad), allow_inspect=True)

    def test_diagnostic_leaks_are_hashed_but_not_saved_or_sent_as_facts(self):
        task = task_for(self.tmp)
        leaks = [{'nested': {'reference_path': '/private/target.laz'}},
                 {'surface_metrics': METRICS}, {'accuracy_l1_m': 0.1},
                 {'raw_score': {'status': 'valid'}}, {'text': 'arm-GE result'},
                 {'text': 'Bearer credential'}]
        for index, leak in enumerate(leaks):
            with self.subTest(leak=leak):
                diagnostics = {'camera.summary': {'description': 'Read camera facts.',
                                                  'source_ids': ['input_manifest'],
                                                  'run': lambda leak=leak: leak}}
                receipt, client = self.diagnostic_episode(
                    [{'tool': 'inspect', 'name': 'camera.summary', 'reason': 'check grouping'}],
                    task=task, diagnostics=diagnostics, output=f'leak-{index}')
                self.assertEqual(receipt['status'], 'failed')
                self.assertEqual(receipt['error'], 'send_gate')
                self.assertEqual(receipt['cost']['diagnostic_calls'], 1)
                self.assertGreater(receipt['cost']['diagnostic_wall_clock_s'], 0)
                self.assertEqual(len(client.payloads), 1)
                self.assertIsNone(receipt['tool_calls'][0]['feedback']['facts'])
                detail = receipt['executed_actions'][0]['diagnostic']
                self.assertEqual(detail['raw_result_sha256'], digest(leak))
                saved = json.loads(Path(detail['result_path']).read_text())
                self.assertIsNone(saved['feedback']['facts'])
                self.assertIsNone(receipt['tool_calls'][0]['score'])
                self.assertEqual(receipt['cost']['score_calls'], 0)

    def test_diagnostic_exception_records_failure_and_elapsed_without_raw_message(self):
        invoked = []

        def fail():
            invoked.append(True)
            raise RuntimeError('secret provider text must not enter feedback')

        diagnostics = {'camera.summary': {'description': 'Read camera facts.',
                                          'source_ids': ['input_manifest'], 'run': fail}}
        receipt, client = self.diagnostic_episode(
            [{'tool': 'inspect', 'name': 'camera.summary', 'reason': 'check grouping'}],
            diagnostics=diagnostics)
        self.assertEqual(invoked, [True])
        self.assertEqual(receipt['tool_calls'][0]['exit_status'], 'failed')
        self.assertEqual(receipt['tool_calls'][0]['feedback']['reason'], 'diagnostic_failed')
        self.assertGreater(receipt['tool_calls'][0]['elapsed_s'], 0)
        self.assertEqual(receipt['cost']['diagnostic_calls'], 1)
        self.assertIsNone(receipt['executed_actions'][0]['diagnostic']['raw_result_sha256'])
        self.assertNotIn('secret provider', json.dumps(receipt))
        next_state = json.loads(client.payloads[1]['messages'][1]['content'])
        self.assertEqual(latest_feedback(next_state), receipt['tool_calls'][0]['feedback'])
        self.assertIsNone(latest_feedback(next_state)['facts'])
        self.assertEqual(receipt['cost']['score_calls'], 0)

    def test_missing_diagnostic_evidence_writes_receipt_without_calling_extractor(self):
        task = task_for(self.tmp)
        (self.tmp / 'mesh-help.txt').unlink()
        invoked = []
        diagnostics = {'tool.summary': {'description': 'Read supported parameters.',
                                        'source_ids': ['mesh_help'],
                                        'run': lambda: invoked.append(True)}}
        receipt, client = self.diagnostic_episode(
            [{'tool': 'inspect', 'name': 'tool.summary', 'reason': 'check parameters'}],
            task=task, diagnostics=diagnostics)
        self.assertEqual(invoked, [])
        self.assertEqual(receipt['tool_calls'][0]['feedback']['reason'], 'source_missing')
        self.assertEqual(receipt['cost']['diagnostic_calls'], 1)
        self.assertFalse(receipt['executed_actions'][0]['diagnostic']['callback_invoked'])
        self.assertIsNone(receipt['tool_calls'][0]['feedback']['facts'])
        self.assertEqual(receipt['turns'][0]['request']['context_sources']['mesh_help']['status'],
                         'missing')
        self.assertEqual(len(client.payloads), 2)

    def test_source_drift_during_diagnostic_discards_facts(self):
        task = task_for(self.tmp)

        def mutate():
            (self.tmp / 'mesh-help.txt').write_text('changed')
            return {'decimate_supported': True}

        diagnostics = {'tool.summary': {'description': 'Read supported parameters.',
                                        'source_ids': ['mesh_help'], 'run': mutate}}
        receipt, _ = self.diagnostic_episode(
            [{'tool': 'inspect', 'name': 'tool.summary', 'reason': 'check parameters'}],
            task=task, diagnostics=diagnostics)
        detail = receipt['executed_actions'][0]['diagnostic']
        self.assertEqual(detail['error'], 'source_drifted')
        binding = detail['source_bindings']['mesh_help']
        self.assertEqual(binding['sha256'], binding['observed_before_sha256'])
        self.assertNotEqual(binding['sha256'], binding['observed_after_sha256'])
        self.assertIsNone(receipt['tool_calls'][0]['feedback']['facts'])

    def test_source_drift_before_diagnostic_prevents_extractor_call(self):
        task = task_for(self.tmp)
        (self.tmp / 'mesh-help.txt').write_text('changed before inspection')
        invoked = []
        diagnostics = {'tool.summary': {'description': 'Read supported parameters.',
                                        'source_ids': ['mesh_help'],
                                        'run': lambda: invoked.append(True)}}
        receipt, _ = self.diagnostic_episode(
            [{'tool': 'inspect', 'name': 'tool.summary', 'reason': 'check parameters'}],
            task=task, diagnostics=diagnostics)
        self.assertEqual(invoked, [])
        self.assertEqual(receipt['tool_calls'][0]['feedback']['reason'], 'source_drifted')
        self.assertFalse(receipt['executed_actions'][0]['diagnostic']['callback_invoked'])
        self.assertIsNone(receipt['executed_actions'][0]['diagnostic']['raw_result_sha256'])

    def test_invalid_diagnostic_result_cannot_become_feedback(self):
        task = task_for(self.tmp)
        for index, result in enumerate(([1, 2], {'value': float('nan')})):
            diagnostics = {'tool.summary': {'description': 'Read supported parameters.',
                                            'source_ids': ['mesh_help'],
                                            'run': lambda result=result: result}}
            receipt, _ = self.diagnostic_episode(
                [{'tool': 'inspect', 'name': 'tool.summary', 'reason': 'check parameters'}],
                task=task, diagnostics=diagnostics, output=f'invalid-result-{index}')
            self.assertEqual(receipt['tool_calls'][0]['exit_status'], 'failed')
            self.assertIsNone(receipt['tool_calls'][0]['feedback']['facts'])
            self.assertEqual(receipt['cost']['diagnostic_calls'], 1)

    def test_diagnostic_host_menu_requires_bound_sources_and_callable(self):
        task = task_for(self.tmp)
        good = {'description': 'Read facts.', 'source_ids': ['input_manifest'], 'run': lambda: {}}
        for bad in ({**good, 'source_ids': ['not_bound']}, {**good, 'source_ids': []},
                    {**good, 'source_ids': ['input_manifest', 'input_manifest']},
                    {**good, 'run': 'arbitrary command'}, {**good, 'path': '/tmp/x'}):
            with self.subTest(entry=bad), self.assertRaises(ValueError):
                host.diagnostic_tools({'input.summary': bad}, task)
        self.assertEqual(host.diagnostic_tools(None, task), host.TOOL_SCHEMAS)
        self.assertEqual(host.diagnostic_tools({}, task), host.TOOL_SCHEMAS)
        self.assertEqual(set(host.diagnostic_tools(None, task, read_only=True)), {'stop'})


class TransportFaultTests(unittest.TestCase):
    """A provider fault must be recorded, never raised out of the transport.

    Regression: the non-200 branch set only ``http_status``, leaving the body
    buffer unbound, so every 401/429/500 raised UnboundLocalError out of
    ``complete()`` and killed the episode instead of writing a receipt.
    """

    @staticmethod
    def _complete(fault):
        class Opener:
            def open(self, request, timeout=None):
                raise fault

        client = host.HttpModelClient(endpoint='https://example.invalid',
                                      api_key='k', model='m')
        with patch.object(urllib.request, 'build_opener', lambda *a: Opener()):
            return client.complete(b'{}')

    def test_http_error_status_is_recorded_not_raised(self):
        for code in (401, 429, 500):
            with self.subTest(code=code):
                record = self._complete(urllib.error.HTTPError(
                    'https://example.invalid', code, 'fault', {},
                    io.BytesIO(b'{"error":"provider text"}')))
                self.assertEqual(record['status'], 'error')
                self.assertEqual(record['error'], 'response_fault')
                self.assertEqual(record['http_status'], code)

    def test_transport_exception_is_recorded_not_raised(self):
        record = self._complete(OSError('connection reset'))
        self.assertEqual((record['status'], record['error'], record['http_status']),
                         ('error', 'response_fault', None))
        self.assertEqual(record['timing']['transport_error_type'], 'OSError')
        timeout = self._complete(TimeoutError('private transport detail'))
        self.assertEqual(host._raw_view(b'', timeout, None)['transport']['transport_error_type'],
                         'TimeoutError')
        self.assertNotIn('private transport detail', repr(timeout))

    def test_provider_error_body_never_persists(self):
        record = self._complete(urllib.error.HTTPError(
            'https://example.invalid', 429, 'fault', {}, io.BytesIO(b'{"error":"secret"}')))
        self.assertEqual(record['body'], b'')

    def test_recorded_keys_are_exactly_the_client_contract(self):
        self.assertEqual(set(self._complete(OSError('x'))), set(host.CLIENT_KEYS))

    def test_redirect_handler_refuses(self):
        self.assertIsNone(host._NoRedirect().redirect_request(
            None, None, 302, 'Found', {}, 'https://elsewhere.invalid'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
