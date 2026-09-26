"""Checks use synthetic responses against read-only real evidence; never run geometry."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import continuation as m


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = m.old.ROOT / 'out/geopilot-learning-20260923/plan.json'
        cls.context = m.compile_context(cls.plan)

    def decision(self):
        p = copy.deepcopy(self.context['parent'])
        p.update(program_id='p2', parent_id='p0')
        p['nodes'][2]['parameters']['decimate'] = 0.5
        return {'hypothesis': 'synthetic single parameter test', 'expected_effect': 'no research claim',
            'risks': ['fixture'], 'competing_explanations': ['unresolved cause'],
            'evidence_refs': ['attempt-p1'], 'history_update': {'evidence_ids': ['attempt-p1'],
            'changed_reasoning': 'separate combined intervention', 'unresolved': 'causality'},
            'test_prediction': {'observed_if_supported': 'fixture', 'observed_if_not_supported': 'fixture'}, 'program': p}

    def response(self, request):
        return {'status': 'ok', 'decision': self.decision(), 'test_fixture': True,
            'model': 'MiniMax-M3', 'response_model': 'MiniMax-M3',
            'request_sha256': m.old.provider_request_sha(request),
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}

    def test_history_is_actual_and_separate(self):
        c = self.context
        self.assertEqual(c['parent']['program_id'], 'p0')
        self.assertEqual(c['history'][0]['program']['program_id'], 'p1')
        self.assertEqual(c['history'][0]['expected_effect'], m.old.load(
            m.old.ROOT/'out/geopilot-learning-20260921/proposal-r2/response.json')['decision']['expected_effect'])
        self.assertGreater(c['history'][0]['delta_from_stable']['accuracy_l1_m'], 0)
        self.assertLess(c['history'][0]['delta_from_stable']['completeness_0_20'], 0)
        self.assertNotEqual(c['history'][0]['diagnostics']['dense_points'], c['diagnostics']['dense_points'])

    def test_single_change_and_history_citation(self):
        self.assertEqual(len(m.validate_decision(self.context, self.decision())), 1)
        for change in ('none', 'multiple', 'topology', 'missing_history', 'wrong_parent', 'no_prediction'):
            d = self.decision()
            if change == 'none': d['program']['nodes'][2]['parameters']['decimate'] = 0.25
            if change == 'multiple': d['program']['nodes'][2]['parameters']['smooth'] = 0
            if change == 'topology': d['program']['nodes'][2]['success'] = 'mesh'
            if change == 'missing_history': d['history_update']['evidence_ids'] = []
            if change == 'wrong_parent': d['program']['parent_id'] = 'p1'
            if change == 'no_prediction': d['test_prediction'] = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.validate_decision(self.context, d)

    def test_consistent_echoes_do_not_change_program_contract(self):
        d = self.decision()
        d.update(candidate_id='p2', parent_program_id='p0', mode='single_parameter',
                 intervention={'node_id': 'mesh', 'parameter': 'decimate', 'from': 0.25, 'to': 0.5})
        d['evidence_refs'].append('stable-score')
        d['history_update']['evidence_ids'].append('stable-score')
        m.validate_decision(self.context, d)
        d['intervention']['to'] = 1
        with self.assertRaisesRegex(ValueError, 'metadata'):
            m.validate_decision(self.context, d)
        d.pop('intervention'); d['unregistered_field'] = True
        with self.assertRaisesRegex(ValueError, 'response'):
            m.validate_decision(self.context, d)

    def test_invalid_history_prevents_call(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); plan = m.old.load(self.plan)
            broken = root / 'response.json'
            value = m.old.load(Path(plan['history'][0]['response'])); value['status'] = 'error'
            broken.write_text(json.dumps(value)); plan['history'][0]['response'] = str(broken)
            path = root / 'plan.json'; path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'provider rejected'):
                m.propose(path, root / 'out', lambda _: self.fail('provider called'))
            self.assertFalse((root / 'out').exists())
            plan = m.old.load(self.plan); plan['history'] *= 2; path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'duplicate history'):
                m.compile_context(path)

    def test_real_bridge_v2_launch_boundary_and_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'proposal'
            m.propose(self.plan, output, self.response)
            m.verify_request(output/'request.json', output/'response.json')
            code = "import {verifyCandidate} from './code/geopilot_rsih/launch.mjs'; verifyCandidate(process.argv[1],process.argv[2],'/usr/bin/python3');"
            subprocess.run(['node', '--input-type=module', '-e', code, str(output/'candidate.json'),
                            str(output/'program.json')], cwd=m.old.ROOT, check=True)
            request = m.old.load(output/'request.json')
            request['context']['history'][0]['metrics']['accuracy_l1_m'] = 0
            (output/'request.json').write_text(json.dumps(request))
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                m.verify_request(output/'request.json', output/'response.json')
        # Updating new knowledge did not replace the legacy file or invalidate the old request.
        p = m.old.ROOT/'out/geopilot-learning-20260921/proposal-r2'
        m.old.verify_request(p/'request.json', p/'response.json')


if __name__ == '__main__':
    unittest.main()
