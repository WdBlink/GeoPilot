"""Synthetic consumer/bridge checks, never model or reconstruction evidence."""
import copy
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import learning as m


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='geopilot-learning-test-')
        self.home = Path(self.tmp.name).resolve()
        self.run = self.home / 'method'
        self.run.mkdir()
        old = m.ROOT / 'out/geopilot-rsih-p0-mu9geojt-r3'
        for name in ('program.json', 'genome.json'):
            shutil.copyfile(old / name, self.run / name)
        launch = m.load(old / 'launch.json')
        launch['effective'] = {str(self.run / n): m.sha(self.run / n) for n in ('program.json', 'genome.json')}
        launch['bindings'] = {'synthetic-fixture': 'not-production'}
        m.write(self.run / 'launch.json', launch)
        (self.run / 'submission').mkdir()
        (self.run / 'submission/mesh.ply').write_bytes(b'explicit non-geometric fixture')
        m.write(self.run / 'submission/submission.json', {'fixture': True})
        run = m.load(old / 'run.json')
        run.update(submission_mesh_sha256=m.sha(self.run / 'submission/mesh.ply'),
                   submission_contract_sha256=m.sha(self.run / 'submission/submission.json'))
        m.write(self.run / 'run.json', run)
        result = m.load(old / 'result.json')
        result['state'] = {'diagnostics': {'mesh_faces': {'value': 3000, 'status': 'valid'}}}
        m.write(self.run / 'result.json', result)
        validation = m.load(old / 'validation.json')
        accepted = json.loads(validation['stdout'])
        accepted['bindings'].update(mesh_sha256=run['submission_mesh_sha256'],
                                    submission_contract_sha256=run['submission_contract_sha256'])
        validation['stdout'] = json.dumps(accepted)
        m.write(self.run / 'validation.json', validation)
        bindings = copy.deepcopy(accepted['bindings'])
        protocol = m.HERE / 'evaluator/protocol_v1.json'
        bindings.update(protocol_sha256=m.sha(protocol), runtime_implementation_id=m.load(protocol)['runtime']['implementation_id'])
        self.score = self.home / 'score.json'
        self.manifest = self.home / 'run_manifest.json'
        self.value = {'status': 'valid', 'bindings': bindings, 'configuration_id': accepted['configuration_id'],
                      'scene_id': 'Dataset-1', 'track': 'rgb-oriented',
                      'surface_metrics': {k: 0.1 for k in m.METRICS}, 'structure_diagnostics': {'fixture': True}}
        self.save_score()
        self.args = SimpleNamespace(run=self.run, score=self.score, manifest=self.manifest,
                                    output=self.home / 'proposal', node='node')

    def tearDown(self):
        self.tmp.cleanup()

    def save_score(self):
        self.score.write_text(json.dumps(self.value))
        self.manifest.write_text(json.dumps({'status': 'valid', 'bindings': self.value['bindings'],
                                            'output_sha256': {'score.json': m.sha(self.score)}}))

    def decision(self):
        p = m.load(self.run / 'program.json')
        p.update(program_id='p1', parent_id='p0')
        p['nodes'][2]['success'] = 'check'
        p['nodes'].extend([
            {'id': 'check', 'kind': 'check', 'metric': 'mesh_faces', 'op': 'gt', 'value': 1000,
             'success': 'refine', 'failure': 'end', 'unknown': 'end'},
            {'id': 'refine', 'kind': 'tool', 'action': 'refine', 'parameters': {'resolution-level': 1},
             'success': 'end', 'failure': 'end'}])
        return {'hypothesis': 'synthetic check only', 'expected_effect': 'no research claim',
                'risks': ['fixture'], 'competing_explanations': ['fixture'],
                'evidence_refs': ['r3-run'], 'program': p}

    def response(self, request, decision=None):
        return {'status': 'ok', 'decision': decision or self.decision(), 'test_fixture': True,
                'model': 'MiniMax-M3', 'response_model': 'MiniMax-M3',
                'request_sha256': m.provider_request_sha(request),
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}

    def test_actual_bridge_and_candidate_binding(self):
        calls = []
        def fake(request):
            calls.append(request)
            return self.response(request)
        m.propose(self.args, fake)
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0]['context']['registry'], dict)
        self.assertEqual(m.load(self.args.output / 'p1.json'), self.decision()['program'])
        candidate = m.load(self.args.output / 'candidate.json')
        self.assertEqual(candidate['response_sha256'], m.sha(self.args.output / 'response.json'))
        import subprocess
        script = "import {verifyCandidate} from './code/geopilot_rsih/launch.mjs'; verifyCandidate(process.argv[1],process.argv[2],'/usr/bin/python3');"
        subprocess.run(['node', '--input-type=module', '-e', script,
                        str(self.args.output / 'candidate.json'), str(self.args.output / 'p1.json')],
                       cwd=m.ROOT, check=True)
        request_path = self.args.output / 'request.json'
        request = m.load(request_path)
        del request['context']['evidence_bindings'][str(self.score)]
        request_path.write_text(json.dumps(request))
        with self.assertRaisesRegex(ValueError, 'source evidence'):
            m.verify_request(request_path, self.args.output / 'response.json')

    def test_invalid_score_never_calls_provider(self):
        self.value['status'] = 'error'
        self.save_score()
        with self.assertRaisesRegex(ValueError, 'invalid score'):
            m.propose(self.args, lambda _: self.fail('provider called'))
        self.assertFalse(self.args.output.exists())

    def test_identity_and_metric_rejections(self):
        (self.run / 'submission/mesh.ply').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'actual submission'):
            m.checked(self.run, self.score, self.manifest)
        (self.run / 'submission/mesh.ply').write_bytes(b'explicit non-geometric fixture')
        for value in (True, float('nan'), -1):
            self.value['surface_metrics']['accuracy_l1_m'] = value
            self.save_score()
            with self.assertRaises(ValueError):
                m.checked(self.run, self.score, self.manifest)

    def test_illegal_model_program_is_preserved_and_rejected(self):
        def fake(request):
            decision = self.decision()
            decision['program']['nodes'][-1]['parameters']['unsupported'] = 1
            return self.response(request, decision)
        with self.assertRaises(Exception):
            m.propose(self.args, fake)
        self.assertTrue((self.args.output / 'response.json').exists())
        self.assertTrue((self.args.output / 'rejected.json').exists())
        self.assertFalse((self.args.output / 'candidate.json').exists())

    def test_wrong_provider_request_is_not_a_candidate(self):
        def fake(request):
            response = self.response(request)
            response['request_sha256'] = '0' * 64
            return response
        with self.assertRaisesRegex(ValueError, 'provider request/model'):
            m.propose(self.args, fake)
        self.assertTrue((self.args.output / 'response.json').exists())
        self.assertFalse((self.args.output / 'candidate.json').exists())

    def test_provider_wire_payload_and_legacy_default(self):
        from experiments.geopilot_v1 import provider
        for limit in (4096, 16384):
            request = {'system': m.SYSTEM, 'context': {'fixture': True}, 'model': {'max_tokens': limit}}
            def fake_urlopen(wire, timeout):
                self.assertEqual(json.loads(wire.data)['max_tokens'], limit)
                self.assertEqual(m.hashlib.sha256(wire.data).hexdigest(), m.provider_request_sha(request))
                return io.BytesIO(json.dumps({'model': 'MiniMax-M3', 'usage': {}, 'choices': [
                    {'finish_reason': 'stop', 'message': {'content': '{}'}}]}).encode())
            with patch.object(provider, 'config', return_value={'MINIMAX_API_KEY': 'fixture',
                    'MINIMAX_BASE_URL': 'https://fixture.invalid', 'MINIMAX_MODEL': 'MiniMax-M3'}), \
                    patch.object(provider.urllib.request, 'urlopen', side_effect=fake_urlopen):
                kwargs = {} if limit == 4096 else {'max_tokens': limit}
                response = provider.ask(request['system'], request['context'], [], **kwargs)
                self.assertEqual(response['status'], 'ok')
                self.assertEqual(response['finish_reason'], 'stop')


if __name__ == '__main__':
    unittest.main()
