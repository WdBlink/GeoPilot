"""The release gate cannot turn missing, changed, or failed evidence into READY."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from usegeo_mesh_benchmark import release
from usegeo_mesh_benchmark import benchmark
from usegeo_mesh_benchmark.tests import test_benchmark as fixture_tests


class ReleaseTests(unittest.TestCase):
    def test_portable_gate_and_fail_closed_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "evidence.txt"
            artifact.write_text("synthetic validator fixture, not a real acceptance report")
            ref = {"path": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
            sources = {}
            for suffix in release.SOURCE_SUFFIXES:
                source = Path(release.__file__).resolve().parent.parent / suffix
                target = root / 'source' / suffix
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                sources[str(target.relative_to(root))] = hashlib.sha256(target.read_bytes()).hexdigest()
            protocol_ref = {'path':'source/usegeo_mesh_benchmark/protocol_v1.json',
                            'sha256':sources['source/usegeo_mesh_benchmark/protocol_v1.json']}
            checks = [{"check_id": key, "req_id": value, "status": "PASS",
                       "expected": "synthetic expectation", "observed": "synthetic observation",
                       "evidence": [ref]} for key, value in list(release.CHECK_REQUIREMENTS.items())[:14]]
            campaigns = [{"scene": f"Dataset-{scene}", "round": repeat,
                          "run_id": f"fixture-{scene}-{repeat}", "kind": "formal_fixture",
                          "status": "valid", "output": ref}
                         for scene in (1, 2, 3) for repeat in (1, 2)]
            campaigns += [{"scene": f"Dataset-{scene}", "run_id": f"official-{scene}",
                           "kind": "official_compat", "status": "timeout", "output": ref}
                          for scene in (1, 2, 3)]
            data = {"schema_version": release.SCHEMA_VERSION, "release_id": "synthetic",
                    "created_at": "2026-09-18T00:00:00Z", "scope": "evaluator_release",
                    "source_files": sources, "protocol": protocol_ref,
                    "criteria": ref, "data_lock": ref,
                    "runtime": {"versions": benchmark.runtime_versions(), "lock": ref},
                    "hardware": {"cpu": "synthetic", "memory_bytes": 1, "os": "synthetic"},
                    "commands": [{"cwd": ".", "argv": ["synthetic"], "exit_code": 0, "log": ref}],
                    "checks": checks, "campaigns": campaigns,
                    "review": {"reviewer": "independent", "implementer": "author",
                               "status": "PASS", "report": ref}, "readiness": "READY"}
            path = root / "release_manifest.json"

            def check(value):
                path.write_text(json.dumps(value))
                return release.validate(path)

            self.assertEqual(check(data), "READY")
            for change in (
                lambda d: d["checks"].pop(),
                lambda d: d["checks"][0].update(status="FAIL"),
                lambda d: d["checks"].append(d["checks"][0]),
                lambda d: d["campaigns"].pop(),
                lambda d: d.update(scope="paper_results"),
                lambda d: d["review"].update(reviewer="author"),
                lambda d: d["protocol"].update(path="../escape"),
                lambda d: d["commands"][0].update(exit_code=True),
                lambda d: d.update(created_at="not-a-dateZ"),
                lambda d: d.update(created_at="2026-02-30T00:00:00Z"),
                lambda d: d["hardware"].update(memory_bytes=True),
                lambda d: d["hardware"].update(memory_bytes=-1),
                lambda d: d["hardware"].update(cpu=["cpu"]),
                lambda d: d["runtime"]["versions"].update(python=True),
                lambda d: d["review"].update(reviewer=["reviewer"]),
                lambda d: d["checks"][0].update(expected=True),
                lambda d: d["checks"].__setitem__(0, None),
                lambda d: d["commands"].__setitem__(0, []),
                lambda d: d["campaigns"].__setitem__(0, 1),
                lambda d: d["checks"][0].update(check_id=[]),
                lambda d: d["checks"][0].update(status=[]),
                lambda d: d["campaigns"][0].update(scene=[]),
                lambda d: d["commands"][0].update(argv=[""]),
                lambda d: d['source_files'].pop('source/usegeo_mesh_benchmark/official_compat.py'),
            ):
                altered = json.loads(json.dumps(data))
                change(altered)
                with self.assertRaises(ValueError):
                    check(altered)
            incomplete = json.loads(json.dumps(data))
            incomplete["checks"][0]["status"] = "BLOCKED"
            incomplete["readiness"] = "NOT_READY"
            self.assertEqual(check(incomplete), "NOT_READY")
            path.write_text('{"schema_version":"x","schema_version":"y"}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                release.validate(path)
            link = root / 'linked.txt'
            link.symlink_to(artifact)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                release.evidence(root, {'path': link.name, 'sha256': ref['sha256']})

            # Evidence-shaped synthetic values test identity only, not scientific truth.
            def save(name, value):
                target = root / name
                target.write_text(json.dumps(value))
                return {'path': name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}

            evaluator_ref = save('evaluator.json', data)
            paper = json.loads(json.dumps(data))
            paper.update(scope='paper_results', evaluator_release=evaluator_ref)
            paper['checks'] = [{'check_id': k, 'req_id': v, 'status':'PASS',
                'expected':'synthetic', 'observed':'synthetic', 'evidence':[ref]}
                for k,v in release.CHECK_REQUIREMENTS.items()]
            experiments = []
            for scene in (1,2,3):
                exp = {'schema_version':'usegeo-mesh-experiment-1.0', 'experiment_id':f'synthetic-{scene}',
                    'method_id':'synthetic', 'version':'1', 'configuration_id':'fixed', 'seed':1,
                    'track':'rgb-oriented', 'scene_id':f'Dataset-{scene}', 'run_id':f'method-{scene}',
                    'status':'valid', 'release_id':data['release_id'], 'release_lock_sha256':evaluator_ref['sha256'],
                    'data_lock_sha256':ref['sha256'], 'preregistration':ref, 'allowed_inputs':ref,
                    'run_manifest':ref, 'mesh':ref, 'score':ref,
                    'method_cost':{'seconds':1}, 'evaluator_resources':{'seconds':1}}
                result_dir = fixture_tests.IsolationAndAggregationTests()._result(root / f'result-{scene}',
                    exp['scene_id'], config=exp['configuration_id'])
                score = json.loads((result_dir / 'score.json').read_text())
                recorded = json.loads((result_dir / 'run_manifest.json').read_text())
                score['bindings'].update(mesh_sha256=ref['sha256'], input_manifest_sha256=ref['sha256'],
                    protocol_sha256=protocol_ref['sha256'], bundle_lock_sha256=ref['sha256'])
                exp['score'] = save(f'result-{scene}/score.json', score)
                recorded.update(bindings=score['bindings'], output_sha256={'score.json':exp['score']['sha256']},
                    method_metadata={key:exp[key] for key in ('method_id','version','seed')})
                exp['run_manifest'] = save(f'result-{scene}/run_manifest.json', recorded)
                exp['evaluator_resources'] = {key:recorded[key] for key in ('elapsed_seconds','peak_worker_rss_bytes')}
                experiments.append(exp)
                paper['campaigns'].append({'scene':f'Dataset-{scene}', 'run_id':f'method-{scene}',
                    'kind':'method', 'status':'valid', 'output':exp['score'],
                    'experiment':save(f'experiment-{scene}.json', exp)})
            self.assertEqual(check(paper), 'READY')
            for field, invalid in [('release_id','wrong'), ('release_lock_sha256','0'*64),
                ('data_lock_sha256','0'*64), ('configuration_id','different'),
                ('seed',True), ('scene_id','Dataset-2'), ('mesh',None), ('score',None)]:
                broken = json.loads(json.dumps(experiments[0]))
                broken[field] = invalid
                altered = json.loads(json.dumps(paper))
                altered['campaigns'][-3]['experiment'] = save('broken.json', broken)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    check(altered)
            altered = json.loads(json.dumps(paper))
            altered['campaigns'][-1].pop('experiment')
            with self.assertRaises(ValueError):
                check(altered)
            # Rehashed but internally inconsistent evidence must not become READY.
            wrong_score = json.loads((root / 'result-1/score.json').read_text())
            wrong_score['bindings']['mesh_sha256'] = '0'*64
            (root / 'wrong').mkdir()
            wrong_ref = save('wrong/score.json', wrong_score)
            broken = json.loads(json.dumps(experiments[0]))
            broken['score'] = wrong_ref
            wrong_run = json.loads((root / 'result-1/run_manifest.json').read_text())
            wrong_run.update(bindings=wrong_score['bindings'], output_sha256={'score.json':wrong_ref['sha256']})
            broken['run_manifest'] = save('wrong/run_manifest.json', wrong_run)
            altered = json.loads(json.dumps(paper))
            altered['campaigns'][-3].update(output=wrong_ref, experiment=save('wrong-experiment.json',broken))
            with self.assertRaisesRegex(ValueError, 'mesh_sha256'):
                check(altered)
            artifact.write_text("changed")
            with self.assertRaises(ValueError):
                check(data)


if __name__ == "__main__":
    unittest.main()
