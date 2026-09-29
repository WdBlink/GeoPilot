"""Offline checks for the registered numerical configuration set.

No reconstruction, sparse/dense/mesh pass, evaluator call or external model call happens
here. The tests that need the frozen chain are skipped unless `outer_numerical` and
`outer_score` import, which requires the baseline runtime recorded in
`publisher_raster_full_sparse.py` (pycolmap); run this file under that interpreter to get
the full set instead of skips.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import outer_config

IMPORT_ERROR = None
try:
    import outer_numerical as numerical
    import outer_score as scoring
except ImportError as exc:  # pragma: no cover - depends on the interpreter, not the logic
    numerical = scoring = None
    IMPORT_ERROR = repr(exc)

needs_runtime = unittest.skipIf(numerical is None, 'frozen baseline runtime required: ' + str(IMPORT_ERROR))

#: The recorded preflight receipt that predates the configuration interface. It is the
#: oracle for "config=None changes nothing": same keys, same values, plus `config`.
RECORDED_PREFLIGHT = Path('out/geopilot-research-20260926/jev-validation/numerical-preflight-d2-f-u/result.json')
RECORDED_COMPLETED = Path('out/geopilot-research-20260926/jev-validation/integration-d2-f-u/result.json')
#: Contract keys whose values are environment-specific even for a synthetic campaign.
ENVIRONMENT_KEYS = frozenset({'bindings', 'inputs_sha256', 'output_hashes', 'elapsed_seconds'})


def reject(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, RuntimeError, KeyError):
        return
    raise AssertionError('Unsafe input accepted: ' + getattr(function, '__name__', str(function)))


class RegisteredSetTest(unittest.TestCase):
    """The closed set itself: no free-form merging, no dead knob, no single point."""

    def test_registry_is_closed_proven_only_and_non_degenerate(self):
        ids = [entry['id'] for entry in outer_config.REGISTERED_CONFIGS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(outer_config.BASELINE_ID, ids)
        self.assertGreaterEqual(len(ids), 2)
        for entry in outer_config.REGISTERED_CONFIGS:
            knobs = outer_config.changed_keys(entry['parameters'])
            self.assertEqual(knobs, entry['knobs'], entry['id'])
            self.assertEqual(outer_config.audit_wiring(entry['id']), entry['id'])
            for knob in knobs:
                self.assertIn(knob, outer_config.REGISTRABLE)
                self.assertEqual(outer_config.KEY_VERDICTS[knob]['verdict'], 'PROVEN')
        actions = [entry['id'] for entry in outer_config.REGISTERED_CONFIGS
                   if entry['id'] != outer_config.BASELINE_ID]
        self.assertTrue(actions, 'a single-point registry is not an action space')

    def test_dead_and_unverified_keys_are_documented_and_not_registered(self):
        verdicts = {key: value['verdict'] for key, value in outer_config.KEY_VERDICTS.items()}
        self.assertEqual(verdicts['undistort_max_image_size'], 'DEAD')
        self.assertEqual(verdicts['undistort_call'], 'DEAD')
        self.assertEqual(verdicts['transform_application'], 'DEAD')
        self.assertEqual(verdicts['export_rtol'], 'DEAD')
        self.assertEqual(verdicts['export_atol_m'], 'CONSUMED_NOT_GEOMETRIC')
        self.assertEqual(verdicts['mesh.decimate'], 'PROVEN')
        self.assertEqual(verdicts['densify.resolution-level'], 'PROVEN')
        for value in outer_config.KEY_VERDICTS.values():
            self.assertTrue(value['consumed_at'], 'every key needs an exact consumption site')
        registered = {knob for entry in outer_config.REGISTERED_CONFIGS for knob in entry['knobs']}
        self.assertEqual(registered & {'undistort_max_image_size', 'undistort_call',
                                       'transform_application', 'export_atol_m', 'export_rtol'}, set())

    def test_resolve_none_is_the_frozen_baseline_and_results_are_copies(self):
        self.assertEqual(outer_config.resolve(None), outer_config.BASELINE)
        self.assertEqual(outer_config.resolve(outer_config.BASELINE_ID), outer_config.BASELINE)
        resolved = outer_config.resolve('mesh-decimate-025-v1')
        resolved['mesh']['decimate'] = 0.9
        self.assertEqual(outer_config.resolve('mesh-decimate-025-v1')['mesh']['decimate'], 0.25)
        self.assertEqual(outer_config.BASELINE['mesh']['decimate'], 0.5)

    def test_unregistered_configurations_raise(self):
        for bad in ({}, {'mesh': {'decimate': 0.9}}, {'mesh': {'decimate': True}},
                    {'undistort_max_image_size': 4800}, {'densify': {'resolution-level': 1, 'x': 2}},
                    {'export_atol_m': 1e-6}, 'baseline-v2', '', {'id': 'baseline-v2'}, 5, ['mesh-decimate-025-v1']):
            with self.subTest(bad=bad):
                reject(outer_config.require_registered, bad)
                reject(outer_config.resolve, bad)

    def test_ambiguous_override_is_refused_not_guessed(self):
        # {'mesh': {}} matches nothing (empty), but a future widening could make one
        # override match two entries; the registry must name the ambiguity instead.
        registry = list(outer_config.REGISTERED_CONFIGS)
        widened = copy.deepcopy(registry[-1])
        widened['id'] = 'widened-v1'
        widened['parameters']['mesh'] = {'decimate': 0.5}
        widened['knobs'] = ('mesh.decimate',)
        with patch.object(outer_config, 'REGISTERED_CONFIGS', tuple(registry + [widened])):
            with self.assertRaises(ValueError) as caught:
                outer_config.require_registered({'mesh': {'decimate': 0.5}})
            self.assertIn('ambiguous', str(caught.exception))
            self.assertEqual(outer_config.require_registered({'mesh': {'decimate': 0.25}}), 'mesh-decimate-025-v1')

    def test_a_knob_found_dead_fails_loudly_instead_of_running(self):
        dead = copy.deepcopy(outer_config.KEY_VERDICTS)
        dead['mesh.decimate'] = {**dead['mesh.decimate'], 'verdict': 'DEAD', 'live': False}
        with patch.object(outer_config, 'KEY_VERDICTS', dead):
            with self.assertRaises(ValueError) as caught:
                outer_config.audit_wiring('mesh-decimate-025-v1')
            self.assertIn('not PROVEN', str(caught.exception))
            with self.assertRaises(ValueError):
                outer_config._self_check()
        unwired = copy.deepcopy(outer_config.KEY_VERDICTS)
        unwired['mesh.decimate'] = {**unwired['mesh.decimate'], 'action': None}
        with patch.object(outer_config, 'KEY_VERDICTS', unwired):
            with self.assertRaises(ValueError) as caught:
                outer_config.audit_wiring('mesh-decimate-025-v1')
            self.assertIn('argv', str(caught.exception))
        # A knob pinned to the binary's own default is registered wiring that does nothing.
        inert = copy.deepcopy(outer_config.REGISTERED_CONFIGS[2])
        inert['id'] = 'densify-resolution-1-v1'
        inert['parameters']['densify']['resolution-level'] = 1
        with patch.object(outer_config, 'REGISTERED_CONFIGS', tuple(list(outer_config.REGISTERED_CONFIGS) + [inert])):
            with self.assertRaises(ValueError) as caught:
                outer_config.audit_wiring(inert['id'])
            self.assertIn('declared knobs', str(caught.exception))
        unlive = copy.deepcopy(outer_config.REGISTERED_CONFIGS[2])
        unlive['id'] = 'densify-resolution-5-v1'
        unlive['parameters']['densify']['resolution-level'] = 5
        with patch.object(outer_config, 'REGISTERED_CONFIGS', tuple(list(outer_config.REGISTERED_CONFIGS) + [unlive])):
            with self.assertRaises(ValueError) as caught:
                outer_config.audit_wiring(unlive['id'])
            self.assertIn('inert binary default', str(caught.exception))

    def test_argv_rule_matches_the_frozen_worker_assembler(self):
        self.assertEqual(outer_config.argv_flags({'decimate': 0.5, 'smooth': 0}), ['--decimate', '0.5', '--smooth', '0'])
        self.assertIn('--decimate', outer_config.argv_flags(outer_config.resolve('mesh-decimate-025-v1')['mesh']))
        self.assertIn('--resolution-level',
                      outer_config.argv_flags(outer_config.resolve('densify-resolution-2-v1')['densify']))


class RecordedReceiptTest(unittest.TestCase):
    """`config=None` must reproduce the pre-existing receipt shape, key for key."""

    @unittest.skipUnless(RECORDED_PREFLIGHT.is_file(), 'recorded preflight receipt is absent')
    @needs_runtime
    def test_default_config_reproduces_the_recorded_receipt(self):
        recorded = json.loads((numerical.ROOT / RECORDED_PREFLIGHT).read_text())
        with tempfile.TemporaryDirectory(dir=numerical.ROOT / 'out') as folder:
            produced = self._preflight(Path(folder) / 'config-default', outer_config.resolve(None))
        self.assertEqual(set(produced) - set(recorded), {'config'})
        self.assertEqual(set(recorded) - set(produced), set())
        for key in sorted(set(recorded) - ENVIRONMENT_KEYS):
            self.assertEqual(produced[key], recorded[key], key)
        self.assertEqual(produced['config'], numerical.PARAMETERS)
        self.assertEqual(produced['parameters'], recorded['parameters'])

    def _preflight(self, output, parameters):
        scene = numerical.scenes.scene_record('Dataset-2')
        campaign = {'scene': scene, 'parameters': parameters, 'config': parameters,
                    'resource_guards': numerical.DEFAULT_GUARDS, 'bindings': {}}
        stage = {'stage': 'probe', 'denied_inputs': [], 'worker_bindings': {}}
        with patch.object(numerical, 'ensure_idle'), \
             patch.object(numerical, 'prepare', return_value=campaign), \
             patch.object(numerical.scenes, 'stage_manifest', return_value=dict(stage)), \
             patch.object(numerical, 'launch_worker', return_value={}):
            numerical.execute('Dataset-2', 'F-U', Path('unused'), output, numerical.DEFAULT_GUARDS)
        return json.loads((output / 'result.json').read_text())


class ThreadingTest(unittest.TestCase):
    """The effective parameters must reach the identity, the worker action and the argv."""

    @needs_runtime
    def test_baseline_parameters_equal_the_registered_baseline(self):
        self.assertEqual(numerical.PARAMETERS, outer_config.BASELINE)
        self.assertEqual(numerical.PARAMETERS, scoring.PARAMETERS)
        self.assertEqual(numerical.worker_parameters({'parameters': numerical.PARAMETERS}),
                         numerical.PARAMETERS)

    @needs_runtime
    def test_worker_manifest_gate_accepts_registered_and_refuses_the_rest(self):
        registered = outer_config.resolve('mesh-decimate-025-v1')
        self.assertEqual(numerical.worker_parameters({'parameters': registered, 'config': registered}), registered)
        reject(numerical.worker_parameters, {'parameters': numerical.PARAMETERS, 'config': registered})
        reject(numerical.worker_parameters, {'parameters': registered})
        reject(numerical.worker_parameters,
               {'parameters': registered, 'config': {'mesh': {'decimate': 0.9}}})
        reject(numerical.worker_parameters,
               {'parameters': numerical.PARAMETERS, 'config': {'undistort_max_image_size': 4800}})
        # A legacy manifest carries no configuration and must keep resolving to the default.
        self.assertEqual(numerical.worker_parameters({'parameters': numerical.PARAMETERS}), numerical.PARAMETERS)

    @needs_runtime
    def test_config_changes_the_worker_action_and_the_assembled_argv(self):
        for action, key, config_id, default, changed in (
                ('densify', 'resolution-level', 'densify-resolution-2-v1', '1', '2'),
                ('mesh', 'decimate', 'mesh-decimate-025-v1', '0.5', '0.25')):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                baseline = self._argv(root, action, numerical.PARAMETERS)
                configured = self._argv(root, action, outer_config.resolve(config_id))
            self.assertIn('--' + key, baseline, action)
            self.assertEqual(baseline[baseline.index('--' + key) + 1], default, action)
            self.assertIn('--' + key, configured, action)
            self.assertEqual(configured[configured.index('--' + key) + 1], changed, action)
            swap = lambda argv: [changed if item == default else item for item in argv]
            self.assertEqual(swap(baseline), configured,
                             'the registered knob must be the only argv difference for ' + action)

    def _argv(self, root, action, parameters):
        """The argv the frozen worker runtime actually assembles, without running it."""
        out = root / action
        out.mkdir(exist_ok=True)
        parent_artifact = root / ('scene.mvs' if action == 'densify' else 'dense.ply')
        parent_artifact.write_bytes(b'synthetic parent artifact')
        state = {'stage': 'parent', 'diagnostics': {},
                 'scene' if action == 'densify' else 'dense': str(parent_artifact),
                 'artifact_hashes': {str(parent_artifact): numerical.ba.sha(parent_artifact)},
                 'mvs_workspace': str(out),
                 'transform': [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]}
        request = numerical.action_request(root / 'inputs', out, action, parameters, state)
        self.assertEqual(request['node']['parameters'], parameters[action])
        calls = []
        with patch.object(numerical.tools, 'invoke', lambda argv, output, workdir=None: calls.append(argv)):
            with self.assertRaises((OSError, ValueError)):
                numerical.tools.real_action(request)
        self.assertEqual(len(calls), 1, 'the frozen worker must assemble exactly one argv')
        return calls[0]


class ScoreBindingTest(unittest.TestCase):
    """Legacy receipts score; retired adapter bytes stay verifiable; config never self-certifies."""

    @needs_runtime
    def test_legacy_receipt_without_config_still_scores(self):
        from test_outer_numerical import source_fixture
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            record, manifest = source_fixture(root)
            path = root / 'result.json'
            self.assertNotIn('config', record)
            scoring.validate_source(path, numerical.ba.sha(path), 'Dataset-2', manifest)
            scoring.validate_source(path, numerical.ba.sha(path), 'Dataset-2', manifest, numerical.PARAMETERS)
            scoring.validate_source(path, numerical.ba.sha(path), 'Dataset-2', manifest,
                                    outer_config.resolve(outer_config.BASELINE_ID))
            self.assertEqual(outer_config.resolve(None), numerical.PARAMETERS)
            # A receipt that carries a configuration must carry the host-held one.
            configured = copy.deepcopy(record)
            configured['config'] = outer_config.resolve('mesh-decimate-025-v1')
            self.write(path, configured)
            reject(scoring.validate_source, path, numerical.ba.sha(path), 'Dataset-2', manifest)
            self.write(path, record)
            self.assertEqual(outer_config.changed_keys(outer_config.resolve('mesh-decimate-025-v1')), ('mesh.decimate',))

    @needs_runtime
    def test_retired_adapter_digest_is_a_closed_allowlist(self):
        from test_outer_numerical import source_fixture
        retired = scoring.LEGACY_ADAPTER_SHA['outer_numerical.py']
        entry = str(numerical.ENTRY)
        self.assertNotEqual(retired, numerical.ba.sha(numerical.ENTRY))
        recorded = numerical.ROOT / RECORDED_COMPLETED
        if recorded.is_file():
            self.assertEqual(json.loads(recorded.read_text())['bindings'][entry], retired)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            record, manifest = source_fixture(root)
            record['bindings'][entry] = retired
            campaign = json.loads((root / 'inputs.json').read_text())
            campaign['bindings'][entry] = retired
            self.write(root / 'inputs.json', campaign)
            record['output_hashes'][str(root / 'inputs.json')] = numerical.ba.sha(root / 'inputs.json')
            record['inputs_sha256'] = record['output_hashes'][str(root / 'inputs.json')]
            path = root / 'result.json'
            self.write(path, record)
            scoring.validate_source(path, numerical.ba.sha(path), 'Dataset-2', manifest)
            # The legacy path structurally cannot carry a new arm.
            reject(scoring.validate_source, path, numerical.ba.sha(path), 'Dataset-2', manifest,
                   outer_config.resolve('mesh-decimate-025-v1'))
            self.write(path, record)
            other = copy.deepcopy(record)
            other['config'] = outer_config.resolve('mesh-decimate-025-v1')
            self.write(path, other)
            reject(scoring.validate_source, path, numerical.ba.sha(path), 'Dataset-2', manifest)
            self.write(path, record)
            unknown = copy.deepcopy(record)
            unknown['bindings'][entry] = '0' * 64
            self.write(path, unknown)
            reject(scoring.validate_source, path, numerical.ba.sha(path), 'Dataset-2', manifest)

    @needs_runtime
    def test_dropping_an_adapter_binding_is_not_a_retired_digest(self):
        """A missing binding must not pass as an accepted retired digest.

        `LEGACY_ADAPTER_SHA` enumerates one name, but the retirement loop walks four. For
        the three names it does not enumerate, both the allowlist lookup and the binding
        lookup return None, and `None != None` is false, so a receipt that simply drops a
        helper from its binding set would have been accepted. Presence is now checked
        before the digest comparison.
        """
        from test_outer_numerical import source_fixture
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            record, manifest = source_fixture(root)
            path = root / 'result.json'
            for name in ('outer_numerical.py', 'publisher_raster_scenes23.py',
                         'publisher_raster_mesh.py', 'publisher_raster_mesh_r2.py'):
                dropped = copy.deepcopy(record)
                dropped['bindings'].pop(str(numerical.ENTRY.with_name(name)), None)
                self.write(path, dropped)
                with self.subTest(dropped=name):
                    reject(scoring.validate_source, path, numerical.ba.sha(path),
                           'Dataset-2', manifest)
            self.write(path, record)

    @needs_runtime
    def test_scorer_rejects_an_unregistered_host_configuration(self):
        with tempfile.TemporaryDirectory(dir=numerical.ROOT / 'out') as folder:
            target = Path(folder) / 'never-created'
            for bad in ({'mesh': {'decimate': 0.9}}, {'undistort_max_image_size': 4800}, 'baseline-v2'):
                with self.subTest(bad=bad):
                    reject(scoring.execute, 'Dataset-2', target, scoring.DEFAULT_GUARDS, config=bad)
            self.assertFalse(target.exists(), 'an unregistered configuration must not create an output')

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, allow_nan=False))


if __name__ == '__main__':
    unittest.main()
