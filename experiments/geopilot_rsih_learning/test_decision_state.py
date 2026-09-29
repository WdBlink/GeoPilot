"""Focused pure transition tests; no numerical chain, filesystem or API calls."""
import copy
import unittest

from decision_state import transition


def observation(step=0, sha='sha-a', status='ok'):
    return {'step_index': step, 'name': 'inspect', 'feedback': {
        'kind': 'diagnostic', 'status': status, 'sources': {'SOURCE': sha},
        'facts': {'source_sha256': {'SOURCE': sha}} if status == 'ok' else None}}


def claim(**changes):
    row = {'claim_id': 'c', 'observation_refs': ['step-0'], 'alternatives': 'alternative',
           'proposed_action': 'inspect output', 'predicted_observation': 'fewer faces',
           'observed_result_refs': [], 'status': 'not_tested'}
    return {**row, **changes}


class StateTests(unittest.TestCase):
    def test_new_observation_requires_explicit_backfill_without_mutation(self):
        calls = [observation()]
        state = transition(tool_calls=calls, rows=[claim()])
        saved = copy.deepcopy(state)
        calls.append(observation(1))
        pending = transition(state, tool_calls=calls)
        self.assertTrue(pending['claims']['c']['needs_update'])
        self.assertEqual(state, saved)
        updated = transition(pending, tool_calls=calls, rows=[claim(
            observed_result_refs=['step-1'], status='supported')])
        self.assertFalse(updated['claims']['c']['needs_update'])
        self.assertEqual(updated['claims']['c']['status'], 'supported')

    def test_future_invented_and_failed_source_refs_rejected(self):
        for ref in ['step-1', 'made-up', 'step-0000']:
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                transition(tool_calls=[observation()], rows=[claim(observation_refs=[ref])])
        failed = observation(status='failed')
        transition(tool_calls=[failed], rows=[claim()])  # failure is an observation
        with self.assertRaises(ValueError):
            transition(tool_calls=[failed], rows=[claim(observation_refs=['SOURCE'])])
        with self.assertRaises(ValueError):
            transition(tool_calls=[observation()], rows=[claim(status='supported')])

    def test_predictions_cannot_be_backfilled_as_observations_or_rewritten(self):
        calls = [observation()]
        with self.assertRaises(ValueError):
            transition(tool_calls=calls, rows=[claim(observed_result_refs='fewer faces')])
        state = transition(tool_calls=calls, rows=[claim()])
        with self.assertRaisesRegex(ValueError, 'prediction_changed'):
            transition(state, tool_calls=calls, rows=[claim(predicted_observation='more faces',
                       observed_result_refs=['step-0'], status='supported')])

    def test_source_and_code_drift_invalidate_and_stale_evidence_rejected(self):
        calls = [observation()]
        state = transition(tool_calls=calls, code_version='v1', rows=[claim(
            observed_result_refs=['SOURCE'], status='supported')])
        stale = transition(state, tool_calls=calls, code_version='v1', source_versions={'SOURCE': 'sha-b'})
        self.assertEqual(stale['stale_observation_ids'], ['step-0'])
        self.assertEqual(stale['claims']['c']['status'], 'unresolved')
        with self.assertRaises(ValueError):
            transition(stale, tool_calls=calls, code_version='v1', source_versions={'SOURCE': 'sha-b'},
                       rows=[claim(observed_result_refs=['SOURCE'], status='supported')])
        newcode = transition(state, tool_calls=calls, code_version='v2')
        self.assertTrue(newcode['claims']['c']['needs_update'])
        self.assertEqual(newcode['claims']['c']['status'], 'unresolved')

    def test_frozen_memory_preserves_exceptions_and_requires_real_evidence(self):
        frozen = [{'id': 'E', 'conditions': 'matching inputs',
                   'counterevidence_and_boundaries': ['counterexample'], 'unproven': ['transfer']}]
        original = copy.deepcopy(frozen)
        calls = [observation()]
        state = transition(tool_calls=calls, frozen_experience=frozen)
        self.assertEqual(state['experience']['E']['status'], 'unresolved')
        with self.assertRaises(ValueError):
            transition(state, tool_calls=calls, frozen_experience=frozen,
                       experience_rows=[{'entry_id': 'E', 'applicability_refs': [], 'status': 'eligible'}])
        row = {'entry_id': 'E', 'applicability_refs': ['SOURCE'], 'status': 'inapplicable'}
        updated = transition(state, tool_calls=calls, frozen_experience=frozen, experience_rows=[row])
        self.assertEqual(updated['experience']['E']['counterevidence_and_boundaries'], ['counterexample'])
        self.assertEqual(frozen, original)
        frozen[0]['conditions'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'frozen_experience_changed'):
            transition(updated, tool_calls=calls, frozen_experience=frozen)

    def test_duplicate_history_is_deduplicated_not_rewritten(self):
        calls = [observation()]
        state = transition(tool_calls=calls, history=calls)
        self.assertEqual(len(state['observations']), 1)
        with self.assertRaisesRegex(ValueError, 'history_rewritten'):
            transition(state, tool_calls=[])
        with self.assertRaisesRegex(ValueError, 'conflicting_actual'):
            transition(tool_calls=calls, history=[observation(sha='other')])


if __name__ == '__main__':
    unittest.main()
