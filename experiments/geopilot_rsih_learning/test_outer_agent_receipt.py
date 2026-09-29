"""Determinism checks for the four-arm receipt writer.

The receipt is the paper's evidence artifact: it is hashed, archived and
re-verified, so every field has to be a pure function of the run. The checks
here are limited to that property. No model API, no numerical chain, no
scorer, no network. Run with the project research Python and ``-B``.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import outer_agent_receipt as receipts
from outer_agent_receipt import ERRORS, normalise_error

HERE = Path(__file__).resolve().parent

# Messages chosen to carry more than one vocabulary term, since those are the
# only cases where scan order decides the answer.
MESSAGES = (
    'unregistered_config rejected: invalid_schema',
    'invalid_schema',
    'model_identity_mismatch: model_identity_missing',
    'step_budget exhausted, call_budget exhausted',
    'numerical_failed while scoring_failed',
    'provider said something with no vocabulary term at all',
)


def _label(message: str, seed: str) -> str:
    """Return normalise_error(message) from a fresh interpreter at a fixed seed."""
    completed = subprocess.run(
        [sys.executable, '-B', '-c',
         'import outer_agent_receipt as r;'
         'print(r.normalise_error(Exception(%r)))' % message],
        cwd=HERE, capture_output=True, text=True, env={'PYTHONHASHSEED': seed, 'PATH': ''},
    )
    if completed.returncode != 0:
        raise AssertionError('seed %s failed: %s' % (seed, completed.stderr.strip()))
    return completed.stdout.strip()


class ErrorVocabularyTests(unittest.TestCase):
    def test_every_returned_label_is_in_the_vocabulary(self):
        for message in MESSAGES:
            self.assertIn(normalise_error(Exception(message)), ERRORS)

    def test_unmatched_message_falls_back_to_invalid_schema(self):
        self.assertEqual(normalise_error(Exception('nothing familiar here')),
                         'invalid_schema')

    def test_specific_term_beats_generic_term(self):
        """A wrapped message records the cause, not the generic wrapper."""
        self.assertEqual(normalise_error(Exception('unregistered_config rejected: invalid_schema')),
                         'unregistered_config')
        self.assertEqual(normalise_error(Exception(MESSAGES[2])),
                         'model_identity_mismatch')

    def test_same_message_gives_same_label_under_every_hash_seed(self):
        """Regression: ERRORS is a frozenset, so scanning it directly let
        PYTHONHASHSEED decide which term a receipt recorded."""
        for message in MESSAGES:
            labels = {_label(message, seed) for seed in ('1', '2', '3', '4', '5')}
            self.assertEqual(len(labels), 1,
                             'hash seed changed the label for %r: %s' % (message, labels))

    def test_scan_order_is_fixed_not_set_order(self):
        self.assertEqual(list(receipts.ERRORS_BY_SPECIFICITY),
                         sorted(receipts.ERRORS, key=lambda t: (-len(t), t)))
        self.assertNotIsInstance(receipts.ERRORS_BY_SPECIFICITY, (set, frozenset))


class IdentityDigestTests(unittest.TestCase):
    def test_digest_ignores_key_insertion_order(self):
        self.assertEqual(receipts.digest({'a': 1, 'b': 2}),
                         receipts.digest({'b': 2, 'a': 1}))

    def test_canonical_rejects_nan(self):
        with self.assertRaises(ValueError):
            receipts.digest({'x': float('nan')})


if __name__ == '__main__':
    unittest.main(verbosity=2)
