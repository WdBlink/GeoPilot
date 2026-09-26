"""Offline public-release configuration check; no API request is sent."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from provider import config


class ConfigTest(unittest.TestCase):
    def test_environment_only_and_explicit_failures(self):
        values = {'MINIMAX_API_KEY': 'synthetic-not-a-key',
                  'MINIMAX_BASE_URL': 'https://example.invalid/v1',
                  'MINIMAX_MODEL': 'MiniMax-M3'}
        with patch.object(Path, 'read_text', side_effect=AssertionError('private file read')):
            with patch.dict(os.environ, values, clear=True):
                self.assertEqual(config(), values)
            for changed in ({}, {**values, 'MINIMAX_BASE_URL': 'http://example.invalid'},
                            {**values, 'MINIMAX_MODEL': 'other'}):
                with patch.dict(os.environ, changed, clear=True), self.assertRaises(ValueError):
                    config()


if __name__ == '__main__':
    unittest.main()
