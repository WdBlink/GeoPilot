"""Regression for exit between an RSS snapshot and leader polling."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import supervise


class SupervisionExitTest(unittest.TestCase):
    def run_case(self, members, **limits):
        proc = Mock(pid=99001)
        proc.poll.return_value = 0
        proc.wait.return_value = 0
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(supervise.subprocess, 'Popen', return_value=proc), \
                patch.object(supervise, 'group_rss', return_value=[1024]), \
                patch.object(supervise, 'group_members', return_value=members), \
                patch.object(supervise, 'stop_group') as stop:
            result = supervise.supervise(['unused'], Path(directory), **limits)
            stop.assert_called_once_with(proc.pid)
        return result

    def test_leader_exits_after_rss_snapshot(self):
        result = self.run_case([])
        self.assertEqual((result['returncode'], result['reason']), (0, None))
        self.assertEqual(result['terminal_group_members'], [])
        self.assertEqual(result['peak_worker_process_group_rss_bytes'], 1024)

    def test_live_child_still_rejects_and_records_identity(self):
        members = [{'pid': 99002, 'rss_bytes': 2048, 'state': 'S'}]
        result = self.run_case(members)
        self.assertEqual(result['reason'], 'RuntimeError: CHILD_SURVIVED_LEADER')
        self.assertEqual(result['terminal_group_members'], members)
        self.assertEqual(result['peak_worker_process_group_rss_bytes'], 2048)

    def test_limits_are_not_bypassed_by_exit(self):
        for limits, reason in [({'max_rss': 512}, 'RSS_LIMIT'),
                               ({'timeout': -1}, 'TIME_LIMIT')]:
            with self.subTest(reason=reason):
                result = self.run_case([], **limits)
                self.assertEqual(result['reason'], 'RuntimeError: ' + reason)

    def test_group_snapshot_excludes_other_groups_and_zombies(self):
        output = Mock(stdout='11 20 3 S\n12 20 9 Z\n13 21 4 R\n')
        with patch.object(supervise.subprocess, 'run', return_value=output):
            self.assertEqual(supervise.group_members(20),
                             [{'pid': 11, 'rss_bytes': 3072, 'state': 'S'}])


if __name__ == '__main__':
    unittest.main()
