"""Observe one sandboxed worker process group and retain measured costs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def group_members(pgid):
    listing = subprocess.run(['/bin/ps', '-axo', 'pid=,pgid=,rss=,state='],
                             capture_output=True, text=True, check=True)
    rows = [line.split() for line in listing.stdout.splitlines()]
    return [{'pid': int(row[0]), 'rss_bytes': int(row[2]) * 1024, 'state': row[3]}
            for row in rows if len(row) == 4 and int(row[1]) == pgid and 'Z' not in row[3]]


def group_rss(pgid):
    return [member['rss_bytes'] for member in group_members(pgid)]


def stop_group(pgid):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + (5 if sig == signal.SIGTERM else 2)
        while time.monotonic() < deadline:
            if not group_rss(pgid):
                return
            time.sleep(.1)
    if group_rss(pgid):
        raise RuntimeError('worker process group survived SIGKILL')


def supervise(command, output, timeout=14400, max_rss=48 * 1024**3):
    """Count the entire group, including children after its leader exits."""
    start, peak, reason = time.monotonic(), 0, None
    terminal_group_members = None
    with (output / 'runtime.stdout').open('xb') as stdout, (output / 'runtime.stderr').open('xb') as stderr:
        proc = subprocess.Popen(command, cwd=output, stdout=stdout, stderr=stderr,
                                start_new_session=True)
        previous = signal.getsignal(signal.SIGTERM)
        def cancel(_signum, _frame):
            raise KeyboardInterrupt('Cancelled by signal')
        signal.signal(signal.SIGTERM, cancel)
        try:
            while True:
                values = group_rss(proc.pid)
                if values:
                    peak = max(peak, sum(values))
                elif proc.poll() is None:
                    raise RuntimeError('RSS unavailable')
                if peak > max_rss:
                    raise RuntimeError('RSS_LIMIT')
                if time.monotonic() - start > timeout:
                    raise RuntimeError('TIME_LIMIT')
                if proc.poll() is not None:
                    # The earlier sample may contain the leader that just exited.
                    terminal_group_members = group_members(proc.pid)
                    peak = max(peak, sum(member['rss_bytes'] for member in terminal_group_members))
                    if terminal_group_members:
                        raise RuntimeError('CHILD_SURVIVED_LEADER')
                    break
                time.sleep(.5)
        except BaseException as exc:
            reason = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                stop_group(proc.pid)
            finally:
                returncode = proc.wait()
                signal.signal(signal.SIGTERM, previous)
    return {'returncode': returncode, 'reason': reason,
            'wall_seconds': time.monotonic() - start,
            'peak_worker_process_group_rss_bytes': peak,
            'terminal_group_members': terminal_group_members,
            'resource_sampling_seconds': .5}


if __name__ == '__main__':
    output = Path(sys.argv[1])
    command = sys.argv[2:]
    if not command:
        raise SystemExit('command required')
    result = supervise(command, output)
    (output / 'supervision.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    raise SystemExit(0 if result['returncode'] == 0 and result['reason'] is None else 1)
