#!/usr/bin/env python3
"""Fetch and verify the separately licensed RSI-Harness dependency, then build it."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def verify(source, lock):
    for name, expected in lock['files_sha256'].items():
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or unsafe pinned source: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Pinned source hash mismatch: {name}')
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != lock['commit']:
        raise ValueError('Upstream commit mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Verify an existing checkout without network or build')
    args = parser.parse_args()
    source = ROOT / 'code/geopilot_rsih/upstream'
    lock = json.loads((source.parent / 'upstream-lock.json').read_text())
    if args.verify_only:
        verify(source, lock)
    else:
        if source.exists():
            parser.error('upstream already exists; use --verify-only (existing files are never replaced)')
        print('RSI-Harness is fetched separately; no license was declared in the pinned snapshot.', flush=True)
        print('GeoPilot does not relicense that dependency. Review its upstream terms before reuse.', flush=True)
        # Build in a disposable sibling; a failed download/build never leaves a half-installed runtime.
        with tempfile.TemporaryDirectory(prefix='.rsih-setup-', dir=source.parent) as temporary:
            checkout = Path(temporary) / 'source'
            subprocess.run(['git', 'init', '-q', str(checkout)], check=True)
            subprocess.run(['git', '-C', str(checkout), 'fetch', '--depth', '1', lock['repository'], lock['commit']], check=True)
            subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
            verify(checkout, lock)
            subprocess.run(['npm', 'ci', '--ignore-scripts', '--no-audit', '--no-fund'], cwd=checkout, check=True)
            subprocess.run(['npm', 'run', 'build'], cwd=checkout, check=True)
            verify(checkout, lock)
            shutil.move(str(checkout), source)
    print(f'Verified {len(lock["files_sha256"])} pinned source files at {lock["commit"]}')


if __name__ == '__main__':
    main()
