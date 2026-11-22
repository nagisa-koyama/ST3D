"""The launcher's code snapshot survives a concurrent git operation (tools/scripts/run_sourceonly_2gpu.sh).

Rows 35179 / 35181 (experiments_md T05 §5 item 51) died at 2026-11-22 14:28, before training: a git fetch in the shared
checkout removed its `.git/objects/pack/tmp_pack_*` while the job's rsync was copying the tree, rsync exited 24
("some files vanished") and `set -euo pipefail` stopped the job. These tests run the launcher's own snapshot block,
read from the script, under `set -euo pipefail` with a stub rsync: exit 24 must continue to the "code frozen" line,
any other failure must still stop the job. CPU only.
"""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'tools' / 'scripts' / 'run_sourceonly_2gpu.sh'


def _snapshot_block():
    src = LAUNCHER.read_text()
    start = src.index('SNAP=$SCRATCH/ST3D\n')
    end = src.index('echo "=== code frozen at', start)
    return src[start:src.index('\n', end) + 1]


def _run(tmp_path, rc):
    stub = tmp_path / 'bin'
    stub.mkdir(exist_ok=True)
    (stub / 'rsync').write_text('#!/usr/bin/env bash\nexit %d\n' % rc)
    (stub / 'rsync').chmod(0o755)
    script = 'set -euo pipefail\nREPO=%s\nSCRATCH=%s\n%s' % (ROOT, tmp_path / 'scratch', _snapshot_block())
    env = dict(os.environ, PATH='%s:%s' % (stub, os.environ['PATH']))
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True, env=env)


def test_the_block_is_the_launchers_rsync_snapshot():
    block = _snapshot_block()
    assert 'rsync -a' in block and '"$REPO/" "$SNAP/"' in block


@pytest.mark.parametrize('rc', [0, 24])
def test_a_clean_copy_or_a_vanished_file_reaches_the_frozen_line(tmp_path, rc):
    r = _run(tmp_path, rc)
    assert r.returncode == 0, r.stderr
    assert '=== code frozen at' in r.stdout


@pytest.mark.parametrize('rc', [1, 23, 255])
def test_any_other_rsync_failure_still_stops_the_job(tmp_path, rc):
    r = _run(tmp_path, rc)
    assert r.returncode == rc
    assert '=== code frozen at' not in r.stdout
