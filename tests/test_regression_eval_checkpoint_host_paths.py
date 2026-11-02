"""analysis/eval_checkpoint.sh checks its config and checkpoint on the HOST, before any container starts.

Defect (a usage trap, twice): rescoring jobs 27280 and 27568 were submitted with `/storage/...` checkpoint
paths - the CONTAINER's bind of /home/koyama/data/ - and died at once with "no such checkpoint", because the
script tests the file on the host. The fix is a rule (memory/repo/bug_prevention_checklist.md rule 6: host-side
scripts take /home/koyama/data/..., never /storage/...), so this test pins the behaviour the rule depends on:
(1) a container-only path is refused with exit 2 and a message naming the offending argument, (2) that refusal
happens BEFORE `singularity exec`, so a wrong path costs seconds, not a GPU allocation, and (3) the host path of
the same file passes the check and is handed to test.py unchanged inside the bind that maps it to /storage.
Reports: experiments_md/20261003_14 (infrastructure), 20261003_13 §3.M.

CPU only: `singularity` is shadowed by a fake on PATH that records its argv and exits 0, so nothing runs.
Limit: the suite itself runs INSIDE the container, where /storage is bound, so a real file's /storage spelling
cannot be made to fail here the way it fails on a compute node (no /storage on the host); the refusal is therefore
exercised with a /storage path that exists nowhere, which takes the identical branch of the script.
"""
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools' / 'analysis' / 'eval_checkpoint.sh'
CFG = 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti.yaml'
HOST_DATA = Path('/home/koyama/data')


def _some_host_data_file():
    """Any regular file under the data root, in its host AND container spelling."""
    if not HOST_DATA.is_dir():
        pytest.skip('data root %s not present on this host' % HOST_DATA)
    for p in sorted(HOST_DATA.iterdir()):
        if p.is_file():
            return str(p), '/storage/' + p.name
    pytest.skip('no regular file directly under %s' % HOST_DATA)


@pytest.fixture
def fake_singularity(tmp_path):
    """A `singularity` on PATH that logs its argv to a file and exits 0."""
    log = tmp_path / 'singularity_argv.txt'
    exe = tmp_path / 'bin' / 'singularity'
    exe.parent.mkdir()
    exe.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "%s"\nexit 0\n' % ('%s', log))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ, PATH=str(exe.parent) + os.pathsep + os.environ.get('PATH', ''))
    return env, log


def _run(args, env):
    pinned = Path('/home/koyama/code/ST3D/tools')
    if not pinned.is_dir() or not (ROOT / 'tools').samefile(pinned):
        pytest.skip('the script hardcodes /home/koyama/code/ST3D/tools; this checkout is elsewhere')
    return subprocess.run(['bash', str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)


def test_a_container_path_is_refused_before_the_launch(fake_singularity):
    env, log = fake_singularity
    storage_path = '/storage/checkpoints/does-not-exist-on-the-host/checkpoint_epoch_20.pth'
    r = _run([CFG, storage_path], env)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert 'no such checkpoint' in r.stderr and storage_path in r.stderr, r.stderr
    assert not log.exists(), 'singularity was launched although the checkpoint check failed'


def test_a_missing_config_is_refused_the_same_way(fake_singularity):
    env, log = fake_singularity
    host_path, _ = _some_host_data_file()
    r = _run(['cfgs/da-ieee-access/does-not-exist.yaml', host_path], env)
    assert r.returncode == 2 and 'no such config' in r.stderr, (r.returncode, r.stderr)
    assert not log.exists()


def test_the_host_path_of_the_same_file_passes_and_reaches_test_py(fake_singularity):
    env, log = fake_singularity
    host_path, _ = _some_host_data_file()
    r = _run([CFG, host_path, 'unit_test_tag'], env)
    assert r.returncode == 0, (r.returncode, r.stderr)
    argv = log.read_text(encoding='utf-8').splitlines()
    assert '--bind' in argv and '/home/koyama/data/:/storage' in argv, argv
    assert 'test.py' in argv[argv.index('python3') + 1], argv
    assert argv[argv.index('--ckpt') + 1] == host_path, 'the checkpoint path must be handed on unchanged'
    assert argv[argv.index('--cfg_file') + 1] == CFG
    assert argv[argv.index('--eval_tag') + 1] == 'unit_test_tag'


def test_the_file_checks_precede_the_launch_in_the_script_text():
    """Static form of the same guarantee, for a reader of the script."""
    lines = SCRIPT.read_text(encoding='utf-8').splitlines()
    cfg_check = next(i for i, l in enumerate(lines) if re.search(r'\[ -f "\$CFG" \]', l))
    ckpt_check = next(i for i, l in enumerate(lines) if re.search(r'\[ -f "\$CKPT" \]', l))
    launch = next(i for i, l in enumerate(lines) if l.lstrip().startswith('singularity exec'))
    assert cfg_check < launch and ckpt_check < launch
    assert 'exit 2' in lines[cfg_check] and 'exit 2' in lines[ckpt_check]
