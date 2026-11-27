"""scripts/submit.sh refuses a --wrap that /bin/sh (dash) cannot run, before it queues.

Why (experiments_md 20261003_05 §5 item 59, 2026-11-27): item 59a's scoring job 37205 was submitted with
`--wrap="set -eo pipefail; singularity exec ... | tee ..."`. sbatch runs a wrap as `#!/bin/sh`, i.e. dash, which has no
pipefail: the job waited for 37203 / 37204 and then died at its first line (`set: Illegal option -o pipefail`). Its 59b twin
37213 was pending with the same line. Both were resubmitted (37218, 37219) with the pipe inside `bash -c '...'`.
Run from the repo root (container or host; stdlib only).
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'tools' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import wrap_check  # noqa: E402

SIF = '/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif'
INNER = ("python3 analysis/item59_bn_split.py manifest /home/koyama/data/item59/item59a_manifest.json && python3 "
         "analysis/item38_kitti_rings.py ap /home/koyama/data/item59/item59a_manifest.json "
         "/home/koyama/data/item59/item59a_ap.json --real-kernel --workers 4")
EXEC = 'singularity exec --nv --bind /home/koyama/data/:/storage %s bash -c ' % SIF
# 37205's wrap, as sacct records its submit line
PRE_FIX = ("set -eo pipefail; " + EXEC + "'set -eo pipefail; cd /home/koyama/code/ST3D/tools && " + INNER + "' | tee "
           "/home/koyama/data/item59/item59a_read.txt")
# 37218's wrap: the pipe and pipefail inside bash -c
FIXED = (EXEC + "'set -eo pipefail; cd /home/koyama/code/ST3D/tools; { " + INNER + "; } | tee "
         "/home/koyama/data/item59/item59a_read.txt'")


def test_pre_fix_wrap_is_refused_and_the_fixed_one_is_not():
    assert wrap_check.problems(PRE_FIX, dash='')
    assert not wrap_check.problems(FIXED, dash='')


def test_pipefail_counts_only_at_the_top_level():
    assert wrap_check.problems('set -o pipefail; a | b', dash='')
    assert not wrap_check.problems("bash -c 'set -o pipefail; a | b'", dash='')
    assert not wrap_check.problems('bash -c "set -o pipefail; a | b"', dash='')
    assert not wrap_check.problems('echo \\"; bash -c "set -o pipefail"', dash='')   # an escaped quote opens nothing
    assert not wrap_check.problems('set -e; C=$(x); y "$C"', dash='')                 # launch-job §5's pattern


def test_wrap_values_are_found_in_both_spellings():
    assert wrap_check.wrap_texts(['--gres=gpu:1', '--wrap=a b', '--job-name=x']) == ['a b']
    assert wrap_check.wrap_texts(['--partition', 'a100', '--wrap', 'c d']) == ['c d']
    assert wrap_check.wrap_texts(['scripts/run_sourceonly_2gpu.sh', 'cfg.yaml']) == []


def test_dash_syntax_errors_are_refused_when_dash_exists():
    import shutil
    if not shutil.which('dash'):
        return
    assert wrap_check.problems('cat <<< x')                  # a bash here-string: not dash syntax
    assert not wrap_check.problems(FIXED)


def _submit(tmp_path, wrap):
    calls = tmp_path / 'sbatch_calls.txt'
    fake = tmp_path / 'sbatch'
    fake.write_text('#!/bin/sh\necho "$@" >> %s\necho 99999\n' % calls)
    fake.chmod(0o755)
    env = dict(os.environ, PATH='%s:%s' % (tmp_path, os.environ['PATH']), WANDB_NOTES='test of the wrap check')
    env.pop('EST_H', None)
    p = subprocess.run(['bash', str(SCRIPTS / 'submit.sh'), '--gres=gpu:1', '--wrap=' + wrap], env=env,
                       capture_output=True, text=True, timeout=60)
    return p, (calls.read_text() if calls.exists() else '')


def test_submit_sh_refuses_the_pre_fix_wrap_before_sbatch(tmp_path):
    p, calls = _submit(tmp_path, PRE_FIX)
    assert p.returncode == 2 and 'pipefail' in p.stderr and calls == ''


def test_submit_sh_passes_the_fixed_wrap_to_sbatch(tmp_path):
    p, calls = _submit(tmp_path, FIXED)
    assert p.returncode == 0, p.stderr
    assert 'submitted job 99999' in p.stdout and calls.count('--wrap=') == 1
