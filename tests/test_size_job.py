"""Pins the sizing rules scripts/submit.sh applies when EST_H is set (tools/scripts/size_job.py).

Each case runs the script as submit.sh does - its own argument list, EST_H in the environment - and
asserts on the sbatch flags it prints. The configs are real ones, loaded through the repo's own
resolver, so a config whose loader count changes shows up here.
"""
import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
S = 'scripts/run_sourceonly_2gpu.sh'
SOURCEONLY = 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti.yaml'          # 1 loader, 4 workers
SELF_TRAIN = 'cfgs/da-ieee-access-tier1/centerpoint-foreground-lyft2nuscenes-4ep.yaml'  # 2 src + tgt
SIZING_ENV = ('EST_H', 'NGPU', 'CPUS_PER_GPU', 'DDP_ABOVE_H', 'CPUS_PER_GPU_MAX', 'SAFETY', 'ENTRY')


def flags(est_h, *argv, **env):
    e = {k: v for k, v in os.environ.items() if k not in SIZING_ENV}
    e.update(EST_H=str(est_h), **env)
    out = subprocess.run([sys.executable, 'scripts/size_job.py', *argv], cwd=TOOLS, env=e,
                         capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_under_a_day_is_one_gpu_with_a_proportionate_limit():
    assert flags(8, S, SOURCEONLY) == ['--gres=gpu:1', '--cpus-per-task=6', '--time=0-13:00:00']


def test_over_a_day_is_two_gpus_and_half_the_time():
    assert flags(40, S, SOURCEONLY) == ['--gres=gpu:2', '--cpus-per-task=12', '--time=1-07:00:00']


def test_exactly_a_day_stays_on_one_gpu():
    assert flags(24, S, SOURCEONLY)[0] == '--gres=gpu:1'


def test_cpus_count_every_training_loader_and_cap_at_node61_share():
    # 2 Lyft platform loaders + 1 target loader, 4 workers each: 14, capped to 10 per GPU.
    assert flags(8, S, SELF_TRAIN)[1] == '--cpus-per-task=10'
    assert flags(40, S, SELF_TRAIN)[1] == '--cpus-per-task=20'


def test_explicit_gres_sizes_cpus_and_time_for_that_count():
    assert flags(4, '--gres=gpu:2', S, SOURCEONLY) == [
        '--gres=gpu:2', '--cpus-per-task=12', '--time=0-04:00:00']


def test_scripts_without_ddp_never_get_two_gpus_by_rule():
    out = flags(30, '--wrap=bash analysis/eval_checkpoint.sh ' + SOURCEONLY + ' ck.pth')
    assert out == ['--gres=gpu:1', '--cpus-per-task=6', '--time=1-22:00:00']


def test_beyond_seven_days_narrows_to_rtx8000():
    assert flags(250, S, SOURCEONLY)[-1] == '--partition=rtx8000'


def test_missing_config_refuses():
    e = {**os.environ, 'EST_H': '1'}
    r = subprocess.run([sys.executable, 'scripts/size_job.py', S, 'cfgs/no/such.yaml'], cwd=TOOLS,
                       env=e, capture_output=True, text=True)
    assert r.returncode != 0 and r.stdout == ''
