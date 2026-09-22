"""An adam_onecycle config may omit the LambdaLR-only keys.

build_scheduler() read DECAY_STEP_LIST eagerly, before branching on the optimizer, so a onecycle
config that dropped it - correctly, as a key whose value that branch never uses - died with
AttributeError. Job 25745 was lost to this. The same shape as FOV_POINTS_ONLY: absent was fatal
while present-and-ignored was fine.
"""
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))

from train_utils.optimization import build_scheduler  # noqa: E402

BASE = dict(OPTIMIZER='adam_onecycle', LR=0.003, MOMS=[0.95, 0.85], DIV_FACTOR=10, PCT_START=0.4)


def _opt():
    return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.003)


def test_onecycle_builds_without_the_lambdalr_only_keys():
    sched, warmup = build_scheduler(_opt(), total_iters_each_epoch=10, total_epochs=2,
                                    last_epoch=-1, optim_cfg=EasyDict(BASE))
    assert sched is not None and warmup is None


def test_onecycle_still_builds_when_they_are_present():
    cfg = EasyDict(dict(BASE, DECAY_STEP_LIST=[15, 19], LR_DECAY=0.1, LR_CLIP=1e-7,
                        LR_WARMUP=False, WARMUP_EPOCH=1))
    assert build_scheduler(_opt(), 10, 2, -1, cfg)[0] is not None


def test_lambdalr_branch_still_requires_them():
    """The keys are only dead for onecycle; the other branch genuinely consumes them."""
    cfg = EasyDict(dict(BASE, OPTIMIZER='adam', DECAY_STEP_LIST=[1], LR_DECAY=0.1, LR_CLIP=1e-7,
                        LR_WARMUP=False))
    assert build_scheduler(_opt(), 10, 2, -1, cfg)[0] is not None
