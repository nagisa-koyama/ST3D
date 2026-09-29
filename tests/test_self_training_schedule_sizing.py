"""The OneCycle plan of a self-training run must be sized by the same rule the loop runs by.

Until 2026-09-29 train.py sized the plan from the TARGET loader for every SELF_TRAIN run while
train_model_st ran the loop by SELF_TRAIN.EPOCH_FOLLOWS. Every da-ieee-access self-training config
follows the SOURCE (Lyft: 3,150 iterations per epoch against nuScenes' 4,689), so those rows took
94,500 steps on a plan of 140,670 and stopped at 45% of the anneal - final LR ~1.7e-3 against ~0 for
a source-only row. Both now call self_training_iters_per_epoch; these tests pin the rule and that
both callers use it. experiments_md/20260928_01 section 4.
"""
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))
import _init_path  # noqa: F401,E402
from train_utils.train_st_utils import self_training_iters_per_epoch  # noqa: E402


class _Loader:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n




def _lyft_like():
    # two Lyft platforms at global batch 6, and the nuScenes target
    return [_Loader(2583), _Loader(567)], _Loader(4689)


def test_source_following_epoch_sums_every_source_loader():
    srcs, tgt = _lyft_like()
    iters, follows = self_training_iters_per_epoch(EasyDict(EPOCH_FOLLOWS='source'), srcs, tgt)
    assert (iters, follows) == (3150, 'source')


def test_default_follows_the_target_so_pre_existing_configs_are_unchanged():
    srcs, tgt = _lyft_like()
    assert self_training_iters_per_epoch(EasyDict(), srcs, tgt) == (4689, 'target')


def test_merge_all_iters_divides_the_target_and_refuses_source():
    srcs, tgt = _lyft_like()
    assert self_training_iters_per_epoch(EasyDict(EPOCH_FOLLOWS='target'), srcs, tgt, True, 30) \
        == (4689 // 30, 'target')
    with pytest.raises(AssertionError, match='TARGET length'):
        self_training_iters_per_epoch(EasyDict(EPOCH_FOLLOWS='source'), srcs, tgt, True, 30)


def test_an_unknown_value_is_refused():
    srcs, tgt = _lyft_like()
    with pytest.raises(AssertionError, match='EPOCH_FOLLOWS'):
        self_training_iters_per_epoch(EasyDict(EPOCH_FOLLOWS='both'), srcs, tgt)


def test_plan_and_loop_agree_for_the_family():
    """The plan is built from the number the loop will actually run - the whole point."""
    srcs, tgt = _lyft_like()
    cfg = EasyDict(EPOCH_FOLLOWS='source')
    plan_iters, _ = self_training_iters_per_epoch(cfg, srcs, tgt)
    loop_iters, _ = self_training_iters_per_epoch(cfg, srcs, tgt)
    assert plan_iters * 30 == loop_iters * 30 == 94500


def test_both_callers_use_the_shared_rule():
    train_py = (ROOT / 'tools/train.py').read_text(encoding='utf-8')
    st = (ROOT / 'tools/train_utils/train_st_utils.py').read_text(encoding='utf-8')
    assert 'self_training_iters_per_epoch(' in train_py.split('create scheduler')[1]
    loop = st[st.index('def train_model_st('):]
    assert 'self_training_iters_per_epoch(' in loop
    assert "cfg.SELF_TRAIN.get('EPOCH_FOLLOWS'" not in loop, 'the loop must not re-derive the rule'
    # the old, target-only sizing must be gone from train.py's SELF_TRAIN branch
    branch = train_py.split("if cfg.get('SELF_TRAIN', None):\n        # The SAME rule")[1].split('else:')[0]
    assert 'total_iters_each_epoch_per_dataloader(target_loader' not in branch
