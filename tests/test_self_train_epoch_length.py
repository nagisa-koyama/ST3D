"""SELF_TRAIN.EPOCH_FOLLOWS decides what an epoch counts.

train.py measures an epoch as one pass over the SOURCE. train_st_utils.py has always measured it
as one pass over the TARGET. At the same NUM_EPOCHS those differ, so a self-training arm and a
source-only arm are not comparable unless the self-training arm is told to follow the source -
otherwise part of what an ablation measures is training length.

The default must stay 'target': every existing SELF_TRAIN config assumes it, and silently changing
the number of gradient steps would alter every reproduction.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SRC = (ROOT / 'tools/train_utils/train_st_utils.py').read_text(encoding='utf-8')


def test_default_is_target_so_existing_configs_are_unchanged():
    assert "cfg.SELF_TRAIN.get('EPOCH_FOLLOWS', 'target')" in SRC


def test_source_branch_sums_every_source_loader():
    """Multi-source runs declare DATA_CONFIGS; one pass means all of them, not the first."""
    assert 'sum(len(r.dataloader) for r in source_readers)' in SRC


def test_the_value_is_validated():
    assert "must be 'source' or 'target'" in SRC


def test_merge_all_iters_is_refused_when_following_the_source():
    """It divides the TARGET length, so it is meaningless under source-driven epochs."""
    i = SRC.index('if merge_all_iters_to_one_epoch:')
    assert "assert epoch_follows == 'target'" in SRC[i:i + 400]


def test_target_iterator_restart_is_gated_on_the_lengths_matching():
    """Under source-driven epochs the target must run continuously across epoch boundaries."""
    assert 'if total_it_each_epoch == len(target_loader):' in SRC


def test_target_is_cycled_on_exhaustion():
    """Source-driven epochs can outrun the target, so StopIteration must restart it."""
    i = SRC.index('except StopIteration:')
    assert 'dataloader_iter = iter(target_loader)' in SRC[i:i + 200]


@pytest.mark.parametrize('cfg_name', [
    'centerpoint-accum-foreground-nuscenes2kitti.yaml',
    'centerpoint-foreground-lyft2nuscenes.yaml',
])
def test_ablation_self_training_arms_follow_the_source(cfg_name):
    """Otherwise the arms of each ablation differ in gradient steps per epoch."""
    import yaml
    text = (ROOT / 'tools/cfgs/da-ieee-access' / cfg_name).read_text(encoding='utf-8')
    cfg = yaml.safe_load(text)
    assert cfg['SELF_TRAIN'].get('EPOCH_FOLLOWS') == 'source'
