"""A SELF_TRAIN config must set USE_PSEUDO_LABEL on its target, or it dies minutes in.

Job 25933 (the Lyft foreground row) reached the training loop and raised:

    train_st_utils.py:138  pos_pseudo_bbox = target_batch['pos_ps_bbox'].mean(dim=0)
    KeyError: 'pos_ps_bbox'

`pos_ps_bbox` and `ign_ps_bbox` are set only by `DatasetTemplate.fill_pseudo_labels`, which each
dataset's `__getitem__` calls only when `USE_PSEUDO_LABEL` is set AND self.training. Without the
key the target yields its REAL labels (which self-training must not use) and no counters, and
`train_one_epoch_st` reads those counters every iteration.

All FOUR da-ieee-access foreground rows declared SELF_TRAIN and none declared USE_PSEUDO_LABEL, so
none of them could ever have run - including the single-source PandaSet one that earlier looked
fine because it passed an unrelated arity check. The MIRU2025 self-training configs, which did run,
all carry it.

This test is static: reaching fill_pseudo_labels for real needs saved pseudo-labels on disk, which
only exist after a generation pass on a GPU.
"""
import re
from pathlib import Path

import pytest
import yaml

FAMILY = Path(__file__).resolve().parent.parent / 'tools' / 'cfgs' / 'da-ieee-access'


def _self_training_configs():
    out = []
    for path in sorted(FAMILY.glob('centerpoint-*.yaml')):
        text = path.read_text(encoding='utf-8')
        if re.search(r'^SELF_TRAIN:', text, re.M):
            out.append(path)
    return out


def test_the_family_has_self_training_rows_to_check():
    """Guard against the glob silently matching nothing and the suite passing vacuously."""
    assert _self_training_configs(), 'no SELF_TRAIN configs found - has the family moved?'


@pytest.mark.parametrize('path', _self_training_configs(), ids=lambda p: p.stem)
def test_target_declares_use_pseudo_label(path):
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    tar = cfg.get('DATA_CONFIG_TAR')
    assert tar is not None, f'{path.name}: SELF_TRAIN with no DATA_CONFIG_TAR'
    assert tar.get('USE_PSEUDO_LABEL') is True, (
        f'{path.name}: SELF_TRAIN is set but DATA_CONFIG_TAR lacks USE_PSEUDO_LABEL: True, so '
        f'fill_pseudo_labels never runs and train_one_epoch_st raises KeyError on pos_ps_bbox'
    )


@pytest.mark.parametrize('path', _self_training_configs(), ids=lambda p: p.stem)
def test_target_does_not_also_ask_for_motion_compensation(path):
    """The two are mutually exclusive, and the loader raises rather than silently choosing.

    GT_BOXES_MOTION_COMPENSATION consumes gt_boxes. On a USE_PSEUDO_LABEL dataset those are the
    target's REAL labels - the ones the method may not see - so motion_compensation.py refuses the
    combination. Adding USE_PSEUDO_LABEL to a target that already asked for compensation would turn
    a silent problem into a hard failure at startup, so pin that it does not happen.
    """
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    tar = cfg['DATA_CONFIG_TAR']
    assert not tar.get('GT_BOXES_MOTION_COMPENSATION', False), (
        f'{path.name}: target asks for both USE_PSEUDO_LABEL and GT_BOXES_MOTION_COMPENSATION; '
        f'the loader raises on that combination - compensation belongs on the labelled source'
    )
