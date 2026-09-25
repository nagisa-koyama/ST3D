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


# --- the runtime guard ---------------------------------------------------------------------

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from easydict import EasyDict  # noqa: E402
from pcdet.datasets import assert_target_labels_are_not_used  # noqa: E402


def _cfg(self_train=False, pseudo=False, unsupervised=False, has_target=True):
    c = {}
    if has_target:
        tar = {}
        if pseudo:
            tar['USE_PSEUDO_LABEL'] = True
        if unsupervised:
            tar['UNSUPERVISED'] = True
        c['DATA_CONFIG_TAR'] = tar
    if self_train:
        c['SELF_TRAIN'] = {'TAR': {'LOSS_WEIGHT': 1.0}}
    return EasyDict(c)


def test_self_training_without_pseudo_labels_is_refused():
    """The exact shape of job 25933 - and the dangerous part is what it would have done if the
    missing counter had been optional: trained on the target's real labels and reported it as UDA."""
    with pytest.raises(AssertionError, match='USE_PSEUDO_LABEL'):
        assert_target_labels_are_not_used(_cfg(self_train=True), True)


def test_self_training_with_pseudo_labels_is_allowed():
    assert_target_labels_are_not_used(_cfg(self_train=True, pseudo=True), True)


def test_trained_target_with_neither_flag_is_refused():
    """A DANN/UADA3D target with no flag: its GT reaches the augmentor and the optimiser."""
    with pytest.raises(AssertionError, match='UNSUPERVISED'):
        assert_target_labels_are_not_used(_cfg(), True)


def test_unsupervised_target_is_allowed():
    assert_target_labels_are_not_used(_cfg(unsupervised=True), True)


def test_both_flags_together_are_refused():
    """They are different mechanisms and UNSUPERVISED would suppress the pseudo-label path."""
    with pytest.raises(ValueError, match='both'):
        assert_target_labels_are_not_used(_cfg(self_train=True, pseudo=True, unsupervised=True), True)


def test_source_only_config_with_no_target_is_untouched():
    """DATA_CONFIG_TAR is eval-only for the source-only family; it is never trained on."""
    assert_target_labels_are_not_used(_cfg(has_target=False), True)


@pytest.mark.parametrize('path', _self_training_configs(), ids=lambda p: p.stem)
def test_every_real_self_training_config_passes_the_guard(path):
    cfg = EasyDict(yaml.safe_load(path.read_text(encoding='utf-8')))
    assert_target_labels_are_not_used(cfg, True)


def test_source_only_eval_target_is_not_refused():
    """A source-only row declares DATA_CONFIG_TAR as an EVALUATION target only. Reading its real
    labels there is the point, so the guard must not fire when nothing trains on it."""
    assert_target_labels_are_not_used(_cfg(), False)
