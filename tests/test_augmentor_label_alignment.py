"""DataAugmentor.forward filters gt_classes / gt_scores with the boxes (experiments_md 20261003_05 §5 item 24, R7).

prepare_data indexes gt_classes by POSITION after the augmentor's class filter, so an unfiltered array shifted every
label after a dropped non-class box (a KITTI Van before a Car made the Car IGNORE; a Cyclist after two Cars became a
Car). These tests pin the fix on the augmentor alone, with an empty augmentation queue.
"""
import numpy as np
import pytest
from easydict import EasyDict

from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
from pcdet.utils import common_utils

IGN = common_utils.IGNORE_CLASS_LABEL


def _augmentor():
    return DataAugmentor('.', EasyDict(AUG_CONFIG_LIST=[], DISABLE_AUG_LIST=[]), ['Car', 'Pedestrian', 'Cyclist'])


def _frame(names, gt_classes=None, gt_scores=None, ignore_mask=None):
    names = np.array(names)
    d = {'points': np.zeros((4, 4), np.float32), 'gt_boxes': np.arange(len(names) * 7, dtype=np.float32).reshape(-1, 7),
         'gt_names': names, 'gt_boxes_mask': np.isin(names, ['Car', 'Pedestrian', 'Cyclist'])}
    if gt_classes is not None:
        d['gt_classes'] = np.asarray(gt_classes, np.int32)
    if gt_scores is not None:
        d['gt_scores'] = np.asarray(gt_scores, np.float32)
    if ignore_mask is not None:
        d['gt_boxes_ignore_mask'] = np.asarray(ignore_mask, bool)
    return d


def test_classes_follow_boxes_past_dropped_non_class_boxes():
    # ['Van', 'Car', 'Car', 'Cyclist'] with the second Car ignored: base labels [1, 1, 3] -> [1, -99, 3], not shifted
    out = _augmentor().forward(_frame(['Van', 'Car', 'Car', 'Cyclist'], [IGN, 1, IGN, 3], [1, .9, .8, .7]))
    assert out['gt_names'].tolist() == ['Car', 'Car', 'Cyclist']
    assert out['gt_classes'].tolist() == [1, IGN, 3]
    np.testing.assert_allclose(out['gt_scores'], [.9, .8, .7])
    assert out['gt_boxes'][:, 0].tolist() == [7.0, 14.0, 21.0]  # rows 1-3 of the input, limit_period touches col 6


def test_all_class_names_is_a_no_op_for_the_pseudo_label_route():
    # pseudo-label names are derived from gt_classes, so every box is in CLASS_NAMES and nothing moves
    out = _augmentor().forward(_frame(['Car', 'Pedestrian', 'Car'], [-1, 2, 1], [.3, .9, .8]))
    assert out['gt_classes'].tolist() == [-1, 2, 1]
    np.testing.assert_allclose(out['gt_scores'], [.3, .9, .8])


def test_absent_classes_unchanged_behaviour():
    out = _augmentor().forward(_frame(['Truck', 'Car', 'Pedestrian']))
    assert 'gt_classes' not in out and out['gt_names'].tolist() == ['Car', 'Pedestrian']


def test_ignore_other_classes_keeps_classes_of_kept_boxes():
    names = ['Truck', 'Car', 'Pedestrian']
    out = _augmentor().forward(_frame(names, [IGN, 1, 2], [1, 1, 1], ignore_mask=[True, False, False]))
    assert out['gt_names'].tolist() == names and out['gt_classes'].tolist() == [IGN, 1, 2]


def test_length_mismatch_is_loud():
    with pytest.raises(AssertionError, match='gt_classes has 2 entries for 3 boxes'):
        _augmentor().forward(_frame(['Van', 'Car', 'Car'], [1, 1]))
