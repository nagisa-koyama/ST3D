"""CenterHead SEP_LOSS_WEIGHTS: ST3D++'s per-domain loss weights on CenterPoint (experiments_md 20261003_05 §5 item 51).

The self-training loop stamps cfg.SELF_TRAIN.{SRC,TAR}.SEP_LOSS_WEIGHTS on a batch (train_st_utils). Before
2026-11-22 CenterPoint ignored the key silently. CenterHead now takes {HM_CLASS_WEIGHTS: per class, LOC_WEIGHT} and
refuses SECOND-IoU's [cls, box, iou] list. Absent = unchanged. CPU only; the head is built as in
test_center_head_ignore_regions.py (its __init__ calls .cuda()).
"""
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_center_head_ignore_regions import FMAP, _box, _head  # noqa: E402

from pcdet.utils import loss_utils  # noqa: E402


def _setup(sep):
    head = _head()
    gt = torch.tensor([[_box(-4.0, 0.0, 1), _box(4.0, 0.0, 2), _box(0.0, 5.0, 3)]], dtype=torch.float32)
    targets = head.assign_targets(gt, feature_map_size=FMAP)
    torch.manual_seed(0)
    pred = {'hm': torch.randn(1, 3, *FMAP), 'center': torch.randn(1, 2, *FMAP), 'center_z': torch.randn(1, 1, *FMAP),
            'dim': torch.randn(1, 3, *FMAP), 'rot': torch.randn(1, 2, *FMAP)}
    head.forward_ret_dict = {'pred_dicts': [{k: v.clone() for k, v in pred.items()}], 'target_dicts': targets,
                             'sep_loss_weights': sep}
    loss, tb = head.get_loss()
    return float(loss), tb


def test_absent_is_unchanged_and_all_ones_equals_absent():
    ref, tb = _setup(None)
    ones, tb1 = _setup(EasyDict({'HM_CLASS_WEIGHTS': [1.0, 1.0, 1.0], 'LOC_WEIGHT': 1.0}))
    assert ones == pytest.approx(ref, rel=1e-6)
    assert tb1['hm_loss_head_0'] == pytest.approx(tb['hm_loss_head_0'], rel=1e-6)


def test_class_weight_zero_removes_that_channel_and_keeps_the_normaliser():
    pred = torch.rand(2, 3, 8, 8).clamp(0.05, 0.95)
    gt = torch.zeros(2, 3, 8, 8)
    gt[0, 0, 1, 1] = gt[0, 1, 4, 4] = gt[1, 2, 6, 2] = 1.0
    full = loss_utils.neg_loss_cornernet(pred, gt)
    parts = [loss_utils.neg_loss_cornernet(pred, gt, class_weights=[float(c == k) for c in range(3)]) for k in range(3)]
    assert float(sum(parts)) == pytest.approx(float(full), rel=1e-6)  # channels add up under one normaliser (3 peaks)
    w = loss_utils.neg_loss_cornernet(pred, gt, class_weights=[0.0, 1.0, 1.0])
    assert float(w) == pytest.approx(float(parts[1] + parts[2]), rel=1e-6)


def test_st3dpp_source_weights_zero_the_car_heatmap_and_keep_the_box_loss():
    _, ref = _setup(None)
    _, tb = _setup(EasyDict({'HM_CLASS_WEIGHTS': [0.0, 1.0, 1.0], 'LOC_WEIGHT': 1.0}))
    assert tb['loc_loss_head_0'] == pytest.approx(ref['loc_loss_head_0'], rel=1e-6)
    assert tb['hm_loss_head_0'] < ref['hm_loss_head_0']
    _, tb0 = _setup(EasyDict({'HM_CLASS_WEIGHTS': [1.0, 1.0, 1.0], 'LOC_WEIGHT': 0.0}))
    assert tb0['loc_loss_head_0'] == 0.0 and tb0['hm_loss_head_0'] == pytest.approx(ref['hm_loss_head_0'], rel=1e-6)


def test_the_second_iou_list_format_and_wrong_lengths_are_refused():
    with pytest.raises(ValueError):
        _setup([0.0, 1.0, 0.0])
    with pytest.raises(AssertionError):
        _setup(EasyDict({'HM_CLASS_WEIGHTS': [0.0, 1.0]}))
    with pytest.raises(AssertionError):
        _setup(EasyDict({'HM_CLASS_WEIGHT': [0.0, 1.0, 1.0]}))  # misspelled key


def test_each_forward_overwrites_the_previous_batch_weights():
    head = _head()
    head.forward_ret_dict = {'sep_loss_weights': {'LOC_WEIGHT': 0.0}}
    head.shared_conv = torch.nn.Identity()
    head.domain_discriminator = None
    head.heads_list = []
    head.predict_boxes_when_training = False
    head.training = True
    try:
        head.forward({'spatial_features_2d': torch.zeros(1, 1, *FMAP)})  # no SEP_LOSS_WEIGHTS on this batch
    except (IndexError, KeyError):
        pass  # the stub has no heads; the weights are set before they are needed
    assert head.forward_ret_dict['sep_loss_weights'] is None
