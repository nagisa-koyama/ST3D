"""PL-GT (ANALYSIS; experiments_md 20261027_05): the target's ground truth written as FROZEN pseudo-labels.

Bug-prevention rule 1 (memory/repo/bug_prevention_checklist.md): every consumer is tested against the sentinels. The
GT -> pseudo-label conversion must emit only positive model-vocabulary labels (no ignore band -1..-C, no
IGNORE_CLASS_LABEL -99, no background 0) at score 1.0, through the filters the oracle's training labels pass; the
validator refuses each sentinel; the frozen wrapper cannot be read through any other pseudo-label path; the loader keeps
the stored point count under the key only; and CenterHead turns every converted box into a positive peak with nothing
ignored. CPU only (CenterHead is built without __init__, as in test_center_head_ignore_regions.py).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.models.dense_heads.center_head import CenterHead  # noqa: E402
from pcdet.utils import common_utils, self_training_utils as stu  # noqa: E402
from pcdet.utils.ontology_mapping import get_ontology_mapping  # noqa: E402

NAMES = ['Car', 'Pedestrian', 'Cyclist']
WAYMO_TO_KITTI = get_ontology_mapping('waymo', 'kitti')


def _frame():
    names = np.array(['Vehicle', 'Pedestrian', 'Cyclist', 'Sign', 'unknown', 'Vehicle', 'Pedestrian'])
    boxes = np.array([[10, 0, 1, 4.5, 1.9, 1.6, 0.3],
                      [5, 3, 1, 0.8, 0.8, 1.8, -1.0],
                      [-8, 2, 1, 1.8, 0.6, 1.7, 3.0],
                      [3, -3, 2, 0.5, 0.1, 0.8, 0.0],
                      [20, 20, 1, 2.0, 2.0, 2.0, 0.0],
                      [30, -5, 1, 4.0, 1.8, 1.5, 1.2],     # empty: stored count 0
                      [-2, -6, 1, 0.7, 0.7, 1.7, 0.1]], dtype=np.float64)
    npts = np.array([120, 15, 3, 40, 9, 0, 1])
    return names, boxes, npts


def test_waymo_ontology_is_the_oracles():
    assert WAYMO_TO_KITTI == {'Vehicle': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist', 'Sign': 'Misc'}


def test_conversion_keeps_oracle_labels_only_positive_and_scored_one():
    names, boxes, npts = _frame()
    e, st = stu.target_gt_to_pseudo_label_entry(names, boxes, npts, WAYMO_TO_KITTI, NAMES, min_points=1)
    g = e['gt_boxes']
    assert g.shape == (4, 9) and g.dtype == np.float64
    assert g[:, 7].tolist() == [1, 2, 3, 2]                      # Vehicle -> Car; Sign, unknown, empty dropped
    assert (g[:, 7] > 0).all() and not (g[:, 7] == common_utils.IGNORE_CLASS_LABEL).any()
    assert (g[:, 8] == 1.0).all()
    np.testing.assert_array_equal(g[:, :7], boxes[[0, 1, 2, 6]])
    assert e['teacher_classes'].tolist() == ['Car', 'Pedestrian', 'Cyclist', 'Pedestrian']
    assert e['num_points_in_gt'].tolist() == [120, 15, 3, 1]
    assert (e['cls_scores'] == 1).all() and (e['iou_scores'] == 1).all() and (e['memory_counter'] == 0).all()
    assert st == {'raw': 7, 'unknown': 1, 'empty': 1, 'other_class': 1, 'kept_Car': 1, 'kept_Pedestrian': 2,
                  'kept_Cyclist': 1, 'kept': 4}
    stu.validate_target_gt_pseudo_labels({'f': e}, NAMES)


def test_conversion_applies_shift_coor_and_min_points():
    names, boxes, npts = _frame()
    e, _ = stu.target_gt_to_pseudo_label_entry(names, boxes, npts, WAYMO_TO_KITTI, NAMES, min_points=2,
                                               shift_coor=[0.0, 0.0, 1.5])
    assert e['gt_boxes'][:, 7].tolist() == [1, 2, 3]            # the 1-point pedestrian falls under MIN_POINTS 2
    np.testing.assert_allclose(e['gt_boxes'][:, 2], boxes[[0, 1, 2], 2] + 1.5)


def test_conversion_of_an_empty_frame():
    e, st = stu.target_gt_to_pseudo_label_entry(np.array([], dtype='<U10'), np.zeros((0, 7)), np.zeros(0, int),
                                                WAYMO_TO_KITTI, NAMES)
    assert e['gt_boxes'].shape == (0, 9) and st['kept'] == 0
    stu.validate_target_gt_pseudo_labels({'f': e}, NAMES)


def test_conversion_refuses_a_name_outside_the_map():
    names, boxes, npts = _frame()
    names = names.copy()
    names[0] = 'Truck'
    with pytest.raises(KeyError, match='Truck'):
        stu.target_gt_to_pseudo_label_entry(names, boxes, npts, WAYMO_TO_KITTI, NAMES)


def _entry():
    names, boxes, npts = _frame()
    return stu.target_gt_to_pseudo_label_entry(names, boxes, npts, WAYMO_TO_KITTI, NAMES)[0]


@pytest.mark.parametrize('label', [-1, -2, -3, common_utils.IGNORE_CLASS_LABEL, 0, 4, 1.5])
def test_validator_refuses_every_sentinel_label(label):
    e = _entry()
    e['gt_boxes'][0, 7] = label
    with pytest.raises(AssertionError):
        stu.validate_target_gt_pseudo_labels({'f': e}, NAMES)


@pytest.mark.parametrize('col, value', [(8, 0.9), (3, 0.0), (4, -1.0), (0, np.nan)])
def test_validator_refuses_bad_score_or_box(col, value):
    e = _entry()
    e['gt_boxes'][0, col] = value
    with pytest.raises(AssertionError):
        stu.validate_target_gt_pseudo_labels({'f': e}, NAMES)


def test_validator_refuses_an_empty_box_and_misaligned_fields():
    e = _entry()
    e['num_points_in_gt'][0] = 0
    with pytest.raises(AssertionError):
        stu.validate_target_gt_pseudo_labels({'f': e}, NAMES)
    e = _entry()
    e['teacher_classes'] = e['teacher_classes'][::-1]
    with pytest.raises(AssertionError):
        stu.validate_target_gt_pseudo_labels({'f': e}, NAMES)


def test_load_needs_the_wrapper_and_the_class_names(tmp_path):
    labels = {'seg_000': _entry()}
    p = tmp_path / 'plain.pkl'
    pickle.dump(labels, open(p, 'wb'))
    with pytest.raises(AssertionError, match='not a'):
        stu.load_analysis_target_gt_pseudo_labels(str(p), NAMES)
    w = tmp_path / 'wrapped.pkl'
    pickle.dump({'format': stu.ANALYSIS_TARGET_GT_FORMAT, 'meta': {'class_names': NAMES}, 'labels': labels}, open(w, 'wb'))
    got, meta, counts = stu.load_analysis_target_gt_pseudo_labels(str(w), NAMES)
    assert counts.tolist() == [1, 2, 1] and list(got) == ['seg_000']
    with pytest.raises(AssertionError):
        stu.load_analysis_target_gt_pseudo_labels(str(w), ['Car', 'Pedestrian'])


def test_the_wrapper_fails_loudly_through_any_other_pseudo_label_path():
    """Handed to INIT_PS or left as a ps_label_e*.pkl, the wrapper has no frame-id keys: the first frame raises."""
    wrapper = {'format': stu.ANALYSIS_TARGET_GT_FORMAT, 'meta': {}, 'labels': {'seg_000': _entry()}}
    with pytest.raises(ValueError, match='Cannot find pseudo label'):
        stu.load_ps_label('seg_000', wrapper)


def test_path_helper_is_none_without_the_key():
    assert stu.analysis_target_gt_pseudo_label_path(EasyDict({'DATA_CONFIG_TAR': {'USE_PSEUDO_LABEL': True}})) is None
    assert stu.analysis_target_gt_pseudo_label_path(EasyDict({})) is None
    c = EasyDict({'DATA_CONFIG_TAR': {common_utils.ANALYSIS_TARGET_GT_KEY: '/x.pkl'}})
    assert stu.analysis_target_gt_pseudo_label_path(c) == '/x.pkl'


def _ds(with_key, labels):
    ds = DatasetTemplate.__new__(DatasetTemplate)
    cfg = {'USE_PSEUDO_LABEL': True}
    if with_key:
        cfg[common_utils.ANALYSIS_TARGET_GT_KEY] = '/x.pkl'
    ds.dataset_cfg = EasyDict(cfg)
    ds.class_names = list(NAMES)
    ds.pseudo_labels = labels
    return ds


@pytest.mark.parametrize('with_key', [True, False])
def test_fill_pseudo_labels_keeps_the_stored_count_only_under_the_key(with_key):
    e = _entry()
    ds = _ds(with_key, {'seg_000': e})
    # Waymo's __getitem__ puts the INFO's per-box count here (all 7 raw boxes), misaligned with any pseudo-label set
    d = {'frame_id': 'seg_000', 'num_points_in_gt': np.arange(7)}
    ds.fill_pseudo_labels(d)
    assert d['gt_boxes'].shape == (4, 7) and d['gt_names'].tolist() == ['Car', 'Pedestrian', 'Cyclist', 'Pedestrian']
    assert d['pos_ps_bbox'].tolist() == [1, 2, 1] and d['ign_ps_bbox'].tolist() == [0, 0, 0]
    if with_key:
        assert d['num_points_in_gt'].tolist() == [120, 15, 3, 1]
    else:
        assert 'num_points_in_gt' not in d     # unchanged behaviour: the loader recounts on the cloud


def test_center_head_makes_every_converted_box_a_positive_and_ignores_nothing():
    head = CenterHead.__new__(CenterHead)
    torch.nn.Module.__init__(head)
    head.model_cfg = EasyDict({'TARGET_ASSIGNER_CONFIG': {'FEATURE_MAP_STRIDE': 4, 'NUM_MAX_OBJS': 50,
                                                          'GAUSSIAN_OVERLAP': 0.1, 'MIN_RADIUS': 2}})
    head.class_names = list(NAMES)
    head.class_names_each_head = [list(NAMES)]
    head.point_cloud_range = np.array([-40.0, -40.0, -2.0, 40.0, 40.0, 4.0], dtype=np.float32)
    head.voxel_size = [0.1, 0.1, 0.15]
    g = _entry()['gt_boxes']
    gt = np.concatenate([g[:, :7], g[:, 7:8]], axis=1)          # as prepare_data serves it: (N, 8), label last
    t = head.assign_targets(torch.from_numpy(np.concatenate([gt, np.zeros((1, 8))])[None]).float(),
                            feature_map_size=[200, 200])
    hm = t['heatmaps'][0][0]
    assert [int(hm[c].eq(1).sum()) for c in range(3)] == [1, 2, 1]
    assert t['heatmap_masks'][0] is None
    assert int(t['masks'][0].sum()) == 4
