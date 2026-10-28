"""PL-MATCH (ANALYSIS; experiments_md 20261028_05; design T05 §5 item 23, fixed 2026-10-28 20:01, experiments_md 2e78a9c):
a row's own frozen pseudo-labels matched against the target's ground truth (MB: matched positives take their GT box;
MB-FP: plus unmatched positives removed; the ignore band kept in both).

Bug-prevention rule 1 (memory/repo/bug_prevention_checklist.md): unlike PL-GT, these files KEEP the pseudo-label
sentinels (the ignore band -1..-C, the teacher's scores), the exact path of the 2026-10-06 seven-month CenterHead bug.
Every sentinel goes through the new frozen-file path here: the matcher never matches a negative label, the validator
accepts only +-1..C, refuses 0, -99 and |label| > C, the loader serves a negative label as an ignore band (gt_classes
negative, counted as ign_ps_bbox) with a recounted point count, and CenterHead turns it into an ignored footprint,
never a positive of any class. The guard still refuses without the key; PL-GT's path and every other config are
unchanged. CPU only (CenterHead built without __init__, as in test_center_head_ignore_regions.py).
"""
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets import assert_target_labels_are_not_used  # noqa: E402
from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.models.dense_heads.center_head import CenterHead  # noqa: E402
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu  # noqa: E402
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils  # noqa: E402
from pcdet.utils import common_utils, self_training_utils as stu  # noqa: E402

NAMES = ['Car', 'Pedestrian', 'Cyclist']
KEY = common_utils.ANALYSIS_TARGET_PL_MATCH_KEY
FAMILY = Path(__file__).resolve().parent.parent / 'tools' / 'cfgs' / 'da-ieee-access'
TOOLS = FAMILY.parent.parent
BASE = FAMILY / 'centerpoint-gblobs-accum-legaldepth-st3d-nuscenes2waymo.yaml'          # 27704
ROWS = [FAMILY / 'centerpoint-gblobs-accum-legaldepth-plmatch-mb-nuscenes2waymo.yaml',
        FAMILY / 'centerpoint-gblobs-accum-legaldepth-plmatch-mbfp-nuscenes2waymo.yaml']


def iou(a, b):
    a = np.ascontiguousarray(np.asarray(a)[:, :7], dtype=np.float32)
    b = np.ascontiguousarray(np.asarray(b)[:, :7], dtype=np.float32)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return np.asarray(boxes_bev_iou_cpu(a, b), dtype=np.float64)


def _b(x, y, l, w, label, score, heading=0.0, z=1.0, h=1.6):
    return [x, y, z, l, w, h, heading, float(label), float(score)]


def _ps_entry():
    """One frame of a teacher pass: positives, an ignore band, a -1 Car sitting on a GT car."""
    rows = [
        _b(10.0, 0.0, 4.4, 1.8, 1, 0.90),            # 0 Car, overlaps GT car A (IoU high)
        _b(10.3, 0.1, 4.2, 1.9, 1, 0.95),            # 1 Car, overlaps GT car A too, HIGHER score: takes A
        _b(20.0, 5.0, 4.5, 1.9, 1, 0.50),            # 2 Car, overlaps GT car B
        _b(-30.0, 0.0, 4.5, 1.9, 1, 0.70),           # 3 Car, no GT anywhere near: a false positive
        _b(5.0, 3.0, 0.8, 0.8, 2, 0.60, h=1.7),      # 4 Ped, on GT ped
        _b(30.0, -8.0, 4.5, 1.9, -1, 0.15),          # 5 ignore-band Car ON GT car C: never matched
        _b(5.1, 3.1, 0.8, 0.8, -2, 0.25, h=1.7),     # 6 ignore-band Ped on the same GT ped
        _b(7.0, -7.0, 1.8, 0.6, -3, 0.20, h=1.7),    # 7 ignore-band Cyclist
        _b(40.0, 0.0, 4.4, 1.9, 3, 0.80, h=1.7),     # 8 "Cyclist" with a car-sized box on GT CAR D (class mismatch)
    ]
    b = np.array(rows, dtype=np.float64)
    n = len(b)
    return {'gt_boxes': b, 'cls_scores': b[:, 8].astype(np.float32), 'iou_scores': b[:, 8].astype(np.float32),
            'memory_counter': np.zeros(n), 'teacher_classes': np.array(
                ['Car', 'Car', 'Car', 'Car', 'Pedestrian', 'Ignored', 'Ignored', 'Ignored', 'Cyclist'], dtype='<U10')}


def _gt_entry():
    rows = [[10.2, 0.05, 1.0, 4.6, 1.95, 1.7, 0.02, 1, 1.0],     # A
            [20.1, 5.0, 1.0, 4.6, 2.0, 1.7, 0.0, 1, 1.0],        # B
            [30.0, -8.0, 1.0, 4.6, 2.0, 1.7, 0.0, 1, 1.0],       # C (only an ignore-band box on it)
            [40.0, 0.0, 1.0, 4.6, 2.0, 1.7, 0.0, 1, 1.0],        # D (only a cyclist on it)
            [5.05, 3.0, 1.0, 0.7, 0.75, 1.75, 0.3, 2, 1.0]]      # pedestrian
    g = np.array(rows, dtype=np.float64)
    return {'gt_boxes': g, 'num_points_in_gt': np.array([120, 40, 30, 15, 9], dtype=np.int64)}


# --- the matcher -------------------------------------------------------------------------------------------------

def test_greedy_matching_by_score_one_to_one_positives_only():
    ps, gt = _ps_entry(), _gt_entry()
    b, g = ps['gt_boxes'], gt['gt_boxes']
    m, mi = stu.pl_match_greedy(b, b[:, 7], b[:, 8], g, g[:, 7], 3, iou)
    # row 1 (score 0.95) takes A before row 0 (0.90); row 0 then has nothing left at >= 0.5
    assert m.tolist() == [-1, 0, 1, -1, 4, -1, -1, -1, -1]
    assert (mi[m >= 0] >= 0.5).all() and (mi[m < 0] == 0).all()
    assert iou(b[[0]], g[[0]])[0, 0] >= 0.5                       # row 0 WOULD match A: lost on score
    assert iou(b[[5]], g[[2]])[0, 0] >= 0.5                       # the -1 box sits on C and is still unmatched
    assert iou(b[[8]], g[[3]])[0, 0] >= 0.5                       # the cyclist sits on car D: class mismatch


@pytest.mark.parametrize('label', [-1, -2, -3, common_utils.IGNORE_CLASS_LABEL, 0])
def test_matcher_never_matches_a_non_positive_label(label):
    g = np.array([[0.0, 0.0, 1.0, 4.5, 1.9, 1.6, 0.0, 1, 1.0]])
    for c in (1, 2, 3):
        g[0, 7] = c
        b = np.array([[0.0, 0.0, 1.0, 4.5, 1.9, 1.6, 0.0, label, 0.9]])
        m, _ = stu.pl_match_greedy(b, b[:, 7], b[:, 8], g, g[:, 7], 3, iou)
        assert m.tolist() == [-1]


def test_iou_threshold_is_inclusive_at_half():
    g = np.array([[0.0, 0.0, 1.0, 4.0, 2.0, 1.6, 0.0, 1, 1.0]])
    b = np.array([[0.0, 0.0, 1.0, 4.0, 2.0, 1.6, 0.0, 1, 0.9]])
    fake = lambda a, bb: np.full((len(a), len(bb)), 0.5)        # noqa: E731
    assert stu.pl_match_greedy(b, b[:, 7], b[:, 8], g, g[:, 7], 3, fake)[0].tolist() == [0]
    fake = lambda a, bb: np.full((len(a), len(bb)), 0.4999)     # noqa: E731
    assert stu.pl_match_greedy(b, b[:, 7], b[:, 8], g, g[:, 7], 3, fake)[0].tolist() == [-1]


def test_mb_replaces_matched_geometry_only():
    ps, gt = _ps_entry(), _gt_entry()
    e, info = stu.pl_match_entry(ps, gt, 3, iou, 0.5, drop_unmatched_positives=False)
    b, g, o = ps['gt_boxes'], gt['gt_boxes'], e['gt_boxes']
    assert o.shape == b.shape
    for i, j in ((1, 0), (2, 1), (4, 4)):
        np.testing.assert_array_equal(o[i, :7], g[j, :7])        # centre, size, heading = the GT's
        assert o[i, 7] == b[i, 7] and o[i, 8] == b[i, 8]          # class and score kept
    for i in (0, 3, 5, 6, 7, 8):
        np.testing.assert_array_equal(o[i], b[i])                 # unmatched positives and the ignore band unchanged
    assert e['num_points_in_gt'].tolist() == [-1, 120, 40, -1, 9, -1, -1, -1, -1]
    assert e['match_gt_index'].tolist() == [-1, 0, 1, -1, 4, -1, -1, -1, -1]
    assert len(e['gt_boxes']) == 9 and 2 not in e['match_gt_index'] and 3 not in e['match_gt_index']  # GT not added
    np.testing.assert_array_equal(e['teacher_classes'], ps['teacher_classes'])
    counts = stu.validate_pl_match_pseudo_labels({'f': e}, NAMES)
    assert counts['positive'].tolist() == [4, 1, 1] and counts['ignore'].tolist() == [1, 1, 1]
    assert counts['gt_replaced'].tolist() == [2, 1, 0]


def test_mbfp_removes_unmatched_positives_and_keeps_the_ignore_band():
    ps, gt = _ps_entry(), _gt_entry()
    e, _ = stu.pl_match_entry(ps, gt, 3, iou, 0.5, drop_unmatched_positives=True)
    b, o = ps['gt_boxes'], e['gt_boxes']
    assert o[:, 7].tolist() == [1, 1, 2, -1, -2, -3]
    np.testing.assert_array_equal(o[3:], b[5:8])                  # ignore band byte-equal
    assert (e['match_gt_index'][o[:, 7] > 0] >= 0).all()          # no unmatched positive
    assert e['num_points_in_gt'].tolist() == [120, 40, 9, -1, -1, -1]
    assert len(e['cls_scores']) == len(e['teacher_classes']) == len(e['memory_counter']) == 6
    counts = stu.validate_pl_match_pseudo_labels({'f': e}, NAMES)
    assert counts['positive'].tolist() == [2, 1, 0] and counts['ignore'].tolist() == [1, 1, 1]


def test_empty_frames():
    ps = {'gt_boxes': np.zeros((0, 9)), 'cls_scores': np.zeros(0, np.float32), 'iou_scores': np.zeros(0, np.float32),
          'memory_counter': np.zeros(0), 'teacher_classes': np.zeros(0, '<U10')}
    for drop in (False, True):
        e, _ = stu.pl_match_entry(ps, _gt_entry(), 3, iou, 0.5, drop)
        assert e['gt_boxes'].shape == (0, 9)
        e, _ = stu.pl_match_entry(_ps_entry(), {'gt_boxes': np.zeros((0, 9)), 'num_points_in_gt': np.zeros(0, int)},
                                  3, iou, 0.5, drop)
        assert (e['match_gt_index'] == -1).all()
        assert len(e['gt_boxes']) == (9 if not drop else 3)
        stu.validate_pl_match_pseudo_labels({'f': e}, NAMES)


# --- the validator -----------------------------------------------------------------------------------------------

def _mb():
    return stu.pl_match_entry(_ps_entry(), _gt_entry(), 3, iou, 0.5, False)[0]


@pytest.mark.parametrize('label', [-1, -2, -3, 1, 2, 3])
def test_validator_accepts_the_positives_and_the_ignore_band(label):
    e = _mb()
    e['gt_boxes'][0, 7] = label          # row 0 is an unmatched (recount) box
    stu.validate_pl_match_pseudo_labels({'f': e}, NAMES)


@pytest.mark.parametrize('label', [0, common_utils.IGNORE_CLASS_LABEL, 4, -4, 1.5, -0.5, np.nan])
def test_validator_refuses_every_other_sentinel(label):
    e = _mb()
    e['gt_boxes'][0, 7] = label
    with pytest.raises(AssertionError):
        stu.validate_pl_match_pseudo_labels({'f': e}, NAMES)


@pytest.mark.parametrize('col, value', [(8, 0.0), (8, 1.5), (8, np.nan), (3, 0.0), (4, -1.0), (0, np.inf)])
def test_validator_refuses_bad_score_or_box(col, value):
    e = _mb()
    e['gt_boxes'][0, col] = value
    with pytest.raises(AssertionError):
        stu.validate_pl_match_pseudo_labels({'f': e}, NAMES)


def test_validator_refuses_bad_counts():
    e = _mb()
    e['num_points_in_gt'][5] = 30                     # a stored count on an ignore-band box
    e['match_gt_index'][5] = 2
    with pytest.raises(AssertionError):
        stu.validate_pl_match_pseudo_labels({'f': e}, NAMES)
    e = _mb()
    e['num_points_in_gt'][1] = 0                      # a stored count below MIN_POINTS_OF_GT
    with pytest.raises(AssertionError):
        stu.validate_pl_match_pseudo_labels({'f': e}, NAMES)
    e = _mb()
    e['num_points_in_gt'][0] = -2                     # neither the recount sentinel nor a count
    with pytest.raises(AssertionError):
        stu.validate_pl_match_pseudo_labels({'f': e}, NAMES)
    e = _mb()
    e['match_gt_index'][0] = 3                        # audit trail and counts disagree
    with pytest.raises(AssertionError):
        stu.validate_pl_match_pseudo_labels({'f': e}, NAMES)
    e = _mb()
    e['cls_scores'] = e['cls_scores'][:-1]            # misaligned field
    with pytest.raises(AssertionError):
        stu.validate_pl_match_pseudo_labels({'f': e}, NAMES)


def test_load_needs_the_wrapper_and_the_class_names(tmp_path):
    labels = {'seg_000': _mb()}
    p = tmp_path / 'plain.pkl'
    pickle.dump(labels, open(p, 'wb'))
    with pytest.raises(AssertionError, match='not a'):
        stu.load_analysis_pl_match_pseudo_labels(str(p), NAMES)
    q = tmp_path / 'plgt.pkl'                                     # a PL-GT wrapper is not a PL-MATCH file either
    pickle.dump({'format': stu.ANALYSIS_TARGET_GT_FORMAT, 'meta': {'class_names': NAMES}, 'labels': labels}, open(q, 'wb'))
    with pytest.raises(AssertionError, match='not a'):
        stu.load_analysis_pl_match_pseudo_labels(str(q), NAMES)
    w = tmp_path / 'wrapped.pkl'
    pickle.dump({'format': stu.ANALYSIS_PL_MATCH_FORMAT, 'meta': {'class_names': NAMES, 'row': 'MB'},
                 'labels': labels}, open(w, 'wb'))
    got, meta, counts = stu.load_analysis_pl_match_pseudo_labels(str(w), NAMES)
    assert list(got) == ['seg_000'] and counts['positive'].tolist() == [4, 1, 1]
    with pytest.raises(AssertionError):
        stu.load_analysis_pl_match_pseudo_labels(str(w), ['Car', 'Pedestrian'])
    with pytest.raises(AssertionError, match='not a'):                # and PL-GT's loader refuses this one
        stu.load_analysis_target_gt_pseudo_labels(str(w), NAMES)


def test_the_wrapper_fails_loudly_through_any_other_pseudo_label_path():
    wrapper = {'format': stu.ANALYSIS_PL_MATCH_FORMAT, 'meta': {}, 'labels': {'seg_000': _mb()}}
    with pytest.raises(ValueError, match='Cannot find pseudo label'):
        stu.load_ps_label('seg_000', wrapper)


def test_path_helper_is_none_without_the_key():
    assert stu.analysis_pl_match_pseudo_label_path(EasyDict({'DATA_CONFIG_TAR': {'USE_PSEUDO_LABEL': True}})) is None
    assert stu.analysis_pl_match_pseudo_label_path(EasyDict({})) is None
    assert stu.analysis_pl_match_pseudo_label_path(EasyDict({'DATA_CONFIG_TAR': {KEY: ''}})) is None
    assert stu.analysis_pl_match_pseudo_label_path(EasyDict({'DATA_CONFIG_TAR': {KEY: '/x.pkl'}})) == '/x.pkl'
    c = EasyDict({'DATA_CONFIG_TAR': {common_utils.ANALYSIS_TARGET_GT_KEY: '/x.pkl'}})
    assert stu.analysis_pl_match_pseudo_label_path(c) is None


# --- the loader: fill_pseudo_labels ---------------------------------------------------------------------------------

def _ds(key, labels):
    ds = DatasetTemplate.__new__(DatasetTemplate)
    cfg = {'USE_PSEUDO_LABEL': True}
    if key is not None:
        cfg[key] = '/x.pkl'
    ds.dataset_cfg = EasyDict(cfg)
    ds.class_names = list(NAMES)
    ds.pseudo_labels = labels
    return ds


def _cloud(seed=0):
    """Points filling every box of the test frame except the false-positive car at (-30, 0) and the GT ped."""
    rng = np.random.RandomState(seed)
    pts = []
    for x, y, n in ((10.2, 0.0, 50), (20.0, 5.0, 30), (30.0, -8.0, 20), (7.0, -7.0, 5), (40.0, 0.0, 12)):
        pts.append(np.c_[x + rng.uniform(-0.3, 0.3, n), y + rng.uniform(-0.2, 0.2, n), 1.0 + rng.uniform(-0.3, 0.3, n)])
    pts.append(rng.uniform(-60, 60, (200, 3)) * [1, 1, 0] + [0, 0, -1.5])   # ground, below every box
    return np.concatenate(pts).astype(np.float32)


def _recount(points, boxes):
    """What prepare_data computes when num_points_in_gt is absent (the plain pseudo-label path)."""
    return roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, :3]), torch.from_numpy(boxes[:, :7])).numpy().sum(axis=1)


def test_fill_serves_the_ignore_band_as_ignore_and_mixes_the_counts():
    e = _mb()
    pts = _cloud()
    d = {'frame_id': 'seg_000', 'points': pts, 'num_points_in_gt': np.arange(17)}   # the INFO's count, misaligned
    _ds(KEY, {'seg_000': e}).fill_pseudo_labels(d)
    lab = e['gt_boxes'][:, 7]
    np.testing.assert_array_equal(d['gt_classes'], lab)                 # negative stays negative
    assert d['gt_names'].tolist() == ['Car'] * 4 + ['Pedestrian', 'Car', 'Pedestrian', 'Cyclist', 'Cyclist']
    assert d['pos_ps_bbox'].tolist() == [4, 1, 1] and d['ign_ps_bbox'].tolist() == [1, 1, 1]
    rc = _recount(pts, e['gt_boxes'][:, :7])
    exp = np.where(e['num_points_in_gt'] == -1, rc, e['num_points_in_gt'])
    np.testing.assert_array_equal(d['num_points_in_gt'], exp)
    assert d['num_points_in_gt'][4] == 9 and rc[4] == 0                # the GT ped: empty cloud, stored count kept
    assert d['num_points_in_gt'][3] == 0                                # the false-positive car: recounted, empty


def test_fill_under_the_key_equals_the_plain_path_on_a_control_entry():
    """27704's own entry with every count = RECOUNT: the same served arrays as the plain path, and the count the plain
    path's prepare_data would compute."""
    ps = _ps_entry()
    ctl = dict(ps, num_points_in_gt=np.full(9, -1, np.int64), match_gt_index=np.full(9, -1, np.int64),
               match_bev_iou=np.zeros(9))
    pts = _cloud(1)
    a = {'frame_id': 'f', 'points': pts, 'num_points_in_gt': np.arange(5)}
    b = {'frame_id': 'f', 'points': pts, 'num_points_in_gt': np.arange(5)}
    _ds(None, {'f': ps}).fill_pseudo_labels(a)
    _ds(KEY, {'f': ctl}).fill_pseudo_labels(b)
    assert 'num_points_in_gt' not in a
    for k in ('gt_boxes', 'gt_names', 'gt_classes', 'gt_scores', 'pos_ps_bbox', 'ign_ps_bbox'):
        np.testing.assert_array_equal(a[k], b[k])
    np.testing.assert_array_equal(b['num_points_in_gt'], _recount(pts, a['gt_boxes']))


@pytest.mark.parametrize('key', [None, common_utils.ANALYSIS_TARGET_GT_KEY])
def test_other_paths_are_unchanged(key):
    """Without the key the count is popped; under PL-GT's key the STORED count is served (PL-GT unchanged)."""
    names_e = stu.target_gt_to_pseudo_label_entry(
        np.array(['Vehicle', 'Pedestrian']), np.array([[10, 0, 1, 4.5, 1.9, 1.6, 0.3], [5, 3, 1, 0.8, 0.8, 1.8, 0.0]]),
        np.array([7, 3]), {'Vehicle': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist', 'Sign': 'Misc'},
        NAMES)[0]
    d = {'frame_id': 'f', 'points': _cloud(), 'num_points_in_gt': np.arange(9)}
    _ds(key, {'f': names_e}).fill_pseudo_labels(d)
    if key is None:
        assert 'num_points_in_gt' not in d
    else:
        assert d['num_points_in_gt'].tolist() == [7, 3]


# --- CenterHead: what the served frame becomes ----------------------------------------------------------------------

def _head(num_max_objs=50):
    head = CenterHead.__new__(CenterHead)
    torch.nn.Module.__init__(head)
    head.model_cfg = EasyDict({'TARGET_ASSIGNER_CONFIG': {'FEATURE_MAP_STRIDE': 4, 'NUM_MAX_OBJS': num_max_objs,
                                                          'GAUSSIAN_OVERLAP': 0.1, 'MIN_RADIUS': 2}})
    head.class_names = list(NAMES)
    head.class_names_each_head = [list(NAMES)]
    head.point_cloud_range = np.array([-50.0, -50.0, -2.0, 50.0, 50.0, 4.0], dtype=np.float32)
    head.voxel_size = [0.1, 0.1, 0.15]
    return head


def _pix(x, y):
    """(row, col) of the heatmap pixel CenterHead puts a centre in: its own float32 arithmetic, not float64."""
    t = torch.tensor([x, y], dtype=torch.float32)
    cx = ((t[0] - -50.0) / 0.1 / 4).int().item()
    cy = ((t[1] - -50.0) / 0.1 / 4).int().item()
    return cy, cx


def test_center_head_turns_every_negative_label_into_ignore_never_a_positive():
    """The served MB frame (prepare_data's (N, 8): box + gt_classes) through CenterHead: peaks only at the positives,
    each in its own class; every ignore-band box zeroes its footprint in EVERY class channel; -1 (Car) never becomes a
    Cyclist peak (the 2026-10-06 bug), -3 (Cyclist) never a Car peak."""
    e = _mb()
    d = {'frame_id': 'f', 'points': _cloud(), 'num_points_in_gt': np.arange(9)}
    _ds(KEY, {'f': e}).fill_pseudo_labels(d)
    gt = np.concatenate([d['gt_boxes'], d['gt_classes'].reshape(-1, 1)], axis=1)
    t = _head().assign_targets(torch.from_numpy(np.concatenate([gt, np.zeros((2, 8))])[None]).float(),
                               feature_map_size=[250, 250])
    hm, mask = t['heatmaps'][0][0], t['heatmap_masks'][0][0]
    pos = gt[gt[:, 7] > 0]
    exp = [len({_pix(b[0], b[1]) for b in pos if b[7] == c}) for c in (1, 2, 3)]
    assert [int(hm[c].eq(1).sum()) for c in range(3)] == exp == [3, 1, 1]   # rows 0 and 1 share a pixel
    assert int(t['masks'][0].sum()) == len(pos) == 6
    for b in gt[gt[:, 7] < 0]:
        r, c = _pix(b[0], b[1])
        if hm[:, r, c].eq(1).any():          # the -2 Ped sits on the GT-replaced Ped: a positive's peak is never ignored
            assert mask[r, c] == 1
        else:                                # every other ignore-band centre is out of the loss
            assert mask[r, c] == 0
    r, c = _pix(30.0, -8.0)                                                # the -1 Car on GT car C
    assert hm[:, r, c].max() < 1 and mask[r, c] == 0
    r, c = _pix(7.0, -7.0)                                                 # the -3 Cyclist
    assert hm[0, r, c] < 1 and hm[2, r, c] < 1 and mask[r, c] == 0
    assert not (mask.eq(0) & hm.eq(1).any(0)).any()                        # no positive peak is ever ignored


@pytest.mark.parametrize('label', [common_utils.IGNORE_CLASS_LABEL, 0])
def test_center_head_other_sentinels_are_never_positives(label):
    """-99 is an ignore region and 0 is padding: neither ever makes a peak (they cannot come from a validated file)."""
    gt = np.array([[0.0, 0.0, 1.0, 4.5, 1.9, 1.6, 0.0, label]])
    t = _head().assign_targets(torch.from_numpy(gt[None]).float(), feature_map_size=[250, 250])
    assert int(t['heatmaps'][0][0].eq(1).sum()) == 0 and int(t['masks'][0].sum()) == 0


# --- the guard and the configs --------------------------------------------------------------------------------------

def _cfg(self_train=True, pseudo=True, unsupervised=False, value='/home/koyama/data/plmatch/x.pkl', plgt=None):
    tar = {}
    if pseudo:
        tar['USE_PSEUDO_LABEL'] = True
    if unsupervised:
        tar['UNSUPERVISED'] = True
    if value is not None:
        tar[KEY] = value
    if plgt is not None:
        tar[common_utils.ANALYSIS_TARGET_GT_KEY] = plgt
    c = {'DATA_CONFIG_TAR': tar}
    if self_train:
        c['SELF_TRAIN'] = {'TAR': {'LOSS_WEIGHT': 1.0}}
    return EasyDict(c)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *a):
        self.lines.append(('info', msg % a if a else msg))

    def warning(self, msg, *a):
        self.lines.append(('warning', msg % a if a else msg))


def test_key_is_accepted_and_says_analysis_row():
    log = _Log()
    assert_target_labels_are_not_used(_cfg(), True, log)
    assert len(log.lines) == 1 and log.lines[0][0] == 'warning'
    assert 'ANALYSIS ROW: target labels used to build the training targets' in log.lines[0][1]


@pytest.mark.parametrize('kwargs, err', [(dict(pseudo=False), 'USE_PSEUDO_LABEL'), (dict(self_train=False), 'SELF_TRAIN'),
                                         (dict(unsupervised=True), 'UNSUPERVISED'),
                                         (dict(plgt='/home/koyama/data/plgt/x.pkl'), 'both set')])
def test_the_key_does_not_switch_the_checks_off(kwargs, err):
    with pytest.raises(AssertionError, match=err):
        assert_target_labels_are_not_used(_cfg(**kwargs), True)


@pytest.mark.parametrize('value', [None, ''])
def test_without_the_key_the_guard_still_refuses(value):
    """Absent or empty, the key is no bypass: the old refusals fire (job 25933's shape; a trained target, no flag)."""
    with pytest.raises(AssertionError, match='USE_PSEUDO_LABEL'):
        assert_target_labels_are_not_used(_cfg(pseudo=False, value=value), True)
    with pytest.raises(AssertionError, match='UNSUPERVISED'):
        assert_target_labels_are_not_used(_cfg(self_train=False, pseudo=False, value=value), True)
    log = _Log()
    assert_target_labels_are_not_used(_cfg(value=value), True, log)
    assert log.lines == [('info', 'target label policy: pseudo-labels')]


def _resolved(path):
    from pcdet.config import cfg_from_yaml_file
    prev = os.getcwd()
    os.chdir(TOOLS)
    try:
        return cfg_from_yaml_file(str(path.relative_to(TOOLS)), EasyDict())
    finally:
        os.chdir(prev)


def _plain(d):
    if isinstance(d, dict):
        return {k: _plain(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_plain(v) for v in d]
    return d


def test_only_the_plmatch_configs_carry_the_key_raw_text():
    hits = sorted(p.relative_to(TOOLS / 'cfgs').as_posix() for p in (TOOLS / 'cfgs').rglob('*.yaml')
                  if KEY in p.read_text(encoding='utf-8', errors='ignore'))
    assert hits == sorted('da-ieee-access/' + p.name for p in ROWS), hits


def _self_training_configs():
    return [p for p in sorted(FAMILY.glob('centerpoint-*.yaml'))
            if re.search(r'^SELF_TRAIN:', p.read_text(encoding='utf-8'), re.M)]


@pytest.mark.parametrize('path', _self_training_configs(), ids=lambda p: p.stem)
def test_only_the_plmatch_configs_carry_the_key_resolved(path):
    has = bool(_resolved(path).DATA_CONFIG_TAR.get(KEY, None))
    assert has == (path in ROWS), path.name


@pytest.mark.parametrize('path', ROWS, ids=lambda p: p.stem)
def test_plmatch_config_is_27704s_config_plus_the_key(path):
    """The design's ONE change: the resolved row config equals 27704's resolved config except the key, and the key
    names the row's own frozen file under /home/koyama/data."""
    a, b = _plain(_resolved(path)), _plain(_resolved(BASE))
    a.pop('_BASE_CONFIG_', None)
    b.pop('_BASE_CONFIG_', None)
    key = a['DATA_CONFIG_TAR'].pop(KEY)
    row = 'mbfp' if 'mbfp' in path.name else 'mb'
    assert key == '/home/koyama/data/plmatch_20261028/plmatch_%s_27704e0_waymo_train_si10.pkl' % row
    assert a == b
    assert common_utils.ANALYSIS_TARGET_GT_KEY not in a['DATA_CONFIG_TAR']
    assert_target_labels_are_not_used(_resolved(path), True)          # the guard accepts the real config
