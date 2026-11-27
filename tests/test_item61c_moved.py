"""Item 61c, the label-content control (tools/analysis/item61c_moved.py; experiments_md 20261003_05 §5 item 61c, design
fixed 2026-11-27 15:47 JST): 59b's pseudo-labels rotated about the sensor by random angles. CPU only; run from the repo
root."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item61c_moved as m  # noqa: E402

XY = (-75.2, -75.2, 75.2, 75.2)


def ground(n=200000, r=70.0, seed=0):
    """A flat ground of points (z in [0, 0.1]) over a disc, the loader frame's shape."""
    g = np.random.default_rng(seed)
    a, rr = g.uniform(-np.pi, np.pi, n), r * np.sqrt(g.uniform(0, 1, n))
    return np.stack([rr * np.cos(a), rr * np.sin(a), g.uniform(0.0, 0.1, n)], 1)


def boxes():
    # x, y, z, dx, dy, dz, heading, label, score: cars at 8, 25, 35, 55 m, a pedestrian, a cyclist
    return np.array([[8.0, 1.0, 0.75, 4.2, 1.8, 1.6, 0.3, 1, 0.9],
                     [-20.0, 15.0, 0.75, 4.5, 1.9, 1.5, -1.2, 1, 0.5],
                     [30.0, -18.0, 0.75, 4.0, 1.7, 1.5, 2.0, 1, 0.2],
                     [40.0, 38.0, 0.75, 4.4, 1.8, 1.6, 0.0, 1, 0.1],
                     [12.0, -6.0, 0.85, 0.7, 0.7, 1.7, 0.5, 2, 0.3],
                     [-5.0, -22.0, 0.80, 1.8, 0.6, 1.7, -2.5, 3, 0.2]], dtype=np.float32)


def inside_independent(p, b):
    """Point-in-box by a rotation matrix (a different formula from m.points_inside)."""
    R = np.array([[np.cos(b[6]), -np.sin(b[6])], [np.sin(b[6]), np.cos(b[6])]])
    loc = (p[:, :2] - b[:2]) @ R                     # rows times R = R^T applied to each point
    return (np.abs(loc[:, 0]) < b[3] / 2) & (np.abs(loc[:, 1]) < b[4] / 2) & (np.abs(p[:, 2] - b[2]) < b[5] / 2)


def test_move_keeps_range_size_class_score_and_rotates_heading_with_the_centre():
    b, pts = boxes(), ground()
    out, keep, draws = m.move_frame(b, pts, np.random.default_rng([666, 7]), XY)
    assert keep.all() and len(out) == len(b) and (draws >= 1).all()
    src = b.astype(np.float64)
    np.testing.assert_allclose(np.hypot(out[:, 0], out[:, 1]), np.hypot(src[:, 0], src[:, 1]), atol=1e-9)
    np.testing.assert_array_equal(out[:, [2, 3, 4, 5, 7, 8]], src[:, [2, 3, 4, 5, 7, 8]])
    dphi = m.wrap(np.arctan2(out[:, 1], out[:, 0]) - np.arctan2(src[:, 1], src[:, 0]))
    np.testing.assert_allclose(m.wrap(out[:, 6] - src[:, 6] - dphi), 0.0, atol=1e-9)
    assert np.all(np.abs(dphi) > 1e-3)                      # every box actually moved


def test_every_kept_box_holds_a_point_and_clears_every_circle():
    b, pts = boxes(), ground()
    out, keep, _ = m.move_frame(b, pts, np.random.default_rng([667, 3]), XY)
    for k in out:
        assert inside_independent(pts, k).sum() >= 1
    half_o = 0.5 * np.hypot(b[:, 3], b[:, 4])
    half_m = 0.5 * np.hypot(out[:, 3], out[:, 4])
    for i, k in enumerate(out):
        d = np.hypot(b[:, 0] - k[0], b[:, 1] - k[1])
        assert np.all(d >= half_o + half_m[i])                       # clear of every original, its own included
        for j in range(i):
            assert np.hypot(*(out[j, :2] - k[:2])) >= half_m[i] + half_m[j]
    assert np.all((out[:, 0] > XY[0]) & (out[:, 0] < XY[2]) & (out[:, 1] > XY[1]) & (out[:, 1] < XY[3]))


def test_deterministic_and_seed_dependent():
    b, pts = boxes(), ground()
    o1, _, _ = m.move_frame(b, pts, np.random.default_rng([666, 5]), XY)
    o2, _, _ = m.move_frame(b, pts, np.random.default_rng([666, 5]), XY)
    o3, _, _ = m.move_frame(b, pts, np.random.default_rng([667, 5]), XY)
    np.testing.assert_array_equal(o1, o2)
    assert not np.allclose(o1[:, :2], o3[:, :2])
    # the first draws are shared, so a box placed within 100 draws is placed identically at 1,000 (the fallback rule)
    o4, _, _ = m.move_frame(b, pts, np.random.default_rng([666, 5]), XY, tries=1000)
    np.testing.assert_array_equal(o1, o4)


def test_a_box_with_nowhere_to_go_is_dropped_and_the_rest_kept():
    b = boxes()[:2].copy()
    pts = np.concatenate([ground(50000, seed=1)[:, :2], np.full((50000, 1), 5.0)], 1)   # nothing at box height ...
    inbox = np.array([[b[0, 0], b[0, 1], 0.75]])                                         # ... except inside box 0
    pts = np.concatenate([pts, inbox])
    out, keep, draws = m.move_frame(b, pts, np.random.default_rng([1, 1]), XY, tries=50)
    assert not keep.any() and len(out) == 0 and (draws == 50).all()
    ok = np.concatenate([ground(seed=2)])
    out, keep, _ = m.move_frame(b, ok, np.random.default_rng([1, 1]), XY, tries=50)
    assert keep.all()


def test_the_range_bound_is_respected():
    b = np.array([[70.0, 70.0, 0.75, 4.0, 1.8, 1.5, 0.0, 1, 0.5]], dtype=np.float32)   # r = 99 m: only corners fit
    out, keep, _ = m.move_frame(b, ground(400000, r=100.0), np.random.default_rng([3, 3]), XY, tries=1000)
    if keep.any():
        assert np.all(np.abs(out[:, :2]) < 75.2)


@pytest.mark.parametrize('bad', [0, -1, -3, -99, 1.5])
def test_non_positive_or_fractional_labels_are_refused(bad):
    b = boxes()
    b[2, 7] = bad
    with pytest.raises(ValueError):
        m.move_frame(b, ground(1000), np.random.default_rng(0), XY)


def test_empty_frame():
    out, keep, draws = m.move_frame(np.zeros((0, 9), np.float32), ground(10), np.random.default_rng(0), XY)
    assert out.shape == (0, 9) and keep.shape == (0,) and draws.shape == (0,)


def test_pl_entry_aligns_the_side_arrays_and_recounts_everything():
    from pcdet.utils import self_training_utils as stu
    b = boxes()
    e = {'gt_boxes': b, 'cls_scores': b[:, 8].copy(), 'iou_scores': b[:, 8].copy(), 'memory_counter': np.zeros(6),
         'teacher_classes': np.array(['Car'] * 4 + ['Pedestrian', 'Cyclist'])}
    out, keep, _ = m.move_frame(b, ground(), np.random.default_rng([0, 0]), XY)
    keep[1] = False                                                     # pretend box 1 was dropped
    out = out[[0, 2, 3, 4, 5]]
    ent = m.pl_entry(e, out, keep)
    assert ent['gt_boxes'].dtype == b.dtype and len(ent['gt_boxes']) == 5
    np.testing.assert_array_equal(ent['cls_scores'], b[keep, 8])
    np.testing.assert_array_equal(ent['teacher_classes'], e['teacher_classes'][keep])
    assert (ent['num_points_in_gt'] == stu.PL_MATCH_RECOUNT).all() and ent['num_points_in_gt'].dtype == np.int64
    counts = stu.validate_pl_match_pseudo_labels({'000001': ent, '000002': m.pl_entry(e)}, m.NAMES)
    assert counts['positive'].tolist() == [3 + 4, 1 + 1, 1 + 1] and counts['gt_replaced'].sum() == 0
    with pytest.raises(AssertionError):
        m.pl_entry(e, out[:4], keep)                                    # boxes and keep disagree


def test_the_row_config_is_59b_b_bn_plus_only_the_frozen_file():
    import os
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.utils import common_utils
    key = common_utils.ANALYSIS_TARGET_PL_MATCH_KEY
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        for s in (666, 667):
            a, b = EasyDict(), EasyDict()
            cfg_from_yaml_file(m.BBN_CFG % s, a)
            cfg_from_yaml_file(m.ROW_CFG % s, b)
            assert b.DATA_CONFIG_TAR[key] == str(m.out_path('move', s))
            assert key not in a.DATA_CONFIG_TAR
            bb = copy.deepcopy(b)
            del bb.DATA_CONFIG_TAR[key]
            for c in (a, bb):
                c.pop('_BASE_CONFIG_', None)
            assert bb == a
            assert b.OPTIMIZATION.TRAIN_ONLY_BN and not b.SELF_TRAIN.SRC.USE_DATA and b.OPTIMIZATION.NUM_EPOCHS == 10
    finally:
        os.chdir(cwd)


def test_paths_and_words():
    assert str(m.out_path('move', 666)).endswith('item61/moved/plmove_59bbn_s666.pkl')
    assert m.PS_E0[666].endswith('qv08jln7/files/ps_label/ps_label_e0.pkl')
    assert m.PS_E0[667].endswith('q3z6o5zb/files/ps_label/ps_label_e0.pkl')
    assert m.TRIES == (100, 1000) and m.DROP_MAX == 0.02 and m.MIN_POINTS == 1 and m.EPOCH == 10
    assert m.parse_jobs(['moved666=1', 'moved667=2']) == {666: 1, 667: 2}
    assert set(m.WORDS) == {'a normalization-level correction can close most of the far residual', 'it cannot', 'partly'}
