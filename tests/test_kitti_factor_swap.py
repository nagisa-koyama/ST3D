"""TRAIN_FACTOR_SWAP (pcdet/datasets/kitti/kitti_factor_swap.py): each arm changes exactly its factor, is fixed per frame,
and leaves everything else alone. The FACTOR-SWAP S2 rows of experiments_md 20261003_05 §5 item 24 rely on it. CPU only.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.kitti import kitti_factor_swap as kfs  # noqa: E402
from pcdet.utils.common_utils import IGNORE_CLASS_LABEL  # noqa: E402

NAMES = ['Car', 'Pedestrian', 'Cyclist']


def _scan(n_rings=64, per_ring=360, seed=0):
    """A raw KITTI-like scan in stored order: laser-major, each ring one full turn from azimuth ~0 upwards."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_rings):
        el = np.radians(2.0 - 0.4 * k)
        az = np.radians(np.sort(rng.uniform(0.0, 359.9, per_ring)))
        r = rng.uniform(5.0, 60.0, per_ring)
        rows.append(np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el),
                              rng.uniform(0, 1, per_ring)], axis=1))
    return np.concatenate(rows).astype(np.float32)


def test_exactly_one_arm():
    with pytest.raises(AssertionError):
        kfs.arm_of(EasyDict({}))
    with pytest.raises(AssertionError):
        kfs.arm_of(EasyDict({'PITCH_DEG': 1.0, 'RANGE_KEEP': {'EDGES': [0, 1], 'RATE': [1]}}))


def test_ring_thin_rows_halves_rings_and_random_matches_count():
    pts = _scan()
    rows = kfs.apply_points(pts, EasyDict({'RING_THIN': {'MODE': 'rows', 'STRIDE': 2}}), '000001')
    rnd = kfs.apply_points(pts, EasyDict({'RING_THIN': {'MODE': 'random', 'STRIDE': 2}}), '000001')
    assert len(rows) == len(pts) // 2
    assert len(rnd) == len(rows)
    # rows keeps whole rings: every kept elevation level is one of the even rings
    el = np.round(np.degrees(np.arctan2(rows[:, 2], np.hypot(rows[:, 0], rows[:, 1]))), 1)
    assert len(np.unique(el)) == 32
    # random keeps points of every ring
    el_r = np.round(np.degrees(np.arctan2(rnd[:, 2], np.hypot(rnd[:, 0], rnd[:, 1]))), 1)
    assert len(np.unique(el_r)) == 64


def test_cuts_are_fixed_per_frame():
    pts = _scan()
    cfg = EasyDict({'RING_THIN': {'MODE': 'random', 'STRIDE': 2}})
    a = kfs.apply_points(pts, cfg, '000007')
    b = kfs.apply_points(pts, cfg, '000007')
    c = kfs.apply_points(pts, cfg, '000008')
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_range_keep_rates_by_radius():
    pts = np.zeros((4000, 4), np.float32)
    pts[:2000, 0] = 5.0      # bin 0
    pts[2000:, 0] = 25.0     # bin 1
    cfg = EasyDict({'RANGE_KEEP': {'EDGES': [0, 10, 30, 75], 'RATE': [1.0, 0.25, 0.0]}})
    out = kfs.apply_points(pts, cfg, '000003')
    assert np.sum(out[:, 0] == 5.0) == 2000
    assert abs(np.sum(out[:, 0] == 25.0) / 2000 - 0.25) < 0.05
    far = np.zeros((10, 4), np.float32); far[:, 0] = 200.0     # beyond the last edge -> the last rate (0)
    assert len(kfs.apply_points(far, cfg, '000003')) == 0


def test_pitch_moves_points_and_box_centres_by_the_same_rotation():
    cfg = EasyDict({'PITCH_DEG': 1.0})
    pts = np.array([[10.0, 0.0, 0.0, 0.5], [-10.0, 2.0, 0.0, 0.5]], np.float32)
    out = kfs.apply_points(pts, cfg, '000001')
    assert out[0, 2] == pytest.approx(10 * np.sin(np.radians(1.0)), abs=1e-5)   # ahead rises
    assert out[1, 2] == pytest.approx(-10 * np.sin(np.radians(1.0)), abs=1e-5)  # behind falls
    assert out[1, 1] == pts[1, 1] and out[0, 3] == pts[0, 3]                    # y and intensity untouched
    boxes = np.array([[10.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.3]], np.float32)
    ob = kfs.apply_boxes(boxes, np.array(['Car']), cfg, '000001')
    assert np.allclose(ob[0, :3], out[0, :3], atol=1e-5)
    assert np.array_equal(ob[0, 3:], boxes[0, 3:])                               # upright, same size and yaw


def test_pitch_mixture_draws_one_angle_per_frame_for_points_and_boxes():
    cfg = EasyDict({'PITCH_DEG': [1.0, 3.0], 'PITCH_PROB': [0.6, 0.4]})
    pt = np.array([[10.0, 0.0, 0.0, 0.0]], np.float32)
    bx = np.array([[10.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0]], np.float32)
    angles = []
    for i in range(400):
        fid = '%06d' % i
        zp = kfs.apply_points(pt, cfg, fid)[0, 2]
        zb = kfs.apply_boxes(bx, np.array(['Car']), cfg, fid)[0, 2]
        assert zp == pytest.approx(zb, abs=1e-6)
        angles.append(np.degrees(np.arcsin(zp / 10.0)))
    angles = np.round(angles, 3)
    assert set(np.unique(angles)) == {1.0, 3.0}
    assert abs(np.mean(angles == 1.0) - 0.6) < 0.08


def test_gt_dim_scale_keeps_the_bottom_and_the_points():
    cfg = EasyDict({'GT_DIM_SCALE': {'Car': 1.2}})
    boxes = np.array([[10.0, 0.0, -0.9, 4.0, 1.8, 1.6, 0.0],
                      [5.0, 1.0, -0.8, 0.8, 0.6, 1.7, 0.0]], np.float32)
    names = np.array(['Car', 'Pedestrian'])
    ob = kfs.apply_boxes(boxes, names, cfg, '000001')
    assert np.allclose(ob[0, 3:6], boxes[0, 3:6] * 1.2)
    assert ob[0, 2] - ob[0, 5] / 2 == pytest.approx(boxes[0, 2] - boxes[0, 5] / 2, abs=1e-5)   # bottom fixed
    assert np.array_equal(ob[1], boxes[1])                                                       # other class
    pts = _scan(n_rings=4, per_ring=50)
    assert np.array_equal(kfs.apply_points(pts, cfg, '000001'), pts)


def test_gt_ignore_by_range_sets_ignore_label_only_on_far_cars():
    cfg = EasyDict({'GT_IGNORE_BY_RANGE': {'CLASS': 'Car', 'EDGES': [0, 30, 1000], 'KEEP': [1.0, 0.0]}})
    boxes = np.array([[10.0, 0, 0, 4, 2, 1.5, 0], [50.0, 0, 0, 4, 2, 1.5, 0],
                      [50.0, 0, 0, 0.8, 0.6, 1.7, 0], [50.0, 0, 0, 5, 2, 2, 0]], np.float32)
    names = np.array(['Car', 'Car', 'Pedestrian', 'Van'])
    cls, scores = kfs.ignore_labels(boxes, names, NAMES, cfg, '000001')
    assert cls.tolist() == [1, IGNORE_CLASS_LABEL, 2, IGNORE_CLASS_LABEL]
    assert scores.tolist() == [1.0] * 4


def test_gt_ignore_by_range_fraction_and_determinism():
    cfg = EasyDict({'GT_IGNORE_BY_RANGE': {'CLASS': 'Car', 'EDGES': [0, 1000], 'KEEP': [0.3]}})
    boxes = np.tile(np.array([[20.0, 0, 0, 4, 2, 1.5, 0]], np.float32), (2000, 1))
    names = np.array(['Car'] * 2000)
    a, _ = kfs.ignore_labels(boxes, names, NAMES, cfg, '000005')
    b, _ = kfs.ignore_labels(boxes, names, NAMES, cfg, '000005')
    assert np.array_equal(a, b)
    assert abs(np.mean(a == 1) - 0.3) < 0.04


def test_region_thin_foreground_only_thins_inside_boxes():
    rng = np.random.default_rng(1)
    inside = np.c_[rng.uniform(9, 11, (1000, 1)), rng.uniform(-0.5, 0.5, (1000, 1)), rng.uniform(-0.5, 0.5, (1000, 1)),
                   np.zeros((1000, 1))]
    outside = np.c_[rng.uniform(30, 40, (1000, 1)), rng.uniform(-5, 5, (1000, 2)), np.zeros((1000, 1))]
    pts = np.concatenate([inside, outside]).astype(np.float32)
    boxes = np.array([[10.0, 0.0, 0.0, 3.0, 2.0, 2.0, 0.0]], np.float32)
    cfg = EasyDict({'REGION_THIN': {'SIDE': 'foreground', 'KEEP': 0.5}, 'BOX_MARGIN': 0.25})
    out = kfs.region_thin(pts, boxes, '000001', cfg)
    assert np.sum(out[:, 0] > 20) == 1000
    assert np.sum(out[:, 0] < 20) == 500
