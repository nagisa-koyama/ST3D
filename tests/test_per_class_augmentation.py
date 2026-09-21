"""Tests for per-class ROS and SN parameters.

`random_object_scaling` (ROS) and `normalize_object_size` (SN) were class-blind: one interval and
one offset applied to every object. That is wrong in direction, not only magnitude - measured
against KITTI, cars need roughly -0.9 m of length while pedestrians need about +0.1, so a
car-derived offset moves pedestrians the wrong way and can drive their dimensions negative.

Both now accept either the original single setting or a per-class dict. See
experiments_md/20260922_03_deriving_geometry_alignment_parameters.md for the measured values.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.augmentor import augmentor_utils  # noqa: E402


def _scene():
    """Two well-separated boxes - one Car, one Pedestrian - each with points inside."""
    boxes = np.array([
        [0.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0],      # Car
        [20.0, 0.0, 0.0, 0.8, 0.8, 1.8, 0.0],     # Pedestrian
    ])
    rng = np.random.RandomState(0)
    pts = []
    for box in boxes:
        local = (rng.rand(40, 3) - 0.5) * box[3:6] * 0.5
        pts.append(local + box[:3])
    return boxes, np.concatenate(pts).astype(np.float64), np.array(['Car', 'Pedestrian'])


def test_resolve_broadcasts_a_plain_list():
    out = augmentor_utils.resolve_per_object_param([0.9, 1.1], ['Car', 'Pedestrian'], 2)
    assert out.shape == (2, 2)
    assert np.allclose(out, [[0.9, 1.1], [0.9, 1.1]])


def test_resolve_looks_up_per_class():
    param = {'Car': [0.75, 1.0], 'Pedestrian': [0.85, 1.15]}
    out = augmentor_utils.resolve_per_object_param(param, ['Pedestrian', 'Car'], 2)
    assert np.allclose(out, [[0.85, 1.15], [0.75, 1.0]])


def test_resolve_accepts_head_per_dataset_prefixes():
    """head-per-dataset configs carry names like 'nuscenes:car'."""
    out = augmentor_utils.resolve_per_object_param({'car': [0.75, 1.0]}, ['nuscenes:car'], 2)
    assert np.allclose(out, [[0.75, 1.0]])


def test_resolve_falls_back_to_DEFAULT():
    param = {'Car': [0.75, 1.0], 'DEFAULT': [0.95, 1.05]}
    out = augmentor_utils.resolve_per_object_param(param, ['Car', 'Tram'], 2)
    assert np.allclose(out, [[0.75, 1.0], [0.95, 1.05]])


def test_resolve_marks_unlisted_classes_nan():
    out = augmentor_utils.resolve_per_object_param({'Car': [0.75, 1.0]}, ['Car', 'Tram'], 2)
    assert np.allclose(out[0], [0.75, 1.0])
    assert np.isnan(out[1]).all()


def test_sn_applies_a_different_offset_per_class():
    boxes, points, names = _scene()
    size_res = {'Car': [-0.9, -0.4, -0.2], 'Pedestrian': [0.1, 0.0, 0.05]}
    _, out = augmentor_utils.normalize_object_size(
        boxes.copy(), points.copy(), np.ones(2, dtype=bool), size_res, gt_names=names)
    assert np.allclose(out[0, 3:6], [4.0 - 0.9, 2.0 - 0.4, 1.6 - 0.2])
    assert np.allclose(out[1, 3:6], [0.8 + 0.1, 0.8, 1.8 + 0.05])


def test_sn_leaves_unlisted_classes_untouched():
    boxes, points, names = _scene()
    _, out = augmentor_utils.normalize_object_size(
        boxes.copy(), points.copy(), np.ones(2, dtype=bool), {'Car': [-0.9, -0.4, -0.2]},
        gt_names=names)
    assert np.allclose(out[0, 3:6], [3.1, 1.6, 1.4])
    assert np.allclose(out[1, 3:6], boxes[1, 3:6])          # Pedestrian unchanged


def test_sn_is_unchanged_for_a_plain_list():
    """The original class-blind form must behave exactly as before."""
    boxes, points, names = _scene()
    _, with_names = augmentor_utils.normalize_object_size(
        boxes.copy(), points.copy(), np.ones(2, dtype=bool), [-0.2, -0.1, -0.1], gt_names=names)
    _, without = augmentor_utils.normalize_object_size(
        boxes.copy(), points.copy(), np.ones(2, dtype=bool), [-0.2, -0.1, -0.1])
    assert np.allclose(with_names, without)
    assert np.allclose(with_names[0, 3:6], [3.8, 1.9, 1.5])
    assert np.allclose(with_names[1, 3:6], [0.6, 0.7, 1.7])


def test_ros_draws_from_a_different_interval_per_class():
    """Degenerate intervals make the draw deterministic, so the scale is checkable exactly."""
    boxes, points, names = _scene()
    perturb = {'Car': [2.0, 2.0], 'Pedestrian': [0.5, 0.5]}
    _, out = augmentor_utils.scale_pre_object(
        boxes.copy(), points.copy(), np.ones(2, dtype=bool), perturb, gt_names=names)
    assert np.allclose(out[0, 3:6], boxes[0, 3:6] * 2.0)
    assert np.allclose(out[1, 3:6], boxes[1, 3:6] * 0.5)


def test_ros_leaves_unlisted_classes_untouched():
    boxes, points, names = _scene()
    _, out = augmentor_utils.scale_pre_object(
        boxes.copy(), points.copy(), np.ones(2, dtype=bool), {'Car': [2.0, 2.0]}, gt_names=names)
    assert np.allclose(out[0, 3:6], boxes[0, 3:6] * 2.0)
    assert np.allclose(out[1, 3:6], boxes[1, 3:6])


def test_ros_is_unchanged_for_a_plain_list():
    boxes, points, names = _scene()
    _, out = augmentor_utils.scale_pre_object(
        boxes.copy(), points.copy(), np.ones(2, dtype=bool), [0.5, 0.5], gt_names=names)
    assert np.allclose(out[0, 3:6], boxes[0, 3:6] * 0.5)
    assert np.allclose(out[1, 3:6], boxes[1, 3:6] * 0.5)


def test_per_class_dict_without_gt_names_raises():
    with pytest.raises(AssertionError):
        augmentor_utils.resolve_per_object_param({'Car': [0.75, 1.0]}, None, 2)
