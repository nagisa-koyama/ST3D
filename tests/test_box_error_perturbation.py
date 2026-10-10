"""Sentinel tests for tools/analysis/oracle_box_error_sensitivity.py (experiments_md 20261010_11, ANALYSIS).

The perturbation touches `boxes_lidar` only. What it must never do: change a detection's score / class / count, move a
box that was not asked to move, turn an empty frame into a non-empty one, or return a heading outside [-pi, pi].
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'analysis'))
import oracle_box_error_sensitivity as m  # noqa: E402

ARMS = m.build_arms()


def boxes():
    return np.array([[10.0, 2.0, -1.0, 4.5, 1.9, 1.6, 0.0],
                     [-20.0, 5.0, -1.2, 4.0, 1.8, 1.5, np.pi / 2],
                     [30.0, -8.0, -0.9, 5.0, 2.0, 1.8, 3.0]], dtype=np.float32)


def test_base_and_zero_arms_are_identity():
    b = boxes()
    assert np.array_equal(m.perturb_boxes(b, 'base', {}, np.random.default_rng(0)), b)
    out = m.perturb_boxes(b, 'size', {'scale': (1.0, 1.0, 1.0)}, np.random.default_rng(0))
    assert np.allclose(out, b)
    assert np.allclose(m.perturb_boxes(b, 'yaw_bias', {'deg': 0.0}, np.random.default_rng(0))[:, 6], b[:, 6], atol=1e-6)


def test_empty_and_dtype():
    for kind, params in ARMS.values():
        out = m.perturb_boxes(np.zeros((0, 7), np.float32), kind, params, np.random.default_rng(0))
        assert out.shape == (0, 7)
    assert m.perturb_boxes(boxes(), 'size', {'scale': (1.1, 1.1, 1.1)}, np.random.default_rng(0)).dtype == np.float32


def test_size_scales_dims_and_keeps_centre_and_heading():
    b = boxes()
    out = m.perturb_boxes(b, 'size', {'scale': (1.1, 0.9, 1.0)}, np.random.default_rng(0))
    assert np.allclose(out[:, 3:6], b[:, 3:6] * np.array([1.1, 0.9, 1.0]), rtol=1e-5)
    assert np.array_equal(out[:, :3], b[:, :3]) and np.array_equal(out[:, 6], b[:, 6])


def test_yaw_bias_wraps_and_touches_only_heading():
    b = boxes()
    out = m.perturb_boxes(b, 'yaw_bias', {'deg': 20.0}, np.random.default_rng(0))
    assert np.all(np.abs(out[:, 6]) <= np.pi + 1e-6)
    d = np.arctan2(np.sin(out[:, 6] - b[:, 6]), np.cos(out[:, 6] - b[:, 6]))
    assert np.allclose(d, np.deg2rad(20.0), atol=1e-5)
    assert np.array_equal(out[:, :6], b[:, :6])


def test_position_bias_is_along_and_across_the_box_heading():
    b = boxes()
    lon = m.perturb_boxes(b, 'pos_bias', {'axis': 'lon', 'm': 0.5}, np.random.default_rng(0)).astype(np.float64)
    lat = m.perturb_boxes(b, 'pos_bias', {'axis': 'lat', 'm': 0.5}, np.random.default_rng(0)).astype(np.float64)
    z = m.perturb_boxes(b, 'pos_bias', {'axis': 'z', 'm': 0.5}, np.random.default_rng(0)).astype(np.float64)
    h = b[:, 6].astype(np.float64)
    fwd = np.stack([np.cos(h), np.sin(h)], 1)
    side = np.stack([-np.sin(h), np.cos(h)], 1)
    for out, axis_vec in ((lon, fwd), (lat, side)):
        shift = out[:, :2] - b[:, :2]
        assert np.allclose(np.linalg.norm(shift, axis=1), 0.5, atol=1e-5)
        assert np.allclose(shift, 0.5 * axis_vec, atol=1e-5)
        assert np.allclose(out[:, 2], b[:, 2], atol=1e-6)
    assert np.allclose(z[:, 2] - b[:, 2], 0.5, atol=1e-6) and np.allclose(z[:, :2], b[:, :2], atol=1e-6)
    with pytest.raises(ValueError):
        m.perturb_boxes(b, 'pos_bias', {'axis': 'up', 'm': 1.0}, np.random.default_rng(0))


def test_noise_is_zero_mean_seeded_and_confined_to_its_field():
    big = np.tile(boxes(), (4000, 1))
    out = m.perturb_boxes(big, 'yaw_noise', {'sigma_deg': 10.0}, np.random.default_rng(1)).astype(np.float64)
    d = np.arctan2(np.sin(out[:, 6] - big[:, 6]), np.cos(out[:, 6] - big[:, 6]))
    assert abs(d.mean()) < 0.01 and abs(d.std() - np.deg2rad(10.0)) < 0.01
    assert np.array_equal(out[:, :6].astype(np.float32), big[:, :6])
    a = m.perturb_boxes(big, 'size_noise', {'sigma': 0.1}, np.random.default_rng(7))
    b2 = m.perturb_boxes(big, 'size_noise', {'sigma': 0.1}, np.random.default_rng(7))
    assert np.array_equal(a, b2)
    assert (a[:, 3:6] > 0).all()
    pz = m.perturb_boxes(big, 'pos_noise', {'sigma_xy': 0.0, 'sigma_z': 0.2}, np.random.default_rng(2))
    assert np.array_equal(pz[:, :2], big[:, :2]) and not np.array_equal(pz[:, 2], big[:, 2])


def test_perturb_result_changes_only_boxes_and_keeps_empty_frames():
    dets = [{'frame_id': 'a', 'name': np.array(['Car', 'Car']), 'score': np.array([0.9, 0.2], np.float32),
             'boxes_lidar': boxes()[:2].copy(), 'pred_labels': np.array([1, 1])},
            {'frame_id': 'b', 'name': np.array([]), 'score': np.zeros(0, np.float32),
             'boxes_lidar': np.zeros((0, 7), np.float32), 'pred_labels': np.zeros(0, int)}]
    before = copy.deepcopy(dets)
    for kind, params in ARMS.values():
        out = m.perturb_result(dets, kind, params)
        assert [d['frame_id'] for d in out] == ['a', 'b']
        assert np.array_equal(out[0]['score'], before[0]['score']) and np.array_equal(out[0]['name'], before[0]['name'])
        assert out[1]['boxes_lidar'].shape == (0, 7)
    assert all(np.array_equal(d['boxes_lidar'], e['boxes_lidar']) for d, e in zip(dets, before))  # input untouched


def test_every_arm_is_named_and_unique():
    assert 'base' in ARMS and len(ARMS) == len(set(ARMS)) == 34
    assert {k for k, _ in ARMS.values()} == {'base', 'size', 'size_noise', 'yaw_bias', 'yaw_noise', 'pos_bias', 'pos_noise'}
