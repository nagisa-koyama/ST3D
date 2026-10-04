"""LEVEL_COOR geometry (pcdet/utils/level_utils.py).

Pins: (1) levelling a tilted sensor makes the vehicle's ground plane flat in the output frame;
(2) PITCH_DEG / ROLL_DEG re-tilt it with the documented sign (ground RISES to the front / left);
(3) the box transform is exactly inverted by the transpose, so predictions made in the levelled
frame return to the infos' frame; (4) loaders that do not implement it refuse the key.
"""
import numpy as np
import pytest

from pcdet.utils import level_utils


def _rot(axis, deg):
    return level_utils._rodrigues(np.asarray(axis, float), np.radians(deg))


def _ground_in_sensor(sensor_from_vehicle, t_sensor_from_vehicle, n=4000, seed=0):
    """Points on the vehicle's z=0 ground plane, 5-40 m around, expressed in the sensor frame."""
    rng = np.random.default_rng(seed)
    r, a = rng.uniform(5, 40, n), rng.uniform(-np.pi, np.pi, n)
    g_vehicle = np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n)], 1)
    return g_vehicle @ sensor_from_vehicle.T + t_sensor_from_vehicle


def _slopes(points, fwd, left):
    A = np.c_[points[:, 0], points[:, 1], np.ones(len(points))]
    a, b, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)[0]
    g = np.array([a, b, 0.0])
    return np.degrees(np.arctan(g @ fwd)), np.degrees(np.arctan(g @ left))


@pytest.mark.parametrize('yaw', [-90.0, 177.0, 0.0])
def test_levelling_flattens_the_vehicle_ground(yaw):
    # nuScenes-like sensor: yawed, pitched 2.65 deg about the vehicle's lateral axis, rolled 0.3.
    S = (_rot([0, 0, 1], yaw) @ _rot([0, 1, 0], 2.65) @ _rot([1, 0, 0], 0.3)).T  # vehicle -> sensor
    pts = _ground_in_sensor(S, np.array([0.0, -0.9, -1.84]))
    R = level_utils.level_rotation(S)
    lev = pts @ R.T
    assert np.std(lev[:, 2]) < 1e-6
    # heading convention kept: the vehicle's forward axis stays where it was in azimuth
    f0, f1 = S[:, 0], R @ S[:, 0]
    assert abs(np.degrees(np.arctan2(f1[1], f1[0]) - np.arctan2(f0[1], f0[0]))) < 0.2


@pytest.mark.parametrize('pitch,roll', [(-1.3, 0.0), (2.0, 0.0), (0.0, 1.0), (1.5, -0.7)])
def test_retilt_sign_convention(pitch, roll):
    S = (_rot([0, 0, 1], -90.0) @ _rot([1, 0, 0], -2.65)).T
    pts = _ground_in_sensor(S, np.array([0.0, -0.9, -1.84]))
    R = level_utils.level_rotation(S, pitch_deg=pitch, roll_deg=roll)
    lev = pts @ R.T
    fwd, left = R @ S[:, 0], R @ S[:, 1]
    sf, sl = _slopes(lev, fwd, left)
    assert sf == pytest.approx(pitch, abs=0.02)
    assert sl == pytest.approx(roll, abs=0.02)


def test_box_round_trip_and_upright():
    rng = np.random.default_rng(1)
    boxes = np.c_[rng.uniform(-50, 50, (20, 2)), rng.uniform(-2, 0, 20), rng.uniform(1, 5, (20, 3)),
                  rng.uniform(-np.pi, np.pi, 20), rng.uniform(-5, 5, (20, 2))].astype(np.float32)
    S = (_rot([0, 0, 1], 177.0) @ _rot([0, 1, 0], -1.3)).T
    R = level_utils.level_rotation(S, pitch_deg=-1.3)
    there = level_utils.rotate_boxes(boxes, R)
    back = level_utils.rotate_boxes(there, R.T)
    assert np.allclose(back[:, :6], boxes[:, :6], atol=1e-4)
    # Heading is re-read from the rotated heading vector's xy part, which drops a second-order
    # term: at this 2.6 deg combined tilt the round trip is off by ~1e-3 rad (2 mm on a car).
    assert np.allclose(np.angle(np.exp(1j * (back[:, 6] - boxes[:, 6]))), 0, atol=2e-3)
    assert np.allclose(back[:, 7:9], boxes[:, 7:9], rtol=3e-3, atol=1e-3)  # same 2nd-order term
    assert np.allclose(there[:, 3:6], boxes[:, 3:6])  # dimensions untouched


def test_identity_when_nothing_to_do():
    assert np.allclose(level_utils.level_rotation(None), np.eye(3))
    assert np.allclose(level_utils.level_rotation(np.eye(3)), np.eye(3))


def test_unsupported_loader_refuses_the_key():
    from easydict import EasyDict
    from pcdet.datasets.dataset import DatasetTemplate

    class _Logger:
        def info(self, *a, **k):
            pass

    cfg = EasyDict({'LEVEL_COOR': {'PITCH_DEG': 0.0}, 'DATA_PATH': '/tmp', 'POINT_CLOUD_RANGE': [-75.2, -75.2, -2, 75.2, 75.2, 4],
                    'POINT_FEATURE_ENCODING': {'encoding_type': 'absolute_coordinates_encoding',
                                               'used_feature_list': ['x', 'y', 'z'],
                                               'src_feature_list': ['x', 'y', 'z']},
                    'DATA_PROCESSOR': [], 'DATA_AUGMENTOR': {'DISABLE_AUG_LIST': [], 'AUG_CONFIG_LIST': []}})
    with pytest.raises(AssertionError, match='does not apply it'):
        DatasetTemplate(dataset_cfg=cfg, class_names=['Car'], training=False, logger=_Logger())
