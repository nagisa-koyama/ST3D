"""Per-object motion compensation: the rigid box-to-box transform and its guards.

Ego compensation alone leaves a moving object smeared, so the points it contributed in earlier
sweeps fall outside its box in the anchor frame. Over 30 frames that costs a moving object the
whole benefit of accumulating - x1.01 against a static object's x2.74.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcdet.datasets.motion_compensation import (  # noqa: E402
    boxes_to_frame, interpolate_boxes, move_points_between_boxes)


def box(x, y, z=0.0, dx=4.0, dy=2.0, dz=2.0, yaw=0.0):
    return np.array([x, y, z, dx, dy, dz, yaw], dtype=np.float64)


def test_a_point_inside_a_moving_box_follows_it():
    pts = np.array([[10.0, 0.0, 0.0, 0.3]])
    out = move_points_between_boxes(pts, {'a': box(10, 0)}, {'a': box(25, 0)})
    assert out[0][0] == pytest.approx(25.0)
    assert out[0][3] == pytest.approx(0.3), 'intensity must be carried through untouched'


def test_a_point_outside_every_box_is_left_alone():
    pts = np.array([[10.0, 50.0, 0.0, 0.0]])
    out = move_points_between_boxes(pts, {'a': box(10, 0)}, {'a': box(25, 0)})
    assert out[0][1] == pytest.approx(50.0)


def test_rotation_is_applied_about_the_box_centre():
    """A box that turns 90 degrees takes its points around with it."""
    pts = np.array([[11.0, 0.0, 0.0, 0.0]])          # 1 m ahead of the centre
    out = move_points_between_boxes(pts, {'a': box(10, 0, yaw=0.0)},
                                    {'a': box(10, 0, yaw=np.pi / 2)})
    assert out[0][0] == pytest.approx(10.0, abs=1e-9)
    assert out[0][1] == pytest.approx(1.0, abs=1e-9)


def test_a_track_missing_from_the_anchor_frame_is_skipped():
    pts = np.array([[10.0, 0.0, 0.0, 0.0]])
    out = move_points_between_boxes(pts, {'gone': box(10, 0)}, {'other': box(25, 0)})
    assert out[0][0] == pytest.approx(10.0)


def test_a_point_is_claimed_by_at_most_one_track():
    """Overlapping boxes must not move the same point twice."""
    pts = np.array([[10.0, 0.0, 0.0, 0.0]])
    then = {'a': box(10, 0), 'b': box(10.5, 0)}
    now = {'a': box(30, 0), 'b': box(60, 0)}
    out = move_points_between_boxes(pts, then, now)
    assert out[0][0] in (pytest.approx(30.0), pytest.approx(60.0))
    assert out[0][0] != pytest.approx(80.0), 'moved twice'


def test_containment_is_tested_before_anything_moves():
    """Otherwise a box moved into another box's path would sweep up its points."""
    pts = np.array([[10.0, 0.0, 0.0, 0.0], [30.0, 0.0, 0.0, 0.0]])
    out = move_points_between_boxes(pts, {'a': box(10, 0), 'b': box(30, 0)},
                                    {'a': box(30, 0), 'b': box(80, 0)})
    assert sorted(np.round(out[:, 0], 6)) == [30.0, 80.0]


def test_the_input_array_is_not_modified():
    pts = np.array([[10.0, 0.0, 0.0, 0.0]])
    move_points_between_boxes(pts, {'a': box(10, 0)}, {'a': box(25, 0)})
    assert pts[0][0] == pytest.approx(10.0)


def test_interpolation_is_linear_in_position():
    mid = interpolate_boxes({'a': box(0, 0)}, {'a': box(10, 0)}, 0.5)
    assert mid['a'][0] == pytest.approx(5.0)


def test_interpolation_takes_the_shortest_angular_path():
    """Across the +-pi wrap the box must not spin the long way round."""
    mid = interpolate_boxes({'a': box(0, 0, yaw=3.0)}, {'a': box(0, 0, yaw=-3.0)}, 0.5)
    assert abs(abs(mid['a'][6]) - np.pi) < 0.15, 'interpolated %.3f' % mid['a'][6]


def test_a_track_in_only_one_keyframe_is_dropped():
    """It has no defined position in between, and inventing one sweeps up stray points."""
    assert interpolate_boxes({'a': box(0, 0)}, {'b': box(1, 0)}, 0.5) == {}


def test_boxes_to_frame_is_identity_between_a_frame_and_itself():
    S = np.eye(4); S[:3, 3] = [5.0, -2.0, 1.0]
    got = boxes_to_frame(np.array([box(3, 4, yaw=0.5)]), np.array(['car']), ['t0'], S, S)
    assert np.allclose(got['t0'], box(3, 4, yaw=0.5))


def test_boxes_to_frame_applies_the_relative_transform():
    """Two ego frames 10 m apart put the same global box 10 m differently."""
    A = np.eye(4)
    B = np.eye(4); B[0, 3] = -10.0          # ego B sits 10 m further along x in global
    got = boxes_to_frame(np.array([box(0, 0)]), np.array(['car']), ['t0'], A, B)
    assert got['t0'][0] == pytest.approx(-10.0)


def test_classes_filter_excludes_untracked_categories():
    got = boxes_to_frame(np.array([box(0, 0), box(5, 0)]), np.array(['car', 'barrier']),
                         ['t0', 't1'], np.eye(4), np.eye(4), classes={'car'})
    assert list(got) == ['t0']


def test_boxes_without_a_track_id_are_skipped():
    got = boxes_to_frame(np.array([box(0, 0)]), np.array(['car']), [None], np.eye(4), np.eye(4))
    assert got == {}


# --------------------------------------------------------------------------------------------
# The key must be refused where it cannot be honoured. A config key a loader silently ignores is
# worse than one that fails: the run looks like it did what was asked and is quietly a different
# experiment. This project has paid for that failure mode more than once - a correction configured
# but never installed, an N=15 that was really N=10.
# --------------------------------------------------------------------------------------------

from pcdet.datasets.motion_compensation import assert_not_supported  # noqa: E402


class _Cfg(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)


def test_unsupported_dataset_raises_rather_than_ignoring_the_key():
    with pytest.raises(NotImplementedError) as e:
        assert_not_supported(_Cfg(GT_BOXES_MOTION_COMPENSATION=True), 'PandasetDataset')
    msg = str(e.value)
    assert 'PandasetDataset' in msg
    assert 'NuScenesDataset' in msg and 'LyftDataset' in msg, 'must say what DOES support it'


def test_the_message_says_why_each_dataset_cannot():
    with pytest.raises(NotImplementedError) as e:
        assert_not_supported(_Cfg(GT_BOXES_MOTION_COMPENSATION=True), 'KittiDataset')
    msg = str(e.value)
    assert 'no sequences' in msg, 'KITTI cannot ever support it, for a different reason'
    assert 'uuid' in msg and 'obj_ids' in msg, 'PandaSet and Waymo could, and the note should say so'


def test_absent_or_false_is_silent():
    assert_not_supported(_Cfg(), 'KittiDataset')
    assert_not_supported(_Cfg(GT_BOXES_MOTION_COMPENSATION=False), 'WaymoDataset')


@pytest.mark.parametrize('mod,cls', [
    ('pcdet/datasets/kitti/kitti_dataset.py', 'KittiDataset'),
    ('pcdet/datasets/waymo/waymo_dataset.py', 'WaymoDataset'),
    ('pcdet/datasets/pandaset/pandaset_dataset.py', 'PandasetDataset'),
])
def test_every_unsupported_loader_calls_the_guard(mod, cls):
    src = (Path(__file__).resolve().parent.parent / mod).read_text(encoding='utf-8')
    assert "assert_not_supported(self.dataset_cfg, '%s')" % cls in src
