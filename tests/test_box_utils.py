"""Tests for pcdet/utils/box_utils.py - the geometry every dataset and the eval path sit on.

This module had no tests. It decides which GT survives `POINT_CLOUD_RANGE`, what a box's corners
are (hence every IoU and the KITTI camera-FOV prediction filter), and how boxes convert between
the two LiDAR conventions this codebase carries. Its failure mode is the worst kind: a wrong
rotation or a dropped half-height produces numbers that look entirely plausible.

Three things get particular attention because they are the ones that bite silently:

  * `mask_boxes_outside_range_numpy`'s `min_num_corners`. It is a CORNER count, not a centre
    test, so a box straddling the boundary is kept at 1 and dropped at 8 - and the repo's own
    audit found the GT branch of the range filter is gated on `self.training`, so predictions and
    ground truth are not filtered alike. The semantics are worth pinning wherever they are used.
  * the fakelidar conversions, which simultaneously swap dx/dy, shift z by half the height and
    negate-and-offset the heading. Round-tripping is the only cheap way to be sure all three
    happened in both directions.
  * `boxes3d_lidar_to_aligned_bev_boxes`, which picks dx/dy or dy/dx depending on whether the
    heading is nearer an axis or nearer 45 degrees. That branch is easy to read past.

Everything here is CPU-only and exact; where a value is analytically known the test asserts the
value, not merely a property.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.utils import box_utils  # noqa: E402

# A 4 x 2 x 1.5 m box at the origin, axis-aligned. Chosen so every corner is a round number.
UNIT_BOX = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])


# --- corners --------------------------------------------------------------------------------------

def test_corners_of_an_axis_aligned_box_are_the_half_extents():
    corners = box_utils.boxes_to_corners_3d(UNIT_BOX)[0]
    assert corners.shape == (8, 3)
    assert np.allclose(np.abs(corners).max(axis=0), [2.0, 1.0, 0.75])
    assert np.allclose(corners.mean(axis=0), [0.0, 0.0, 0.0])


def test_corners_follow_the_box_centre():
    box = UNIT_BOX.copy()
    box[0, 0:3] = [10.0, -5.0, 2.0]
    corners = box_utils.boxes_to_corners_3d(box)[0]
    assert np.allclose(corners.mean(axis=0), [10.0, -5.0, 2.0])


def test_a_quarter_turn_swaps_the_footprint():
    """Heading is about +z, so dx and dy exchange roles at 90 degrees."""
    box = UNIT_BOX.copy()
    box[0, 6] = np.pi / 2
    corners = box_utils.boxes_to_corners_3d(box)[0]
    assert np.allclose(np.abs(corners).max(axis=0), [1.0, 2.0, 0.75], atol=1e-6)


def test_corner_count_and_ordering_are_stable_under_rotation():
    """The z split in the template must stay a z split whatever the heading."""
    box = UNIT_BOX.copy()
    box[0, 6] = 0.7
    corners = box_utils.boxes_to_corners_3d(box)[0]
    assert (corners[0:4, 2] < 0).all() and (corners[4:8, 2] > 0).all()


def test_corners_accept_float32_torch_and_return_torch():
    out = box_utils.boxes_to_corners_3d(torch.from_numpy(UNIT_BOX).float())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 8, 3)


def test_corners_reject_a_float64_torch_tensor():
    """A latent asymmetry, pinned rather than fixed.

    `check_numpy_to_torch` casts a numpy array to float32 but passes a torch tensor through
    untouched, while `rotate_points_along_z` builds its rotation matrix with an unconditional
    `.float()`. So the numpy path accepts float64 and the torch path does not - it raises
    "expected scalar type Double but found Float". Every live call site passes either numpy or a
    float32 model tensor, so this never fires today; it is recorded here so that the next caller
    to hand it a float64 tensor finds a test rather than a puzzling RuntimeError. Fixing it means
    editing a util shared by the loss and every dataset, which is a wider change than a test.
    """
    with pytest.raises(RuntimeError, match='Double'):
        box_utils.boxes_to_corners_3d(torch.from_numpy(UNIT_BOX).double())


def test_the_numpy_path_accepts_float64():
    """The same input as a numpy array works, which is what makes the asymmetry easy to miss."""
    assert box_utils.boxes_to_corners_3d(UNIT_BOX.astype(np.float64)).shape == (1, 8, 3)


# --- range masking --------------------------------------------------------------------------------

LIMIT = [-10.0, -10.0, -5.0, 10.0, 10.0, 5.0]


def test_a_box_well_inside_the_range_is_kept():
    assert box_utils.mask_boxes_outside_range_numpy(UNIT_BOX.copy(), LIMIT).tolist() == [True]


def test_a_box_well_outside_the_range_is_dropped():
    box = UNIT_BOX.copy()
    box[0, 0] = 100.0
    assert box_utils.mask_boxes_outside_range_numpy(box, LIMIT).tolist() == [False]


def test_min_num_corners_counts_corners_not_centres():
    """A straddling box: kept when one corner suffices, dropped when all eight are required."""
    box = UNIT_BOX.copy()
    box[0, 0] = 9.0          # spans x in [7, 11], so half its corners are past the +10 wall
    assert box_utils.mask_boxes_outside_range_numpy(box, LIMIT, min_num_corners=1).tolist() == [True]
    assert box_utils.mask_boxes_outside_range_numpy(box, LIMIT, min_num_corners=8).tolist() == [False]


def test_the_range_test_is_inclusive_of_its_bounds():
    box = UNIT_BOX.copy()
    box[0, 0] = 8.0          # the +x face lands exactly on the +10 boundary
    assert box_utils.mask_boxes_outside_range_numpy(box, LIMIT, min_num_corners=8).tolist() == [True]


def test_extra_box_columns_are_tolerated():
    """gt_boxes carries a class column appended by prepare_data; the mask must still work."""
    box = np.concatenate([UNIT_BOX, np.ones((1, 1))], axis=1)
    assert box_utils.mask_boxes_outside_range_numpy(box, LIMIT).tolist() == [True]


# --- the two LiDAR conventions ---------------------------------------------------------------------

def test_fakelidar_round_trip_is_the_identity():
    """Swap dx/dy, shift z by half the height, negate and offset the heading - both ways."""
    boxes = np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.3],
                      [-7.0, 0.5, -1.0, 3.6, 1.7, 1.4, -2.1]])
    back = box_utils.boxes3d_kitti_fakelidar_to_lidar(
        box_utils.boxes3d_kitti_lidar_to_fakelidar(boxes))
    assert np.allclose(back[:, 0:6], boxes[:, 0:6], atol=1e-9)
    assert np.allclose(np.cos(back[:, 6]), np.cos(boxes[:, 6]), atol=1e-9)
    assert np.allclose(np.sin(back[:, 6]), np.sin(boxes[:, 6]), atol=1e-9)


def test_lidar_to_fakelidar_moves_z_to_the_box_bottom():
    boxes = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    assert box_utils.boxes3d_kitti_lidar_to_fakelidar(boxes)[0, 2] == pytest.approx(-0.75)


def test_fakelidar_to_lidar_moves_z_back_to_the_box_centre():
    boxes = np.array([[0.0, 0.0, -0.75, 2.0, 4.0, 1.5, 0.0]])
    assert box_utils.boxes3d_kitti_fakelidar_to_lidar(boxes)[0, 2] == pytest.approx(0.0)


def test_the_conversions_swap_the_horizontal_extents():
    boxes = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    assert np.allclose(box_utils.boxes3d_kitti_lidar_to_fakelidar(boxes)[0, 3:6], [2.0, 4.0, 1.5])


def test_the_conversions_do_not_mutate_their_input():
    boxes = np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.3]])
    before = boxes.copy()
    box_utils.boxes3d_kitti_lidar_to_fakelidar(boxes)
    box_utils.boxes3d_kitti_fakelidar_to_lidar(boxes)
    assert np.array_equal(boxes, before)


# --- enlarging ---------------------------------------------------------------------------------------

def test_enlarge_box3d_grows_only_the_extents():
    out = box_utils.enlarge_box3d(UNIT_BOX.copy(), extra_width=(0.2, 0.4, 0.6))
    assert np.allclose(out[0, 3:6], [4.2, 2.4, 2.1])
    assert np.allclose(out[0, 0:3], UNIT_BOX[0, 0:3])
    assert out[0, 6] == pytest.approx(UNIT_BOX[0, 6])


def test_enlarge_box3d_does_not_mutate_its_input():
    boxes = UNIT_BOX.copy()
    box_utils.enlarge_box3d(boxes, extra_width=(1.0, 1.0, 1.0))
    assert np.array_equal(boxes, UNIT_BOX)


# --- BEV projection and IoU ----------------------------------------------------------------------------

def test_aligned_bev_box_of_an_axis_aligned_box_uses_dx_dy():
    bev = box_utils.boxes3d_lidar_to_aligned_bev_boxes(torch.from_numpy(UNIT_BOX)).numpy()[0]
    assert np.allclose(bev, [-2.0, -1.0, 2.0, 1.0])


def test_aligned_bev_box_swaps_dims_past_forty_five_degrees():
    """The branch that is easy to read past: nearer 90 degrees than 0, so dy/dx is the footprint."""
    box = UNIT_BOX.copy()
    box[0, 6] = np.pi / 2
    bev = box_utils.boxes3d_lidar_to_aligned_bev_boxes(torch.from_numpy(box)).numpy()[0]
    assert np.allclose(bev, [-1.0, -2.0, 1.0, 2.0])


def test_aligned_bev_box_is_periodic_in_pi():
    """A box rotated by pi occupies the same footprint, so the projection must agree."""
    box = UNIT_BOX.copy()
    turned = UNIT_BOX.copy()
    turned[0, 6] = np.pi
    a = box_utils.boxes3d_lidar_to_aligned_bev_boxes(torch.from_numpy(box)).numpy()
    b = box_utils.boxes3d_lidar_to_aligned_bev_boxes(torch.from_numpy(turned)).numpy()
    assert np.allclose(a, b, atol=1e-6)


def test_normal_iou_of_a_box_with_itself_is_one():
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    assert box_utils.boxes_iou_normal(boxes, boxes).item() == pytest.approx(1.0)


def test_normal_iou_of_disjoint_boxes_is_zero():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[5.0, 5.0, 6.0, 6.0]])
    assert box_utils.boxes_iou_normal(a, b).item() == pytest.approx(0.0)


def test_normal_iou_of_a_half_overlap_is_one_third():
    """Two unit squares overlapping on half their area: 0.5 / (1 + 1 - 0.5)."""
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[0.5, 0.0, 1.5, 1.0]])
    assert box_utils.boxes_iou_normal(a, b).item() == pytest.approx(1.0 / 3.0)


def test_nearest_bev_iou_of_a_box_with_itself_is_one():
    boxes = torch.from_numpy(UNIT_BOX)
    assert box_utils.boxes3d_nearest_bev_iou(boxes, boxes).item() == pytest.approx(1.0)


# --- convex-hull containment ---------------------------------------------------------------------------

def test_in_hull_separates_inside_from_outside():
    corners = box_utils.boxes_to_corners_3d(UNIT_BOX)[0]
    points = np.array([[0.0, 0.0, 0.0], [1.9, 0.9, 0.7], [3.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    assert box_utils.in_hull(points, corners).tolist() == [True, True, False, False]


def test_in_hull_follows_the_rotated_hull():
    box = UNIT_BOX.copy()
    box[0, 6] = np.pi / 2
    corners = box_utils.boxes_to_corners_3d(box)[0]
    # (1.9, 0, 0) is inside before the turn and outside after it; (0, 1.9, 0) is the reverse.
    assert box_utils.in_hull(np.array([[1.9, 0.0, 0.0], [0.0, 1.9, 0.0]]), corners).tolist() \
        == [False, True]


# --- removing points that fall in boxes -------------------------------------------------------------------

def test_remove_points_in_boxes3d_keeps_only_the_outside():
    points = np.array([[0.0, 0.0, 0.0, 0.5], [9.0, 9.0, 0.0, 0.5]])
    kept = box_utils.remove_points_in_boxes3d(points, UNIT_BOX.copy())
    assert len(kept) == 1
    assert np.allclose(kept[0, 0:3], [9.0, 9.0, 0.0])
