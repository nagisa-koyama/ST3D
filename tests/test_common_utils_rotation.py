"""Tests for common_utils.rotate_points_along_z - the rotation under every box and point transform.

It is reached from `boxes_to_corners_3d`, the augmentor and the loss, so its dtype contract is not
a detail: it decides whether a caller's array can be used at all.

The dtype behaviour was a defect until 2026-09-23. `check_numpy_to_torch` casts a numpy array to
float32 but passes a torch tensor through untouched, while this function built its rotation matrix
with an unconditional `.float()` - so a float64 TENSOR died in matmul with "expected scalar type
Double but found Float" while the identical data as a float64 ARRAY worked. Found by
tests/test_box_utils.py, which pinned the failure before it was fixed. The matrix now takes the
points' dtype.

See experiments_md/20260923_01 §3.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.utils.common_utils import rotate_points_along_z  # noqa: E402

# One point on +x, one on +y, in a single batch.
POINTS = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
QUARTER_TURN = np.array([np.pi / 2])


# --- correctness ----------------------------------------------------------------------------------

def test_zero_angle_is_the_identity():
    out = rotate_points_along_z(POINTS.copy(), np.array([0.0]))
    assert np.allclose(out, POINTS)


def test_a_quarter_turn_maps_x_onto_y():
    """The docstring's convention: angle increases x ==> y."""
    out = rotate_points_along_z(POINTS.copy(), QUARTER_TURN)
    assert np.allclose(out[0, 0], [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(out[0, 1], [-1.0, 0.0, 0.0], atol=1e-6)


def test_z_is_untouched():
    points = np.array([[[1.0, 0.0, 7.0]]])
    assert rotate_points_along_z(points, QUARTER_TURN)[0, 0, 2] == pytest.approx(7.0)


def test_channels_beyond_xyz_are_carried_through():
    points = np.array([[[1.0, 0.0, 0.0, 0.42, 9.0]]])
    out = rotate_points_along_z(points, QUARTER_TURN)
    assert out.shape == (1, 1, 5)
    assert np.allclose(out[0, 0, 3:], [0.42, 9.0])


def test_each_batch_element_uses_its_own_angle():
    points = np.array([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    out = rotate_points_along_z(points, np.array([0.0, np.pi / 2]))
    assert np.allclose(out[0, 0], [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(out[1, 0], [0.0, 1.0, 0.0], atol=1e-6)


def test_rotating_by_an_angle_and_back_is_the_identity():
    rng = np.random.default_rng(0)
    points = rng.uniform(-10, 10, size=(1, 64, 3))
    there = rotate_points_along_z(points.copy(), np.array([0.7]))
    back = rotate_points_along_z(there, np.array([-0.7]))
    assert np.allclose(back, points, atol=1e-5)


# --- the dtype contract, which is what was fixed ------------------------------------------------------

def test_a_float64_tensor_works_and_keeps_its_dtype():
    """The regression test. This raised RuntimeError before 2026-09-23."""
    out = rotate_points_along_z(torch.from_numpy(POINTS).double(), torch.from_numpy(QUARTER_TURN))
    assert out.dtype == torch.float64
    assert torch.allclose(out[0, 0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-12)


def test_a_float32_tensor_still_works_and_keeps_its_dtype():
    out = rotate_points_along_z(torch.from_numpy(POINTS).float(), torch.from_numpy(QUARTER_TURN))
    assert out.dtype == torch.float32


def test_a_half_precision_tensor_works():
    out = rotate_points_along_z(torch.from_numpy(POINTS).half(), torch.from_numpy(QUARTER_TURN))
    assert out.dtype == torch.float16


def test_the_angle_dtype_need_not_match_the_points_dtype():
    """The matrix's own dtype comes from `angle`, so it is converted rather than constructed."""
    for angle in (torch.from_numpy(QUARTER_TURN).float(), torch.from_numpy(QUARTER_TURN).double()):
        for points_dtype in (torch.float32, torch.float64):
            out = rotate_points_along_z(torch.from_numpy(POINTS).to(points_dtype), angle)
            assert out.dtype == points_dtype


def test_float64_tensor_and_float64_array_now_agree():
    """The asymmetry that made the defect easy to miss: one path worked, the other raised."""
    from_array = rotate_points_along_z(POINTS.astype(np.float64), QUARTER_TURN)
    from_tensor = rotate_points_along_z(torch.from_numpy(POINTS).double(),
                                        torch.from_numpy(QUARTER_TURN)).numpy()
    assert np.allclose(from_array, from_tensor, atol=1e-6)


def test_a_numpy_input_still_returns_numpy_at_float32():
    """Unchanged on purpose: check_numpy_to_torch has always cast arrays to float32, and every
    caller that passes an array today gets a float32 result back. Widening the tensor path does
    not change that, and changing it would alter the output dtype of every numpy caller."""
    out = rotate_points_along_z(POINTS.astype(np.float64), QUARTER_TURN)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
