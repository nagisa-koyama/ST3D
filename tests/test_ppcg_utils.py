"""Tests for PPCG's geometry and ray tracer (DALI Tier D2.1/D2.2).

RC-PPCG replaces the returns inside a sparse pseudo box with points from a clean reference object
lying along the same rays. What has to be true for that to be meaningful:

  * the object frame round-trips. A library stores objects centred and heading-aligned and places
    them back at arbitrary poses; if that pair is not an exact inverse, every reference lands
    slightly wrong and nothing downstream can detect it.
  * the ray pattern is preserved exactly - one output point per observed return, so the box's
    point count and hence its density are unchanged. Regenerating a DIFFERENT number of points
    would confound PPCG with the density correction this repo already applies.
  * the chosen reference point is the shallowest among the best-aligned ones, i.e. the object
    self-occludes instead of mixing its front and back faces.
  * angular difference is circular. Upstream's abs(diff % 2pi) scores a -epsilon match as ~2pi,
    the worst possible, which silently corrupts reference selection for half of all near matches.

See experiments_md/20260922_04 section 2.4 for the tier plan and the module docstring of
ppcg_utils.py for the four upstream defects not reproduced.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.utils import ppcg_utils  # noqa: E402

BOX = np.array([12.0, -3.0, -0.8, 4.2, 1.8, 1.6, 0.7])


def _cloud(n=64, seed=0, cols=3):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.0, 1.0, size=(n, cols))
    return pts


# --- the object frame ---------------------------------------------------------------------------

def test_canonicalize_and_place_round_trip():
    world = _cloud(32) * 0.5 + BOX[0:3]
    back = ppcg_utils.place_object_points(
        ppcg_utils.canonicalize_object_points(world, BOX), BOX)
    assert np.allclose(back, world, atol=1e-9)


def test_canonicalize_centres_the_object_on_the_origin():
    """A box-shaped cloud, canonicalized, must be axis-aligned about 0."""
    rng = np.random.default_rng(3)
    local = rng.uniform(-0.5, 0.5, size=(256, 3)) * BOX[3:6]
    world = ppcg_utils.place_object_points(local, BOX)
    canonical = ppcg_utils.canonicalize_object_points(world, BOX)
    assert np.allclose(canonical.mean(axis=0), local.mean(axis=0), atol=1e-9)
    assert np.abs(canonical).max(axis=0) == pytest.approx(np.abs(local).max(axis=0), abs=1e-9)


def test_extra_channels_survive_canonicalization():
    pts = _cloud(16, cols=4)
    out = ppcg_utils.canonicalize_object_points(pts, BOX)
    assert np.array_equal(out[:, 3], pts[:, 3])


def test_geometry_helpers_do_not_mutate_their_input():
    pts = _cloud(16)
    before = pts.copy()
    ppcg_utils.canonicalize_object_points(pts, BOX)
    ppcg_utils.place_object_points(pts, BOX)
    ppcg_utils.scale_object_points(pts, [4.0, 2.0, 1.5], [4.4, 2.2, 1.6])
    assert np.array_equal(pts, before)


def test_empty_clouds_are_handled():
    empty = np.zeros((0, 3))
    assert ppcg_utils.canonicalize_object_points(empty, BOX).shape == (0, 3)
    assert ppcg_utils.place_object_points(empty, BOX).shape == (0, 3)
    assert ppcg_utils.scale_object_points(empty, [4, 2, 1.5], [4, 2, 1.5]).shape == (0, 3)


# --- scaling a reference to a box ----------------------------------------------------------------

def test_scale_object_points_is_per_axis():
    pts = np.array([[1.0, 1.0, 1.0]])
    out = ppcg_utils.scale_object_points(pts, [2.0, 2.0, 2.0], [4.0, 1.0, 6.0])
    assert np.allclose(out, [[2.0, 0.5, 3.0]])


def test_scaling_a_reference_to_its_own_extent_is_the_identity():
    pts = _cloud(16)
    assert np.allclose(ppcg_utils.scale_object_points(pts, [4.2, 1.8, 1.6], [4.2, 1.8, 1.6]), pts)


# --- circular distance, the bug not reproduced ----------------------------------------------------

def test_angular_difference_is_circular():
    assert ppcg_utils.angular_difference(0.01, -0.01) == pytest.approx(0.02)
    assert ppcg_utils.angular_difference(-0.01, 0.0) == pytest.approx(0.01)


def test_angular_difference_wraps_across_pi():
    assert ppcg_utils.angular_difference(np.pi - 0.05, -np.pi + 0.05) == pytest.approx(0.1)


def test_angular_difference_never_exceeds_pi():
    rng = np.random.default_rng(1)
    a, b = rng.uniform(-10, 10, 500), rng.uniform(-10, 10, 500)
    diff = ppcg_utils.angular_difference(a, b)
    assert diff.min() >= 0.0 and diff.max() <= np.pi + 1e-12


def test_upstream_modulo_would_have_scored_a_near_match_as_the_worst():
    """Pins WHY the helper exists rather than just that it works."""
    upstream = np.abs((-0.01) % (2 * np.pi))
    assert upstream > 6.2                                   # ranked as maximally distant
    assert ppcg_utils.angular_difference(-0.01, 0.0) < 0.02  # actually a near-perfect match


# --- the ray tracer -------------------------------------------------------------------------------

def _wall(x, n=400, seed=5, y=(-1.0, 1.0), z=(-0.8, 0.8)):
    """A flat surface at distance x in the object frame, i.e. a fronto-parallel slab."""
    rng = np.random.default_rng(seed)
    return np.stack([np.full(n, x),
                     rng.uniform(*y, size=n),
                     rng.uniform(*z, size=n)], axis=1)


def test_output_has_exactly_one_point_per_observed_return():
    box = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0])
    observed = ppcg_utils.place_object_points(_wall(-1.0, n=25), box)
    reference = _wall(-1.0, n=300, seed=9)
    out = ppcg_utils.ray_constrained_resample(reference, observed, [0, 0, 0], box)
    assert out.shape == (25, 3)


def test_the_shallowest_of_the_aligned_candidates_wins():
    """Two parallel surfaces; every ray must land on the near one."""
    box = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0])
    observed = ppcg_utils.place_object_points(_wall(-1.5, n=20), box)
    reference = np.concatenate([_wall(-1.5, n=300, seed=1), _wall(1.5, n=300, seed=2)])
    out = ppcg_utils.ray_constrained_resample(reference, observed, [0, 0, 0], box, k=10)
    local = ppcg_utils.canonicalize_object_points(out, box)
    assert (local[:, 0] < 0).all()          # the near face, never the far one


def test_a_reference_at_the_box_pose_reproduces_that_surface():
    box = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0])
    observed = ppcg_utils.place_object_points(_wall(-1.0, n=30), box)
    reference = _wall(-1.0, n=800, seed=7)
    out = ppcg_utils.ray_constrained_resample(reference, observed, [0, 0, 0], box)
    local = ppcg_utils.canonicalize_object_points(out, box)
    assert np.allclose(local[:, 0], -1.0, atol=1e-9)


def test_rotation_of_the_box_is_respected():
    """The reference is stored canonically, so a rotated box must yield a rotated surface."""
    box = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 1.6, np.pi / 2])
    observed = ppcg_utils.place_object_points(_wall(-1.0, n=30), box)
    out = ppcg_utils.ray_constrained_resample(_wall(-1.0, n=800, seed=7), observed, [0, 0, 0], box)
    local = ppcg_utils.canonicalize_object_points(out, box)
    assert np.allclose(local[:, 0], -1.0, atol=1e-9)


def test_reference_points_are_not_mutated():
    """Upstream rotates and translates the caller's array in place."""
    box = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.3])
    reference = _wall(-1.0, n=100)
    before = reference.copy()
    observed = ppcg_utils.place_object_points(_wall(-1.0, n=10), box)
    ppcg_utils.ray_constrained_resample(reference, observed, [0, 0, 0], box)
    assert np.array_equal(reference, before)


def test_degenerate_inputs_pass_the_observation_through():
    box = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0])
    observed = ppcg_utils.place_object_points(_wall(-1.0, n=5), box)
    assert np.array_equal(
        ppcg_utils.ray_constrained_resample(np.zeros((0, 3)), observed, [0, 0, 0], box), observed)
    assert ppcg_utils.ray_constrained_resample(
        _wall(-1.0), np.zeros((0, 3)), [0, 0, 0], box).shape == (0, 3)


def test_k_larger_than_the_library_is_clamped():
    box = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0])
    observed = ppcg_utils.place_object_points(_wall(-1.0, n=4), box)
    out = ppcg_utils.ray_constrained_resample(_wall(-1.0, n=3), observed, [0, 0, 0], box, k=10)
    assert out.shape == (4, 3)


def test_extra_channels_come_from_the_observation_not_the_reference():
    box = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0])
    observed = ppcg_utils.place_object_points(_wall(-1.0, n=6), box)
    observed = np.concatenate([observed, np.full((6, 1), 0.42)], axis=1)
    reference = np.concatenate([_wall(-1.0, n=50), np.full((50, 1), 0.99)], axis=1)
    out = ppcg_utils.ray_constrained_resample(reference, observed, [0, 0, 0], box)
    assert out.shape == (6, 4)
    assert np.allclose(out[:, 3], 0.42)


# --- the sensor origin ----------------------------------------------------------------------------

def test_sensor_origin_defaults_to_the_frame_origin():
    assert np.array_equal(ppcg_utils.sensor_origin(EasyDict({})), [0.0, 0.0, 0.0])


def test_sensor_origin_follows_shift_coor():
    cfg = EasyDict({'SHIFT_COOR': [0.0, 0.0, 1.75]})
    assert np.array_equal(ppcg_utils.sensor_origin(cfg), [0.0, 0.0, 1.75])


# --- the packed library ---------------------------------------------------------------------------

def _objects(n=3, seed=2):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        pts = rng.uniform(-1, 1, size=(10 + i * 5, 3)).astype(np.float32)
        out.append({'points': pts, 'box': np.array([1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.3]),
                    'direction': 0.1 * i, 'orientation': 0.3, 'numpts': len(pts),
                    'size': 2.29, 'name': 'obj%d' % i})
    return out


def test_reference_library_round_trips(tmp_path):
    objects = _objects()
    path = tmp_path / 'ref.npz'
    ppcg_utils.save_reference_objects(str(path), objects, metadata={'class_name': 'car'})
    library = ppcg_utils.load_reference_objects(str(path))
    assert len(library) == len(objects)
    for i, obj in enumerate(objects):
        assert np.allclose(library.points(i), obj['points'])
    assert np.allclose(library.lwh[0], [4.0, 2.0, 1.5])
    assert 'car' in library.metadata


def test_reference_library_offsets_do_not_interleave_objects(tmp_path):
    """Variable-length objects packed into one array - the offsets are the only thing keeping
    them apart, so a single off-by-one would silently mix two cars."""
    objects = _objects(4)
    path = tmp_path / 'ref.npz'
    ppcg_utils.save_reference_objects(str(path), objects)
    library = ppcg_utils.load_reference_objects(str(path))
    for i, obj in enumerate(objects):
        assert len(library.points(i)) == len(obj['points'])


def test_an_empty_library_saves_and_loads(tmp_path):
    path = tmp_path / 'ref.npz'
    ppcg_utils.save_reference_objects(str(path), [])
    assert len(ppcg_utils.load_reference_objects(str(path))) == 0
