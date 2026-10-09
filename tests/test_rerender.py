"""Sentinel tests for the spec-conditioned re-rendering operator (pcdet/datasets/rerender_utils.py,
experiments_md 20261009_01): empty cloud, one point, ground only, two planes at a depth edge (no bridging), a box
between two source rings (filled), the K limit and its angular variant, the horizontal variant, attributes."""
import os

import numpy as np
import pytest

from pcdet.datasets import rerender_utils as R

SPEC = dict(thetas=np.radians(np.linspace(-10.0, 2.0, 49)), n_cols=1440, height=2.0)   # 0.25 deg rows, 0.25 deg cols


def ray_point(i, j, r, spec=SPEC, extra=()):
    th = spec['thetas'][i]; az = (j + 0.5) / spec['n_cols'] * 2 * np.pi - np.pi
    return [r * np.cos(th) * np.cos(az), r * np.cos(th) * np.sin(az), spec['height'] + r * np.sin(th), *extra]


def test_empty_cloud():
    out, filled = R.rerender(np.zeros((0, 4), np.float32), SPEC)
    assert out.shape == (0, 4) and filled.shape == (0,)


def test_one_point_lands_on_its_ray():
    p = np.array([ray_point(20, 700, 25.0, extra=(7.0,))], np.float32)
    out, filled = R.rerender(p, SPEC)
    assert out.shape == (1, 4) and not filled.any()
    assert np.allclose(out[0, :3], p[0, :3], atol=1e-4) and out[0, 3] == 7.0


def test_zbuffer_keeps_the_nearest_return_per_cell():
    p = np.array([ray_point(10, 100, 30.0, extra=(1.0,)), ray_point(10, 100, 12.0, extra=(2.0,)),
                  ray_point(10, 100, 50.0, extra=(3.0,))], np.float32)
    out, filled = R.rerender(p, SPEC)
    assert len(out) == 1 and out[0, 3] == 2.0
    assert np.isclose(np.linalg.norm(out[0, :3] - [0, 0, SPEC['height']]), 12.0, atol=1e-3)


def test_ground_only_is_not_filled():
    """Rays every 4th row hit the ground (z = 0): adjacent hits differ in range by metres, so nothing is filled."""
    pts = []
    for i in range(0, 40, 4):                    # rows below the horizon
        th = SPEC['thetas'][i]
        if th >= 0:
            continue
        r = SPEC['height'] / np.sin(-th)
        for j in range(0, SPEC['n_cols'], 3):
            pts.append(ray_point(i, j, r))
    out, filled = R.rerender(np.array(pts, np.float32), SPEC)
    assert filled.sum() == 0 and len(out) == len(pts)
    assert np.abs(out[:, 2]).max() < 1e-3


def test_two_planes_at_a_depth_edge_are_not_bridged():
    """A near wall (20 m) on the upper rows, a far wall (40 m) on the lower rows, source returns every 4th row:
    rows between the two walls must stay empty."""
    pts = []
    for i in range(0, 49, 4):
        r = 20.0 if i >= 24 else 40.0
        for j in range(600, 640):
            pts.append(ray_point(i, j, r))
    out, filled = R.rerender(np.array(pts, np.float32), SPEC)
    rng = np.linalg.norm(out[:, :3] - [0, 0, SPEC['height']], axis=1)
    assert np.all((np.abs(rng - 20.0) < 1e-3) | (np.abs(rng - 40.0) < 1e-3))
    # inside each wall the gaps ARE filled (both neighbours on the same surface)
    assert filled.sum() > 0


def test_box_between_two_rings_is_filled():
    """A car-like vertical patch at constant range seen by two source rings 4 rows apart: the 3 rows between fill."""
    pts = [ray_point(i, j, 30.0) for i in (20, 24) for j in range(700, 720)]
    out, filled = R.rerender(np.array(pts, np.float32), SPEC)
    assert filled.sum() == 3 * 20 and len(out) == 5 * 20
    rng = np.linalg.norm(out[:, :3] - [0, 0, SPEC['height']], axis=1)
    assert np.allclose(rng, 30.0, atol=1e-3)


def test_interpolation_is_linear_in_row_index():
    pts = [ray_point(20, 5, 30.0), ray_point(24, 5, 30.2)]
    out, filled = R.rerender(np.array(pts, np.float32), SPEC)
    rng = np.sort(np.linalg.norm(out[filled][:, :3] - [0, 0, SPEC['height']], axis=1))
    assert np.allclose(rng, [30.05, 30.10, 30.15], atol=1e-3)


def test_range_disagreement_blocks_the_fill():
    pts = [ray_point(20, 5, 30.0), ray_point(24, 5, 30.31)]
    _, filled = R.rerender(np.array(pts, np.float32), SPEC, dr_max=0.3)
    assert filled.sum() == 0


def test_k_limit_and_the_angular_variant():
    pts = [ray_point(10, 5, 30.0), ray_point(20, 5, 30.0)]          # 10 rows (2.5 deg) apart
    _, filled = R.rerender(np.array(pts, np.float32), SPEC, k_rows=4)
    assert filled.sum() == 0                                        # the middle rows are > 4 rows from one side
    _, filled = R.rerender(np.array(pts, np.float32), SPEC, k_rows=5)
    assert filled.sum() == 1                                        # only row 15 is within 5 of both
    _, filled = R.rerender(np.array(pts, np.float32), SPEC, span_max_deg=2.6)
    assert filled.sum() == 9
    _, filled = R.rerender(np.array(pts, np.float32), SPEC, span_max_deg=2.4)
    assert filled.sum() == 0


def test_one_sided_neighbour_is_not_extrapolated():
    pts = [ray_point(20, 5, 30.0)]
    _, filled = R.rerender(np.array(pts, np.float32), SPEC)
    assert filled.sum() == 0


def test_horizontal_variant():
    pts = [ray_point(20, 100, 30.0), ray_point(20, 102, 30.1)]
    _, filled = R.rerender(np.array(pts, np.float32), SPEC)
    assert filled.sum() == 0
    out, filled = R.rerender(np.array(pts, np.float32), SPEC, h_fill=True)
    assert filled.sum() == 1


def test_attributes_of_filled_cells_come_from_the_lower_neighbour():
    pts = [ray_point(20, 5, 30.0, extra=(4.0,)), ray_point(24, 5, 30.0, extra=(9.0,))]
    out, filled = R.rerender(np.array(pts, np.float32), SPEC)
    assert np.all(out[filled][:, 3] == 4.0)


def test_fill_none_is_the_zbuffer_alone():
    pts = [ray_point(i, j, 30.0) for i in (20, 24) for j in range(700, 720)]
    out, filled = R.rerender(np.array(pts, np.float32), SPEC, fill='none')
    assert filled.sum() == 0 and len(out) == 40


def test_out_of_fov_points_are_dropped():
    p = np.array([[10.0, 0.0, 30.0]], np.float32)                   # ~70 deg up
    out, _ = R.rerender(p, SPEC)
    assert len(out) == 0


def test_hdl32e_spec():
    s = R.lattice_spec('hdl32e')
    assert len(s['thetas']) == 32 and s['n_cols'] == 1084 and s['height'] == 1.84
    assert np.allclose(np.degrees(np.diff(s['thetas'])), 41.34 / 31, atol=1e-6)


@pytest.mark.skipif(not os.path.exists(R.WAYMO_TOP_CALIB), reason='Waymo TOP calibration not on disk')
def test_waymo_top_spec_is_the_median_lattice():
    s = R.lattice_spec('waymo_top')
    assert len(s['thetas']) == 64 and s['n_cols'] == 2650 and s['height'] == 2.184
    assert np.all(np.diff(s['thetas']) > 0)
    assert np.isclose(np.degrees(s['thetas'][0]), -17.61, atol=0.02) and np.isclose(np.degrees(s['thetas'][-1]), 2.33, atol=0.02)
