"""tools/analysis/box_pattern_visual.py (CPU; experiments_md 20261018_03).

Pins the sensor's-eye projection: angles are taken about EACH cloud's own sensor. A toy point at a known elevation about
the nuScenes LIDAR_TOP, which SHIFT_COOR moves to (0, 0, 1.75), reads that elevation, and reads a different one about the
shifted ground origin; a rendered RR point sits exactly on its lattice row about (0, 0, 2.184), as `to_lattice` assigns
it; a Waymo point at a known sensor-frame inclination reads it through a pitched, yawed TOP extrinsic, equal to
`waymo_rings.top_columns_and_elevations`. Also: the relative viewing angle, the box outline, the fixed-seed quantile
pick, the block index, the footprint distance and the nearest-row residual.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import box_pattern_visual as V  # noqa: E402


def _rot(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return Rz @ Ry


def test_nuscenes_angles_about_the_shifted_sensor_not_the_ground_origin():
    pose = V.sensor_pose('nus', shift=[0.0, 0.0, 1.75])
    assert np.allclose(pose['origin'], [0, 0, 1.75]) and pose['rot'] is None
    r, el = 20.0, -3.0
    p = np.array([[r, 0.0, 1.75 + r * np.tan(np.radians(el))], [0.0, r, 1.75 + r * np.tan(np.radians(el))]])
    az, e = V.view_angles(p, pose)
    assert np.allclose(e, el, atol=1e-12)
    assert np.allclose(az, [0.0, 90.0], atol=1e-12)
    # the same point about the shifted ground origin reads ~ +2.0 deg instead: the trap the sensor pose avoids
    _, e0 = V.view_angles(p, V.sensor_pose('nus', shift=[0.0, 0.0, 0.0]))
    assert np.all(np.abs(e0 - el) > 4.0)


def test_rr_render_point_sits_on_its_row_about_the_virtual_top():
    from pcdet.datasets.rerender_utils import to_lattice
    th = np.radians(np.linspace(-17.6, 2.3, 64)); h = 2.184; nc = 2650
    pose = V.sensor_pose('rr', height=h)
    k, az, r = 40, np.radians(33.3), 55.0
    p = np.array([[r * np.cos(th[k]) * np.cos(az), r * np.cos(th[k]) * np.sin(az), h + r * np.sin(th[k])]])
    a, e = V.view_angles(p, pose)
    assert abs(e[0] - np.degrees(th[k])) < 1e-10 and abs(a[0] - 33.3) < 1e-10
    row, col, rng, valid = to_lattice(p, h, th, nc)
    assert valid[0] and row[0] == k
    assert V.nearest_row(e, np.degrees(th))[0][0] == k


def test_waymo_angles_in_the_top_sensor_frame():
    from pcdet.datasets.waymo.waymo_rings import top_columns_and_elevations
    E = np.eye(4); E[:3, :3] = _rot(np.radians(148.0), np.radians(0.4)); E[:3, 3] = [1.43, 0.0, 2.184]
    pose = V.sensor_pose('waymo', extrinsic=E)
    assert np.allclose(pose['origin'], [1.43, 0, 2.184]) and abs(pose['yaw_deg'] - 148.0) < 1e-9
    inc, az_s, r = np.radians(-1.5), np.radians(-120.0), 60.0
    ps = np.array([r * np.cos(inc) * np.cos(az_s), r * np.cos(inc) * np.sin(az_s), r * np.sin(inc)])
    pv = (E[:3, :3] @ ps + E[:3, 3])[None]
    a, e = V.view_angles(pv, pose)
    assert abs(e[0] + 1.5) < 1e-9
    assert abs(V.wrap_deg(a[0] - (-120.0 + 148.0))) < 1e-9          # azimuth in the vehicle's axes
    _, el_rings = top_columns_and_elevations(pv, E)
    assert abs(np.degrees(el_rings[0]) - e[0]) < 1e-9
    # ignoring the sensor's pitch (origin only) misreads it by more than half a near-horizon row gap
    _, e_naive = V.view_angles(pv, dict(origin=E[:3, 3], rot=None, yaw_deg=0.0))
    assert abs(e_naive[0] - e[0]) > 0.1


def test_relative_view_angle_and_outline():
    o = np.zeros(3)
    b = np.array([30.0, 30.0, 0.8, 4.5, 1.9, 1.6, np.radians(45.0)])
    assert abs(V.relative_view_angle_deg(b, o)) < 1e-9
    b2 = b.copy(); b2[6] = np.radians(45.0 + 90.0)
    assert abs(V.relative_view_angle_deg(b2, o) - 90.0) < 1e-9
    c = V.box_corners(b)
    assert np.allclose(c.mean(0), b[:3]) and np.isclose(c[:, 2].max() - c[:, 2].min(), b[5])
    polys = V.box_edge_polylines(b, n=5)
    assert len(polys) == 12 and all(len(p) == 5 for p in polys)
    ends = np.concatenate([p[[0, -1]] for p in polys])
    assert all(np.isclose(np.linalg.norm(ends - cc, axis=1), 0).any() for cc in c)


def test_pick_quantile_boxes_fixed_seed_nearest_distinct():
    v = np.r_[np.arange(100), np.full(5, 50)]
    i1, q1, n1 = V.pick_quantile_boxes(v, 7, pool_n=60)
    i2, q2, n2 = V.pick_quantile_boxes(v, 7, pool_n=60)
    assert np.array_equal(i1, i2) and np.allclose(q1, q2) and n1 == 60
    assert len(set(i1.tolist())) == 3
    pool = np.random.default_rng(7).permutation(len(v))[:60]
    assert np.allclose(q1, np.percentile(v[pool], (25, 50, 75)))
    for i, q in zip(i1, q1):
        assert i in pool and abs(v[i] - q) <= np.min(np.abs(v[pool] - q)) + 1.0
    i3, _, n3 = V.pick_quantile_boxes(np.array([5.0, 5.0]), 1)
    assert n3 == 2 and len(i3) == 2


def test_inbox_split_equals_the_cropped_model_input():
    from source_coverage import crop, in_box_indices
    rng = np.random.default_rng(3)
    P = np.c_[rng.uniform(60, 70, 4000), rng.uniform(-5, 5, 4000), rng.uniform(-3.5, 0.5, 4000)].astype(np.float32)
    P[:5, 0] = 80.0                                            # outside the range in x
    box = np.array([65.0, 0.0, -1.8, 4.5, 2.0, 1.6, 0.3])     # straddles the grid's z floor at -2 m
    Pxy = np.ascontiguousarray(P[V.xy_mask(P)], np.float32)
    idx, ing = V._inbox_split(Pxy, box)
    ref = in_box_indices(crop(P), box[None])[0]
    assert (~ing).sum() > 0 and ing.sum() == len(ref)
    assert np.array_equal(np.sort(Pxy[idx[ing]], 0), np.sort(crop(P)[ref], 0))


def test_bottom_elevation_offset():
    assert np.allclose(V.bottom_elevation_offset([0.0, 0.0], [20.0, 60.0], 2.184), 0.0)
    # a bottom 60 m out, 2.6 m below flat ground (Boston's +2.29 deg pitch, behind the ego) sits ~2.5 deg lower
    d = V.bottom_elevation_offset(-60 * np.tan(np.radians(2.29)), 60.0, 2.184)
    assert -2.6 < d < -2.2
    assert np.isclose(V.bottom_elevation_offset(2.184, 30.0, 2.184), np.degrees(np.arctan2(2.184, 30.0)))


def test_block_index_footprint_and_rows():
    k = np.array([[0, 3], [0, 3], [0, 7], [1, 7], [1, 7], [1, 7]])
    assert V.block_index(k).tolist() == [0, 1, 0, 0, 1, 2]
    b = np.array([0.0, 0.0, 0.0, 4.0, 2.0, 1.5, np.pi / 2])
    d = V.footprint_distance(np.array([[0.0, 0.0], [0.0, 3.0], [2.0, 0.0]]), b)
    assert np.allclose(d, [0.0, 1.0, 1.0])
    k, res = V.nearest_row([-1.04, -0.5], [-1.0, -0.2, 0.3])
    assert k.tolist() == [0, 1] and np.allclose(res, [0.04, 0.3])
    w, h = V.sheet_window(30.0)
    assert np.isclose(w, 2 * np.degrees(np.arctan(3.2 / 30))) and np.isclose(h, 0.42 * w)
