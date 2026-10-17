"""tools/analysis/source_coverage.py (CPU; experiments_md 20261018_01).

Pins the per-box descriptors on the model grid (empty box, one-point box, the per-voxel cap, the z cut), the ring and
bin edges, the label sentinels (negative pseudo-label ignore, -99 IGNORE_CLASS_LABEL, other classes) and the nuScenes
other-vehicle selection from the infos, the B3 relocation geometry (exact range, azimuth, heading relative to the line
of sight, the side facing the sensor, bottom on the ground, a donor with no points), the render (one point per pixel,
nearest kept, snapped onto its ray, the same rows and columns as lattice_pixels and as rerender_utils' dense path),
G1's row coverage with occlusion, the z-buffer keep, the phantom count, and the coverage / A2 / reach logic.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import source_coverage as S  # noqa: E402
from pcdet.datasets import rerender_utils as RR  # noqa: E402
from pcdet.datasets.processor.point_sampler import lattice_pixels  # noqa: E402
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils  # noqa: E402
from pcdet.utils import box_utils  # noqa: E402

TH = np.radians(np.concatenate([np.linspace(-17.6, -3.0, 40, endpoint=False), np.linspace(-3.0, 2.4, 30)]))
NC, H = 2650, 2.184


def vox_centre(i, j, k):
    return S.PCR[:3] + (np.array([i, j, k]) + 0.5) * S.VOX


def test_descriptors_empty_and_one_point():
    assert S.descriptors(np.zeros((0, 3))) == (0, 0, 0, 0.0)
    assert S.descriptors(vox_centre(800, 760, 15)[None]) == (1, 1, 1, 1.0)


def test_descriptors_counts_layers_and_cap():
    rng = np.random.default_rng(0)
    jit = lambda n: rng.uniform(-0.02, 0.02, (n, 3))
    a = vox_centre(800, 760, 14) + jit(12)          # 12 points, capped at 10
    b = vox_centre(801, 760, 14) + jit(3)           # same layer
    c = vox_centre(800, 760, 16) + jit(1)           # another layer
    n_pts, n_vox, n_lay, ppv = S.descriptors(np.concatenate([a, b, c]))
    assert (n_pts, n_vox, n_lay) == (16, 3, 2) and ppv == pytest.approx((10 + 3 + 1) / 3)


def test_grid_mask_edges_and_crop():
    x = np.array([[0, 0, -2.0], [0, 0, 3.999], [0, 0, 4.0], [75.2, 0, 0], [-75.2, 0, 0], [0, -75.2, 0], [0, 75.2, 0]])
    assert S.grid_mask(x).tolist() == [True, True, False, False, True, True, False]
    assert S.crop(np.c_[x, np.ones(len(x))]).shape == (4, 3)


def test_box_rows_empty_box_and_one_point_box():
    boxes = np.array([[30.0, 0, 0.8, 4.0, 2.0, 1.6, 0.0], [0.0, 40.0, 0.8, 5.0, 2.0, 1.6, 0.3]])
    pts = np.array([[0.2, 40.1, 0.5], [60, 60, 0.5]], np.float32)
    rows = S.box_rows(pts, boxes)
    assert rows.shape == (2, len(S.BOX_COLS))
    assert rows[0, 1:5].tolist() == [0, 0, 0, 0] and rows[0, 0] == pytest.approx(30.0)
    assert rows[1, 1:5].tolist() == [1, 1, 1, 1] and rows[1, 5] == 5.0 and rows[1, 0] == pytest.approx(40.0)
    assert S.box_rows(pts, np.zeros((0, 7))).shape == (0, len(S.BOX_COLS))
    assert S.box_rows(np.zeros((0, 3), np.float32), boxes)[:, 1].tolist() == [0, 0]


def test_in_box_indices_match_pcdet_on_the_whole_cloud():
    rng = np.random.default_rng(1)
    pts = rng.uniform([-5, -5, -1], [45, 45, 3], (20000, 3)).astype(np.float32)
    boxes = np.array([[10, 10, 0.8, 4.5, 1.9, 1.6, 0.4], [30, 5, 1.0, 12.0, 2.5, 3.2, -1.1], [3, 40, 0.8, 4.0, 2.0, 1.5, 2.9]])
    full = roiaware_pool3d_utils.points_in_boxes_cpu(pts, boxes.astype(np.float32)) > 0
    for j, idx in enumerate(S.in_box_indices(pts, boxes)):
        assert np.array_equal(idx, np.nonzero(full[j])[0])


def test_rings_and_bins():
    assert S.ring_of([0, 9.99, 10, 40, 59.99, 60, 74.99, 75, 80, -1]).tolist() == [0, 0, 1, 4, 5, 6, 6, -1, -1, -1]
    nv = [0, 1, 4, 5, 9, 10, 19, 20, 49, 50, 99, 100, 199, 200, 1000]
    assert S.bin_of(nv, S.NVOX_EDGES).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    nl = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 30]
    assert S.bin_of(nl, S.NLAY_EDGES).tolist() == [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    assert S.cell_of(80.0, 5, 5) == -1
    c = S.cell_of(55.0, 25, 6)
    assert c == (5 * len(S.NVOX_EDGES) + 4) * len(S.NLAY_EDGES) + 4 and 0 <= c < S.N_CELLS


def test_car_boxes_label_sentinels():
    b = lambda x, lab: [x, 0, 0, 4, 2, 1.5, 0, lab]
    gt = np.array([b(10, 1), b(20, -1), b(30, -99), b(40, 2), b(50, 1), b(60, -3)], np.float32)
    cars = S.car_boxes(gt)
    assert cars.shape == (2, 7) and cars[:, 0].tolist() == [10, 50]
    assert S.car_boxes(None).shape == (0, 7) and S.car_boxes(np.zeros((0, 8))).shape == (0, 7)


def test_other_vehicle_boxes_from_infos():
    names = np.array(['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'ignore', 'pedestrian', 'truck'])
    npts = np.array([5, 3, 0, 2, 1, 9, 4, 7])
    gb = np.zeros((8, 9)); gb[:, 0] = np.arange(8) * 5.0; gb[:, 3:6] = (6.0, 2.5, 3.0); gb[7, 0] = 80.0
    out = S.other_vehicle_boxes(dict(gt_names=names, num_lidar_pts=npts, gt_boxes=gb), [0.0, 0.0, 1.75])
    assert out.shape == (3, 7)
    assert out[:, 0].tolist() == [5.0, 15.0, 20.0]          # truck, trailer, construction_vehicle (bus: 0 points)
    assert np.allclose(out[:, 2], 1.75)
    assert S.other_vehicle_boxes(dict(gt_names=np.array([]), gt_boxes=np.zeros((0, 9))), [0, 0, 1.75]).shape == (0, 7)


def _box_points(box, n, rng):
    loc = rng.uniform(-0.48, 0.48, (n, 3)) * np.asarray(box[3:6])
    c, s = np.cos(box[6]), np.sin(box[6])
    return np.stack([loc[:, 0] * c - loc[:, 1] * s, loc[:, 0] * s + loc[:, 1] * c, loc[:, 2]], 1) + np.asarray(box[:3])


def _local(p, box):
    d = np.asarray(p) - np.asarray(box[:3]); c, s = np.cos(-box[6]), np.sin(-box[6])
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c, d[:, 2]], 1)


def test_relocate_geometry():
    rng = np.random.default_rng(2)
    box = np.array([12.0, -9.0, -0.6, 4.5, 1.9, 1.6, 0.7])
    pts = _box_points(box, 300, rng)
    az0 = np.arctan2(box[1], box[0])
    near_corner = lambda b: np.argmin(np.linalg.norm(box_utils.boxes_to_corners_3d(b[None])[0] - [0, 0, H], axis=1))
    for rn in S.R_NEW:
        nb, q = S.relocate(box, pts, rn)
        assert np.hypot(nb[0], nb[1]) == pytest.approx(rn, abs=1e-9)
        assert np.arctan2(nb[1], nb[0]) == pytest.approx(az0, abs=1e-12)
        assert nb[6] == box[6] and np.array_equal(nb[3:6], box[3:6])
        assert nb[2] - nb[5] / 2 == pytest.approx(0.0, abs=1e-12)                     # bottom on the ground
        assert np.allclose(_local(q, nb), _local(pts, box), atol=1e-9)              # rigid, same body coordinates
        assert near_corner(nb) == near_corner(box)                                  # the same side faces the sensor
        inside = roiaware_pool3d_utils.points_in_boxes_cpu(q.astype(np.float32), nb[None].astype(np.float32))[0]
        assert inside.all()
    with pytest.raises(ValueError):
        S.relocate([0, 0, 0, 4, 2, 1.5, 0], pts, 40.0)


def test_render_snaps_one_nearest_point_per_pixel_onto_its_ray():
    rng = np.random.default_rng(3)
    box = np.array([42.0, 30.0, 0.8, 4.6, 1.9, 1.6, 0.4])
    pts = _box_points(box, 4000, rng)
    xyz, rows, cols, k = S.render(pts, TH, NC, H)
    assert len(xyz) == len(set(zip(rows.tolist(), cols.tolist()))) > 10
    row0, col0, rng0, valid0 = RR.to_lattice(pts, H, TH, NC)
    for t in range(0, len(k), max(1, len(k) // 50)):                                # nearest of its cell
        cell = valid0 & (row0 == rows[t]) & (col0 == cols[t])
        assert rng0[k[t]] == rng0[cell].min()
    d = xyz - [0, 0, H]
    assert np.allclose(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])), TH[rows], atol=1e-9)
    az = np.arctan2(d[:, 1], d[:, 0])
    assert np.allclose(np.angle(np.exp(1j * (az - ((cols + 0.5) / NC * 2 * np.pi - np.pi)))), 0, atol=1e-9)
    assert np.allclose(np.linalg.norm(d, axis=1), rng0[k], atol=1e-9)
    pix, lrow, _ = lattice_pixels(xyz, TH, NC, H)                                   # same pixel as lattice_pixels
    assert np.array_equal(lrow, len(TH) - 1 - rows) and np.array_equal(pix, lrow * NC + cols)
    img, idx = RR.zbuffer(row0, col0, rng0, valid0, len(TH), NC)                    # same as the dense path
    dense, _, _ = RR.from_lattice(img, H, TH, NC)
    assert np.allclose(np.sort(dense, 0), np.sort(xyz, 0), atol=1e-9)


def test_render_and_coverage_of_a_donor_with_zero_points():
    nb, q = S.relocate([20.0, 5.0, 0.0, 4.5, 1.9, 1.6, 0.2], np.zeros((0, 3)), 50.0)
    xyz, rows, cols, k = S.render(q, TH, NC, H)
    assert xyz.shape == (0, 3) and rows.shape == cols.shape == k.shape == (0,)
    rc = S.box_rows(S.crop(xyz), nb[None])
    assert rc[0, 1:5].tolist() == [0, 0, 0, 0]
    cov = S.lattice_coverage(S.crop(xyz), nb[None], [np.zeros(0, np.int64)], TH, NC, H)[0]
    assert cov[0] == 0.0 and np.isnan(cov[1]) and cov[2] > 0
    assert S.phantom_count(S.crop(xyz), nb, q) == 0


def test_lattice_coverage_rows_ppp_and_occlusion():
    from lattice_sampler_precheck import silhouette
    box = np.array([30.0, 4.0, 0.8, 4.5, 1.9, 1.6, 0.3])
    inc_desc = np.sort(TH)[::-1]
    sp, srow, tin = silhouette(box, inc_desc, hs=H, n_cols=NC)
    rows_all = np.unique(srow); assert len(rows_all) >= 4
    def on_ray(p, r):
        th = inc_desc[p // NC]; az = ((p % NC) + 0.5) / NC * 2 * np.pi - np.pi
        return np.stack([r * np.cos(th) * np.cos(az), r * np.cos(th) * np.sin(az), H + r * np.sin(th)], 1)
    use = np.isin(srow, rows_all[:len(rows_all) // 2])
    pts = on_ray(sp[use], tin[use] + 0.05)
    idx = S.in_box_indices(pts, box[None])
    assert len(idx[0]) == len(pts)
    rc, ppp, vis = S.lattice_coverage(pts, box[None], idx, TH, NC, H)[0]
    assert rc == pytest.approx((len(rows_all) // 2) / len(rows_all)) and ppp == 1.0 and vis == len(rows_all)
    extra = on_ray(sp[use][:1], tin[use][:1] + 0.15)                                 # a second point in one pixel
    p2 = np.concatenate([pts, extra]); i2 = S.in_box_indices(p2, box[None])
    assert S.lattice_coverage(p2, box[None], i2, TH, NC, H)[0][1] == pytest.approx((len(pts) + 1) / len(pts))
    last = srow == rows_all[-1]                                                      # occlude the whole last row
    occ = on_ray(sp[last], tin[last] - 2.0)
    p3 = np.concatenate([pts, occ]); i3 = S.in_box_indices(p3, box[None])
    assert len(i3[0]) == len(pts)
    rc3, _, vis3 = S.lattice_coverage(p3, box[None], i3, TH, NC, H)[0]
    assert vis3 == len(rows_all) - 1 and rc3 == pytest.approx((len(rows_all) // 2) / (len(rows_all) - 1))


def test_zbuffer_keep_mask_keeps_the_nearest_per_pixel():
    d = np.array([np.cos(np.radians(-1.0)), 0.0, np.sin(np.radians(-1.0))])
    p = np.array([[0, 0, H] + 30 * d, [0, 0, H] + 31 * d, [0, 0, H] + 30 * np.array([0, 1, 0]) * 1.0 + [0, 0, -0.5]])
    assert S.zbuffer_keep_mask(p, TH, NC, H).tolist() == [True, False, True]


def test_phantom_count():
    box = np.array([50.0, 0.0, 0.8, 4.5, 1.9, 1.6, 0.0])
    donor = np.array([[50.0, 0.0, 0.8], [53.0, 0.0, 0.8]])
    rendered = np.array([[50.1, 0.0, 0.8], [53.1, 0.0, 0.8], [56.0, 0.0, 0.8]], np.float32)
    assert S.phantom_count(rendered, box, donor) == 1
    assert S.phantom_count(np.zeros((0, 3), np.float32), box, donor) == 0


def test_covered_a2_and_reach():
    s_src = np.zeros(S.N_CELLS); s_tgt = np.zeros(S.N_CELLS)
    c1, c2, c3, c4 = (int(S.cell_of(r, nv, nl)) for r, nv, nl in ((55, 25, 6), (55, 60, 8), (65, 3, 2), (65, 8, 3)))
    s_src[[c1, c2, c3, c4]] = [100, 40, 100, 0]; s_tgt[[c1, c2, c3, c4]] = [1000, 10, 0, 500]
    share = S.covered(s_src, s_tgt, 'share')
    # S_src = 240, S_tgt = 1510: rho_share(c1) = (100/240) / (1000/1510) = 0.63; c2 below the floor; c3 has no target
    assert share[c1] and not share[c2] and share[c3] and not share[c4]
    count = S.covered(s_src, s_tgt, 'count')
    assert not count[c1] and count[c3]                       # 100 / 1000 = 0.1 < 0.2
    run = S.covered(s_src, s_tgt, 'run', scale_src=20, scale_tgt=7)
    assert run[c1]                                           # 2000 / 7000 = 0.29
    assert S.uncovered_share(np.array([c1, c4, c4, -1]), share) == pytest.approx(2 / 3)
    lacks = S.a2_reading([(0.8, 0.2), (0.7, 0.3), (0.2, 0.1)])
    assert lacks[0].startswith('the source lacks cars like') and lacks[1] == [True, True, False]
    sub = S.a2_reading([(0.8, 0.6), (0.7, 0.5), (0.9, 0.2)])
    assert sub[0].startswith('the source lacks such cars at range') and sub[2] == [False, False, True]
    has = S.a2_reading([(0.3, 0.1), (0.6, 0.1), (np.nan, np.nan)])
    assert has[0].startswith('the source has them')
    row = lambda r, nv, nl: [r, nv, nv, nl, 1.0, 4.5, 0, 1]
    misses = np.array([row(55, 25, 6), row(55, 25, 6), row(55, 3, 2), row(65, 25, 6)])
    op = np.array([row(52, 30, 5)] * 99 + [row(58, 3, 2)])          # the (3, 2) cell holds exactly 1%
    assert S.reach(misses, op, 50, 60) == pytest.approx(1.0)
    assert S.reach(misses, op[:99], 50, 60) == pytest.approx(2 / 3)
    assert np.isnan(S.reach(misses, op, 60, 75))
