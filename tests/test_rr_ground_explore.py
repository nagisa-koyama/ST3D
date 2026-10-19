"""tools/analysis/rr_ground_explore.py (CPU; experiments_md 20261018_02 §11): the leave-one-out cloud, the regions (ring,
radius, annular sectors), raw and cell percentiles, the plane fallback, the source-label variants, the D1 v4 reference, the
frame sample and the selection rule, on a toy with a sloped ground, a car and a wall.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import rr_ground_explore as GE  # noqa: E402
from pcdet.datasets.augmentor import relocate_render as R  # noqa: E402

SLOPE = 0.05


def ground_z(x, y):
    return SLOPE * np.asarray(x) + 0.0 * np.asarray(y)


def toy():
    """Sloped ground over x 8..32, y -12..12 (0.25 m), a car at (20, 0) on the ground, a wall at y = 3.5 (x 16..24,
    up to 3 m above the ground)."""
    g = np.mgrid[8:32:0.25, -12:12:0.25].reshape(2, -1).T
    ground = np.c_[g, ground_z(g[:, 0], g[:, 1])]
    box = np.array([20.0, 0.0, ground_z(20.0, 0.0) + 0.75, 4.5, 1.9, 1.5, 0.0])
    rng = np.random.default_rng(0)
    loc = rng.uniform([-2.25, -0.95, -0.75], [2.25, 0.95, 0.75], (400, 3))
    carp = loc + box[:3]
    wx, wz = np.mgrid[16:24:0.2, 0.2:3.0:0.2].reshape(2, -1)
    wall = np.c_[wx, np.full(len(wx), 3.5), ground_z(wx, 0) + wz]
    return np.concatenate([ground, carp, wall]), box, len(ground), len(carp)


def test_leave_one_out_removes_the_car_and_its_grown_footprint_only():
    P, box, ng, nc = toy()
    keep = GE.leave_one_out(P, box, R._BevGrid(P))
    d = R.footprint_distance(P[:, :2], box)
    assert not keep[ng:ng + nc].any()                       # every in-box point
    assert np.array_equal(keep, d > GE.D_LO)                 # exactly the 0.5 m grown footprint, at any height
    assert np.array_equal(GE.leave_one_out(P, box, None), keep)


def test_e1_e2_raw_equal_the_loader_and_the_region_masks():
    P, box, ng, nc = toy()
    X = P[GE.leave_one_out(P, box)]
    pl = R.ground_plane(X, 40.0)
    assert pl is not None and abs(pl[0] - SLOPE) < 0.01
    est = GE.estimate_car(X, box, np.zeros((0, 7)), pl, R.ground_plane(X, 75.2))
    m = GE.region_masks(X, box)
    for g in ('E1', 'E2'):
        for p in GE.PCTS:
            z, fb = est[f'{g}-raw-p{p}']
            assert not fb and np.isclose(z, np.percentile(X[m[g], 2], p))
    d = R.footprint_distance(X[:, :2], box)
    assert np.array_equal(m['E1'], (d > 0.5) & (d <= 3.0))
    assert np.array_equal(m['E2'], (d > 0.5) & (np.hypot(X[:, 0] - 20, X[:, 1]) <= 10.0))
    # the wall stands 2.55 m from the car's side: inside E1 (<= 3 m) and above the ground, so a low percentile barely
    # moves; on the 0.05 slope the ring's p10 lies below the centre's ground (the ring reaches 5.25 m down the slope)
    nowall = X[:ng - int((~GE.leave_one_out(P[:ng], box)).sum())]
    e2 = GE.estimate_car(nowall, box, np.zeros((0, 7)), pl, pl)
    gt = box[2] - box[5] / 2
    assert abs(est['E1-raw-p10'][0] - e2['E1-raw-p10'][0]) < 0.05 and est['E1-raw-p10'][0] < gt
    assert est['E2-raw-p30'][0] > est['E2-raw-p5'][0]
    assert abs(est['E0-40'][0] - gt) < 0.02                  # the plane recovers the slope


def test_e3_sector_bounds():
    P, box, ng, nc = toy()
    X = P[GE.leave_one_out(P, box)]
    m = GE.region_masks(X, box)
    Rg = np.hypot(X[:, 0], X[:, 1]); az = np.arctan2(X[:, 1], X[:, 0])
    d = R.footprint_distance(X[:, :2], box)
    for dr, w in GE.E3_SHAPES:
        half = w / 20.0
        exp = (d > 0.5) & (Rg >= 20 - dr) & (Rg <= 20 + dr) & (np.abs(az) <= half)
        assert np.array_equal(m[f'E3-dr{dr:g}-w{w:g}'], exp)
    # a car near the sensor: the half-width is capped at pi (the whole annulus)
    near = np.array([1.0, 0.0, 0.75, 4.5, 1.9, 1.5, 0.0])
    mm = GE.region_masks(X, near)['E3-dr2-w10']
    assert np.array_equal(mm, (R.footprint_distance(X[:, :2], near) > 0.5) & (Rg >= -1.0) & (Rg <= 3.0))


def test_cell_values_are_the_second_lowest_of_cells_with_three_points():
    pts = np.array([[0.2, 0.2, 3.0], [0.4, 0.4, 1.0], [0.6, 0.6, 2.0],       # cell (0, 0): 3 points -> 2.0
                    [1.5, 0.5, 5.0], [1.6, 0.5, 4.0]])                         # cell (1, 0): 2 points -> dropped
    assert np.array_equal(GE.cell_values(pts), [2.0])
    assert len(GE.cell_values(np.zeros((0, 3)))) == 0


def test_fallback_to_the_plane_when_too_few_points_or_cells():
    pl = np.array([0.0, 0.0, -0.3])
    box = np.array([50.0, 0.0, 0.5, 4.5, 1.9, 1.5, 0.0])
    X = np.array([[52.6, 0.0, 0.0], [47.0, 0.0, 0.1]] * 3, np.float64)        # 6 points in E1: < 10, 1 cell each
    est = GE.estimate_car(X, box, np.zeros((0, 7)), pl, None)
    for k in GE.e14_names():
        assert est[k] == (-0.3, True), k
    assert est['E0-40'] == (-0.3, False) and est['E0-75'] == (-0.3, True)     # no 4-75.2 m plane: counted
    assert est['D1v4ref'] == (-0.3, True)
    assert np.isnan(est['E5a'][0]) and est['E5a'][1] and est['E5b-R10'][1]


def test_e5_source_label_heights():
    pl = np.array([0.01, 0.0, 0.0])                       # z = 0.01 x
    box = np.array([20.0, 0.0, 0.75, 4.5, 1.9, 1.5, 0.0])
    oth = np.array([[25.0, 0.0, 0.25 + 0.75, 4.5, 1.9, 1.5, 0.0],       # bottom 0.25, plane 0.25 -> residual 0.00
                    [20.0, 8.0, 0.40 + 0.75, 4.5, 1.9, 1.5, 0.0],       # bottom 0.40, plane 0.20 -> +0.20
                    [20.0, -15.0, 0.50 + 0.75, 4.5, 1.9, 1.5, 0.0],     # bottom 0.50, plane 0.20 -> +0.30
                    [60.0, 0.0, 9.0, 4.5, 1.9, 1.5, 0.0]])              # 40 m away: beyond every radius
    est = GE.estimate_car(np.zeros((0, 3)), box, oth, pl, pl)
    assert np.isclose(est['E5a'][0], 0.20 + 0.20) and not est['E5a'][1]
    w = 1 / np.array([5.0, 8.0])
    assert np.isclose(est['E5b-R10'][0], (w * [0.25, 0.40]).sum() / w.sum())
    w = 1 / np.array([5.0, 8.0, 15.0])
    assert np.isclose(est['E5b-R20'][0], (w * [0.25, 0.40, 0.50]).sum() / w.sum())


def test_d1v4_reference_equals_the_gate_on_the_full_cloud():
    P, box, ng, nc = toy()
    X = P[GE.leave_one_out(P, box)]
    pl = R.ground_plane(X, 40.0)
    est = GE.estimate_car(X, box, np.zeros((0, 7)), pl, pl)
    v = R.GroundCells(X, 1.0, 3).local(20.0, 0.0, 5.0)
    assert len(v) >= 3 and np.isclose(est['D1v4ref'][0], np.percentile(v, 20)) and not est['D1v4ref'][1]


def test_frame_rows_cars_only_within_range_and_planes_refitted():
    P, box, ng, nc = toy()
    far = box.copy(); far[0] = 80.0
    ped = box.copy(); ped[1] = -6.0
    boxes = np.stack([box, far, ped])
    rows, dropped = GE.frame_rows(P, boxes, np.array(['Car', 'Car', 'Pedestrian']))
    assert len(rows) == 1 and dropped == 0                   # the 80 m car and the pedestrian are not evaluated
    V = GE.variant_names(); nv = len(V)
    r, gtb, nrem = rows[0][:3]
    assert np.isclose(r, 20.0) and np.isclose(gtb, box[2] - box[5] / 2) and nrem >= nc
    e = dict(zip(V, rows[0][3:3 + nv])); f = dict(zip(V, rows[0][3 + nv:]))
    # the only other Car is 60 m away and the pedestrian 6 m away is not a car: every E5 variant falls back
    assert f['E5a'] == 1.0 and f['E5b-R10'] == 1.0 and f['E5b-R20'] == 1.0 and np.isnan(e['E5a'])
    X = P[GE.leave_one_out(P, box)]
    pl = R.ground_plane(X, 40.0)
    direct = GE.estimate_car(X, box, np.zeros((0, 7)), pl, R.ground_plane(X, 75.2))
    for k in GE.e14_names() + ['E0-40', 'E0-75', 'D1v4ref']:
        assert np.isclose(e[k], direct[k][0]) and f[k] == float(direct[k][1]), k
    assert f['E1-raw-p10'] == 0.0 and abs(e['E0-40'] - gtb) < 0.02


def test_sample_frames_disjoint_from_the_gate_and_deterministic():
    g = GE.gate_frames(15695)
    assert g[1] == 78 and len(g) == 200
    s = GE.sample_frames(15695)
    assert len(s) == 400 and len(np.unique(s)) == 400 and not np.isin(s, g).any()
    assert np.array_equal(s, GE.sample_frames(15695)) and np.all(np.diff(s) > 0)


def test_selection_rule_ties_within_one_point_go_to_the_smaller_median():
    t = {'a': dict(gt05=0.200, med=0.30), 'b': dict(gt05=0.209, med=0.20), 'c': dict(gt05=0.211, med=0.10)}
    assert GE.select(t) == 'b'
    t['b']['gt05'] = 0.215
    assert GE.select(t) == 'a'


def test_resolve_e5_fallback_takes_the_best_e1_e4_variant():
    V = GE.variant_names(); nv = len(V)
    n = 30
    rows = np.zeros((n, 5 + 2 * nv)); rows[:, 2] = 50.0                     # all at 50 m, GT bottom 0
    for j, k in enumerate(V):
        rows[:, 5 + j] = 2.0                                                # every estimate 2 m off ...
    best = 'E3-dr5-w10-cell-p20'
    rows[:, 5 + V.index(best)] = 0.01                                       # ... except one E1-E4 variant
    rows[:10, 5 + V.index('E5a')] = np.nan; rows[:10, 5 + nv + V.index('E5a')] = 1.0
    est, fb, b14 = GE.resolve(rows, V)
    assert b14 == best
    assert np.allclose(est['E5a'][:10], 0.01) and np.allclose(est['E5a'][10:], 2.0) and fb['E5a'][:10].all()
