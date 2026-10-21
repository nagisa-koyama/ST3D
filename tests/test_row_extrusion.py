"""pcdet/datasets/augmentor/row_extrusion.py (CPU; experiments_md 20261021_02, T05 §5 RX; bug-prevention checklist).

Pins, before any gate number: an empty cloud, one point, no boxes, a box with no points; a wall + car + ground toy
(extrusion stops at the box top, at 0.665 deg outside and at S; the shadow of a farther ground return, a nearer one kept;
nothing enters a box from outside); ties to the lower row; features copied from the seed; the arms (L nothing outside, W no
box); G1's ground fill; the sentinels (masked, ignore-mask, negative and -99 class labels, a misleading label column: never
Car boxes, always boxes); the transplant (nearest library relh, a zero-hit row dropped, cells and aspects); S equal to E5a;
the in-box test equal to roiaware's; the silhouette relh equal to box_pattern_visual's; run lengths and medians; the hook
(absent by default, training only, the global RNG untouched, later augmentation identical); q's counts; the configs (one
variable against 27471's loader and RR's rr7a lineage; no other config sets the key).
"""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

from pcdet.datasets.augmentor import relocate_render as rr  # noqa: E402
from pcdet.datasets.augmentor import row_extrusion as RX  # noqa: E402

TH = np.radians(np.linspace(-12.0, 2.0, 57))             # 0.25 deg rows, ascending
NC, H = 1440, 2.184                                        # 0.25 deg columns
LAT = dict(thetas=TH, n_cols=NC, height=H)
PLANE0 = np.array([0.0, 0.0, 0.0])


def row_of(deg):
    return int(np.argmin(np.abs(TH - np.radians(deg))))


def col_az(c):
    return (c + 0.5) / NC * 2 * np.pi - np.pi


def on_ray(rho, row, col, inten=1.0, t=0.0):
    """A point on the centre ray of pixel (row, col) at horizontal range rho."""
    az = col_az(col)
    return [rho * np.cos(az), rho * np.sin(az), H + rho * np.tan(TH[row]), inten, t]


def at_x(x, row, col, inten=1.0, t=0.0):
    """A point on the centre ray of (row, col) where it crosses the vertical plane at x."""
    return on_ray(x / np.cos(col_az(col)), row, col, inten, t)


class OneCarLibrary(RX.Library):
    """Every cell holds one car with the given rows."""

    def __init__(self, relh=(0.5,), hit=(1.0,), q=(1.0,) * 5):
        n = len(relh)
        super().__init__(np.arange(10), np.zeros(10, np.int64), np.full(10, n), np.asarray(relh, float),
                         np.asarray(hit, float), np.asarray(q, float))


CAR = np.array([30.0, 0.0, 0.75, 4.0, 2.0, 1.5, 0.0])       # x 28..32, y -1..1, z 0..1.5
PED = None
CAR_COLS = list(range(714, 726))                            # azimuth -1.4..1.4 deg: the car's front face at x = 28.05
WALL_COLS = list(range(740, 751))
WALL2_COLS = list(range(800, 806))
POLE_COL = 780


def toy():
    """Wall + car + ground (+ a low wall, a pole under a pedestrian box, a kerb and a seen-through ground return)."""
    pts = []
    for c in CAR_COLS:                                      # two HDL-like lines on the car's face
        for deg in (-3.5, -2.5):
            pts.append(at_x(28.05, row_of(deg), c, inten=10 + c, t=0.05))
    for c in WALL_COLS:                                     # a wall at x = 50, two lines
        for deg in (-1.0, 0.0):
            pts.append(at_x(50.0, row_of(deg), c, inten=200 + c + (1000 if deg == 0.0 else 0), t=0.1))
    for c in WALL2_COLS:                                    # a low wall, one line at z ~ 0.37
        pts.append(on_ray(52.0, row_of(-2.0), c, inten=300 + c))
    pts.append(on_ray(35.0, row_of(-2.5), POLE_COL, inten=400))        # a pole point, z = 0.656
    pts.append(on_ray(27.5, row_of(-4.25), CAR_COLS[0], inten=500))    # kerb in front of the car: z 0.14, nearer
    pts.append(on_ray(2.184 / np.tan(np.radians(3.0)), row_of(-3.0), CAR_COLS[1], inten=600))  # ground seen through
    pts = np.array(pts, np.float32)
    az = col_az(POLE_COL)
    ped = np.array([35 * np.cos(az), 35 * np.sin(az), 1.2, 0.6, 0.6, 0.6, 0.0])   # z 0.9..1.5 above the pole
    boxes = np.stack([CAR, ped])
    return pts, boxes, np.array([True, False])


def run(arm='H', ground='G0', pts=None, boxes=None, car=None, lib=None, plane=PLANE0, seed=0):
    if pts is None:
        pts, boxes, car = toy()
    return RX.row_extrusion(pts, boxes, car, arm, ground, LAT, lib, np.random.default_rng(seed), plane=plane,
                            keep_record=True)


def pix_of(xyz):
    from pcdet.datasets.rerender_utils import to_lattice
    row, col, _, valid = to_lattice(np.asarray(xyz, np.float64), H, TH, NC)
    assert valid.all()
    return row, col


# ---------------------------------------------------------------------------------------------- edge cases
def test_empty_cloud():
    out, rec = RX.row_extrusion(np.zeros((0, 5), np.float32), np.zeros((0, 7)), np.zeros(0, bool), lat=LAT,
                                plane=PLANE0)
    assert out.shape == (0, 5) and rec['n_new'] == 0 and rec.get('no_points')


def test_one_point_extrudes_within_half_a_line_outside():
    p = np.array([on_ray(30.0, row_of(-2.0), 700, inten=7, t=0.3)], np.float32)
    out, rec = run(pts=p, boxes=np.zeros((0, 7)), car=np.zeros(0, bool))
    new = out[1:]
    rows, cols = pix_of(new[:, :3])
    assert sorted(rows) == sorted(row_of(d) for d in (-2.5, -2.25, -1.75, -1.5))
    assert (cols == 700).all()
    assert np.allclose(new[:, :2], p[0, :2]) and np.all(new[:, 3] == 7) and np.all(new[:, 4] == np.float32(0.3))
    assert np.array_equal(out[0], p[0])


def test_no_plane_leaves_the_frame_unchanged():
    p = np.array([on_ray(30.0, row_of(-2.0), 700)], np.float32)
    out, rec = RX.row_extrusion(p, np.zeros((0, 7)), np.zeros(0, bool), lat=LAT, plane=None)
    assert rec['no_plane'] and np.array_equal(out, p)


def test_no_boxes_h_equals_w():
    pts, _, _ = toy()
    a, _ = run('H', pts=pts, boxes=np.zeros((0, 7)), car=np.zeros(0, bool))
    b, _ = run('W', pts=pts, boxes=np.zeros((0, 7)), car=np.zeros(0, bool))
    assert np.array_equal(a, b)


def test_a_box_with_no_points_changes_nothing_but_draws_its_library_car():
    p = np.array([at_x(50.0, row_of(-1.0), c) for c in WALL_COLS], np.float32)
    empty_car = np.array([[20.0, -20.0, 0.75, 4.0, 2.0, 1.5, 0.3]])
    a, ra = run(pts=p, boxes=np.zeros((0, 7)), car=np.zeros(0, bool), lib=OneCarLibrary())
    b, rb = run(pts=p, boxes=empty_car, car=np.array([True]), lib=OneCarLibrary())
    assert np.array_equal(a, b) and len(rb['lib_pick']) == 1 and len(ra['lib_pick']) == 0


# -------------------------------------------------------------------------------------------- the toy scene
def test_car_extrusion_fills_the_box_height_and_stops_at_its_top():
    out, rec = run()
    t = rec['new_type']; nx = rec['new_xyz']
    car_new = nx[t == RX.T_CAR]
    rows, cols = pix_of(car_new)
    assert set(cols) <= set(CAR_COLS)
    assert rr.points_in_box(car_new.astype(np.float32), CAR).all()
    # rows -4.25 .. -1.5 deg (z 0.10 .. 1.45) in every car column, except the seed rows and the kerb pixel (nearer return)
    want = {row_of(d) for d in np.arange(-4.25, -1.4, 0.25)} - {row_of(-3.5), row_of(-2.5)}
    for c in CAR_COLS:
        got = set(rows[cols == c])
        exp = want - ({row_of(-4.25)} if c == CAR_COLS[0] else set())
        assert got == exp, (c, sorted(got), sorted(exp))
    assert row_of(-1.25) not in rows and row_of(-4.5) not in rows        # above the roof, below the bottom
    # the inside rule has no elevation bound: -1.5 deg is 1.0 deg above the upper line
    assert row_of(-1.5) in rows


def test_wall_extrusion_stops_at_half_a_line_and_ties_go_to_the_lower_row():
    out, rec = run()
    nx = rec['new_xyz']; t = rec['new_type']
    rows, cols = pix_of(nx)
    for c in WALL_COLS:
        got = set(rows[(cols == c) & (t == RX.T_BG)])
        assert got == {row_of(d) for d in (-1.5, -1.25, -0.75, -0.5, -0.25, 0.25, 0.5)}, (c, sorted(got))
    # the tie at -0.5 deg (2 rows from both lines) takes the lower (-1.0 deg) seed: its features
    k = np.nonzero((cols == WALL_COLS[0]) & (rows == row_of(-0.5)))[0]
    src = out[(rec['keep_src'].sum() + np.cumsum(rec['new_kept']) - 1)[k]]
    seed_low = toy()[0][len(CAR_COLS) * 2]                  # the wall's first point: column WALL_COLS[0], -1.0 deg
    assert src[0, 3] == seed_low[3] == 200 + WALL_COLS[0] and np.allclose(src[0, :2], seed_low[:2])
    assert np.isclose(np.degrees(np.arctan2(seed_low[2] - H, np.hypot(*seed_low[:2]))), -1.0, atol=1e-4)


def test_outside_extrusion_never_goes_below_s():
    out, rec = run()
    nx = rec['new_xyz']; t = rec['new_type']
    rows, cols = pix_of(nx)
    for c in WALL2_COLS:
        got = set(rows[(cols == c) & (t == RX.T_BG)])
        assert got == {row_of(d) for d in (-2.25, -1.75, -1.5)}, (c, sorted(got))   # -2.5 deg would be z = -0.09
    assert (nx[t == RX.T_BG, 2] >= 0).all()


def test_nothing_enters_a_box_from_outside():
    out, rec = run()
    pts, boxes, _ = toy()
    nx = rec['new_xyz']; t = rec['new_type']
    rows, cols = pix_of(nx)
    got = set(rows[cols == POLE_COL])
    assert got == {row_of(d) for d in (-3.0, -2.75, -2.25)}, sorted(got)          # -2.0 deg lands in the pedestrian box
    assert not RX.points_in_boxes(nx[t == RX.T_BG], boxes).any()
    new_out = out[rec['keep_src'].sum():, :3]
    assert len(new_out) == rec['n_kept'] and not rr.points_in_box(new_out, boxes[1]).any()


def test_shadow_removes_a_farther_ground_return_and_keeps_a_nearer_one():
    pts, boxes, car = toy()
    out, rec = run()
    assert rec['n_shadow'] == 1
    gone = pts[~rec['keep_src']]
    assert len(gone) == 1 and gone[0, 3] == 600                                    # the seen-through ground return
    assert (out[:, 3] == 500).sum() == 1                                           # the kerb stays
    rows, cols = pix_of(rec['new_xyz'])
    assert ((cols == CAR_COLS[1]) & (rows == row_of(-3.0))).sum() == 1             # its pixel now holds a car point
    assert not ((cols == CAR_COLS[0]) & (rows == row_of(-4.25))).any()             # the kerb's pixel is not filled


def test_features_are_copied_from_the_seed():
    out, rec = run()
    pts = toy()[0]
    new = out[rec['keep_src'].sum():]
    for p in new:
        same_xy = np.nonzero(np.isclose(pts[:, 0], p[0]) & np.isclose(pts[:, 1], p[1]))[0]
        assert len(same_xy) and p[3] in pts[same_xy, 3] and p[4] in pts[same_xy, 4]


def test_seed_rows_ties_and_self():
    obj = np.zeros((9, 3), bool)
    obj[2, 0] = obj[6, 0] = True; obj[5, 1] = True
    s = RX.nearest_seed_rows(obj)
    assert list(s[:, 0]) == [2, 2, 2, 2, 2, 6, 6, 6, 6]          # row 4 is 2 from both: the lower (2)
    assert list(s[:, 1]) == [5] * 9 and list(s[:, 2]) == [-1] * 9
    assert s[2, 0] == 2 and s[6, 0] == 6                         # an object pixel is its own seed


def test_arm_l_extrudes_only_in_car_boxes():
    out, rec = run('L')
    assert set(rec['new_type'].tolist()) <= {RX.T_CAR} and rec['n_new_type'][RX.T_CAR] > 0
    _, rh = run('H')
    assert np.array_equal(rec['new_xyz'], rh['new_xyz'][rh['new_type'] == RX.T_CAR])


def test_arm_w_uses_no_box():
    out, rec = run('W')
    assert set(rec['new_type'].tolist()) <= {RX.T_BG}
    rows, cols = pix_of(rec['new_xyz'])
    assert row_of(-2.0) in set(rows[cols == POLE_COL])           # into the pedestrian box: W has no boxes
    car_rows = set(rows[cols == CAR_COLS[2]])                    # the outside rule on the car: +-0.665 deg only
    assert car_rows == {row_of(d) for d in (-4.0, -3.75, -3.25, -3.0, -2.75, -2.25, -2.0)}, sorted(car_rows)
    assert rec['lib_pick'] == []


def test_q_keeps_the_share_of_new_points():
    pts, boxes, car = toy()
    lib = OneCarLibrary(q=(0.3,) * 5)
    keep = []
    for s in range(30):
        _, rec = run('W', pts=pts, boxes=boxes, car=car, lib=lib, seed=s)
        keep.append(rec['new_kept'].mean())
    assert abs(np.mean(keep) - 0.3) < 0.05


# ------------------------------------------------------------------------------------------------------ G1
def test_g1_fills_between_two_ground_pixels_and_beyond_the_last_up_to_75_2_m():
    c, c2 = 300, 310
    g = lambda row, col, i: on_ray(H / np.tan(-TH[row]), row, col, inten=i)
    pts = [g(20, c, 1), g(24, c, 2), g(20, c2, 3), g(24, c2, 4)]
    pts.append(on_ray((H - 0.5) / np.tan(-TH[10]), 10, c2, inten=5))           # an object pixel below, column c2
    pts = np.array(pts, np.float32)
    out, rec = run('H', 'G1', pts=pts, boxes=np.zeros((0, 7)), car=np.zeros(0, bool))
    t = rec['new_type'] == RX.T_G1
    nx = rec['new_xyz'][t]
    rows, cols = pix_of(nx)
    assert set(rows[cols == c]) == set(range(21, 24)) | set(range(25, row_of(-1.75) + 1))
    assert set(rows[cols == c2]) == set(range(21, 24))                         # beyond blocked by the object pixel
    assert np.allclose(nx[:, 2], 0.0, atol=1e-4) and np.hypot(nx[:, 0], nx[:, 1]).max() <= 75.2
    new = out[len(pts) - rec['n_shadow']:]
    g1_feat = new[-int(t.sum()):, 3]
    assert set(g1_feat.tolist()) == {1.0, 2.0, 3.0}            # between: the lower ground pixel; beyond: the last one


def test_g0_has_no_ground_points():
    _, rec = run('H', 'G0')
    assert rec['n_new_type'][RX.T_G1] == 0


# ------------------------------------------------------------------------------------------------- sentinels
def test_sentinel_boxes_are_boxes_but_never_car_boxes():
    names = np.array(['Car', 'Car', 'Car', 'Car', 'Pedestrian'])
    boxes = np.tile(np.r_[CAR, 1.0], (5, 1))                    # column 7 says 'Car' (label 1) for every row
    for k in range(1, 5):
        boxes[k, 1] = 6.0 * k
    d = dict(gt_boxes=boxes, gt_names=names, gt_boxes_mask=np.array([True, False, True, True, True]),
             gt_classes=np.array([1, 1, -1, -99, 2]))
    car = rr.gt_car_mask(d)
    assert car.tolist() == [True, False, False, False, False]
    d['gt_boxes_ignore_mask'] = np.array([True, False, False, False, False])
    assert not rr.gt_car_mask(d).any()
    pts = []
    for k in range(5):                                          # points at each box's centre range, z 0.5-0.9
        az = np.arctan2(boxes[k, 1], boxes[k, 0])
        cols = int(np.floor((az + np.pi) / (2 * np.pi) * NC))
        pts += [on_ray(np.hypot(boxes[k, 0], boxes[k, 1]), row_of(-2.5), cc) for cc in (cols - 1, cols, cols + 1)]
    pts = np.array(pts, np.float32)
    for k in range(5):
        assert rr.points_in_box(pts[3 * k:3 * k + 3], boxes[k]).all()
    out, rec = run('H', pts=pts, boxes=boxes, car=car, lib=OneCarLibrary())
    nb = rec['new_box']; t = rec['new_type']
    assert set(nb[t == RX.T_CAR]) == {0}                        # only the real Car box is transplanted
    assert {1, 2, 3, 4} <= set(nb[t == RX.T_OTHER])             # sentinels still bound the extrusion as boxes
    assert [j for j, _ in rec['lib_pick']] == [0]


# ------------------------------------------------------------------------------------------------ transplant
def test_transplant_takes_the_nearest_library_relh_and_drops_rows_without_a_hit():
    ur, urelh = RX.silhouette_row_relh(CAR, LAT)
    assert len(ur) >= 4 and np.all(np.diff(urelh) >= -1e-9)    # higher rows enter higher on the box
    lr = np.array([0.0, 0.4, 0.6, 1.0]); lh = np.array([0.0, 0.5, 0.25, 1.0])
    z = H + 28.1 * np.tan(TH[ur])
    p = RX.transplant_prob(ur, z, CAR, lr, lh, LAT)
    exp = lh[np.argmin(np.abs(urelh[:, None] - lr[None]), 1)]
    assert np.array_equal(p, exp)
    out, rec = run(lib=OneCarLibrary(relh=(0.5,), hit=(0.0,)))
    assert rec['n_new_type'][RX.T_CAR] > 0 and rec['n_kept_type'][RX.T_CAR] == 0
    out, rec = run(lib=OneCarLibrary(relh=(0.5,), hit=(1.0,)))
    assert rec['n_kept_type'][RX.T_CAR] == rec['n_new_type'][RX.T_CAR]


def test_cells_range_clamping_and_aspect_bounds():
    def box(r, alpha_deg, phi=0.3):
        return np.array([r * np.cos(phi), r * np.sin(phi), 0.8, 4, 2, 1.5, phi + np.radians(alpha_deg)])
    assert RX.cell_of(box(10, 0)) == 0 and RX.cell_of(box(80, 0)) == 8 and RX.cell_of(box(74.9, 90)) == 9
    assert RX.cell_of(box(25.01, 0)) == 2 and RX.cell_of(box(24.99, 0)) == 0 and RX.cell_of(box(45, 10)) == 4
    for a, side in ((90.0, 1), (10.0, 0), (170.0, 0), (-100.0, 1), (-20.0, 0)):
        assert RX.cell_of(box(30, a)) % 2 == side, a
    assert RX.aspect_side(np.radians([45.0, 135.0, -45.0, -135.0])).all()
    assert not RX.aspect_side(np.radians([44.9, 135.1, 0.0, 180.0])).any()


# ------------------------------------------------------------------------------------- shared definitions
def test_surface_equals_e5a_at_box_centres():
    rng = np.random.default_rng(0)
    plane = np.array([0.02, -0.01, 0.1])
    cars = np.array([[10, 5, 0.9, 4, 2, 1.6, 0], [25, -8, 0.7, 4, 2, 1.5, 1], [70, 40, 1.4, 4, 2, 1.6, 2]], float)
    S = RX.Surface(plane, cars)
    xyz = rng.uniform(-50, 50, (100, 3))
    grid = rr._BevGrid(xyz[:, :2])
    for q in ([12, 0], [40, 20], [-60, -60]):
        box = np.array([q[0], q[1], 0, 4, 2, 1.5, 0.0])
        z, kind = rr.e5a_bottom(xyz, grid, box, plane, cars)
        if kind == 0:
            assert np.isclose(S(np.array([q[0]]), np.array([q[1]]))[0], z)
        else:
            assert np.isclose(S(np.array([q[0]]), np.array([q[1]]))[0], rr.plane_z(plane, q[0], q[1]))


def test_in_box_test_equals_relocate_render_and_roiaware():
    import torch
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    rng = np.random.default_rng(1)
    pts = rng.uniform([-5, -5, -1], [5, 5, 2], (20000, 3)).astype(np.float32)
    boxes = np.array([[0, 0, 0.5, 4, 2, 1.5, 0.3], [2, -1, 0.2, 1, 1, 1, -1.0], [-3, 3, 1, 2, 3, 2, 2.0]], np.float32)
    m = RX.points_in_boxes(pts, boxes)
    for j, b in enumerate(boxes):
        assert np.array_equal(m[:, j], rr.points_in_box(pts, b))
    ref = roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(pts), torch.from_numpy(boxes)).numpy().T > 0
    assert np.array_equal(m, ref)


def test_silhouette_relh_equals_box_pattern_visual():
    import box_pattern_visual as BV
    lat = RX.lattice()
    th = lat['thetas']; R = len(th)
    pose = BV.sensor_pose('rr', height=lat['height'])
    for box in ([30, 4, 0.8, 4.4, 1.9, 1.6, 0.4], [55, -20, 0.9, 4.8, 2.0, 1.5, 2.0], [-12, 9, 0.7, 4, 1.8, 1.5, -1.0]):
        box = np.array(box, float)
        ur, urelh = RX.silhouette_row_relh(box, lat)
        sil, tin, tout, ent = BV.silhouette_pixels(box, pose, th[::-1], 'rr', lat['n_cols'], full=True)
        rows = R - 1 - sil // lat['n_cols']
        relh = np.clip((ent[:, 2] - (box[2] - box[5] / 2)) / box[5], 0, 1)
        u2 = np.unique(rows)
        assert np.array_equal(ur, u2)
        assert np.allclose(urelh, [np.median(relh[rows == r]) for r in u2], atol=1e-9)


def test_run_lengths_and_medians():
    has = np.array([[1, 0], [1, 1], [1, 1], [0, 1], [1, 1]], bool)
    rng = np.array([[10, 0], [10.2, 5], [10.6, 5.1], [0, 5.2], [3, 9]], float)
    rl = RX.run_lengths(has, rng)
    assert rl[:, 0].tolist() == [2, 2, 1, 0, 1] and rl[:, 1].tolist() == [0, 3, 3, 3, 1]
    rng2 = np.random.default_rng(0)
    for n in (1, 2, 7, 10):
        v = rng2.integers(0, 9, n)
        assert RX.hist_median(np.bincount(v, minlength=12)) == np.median(v)
    assert np.isnan(RX.hist_median(np.zeros(5)))


def test_q_counts_on_a_toy_range_image():
    from rx_build_library import q_counts
    has = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], bool)
    rng = np.array([[10, 10, 10], [10, 0, 30], [10.2, 10.5, 10], [0, 10, 10]], float)
    rho = np.full(has.shape, 30.0)
    ok = np.ones(has.shape, bool); ok[0, 2] = False
    num, den = q_counts(has, rng, rho, ok)
    # row 1: col 0 (10 vs 10.2, returns), col 1 (10 vs 10.5: > 0.3, no), col 2 (row 0 not ok); row 2: col 2 (30 vs 10: no),
    # col 0 (row 3 empty), col 1 (row 1 empty)
    assert den[1] == 1 and num[1] == 1 and den.sum() == 1


# ---------------------------------------------------------------------------------------------------- hook
def _library_file(tmp_path, q=1.0):
    p = tmp_path / 'lib.npz'
    np.savez(p, cell=np.arange(10), row_start=np.zeros(10, np.int64), row_count=np.ones(10, np.int64),
             row_relh=np.array([0.5]), row_hit_rate=np.array([1.0]), q=np.full(5, q))
    return str(p)


def _dataset(rx=None, training=True):
    from pcdet.datasets.dataset import DatasetTemplate
    cfg = EasyDict({
        'POINT_CLOUD_RANGE': [-75.2, -75.2, -2, 75.2, 75.2, 4],
        'POINT_FEATURE_ENCODING': {'encoding_type': 'absolute_coordinates_encoding', 'used_feature_list': ['x', 'y', 'z'],
                                   'src_feature_list': ['x', 'y', 'z', 'intensity', 'timestamp']},
        'ONTOLOGY': 'kitti', 'DATA_PROCESSOR': [],
        'DATA_AUGMENTOR': {'DISABLE_AUG_LIST': [], 'AUG_CONFIG_LIST': [
            {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x', 'y']},
            {'NAME': 'random_world_rotation', 'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]}]},
    })
    if rx is not None:
        cfg['ROW_EXTRUSION'] = rx
    return DatasetTemplate(dataset_cfg=cfg, class_names=['Car', 'Pedestrian', 'Cyclist'], training=training,
                           root_path=Path('/tmp'), logger=mock.MagicMock())


def _frame():
    rng = np.random.default_rng(0)
    n = 40000
    ground = np.c_[rng.uniform(-20, 20, (n, 2)), rng.normal(0, 0.02, n), rng.uniform(0, 255, n), np.zeros(n)]
    car = np.array([[20.0 + 0.0 * k, -1 + 0.1 * k, 0.3 + 0.1 * (k % 10), 50, 0] for k in range(20)])
    pts = np.concatenate([ground, car]).astype(np.float32)
    boxes = np.array([[21.0, 0.0, 0.75, 4.0, 2.0, 1.5, 0.0], [10.0, 8.0, 0.8, 0.8, 0.8, 1.7, 0.0]], np.float32)
    return {'points': pts, 'gt_boxes': boxes, 'gt_names': np.array(['Car', 'Pedestrian']), 'frame_id': '0',
            'num_points_in_gt': np.array([20, 3], np.int32)}


def test_hook_absent_by_default_and_at_evaluation(tmp_path):
    rx = EasyDict(ARM='H', GROUND='G0', LIBRARY=_library_file(tmp_path))
    with mock.patch.object(RX, 'apply_in_loader', wraps=RX.apply_in_loader) as m:
        _dataset().prepare_data(_frame()); assert m.call_count == 0
        _dataset(rx, training=False).prepare_data(_frame()); assert m.call_count == 0
        _dataset(rx).prepare_data(_frame()); assert m.call_count == 1


def test_hook_leaves_the_global_rng_and_the_augmented_boxes_unchanged(tmp_path):
    rx = EasyDict(ARM='H', GROUND='G0', LIBRARY=_library_file(tmp_path))
    np.random.seed(5); off = _dataset().prepare_data(_frame()); st_off = np.random.get_state()
    np.random.seed(5); on = _dataset(rx).prepare_data(_frame()); st_on = np.random.get_state()
    assert np.array_equal(off['gt_boxes'], on['gt_boxes']) and list(off['gt_names']) == list(on['gt_names'])
    assert np.array_equal(st_off[1], st_on[1]) and st_off[2] == st_on[2]
    assert len(on['points']) > len(off['points'])
    np.random.seed(5)
    d = {k: np.copy(v) for k, v in _frame().items() if isinstance(v, np.ndarray)}
    d['gt_boxes_mask'] = np.array([True, True])
    st = np.random.get_state(); RX.apply_in_loader(d, rx); st2 = np.random.get_state()
    assert np.array_equal(st[1], st2[1]) and st[2] == st2[2]


def test_loader_hook_equals_the_operator_with_the_global_generator(tmp_path):
    rx = EasyDict(ARM='H', GROUND='G0', LIBRARY=_library_file(tmp_path, q=0.5))
    f = _frame()
    d = {**f, 'gt_boxes_mask': np.array([True, True])}
    np.random.seed(11)
    out = RX.apply_in_loader({k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in d.items()}, rx)['points']
    np.random.seed(11)
    ref, _ = RX.row_extrusion(f['points'], f['gt_boxes'], rr.gt_car_mask(d), 'H', 'G0', RX.lattice(),
                              RX.load_library(rx.LIBRARY), rr.bg_generator(np.random))
    assert np.array_equal(out, ref)


def test_config_keys_are_validated():
    with pytest.raises(ValueError):
        RX.config_parts(EasyDict(ARM='X', GROUND='G0', LIBRARY='x'))
    with pytest.raises(ValueError):
        RX.config_parts(EasyDict(ARM='H', GROUND='G0'))
    with pytest.raises(ValueError):
        RX.row_extrusion(np.zeros((1, 3), np.float32), np.zeros((0, 7)), np.zeros(0, bool), arm='H', ground='G2', lat=LAT)


def test_library_with_an_empty_cell_is_refused():
    lib = RX.Library(np.arange(9), np.zeros(9, np.int64), np.ones(9, np.int64), [0.5], [1.0], np.ones(5))
    with pytest.raises(ValueError):
        lib.check()


# --------------------------------------------------------------------------------------------------- configs
def _load(n):
    from pcdet.config import cfg_from_yaml_file
    return cfg_from_yaml_file('cfgs/da-ieee-access/' + n, EasyDict())


def test_rx_configs_change_one_variable(monkeypatch):
    monkeypatch.chdir(ROOT / 'tools')
    hg0 = _load('centerpoint-gblobs-accum-legaldepth-rx-hg0-nuscenes2waymo.yaml')
    base = _load('centerpoint-gblobs-accum-legaldepth-nuscenes2kitti.yaml')          # 27471
    waymo = _load('centerpoint-gblobs-sourceonly-nuscenes2waymo.yaml')                # 27596's target block
    rr7a = _load('centerpoint-gblobs-accum-legaldepth-rr7a-nuscenes2waymo.yaml')      # 29690
    assert hg0.MODEL == base.MODEL and hg0.OPTIMIZATION == base.OPTIMIZATION and hg0.CLASS_NAMES == base.CLASS_NAMES
    assert hg0.DATA_CONFIG_TAR == waymo.DATA_CONFIG_TAR == rr7a.DATA_CONFIG_TAR
    for key in ('NUSCENES_N008', 'NUSCENES_N015'):
        a, b, r = hg0.DATA_CONFIGS[key], base.DATA_CONFIGS[key], rr7a.DATA_CONFIGS[key]
        assert dict(a.ROW_EXTRUSION) == dict(ARM='H', GROUND='G0', LIBRARY='/home/koyama/data/row_extrusion_20261021/library.npz')
        assert a.DATA_AUGMENTOR.AUG_CONFIG_LIST == b.DATA_AUGMENTOR.AUG_CONFIG_LIST == r.DATA_AUGMENTOR.AUG_CONFIG_LIST[1:]
        assert r.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NAME == 'relocate_render'
        strip = lambda c: {k: v for k, v in c.items() if k not in ('ROW_EXTRUSION', 'DATA_AUGMENTOR', '_BASE_CONFIG_')}
        assert strip(a) == strip(b) == strip(r)
        assert a.DATA_AUGMENTOR.DISABLE_AUG_LIST == b.DATA_AUGMENTOR.DISABLE_AUG_LIST
        assert 'ROW_EXTRUSION' not in b and 'ROW_EXTRUSION' not in r
    assert hg0.DATA_CONFIGS.NUSCENES_N008.MAX_SWEEPS == 15 and hg0.DATA_CONFIGS.NUSCENES_N015.MAX_SWEEPS == 10
    for tag, arm, ground in (('hg1', 'H', 'G1'), ('lg0', 'L', 'G0'), ('wg0', 'W', 'G0')):
        c = _load(f'centerpoint-gblobs-accum-legaldepth-rx-{tag}-nuscenes2waymo.yaml')
        assert c.MODEL == hg0.MODEL and c.OPTIMIZATION == hg0.OPTIMIZATION and c.DATA_CONFIG_TAR == hg0.DATA_CONFIG_TAR
        for key in ('NUSCENES_N008', 'NUSCENES_N015'):
            e = c.DATA_CONFIGS[key]
            assert dict(e.ROW_EXTRUSION) == dict(ARM=arm, GROUND=ground, LIBRARY=hg0.DATA_CONFIGS[key].ROW_EXTRUSION.LIBRARY)
            strip = lambda x: {k: v for k, v in x.items() if k not in ('ROW_EXTRUSION', '_BASE_CONFIG_')}
            assert strip(e) == strip(hg0.DATA_CONFIGS[key])


def test_no_other_config_sets_row_extrusion():
    hits = [p for p in (ROOT / 'tools' / 'cfgs').rglob('*.yaml') if 'ROW_EXTRUSION' in p.read_text(errors='replace')]
    assert sorted(p.name for p in hits) == sorted(f'centerpoint-gblobs-accum-legaldepth-rx-{t}-nuscenes2waymo.yaml'
                                                  for t in ('hg0', 'hg1', 'lg0', 'wg0'))
