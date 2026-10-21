"""Row extrusion (RX) on the Waymo lattice: a training-time source operator for nuScenes -> Waymo (experiments_md
20261021_02; design fixed in experiments_md T05 §5, commits 9a7c1f5 and b479ec7, before any RX number).

Idea (the user's, 2026-10-21). A far source car carries one or two HDL-32E lines. RX stacks each line in height at Waymo
TOP's rows, as a depth-image operation, with the source label giving where the car's surface ends. Source returns are never
moved; RX only adds points (and removes the ground returns its new points shadow).

Frame. The loader's frame at RR's slot (dataset.prepare_data, just before the augmentor queue, after the zero-point
filter): nuScenes LIDAR_TOP with SHIFT_COOR (0, 0, 1.75) added, so the ground is at z ~ 0. The virtual Waymo TOP sits at
(0, 0, 2.184). Rows are indexed in ASCENDING inclination (rerender_utils.to_lattice).

Steps (T05 §5; the readings of 20261021_02 §1.2 in brackets):
1. Range image: the TRAIN-median TOP lattice (relocate_render.waymo_top_train_lattice), 2,650 columns; each pixel holds the
   nearest source return (3D range from the virtual origin). Points outside TOP's vertical field of view are untouched.
2. Pixel classes: ground if the pixel's return lies <= 0.3 m above S, else object. S = the frame's RANSAC plane
   (relocate_render.ground_plane, 4-40 m) plus E5a's offset: the median of (box bottom - plane at its centre) over the
   frame's source Car boxes whose BEV centres lie within 30 m of (x, y); the plane alone where none does.
   [A frame without a plane is returned unchanged and counted.]
3. Extrusion: every pixel of the field of view takes as its seed the nearest object pixel in its column, by row index, ties
   to the lower row. [An object pixel is its own seed, so it never changes; the occupied-pixel rule therefore acts on
   ground pixels.] A pixel is filled only if the seed's bound holds, otherwise it stays as it is and no other seed is tried.
   - New point: at the seed's (x, y), z = 2.184 + hypot(x, y) tan(inclination of the pixel's row); every feature other than
     x, y, z copied from the seed.
   - Seed inside a labelled box (any class, every gt_boxes row): the new point lies inside the same box; no other bound.
   - Seed outside every box: |elevation(new) - elevation(seed)| <= 0.665 deg about the virtual origin; z >= S(x, y); the
     new point lies in no box.
   - Occupied pixel: filled only if its return is > 0.5 m farther than the new point; that return [the pixel's held,
     nearest return] is removed (shadow) and counted.
4. Return model, on new points only (source returns are always kept):
   - in Car boxes, transplant: per source Car box one library car is drawn uniformly from its cell (range bin 15-25 /
     25-40 / 40-50 / 50-60 / 60-75 m, clamped; aspect side if |relative view angle| in [45, 135] deg, else end); for each
     lattice row of the source box's silhouette the library row with the nearest relh gives the keep probability (its hit
     rate; a row without a hit drops the row's new points);
   - everywhere else: keep probability q(bin) of the new point's horizontal range (0-20 / 20-40 / 40-50 / 50-60 / 60-75 m).
5. Ground: G0 fills nothing; G1 fills still-empty pixels between two ground pixels of the column (the ray's intersection
   with the segment joining them in (horizontal range, z)) and, beyond the column's last ground pixel, with the ray's
   intersection with S up to 75.2 m if no object pixel lies below it in the column; G1 points are kept with q.
Arms: H (the operator above), L (extrusion in Car boxes only, nothing outside), W (no boxes: S = the plane alone, the
outside rule and q everywhere, no transplant); ground G0 / G1.

Random numbers: a generator seeded from the global numpy state WITHOUT advancing it (relocate_render.bg_generator), so every
later draw of the frame (augmentations, shuffle) is what it is with RX off. Draw order: one integer per Car box (gt order;
H and L), then one uniform per new point (pixel order), then one per G1 point.

Legality (memory/repo/uda_legality_rule.md): the lattice, mount height and column count are the target's published
calibration (TRAIN segments); the library and q come from teacher 27471's detections on Waymo TRAIN (fit-set segments), no
Waymo label; the boxes and E5a are SOURCE labels.

    out_points, record = row_extrusion(points, gt_boxes, car_mask, arm='H', ground='G0', lat=..., library=..., rng=...)
    data_dict = apply_in_loader(data_dict, cfg)       # the DATA_CONFIG.ROW_EXTRUSION hook
"""
import functools
import os
import sys

import numpy as np

from ..rerender_utils import to_lattice
from . import relocate_render as rr

MOUNT_HEIGHT = 2.184
N_COLS = 2650
GROUND_TOL = 0.3              # m above S: ground pixel
E5A_RADIUS = 30.0             # m, BEV centre distance of the Car boxes giving the offset
PLANE_RMAX = 40.0             # m, the RANSAC plane's outer range
ELEV_TOL_DEG = 0.665          # half of HDL-32E's 1.33 deg
SHADOW_M = 0.5                # an occupied pixel is filled only if its return is this much farther
G1_RMAX = 75.2                # m, horizontal range limit of G1's ground beyond the last ground pixel
LIB_BINS = ((15.0, 25.0), (25.0, 40.0), (40.0, 50.0), (50.0, 60.0), (60.0, 75.0))
Q_BINS = ((0.0, 20.0), (20.0, 40.0), (40.0, 50.0), (50.0, 60.0), (60.0, 75.0))
SIDE_DEG = (45.0, 135.0)
ARMS = ('H', 'L', 'W')
GROUNDS = ('G0', 'G1')
# new-point types
T_CAR, T_OTHER, T_BG, T_G1 = 0, 1, 2, 3


# ------------------------------------------------------------------------------------------------------- lattice
def lattice(calib=rr.WAYMO_TOP_CALIB, train_list=rr.WAYMO_TRAIN_LIST):
    """The virtual Waymo TOP: the element-wise median of the 798 TRAIN segments' TOP inclinations (ascending, rad),
    2,650 columns, 2.184 m (relocate_render.waymo_top_train_lattice, checked against the published extrinsic z)."""
    lat = rr.waymo_top_train_lattice(calib, train_list, N_COLS, MOUNT_HEIGHT)
    if not all(abs(z - lat['height']) < 1e-3 for z in lat['extrinsic_z']):
        raise ValueError(f"mount height {lat['height']} disagrees with the published extrinsic z {lat['extrinsic_z']}")
    return lat


def range_image(xyz, lat):
    """Nearest return per pixel: (idx (R, C) int64, -1 = empty; rng (R, C), inf = empty; to_lattice's (row, col, rng,
    valid) of every point). Ties in range keep the lower point index."""
    th, nc, h = np.asarray(lat['thetas'], np.float64), int(lat['n_cols']), float(lat['height'])
    nr = len(th)
    idx = np.full(nr * nc, -1, np.int64); rim = np.full(nr * nc, np.inf)
    p = np.asarray(xyz, np.float64).reshape(-1, np.shape(xyz)[-1] if np.ndim(xyz) == 2 else 3)[:, :3]
    if not len(p):
        e = np.zeros(0, np.int64)
        return idx.reshape(nr, nc), rim.reshape(nr, nc), (e, e, np.zeros(0), np.zeros(0, bool))
    row, col, rng, valid = to_lattice(p, h, th, nc)
    sel = np.nonzero(valid)[0]
    if len(sel):
        cell = row[sel] * nc + col[sel]
        o = sel[np.lexsort((rng[sel], cell))]
        c = row[o] * nc + col[o]
        first = np.ones(len(o), bool); first[1:] = c[1:] != c[:-1]
        k = o[first]
        idx[row[k] * nc + col[k]] = k; rim[row[k] * nc + col[k]] = rng[k]
    return idx.reshape(nr, nc), rim.reshape(nr, nc), (row, col, rng, valid)


# ------------------------------------------------------------------------------------------------------- surface S
class Surface:
    """S(x, y) = plane(x, y) + E5a's offset: the median of (box bottom - plane at the box centre) over the Car boxes whose
    BEV centres lie within `radius` (<=, as relocate_render.e5a_bottom) of (x, y); 0 (the plane alone) where none does."""

    def __init__(self, plane, car_boxes=None, radius=E5A_RADIUS):
        self.plane = np.asarray(plane, np.float64)
        b = np.zeros((0, 7)) if car_boxes is None else np.asarray(car_boxes, np.float64).reshape(-1, np.shape(car_boxes)[-1])
        self.cx, self.cy = b[:, 0].copy(), b[:, 1].copy()
        self.res = (b[:, 2] - b[:, 5] / 2) - rr.plane_z(self.plane, b[:, 0], b[:, 1]) if len(b) else np.zeros(0)
        self.radius = float(radius)

    def offset(self, x, y, chunk=40000):
        x = np.asarray(x, np.float64).ravel(); y = np.asarray(y, np.float64).ravel()
        out = np.zeros(len(x))
        k = len(self.res)
        if not k or not len(x):
            return out
        for s in range(0, len(x), chunk):
            m = np.hypot(x[s:s + chunk, None] - self.cx[None], y[s:s + chunk, None] - self.cy[None]) <= self.radius
            any_ = m.any(1)
            if not any_.any():
                continue
            sub = np.nonzero(any_)[0]
            packed = np.packbits(m[sub], axis=1)
            u, inv = np.unique(packed, axis=0, return_inverse=True)
            inv = np.asarray(inv).ravel()
            med = np.array([np.median(self.res[np.unpackbits(r)[:k].astype(bool)]) for r in u])
            out[s + sub] = med[inv]
        return out

    def __call__(self, x, y):
        return rr.plane_z(self.plane, np.asarray(x, np.float64), np.asarray(y, np.float64)) + self.offset(x, y)


# ------------------------------------------------------------------------------------------------------ geometry
def nearest_seed_rows(obj):
    """Per pixel the nearest object row in its column by row index, ties to the lower row (-1 if the column has none).
    An object pixel is its own seed."""
    obj = np.asarray(obj, bool)
    R = obj.shape[0]
    r = np.arange(R)[:, None]
    below = np.maximum.accumulate(np.where(obj, r, -1), axis=0)
    above = np.minimum.accumulate(np.where(obj, r, R)[::-1], axis=0)[::-1]
    db = np.where(below >= 0, r - below, np.iinfo(np.int64).max)
    da = np.where(above < R, above - r, np.iinfo(np.int64).max)
    seed = np.where(db <= da, below, above)
    return np.where((below >= 0) | (above < R), seed, -1)


def _f32(a):
    return np.asarray(a, np.float32)


def bev_in_boxes(xy, boxes):
    """(n, m) BEV part of relocate_render.points_in_box (roiaware's test, float32, 1 cm margin)."""
    p = _f32(xy).reshape(-1, 2); b = _f32(boxes).reshape(-1, np.shape(boxes)[-1] if len(boxes) else 7)
    if not len(p) or not len(b):
        return np.zeros((len(p), len(b)), bool)
    sx = p[:, 0:1] - b[None, :, 0]; sy = p[:, 1:2] - b[None, :, 1]
    ca = np.cos(-b[:, 6]); sa = np.sin(-b[:, 6])
    lx = sx * ca + sy * (-sa); ly = sx * sa + sy * ca
    m = np.float64(np.float32(1e-2))
    return (np.abs(lx).astype(np.float64) < b[:, 3].astype(np.float64) / 2.0 + m) & \
        (np.abs(ly).astype(np.float64) < b[:, 4].astype(np.float64) / 2.0 + m)


def z_in_boxes(z, boxes):
    """(n, m) z part of relocate_render.points_in_box: |z - cz| <= dz / 2 in float32."""
    z = _f32(z).reshape(-1); b = _f32(boxes).reshape(-1, np.shape(boxes)[-1] if len(boxes) else 7)
    if not len(z) or not len(b):
        return np.zeros((len(z), len(b)), bool)
    return np.abs(z[:, None] - b[None, :, 2]).astype(np.float64) <= b[:, 5].astype(np.float64)[None] / 2.0


def points_in_boxes(xyz, boxes):
    """(n, m) relocate_render.points_in_box for every pair (unit-tested equal)."""
    p = np.asarray(xyz).reshape(-1, 3)
    return bev_in_boxes(p[:, :2], boxes) & z_in_boxes(p[:, 2], boxes)


def bin_index(r, bins, clamp=True):
    """Index of the [lo, hi) bin of each r; with clamp, below the first bin -> 0 and at or above the last -> last."""
    r = np.asarray(r, np.float64)
    edges = np.array([b[0] for b in bins] + [bins[-1][1]])
    k = np.searchsorted(edges, r, side='right') - 1
    if clamp:
        return np.clip(k, 0, len(bins) - 1)
    return np.where((k >= 0) & (k < len(bins)), k, -1)


def aspect_side(alpha_rad):
    """True for the side aspect: |relative view angle| in [45, 135] deg."""
    a = np.abs(np.degrees(np.asarray(alpha_rad, np.float64)))
    return (a >= SIDE_DEG[0]) & (a <= SIDE_DEG[1])


def cell_of(box):
    """Library cell (range bin * 2 + side) of a source box (loader frame, about the virtual TOP's vertical axis)."""
    b = np.asarray(box, np.float64)
    k = int(bin_index(np.hypot(b[0], b[1]), LIB_BINS))
    return k * 2 + int(aspect_side(rr.relative_view_angle(b)))


def silhouette_row_relh(box, lat):
    """Per ascending lattice row of the box's (full) silhouette the median relative entry height (relocate_render's
    silhouette; entry z clipped into the box height as box_pattern_visual.row_records): (rows, relh)."""
    th, nc, h = np.asarray(lat['thetas'], np.float64), lat['n_cols'], lat['height']
    sp, srow, tin = rr.silhouette(box, th, nc, h)
    if not len(sp):
        return np.zeros(0, np.int64), np.zeros(0)
    b = np.asarray(box, np.float64)
    z_ent = h + tin * np.sin(th[srow])
    relh = np.clip((z_ent - (b[2] - b[5] / 2)) / b[5], 0.0, 1.0)
    ur, inv = np.unique(srow, return_inverse=True)
    return ur, np.array([np.median(relh[inv == j]) for j in range(len(ur))])


# ------------------------------------------------------------------------------------------------------- library
class Library:
    """The transplant library and q (tools/analysis/rx_build_library.py): per car its cell and rows (relh, hit rate);
    q per Q_BINS bin."""

    def __init__(self, cell, row_start, row_count, relh, hit_rate, q):
        self.cell = np.asarray(cell, np.int64); self.row_start = np.asarray(row_start, np.int64)
        self.row_count = np.asarray(row_count, np.int64)
        self.relh = np.asarray(relh, np.float64); self.hit_rate = np.asarray(hit_rate, np.float64)
        self.q = np.asarray(q, np.float64).reshape(len(Q_BINS))
        self.by_cell = [np.nonzero(self.cell == c)[0] for c in range(2 * len(LIB_BINS))]

    @classmethod
    def from_npz(cls, path):
        d = np.load(path)
        return cls(d['cell'], d['row_start'], d['row_count'], d['row_relh'], d['row_hit_rate'], d['q'])

    def check(self):
        empty = [c for c, ix in enumerate(self.by_cell) if not len(ix)]
        if empty:
            raise ValueError(f'row extrusion library: empty cells {empty}')
        if not np.all((self.q >= 0) & (self.q <= 1)):
            raise ValueError(f'row extrusion library: q outside [0, 1]: {self.q}')
        return self

    def rows(self, k):
        s, n = self.row_start[k], self.row_count[k]
        return self.relh[s:s + n], self.hit_rate[s:s + n]


@functools.lru_cache(maxsize=4)
def load_library(path):
    """Read-only, loaded lazily in each worker (never pickled with the dataset)."""
    return Library.from_npz(path).check()


def transplant_prob(rows_new, z_new, box, lib_relh, lib_hit, lat):
    """Keep probability of each new point of one source Car box: the hit rate of the library row with the nearest relh to
    the relh of the new point's lattice row (median entry relh of the box's silhouette pixels in that row; a row outside
    the silhouette uses the new point's own relative height). Ties go to the first library row."""
    b = np.asarray(box, np.float64)
    ur, urelh = silhouette_row_relh(b, lat)
    relh = np.clip((np.asarray(z_new, np.float64) - (b[2] - b[5] / 2)) / b[5], 0.0, 1.0)
    if len(ur):
        pos = np.clip(np.searchsorted(ur, rows_new), 0, len(ur) - 1)
        has = ur[pos] == rows_new
        relh = np.where(has, urelh[pos], relh)
    if not len(lib_relh):
        return np.zeros(len(relh))
    j = np.argmin(np.abs(relh[:, None] - np.asarray(lib_relh)[None]), axis=1)
    return np.asarray(lib_hit, np.float64)[j]


# ------------------------------------------------------------------------------------------------------ operator
def row_extrusion(points, gt_boxes, car_mask, arm='H', ground='G0', lat=None, library=None, rng=None, plane='fit',
                  keep_record=False):
    """The operator (module docstring). points (N, 3 + C) in the loader frame; gt_boxes (M, >= 7), every labelled box;
    car_mask (M,) the source Car boxes (relocate_render.gt_car_mask). plane: 'fit' (the frame's RANSAC plane, 4-40 m) or
    (a, b, c) / None for tests. Returns (points (N - shadowed + kept new, 3 + C), record)."""
    if arm not in ARMS or ground not in GROUNDS:
        raise ValueError(f'row extrusion: arm {arm!r} (H | L | W), ground {ground!r} (G0 | G1)')
    lat = lattice() if lat is None else lat
    rng = np.random.default_rng(0) if rng is None else rng
    pts = np.asarray(points)
    boxes_all = np.asarray(gt_boxes, np.float64).reshape(-1, np.shape(gt_boxes)[-1] if len(gt_boxes) else 7)[:, :7]
    car = np.asarray(car_mask, bool).reshape(-1) if len(boxes_all) else np.zeros(0, bool)
    assert len(car) == len(boxes_all), (len(car), len(boxes_all))
    rec = dict(arm=arm, ground=ground, n_in=len(pts), no_plane=False, n_new=0, n_kept=0, n_shadow=0,
               n_new_type=[0, 0, 0, 0], n_kept_type=[0, 0, 0, 0], n_below_s_new=0, n_below_s_kept=0, n_g1_between=0,
               n_g1_beyond=0, n_obj_pix=0, n_ground_pix=0, n_empty_pix=0, n_cand=0, lib_pick=[])
    if len(pts) == 0:
        rec['no_points'] = True
        return pts.copy(), rec
    xyz = np.asarray(pts[:, :3], np.float64)
    th, nc, h = np.asarray(lat['thetas'], np.float64), int(lat['n_cols']), float(lat['height'])
    nr = len(th)
    if isinstance(plane, str):
        plane = rr.ground_plane(xyz, PLANE_RMAX)
    if plane is None:
        rec['no_plane'] = True
        return pts.copy(), rec
    # arm's box sets
    if arm == 'W':
        rule_boxes = np.zeros((0, 7)); rule_car = np.zeros(0, bool); s_cars = np.zeros((0, 7))
    elif arm == 'L':
        rule_boxes = boxes_all[car]; rule_car = np.ones(len(rule_boxes), bool); s_cars = boxes_all[car]
    else:
        rule_boxes = boxes_all; rule_car = car; s_cars = boxes_all[car]
    rule_ix = np.nonzero(car)[0] if arm == 'L' else np.arange(len(rule_boxes))      # index into gt_boxes
    S = Surface(plane, s_cars)
    # 1-2: range image and pixel classes
    idx, rim, _ = range_image(xyz, lat)
    occ = idx >= 0
    s_pix = np.full(idx.shape, np.nan)
    if occ.any():
        pp = xyz[idx[occ]]
        s_pix[occ] = S(pp[:, 0], pp[:, 1])
    zpix = np.full(idx.shape, np.nan); zpix[occ] = xyz[idx[occ], 2]
    with np.errstate(invalid='ignore'):
        gpix = occ & (zpix <= s_pix + GROUND_TOL)
    obj = occ & ~gpix
    rec.update(n_obj_pix=int(obj.sum()), n_ground_pix=int(gpix.sum()), n_empty_pix=int((~occ).sum()))
    # 3: extrusion
    seed_row = nearest_seed_rows(obj)
    cand = ~obj & (seed_row >= 0)
    ci, cc = np.nonzero(cand)
    rec['n_cand'] = len(ci)
    new_xyz = np.zeros((0, 3)); new_src = np.zeros(0, np.int64); new_type = np.zeros(0, np.int64)
    new_box = np.zeros(0, np.int64); new_row = np.zeros(0, np.int64); shadow_idx = np.zeros(0, np.int64)
    new_pix = np.zeros(0, np.int64)
    if len(ci):
        sr = seed_row[ci, cc]
        sp = idx[sr, cc]                                   # seed point per candidate
        s_at = s_pix[sr, cc]                               # S at the seed's (x, y)
        useed, uinv = np.unique(sp, return_inverse=True)
        uinv = np.asarray(uinv).ravel()
        s_xyz = xyz[useed]
        rho_s = np.hypot(s_xyz[:, 0], s_xyz[:, 1])
        z_new = h + rho_s[uinv] * np.tan(th[ci])
        el_s = np.arctan2(s_xyz[:, 2] - h, np.maximum(rho_s, 1e-6))
        if len(rule_boxes):
            bev_s = bev_in_boxes(s_xyz[:, :2], rule_boxes)                     # (u, m)
            in_s = bev_s & z_in_boxes(s_xyz[:, 2], rule_boxes)
        else:
            bev_s = np.zeros((len(useed), 0), bool); in_s = bev_s
        seed_in = in_s.any(1)[uinv]
        ok = np.zeros(len(ci), bool); box_of = np.full(len(ci), -1, np.int64)
        # seeds inside a box: the new point inside the same (a seed-containing) box
        a = np.nonzero(seed_in)[0]
        if len(a):
            zin = z_in_boxes(z_new[a], rule_boxes) & in_s[uinv[a]]
            ok[a] = zin.any(1)
            first = np.argmax(zin, axis=1)
            box_of[a] = np.where(ok[a], rule_ix[first], -1)
        # seeds outside every box (not in L)
        if arm != 'L':
            o = np.nonzero(~seed_in)[0]
            o = o[np.abs(th[ci[o]] - el_s[uinv[o]]) <= np.radians(ELEV_TOL_DEG)]
            o = o[z_new[o] >= s_at[o]]
            if len(o) and len(rule_boxes):
                o = o[~(bev_s[uinv[o]] & z_in_boxes(z_new[o], rule_boxes)).any(1)]
            ok[o] = True
        # occupied (ground) pixels: only if their return is > SHADOW_M farther than the new point
        rng_new = np.hypot(rho_s[uinv], z_new - h)
        held = idx[ci, cc]
        occupied = held >= 0
        ok &= ~occupied | (rim[ci, cc] > rng_new + SHADOW_M)
        sel = np.nonzero(ok)[0]
        shadow_idx = held[sel][occupied[sel]]
        new_xyz = np.c_[s_xyz[uinv[sel], 0], s_xyz[uinv[sel], 1], z_new[sel]]
        new_src = sp[sel]; new_row = ci[sel]; new_pix = ci[sel] * nc + cc[sel]
        new_box = box_of[sel]
        is_car = np.zeros(len(sel), bool)
        if len(boxes_all):
            is_car = (new_box >= 0) & car[np.maximum(new_box, 0)]
        new_type = np.where(new_box < 0, T_BG, np.where(is_car, T_CAR, T_OTHER))
    rec['n_shadow'] = int(len(shadow_idx))
    # 4: return model
    p = np.zeros(len(new_xyz))
    rho_new = np.hypot(new_xyz[:, 0], new_xyz[:, 1]) if len(new_xyz) else np.zeros(0)
    if library is not None:
        q = library.q
    else:
        q = np.ones(len(Q_BINS))
    nq = new_type != T_CAR
    p[nq] = q[bin_index(rho_new[nq], Q_BINS)]
    car_ix = np.nonzero(car)[0] if arm in ('H', 'L') else np.zeros(0, np.int64)
    picks = {}
    for j in car_ix:                                   # one draw per Car box, gt order (H, L)
        if library is None:
            picks[j] = -1; continue
        members = library.by_cell[cell_of(boxes_all[j])]
        picks[j] = int(members[int(rng.integers(len(members)))]) if len(members) else -1
    rec['lib_pick'] = [(int(j), int(k)) for j, k in picks.items()]
    for j in np.unique(new_box[new_type == T_CAR]):
        m = (new_type == T_CAR) & (new_box == j)
        if library is None or picks.get(j, -1) < 0:
            p[m] = 1.0 if library is None else 0.0
            continue
        lr, lh = library.rows(picks[j])
        p[m] = transplant_prob(new_row[m], new_xyz[m, 2], boxes_all[j], lr, lh, lat)
    u = rng.random(len(new_xyz))
    kept = u < p
    # 5: G1 ground fill (still-empty pixels)
    g1_xyz = np.zeros((0, 3)); g1_src = np.zeros(0, np.int64); g1_kept = np.zeros(0, bool); g1_pix = np.zeros(0, np.int64)
    if ground == 'G1':
        filled = np.zeros(nr * nc, bool); filled[new_pix] = True
        g1_xyz, g1_src, g1_kind, g1_pix = ground_fill(idx, gpix, obj, filled.reshape(nr, nc), xyz, S, lat)
        rec['n_g1_between'] = int((g1_kind == 0).sum()); rec['n_g1_beyond'] = int((g1_kind == 1).sum())
        pg = q[bin_index(np.hypot(g1_xyz[:, 0], g1_xyz[:, 1]), Q_BINS)] if len(g1_xyz) else np.zeros(0)
        g1_kept = rng.random(len(g1_xyz)) < pg
    # output
    keep_src = np.ones(len(pts), bool); keep_src[shadow_idx] = False
    add = [pts[new_src[kept]].copy(), pts[g1_src[g1_kept]].copy()]
    add[0][:, :3] = new_xyz[kept].astype(pts.dtype); add[1][:, :3] = g1_xyz[g1_kept].astype(pts.dtype)
    out = np.concatenate([pts[keep_src]] + add, axis=0)
    all_xyz = np.concatenate([new_xyz, g1_xyz]); all_type = np.r_[new_type, np.full(len(g1_xyz), T_G1)]
    all_kept = np.r_[kept, g1_kept]
    below = all_xyz[:, 2] < S(all_xyz[:, 0], all_xyz[:, 1]) if len(all_xyz) else np.zeros(0, bool)
    rec.update(n_new=int(len(all_xyz)), n_kept=int(all_kept.sum()),
               n_new_type=[int((all_type == t).sum()) for t in range(4)],
               n_kept_type=[int(((all_type == t) & all_kept).sum()) for t in range(4)],
               n_below_s_new=int(below.sum()), n_below_s_kept=int((below & all_kept).sum()), n_out=len(out),
               n_src_kept=int(keep_src.sum()), plane=np.asarray(plane, np.float64))
    if keep_record:
        rec.update(new_xyz=all_xyz, new_type=all_type, new_kept=all_kept, new_box=np.r_[new_box, np.full(len(g1_xyz), -1)],
                   new_pix=np.r_[new_pix, g1_pix], new_below=below, shadow_idx=shadow_idx, keep_src=keep_src, p=np.r_[p, np.full(len(g1_xyz), np.nan)])
    return out, rec


def intersect_rho(num, den):
    """Horizontal range of the ray z = h + rho tan(theta) on z = plane + offset: rho = (c + off - h) / (tan(theta) - a cos
    - b sin); nan where it is not a positive finite intersection."""
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.asarray(num, np.float64) / np.asarray(den, np.float64)
    return np.where(np.isfinite(r) & (r > 0), r, np.nan)


def ground_fill(idx, gpix, obj, filled, xyz, S, lat, n_iter=5):
    """G1 (module docstring step 5) on the pixels empty in the input and not filled by the extrusion. Returns (xyz (k, 3),
    the source point whose features each takes, kind (0 between, 1 beyond), pixel ids). The ray is the pixel's centre ray
    (row inclination, column-centre azimuth)."""
    th, nc, h = np.asarray(lat['thetas'], np.float64), int(lat['n_cols']), float(lat['height'])
    R, C = idx.shape
    occ = idx >= 0
    empty = ~occ & ~filled
    r = np.arange(R)[:, None]
    # nearest occupied pixel strictly below / above (input image)
    below = np.maximum.accumulate(np.where(occ, r, -1), axis=0)
    below = np.vstack([np.full((1, C), -1), below[:-1]])
    above = np.minimum.accumulate(np.where(occ, r, R)[::-1], axis=0)[::-1]
    above = np.vstack([above[1:], np.full((1, C), R)])
    cols = np.broadcast_to(np.arange(C), (R, C))
    bl = np.clip(below, 0, R - 1); ab = np.clip(above, 0, R - 1)
    between = empty & (below >= 0) & (above < R) & gpix[bl, cols] & gpix[ab, cols]
    out_xyz, out_src, out_kind, out_pix = [], [], [], []
    bi, bc = np.nonzero(between)
    if len(bi):
        az = (bc + 0.5) / nc * 2 * np.pi - np.pi
        pa = xyz[idx[below[bi, bc], bc]]; pb = xyz[idx[above[bi, bc], bc]]
        ra, za = np.hypot(pa[:, 0], pa[:, 1]), pa[:, 2]; rb, zb = np.hypot(pb[:, 0], pb[:, 1]), pb[:, 2]
        t_ = np.tan(th[bi])
        fa = za - h - ra * t_; fb = zb - h - rb * t_
        with np.errstate(divide='ignore', invalid='ignore'):
            t = fa / (fa - fb)
        ok = (np.sign(fa) != np.sign(fb)) | (fa == 0) | (fb == 0)
        t = np.where(fa == fb, 0.0, t)
        ok &= (t >= 0) & (t <= 1) & np.isfinite(t)
        rho = ra + t * (rb - ra)
        ok &= rho > 0
        sel = np.nonzero(ok)[0]
        out_xyz.append(np.c_[rho[sel] * np.cos(az[sel]), rho[sel] * np.sin(az[sel]), h + rho[sel] * t_[sel]])
        out_src.append(idx[below[bi[sel], bc[sel]], bc[sel]]); out_kind.append(np.zeros(len(sel), np.int64))
        out_pix.append(bi[sel] * nc + bc[sel])
    # beyond the column's last (highest) ground pixel, no object pixel below
    last_g = np.where(gpix.any(0), R - 1 - np.argmax(gpix[::-1], axis=0), -1)
    obj_below = np.vstack([np.zeros((1, C), bool), np.logical_or.accumulate(obj, axis=0)[:-1]])
    beyond = empty & (last_g[None] >= 0) & (r > last_g[None]) & ~obj_below
    yi, yc = np.nonzero(beyond)
    if len(yi):
        az = (yc + 0.5) / nc * 2 * np.pi - np.pi
        ca, sa, t_ = np.cos(az), np.sin(az), np.tan(th[yi])
        a, b, c = S.plane
        den = t_ - a * ca - b * sa
        off = np.zeros(len(yi))
        rho = intersect_rho(c + off - h, den)
        for _ in range(n_iter - 1):                    # fixed point: the offset at the last intersection
            fin = np.isfinite(rho)
            off = np.zeros(len(yi))
            if fin.any():
                off[fin] = S.offset(rho[fin] * ca[fin], rho[fin] * sa[fin])
            rn = intersect_rho(c + off - h, den)
            done = np.array_equal(np.isfinite(rn), fin) and np.all(np.abs(rn[fin] - rho[fin]) < 1e-6)
            rho = rn
            if done:
                break
        ok = np.isfinite(rho) & (rho <= G1_RMAX)
        sel = np.nonzero(ok)[0]
        out_xyz.append(np.c_[rho[sel] * ca[sel], rho[sel] * sa[sel], h + rho[sel] * t_[sel]])
        out_src.append(idx[last_g[yc[sel]], yc[sel]]); out_kind.append(np.ones(len(sel), np.int64))
        out_pix.append(yi[sel] * nc + yc[sel])
    if not out_xyz:
        return np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    return (np.concatenate(out_xyz), np.concatenate(out_src), np.concatenate(out_kind), np.concatenate(out_pix))


# ------------------------------------------------------------------------------------------- range-image statistics
def run_lengths(has, rng, dr=0.3):
    """Per pixel of an (R, C) range image (rows in elevation order) the length of its run: consecutive returning rows of
    the column whose adjacent ranges differ by <= dr (0 for a pixel without a return)."""
    has = np.asarray(has, bool); rng = np.asarray(rng, np.float64)
    R, C = has.shape
    with np.errstate(invalid='ignore'):
        link = has[:-1] & has[1:] & (np.abs(rng[:-1] - rng[1:]) <= dr)          # (R - 1, C): row i linked to i + 1
    start = np.ones((R, C), bool); start[1:] = ~link
    rid = np.cumsum(start.T.ravel()).reshape(C, R).T                            # run ids, column-major
    cnt = np.bincount(rid.ravel())
    out = cnt[rid]
    return np.where(has, out, 0)


def hist_median(h):
    """numpy's median of the integer sample whose histogram is h (h[v] = count of value v); nan if empty."""
    h = np.asarray(h, np.int64); n = int(h.sum())
    if n == 0:
        return np.nan
    c = np.cumsum(h)
    lo = int(np.searchsorted(c, (n - 1) // 2 + 1)); hi = int(np.searchsorted(c, n // 2 + 1))
    return (lo + hi) / 2.0


# ------------------------------------------------------------------------------------------------- loader hook
_STATS = {}


def config_parts(cfg):
    arm = str(cfg.get('ARM', 'H')); ground = str(cfg.get('GROUND', 'G0'))
    if arm not in ARMS or ground not in GROUNDS:
        raise ValueError(f'ROW_EXTRUSION: ARM {arm!r} (H | L | W), GROUND {ground!r} (G0 | G1)')
    lib = cfg.get('LIBRARY', None)
    if lib is None:
        raise ValueError('ROW_EXTRUSION: LIBRARY (tools/analysis/rx_build_library.py output) is required')
    return arm, ground, str(lib)


@functools.lru_cache(maxsize=2)
def _lattice_cached():
    return lattice()


def apply_in_loader(data_dict, cfg):
    """DATA_CONFIG.ROW_EXTRUSION = {ARM: H | L | W, GROUND: G0 | G1, LIBRARY: <library.npz>}: the hook at RR's slot
    (dataset.prepare_data, before the augmentor queue, training on the source only). Only 'points' changes; boxes, names
    and masks are untouched. The global numpy RNG is not advanced. Each worker prints a cumulative line every 500 frames."""
    arm, ground, lib_path = config_parts(cfg)
    lib = load_library(lib_path)
    rng = rr.bg_generator(np.random)
    car = rr.gt_car_mask(data_dict)
    out, rec = row_extrusion(data_dict['points'], data_dict['gt_boxes'], car, arm, ground, _lattice_cached(), lib, rng)
    data_dict['points'] = out
    st = _STATS.setdefault(os.getpid(), [0, 0, 0, 0, 0])
    st[0] += 1; st[1] += rec['n_kept']; st[2] += rec['n_shadow']; st[3] += int(rec['no_plane']); st[4] += rec['n_in']
    if st[0] % 500 == 0:
        print('[RX %s-%s stats] pid %d: frames %d, new points kept %d (%.0f per frame, %.2f%% of input), shadowed %d, '
              'frames without a plane %d' % (arm, ground, os.getpid(), st[0], st[1], st[1] / st[0],
                                               100.0 * st[1] / max(st[4], 1), st[2], st[3]), file=sys.stderr, flush=True)
    return data_dict
