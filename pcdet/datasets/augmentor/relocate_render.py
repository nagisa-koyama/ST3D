"""Relocate-and-render (RR): a training-time source augmentation (experiments_md 20261018_02).

A near, dense source car has its whole visible surface. RR copies it, moves the copy farther out (rotated about the
sensor's vertical axis to a new azimuth, then slid along its new ray to a new range, so the face the sensor sees is
kept), and re-renders the copy through a TARGET sensor's published lattice (Waymo TOP: beam inclinations, columns per
revolution, mount height). That supplies the scan lines a far target car has and the source never recorded. The design
and every value are fixed in experiments_md T05 §5 (commit 1963034) and 20261018_02 §1; nothing here is tuned on target
results, and target point clouds and labels are never read: the lattice is the target's published calibration.

Frame. Everything runs in the loader's frame at the augmentor: the source sensor frame with SHIFT_COOR added, so the
ground is at z ~ 0. The virtual target sensor sits at (0, 0, height). Azimuth and BEV range are about the vertical axis
x = y = 0, which the source sensor, the ground origin and the virtual sensor share; elevations are about the virtual
sensor. Rows are indexed in ASCENDING inclination (`rerender_utils.to_lattice`'s convention).

Object-level API (stable; also for analyses that render one object at fixed ranges, object-only, no scene)
---------------------------------------------------------------------------------------------------------
    lat = waymo_top_train_lattice()
        dict(thetas = 64 ascending inclinations in rad (the element-wise median over the Waymo TRAIN segments of
        top_calib.pkl), n_cols = 2650, height = 2.184, n_segments, extrinsic_z).
    box2, xyz2 = relocate(box, xyz, r_new, phi_new)
        Rigid move: rotate box (and points) about the z axis by phi_new - azimuth(box), then translate along the new
        ray so the BEV centre range is r_new. Heading += the same angle (velocity columns 7:9 rotated too); z, size and
        the heading relative to the line of sight are unchanged.
    xyz, rows, cols, kept = render_object(xyz, thetas, n_cols, height)
        Object-only render: each point goes to (nearest inclination, azimuth column) about (0, 0, height); per pixel the
        NEAREST point is kept and SNAPPED onto the pixel's centre ray (row inclination, column-centre azimuth) at its
        own range. Returns snapped xyz (m, 3), ascending row index, column, and the index of the kept input point.
        Ported from ST3D tools/analysis/source_coverage.py `render` (ec2526a, 20261018_01 B3); a unit test pins the two
        to identical output.
    out = render_object_at(box, points, r_new, phi_new, lat, time_column=None)
        relocate + render_object for an (N, 3 + C) cloud; columns beyond x, y, z are carried from the kept point, and
        `time_column` (if given) is set to 0, a single-scan point's time lag. Returns dict(box, moved, points, rows,
        cols, kept).

Scene-level API (the `relocate_render` entry of DataAugmentor)
--------------------------------------------------------------
    data_dict, record = relocate_render_scene(data_dict, params, lat, rng=np.random)
        params: the augmentor config (keys of DEFAULTS). Donors: boxes named DONOR_CLASS with gt_boxes_mask true, not
        ignore boxes (gt_boxes_ignore_mask, or gt_classes <= 0), 0 < r < R_MAX, >= MIN_DONOR_POINTS points inside
        (roiaware points_in_boxes_cpu's test). Visited in random order until MAX_COPIES copies are placed; per donor up
        to MAX_TRIES draws of r' ~ U[r_donor, R_MAX], phi' ~ U[0, 2 pi), rejected if the copy's BEV box grown by
        BEV_MARGIN on every side overlaps any GT box (every row, any label) or a placed copy, has a corner outside
        POINT_CLOUD_RANGE (x, y), or holds more than MAX_SCENE_POINTS scene points higher than CLEARANCE above the
        copy's bottom. Occlusion per copy: scene points inside the copy's box are removed; a rendered copy point is
        dropped if its pixel holds a scene point nearer by more than OCCLUSION_M; scene points farther than the copy
        point in every pixel the copy renders into are removed (shadow). A copy left with no point is not added (its
        edits are undone). Copies are appended to gt_boxes / gt_names / gt_boxes_mask (and gt_boxes_ignore_mask,
        gt_classes, gt_scores, num_points_in_gt when present). Original boxes are never moved or removed.
        record: counts and per-copy details (for the gate and the target dump).
"""
import functools
import pickle

import numpy as np

from ..rerender_utils import to_lattice

WAYMO_TOP_CALIB = '/home/koyama/data/waymo_top_calib/top_calib.pkl'
WAYMO_TRAIN_LIST = '/home/koyama/data/waymo_open_dataset_v_1_4_0/pcdet_structure/ImageSets/train.txt'

DEFAULTS = dict(
    DONOR_CLASS='Car', MIN_DONOR_POINTS=200, MAX_COPIES=8, MAX_TRIES=20, R_MAX=75.2, BEV_MARGIN=0.5,
    MAX_SCENE_POINTS=5, CLEARANCE=0.3, OCCLUSION_M=0.5, POINT_CLOUD_RANGE=(-75.2, -75.2, -2.0, 75.2, 75.2, 4.0),
    TIME_COLUMN=4, LATTICE='waymo_top_train', N_COLS=2650, MOUNT_HEIGHT=2.184,
    CALIB=WAYMO_TOP_CALIB, TRAIN_SEGMENTS=WAYMO_TRAIN_LIST,
)


# ------------------------------------------------------------------------------------------------------- lattice
@functools.lru_cache(maxsize=4)
def _train_median(calib_path, train_list):
    d = pickle.load(open(calib_path, 'rb'))
    names = [ln.strip() for ln in open(train_list) if ln.strip()]
    keys = [n[:-len('.tfrecord')] if n.endswith('.tfrecord') else n for n in names]
    missing = [k for k in keys if k not in d]
    if missing or not keys:
        raise KeyError(f'{len(missing)} of {len(keys)} train segments missing from {calib_path}: {missing[:3]}')
    inc = np.array([np.sort(np.asarray(d[k]['inclinations'], np.float64).ravel()) for k in keys])
    z = tuple(sorted({round(float(np.asarray(d[k]['extrinsic'])[2, 3]), 4) for k in keys}))
    return np.median(inc, 0), len(keys), z


def waymo_top_train_lattice(calib_path=WAYMO_TOP_CALIB, train_list=WAYMO_TRAIN_LIST, n_cols=2650, height=2.184):
    """Waymo TOP lattice from the published calibration of the TRAIN segments only (see the module docstring)."""
    thetas, n, z = _train_median(calib_path, train_list)
    return dict(thetas=thetas.copy(), n_cols=int(n_cols), height=float(height), n_segments=n, extrinsic_z=z)


def lattice_from_config(cfg):
    p = {**DEFAULTS, **{k: cfg[k] for k in cfg if k in DEFAULTS}}
    if p['LATTICE'] != 'waymo_top_train':
        raise ValueError(f"relocate_render: unknown LATTICE {p['LATTICE']!r} (waymo_top_train)")
    lat = waymo_top_train_lattice(p['CALIB'], p['TRAIN_SEGMENTS'], p['N_COLS'], p['MOUNT_HEIGHT'])
    if not all(abs(z - lat['height']) < 1e-3 for z in lat['extrinsic_z']):
        raise ValueError(f"MOUNT_HEIGHT {lat['height']} disagrees with the published extrinsic z {lat['extrinsic_z']}")
    return lat


# ------------------------------------------------------------------------------------------------ box geometry
def bev_corners(boxes, grow=0.0):
    """(N, 4, 2) BEV corners of (N, >= 7) boxes, each side moved out by `grow`."""
    b = np.asarray(boxes, np.float64).reshape(-1, np.shape(boxes)[-1])
    hx = b[:, 3] / 2 + grow; hy = b[:, 4] / 2 + grow
    c, s = np.cos(b[:, 6]), np.sin(b[:, 6])
    lx = np.array([1, 1, -1, -1], np.float64); ly = np.array([1, -1, -1, 1], np.float64)
    x = b[:, :1] + (lx * hx[:, None]) * c[:, None] - (ly * hy[:, None]) * s[:, None]
    y = b[:, 1:2] + (lx * hx[:, None]) * s[:, None] + (ly * hy[:, None]) * c[:, None]
    return np.stack([x, y], -1)


def rects_overlap(a, bs):
    """True where the convex quad a (4, 2) and each quad of bs (N, 4, 2) intersect with positive area (separating
    axes; touching edges do not count)."""
    bs = np.asarray(bs, np.float64).reshape(-1, 4, 2)
    if not len(bs):
        return np.zeros(0, bool)
    a = np.asarray(a, np.float64)
    out = np.ones(len(bs), bool)
    axes = [a[1] - a[0], a[3] - a[0]]
    for ax in axes:
        n = np.array([-ax[1], ax[0]])
        pa = a @ n; pb = bs @ n
        out &= ~((pa.max() <= pb.min(1)) | (pb.max(1) <= pa.min()))
    for k in (1, 3):
        e = bs[:, k] - bs[:, 0]
        n = np.stack([-e[:, 1], e[:, 0]], 1)
        pa = np.einsum('ij,nj->ni', a, n); pb = np.einsum('nij,nj->ni', bs, n)
        out &= ~((pa.max(1) <= pb.min(1)) | (pb.max(1) <= pa.min(1)))
    return out


def points_in_box(xyz, box):
    """Mask of points inside a 3D box with roiaware_pool3d.points_in_boxes_cpu's test (float32 inputs, the kernel's
    1 cm BEV margin): |z - cz| <= dz / 2, |local x| < dx / 2 + 0.01, |local y| < dy / 2 + 0.01."""
    p = np.asarray(xyz, np.float32)[:, :3]; b = np.asarray(box, np.float32)[:7]
    if not len(p):
        return np.zeros(0, bool)
    sx = p[:, 0] - b[0]; sy = p[:, 1] - b[1]
    ca = np.cos(-b[6]); sa = np.sin(-b[6])
    lx = sx * ca + sy * (-sa); ly = sx * sa + sy * ca
    m = np.float64(np.float32(1e-2))
    inz = np.abs(p[:, 2] - b[2]).astype(np.float64) <= np.float64(b[5]) / 2.0
    return inz & (np.abs(lx).astype(np.float64) < np.float64(b[3]) / 2.0 + m) & \
        (np.abs(ly).astype(np.float64) < np.float64(b[4]) / 2.0 + m)


def points_in_rect(xy, box, grow):
    """Mask of points whose BEV position lies in the box's rectangle grown by `grow` on every side (closed)."""
    p = np.asarray(xy, np.float64)
    if not len(p):
        return np.zeros(0, bool)
    b = np.asarray(box, np.float64)
    d0 = p[:, 0] - b[0]; d1 = p[:, 1] - b[1]
    c, s = np.cos(b[6]), np.sin(b[6])
    lx = d0 * c + d1 * s; ly = -d0 * s + d1 * c
    return (np.abs(lx) <= b[3] / 2 + grow) & (np.abs(ly) <= b[4] / 2 + grow)


# ----------------------------------------------------------------------------------------- object-level API
def relocate(box, xyz, r_new, phi_new):
    """See the module docstring. box (>= 7,), xyz (k, >= 3) -> (box (same length), xyz (k, 3) float64)."""
    b = np.asarray(box, np.float64).ravel().copy()
    r = float(np.hypot(b[0], b[1]))
    if not r > 0:
        raise ValueError('a box at r = 0 has no azimuth')
    d = float(phi_new) - float(np.arctan2(b[1], b[0]))
    c, s = np.cos(d), np.sin(d)
    u = np.array([np.cos(phi_new), np.sin(phi_new)])
    nb = b.copy()
    nb[0:2] = float(r_new) * u
    nb[6] = b[6] + d
    if len(b) >= 9:
        nb[7:9] = [c * b[7] - s * b[8], s * b[7] + c * b[8]]
    p = np.asarray(xyz, np.float64).reshape(len(xyz), -1)[:, :3].copy()
    x, y = p[:, 0].copy(), p[:, 1].copy()
    p[:, 0] = c * x - s * y + (float(r_new) - r) * u[0]
    p[:, 1] = s * x + c * y + (float(r_new) - r) * u[1]
    return nb, p


def _render(xyz, thetas, n_cols, height):
    """render_object plus the kept points' ranges."""
    p = np.asarray(xyz, np.float64).reshape(-1, 3)
    empty = (np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    if not len(p):
        return empty
    row, col, rng, valid = to_lattice(p, height, thetas, n_cols)
    sel = np.nonzero(valid)[0]
    if not len(sel):
        return empty
    cell = row[sel] * n_cols + col[sel]
    o = sel[np.lexsort((rng[sel], cell))]
    c = row[o] * n_cols + col[o]
    first = np.ones(len(o), bool); first[1:] = c[1:] != c[:-1]
    k = o[first]
    rr, th = rng[k], np.asarray(thetas, np.float64)[row[k]]
    az = (col[k] + 0.5) / n_cols * 2 * np.pi - np.pi
    out = np.stack([rr * np.cos(th) * np.cos(az), rr * np.cos(th) * np.sin(az), height + rr * np.sin(th)], 1)
    return out, row[k], col[k], k, rr


def render_object(xyz, thetas, n_cols, height):
    """See the module docstring (port of tools/analysis/source_coverage.py `render`, ST3D ec2526a)."""
    return _render(np.asarray(xyz, np.float64)[:, :3] if np.ndim(xyz) == 2 else xyz, thetas, n_cols, height)[:4]


def render_object_at(box, points, r_new, phi_new, lat, time_column=None):
    """relocate + render_object, carrying feature columns (see the module docstring)."""
    pts = np.asarray(points)
    nb, q = relocate(box, pts[:, :3], r_new, phi_new)
    xyz, rows, cols, kept, _ = _render(q, lat['thetas'], lat['n_cols'], lat['height'])
    out = np.zeros((len(kept), pts.shape[1]), pts.dtype)
    out[:, :3] = xyz
    if pts.shape[1] > 3:
        out[:, 3:] = pts[kept, 3:]
        if time_column is not None and time_column < pts.shape[1]:
            out[:, time_column] = 0
    return dict(box=nb, moved=q, points=out, rows=rows, cols=cols, kept=kept)


# ------------------------------------------------------------------------------------------ scene-level helpers
class _BevGrid:
    """Points bucketed by 1 m BEV cell (radix sort on uint16 ids), for rectangle candidate queries."""
    CELL, LIM = 1.0, 80.0

    def __init__(self, xy):
        n = int(np.ceil(2 * self.LIM / self.CELL)); self.n = n
        ix = np.clip(np.floor((xy[:, 0] + self.LIM) / self.CELL), 0, n - 1).astype(np.int64)
        iy = np.clip(np.floor((xy[:, 1] + self.LIM) / self.CELL), 0, n - 1).astype(np.int64)
        cid = (ix * n + iy).astype(np.uint16)
        self.order = np.argsort(cid, kind='stable')
        self.start = np.concatenate([[0], np.cumsum(np.bincount(cid, minlength=n * n))])

    def query(self, corners, pad=0.05):
        """Indices of points in the cells overlapping the AABB of `corners` (k, 2), padded."""
        lo = np.floor((corners.min(0) - pad + self.LIM) / self.CELL).astype(np.int64)
        hi = np.floor((corners.max(0) + pad + self.LIM) / self.CELL).astype(np.int64)
        lo = np.clip(lo, 0, self.n - 1); hi = np.clip(hi, 0, self.n - 1)
        parts = [self.order[self.start[ix * self.n + lo[1]]:self.start[ix * self.n + hi[1] + 1]]
                 for ix in range(lo[0], hi[0] + 1)]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)


class _ColIndex:
    """Points sorted by lattice column, for 'points in these columns' queries."""

    def __init__(self, xy, n_cols):
        az = np.arctan2(xy[:, 1], xy[:, 0])
        col = np.clip(np.floor((az + np.pi) / (2 * np.pi) * n_cols), 0, n_cols - 1).astype(np.uint16)
        self.order = np.argsort(col, kind='stable')
        self.start = np.concatenate([[0], np.cumsum(np.bincount(col, minlength=n_cols))])

    def query(self, cols):
        cols = np.unique(cols)
        if not len(cols):
            return np.zeros(0, np.int64)
        return np.concatenate([self.order[self.start[c]:self.start[c + 1]] for c in cols])


def _params(cfg):
    p = dict(DEFAULTS)
    if cfg is not None:
        for k in cfg:
            if k in DEFAULTS:
                p[k] = cfg[k]
    return p


def donor_mask(data_dict, donor_class='Car', r_max=75.2):
    """Boxes that may serve as donors before the point count: the donor class by name, gt_boxes_mask true, not an
    ignore box (gt_boxes_ignore_mask, or gt_classes <= 0 when present), 0 < BEV range < r_max. Labels are compared,
    never used as indices."""
    boxes = np.asarray(data_dict['gt_boxes'])
    n = len(boxes)
    if n == 0:
        return np.zeros(0, bool)
    m = np.asarray(data_dict['gt_names']) == donor_class
    if 'gt_boxes_mask' in data_dict:
        m &= np.asarray(data_dict['gt_boxes_mask'], bool)
    ign = data_dict.get('gt_boxes_ignore_mask', None)
    if ign is not None:
        m &= ~np.asarray(ign, bool)
    cls = data_dict.get('gt_classes', None)
    if cls is not None:
        m &= np.asarray(cls) > 0
    r = np.hypot(boxes[:, 0], boxes[:, 1])
    return m & (r > 0) & (r < r_max)


# ------------------------------------------------------------------------------------------- scene-level API
def relocate_render_scene(data_dict, params=None, lat=None, rng=np.random, keep_points=False):
    """See the module docstring. keep_points=True stores each copy's moved donor xyz and rendered xyz in the record
    (for the gate's phantom check); the loader leaves it False."""
    p = _params(params)
    if lat is None:
        lat = lattice_from_config(p)
    thetas, n_cols, height = lat['thetas'], lat['n_cols'], lat['height']
    rec = dict(n_boxes=0, n_candidates=0, n_eligible=0, n_placed=0, n_no_place=0, n_zero=0,
               reject=dict(overlap=0, range=0, scene=0), copies=[], removed_orig=np.zeros(0, np.int64))
    pts = data_dict.get('points', None)
    boxes = data_dict.get('gt_boxes', None)
    if pts is None or boxes is None or len(pts) == 0 or len(boxes) == 0:
        return data_dict, rec
    pts = np.asarray(pts); boxes = np.asarray(boxes)
    n_box = len(boxes); rec['n_boxes'] = n_box
    cand = np.nonzero(donor_mask(data_dict, p['DONOR_CLASS'], p['R_MAX']))[0]
    rec['n_candidates'] = int(len(cand))
    if not len(cand):
        return data_dict, rec

    xyz = pts[:, :3]
    grid = _BevGrid(xyz)
    donors = []
    for j in cand:
        c = grid.query(bev_corners(boxes[j:j + 1], 0.02)[0])
        idx = c[points_in_box(xyz[c], boxes[j])]
        if len(idx) >= p['MIN_DONOR_POINTS']:
            donors.append((int(j), np.sort(idx)))
    rec['n_eligible'] = len(donors)
    if not donors:
        return data_dict, rec

    pcr = np.asarray(p['POINT_CLOUD_RANGE'], np.float64)
    alive = np.ones(len(pts), bool)
    colidx = None
    # added (copy) points: xyzf rows, owning copy, range about the virtual sensor, lattice pixel
    add_pts = np.zeros((0, pts.shape[1]), pts.dtype); add_id = np.zeros(0, np.int64)
    add_rng = np.zeros(0); add_pix = np.zeros(0, np.int64); add_alive = np.zeros(0, bool)
    placed_boxes = []
    rows_gt = boxes[:, :7].astype(np.float64)
    gt_corners = bev_corners(rows_gt)
    removed_orig = []

    for di in rng.permutation(len(donors)):
        if len(placed_boxes) >= p['MAX_COPIES']:
            break
        j, idx = donors[di]
        b = boxes[j]
        r_d = float(np.hypot(b[0], b[1]))
        nb = q = None
        tries = 0
        for _ in range(int(p['MAX_TRIES'])):
            tries += 1
            r_new = rng.uniform(r_d, p['R_MAX']); phi_new = rng.uniform(0.0, 2 * np.pi)
            cb, _q = relocate(b, xyz[idx], r_new, phi_new)
            g = bev_corners(cb[None], p['BEV_MARGIN'])[0]
            if (g[:, 0] < pcr[0]).any() or (g[:, 0] > pcr[3]).any() or (g[:, 1] < pcr[1]).any() or \
                    (g[:, 1] > pcr[4]).any():
                rec['reject']['range'] += 1; continue
            others = gt_corners if not placed_boxes else \
                np.concatenate([gt_corners, bev_corners(np.array([pb[:7] for pb in placed_boxes]))])
            if rects_overlap(g, others).any():
                rec['reject']['overlap'] += 1; continue
            zb = cb[2] - cb[5] / 2 + p['CLEARANCE']
            c = grid.query(g)
            c = c[alive[c] & (xyz[c, 2] > zb)]
            n_sc = int(points_in_rect(xyz[c, :2], cb, p['BEV_MARGIN']).sum())
            if len(add_pts):
                a = add_alive & (add_pts[:, 2] > zb)
                n_sc += int(points_in_rect(add_pts[a, :2], cb, p['BEV_MARGIN']).sum())
            if n_sc > p['MAX_SCENE_POINTS']:
                rec['reject']['scene'] += 1; continue
            nb, q = cb, _q
            break
        if nb is None:
            rec['n_no_place'] += 1
            continue

        rxyz, rrow, rcol, kept, rr = _render(q, thetas, n_cols, height)
        if not len(kept):                                            # nothing inside the lattice's field of view
            rec['n_zero'] += 1
            continue
        cpix = rrow * n_cols + rcol                                  # one rendered point per pixel
        # (1) scene points inside the copy's box
        c = grid.query(bev_corners(nb[None], 0.02)[0])
        c = c[alive[c]]
        inb = c[points_in_box(xyz[c], nb)]
        alive[inb] = False
        add_in = np.zeros(len(add_pts), bool)
        if len(add_pts):
            add_in = add_alive & points_in_box(add_pts[:, :3], nb)
            add_alive &= ~add_in
        # (2) scene points in the copy's pixels
        if colidx is None:
            colidx = _ColIndex(xyz[:, :2], n_cols)
        sc = colidx.query(rcol)
        sc = sc[alive[sc]]
        srow, scol, srng, sval = to_lattice(xyz[sc], height, thetas, n_cols)
        spix = np.where(sval, srow * n_cols + scol, -1)
        order = np.argsort(cpix); cs = cpix[order]; rs = rr[order]
        pos = np.clip(np.searchsorted(cs, spix), 0, len(cs) - 1)
        hit = (spix >= 0) & (cs[pos] == spix)
        apos = np.zeros(0, np.int64); ahit = np.zeros(len(add_pts), bool)
        if len(add_pts):
            apos = np.clip(np.searchsorted(cs, add_pix), 0, len(cs) - 1)
            ahit = add_alive & (add_pix >= 0) & (cs[apos] == add_pix)
        # (3) a copy point is dropped if its pixel holds a scene point nearer by more than OCCLUSION_M
        near = np.full(len(cs), np.inf)
        if hit.any():
            np.minimum.at(near, pos[hit], srng[hit])
        if ahit.any():
            np.minimum.at(near, apos[ahit], add_rng[ahit])
        keep_sorted = ~(near < rs - p['OCCLUSION_M'])
        # (4) shadow: scene points farther than the copy point in every pixel the copy renders into
        sh = hit & (srng > rs[pos])
        shadow = sc[sh]
        alive[shadow] = False
        ash = (ahit & (add_rng > rs[apos])) if len(add_pts) else ahit
        add_alive &= ~ash
        keep = np.zeros(len(cpix), bool); keep[order[keep_sorted]] = True
        if not keep.any():                                           # MIN_POINTS_OF_GT = 1: undo, skip the donor
            alive[inb] = True; alive[shadow] = True
            if len(add_pts):
                add_alive |= add_in | ash
            rec['n_zero'] += 1
            continue
        removed_orig.extend([inb, shadow])
        cid = len(placed_boxes)
        newp = np.zeros((int(keep.sum()), pts.shape[1]), pts.dtype)
        newp[:, :3] = rxyz[keep]
        if pts.shape[1] > 3:
            newp[:, 3:] = pts[idx[kept[keep]], 3:]
            tc = p['TIME_COLUMN']
            if tc is not None and 3 <= int(tc) < pts.shape[1]:
                newp[:, int(tc)] = 0
        add_pts = np.concatenate([add_pts, newp]); add_id = np.concatenate([add_id, np.full(len(newp), cid)])
        add_rng = np.concatenate([add_rng, rr[keep]]); add_pix = np.concatenate([add_pix, cpix[keep]])
        add_alive = np.concatenate([add_alive, np.ones(len(newp), bool)])
        placed_boxes.append(nb.astype(boxes.dtype))
        crec = dict(donor=j, donor_box=np.asarray(b, np.float64).copy(), r_donor=r_d,
                    phi_donor=float(np.arctan2(b[1], b[0])), box=nb.copy(), r_new=float(np.hypot(nb[0], nb[1])),
                    phi_new=float(np.arctan2(nb[1], nb[0])), tries=tries, n_donor=int(len(idx)),
                    n_rendered=int(len(cpix)), n_occluded=int((~keep).sum()), n_kept=int(keep.sum()),
                    n_inbox_removed=int(len(inb) + add_in.sum()), n_shadow_removed=int(len(shadow) + ash.sum()))
        if keep_points:
            crec['moved'] = q.copy(); crec['rendered'] = rxyz[keep].copy()
        rec['copies'].append(crec)

    if not placed_boxes:
        return data_dict, rec
    k = len(placed_boxes)
    out_add = add_pts[add_alive]
    data_dict['points'] = np.concatenate([pts[alive], out_add]).astype(pts.dtype, copy=False)
    n_alive = int(alive.sum())
    ids = add_id[add_alive]
    for cid, crec in enumerate(rec['copies']):
        crec['point_index'] = n_alive + np.nonzero(ids == cid)[0]
        crec['n_final'] = int(len(crec['point_index']))
    data_dict['gt_boxes'] = np.concatenate([boxes, np.stack(placed_boxes)]).astype(boxes.dtype, copy=False)
    donor_ids = [c['donor'] for c in rec['copies']]
    data_dict['gt_names'] = np.concatenate([np.asarray(data_dict['gt_names']),
                                            np.asarray(data_dict['gt_names'])[donor_ids]])
    if 'gt_boxes_mask' in data_dict:
        data_dict['gt_boxes_mask'] = np.concatenate([np.asarray(data_dict['gt_boxes_mask'], bool), np.ones(k, bool)])
    if data_dict.get('gt_boxes_ignore_mask', None) is not None:
        data_dict['gt_boxes_ignore_mask'] = np.concatenate(
            [np.asarray(data_dict['gt_boxes_ignore_mask'], bool), np.zeros(k, bool)])
    for key in ('gt_classes', 'gt_scores'):
        if data_dict.get(key, None) is not None and len(data_dict[key]) == n_box:
            v = np.asarray(data_dict[key]); data_dict[key] = np.concatenate([v, v[donor_ids]])
    if data_dict.get('num_points_in_gt', None) is not None and len(data_dict['num_points_in_gt']) == n_box:
        v = np.asarray(data_dict['num_points_in_gt'])
        data_dict['num_points_in_gt'] = np.concatenate([v, np.array([c['n_final'] for c in rec['copies']], v.dtype)])
    rec['n_placed'] = k
    rec['removed_orig'] = np.unique(np.concatenate(removed_orig)) if removed_orig else np.zeros(0, np.int64)
    rec['alive_orig'] = alive
    return data_dict, rec


# =========================================================================================================== RR v2
# experiments_md 20261018_02 §6 (design fixed in T05 §5, experiments_md 4e290c1): a track-level donor database, the
# copy's bottom on a per-frame RANSAC ground plane, a visibility rule, row-coverage re-placement and a uniform random
# sparsifier. Reached only by a config with VERSION: 2 (absent = v1, unchanged).
DEFAULTS_V2 = dict(N_COPIES=8, R_MIN=5.0, RC_MIN=0.8, VIS_MAX_SHARE=0.1, VIS_HEIGHT=0.3, RC_OCCL_M=0.5,
                   COPY_CLASS_LABEL=1, DB_POINTS=None, DB_INDEX=None, VOXEL=0.02)
# ground plane: tools/analysis/ground_tilt.py (20261004_02), ported verbatim in its parameters
GP_RMIN, GP_RMAX, GP_CELL, GP_MIN_PTS, GP_ITERS, GP_THR, GP_REFIT, GP_MAX_SLOPE, GP_MIN_CELLS = \
    4.0, 40.0, 1.0, 3, 300, 0.12, 0.08, 0.2, 30


def lowest_per_cell(pts):
    """ground_tilt.lowest_per_cell(pts) (4-40 m, 1 m cells, >= 3 points, the SECOND-lowest return per cell), in the same
    output order (cells by (floor x, floor y)), with a radix sort instead of a lexsort."""
    p = np.asarray(pts)
    r = np.hypot(p[:, 0], p[:, 1])
    p = p[(r >= GP_RMIN) & (r <= GP_RMAX)]
    if len(p) == 0:
        return np.zeros((0, 3))
    ij = np.floor(p[:, :2] / GP_CELL).astype(np.int64)
    off = int(np.ceil(GP_RMAX / GP_CELL)) + 1; w = 2 * off + 1
    key = ((ij[:, 0] + off) * w + (ij[:, 1] + off)).astype(np.uint16 if w * w < 65536 else np.int64)
    order = np.argsort(key, kind='stable')
    k = key[order]; z = np.asarray(p[order, 2], np.float64)
    first = np.r_[True, k[1:] != k[:-1]]
    starts = np.flatnonzero(first); counts = np.diff(np.r_[starts, len(k)])
    zmin = np.minimum.reduceat(z, starts)
    pos = np.arange(len(z))
    ismin = z == np.repeat(zmin, counts)
    fmin = np.minimum.reduceat(np.where(ismin, pos, len(z)), starts)          # first occurrence of the minimum
    z2 = z.copy(); z2[fmin] = np.inf
    zsec = np.minimum.reduceat(z2, starts)
    issec = (z2 == np.repeat(zsec, counts)) & np.isfinite(z2)
    fsec = np.minimum.reduceat(np.where(issec, pos, len(z)), starts)
    take = np.where(counts >= 2, fsec, fmin)
    keep = counts >= GP_MIN_PTS
    return np.asarray(p[order[take[keep]], :3], np.float64)


def fit_plane(g, rng=None):
    """ground_tilt.fit_plane(g) with its parameters (300 iterations, 0.12 m, slopes <= 0.2, refits at 0.12 / 0.08 m).
    Returns (a, b, c) of z = a x + b y + c, or None. The 300 candidate triples are drawn with the same sequential
    rng.choice calls as ground_tilt; solving and counting are batched (same result: the first candidate with the most
    inliers); a singular triple falls back to ground_tilt's loop."""
    if len(g) < GP_MIN_CELLS:
        return None
    rng = rng or np.random.default_rng(0)
    A = np.c_[g[:, 0], g[:, 1], np.ones(len(g))]
    tri = np.array([rng.choice(len(g), 3, replace=False) for _ in range(GP_ITERS)])
    try:
        coefs = np.linalg.solve(A[tri], g[tri, 2][..., None])[..., 0]
        okc = (np.abs(coefs[:, 0]) <= GP_MAX_SLOPE) & (np.abs(coefs[:, 1]) <= GP_MAX_SLOPE)
        if not okc.any():
            return None
        n = (np.abs(A @ coefs[okc].T - g[:, 2:3]) < GP_THR).sum(0)
        best = coefs[okc][int(np.argmax(n))]
    except np.linalg.LinAlgError:
        best, best_n = None, -1
        for idx in tri:
            try:
                coef = np.linalg.solve(A[idx], g[idx, 2])
            except np.linalg.LinAlgError:
                continue
            if abs(coef[0]) > GP_MAX_SLOPE or abs(coef[1]) > GP_MAX_SLOPE:
                continue
            nn = np.sum(np.abs(A @ coef - g[:, 2]) < GP_THR)
            if nn > best_n:
                best, best_n = coef, nn
        if best is None:
            return None
    for t in (GP_THR, GP_REFIT):
        inl = np.abs(A @ best - g[:, 2]) < t
        best = np.linalg.lstsq(A[inl], g[inl, 2], rcond=None)[0]
    return best


def ground_plane(xyz):
    """The frame's ground plane (a, b, c), RANSAC as ground_tilt.py, numpy default_rng(0) per frame; None if none."""
    return fit_plane(lowest_per_cell(xyz), np.random.default_rng(0))


def plane_z(plane, x, y):
    return plane[0] * np.asarray(x) + plane[1] * np.asarray(y) + plane[2]


# ------------------------------------------------------------------------------------------- box-frame transforms
def to_box_frame(xyz, box):
    """Points (k, 3) in the loader frame -> the box frame (centre at 0, x along the heading)."""
    p = np.asarray(xyz, np.float64)[:, :3]; b = np.asarray(box, np.float64)
    c, s = np.cos(b[6]), np.sin(b[6])
    dx, dy = p[:, 0] - b[0], p[:, 1] - b[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, p[:, 2] - b[2]], 1)


def from_box_frame(local, box):
    """Inverse of to_box_frame."""
    q = np.asarray(local, np.float64)[:, :3]; b = np.asarray(box, np.float64)
    c, s = np.cos(b[6]), np.sin(b[6])
    return np.stack([b[0] + c * q[:, 0] - s * q[:, 1], b[1] + s * q[:, 0] + c * q[:, 1], b[2] + q[:, 2]], 1)


def relative_view_angle(box):
    """wrap(heading - azimuth of the box centre): the heading relative to the line of sight."""
    b = np.asarray(box, np.float64)
    return float((b[6] - np.arctan2(b[1], b[0]) + np.pi) % (2 * np.pi) - np.pi)


class VoxelAccumulator:
    """Exact centroid voxel downsampling of a union of point sets in one frame (sums and counts per voxel)."""

    def __init__(self, voxel=0.02, n_feat=4):
        self.voxel, self.n_feat = float(voxel), n_feat
        self.keys = np.zeros(0, np.int64); self.sums = np.zeros((0, n_feat)); self.cnt = np.zeros(0, np.int64)
        self.n_raw = 0

    def add(self, pts):
        p = np.asarray(pts, np.float64)[:, :self.n_feat]
        if not len(p):
            return
        self.n_raw += len(p)
        ijk = np.floor(p[:, :3] / self.voxel).astype(np.int64) + (1 << 20)
        key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
        keys = np.concatenate([self.keys, key]); sums = np.concatenate([self.sums, p])
        cnt = np.concatenate([self.cnt, np.ones(len(p), np.int64)])
        u, inv = np.unique(keys, return_inverse=True)
        s = np.zeros((len(u), self.n_feat)); np.add.at(s, inv, sums)
        c = np.bincount(inv, weights=cnt, minlength=len(u)).astype(np.int64)
        self.keys, self.sums, self.cnt = u, s, c

    def points(self):
        return self.sums / self.cnt[:, None] if len(self.cnt) else np.zeros((0, self.n_feat))


class DonorDB:
    """The platform's donor database: points memory-mapped read-only on first use (never pickled), index in memory."""

    def __init__(self, points_path, index_path):
        self.points_path, self.index_path = str(points_path), str(index_path)
        z = np.load(self.index_path, allow_pickle=False)
        self.offset, self.count = z['offset'].astype(np.int64), z['count'].astype(np.int64)
        self.size = z['size'].astype(np.float64)
        self.obs_offset, self.obs_count = z['obs_offset'].astype(np.int64), z['obs_count'].astype(np.int64)
        self.obs_alpha = z['obs_alpha'].astype(np.float64)
        self.n = len(self.count)
        self._pts = None

    def __getstate__(self):
        d = dict(self.__dict__); d['_pts'] = None
        return d

    def _points(self):
        if self._pts is None:
            self._pts = np.load(self.points_path, mmap_mode='r')
        return self._pts

    def points(self, i):
        return np.asarray(self._points()[self.offset[i]:self.offset[i] + self.count[i]], np.float64)

    def alphas(self, i):
        return self.obs_alpha[self.obs_offset[i]:self.obs_offset[i] + self.obs_count[i]]


def db_from_config(cfg):
    return DonorDB(cfg['DB_POINTS'], cfg['DB_INDEX'])


def place_instance(local, size, r, phi, alpha, plane, bottom_fn=None):
    """Box (7,) and loader-frame points of a database instance at range r, azimuth phi, heading phi + alpha, the box's
    bottom on the plane at its BEV centre (v2), or at bottom_fn(x, y) (v3)."""
    x, y = r * np.cos(phi), r * np.sin(phi)
    zb = float(plane_z(plane, x, y)) if bottom_fn is None else float(bottom_fn(x, y))
    box = np.array([x, y, zb + size[2] / 2, size[0], size[1], size[2], phi + alpha])
    return box, from_box_frame(local, box)


# ------------------------------------------------------------------------------------- silhouette and row coverage
def box_corners(box):
    b = np.asarray(box, np.float64)
    t = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
                  [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]], np.float64) / 2 * b[3:6]
    c, s = np.cos(b[6]), np.sin(b[6])
    return np.stack([t[:, 0] * c - t[:, 1] * s + b[0], t[:, 0] * s + t[:, 1] * c + b[1], t[:, 2] + b[2]], 1)


def silhouette(box, thetas, n_cols, height):
    """Lattice pixels whose ray from (0, 0, height) hits the box: (pixel ids, ASCENDING rows, entry range). Port of
    tools/analysis/lattice_sampler_precheck.py `silhouette` (rows there are descending; the hit set is the same)."""
    th = np.asarray(thetas, np.float64); b = np.asarray(box, np.float64)
    c = box_corners(b)
    el = np.arctan2(c[:, 2] - height, np.hypot(c[:, 0], c[:, 1])); az0 = np.arctan2(b[1], b[0])
    daz = (np.arctan2(c[:, 1], c[:, 0]) - az0 + np.pi) % (2 * np.pi) - np.pi
    rows = np.nonzero((th >= el.min() - 0.01) & (th <= el.max() + 0.01))[0]
    c_lo = int(np.floor((az0 + daz.min() + np.pi) / (2 * np.pi) * n_cols)) - 1
    c_hi = int(np.floor((az0 + daz.max() + np.pi) / (2 * np.pi) * n_cols)) + 1
    cols = np.arange(c_lo, c_hi + 1)
    if not len(rows) or len(cols) > n_cols // 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    R, C = np.meshgrid(rows, cols, indexing='ij'); R, C = R.ravel(), C.ravel()
    phi = th[R]; az = ((C % n_cols) + 0.5) / n_cols * 2 * np.pi - np.pi
    d = np.stack([np.cos(phi) * np.cos(az), np.cos(phi) * np.sin(az), np.sin(phi)], 1)
    ch, sh = np.cos(-b[6]), np.sin(-b[6])
    rot = lambda v: np.stack([v[..., 0] * ch - v[..., 1] * sh, v[..., 0] * sh + v[..., 1] * ch, v[..., 2]], -1)
    o = rot(np.array([0.0, 0.0, height]) - b[:3]); dd = rot(d); e = b[3:6] / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (-e - o) / dd; t2 = (e - o) / dd
    t1 = np.where(np.isfinite(t1), t1, -np.inf); t2 = np.where(np.isfinite(t2), t2, np.inf)
    tmin = np.minimum(t1, t2).max(1); tmax = np.maximum(t1, t2).min(1)
    hit = tmax >= np.maximum(tmin, 0)
    return (R[hit] * n_cols + C[hit] % n_cols), R[hit], np.maximum(tmin[hit], 0)


def row_coverage(box, copy_pix, copy_rng, scene_pix, scene_rng, thetas, n_cols, height, occl_m=0.5, sil=None):
    """RC of a copy exactly as `source_coverage.lattice_coverage` defines it: rows of the box's VISIBLE silhouette holding
    a copy point / rows of the visible silhouette. A silhouette pixel is occluded when the nearest return of the pixel
    (copy and scene points together) is not a copy point and lies more than occl_m in front of the box entry. Pixels are
    ascending-row ids (-1 = outside the field of view). Returns (RC or nan, number of visible rows)."""
    sp, srow, tin = silhouette(box, thetas, n_cols, height) if sil is None else sil
    if not len(sp):
        return np.nan, 0
    cp = np.asarray(copy_pix, np.int64); sc = np.asarray(scene_pix, np.int64)
    pix = np.concatenate([cp, sc]); rng = np.concatenate([np.asarray(copy_rng, np.float64),
                                                          np.asarray(scene_rng, np.float64)])
    inbox = np.r_[np.ones(len(cp), bool), np.zeros(len(sc), bool)]
    ok = np.nonzero(pix >= 0)[0]
    occl = np.zeros(len(sp), bool)
    if len(ok):
        o = ok[np.lexsort((rng[ok], pix[ok]))]
        first = np.ones(len(o), bool); first[1:] = pix[o][1:] != pix[o][:-1]
        nidx = o[first]; up = pix[nidx]
        pos = np.clip(np.searchsorted(up, sp), 0, len(up) - 1); has = up[pos] == sp
        occl = has & (rng[nidx][pos] < tin - occl_m) & ~inbox[nidx][pos]
    vp, vrow = sp[~occl], srow[~occl]
    if not len(vrow):
        return np.nan, 0
    cpv = cp[cp >= 0]
    occ = np.unique(cpv[np.isin(cpv, vp)])
    nvr = len(np.unique(vrow))
    return len(np.unique(occ // n_cols)) / nvr, nvr


def _randint(rng, n):
    return int(rng.randint(n)) if hasattr(rng, 'randint') else int(rng.integers(n))


def _params_v2(cfg):
    p = dict(DEFAULTS); p.update(DEFAULTS_V2); p.update(DEFAULTS_V3); p['VERSION'] = 2
    if cfg is not None:
        for k in cfg:
            if k in p:
                p[k] = cfg[k]
    return p


def relocate_render_scene_v2(data_dict, params=None, lat=None, db=None, rng=np.random, keep_points=False):
    """RR v2 (experiments_md 20261018_02 §6). See the module's v2 block; v1 helpers are reused unchanged."""
    p = _params_v2(params)
    if lat is None:
        lat = lattice_from_config(p)
    if db is None:
        db = db_from_config(p)
    thetas, n_cols, height = lat['thetas'], lat['n_cols'], lat['height']
    rec = dict(version=2, n_boxes=0, plane=None, n_attempted=0, n_placed=0, n_no_place=0, n_zero=0,
               reject=dict(range=0, overlap=0, scene=0, empty=0, visibility=0, rc=0), copies=[],
               removed_orig=np.zeros(0, np.int64))
    pts = data_dict.get('points', None)
    boxes = data_dict.get('gt_boxes', None)
    if pts is None or boxes is None or len(pts) == 0 or db.n == 0:
        return data_dict, rec
    pts = np.asarray(pts); boxes = np.asarray(boxes)
    if boxes.ndim != 2:
        boxes = boxes.reshape(-1, 7)
    n_box = len(boxes); rec['n_boxes'] = n_box
    xyz = pts[:, :3]
    plane = ground_plane(xyz)
    if plane is None:
        return data_dict, rec
    rec['plane'] = np.asarray(plane, np.float64)
    v3 = int(p['VERSION']) == 3
    rec['version'] = 3 if v3 else 2
    if v3:                                          # §7: ground cells of the cloud RR receives, built once per frame
        cells = GroundCells(xyz, p['GROUND_CELL'], int(p['GROUND_CELL_MIN_PTS']))
        last_fb = [False]

        def bottom_fn(x, y):
            zb, fb = local_bottom(cells, x, y, plane, p['GROUND_RADIUS'], int(p['GROUND_MIN_CELLS']), p['GROUND_PCT'])
            last_fb[0] = fb
            return zb
    else:
        bottom_fn = None
    pcr = np.asarray(p['POINT_CLOUD_RANGE'], np.float64)
    grid = _BevGrid(xyz)
    colidx = None
    alive = np.ones(len(pts), bool)
    add_pts = np.zeros((0, pts.shape[1]), pts.dtype); add_id = np.zeros(0, np.int64)
    add_rng = np.zeros(0); add_pix = np.zeros(0, np.int64); add_alive = np.zeros(0, bool)
    placed_boxes = []
    occ_corners = bev_corners(boxes[:, :7].astype(np.float64)) if n_box else np.zeros((0, 4, 2))
    removed_orig = []
    vis_h = float(p['VIS_HEIGHT'])

    for _ in range(int(p['N_COPIES'])):
        rec['n_attempted'] += 1
        inst = _randint(rng, db.n)
        local = db.points(inst); size = db.size[inst]; alphas = db.alphas(inst)
        lxyz = local[:, :3]
        tries, fresh, acc = 0, True, None
        r = phi = alpha = None
        while tries < int(p['MAX_TRIES']):
            tries += 1
            if fresh:
                r = rng.uniform(p['R_MIN'], p['R_MAX']); phi = rng.uniform(0.0, 2 * np.pi)
                alpha = float(alphas[_randint(rng, len(alphas))])
            else:
                r = rng.uniform(r, p['R_MAX'])
            fresh = True
            cb, q = place_instance(lxyz, size, r, phi, alpha, plane, bottom_fn)
            # v1 free-space check
            g = bev_corners(cb[None], p['BEV_MARGIN'])[0]
            if (g[:, 0] < pcr[0]).any() or (g[:, 0] > pcr[3]).any() or (g[:, 1] < pcr[1]).any() or \
                    (g[:, 1] > pcr[4]).any():
                rec['reject']['range'] += 1; continue
            if len(occ_corners) and rects_overlap(g, occ_corners).any():
                rec['reject']['overlap'] += 1; continue
            zb = cb[2] - cb[5] / 2 + p['CLEARANCE']
            c = grid.query(g)
            c = c[alive[c] & (xyz[c, 2] > zb)]
            n_sc = int(points_in_rect(xyz[c, :2], cb, p['BEV_MARGIN']).sum())
            if len(add_pts):
                a = add_alive & (add_pts[:, 2] > zb)
                n_sc += int(points_in_rect(add_pts[a, :2], cb, p['BEV_MARGIN']).sum())
            if n_sc > p['MAX_SCENE_POINTS']:
                rec['reject']['scene'] += 1; continue
            # render
            rxyz, rrow, rcol, kept, rr = _render(q, thetas, n_cols, height)
            if not len(kept):
                rec['reject']['empty'] += 1; continue
            cpix = rrow * n_cols + rcol
            if colidx is None:
                colidx = _ColIndex(xyz[:, :2], n_cols)
            am_all = add_alive & ~points_in_box(add_pts[:, :3], cb) if len(add_pts) else np.zeros(0, bool)

            def scene_in(cols):
                """Scene points in these lattice columns: alive originals and alive copies, none inside the copy's box
                (they are removed on acceptance). Returns (original ids, row, col, range, valid, xyz)."""
                sc_ = colidx.query(cols)
                sc_ = sc_[alive[sc_]]
                sc_ = sc_[~points_in_box(xyz[sc_], cb)] if len(sc_) else sc_
                r_, c_, g_, v_ = to_lattice(xyz[sc_], height, thetas, n_cols)
                x_ = np.asarray(xyz[sc_], np.float64)
                if len(add_pts):
                    am = am_all & np.isin(add_pix % n_cols, cols)
                    if am.any():
                        ap = add_pts[am, :3].astype(np.float64)
                        ar, ac, ag, av = to_lattice(ap, height, thetas, n_cols)
                        r_, c_, g_, v_ = np.r_[r_, ar], np.r_[c_, ac], np.r_[g_, ag], np.r_[v_, av]
                        x_ = np.concatenate([x_, ap])
                return sc_, r_, c_, g_, v_, x_
            # visibility: columns with a nearer scene point > VIS_HEIGHT above the plane inside the copy's row span
            ucol, inv = np.unique(rcol, return_inverse=True)
            sc, srow, scol, srng, sval, sxyz = scene_in(ucol)
            rmin = np.full(len(ucol), np.inf); np.minimum.at(rmin, inv, rr)
            hz = sxyz[:, 2] - plane_z(plane, sxyz[:, 0], sxyz[:, 1]) if len(sxyz) else np.zeros(0)
            vm = sval & (srow >= rrow.min()) & (srow <= rrow.max()) & (hz > vis_h) & np.isin(scol, ucol)
            blocked = np.zeros(len(ucol), bool)
            if vm.any():
                near = np.full(len(ucol), np.inf)
                np.minimum.at(near, np.searchsorted(ucol, scol[vm]), srng[vm])
                blocked = near < rmin
            vshare = float(blocked.mean())
            if vshare >= p['VIS_MAX_SHARE']:
                rec['reject']['visibility'] += 1; continue
            # row coverage of the render in this scene: the scene of the silhouette's extra columns is added
            sil = silhouette(cb, thetas, n_cols, height)
            extra = np.setdiff1d(sil[0] % n_cols, ucol)
            if len(extra):
                e = scene_in(extra)
                n_o = len(sc); n_a = len(srow) - n_o; n_eo = len(e[0])
                # keep originals first, then copies, in both parts (the shadow step below indexes the originals)
                srow = np.r_[srow[:n_o], e[1][:n_eo], srow[n_o:], e[1][n_eo:]]
                scol = np.r_[scol[:n_o], e[2][:n_eo], scol[n_o:], e[2][n_eo:]]
                srng = np.r_[srng[:n_o], e[3][:n_eo], srng[n_o:], e[3][n_eo:]]
                sval = np.r_[sval[:n_o], e[4][:n_eo], sval[n_o:], e[4][n_eo:]]
                sc = np.r_[sc, e[0]]
            spix = np.where(sval, srow * n_cols + scol, -1)
            rc, _ = row_coverage(cb, cpix, rr, spix, srng, thetas, n_cols, height, p['RC_OCCL_M'], sil=sil)
            if not (rc >= p['RC_MIN']):
                rec['reject']['rc'] += 1; fresh = False; continue
            acc = (cb, q, rxyz, rrow, rcol, kept, rr, cpix, sc, srng, sval, spix, rc, vshare)
            break
        if acc is None:
            rec['n_no_place'] += 1
            continue
        cb, q, rxyz, rrow, rcol, kept, rr, cpix, sc, srng, sval, spix, rc, vshare = acc
        # v1 occlusion: (1) scene points inside the box
        cc = grid.query(bev_corners(cb[None], 0.02)[0]); cc = cc[alive[cc]]
        inb = cc[points_in_box(xyz[cc], cb)]
        alive[inb] = False
        add_in = np.zeros(len(add_pts), bool)
        if len(add_pts):
            add_in = add_alive & points_in_box(add_pts[:, :3], cb)
            add_alive &= ~add_in
        # (3)/(4) on the original scene candidates of the copy's columns, and on the alive copies
        order = np.argsort(cpix); cs = cpix[order]; rs = rr[order]
        n_orig = len(sc)
        osp, osr = spix[:n_orig], srng[:n_orig]
        pos = np.clip(np.searchsorted(cs, osp), 0, len(cs) - 1); hit = (osp >= 0) & (cs[pos] == osp)
        apos = np.zeros(0, np.int64); ahit = np.zeros(len(add_pts), bool)
        if len(add_pts):
            apos = np.clip(np.searchsorted(cs, add_pix), 0, len(cs) - 1)
            ahit = add_alive & (add_pix >= 0) & (cs[apos] == add_pix)
        nearest = np.full(len(cs), np.inf)
        if hit.any():
            np.minimum.at(nearest, pos[hit], osr[hit])
        if ahit.any():
            np.minimum.at(nearest, apos[ahit], add_rng[ahit])
        keep_sorted = ~(nearest < rs - p['OCCLUSION_M'])
        sh = hit & (osr > rs[pos])
        shadow = sc[sh]
        alive[shadow] = False
        ash = (ahit & (add_rng > rs[apos])) if len(add_pts) else ahit
        add_alive &= ~ash
        vis = np.zeros(len(cpix), bool); vis[order[keep_sorted]] = True        # after the 0.5 m occlusion drop
        # sparsifier: p ~ U(0, 1], each remaining rendered point kept with p
        pk = 1.0 - rng.uniform(0.0, 1.0)
        u = np.asarray(rng.uniform(0.0, 1.0, len(cpix)), np.float64)
        keep = vis & (u < pk)
        if not keep.any():
            alive[inb] = True; alive[shadow] = True
            if len(add_pts):
                add_alive |= add_in | ash
            rec['n_zero'] += 1
            continue
        removed_orig.extend([inb, shadow])
        cid = len(placed_boxes)
        newp = np.zeros((int(keep.sum()), pts.shape[1]), pts.dtype)
        newp[:, :3] = rxyz[keep]
        if pts.shape[1] > 3 and local.shape[1] > 3:
            newp[:, 3] = local[kept[keep], 3]
        add_pts = np.concatenate([add_pts, newp]); add_id = np.concatenate([add_id, np.full(len(newp), cid)])
        add_rng = np.concatenate([add_rng, rr[keep]]); add_pix = np.concatenate([add_pix, cpix[keep]])
        add_alive = np.concatenate([add_alive, np.ones(len(newp), bool)])
        cb_store = np.zeros(boxes.shape[1]) if boxes.ndim == 2 else np.zeros(7)
        cb_store[:7] = cb
        placed_boxes.append(cb_store.astype(boxes.dtype))
        occ_corners = np.concatenate([occ_corners, bev_corners(placed_boxes[-1][None, :7].astype(np.float64))])
        crec = dict(instance=inst, box=cb.copy(), r_new=float(np.hypot(cb[0], cb[1])), phi_new=float(phi), alpha=alpha,
                    tries=tries, rc_loader=float(rc), vis_share_loader=vshare, n_db=int(len(local)),
                    n_rendered=int(len(cpix)), n_visible=int(vis.sum()), p_keep=float(pk), n_kept=int(keep.sum()),
                    n_inbox_removed=int(len(inb) + add_in.sum()), n_shadow_removed=int(len(shadow) + ash.sum()),
                    ground_fallback=bool(last_fb[0]) if v3 else None, bottom=float(cb[2] - cb[5] / 2))
        if keep_points:
            crec['placed'] = q.copy(); crec['rendered_unsparsified'] = rxyz[vis].copy(); crec['rendered'] = rxyz[keep]
        rec['copies'].append(crec)

    if not placed_boxes:
        return data_dict, rec
    k = len(placed_boxes)
    data_dict['points'] = np.concatenate([pts[alive], add_pts[add_alive]]).astype(pts.dtype, copy=False)
    n_alive = int(alive.sum()); ids = add_id[add_alive]
    for cid, crec in enumerate(rec['copies']):
        crec['point_index'] = n_alive + np.nonzero(ids == cid)[0]
        crec['n_final'] = int(len(crec['point_index']))
    data_dict['gt_boxes'] = np.concatenate([boxes, np.stack(placed_boxes)]).astype(boxes.dtype, copy=False)
    data_dict['gt_names'] = np.concatenate([np.asarray(data_dict['gt_names']), np.array([p['DONOR_CLASS']] * k)])
    if 'gt_boxes_mask' in data_dict:
        data_dict['gt_boxes_mask'] = np.concatenate([np.asarray(data_dict['gt_boxes_mask'], bool), np.ones(k, bool)])
    if data_dict.get('gt_boxes_ignore_mask', None) is not None:
        data_dict['gt_boxes_ignore_mask'] = np.concatenate(
            [np.asarray(data_dict['gt_boxes_ignore_mask'], bool), np.zeros(k, bool)])
    if data_dict.get('gt_classes', None) is not None and len(data_dict['gt_classes']) == n_box:
        v = np.asarray(data_dict['gt_classes'])
        data_dict['gt_classes'] = np.concatenate([v, np.full(k, int(p['COPY_CLASS_LABEL']), v.dtype)])
    if data_dict.get('gt_scores', None) is not None and len(data_dict['gt_scores']) == n_box:
        v = np.asarray(data_dict['gt_scores']); data_dict['gt_scores'] = np.concatenate([v, np.ones(k, v.dtype)])
    if data_dict.get('num_points_in_gt', None) is not None and len(data_dict['num_points_in_gt']) == n_box:
        v = np.asarray(data_dict['num_points_in_gt'])
        data_dict['num_points_in_gt'] = np.concatenate([v, np.array([c['n_final'] for c in rec['copies']], v.dtype)])
    rec['n_placed'] = k
    rec['removed_orig'] = np.unique(np.concatenate(removed_orig)) if removed_orig else np.zeros(0, np.int64)
    rec['alive_orig'] = alive
    return data_dict, rec


# =========================================================================================================== RR v3
# experiments_md 20261018_02 §7 (design fixed in T05 §5, experiments_md 9334105): v2 with the copy's bottom height from the
# local ground cells (the 20th percentile of the second-lowest z of the 1 m cells within 5 m holding >= 3 points), the
# plane as fallback. Reached only by VERSION: 3; v1 and v2 unchanged.
DEFAULTS_V3 = dict(GROUND_CELL=1.0, GROUND_CELL_MIN_PTS=3, GROUND_RADIUS=5.0, GROUND_MIN_CELLS=3, GROUND_PCT=20.0)


class GroundCells:
    """1 m BEV cells of a cloud holding >= min_pts points, each valued at its second-lowest z (ties by point order, as
    ground_tilt.lowest_per_cell), on a dense grid over +-(LIM) m."""
    LIM = 82

    def __init__(self, xyz, cell=1.0, min_pts=3):
        self.cell = float(cell)
        p = np.asarray(xyz)
        n = 2 * self.LIM + 1
        self.vals = np.full((n, n), np.nan)
        if not len(p):
            return
        ij = np.floor(p[:, :2] / self.cell).astype(np.int64)
        ok = (np.abs(ij) < self.LIM).all(1)
        ij, z = ij[ok], np.asarray(p[ok, 2], np.float64)
        if not len(z):
            return
        key = ((ij[:, 0] + self.LIM) * n + (ij[:, 1] + self.LIM)).astype(np.uint16 if n * n < 65536 else np.int64)
        order = np.argsort(key, kind='stable')
        k = key[order]; zz = z[order]
        first = np.r_[True, k[1:] != k[:-1]]
        starts = np.flatnonzero(first); counts = np.diff(np.r_[starts, len(k)])
        pos = np.arange(len(zz))
        zmin = np.minimum.reduceat(zz, starts)
        fmin = np.minimum.reduceat(np.where(zz == np.repeat(zmin, counts), pos, len(zz)), starts)
        z2 = zz.copy(); z2[fmin] = np.inf
        zsec = np.minimum.reduceat(z2, starts)
        keep = counts >= min_pts
        kk = k[starts[keep]].astype(np.int64)
        self.vals[kk // n, kk % n] = zsec[keep]

    def local(self, x, y, radius=5.0):
        """Values of the ground cells whose centre lies within `radius` (BEV) of (x, y)."""
        c = self.cell; n = self.vals.shape[0]
        i0, i1 = int(np.floor((x - radius) / c)), int(np.floor((x + radius) / c))
        j0, j1 = int(np.floor((y - radius) / c)), int(np.floor((y + radius) / c))
        I = np.arange(max(i0, -self.LIM + 1), min(i1, self.LIM - 1) + 1)
        J = np.arange(max(j0, -self.LIM + 1), min(j1, self.LIM - 1) + 1)
        if not len(I) or not len(J):
            return np.zeros(0)
        v = self.vals[I[:, None] + self.LIM, J[None, :] + self.LIM]
        d = np.hypot((I[:, None] + 0.5) * c - x, (J[None, :] + 0.5) * c - y)
        m = (d <= radius) & np.isfinite(v)
        return v[m]


def local_bottom(cells, x, y, plane, radius=5.0, min_cells=3, pct=20.0):
    """(bottom z, fallback?): the pct-th percentile of the local ground cells' values, or the plane at (x, y)."""
    v = cells.local(x, y, radius)
    if len(v) >= min_cells:
        return float(np.percentile(v, pct)), False
    return float(plane_z(plane, x, y)), True


def footprint_distance(xy, box):
    """BEV distance of points to a box's rectangle (0 inside)."""
    p = np.asarray(xy, np.float64); b = np.asarray(box, np.float64)
    c, s = np.cos(b[6]), np.sin(b[6])
    dx, dy = p[:, 0] - b[0], p[:, 1] - b[1]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return np.hypot(np.maximum(np.abs(lx) - b[3] / 2, 0), np.maximum(np.abs(ly) - b[4] / 2, 0))


def ground_reference(scene_xyz, box, d_lo=0.5, d_hi=3.0, min_points=10, pct=5.0):
    """D1 v3's independent ground reference of a copy (20261018_02 §7.2): the pct-th percentile of z over the scene points
    whose BEV distance to the box's rectangle lies in (d_lo, d_hi]; (reference or nan, number of points)."""
    p = np.asarray(scene_xyz, np.float64)
    if not len(p):
        return np.nan, 0
    d = footprint_distance(p[:, :2], box)
    m = (d > d_lo) & (d <= d_hi)
    n = int(m.sum())
    return (float(np.percentile(p[m, 2], pct)) if n >= min_points else np.nan), n
