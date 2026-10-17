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
