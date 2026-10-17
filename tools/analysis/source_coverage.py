"""Does the nuScenes source contain cars like the Waymo cars 27471 misses, and can an operator add them?

experiments_md 20261018_01 holds the definitions (§2) and the readings pre-declared before this ran (§3). CPU only.
ANALYSIS: the Waymo-val split and the Waymo-train support read Waymo labels to diagnose; nothing here chooses a method or
a value (memory/repo/uda_legality_rule.md). The operators' label-free gate (G1-G3) uses only the target spec and SOURCE
labels.

Descriptors per GT box, on the MODEL INPUT (27471's grid, which 26398 and 27596 share): points with x, y in
[-75.2, 75.2) and z in [-2, 4); voxel = floor((p - [-75.2, -75.2, -2]) / [0.1, 0.1, 0.15]);
- r: BEV centre range; rings 0-10 ... 50-60, 60-75 m (>= 75 m: ring -1);
- n_vox: occupied voxels holding an in-box point; n_layers: distinct voxel z-indices among them;
- ppv: in-box points per occupied voxel, each voxel capped at MAX_POINTS_PER_VOXEL = 10; length = box dx.

Modes (run from ST3D/tools, inside the container):
    python analysis/source_coverage.py nus --cfg <27471 cfg> --stride 4 --out <npz> [--augment] [--ops]
           [--shard k/N] [--workers 8] [--max_frames m] [--dump <pkl>]
    python analysis/source_coverage.py waymo_train --cfg <26398 cfg> --stride 4 --out <npz> [--workers 8] ...
    python analysis/source_coverage.py waymo_val --pass_pkl <pass.pkl> --out <npz> [--workers 8] ...
    python analysis/source_coverage.py report --dir <dir> [--smoke]

`nus --ops` adds, per frame, the Part B operators on the same anchor: B1a (Boston 30 / Singapore 20) and B1b (60 / 60)
deeper consecutive accumulation with the existing loader; B2a / B2b = those clouds after the virtual Waymo TOP z-buffer
keep (nearest point per lattice pixel, kept in place); B3 = source Car donors (r <= 30 m, >= 200 in-box points) moved
radially to 40 / 50 / 60 / 70 m with the bottom on z = 0, rendered object-only through the Waymo TOP lattice (nearest
per pixel, snapped onto the pixel ray). G1 (lattice-row coverage RC), G2 (points per silhouette pixel) and G3
(phantoms) follow 20261009_01 §4 / §7.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import copy
import json
import pickle
import resource
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS.parent, TOOLS, TOOLS / 'analysis'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _init_path  # noqa: F401,E402  (the live pcdet, not the image's copy; allocator guard)

PCR = np.array([-75.2, -75.2, -2.0, 75.2, 75.2, 4.0])
VOX = np.array([0.1, 0.1, 0.15])
MAX_PPV = 10
VOXEL_CAP = 150000
RINGS = (0, 10, 20, 30, 40, 50, 60, 75)
RING_NAMES = tuple('%g-%g' % (RINGS[k], RINGS[k + 1]) for k in range(len(RINGS) - 1))
NVOX_EDGES = (0, 1, 5, 10, 20, 50, 100, 200)
NVOX_NAMES = ('0', '1-4', '5-9', '10-19', '20-49', '50-99', '100-199', '>=200')
NLAY_EDGES = (0, 1, 2, 3, 5, 7, 10, 13)
NLAY_NAMES = ('0', '1', '2', '3-4', '5-6', '7-9', '10-12', '>=13')
N_CELLS = (len(RINGS) - 1) * len(NVOX_EDGES) * len(NLAY_EDGES)
CAR_LABEL = 1
OTHER_VEHICLES = ('truck', 'bus', 'trailer', 'construction_vehicle')
LONG_M = 6.0
OPS_DEPTHS = (('B1a', {'NUSCENES_N008': 30, 'NUSCENES_N015': 20}), ('B1b', {'NUSCENES_N008': 60, 'NUSCENES_N015': 60}))
DONOR_R_MAX, DONOR_MIN_PTS = 30.0, 200
R_NEW = (40.0, 50.0, 60.0, 70.0)
SPARSE = (10, 50)             # G1 population: 10-49 base points, 20-75 m, moving ego
OCCL_M, PHANTOM_M = 0.5, 0.3
# per-box row: r, n_pts, n_vox, n_layers, ppv, length, sector, moving
BOX_COLS = ('r', 'n_pts', 'n_vox', 'n_layers', 'ppv', 'length', 'sector', 'moving')


# ------------------------------------------------------------------------------------------------ pure geometry
def grid_mask(xyz):
    x = np.asarray(xyz)
    return ((x[:, 0] >= PCR[0]) & (x[:, 0] < PCR[3]) & (x[:, 1] >= PCR[1]) & (x[:, 1] < PCR[4]) &
            (x[:, 2] >= PCR[2]) & (x[:, 2] < PCR[5]))


def crop(xyz):
    """The model input's points: inside the range in x, y and inside the voxel grid in z."""
    x = np.asarray(xyz)[:, :3]
    return np.ascontiguousarray(x[grid_mask(x)], np.float32)


def voxel_keys(xyz):
    ijk = np.floor((np.asarray(xyz, np.float64)[:, :3] - PCR[:3]) / VOX).astype(np.int64)
    return (ijk[:, 0] * 2000 + ijk[:, 1]) * 100 + ijk[:, 2]


def n_voxels(xyz):
    return int(len(np.unique(voxel_keys(xyz)))) if len(xyz) else 0


def descriptors(p):
    """(n_pts, n_vox, n_layers, ppv) of the in-box points p (already on the grid)."""
    if len(p) == 0:
        return 0, 0, 0, 0.0
    u, cnt = np.unique(voxel_keys(p), return_counts=True)
    return len(p), len(u), len(np.unique(u % 100)), float(np.minimum(cnt, MAX_PPV).sum()) / len(u)


def in_box_indices(xyz, boxes, tree=None):
    """Per box, the indices of the points inside it (pcdet's points_in_boxes_cpu, run on a BEV candidate subset)."""
    from scipy.spatial import cKDTree
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    boxes = np.asarray(boxes, np.float32).reshape(-1, 7)
    if not len(boxes):
        return []
    if len(xyz) == 0:
        return [np.zeros(0, np.int64) for _ in boxes]
    if tree is None:
        tree = cKDTree(np.asarray(xyz)[:, :2])
    out = []
    for b in boxes:
        cand = np.asarray(tree.query_ball_point(b[:2], float(np.hypot(b[3], b[4]) / 2 + 0.1)), np.int64)
        if not len(cand):
            out.append(cand); continue
        m = roiaware_pool3d_utils.points_in_boxes_cpu(np.ascontiguousarray(xyz[cand, :3], np.float32), b[None])[0] > 0
        out.append(np.sort(cand[m]))
    return out


def box_rows(xyz, boxes, sector=None, moving=0.0, tree=None, idx=None):
    """(M, 8) rows of BOX_COLS for the boxes on the cloud xyz (already cropped)."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 7)
    if idx is None:
        idx = in_box_indices(xyz, boxes, tree)
    rows = np.zeros((len(boxes), len(BOX_COLS)))
    for j, b in enumerate(boxes):
        rows[j, 0] = np.hypot(b[0], b[1])
        rows[j, 1:5] = descriptors(xyz[idx[j]])
        rows[j, 5] = b[3]
        rows[j, 6] = -1 if sector is None else sector[j]
        rows[j, 7] = moving
    return rows


def ring_of(r):
    r = np.asarray(r, np.float64)
    g = np.searchsorted(RINGS, r, side='right') - 1
    return np.where((r >= 0) & (r < RINGS[-1]), g, -1)


def bin_of(v, edges):
    return np.searchsorted(edges, np.asarray(v, np.float64), side='right') - 1


def cell_of(r, n_vox, n_layers):
    g = ring_of(r)
    c = (g * len(NVOX_EDGES) + bin_of(n_vox, NVOX_EDGES)) * len(NLAY_EDGES) + bin_of(n_layers, NLAY_EDGES)
    return np.where(g >= 0, c, -1)


def car_boxes(gt_boxes):
    """The (M, 7) boxes trained as Car: label == 1 exactly. Negative (pseudo-label ignore) and -99 (IGNORE_CLASS_LABEL)
    labels never match, and no label indexes anything."""
    if gt_boxes is None or len(gt_boxes) == 0:
        return np.zeros((0, 7))
    g = np.asarray(gt_boxes, np.float64)
    return g[g[:, 7] == CAR_LABEL, :7]


def other_vehicle_boxes(info, shift):
    """nuScenes truck / bus / trailer / construction_vehicle info boxes with >= 1 lidar point, SHIFT_COOR applied,
    centre inside the range in x, y: what 27471 trains as background (§2.2)."""
    names = np.asarray(info.get('gt_names', []))
    if not len(names):
        return np.zeros((0, 7))
    m = np.isin(names, OTHER_VEHICLES)
    if 'num_lidar_pts' in info:
        m &= np.asarray(info['num_lidar_pts']) >= 1
    b = np.asarray(info['gt_boxes'], np.float64)[m][:, :7].copy()
    b[:, :3] += np.asarray(shift, np.float64)[:3]
    ok = (b[:, 0] >= PCR[0]) & (b[:, 0] < PCR[3]) & (b[:, 1] >= PCR[1]) & (b[:, 1] < PCR[4])
    return b[ok]


def relocate(box, pts, r_new):
    """Move a box and its points radially to BEV centre range r_new: azimuth and heading kept (so the side facing the
    sensor is kept), bottom on z = 0 (the SHIFT_COOR frame's ground). Returns (box (7,), points (k, 3))."""
    b = np.asarray(box, np.float64)[:7].copy()
    r = float(np.hypot(b[0], b[1]))
    if not r > 0:
        raise ValueError('a donor at r = 0 has no azimuth')
    nb = b.copy()
    nb[0] *= r_new / r; nb[1] *= r_new / r; nb[2] = b[5] / 2.0
    t = nb[:3] - b[:3]
    p = np.asarray(pts, np.float64).reshape(-1, 3)[:, :3] + t
    return nb, p


def render(pts, thetas, n_cols, height):
    """Object-only render through a spinning-lidar lattice at (0, 0, height): row = nearest of the ASCENDING
    inclinations, column = azimuth bin; per pixel the nearest point is kept and SNAPPED onto the pixel ray (row
    inclination, column-centre azimuth) at its range. Returns (xyz (m, 3), rows ascending, cols, kept point index)."""
    from pcdet.datasets.rerender_utils import to_lattice
    p = np.asarray(pts, np.float64).reshape(-1, 3)
    if not len(p):
        return np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    row, col, rng, valid = to_lattice(p, height, thetas, n_cols)
    sel = np.nonzero(valid)[0]
    if not len(sel):
        return np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    cell = row[sel] * n_cols + col[sel]
    o = sel[np.lexsort((rng[sel], cell))]
    c = row[o] * n_cols + col[o]
    first = np.ones(len(o), bool); first[1:] = c[1:] != c[:-1]
    k = o[first]
    rr, th = rng[k], thetas[row[k]]
    az = (col[k] + 0.5) / n_cols * 2 * np.pi - np.pi
    xyz = np.stack([rr * np.cos(th) * np.cos(az), rr * np.cos(th) * np.sin(az), height + rr * np.sin(th)], 1)
    return xyz, row[k], col[k], k


def nearest_per_pixel(xyz, thetas, n_cols, height):
    """lattice_pixels of every point, and per occupied pixel (sorted ids) its nearest point's index and range."""
    from pcdet.datasets.processor.point_sampler import lattice_pixels
    pix, _, rng = lattice_pixels(np.asarray(xyz, np.float64), thetas, n_cols, height)
    ok = np.nonzero(pix >= 0)[0]
    if not len(ok):
        return pix, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    o = ok[np.lexsort((rng[ok], pix[ok]))]
    first = np.ones(len(o), bool); first[1:] = pix[o][1:] != pix[o][:-1]
    nidx = o[first]
    return pix, pix[nidx], nidx, rng[nidx]


def lattice_coverage(xyz, boxes, idx, thetas, n_cols, height, npp=None):
    """G1 / G2 per box: (RC, PPP, visible silhouette rows). RC = rows of the box's VISIBLE silhouette (virtual-lattice
    rays hitting the box; a pixel is occluded when its nearest return is outside the box and > OCCL_M in front of the
    box entry) holding >= 1 in-box point; PPP = in-box points per occupied visible silhouette pixel (20261009_01 §7)."""
    from lattice_sampler_precheck import silhouette
    inc_desc = np.sort(np.asarray(thetas, np.float64))[::-1]
    if npp is None:
        npp = nearest_per_pixel(xyz, thetas, n_cols, height)
    pix, up, nidx, nrng = npp
    out = np.full((len(boxes), 3), np.nan)
    inbox = np.zeros(len(xyz), bool)
    for j, b in enumerate(np.asarray(boxes, np.float64).reshape(-1, 7)):
        sp, srow, tin = silhouette(b, inc_desc, hs=height, n_cols=n_cols)
        if not len(sp):
            continue
        inbox[idx[j]] = True
        occl = np.zeros(len(sp), bool)
        if len(up):
            pos = np.clip(np.searchsorted(up, sp), 0, len(up) - 1); has = up[pos] == sp
            occl = has & (nrng[pos] < tin - OCCL_M) & ~inbox[nidx[pos]]
        vp, vrow = sp[~occl], srow[~occl]
        if len(vrow):
            pp = pix[idx[j]]; pp = pp[pp >= 0]
            isv = np.isin(pp, vp); occ = np.unique(pp[isv])
            out[j] = (len(np.unique(occ // n_cols)) / len(np.unique(vrow)),
                      isv.sum() / len(occ) if len(occ) else np.nan, len(np.unique(vrow)))
        inbox[idx[j]] = False
    return out


def zbuffer_keep_mask(xyz, thetas, n_cols, height):
    """B2's thinning: True for the nearest point of each virtual-lattice pixel (20261008_01's rule), kept in place."""
    from pcdet.datasets.processor.point_sampler import lattice_pixels, zbuffer_keep
    pix, _, rng = lattice_pixels(np.asarray(xyz, np.float64), thetas, n_cols, height)
    return zbuffer_keep(pix, rng)


def phantom_count(rendered, box, donor_pts):
    """G3: rendered points in no (relocated) box with no (relocated) donor point within PHANTOM_M."""
    from scipy.spatial import cKDTree
    if not len(rendered):
        return 0
    inb = np.zeros(len(rendered), bool)
    inb[in_box_indices(rendered, np.asarray(box)[None])[0]] = True
    if not len(donor_pts):
        return int((~inb).sum())
    d = cKDTree(np.asarray(donor_pts)).query(rendered, k=1, distance_upper_bound=PHANTOM_M)[0]
    return int((~inb & ~(d <= PHANTOM_M)).sum())


# --------------------------------------------------------------------------------------- coverage and readings
def support(rows, scale, length_max=None):
    """Instances per epoch per cell (n_vox >= 1 or not), from BOX_COLS rows."""
    rows = np.asarray(rows).reshape(-1, len(BOX_COLS))
    keep = np.ones(len(rows), bool) if length_max is None else rows[:, 5] < length_max
    c = cell_of(rows[keep, 0], rows[keep, 2], rows[keep, 3])
    return np.bincount(c[c >= 0], minlength=N_CELLS)[:N_CELLS] * float(scale)


def reachable_total(s):
    """S: instances with n_vox >= 1 at r < 75 m (the n_vox bin index is the middle cell axis)."""
    s = np.asarray(s).reshape(len(RINGS) - 1, len(NVOX_EDGES), len(NLAY_EDGES))
    return float(s[:, 1:, :].sum())


def covered(s_src, s_tgt, mode='share', floor=50.0, bar=0.2, scale_src=1.0, scale_tgt=1.0):
    """A1: covered iff s_src >= floor (per epoch) AND rho >= bar. mode 'share': rho = (s_src/S_src)/(s_tgt/S_tgt);
    'count': rho = s_src / s_tgt; 'run': rho = (scale_src s_src) / (scale_tgt s_tgt) (per-run totals)."""
    s_src, s_tgt = np.asarray(s_src, np.float64), np.asarray(s_tgt, np.float64)
    if mode == 'share':
        a, b = s_src / max(reachable_total(s_src), 1e-12), s_tgt / max(reachable_total(s_tgt), 1e-12)
    elif mode == 'count':
        a, b = s_src, s_tgt
    elif mode == 'run':
        a, b = s_src * scale_src, s_tgt * scale_tgt
    else:
        raise ValueError(mode)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.where(b > 0, a / np.where(b > 0, b, 1), np.inf)
    return (s_src >= floor) & (rho >= bar)


def uncovered_share(cells, cov):
    cells = np.asarray(cells, np.int64)
    cells = cells[cells >= 0]
    if not len(cells):
        return np.nan
    return float(np.mean(~cov[cells]))


def a2_reading(per_ring):
    """per_ring: [(U_miss, U_found)] for 40-50, 50-60, 60-75 m. Returns (reading, a-flags, ab-flags) (§3.2)."""
    a = [bool(np.isfinite(m) and m >= 0.5) for m, f in per_ring]
    ab = [bool(a[k] and np.isfinite(f) and f <= 0.5 * m) for k, (m, f) in enumerate(per_ring)]
    if sum(ab) >= 2:
        return 'the source lacks cars like the misses', a, ab
    if sum(a) >= 2:
        return ('the source lacks such cars at range, but the found GT lack them about as much; coverage does not '
                'single out the misses'), a, ab
    return 'the source has them; coverage does not explain the misses', a, ab


def reach(miss_rows, op_rows, ring_lo, ring_hi, frac=0.01):
    """Share of the misses in [ring_lo, ring_hi) lying in (n_vox bin, n_layers bin) cells that hold >= frac of the
    operator's instances in that ring (§3.6)."""
    def cells(rows):
        rows = np.asarray(rows).reshape(-1, len(BOX_COLS))
        m = (rows[:, 0] >= ring_lo) & (rows[:, 0] < ring_hi)
        return bin_of(rows[m, 2], NVOX_EDGES) * len(NLAY_EDGES) + bin_of(rows[m, 3], NLAY_EDGES)
    mc, oc = cells(miss_rows), cells(op_rows)
    if not len(mc) or not len(oc):
        return np.nan
    h = np.bincount(oc, minlength=len(NVOX_EDGES) * len(NLAY_EDGES)) / len(oc)
    return float(np.mean(h[mc] >= frac))


# --------------------------------------------------------------------------------------------- data passes
G = {}   # filled in the parent before the pool forks


def lattice():
    from pcdet.datasets.rerender_utils import lattice_spec
    s = lattice_spec('waymo_top')
    return s['thetas'], s['n_cols'], s['height']


def _sample(ds, i, max_sweeps=None):
    if max_sweeps is None:
        return ds[i]
    old = ds.dataset_cfg.MAX_SWEEPS
    ds.dataset_cfg.MAX_SWEEPS = max_sweeps
    try:
        return ds[i]
    finally:
        ds.dataset_cfg.MAX_SWEEPS = old


def _frame_row(plat, i, P, moving, sec):
    return [plat, i, len(P), n_voxels(P), moving, sec]


def nus_frame(task):
    from scipy.spatial import cKDTree
    from lattice_depth_ceiling import travel_direction, sector_of
    import torch
    torch.set_num_threads(1)
    plat, key, i = task
    ds = G['ds'][key]; info = ds.infos[i]; shift = np.asarray(ds.dataset_cfg.SHIFT_COOR, np.float64)
    if G['augment']:
        np.random.seed((plat * 1000003 + i) % (2 ** 31))
    t0 = time.time()
    d = ds[i]; t_load = time.time() - t0
    P = crop(d['points']); cars = car_boxes(d.get('gt_boxes'))
    fwd = travel_direction(info); moving = float(fwd is not None)
    fv = fwd if fwd is not None else np.array([1.0, 0.0])
    sec = sector_of(cars[:, :2], fv) if len(cars) else np.zeros(0, np.int64)
    tree = cKDTree(P[:, :2]) if len(P) else None
    idx = in_box_indices(P, cars, tree)
    out = dict(car=box_rows(P, cars, sec, moving, idx=idx), frame=[_frame_row(plat, i, P, moving, t_load)])
    out['car_meta'] = np.array([[plat, i]] * len(cars), np.float64).reshape(-1, 2)
    if not G['augment']:
        ob = other_vehicle_boxes(info, shift)
        out['other'] = box_rows(P, ob, sector_of(ob[:, :2], fv) if len(ob) else None, moving, tree)
    if G['dump'] is not None and len(G['dumped']) < 3:
        for j in np.nonzero((out['car'][:, 0] >= 20) & (out['car'][:, 1] >= 1))[0][:3 - len(G['dumped'])]:
            G['dumped'].append(dict(mode='nus', plat=plat, frame=i, box=cars[j], points=P[idx[j]], row=out['car'][j]))
    if not G['ops']:
        return out
    th, nc, h = G['lattice']
    rr = out['car'][:, 0]; npts = out['car'][:, 1]
    sp = np.nonzero((npts >= SPARSE[0]) & (npts < SPARSE[1]) & (rr >= 20) & (rr < 75))[0]
    g1 = {'base': lattice_coverage(P, cars[sp], [idx[j] for j in sp], th, nc, h)} if len(sp) else {}
    for name, depths in OPS_DEPTHS:
        t1 = time.time()
        d2 = _sample(ds, i, depths[key]); t2 = time.time() - t1
        P2 = crop(d2['points']); c2 = car_boxes(d2.get('gt_boxes'))
        s2 = sector_of(c2[:, :2], fv) if len(c2) else np.zeros(0, np.int64)
        tr2 = cKDTree(P2[:, :2]) if len(P2) else None
        out['car_' + name] = box_rows(P2, c2, s2, moving, tr2)
        out['frame_' + name] = [_frame_row(plat, i, P2, moving, t2)]
        keep = zbuffer_keep_mask(P2, th, nc, h)
        P3 = np.ascontiguousarray(P2[keep])
        bname = 'B2' + name[2:]
        tr3 = cKDTree(P3[:, :2]) if len(P3) else None
        out['car_' + bname] = box_rows(P3, c2, s2, moving, tr3)
        out['frame_' + bname] = [_frame_row(plat, i, P3, moving, 0.0)]
        if len(sp):
            g1[name] = lattice_coverage(P2, cars[sp], in_box_indices(P2, cars[sp], tr2), th, nc, h)
            g1[bname] = lattice_coverage(P3, cars[sp], in_box_indices(P3, cars[sp], tr3), th, nc, h)
        del d2, P2, P3
    if len(sp):
        out['g1'] = np.concatenate([np.stack([np.full(len(sp), plat), np.full(len(sp), i), rr[sp], sec[sp],
                                              np.full(len(sp), moving), npts[sp]], 1)] +
                                   [g1[k] for k in ('base', 'B1a', 'B2a', 'B1b', 'B2b')], 1)
    # B3: donors -> relocated -> rendered
    don = np.nonzero((rr > 0) & (rr <= DONOR_R_MAX) & (npts >= DONOR_MIN_PTS))[0]
    b3, b3rows, b3in = [], [], []
    for j in don:
        dp = P[idx[j]].astype(np.float64)
        for rn in R_NEW:
            nb, q = relocate(cars[j], dp, rn)
            R, _, _, _ = render(q, th, nc, h)
            Rc = crop(R)
            # Amendment 20261018_01 §3.8: every rendered point is the object's (object-only render of in-box donor
            # points), but snapping moves a surface point by up to half a pixel, so some land just outside the box.
            # The descriptors and RC therefore use ALL rendered points; the declared in-box version is kept beside.
            aidx = [np.arange(len(Rc))]
            ridx = in_box_indices(Rc, nb[None])
            row = box_rows(Rc, nb[None], [sec[j]], moving, idx=aidx)[0]
            row_in = box_rows(Rc, nb[None], [sec[j]], moving, idx=ridx)[0]
            npp = nearest_per_pixel(Rc, th, nc, h)
            cov = lattice_coverage(Rc, nb[None], aidx, th, nc, h, npp)[0]
            cov_in = lattice_coverage(Rc, nb[None], ridx, th, nc, h, npp)[0]
            b3rows.append(row); b3in.append(row_in)
            b3.append([plat, i, rr[j], npts[j], sec[j], moving, rn, len(R), *cov, phantom_count(Rc, nb, q), len(Rc),
                       *cov_in])
            if G['dump'] is not None and len(G['dumped_b3']) < 3 and row[1] >= 1:
                G['dumped_b3'].append(dict(mode='b3', plat=plat, frame=i, donor_box=cars[j], donor_points=dp, r_new=rn,
                                           box=nb, moved=q, rendered=Rc, row=row, row_in=row_in, cov=cov, cov_in=cov_in))
    if b3:
        out['b3'] = np.array(b3, np.float64); out['car_B3'] = np.array(b3rows, np.float64)
        out['car_B3in'] = np.array(b3in, np.float64)
    return out


def waymo_train_frame(i):
    import torch
    torch.set_num_threads(1)
    ds = G['ds']['WAYMO']
    t0 = time.time(); d = ds[i]; t_load = time.time() - t0
    P = crop(d['points']); cars = car_boxes(d.get('gt_boxes'))
    idx = in_box_indices(P, cars)
    out = dict(car=box_rows(P, cars, None, 0.0, idx=idx), frame=[_frame_row(2, i, P, 0.0, t_load)])
    if G['dump'] is not None and len(G['dumped']) < 3:
        for j in np.nonzero((out['car'][:, 0] >= 20) & (out['car'][:, 1] >= 1))[0][:3 - len(G['dumped'])]:
            G['dumped'].append(dict(mode='waymo_train', frame=i, box=cars[j], points=P[idx[j]], row=out['car'][j]))
    return out


def waymo_val_frame(fi):
    import torch
    from pcdet.datasets.waymo.waymo_dataset import WaymoDataset
    torch.set_num_threads(1)
    f = G['frames'][fi]
    pts = WaymoDataset.get_lidar(SimpleNamespace(data_path=G['data_path'], dataset_cfg={}), f['seq'], f['idx'])
    P = crop(pts[:, :3])
    idx = in_box_indices(P, f['gv'])
    out = dict(car=box_rows(P, f['gv'], None, 0.0, idx=idx), frame=[_frame_row(3, fi, P, 0.0, 0.0)])
    if G['dump'] is not None and len(G['dumped']) < 3:
        for j in np.nonzero((out['car'][:, 0] >= 20) & (out['car'][:, 1] >= 1))[0][:3 - len(G['dumped'])]:
            G['dumped'].append(dict(mode='waymo_val', frame=fi, gt_index=j, box=f['gv'][j], points=P[idx[j]],
                                    row=out['car'][j]))
    return fi, out


def _collect(results, keys_order=None):
    acc = {}
    for r in results:
        for k, v in r.items():
            if isinstance(v, list) and not len(v):
                continue
            acc.setdefault(k, []).append(np.asarray(v, np.float64).reshape(len(v), -1) if len(v) else None)
    return {k: np.concatenate([a for a in v if a is not None]) if any(a is not None for a in v) else np.zeros((0, 1))
            for k, v in acc.items()}


def _run_pool(fn, tasks, workers, maxtasks=None):
    """maxtasks: re-fork each worker after that many frames, so copy-on-write growth of the parent's dataset (16.7 GB
    with both nuScenes compensators, smoke 29397) stays bounded."""
    t0 = time.time(); res = []
    if workers <= 1:
        for n, t in enumerate(tasks):
            res.append(fn(t))
            if n % 200 == 0:
                print(f'  {n}/{len(tasks)} tasks, {time.time() - t0:.0f} s', flush=True)
    else:
        with Pool(workers, maxtasksperchild=maxtasks) as pool:
            for n, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=1)):
                res.append(r)
                if n % 200 == 0:
                    print(f'  {n}/{len(tasks)} tasks, {time.time() - t0:.0f} s', flush=True)
    print(f'pool done: {len(tasks)} tasks in {time.time() - t0:.0f} s', flush=True)
    return res


def _print_block(key, dc):
    proc = [p['NAME'] for p in dc.DATA_PROCESSOR]
    vox = [p for p in dc.DATA_PROCESSOR if p['NAME'] == 'transform_points_to_voxels']
    print(f'[{key}] MAX_SWEEPS {dc.get("MAX_SWEEPS", 1)} | SHIFT_COOR {dc.get("SHIFT_COOR", None)} | '
          f'GT_BOXES_MOTION_COMPENSATION {dc.get("GT_BOXES_MOTION_COMPENSATION", None)} | '
          f'SWEEP_SELECTION {dc.get("SWEEP_SELECTION", None)} | DEDUP_SWEEPS {dc.get("DEDUP_SWEEPS", None)} | '
          f'IGNORE_OTHER_CLASSES {dc.get("IGNORE_OTHER_CLASSES", None)} | SAMPLED_INTERVAL {dc.get("SAMPLED_INTERVAL", None)} | '
          f'PCR {list(dc.POINT_CLOUD_RANGE)} | FILTER_MIN_POINTS_IN_GT {dc.get("FILTER_MIN_POINTS_IN_GT", None)} | '
          f'MIN_POINTS_OF_GT {dc.get("MIN_POINTS_OF_GT", None)}', flush=True)
    print(f'[{key}] AUG {[(a["NAME"]) for a in dc.DATA_AUGMENTOR.AUG_CONFIG_LIST]} disabled '
          f'{list(dc.DATA_AUGMENTOR.get("DISABLE_AUG_LIST", []))} | PROCESSOR {proc} | VOXEL '
          f'{[(list(v["VOXEL_SIZE"]), v["MAX_POINTS_PER_VOXEL"], dict(v["MAX_NUMBER_OF_VOXELS"])) for v in vox]}', flush=True)


def _build(dc, cfg, training=True):
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    dc = copy.deepcopy(dc)
    dc.DATA_PROCESSOR = [p for p in dc.DATA_PROCESSOR if p['NAME'] != 'transform_points_to_voxels']
    ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=common_utils.create_logger(), training=training,
                                model_ontology=cfg.get('ONTOLOGY'))
    return ds


def main_nus(a):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.point_calibration import _augmentation_off
    cfg = EasyDict(); cfg_from_yaml_file(a.cfg, cfg)
    keys = ('NUSCENES_N008', 'NUSCENES_N015')
    G.update(ds={}, augment=a.augment, ops=a.ops, dump=a.dump, dumped=[], dumped_b3=[], lattice=lattice())
    tasks = []
    for plat, key in enumerate(keys):
        _print_block(key, cfg.DATA_CONFIGS[key])
        ds = _build(cfg.DATA_CONFIGS[key], cfg)
        print(f'[{key}] {len(ds)} infos; compensator {"ON" if ds._sweep_compensator is not None else "OFF"}; '
              f'augmentor queue {[type(f).__name__ if not hasattr(f, "func") else f.func.__name__ for f in ds.data_augmentor.data_augmentor_queue]}',
              flush=True)
        if not a.augment:
            _augmentation_off(ds).__enter__()      # kept off for the whole pass (forked workers inherit it)
        G['ds'][key] = ds
        t = [(plat, key, i) for i in range(0, len(ds), a.stride)]
        tasks += t[:a.max_frames] if a.max_frames else t
    k, n = (int(v) for v in a.shard.split('/'))
    tasks = tasks[k::n]
    print(f'{len(tasks)} frames (stride {a.stride}, shard {a.shard}, augment {a.augment}, ops {a.ops})', flush=True)
    res = _run_pool(nus_frame, tasks, a.workers if not a.dump else 1, maxtasks=25)
    out = _collect(res)
    np.savez(a.out, stride=a.stride, augment=a.augment, ops=a.ops, shard=a.shard, n_tasks=len(tasks),
             box_cols=np.array(BOX_COLS), **out)
    print(f'saved {a.out}: ' + ', '.join(f'{k} {v.shape}' for k, v in out.items()), flush=True)
    if a.dump:
        pickle.dump(dict(boxes=G['dumped'], b3=G['dumped_b3']), open(a.dump, 'wb'))
        print(f'dumped {len(G["dumped"])} boxes and {len(G["dumped_b3"])} B3 renders to {a.dump}')
    print(f'max RSS (parent) {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB; children '
          f'{resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6:.2f} GB', flush=True)


def main_waymo_train(a):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.point_calibration import _augmentation_off
    cfg = EasyDict(); cfg_from_yaml_file(a.cfg, cfg)
    _print_block('WAYMO', cfg.DATA_CONFIG)
    ds = _build(cfg.DATA_CONFIG, cfg)
    _augmentation_off(ds).__enter__()
    G.update(ds={'WAYMO': ds}, dump=a.dump, dumped=[])
    tasks = list(range(0, len(ds), a.stride))
    tasks = tasks[:a.max_frames] if a.max_frames else tasks
    print(f'[WAYMO] {len(ds)} sampled train frames; {len(tasks)} at stride {a.stride}', flush=True)
    out = _collect(_run_pool(waymo_train_frame, tasks, a.workers if not a.dump else 1))
    np.savez(a.out, stride=a.stride, n_sampled=len(ds), n_tasks=len(tasks), box_cols=np.array(BOX_COLS), **out)
    print(f'saved {a.out}: ' + ', '.join(f'{k} {v.shape}' for k, v in out.items()), flush=True)
    if a.dump:
        pickle.dump(dict(boxes=G['dumped']), open(a.dump, 'wb'))
    print(f'max RSS (parent) {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB', flush=True)


def main_waymo_val(a):
    """The 249,066 Vehicle GT of pass.pkl, in its order (frames in the first run's result.pkl order, boxes in info
    order, names == 'Vehicle'), on the evaluated cloud."""
    p = pickle.load(open(a.pass_pkl, 'rb'))
    infos = pickle.load(open(TOOLS.parent / 'data/waymo/waymo_infos_val.pkl', 'rb'))
    by_fid = {}
    for inf in infos:
        fid = inf['frame_id'] if 'frame_id' in inf else \
            inf['point_cloud']['lidar_sequence'] + '_%03d' % inf['point_cloud']['sample_idx']
        by_fid[fid] = inf
    res = pickle.load(open(p['runs'][0][1], 'rb'))
    fids = [r['frame_id'] for r in res][:p['n_frames']]
    del res
    frames = []
    for fid in fids:
        an = by_fid[fid]['annos']; names = np.asarray(an['name'])
        b = np.asarray(an['gt_boxes_lidar'], np.float64).reshape(len(names), -1)[:, :7] if len(names) else np.zeros((0, 7))
        frames.append(dict(seq=by_fid[fid]['point_cloud']['lidar_sequence'],
                           idx=int(by_fid[fid]['point_cloud']['sample_idx']), gv=b[names == 'Vehicle']))
    del infos, by_fid
    nv = np.array([len(f['gv']) for f in frames]); gfr = np.repeat(np.arange(len(frames)), nv)
    rng = np.concatenate([np.hypot(f['gv'][:, 0], f['gv'][:, 1]) for f in frames])
    assert len(gfr) == len(p['gt']['frame']) and np.array_equal(gfr, p['gt']['frame']), 'GT order differs from pass.pkl'
    assert np.allclose(rng, p['gt']['rng'], atol=1e-3), 'GT ranges differ from pass.pkl'
    print(f'{len(frames)} val frames, {len(gfr)} Vehicle GT; order and ranges equal pass.pkl', flush=True)
    G.update(frames=frames, data_path=TOOLS.parent / 'data/waymo/waymo_processed_data', dump=a.dump, dumped=[])
    tasks = list(range(len(frames)))[:a.max_frames] if a.max_frames else list(range(len(frames)))
    res = _run_pool(waymo_val_frame, tasks, a.workers if not a.dump else 1)
    res.sort(key=lambda t: t[0])
    out = _collect([r for _, r in res])
    flags = {}
    for lab in p['labels']:
        g = p['models'][lab]['gt']
        flags[f'{lab}_found5'] = g['det5_score'] >= 0
        flags[f'{lab}_found7'] = g['det7_score'] >= 0
        flags[f'{lab}_best_iou'] = g['best_iou']
    n = int(nv[:len(tasks)].sum())
    flags = {k: v[:n] for k, v in flags.items()}
    np.savez(a.out, labels=np.array(p['labels']), n_tasks=len(tasks), box_cols=np.array(BOX_COLS),
             gt_npts_stored=p['gt']['npts'][:n], gt_length=p['gt']['length'][:n], **out, **flags)
    print(f'saved {a.out}: ' + ', '.join(f'{k} {v.shape}' for k, v in out.items()), flush=True)
    if a.dump:
        pickle.dump(dict(boxes=G['dumped']), open(a.dump, 'wb'))
    print(f'max RSS (parent) {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['nus', 'waymo_train', 'waymo_val', 'report'])
    ap.add_argument('--cfg'); ap.add_argument('--out'); ap.add_argument('--dir'); ap.add_argument('--pass_pkl')
    ap.add_argument('--stride', type=int, default=4); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_frames', type=int, default=0); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--augment', action='store_true'); ap.add_argument('--ops', action='store_true')
    ap.add_argument('--dump', default=None); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.mode == 'nus':
        main_nus(a)
    elif a.mode == 'waymo_train':
        main_waymo_train(a)
    elif a.mode == 'waymo_val':
        main_waymo_val(a)
    else:
        from source_coverage_report import main_report
        main_report(a)
