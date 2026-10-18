"""Box-level scan-pattern sheets: source, source + RR v2, target, target misses (experiments_md 20261018_03). CPU, ANALYSIS.

The user's request (2026-10-18 ~11:30 JST): show individual boxes of the nuScenes source, of the source after
relocate-and-render (RR v2, experiments_md 20261018_02 §6), and of the Waymo target, at the same range, from a viewpoint
in which the scan pattern (the beam lines) is visible. ANALYSIS ONLY: columns C / D read Waymo val labels to diagnose
(D is the misses of 20261018_01); nothing here chooses a method, a variant or a value (memory/repo/uda_legality_rule.md).

Columns, per range bin (one sheet per bin, rows = the boxes nearest the p25 / p50 / p75 of in-box point count):
- A: nuScenes train `car` boxes of 27471's loader (Boston 15 / Singapore 10 sweeps, per-object compensation,
  augmentation off; population P1 of 20261018_01, `nus_base.npz`), points coloured by sweep time lag;
- B: RR v2 copies of the gate v2 run (the 200 + 200 strided train frames and per-frame seeds of `rr_gate.py run_v2`,
  `gate_v2_*.pkl`), rerun here with the same config, seeds and capture, and checked against the gate's rows; kept points
  after the sparsifier in colour, the unsparsified render (after the 0.5 m occlusion drop) in grey;
- C: Waymo val Vehicle GT on the evaluated cloud (27596's input; P3 of 20261018_01, `waymo_val.npz`);
- D: the oracle-only misses among them (found by 27179 at IoU 0.5, not by 27596; 20261018_01 §2.2).
Every population: length < 6 m, >= 1 in-box point on 27471's grid, BEV centre range in the bin. The pool per
(column, bin) is a fixed-seed random draw of at most POOL_N eligible boxes; per quantile the pool member nearest to it.

Viewpoint (the main panel): azimuth x elevation about THAT cloud's own sensor, the same degrees per inch in every panel
of a sheet (equal aspect, one window per sheet).
- nuScenes (A): LIDAR_TOP, which the loader's SHIFT_COOR moves to (0, 0, 1.75) (the angles are taken about it, never
  about the shifted ground origin; memory/repo/index.md, "Angle statistics inside a loader").
- RR copy (B): the virtual Waymo TOP at (0, 0, 2.184) of the render, in the same nuScenes loader frame.
- Waymo (C, D): the TOP sensor of the segment, from its published extrinsic (vehicle frame, SHIFT_COOR 0): elevation in
  the sensor's axes (as `waymo_rings.top_columns_and_elevations`), azimuth in the vehicle's axes.
Faint lines: the Waymo TOP rows, the TRAIN-median lattice of the render in A and B, the segment's own published
inclinations in C and D; dashed in A: the HDL-32E's nominal 32 rings.

    python analysis/box_pattern_visual.py select --out <dir>
    python analysis/box_pattern_visual.py extract_a --out <dir>
    python analysis/box_pattern_visual.py extract_b --out <dir>
    python analysis/box_pattern_visual.py extract_cd --out <dir>
    python analysis/box_pattern_visual.py plot --out <dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS.parent, TOOLS, TOOLS / 'analysis'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _init_path  # noqa: F401,E402  (the checkout's pcdet, not the image's copy)

BINS = ((15.0, 25.0), (30.0, 40.0), (40.0, 50.0), (50.0, 60.0), (60.0, 75.0))
COLUMNS = ('A', 'B', 'C', 'D')
QS = (25, 50, 75)
POOL_N = 300
SEED = 20261018
LONG_M = 6.0
CTX_M = 3.0                      # context: scene points within this BEV distance of the box's rectangle
LAG_EDGES = (0.2, 0.45)          # s: older-sweep colour classes (0, 0.2], (0.2, 0.45], > 0.45
WINDOW_HALF_M, WINDOW_ASPECT = 3.2, 0.42
NUS_CFG = 'cfgs/da-ieee-access/centerpoint-gblobs-accum-legaldepth-nuscenes2kitti.yaml'      # 27471
RR2_CFG = 'cfgs/da-ieee-access/centerpoint-gblobs-accum-legaldepth-rr2-nuscenes2waymo.yaml'  # gate v2 (29458 / 29459)
SC_DIR = Path('/home/koyama/data/source_coverage_20261018/full')
GATE_DIR = Path('/home/koyama/data/relocate_render_20261018/v2/gate')
DB_DIR = Path('/home/koyama/data/relocate_render_20261018/v2')
PASS_PKL = Path('/home/koyama/data/waymo_ranking_20261017/full/pass.pkl')
TOP_CALIB = Path('/home/koyama/data/waymo_top_calib/top_calib.pkl')
NUS_KEYS = ('NUSCENES_N008', 'NUSCENES_N015')
PLAT_NAME = {0: 'Boston', 1: 'Singapore'}
HDL32E_DEG = np.linspace(-30.67, 10.67, 32)      # nominal (rerender_utils.lattice_spec('hdl32e'))
SC_COLS = ('r', 'n_pts', 'n_vox', 'n_layers', 'ppv', 'length', 'sector', 'moving')


# ------------------------------------------------------------------------------------------- geometry (unit-tested)
def wrap_deg(a):
    return (np.asarray(a, np.float64) + 180.0) % 360.0 - 180.0


def sensor_pose(kind, shift=(0.0, 0.0, 0.0), extrinsic=None, height=2.184):
    """The sensor whose eye the panel takes, in the cloud's own (loader) frame.
    'nus': nuScenes LIDAR_TOP = the frame origin, moved by the loader's SHIFT_COOR, so it sits AT SHIFT_COOR.
    'rr': the render's virtual Waymo TOP at (0, 0, height) of the nuScenes loader frame.
    'waymo': the segment's TOP from its 4x4 sensor -> vehicle extrinsic (+ SHIFT_COOR, 0 for 27596); rot = its rotation,
    yaw_deg its heading in the vehicle's axes (azimuths are reported in the vehicle's axes)."""
    if kind == 'nus':
        return dict(kind=kind, origin=np.asarray(shift, np.float64)[:3].copy(), rot=None, yaw_deg=0.0)
    if kind == 'rr':
        return dict(kind=kind, origin=np.array([0.0, 0.0, float(height)]), rot=None, yaw_deg=0.0)
    if kind == 'waymo':
        E = np.asarray(extrinsic, np.float64)
        return dict(kind=kind, origin=E[:3, 3] + np.asarray(shift, np.float64)[:3], rot=E[:3, :3].copy(),
                    yaw_deg=float(np.degrees(np.arctan2(E[1, 0], E[0, 0]))))
    raise ValueError(kind)


def view_angles(xyz, pose):
    """(azimuth, elevation) in degrees of points about pose['origin']; with pose['rot'] (sensor -> frame rotation) the
    direction is first expressed in the sensor's axes (elevation = the beam inclination's frame), and the azimuth is
    turned back by yaw_deg into the frame's axes."""
    d = np.atleast_2d(np.asarray(xyz, np.float64))[:, :3] - pose['origin']
    if pose['rot'] is not None:
        d = d @ pose['rot']                          # R^T d, row-wise
    az = np.degrees(np.arctan2(d[:, 1], d[:, 0])) + pose['yaw_deg']
    el = np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
    return wrap_deg(az), el


def box_corners(box):
    """(8, 3) corners of an (x, y, z, dx, dy, dz, heading) box: bottom face 0-3, top face 4-7."""
    x, y, z, dx, dy, dz, h = np.asarray(box, np.float64)[:7]
    lx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * dx / 2
    ly = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * dy / 2
    lz = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * dz / 2
    c, s = np.cos(h), np.sin(h)
    return np.stack([x + c * lx - s * ly, y + s * lx + c * ly, z + lz], 1)


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def box_edge_polylines(box, n=32):
    """The 12 edges, each sampled at n points (a straight edge projects to a curve in azimuth x elevation)."""
    c = box_corners(box); t = np.linspace(0.0, 1.0, n)[:, None]
    return [c[i] + t * (c[j] - c[i]) for i, j in EDGES]


def relative_view_angle_deg(box, origin):
    """Heading minus the azimuth of the box centre about the sensor's vertical axis, wrapped to [-180, 180)."""
    b = np.asarray(box, np.float64)
    return float(wrap_deg(np.degrees(b[6] - np.arctan2(b[1] - origin[1], b[0] - origin[0]))))


def footprint_distance(xy, box):
    """BEV distance of points to a box's rectangle (0 inside)."""
    p = np.asarray(xy, np.float64); b = np.asarray(box, np.float64)
    c, s = np.cos(b[6]), np.sin(b[6])
    dx, dy = p[:, 0] - b[0], p[:, 1] - b[1]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return np.hypot(np.maximum(np.abs(lx) - b[3] / 2, 0), np.maximum(np.abs(ly) - b[4] / 2, 0))


def nearest_row(el_deg, rows_deg):
    """Index of the nearest row (rows ascending) and |el - row| in degrees."""
    r = np.asarray(rows_deg, np.float64); e = np.asarray(el_deg, np.float64)
    if not len(e):
        return np.zeros(0, np.int64), np.zeros(0)
    k = np.clip(np.searchsorted(r, e), 1, len(r) - 1)
    k = np.where(np.abs(e - r[k - 1]) <= np.abs(e - r[k]), k - 1, k)
    return k, np.abs(e - r[k])


def pick_quantile_boxes(values, seed, pool_n=POOL_N, qs=QS):
    """Fixed-seed pool (a random permutation's first pool_n), the pool's quantiles (numpy linear) of `values`, and per
    quantile the pool member nearest to it (ties: earliest in pool order), all distinct. Returns (indices into values,
    quantile values, pool size)."""
    v_all = np.asarray(values, np.float64)
    if not len(v_all):
        return np.zeros(0, np.int64), np.full(len(qs), np.nan), 0
    perm = np.random.default_rng(seed).permutation(len(v_all))[:pool_n]
    v = v_all[perm]
    qv = np.percentile(v, qs)
    taken = np.zeros(len(perm), bool); out = []
    for q in qv:
        d = np.abs(v - q); d[taken] = np.inf
        if not np.isfinite(d).any():
            break
        k = int(np.argmin(d)); taken[k] = True; out.append(perm[k])
    return np.asarray(out, np.int64), qv, len(perm)


def sheet_window(r_lo, half_m=WINDOW_HALF_M, aspect=WINDOW_ASPECT):
    """(width, height) in degrees of every panel of a sheet: the angle a +-half_m object subtends at the bin's near edge."""
    w = 2.0 * np.degrees(np.arctan(half_m / r_lo))
    return float(w), float(aspect * w)


def block_index(keys):
    """Position of each row inside its contiguous block of equal keys (rows of one frame are contiguous)."""
    k = np.asarray(keys)
    if not len(k):
        return np.zeros(0, np.int64)
    start = np.r_[True, np.any(k[1:] != k[:-1], axis=1) if k.ndim > 1 else k[1:] != k[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(len(k)), 0))
    return np.arange(len(k)) - first


# -------------------------------------------------------------------------------------------------------- select
def _bin_of(r):
    r = np.asarray(r, np.float64); out = np.full(len(r), -1)
    for b, (lo, hi) in enumerate(BINS):
        out[(r >= lo) & (r < hi)] = b
    return out


def select(a):
    from rr_gate import V2_COPY_COLS
    C = {k: j for j, k in enumerate(V2_COPY_COLS)}
    S = {k: j for j, k in enumerate(SC_COLS)}
    sel = []
    # A: P1 nuScenes car rows (27471 as trained, augmentation off, every 4th info)
    z = np.load(SC_DIR / 'nus_base.npz')
    car, meta = z['car'], z['car_meta']
    jA = block_index(meta)
    okA = (car[:, S['n_pts']] >= 1) & (car[:, S['length']] < LONG_M)
    # B: gate v2 copies (both platforms, frame order), lengths from the donor database index
    cps, jB = [], []
    for key in NUS_KEYS:
        g = pickle.load(open(GATE_DIR / f'gate_v2_{key}.pkl', 'rb'))
        cp = g['copies']
        idx = np.load(DB_DIR / f'db_{key}_index.npz')
        size = idx['size'] if 'size' in idx.files else idx['sizes']
        L = size[cp[:, C['instance']].astype(np.int64), 0]
        cps.append(np.c_[cp, L]); jB.append(block_index(cp[:, [C['plat'], C['frame']]]))
    cpB = np.concatenate(cps); jB = np.concatenate(jB)
    okB = (cpB[:, C['n_pts_grid']] >= 1) & (cpB[:, -1] < LONG_M)
    # C / D: Waymo val Vehicle GT (pass.pkl order) and the oracle-only misses at IoU 0.5
    w = np.load(SC_DIR / 'waymo_val.npz')
    wc = w['car']
    p = pickle.load(open(PASS_PKL, 'rb'))
    gfr = np.asarray(p['gt']['frame'], np.int64)
    assert len(gfr) == len(wc) and np.allclose(p['gt']['rng'], wc[:, S['r']], atol=1e-3), 'waymo_val.npz != pass.pkl'
    jW = np.arange(len(gfr)) - np.searchsorted(gfr, gfr)
    okC = (wc[:, S['n_pts']] >= 1) & (wc[:, S['length']] < LONG_M)
    miss = w['oracle_found5'].astype(bool) & ~w['ours27596_found5'].astype(bool)
    pops = {
        'A': (car[:, S['r']], car[:, S['n_pts']], okA),
        'B': (cpB[:, C['r_new']], cpB[:, C['n_pts_grid']], okB),
        'C': (wc[:, S['r']], wc[:, S['n_pts']], okC),
        'D': (wc[:, S['r']], wc[:, S['n_pts']], okC & miss),
    }
    for ci, col in enumerate(COLUMNS):
        r, npts, ok = pops[col]
        rb = _bin_of(r)
        for b in range(len(BINS)):
            elig = np.nonzero(ok & (rb == b))[0]
            pick, qv, pool_n = pick_quantile_boxes(npts[elig], SEED + 1000 * ci + b)
            for qi, gi in enumerate(elig[pick]):
                e = dict(col=col, bin=b, q=QS[qi], q_value=float(qv[qi]), pool_n=int(pool_n), elig_n=int(len(elig)),
                         g=int(gi))
                if col == 'A':
                    e.update(plat=int(meta[gi, 0]), frame=int(meta[gi, 1]), j=int(jA[gi]), row=car[gi].tolist())
                elif col == 'B':
                    e.update(plat=int(cpB[gi, C['plat']]), frame=int(cpB[gi, C['frame']]), j=int(jB[gi]),
                             row=cpB[gi, :len(V2_COPY_COLS)].tolist(), length=float(cpB[gi, -1]))
                else:
                    e.update(fi=int(gfr[gi]), j=int(jW[gi]), row=wc[gi].tolist())
                sel.append(e)
            print(f'[select] {col} {BINS[b][0]:g}-{BINS[b][1]:g} m: eligible {len(elig)}, pool {pool_n}, '
                  f'p25/p50/p75 n_pts {np.round(qv, 1).tolist()}, picked n_pts '
                  f'{[int(npts[g]) for g in elig[pick]]}', flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(rule=dict(bins=BINS, qs=QS, pool_n=POOL_N, seed=SEED, long_m=LONG_M), entries=sel),
              open(out / 'selection.json', 'w'), indent=1)
    print(f'[select] {len(sel)} boxes -> {out / "selection.json"}', flush=True)


def _load_sel(a, cols):
    s = json.load(open(Path(a.out) / 'selection.json'))
    return [e for e in s['entries'] if e['col'] in cols]


def _desc_match(dsc, row, keys=(1, 2, 3, 4)):
    return bool(all(abs(float(dsc[k - 1]) - float(row[k])) < 1e-6 for k in keys))


def _context(P, box, exclude):
    fd = footprint_distance(P[:, :2], box)
    m = (fd > 0) & (fd <= CTX_M)
    if len(exclude):
        m[exclude] = False
    return np.ascontiguousarray(P[m, :3], np.float32)


def xy_mask(xyz):
    """The range mask in x, y only (the grid's z cut is applied separately, so in-box points it removes can be shown)."""
    from source_coverage import PCR
    x = np.asarray(xyz)
    return (x[:, 0] >= PCR[0]) & (x[:, 0] < PCR[3]) & (x[:, 1] >= PCR[1]) & (x[:, 1] < PCR[4])


def z_in_grid(xyz):
    from source_coverage import PCR
    z = np.asarray(xyz)[:, 2]
    return (z >= PCR[2]) & (z < PCR[5])


def _inbox_split(P3xy, box):
    """In-box points of the x, y-masked cloud and which of them pass the grid's z cut. The in-box test is per point, so
    the in-grid subset equals the in-box points of the fully cropped cloud (the model input)."""
    from source_coverage import in_box_indices
    idx = in_box_indices(P3xy, np.asarray(box)[None])[0]
    return idx, z_in_grid(P3xy[idx])


# ------------------------------------------------------------------------------------------------------ extract A
def extract_a(a):
    import pcdet
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.point_calibration import _augmentation_off
    from source_coverage import _build, car_boxes, descriptors
    print(f'[extract_a] pcdet from {pcdet.__file__}', flush=True)
    sel = _load_sel(a, ('A',))
    cfg = EasyDict(); cfg_from_yaml_file(NUS_CFG, cfg)
    out = []
    for plat, key in enumerate(NUS_KEYS):
        mine = [e for e in sel if e['plat'] == plat]
        if not mine:
            continue
        dc = cfg.DATA_CONFIGS[key]
        print(f'[extract_a] {key}: MAX_SWEEPS {dc.MAX_SWEEPS}, SHIFT_COOR {list(dc.SHIFT_COOR)}, compensation '
              f'{dc.get("GT_BOXES_MOTION_COMPENSATION")}, src features {list(dc.POINT_FEATURE_ENCODING.src_feature_list)}',
              flush=True)
        tcol = list(dc.POINT_FEATURE_ENCODING.src_feature_list).index('timestamp')
        ds = _build(dc, cfg)
        _augmentation_off(ds).__enter__()
        CAP = {}
        fwd0 = ds.data_augmentor.forward

        def fwd(data_dict, fwd0=fwd0):
            o = fwd0(data_dict)
            CAP['points'] = np.copy(o['points'])
            return o
        ds.data_augmentor.forward = fwd
        pose = sensor_pose('nus', shift=dc.SHIFT_COOR)
        for e in mine:
            i = e['frame']; CAP.clear()
            d = ds[i]
            pts = CAP['points']
            cars = car_boxes(d.get('gt_boxes'))
            box = cars[e['j']]
            P = pts[xy_mask(pts[:, :3])]
            P3 = np.ascontiguousarray(P[:, :3], np.float32)
            idx, ing = _inbox_split(P3, box)
            dsc = descriptors(P3[idx[ing]])
            info = ds.infos[i]
            e.update(box=box, pose=pose, pts=P3[idx], ingrid=ing, lag=P[idx, tcol].astype(np.float32),
                     ctx=_context(P3, box, idx),
                     desc=list(dsc), match=_desc_match(dsc, e['row']) and abs(np.hypot(*box[:2]) - e['row'][0]) < 1e-3,
                     alpha=relative_view_angle_deg(box, pose['origin']), platform=PLAT_NAME[plat],
                     frame_id=str(info['lidar_path']), token=str(info.get('token', '')),
                     max_sweeps=int(dc.MAX_SWEEPS))
            print(f'[extract_a] {key} info {i} box {e["j"]}: r {np.hypot(*box[:2]):.2f} desc {dsc} row '
                  f'{e["row"][1:5]} match {e["match"]}; lags {np.unique(np.round(e["lag"], 3))[:20]}', flush=True)
            out.append(e)
        del ds
    pickle.dump(out, open(Path(a.out) / 'extract_A.pkl', 'wb'))
    print(f'[extract_a] {len(out)} boxes, {sum(e["match"] for e in out)} match nus_base.npz', flush=True)


# ------------------------------------------------------------------------------------------------------ extract B
def extract_b(a):
    import pcdet
    from pcdet.datasets.augmentor import relocate_render as rr
    from rr_gate import build, V2_COPY_COLS, PLAT_ID
    from source_coverage import crop, descriptors
    print(f'[extract_b] pcdet from {pcdet.__file__}; RR from {rr.__file__}', flush=True)
    C = {k: j for j, k in enumerate(V2_COPY_COLS)}
    sel = _load_sel(a, ('B',))
    out = []
    for key in NUS_KEYS:
        plat = PLAT_ID[key]
        mine = sorted([e for e in sel if e['plat'] == plat], key=lambda e: e['frame'])
        if not mine:
            continue
        cfg, dcfg, ds = build(RR2_CFG, key)
        aug = ds.data_augmentor
        q0 = aug.data_augmentor_queue[0]
        assert getattr(q0, 'func', None) is not None and q0.func.__name__ == 'relocate_render', 'RR is not first'
        rr_cfg = q0.keywords['config']; lat = q0.keywords['lattice']; db = q0.keywords['db']
        assert int(rr_cfg.VERSION) == 2, rr_cfg.VERSION
        CAP = {}

        def rr_step(data_dict=None):            # the gate's capture (rr_gate.run_v2), nothing else
            o, rec = rr.relocate_render_scene_v2(data_dict, rr_cfg, lat, db, rng=np.random, keep_points=True)
            CAP['rec'] = rec; CAP['points'] = np.copy(o['points'])
            return o
        aug.data_augmentor_queue = [rr_step] + list(aug.data_augmentor_queue)[1:]
        pose = sensor_pose('rr', height=lat['height'])
        for e in mine:
            i = e['frame']; CAP.clear()
            np.random.seed((plat * 1000003 + i) % (2 ** 31))        # the gate's per-frame seed
            ds[i]
            rec, P = CAP['rec'], CAP['points']
            c = rec['copies'][e['j']]
            row = e['row']
            kept = np.ascontiguousarray(P[c['point_index'], :3], np.float32)
            g = crop(kept); dsc = descriptors(g)
            ing = xy_mask(kept) & z_in_grid(kept)
            assert ing.sum() == len(g)
            same = (abs(c['r_new'] - row[C['r_new']]) < 1e-6 and abs(c['p_keep'] - row[C['p_keep']]) < 1e-9 and
                    c['n_kept'] == row[C['n_kept']] and c['n_final'] == row[C['n_final']] and
                    c['instance'] == row[C['instance']] and
                    all(abs(dsc[k] - row[C[n]]) < 1e-6 for k, n in enumerate(('n_pts_grid', 'n_vox', 'n_layers', 'ppv'))))
            iscopy = np.zeros(len(P), bool)
            for cc in rec['copies']:
                iscopy[cc['point_index']] = True
            scene = np.ascontiguousarray(P[~iscopy, :3], np.float32)
            box = np.asarray(c['box'], np.float64)
            e.update(box=box, pose=pose, pts=kept, ingrid=ing, unsp=np.asarray(c['rendered_unsparsified'], np.float32),
                     ctx=_context(scene, box, np.zeros(0, np.int64)), desc=list(dsc), match=bool(same),
                     alpha=relative_view_angle_deg(box, pose['origin']), alpha_rr=float(np.degrees(c['alpha'])),
                     platform=PLAT_NAME[plat], frame_id=f'train info {i}', instance=int(c['instance']),
                     p_keep=float(c['p_keep']), n_rendered=int(c['n_rendered']), n_visible=int(c['n_visible']),
                     n_kept=int(c['n_kept']), n_final=int(c['n_final']), n_db=int(c['n_db']),
                     thetas_deg=np.degrees(lat['thetas']), lattice_segments=int(lat['n_segments']))
            print(f'[extract_b] {key} info {i} copy {e["j"]}: r {c["r_new"]:.2f} p {c["p_keep"]:.3f} kept '
                  f'{c["n_kept"]} final {c["n_final"]} desc {dsc} gate {[row[C[n]] for n in ("n_pts_grid", "n_vox", "n_layers")]}'
                  f' reproduced {same}', flush=True)
            out.append(e)
        del ds
    pickle.dump(out, open(Path(a.out) / 'extract_B.pkl', 'wb'))
    print(f'[extract_b] {len(out)} copies, {sum(e["match"] for e in out)} reproduce the gate v2 rows', flush=True)


# ----------------------------------------------------------------------------------------------------- extract CD
def extract_cd(a):
    from source_coverage import descriptors
    sel = _load_sel(a, ('C', 'D'))
    p = pickle.load(open(PASS_PKL, 'rb'))
    infos = pickle.load(open(TOOLS.parent / 'data/waymo/waymo_infos_val.pkl', 'rb'))
    by_fid = {}
    for inf in infos:
        fid = inf['frame_id'] if 'frame_id' in inf else \
            inf['point_cloud']['lidar_sequence'] + '_%03d' % inf['point_cloud']['sample_idx']
        by_fid[fid] = inf
    res = pickle.load(open(p['runs'][0][1], 'rb'))
    fids = [r['frame_id'] for r in res][:p['n_frames']]
    del res
    calib = pickle.load(open(TOP_CALIB, 'rb'))
    data_path = TOOLS.parent / 'data/waymo/waymo_processed_data'
    out = []
    for e in sel:
        inf = by_fid[fids[e['fi']]]
        an = inf['annos']; names = np.asarray(an['name'])
        gv = np.asarray(an['gt_boxes_lidar'], np.float64).reshape(len(names), -1)[:, :7][names == 'Vehicle']
        box = gv[e['j']]
        seq = inf['point_cloud']['lidar_sequence']; sidx = int(inf['point_cloud']['sample_idx'])
        raw = np.load(data_path / seq / ('%04d.npy' % sidx))
        counts = np.asarray(inf['num_points_of_each_lidar'], np.int64)
        assert counts.sum() == len(raw), 'per-lidar counts do not match the file'
        lid = np.repeat(np.arange(len(counts)), counts)
        keep = raw[:, 5] == -1                                   # WaymoDataset.get_lidar: no-label-zone points dropped
        pts, lid = raw[keep, :3], lid[keep]
        m = xy_mask(pts)
        P3 = np.ascontiguousarray(pts[m], np.float32); L = lid[m]
        idx, ing = _inbox_split(P3, box)
        dsc = descriptors(P3[idx[ing]])
        cal = calib[seq]
        pose = sensor_pose('waymo', extrinsic=cal['extrinsic'])
        e.update(box=box, pose=pose, pts=P3[idx], ingrid=ing, lidar=L[idx].astype(np.int8), ctx=_context(P3, box, idx),
                 desc=list(dsc), match=_desc_match(dsc, e['row']) and abs(np.hypot(*box[:2]) - e['row'][0]) < 1e-3,
                 alpha=relative_view_angle_deg(box, pose['origin']), platform='Waymo', frame_id=f'{seq} {sidx:03d}',
                 segment=seq, sample_idx=sidx, incl_deg=np.sort(np.degrees(np.asarray(cal['inclinations'], np.float64))))
        print(f'[extract_cd] {e["col"]} {seq[:24]} {sidx}: r {np.hypot(*box[:2]):.2f} desc {dsc} row {e["row"][1:5]} '
              f'match {e["match"]}; TOP {int((L[idx] == 0).sum())} of {len(idx)}', flush=True)
        out.append(e)
    pickle.dump(out, open(Path(a.out) / 'extract_CD.pkl', 'wb'))
    print(f'[extract_cd] {len(out)} boxes, {sum(e["match"] for e in out)} match waymo_val.npz', flush=True)


# ---------------------------------------------------------------------------------------------------------- fill
# Silhouette fill (20261018_03 §7): how many of the lattice pixels whose rays hit a box hold a return of the object.
OCCL_M = 0.5            # RR's / G1's occlusion margin: a pixel whose nearest return is outside the box and this much in front
W_TOP = 2650


def col_of_az(az_s, conv, yaw=0.0, n_cols=W_TOP):
    """Lattice column of sensor-frame azimuths. 'rr': floor((az + pi) / 2 pi * n) (rerender_utils.to_lattice);
    'waymo': the TOP range-image column floor(mod(n - (az_v / pi + 1) n / 2, n)), az_v = az_s + yaw
    (waymo_rings.top_columns_and_elevations)."""
    a = np.asarray(az_s, np.float64)
    if conv == 'rr':
        aw = np.mod(a + np.pi, 2 * np.pi) - np.pi
        return np.clip(np.floor((aw + np.pi) / (2 * np.pi) * n_cols).astype(np.int64), 0, n_cols - 1)
    c = np.mod(n_cols - (a + yaw) / np.pi * n_cols / 2.0 - n_cols / 2.0, n_cols)
    return np.floor(c).astype(np.int64) % n_cols


def az_of_col(col, conv, yaw=0.0, n_cols=W_TOP):
    """Sensor-frame azimuth of a column's centre (the inverse of col_of_az)."""
    j = np.asarray(col, np.float64) + 0.5
    if conv == 'rr':
        return j / n_cols * 2 * np.pi - np.pi
    return (2.0 * (n_cols - j) / n_cols - 1.0) * np.pi - yaw


def silhouette_pixels(box, pose, inc_desc, conv, n_cols=W_TOP, margin=0.02, full=False):
    """Lattice pixels whose ray from the sensor hits the box: (pixel ids row * n_cols + col, entry range). Rows index
    `inc_desc` (descending inclinations, row 0 = highest, as lattice_pixels / top_beam_ids). The box is taken into the
    sensor's axes (pose rot: sensor -> frame) and tested with the slab method; candidates are the rows / columns within
    `margin` rad of the corners' angles. full=True also returns the exit range and the entry point in the frame."""
    inc = np.asarray(inc_desc, np.float64)
    R = np.eye(3) if pose['rot'] is None else np.asarray(pose['rot'], np.float64)
    yaw = np.radians(pose['yaw_deg']) if conv == 'waymo' else 0.0
    b = np.asarray(box, np.float64)
    c = R.T @ (b[:3] - pose['origin'])
    ch, sh = np.cos(b[6]), np.sin(b[6])
    Rb = R.T @ np.array([[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]])      # box axes in the sensor frame
    corners = (box_corners(np.r_[0.0, 0.0, 0.0, b[3:6], 0.0]) @ Rb.T) + c
    el = np.arctan2(corners[:, 2], np.hypot(corners[:, 0], corners[:, 1]))
    az = np.arctan2(corners[:, 1], corners[:, 0]); az0 = np.arctan2(c[1], c[0])
    daz = np.radians(wrap_deg(np.degrees(az - az0)))
    rows = np.nonzero((inc >= el.min() - margin) & (inc <= el.max() + margin))[0]
    step = np.pi / n_cols / 2
    if not len(rows) or daz.max() - daz.min() > np.pi / 2:
        if full:
            return np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros((0, 3))
        return np.zeros(0, np.int64), np.zeros(0)
    cols = np.unique(col_of_az(az0 + np.arange(daz.min() - margin, daz.max() + margin + step, step), conv, yaw, n_cols))
    Rr, Cc = np.meshgrid(rows, cols, indexing='ij'); Rr, Cc = Rr.ravel(), Cc.ravel()
    phi, th = inc[Rr], az_of_col(Cc, conv, yaw, n_cols)
    d = np.stack([np.cos(phi) * np.cos(th), np.cos(phi) * np.sin(th), np.sin(phi)], 1)
    o = Rb.T @ (-c); u = d @ Rb; e = b[3:6] / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (-e - o) / u; t2 = (e - o) / u
    t1 = np.where(np.isfinite(t1), t1, -np.inf); t2 = np.where(np.isfinite(t2), t2, np.inf)
    tmin = np.minimum(t1, t2).max(1); tmax = np.maximum(t1, t2).min(1)
    hit = tmax >= np.maximum(tmin, 0)
    t_in = np.maximum(tmin[hit], 0)
    if full:
        ent = (d[hit] * t_in[:, None]) @ R.T + pose['origin']
        return Rr[hit] * n_cols + Cc[hit], t_in, tmax[hit], ent
    return Rr[hit] * n_cols + Cc[hit], t_in


def visible_mask(sil, tin, npp, inbox, occl=OCCL_M):
    """True for silhouette pixels NOT occluded: occluded = the pixel's nearest return is not the object's and lies more
    than occl in front of the box entry. A pixel without any return is visible (RR's / G1's rule)."""
    up, uidx, urng = npp
    vis = np.ones(len(sil), bool)
    if len(up) and len(sil):
        pos = np.clip(np.searchsorted(up, sil), 0, len(up) - 1)
        has = up[pos] == sil
        vis = ~(has & ~inbox[uidx[pos]] & (urng[pos] < tin - occl))
    return vis


BANDS4 = (0.0, 0.25, 0.5, 0.75)
ROW_COLS = ('row', 'n_vis', 'n_hit', 'relh', 'n_pts', 'mean_val', 'n_none', 'n_behind', 'n_near')


def row_records(sil, tin, tout, ent, vis, obj_pix, obj_val, npp, box, n_cols=W_TOP, behind_m=OCCL_M):
    """Per row (beam) of the visible silhouette (20261018_03 §8.1), columns ROW_COLS:
    n_vis visible pixels; n_hit of them holding >= 1 object point; relh = median over its visible pixels of the ray's entry
    point height above the box bottom / box height (clipped to [0, 1]; a ray entering through the roof reads 1); n_pts
    object points in that row; mean_val = mean obj_val of those points (nan if none); and the row's visible pixels WITHOUT
    an object point, by the pixel's own nearest return: n_none (no return at all), n_behind (beyond the box exit +
    behind_m: seen through the box), n_near (any other return)."""
    vp, ve, vto = sil[vis], ent[vis], tout[vis]
    if not len(vp):
        return np.zeros((0, len(ROW_COLS)))
    b = np.asarray(box, np.float64)
    relh = np.clip((ve[:, 2] - (b[2] - b[5] / 2)) / b[5], 0.0, 1.0)
    hit = np.isin(vp, obj_pix)
    up, _, urng = npp
    if len(up):
        pos = np.clip(np.searchsorted(up, vp), 0, len(up) - 1)
        has = up[pos] == vp
        rng = np.where(has, urng[pos], np.nan)
    else:
        has = np.zeros(len(vp), bool); rng = np.full(len(vp), np.nan)
    with np.errstate(invalid='ignore'):
        far = has & (rng > vto + behind_m)
    none, behind, near = ~hit & ~has, ~hit & far, ~hit & has & ~far
    rows = vp // n_cols
    ur, inv = np.unique(rows, return_inverse=True)
    k = len(ur)
    o = np.argsort(inv, kind='stable'); starts = np.r_[0, np.cumsum(np.bincount(inv, minlength=k))[:-1]]
    relh_med = np.array([np.median(relh[o[s:s + n]]) for s, n in zip(starts, np.bincount(inv, minlength=k))])
    op = np.asarray(obj_pix, np.int64); ov = np.asarray(obj_val, np.float64)
    prow = op // n_cols; m = np.isin(prow, ur); pk = np.searchsorted(ur, prow[m])
    n_pts = np.bincount(pk, minlength=k).astype(np.float64)
    sv = np.bincount(pk, weights=ov[m], minlength=k)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_val = np.where(n_pts > 0, sv / np.maximum(n_pts, 1), np.nan)
    return np.c_[ur, np.bincount(inv, minlength=k), np.bincount(inv, weights=hit, minlength=k), relh_med, n_pts, mean_val,
                 np.bincount(inv, weights=none, minlength=k), np.bincount(inv, weights=behind, minlength=k),
                 np.bincount(inv, weights=near, minlength=k)]


def nearest_per_pixel(pix, rng):
    """Per occupied pixel (sorted ids): the index and range of its nearest return (pixel < 0 ignored)."""
    ok = np.nonzero(np.asarray(pix) >= 0)[0]
    if not len(ok):
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    o = ok[np.lexsort((rng[ok], pix[ok]))]
    first = np.r_[True, pix[o][1:] != pix[o][:-1]]
    return pix[o[first]], o[first], rng[o[first]]


def fill_stats(sil, tin, obj_pix_sets, npp, inbox, n_cols=W_TOP, occl=OCCL_M):
    """Visible silhouette = silhouette minus pixels whose nearest return is NOT the object's and lies > occl in front of
    the box entry (RR's / G1's definition). Per object point set (pixel ids): (fill = distinct visible pixels holding a
    point / visible pixels, row coverage = rows of those / rows of the visible silhouette). Returns (n_sil, n_vis,
    [(fill, rc, n_filled)] per set)."""
    vis = visible_mask(sil, tin, npp, inbox, occl)
    vp = np.unique(sil[vis])
    out = []
    for op in obj_pix_sets:
        f = np.intersect1d(np.unique(op), vp)
        out.append((len(f) / len(vp) if len(vp) else np.nan,
                    len(np.unique(f // n_cols)) / len(np.unique(vp // n_cols)) if len(vp) else np.nan, len(f)))
    return len(np.unique(sil)), len(vp), out


def fill_b(a):
    """All gate v2 copies of one platform, re-rendered with the gate's config, seeds and capture (rr_gate.run_v2)."""
    import pcdet
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.processor.point_sampler import lattice_pixels
    from rr_gate import build, V2_COPY_COLS, PLAT_ID
    C = {k: j for j, k in enumerate(V2_COPY_COLS)}
    print(f'[fill_b] pcdet from {pcdet.__file__}', flush=True)
    key = a.key; plat = PLAT_ID[key]
    g = pickle.load(open(GATE_DIR / f'gate_v2_{key}.pkl', 'rb'))
    gcp = g['copies']
    cfg, dcfg, ds = build(RR2_CFG, key)
    aug = ds.data_augmentor; q0 = aug.data_augmentor_queue[0]
    rr_cfg = q0.keywords['config']; lat = q0.keywords['lattice']; db = q0.keywords['db']
    assert int(rr_cfg.VERSION) == 2
    CAP = {}

    def rr_step(data_dict=None):
        o, rec = rr.relocate_render_scene_v2(data_dict, rr_cfg, lat, db, rng=np.random, keep_points=True)
        CAP['rec'] = rec; CAP['points'] = np.copy(o['points'])
        return o
    aug.data_augmentor_queue = [rr_step] + list(aug.data_augmentor_queue)[1:]
    th, nc, h = lat['thetas'], lat['n_cols'], lat['height']
    inc_desc = np.sort(th)[::-1]
    pose = sensor_pose('rr', height=h)
    rows, n_mismatch, row_recs = [], 0, []
    t0 = time.time()
    for n_done, i in enumerate(g['frames']):
        CAP.clear()
        np.random.seed((plat * 1000003 + i) % (2 ** 31))
        ds[i]
        rec, P = CAP['rec'], CAP['points']
        grow = gcp[gcp[:, C['frame']] == i]
        if len(grow) != len(rec['copies']):
            n_mismatch += 1
        if not rec['copies']:
            continue
        iscopy = np.zeros(len(P), bool)
        for c in rec['copies']:
            iscopy[c['point_index']] = True
        scene = P[~iscopy, :3]
        spix, _, srng = lattice_pixels(scene, th, nc, h)
        for j, c in enumerate(rec['copies']):
            if j < len(grow) and not (abs(grow[j, C['r_new']] - c['r_new']) < 1e-6 and grow[j, C['n_final']] == c['n_final']):
                n_mismatch += 1
            ru = np.asarray(c['rendered_unsparsified'], np.float64)
            kp = P[c['point_index'], :3]
            rpix, _, rrng = lattice_pixels(ru, th, nc, h)
            kpix, _, _ = lattice_pixels(kp, th, nc, h)
            allpix = np.r_[spix, rpix]; allrng = np.r_[srng, rrng]
            inbox = np.r_[np.zeros(len(spix), bool), np.ones(len(rpix), bool)]
            npp = nearest_per_pixel(allpix, allrng)
            sil, tin, tout, ent = silhouette_pixels(c['box'], pose, inc_desc, 'rr', nc, full=True)
            n_sil, n_vis, ((f_r, rc_r, nf_r), (f_k, rc_k, nf_k)) = fill_stats(sil, tin, [rpix[rpix >= 0], kpix[kpix >= 0]],
                                                                              npp, inbox, nc)
            if a.rows:                               # §8: per-row records of the render (0) and the kept points (1)
                vis = visible_mask(sil, tin, npp, inbox)
                for which, op in ((0, rpix[rpix >= 0]), (1, kpix[kpix >= 0])):
                    rr_ = row_records(sil, tin, tout, ent, vis, op, np.zeros(len(op)), npp, c['box'], nc)
                    row_recs.append(np.c_[np.tile([plat, i, j, which, c['r_new'], n_vis / max(n_sil, 1),
                                                   float(c['box'][3])], (len(rr_), 1)), rr_])
            insil = float(np.isin(rpix, sil).mean()) if len(rpix) else np.nan            # self-check: ~1
            rows.append([plat, i, j, c['r_new'], c['p_keep'], len(ru), len(kp), n_sil, n_vis, f_r, rc_r, nf_r, f_k, rc_k,
                         nf_k, float(c['box'][3]), insil])
        if n_done % 25 == 0:
            print(f'[fill_b] {key} {n_done + 1}/{len(g["frames"])} frames, {len(rows)} copies, {time.time() - t0:.0f} s',
                  flush=True)
    cols = ('plat', 'frame', 'j', 'r', 'p_keep', 'n_render', 'n_kept', 'n_sil', 'n_vis', 'fill_render', 'rc_render',
            'nfill_render', 'fill_kept', 'rc_kept', 'nfill_kept', 'length', 'render_in_sil')
    rows = np.array(rows, np.float64).reshape(-1, len(cols))
    if a.rows:
        rcols = ('plat', 'frame', 'j', 'which', 'r', 'unocc', 'length') + ROW_COLS
        np.savez(Path(a.out) / f'rows_B_{key}.npz', rows=np.concatenate(row_recs).astype(np.float32),
                 cols=np.array(rcols), n_mismatch=n_mismatch)
        print(f'[fill_b] rows: {sum(len(x) for x in row_recs)} row records -> rows_B_{key}.npz', flush=True)
        return
    np.savez(Path(a.out) / f'fill_B_{key}.npz', rows=rows, cols=np.array(cols), n_mismatch=n_mismatch, n_gate=len(gcp))
    print(f'[fill_b] self-check: rendered pixels in the silhouette, p5 / p50 '
          f'{np.round(np.nanpercentile(rows[:, -1], (5, 50)), 3).tolist()}', flush=True)
    print(f'[fill_b] {key}: {len(rows)} copies (gate {len(gcp)}), frames or copies differing from the gate: {n_mismatch}',
          flush=True)


G = {}


def _fill_cd_frame(fi):
    import torch
    from scipy.spatial import cKDTree
    from pcdet.datasets.waymo.waymo_rings import top_beam_ids
    from source_coverage import in_box_indices
    torch.set_num_threads(1)
    f = G['frames'][fi]
    sel = f['sel']
    if not len(sel):
        return fi, []
    raw = np.load(G['data_path'] / f['seq'] / ('%04d.npy' % f['idx']))
    n_top = int(f['counts'][0])
    cal = G['calib'][f['seq']]
    beam, colf, _, inc_desc, _ = top_beam_ids(raw[:n_top, :3], cal['extrinsic'], cal['inclinations'])
    pix_all = beam * W_TOP + (np.floor(colf).astype(np.int64) % W_TOP)
    pose = sensor_pose('waymo', extrinsic=cal['extrinsic'])
    top = raw[:n_top, :3].astype(np.float64)
    rng_all = np.linalg.norm(top - pose['origin'], axis=1)
    keep = raw[:n_top, 5] == -1                                # evaluated cloud: no-label-zone points dropped
    npp = nearest_per_pixel(pix_all, rng_all)                  # occlusion from every TOP return (first returns)
    P = np.ascontiguousarray(top[keep], np.float32); pk = pix_all[keep]; kidx = np.nonzero(keep)[0]
    tree = cKDTree(P[:, :2]) if len(P) else None
    boxes = f['gv'][sel]
    idx = in_box_indices(P, boxes, tree)
    out = []
    for k, g in enumerate(sel):
        inbox = np.zeros(n_top, bool); inbox[kidx[idx[k]]] = True
        sil, tin = silhouette_pixels(boxes[k], pose, inc_desc, 'waymo', W_TOP)
        n_sil, n_vis, ((fl, rc, nf),) = fill_stats(sil, tin, [pk[idx[k]]], npp, inbox, W_TOP)
        insil = float(np.isin(pk[idx[k]], sil).mean()) if len(idx[k]) else np.nan     # self-check: ~1
        out.append([G['g0'][fi] + g, float(np.hypot(*boxes[k][:2])), len(idx[k]), n_sil, n_vis, fl, rc, nf, insil])
    return fi, out


def fill_cd(a):
    """All Waymo val Vehicle GT of columns C / D in the range bins: silhouette from the segment's own inclinations, its
    TOP extrinsic and the range-image columns; rows of the points from the stored row order (waymo_rings.top_beam_ids);
    first returns (the processed files hold the first return only)."""
    from multiprocessing import Pool
    S = {k: j for j, k in enumerate(SC_COLS)}
    p = pickle.load(open(PASS_PKL, 'rb'))
    infos = pickle.load(open(TOOLS.parent / 'data/waymo/waymo_infos_val.pkl', 'rb'))
    by_fid = {}
    for inf in infos:
        fid = inf['frame_id'] if 'frame_id' in inf else \
            inf['point_cloud']['lidar_sequence'] + '_%03d' % inf['point_cloud']['sample_idx']
        by_fid[fid] = inf
    res = pickle.load(open(p['runs'][0][1], 'rb'))
    fids = [r_['frame_id'] for r_ in res][:p['n_frames']]
    del res
    w = np.load(SC_DIR / 'waymo_val.npz'); wc = w['car']
    rb = _bin_of(wc[:, S['r']])
    ok = (wc[:, S['n_pts']] >= 1) & (wc[:, S['length']] < LONG_M) & (rb >= 0)
    frames, g0 = [], []
    n = 0
    for fid in fids:
        inf = by_fid[fid]; an = inf['annos']; names = np.asarray(an['name'])
        gv = np.asarray(an['gt_boxes_lidar'], np.float64).reshape(len(names), -1)[:, :7][names == 'Vehicle'] \
            if len(names) else np.zeros((0, 7))
        sel = np.nonzero(ok[n:n + len(gv)])[0]
        frames.append(dict(seq=inf['point_cloud']['lidar_sequence'], idx=int(inf['point_cloud']['sample_idx']), gv=gv,
                           sel=sel, counts=np.asarray(inf['num_points_of_each_lidar'], np.int64)))
        g0.append(n); n += len(gv)
    assert n == len(wc)
    del infos, by_fid
    G.update(frames=frames, g0=g0, data_path=TOOLS.parent / 'data/waymo/waymo_processed_data',
             calib=pickle.load(open(TOP_CALIB, 'rb')))
    tasks = [k for k in range(len(frames)) if len(frames[k]['sel'])][:a.max_frames or None]
    print(f'[fill_cd] {len(tasks)} frames, {int(ok.sum())} GT in the bins', flush=True)
    rows = []; t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (fi, out) in enumerate(pool.imap_unordered(_fill_cd_frame, tasks, chunksize=4)):
            rows += out
            if k % 500 == 0:
                print(f'[fill_cd] {k + 1}/{len(tasks)} frames, {len(rows)} GT, {time.time() - t0:.0f} s', flush=True)
    rows = np.array(sorted(rows), np.float64).reshape(-1, 9)
    gi = rows[:, 0].astype(np.int64)
    miss = (w['oracle_found5'].astype(bool) & ~w['ours27596_found5'].astype(bool))[gi]
    cols = ('g', 'r', 'n_top_inbox', 'n_sil', 'n_vis', 'fill', 'rc', 'nfill', 'pts_in_sil')
    print(f'[fill_cd] self-check: in-box TOP points whose pixel is in the silhouette, p5 / p50 '
          f'{np.round(np.nanpercentile(rows[:, 8], (5, 50)), 3).tolist()}', flush=True)
    np.savez(Path(a.out) / 'fill_CD.npz', rows=rows, cols=np.array(cols), miss=miss,
             n_pts_grid=wc[gi, S['n_pts']], n_vox=wc[gi, S['n_vox']])
    print(f'[fill_cd] {len(rows)} GT ({int(miss.sum())} misses) in {time.time() - t0:.0f} s', flush=True)


def _rows_cd_frame(fi):
    """§8: per-row records, per-box records and per-point records of one Waymo val frame (TOP first returns; the stored
    file's RAW intensity, column 3; WaymoDataset.get_lidar applies tanh to it on load)."""
    import torch
    from scipy.spatial import cKDTree
    from pcdet.datasets.waymo.waymo_rings import top_beam_ids
    from source_coverage import in_box_indices
    torch.set_num_threads(1)
    f = G['frames'][fi]
    sel = f['sel']
    if not len(sel):
        return fi, None
    raw = np.load(G['data_path'] / f['seq'] / ('%04d.npy' % f['idx']))
    n_top = int(f['counts'][0])
    cal = G['calib'][f['seq']]
    beam, colf, _, inc_desc, _ = top_beam_ids(raw[:n_top, :3], cal['extrinsic'], cal['inclinations'])
    pix_all = beam * W_TOP + (np.floor(colf).astype(np.int64) % W_TOP)
    pose = sensor_pose('waymo', extrinsic=cal['extrinsic'])
    top = raw[:n_top, :3].astype(np.float64)
    rng_all = np.linalg.norm(top - pose['origin'], axis=1)
    keep = raw[:n_top, 5] == -1
    npp = nearest_per_pixel(pix_all, rng_all)
    P = np.ascontiguousarray(top[keep], np.float32); pk = pix_all[keep]; kidx = np.nonzero(keep)[0]
    inten = raw[:n_top, 3][keep].astype(np.float64)
    tree = cKDTree(P[:, :2]) if len(P) else None
    boxes = f['gv'][sel]
    idx = in_box_indices(P, boxes, tree)
    R, B, Pt = [], [], []
    for k, g in enumerate(sel):
        gid = G['g0'][fi] + g
        bx = boxes[k]
        inbox = np.zeros(n_top, bool); inbox[kidx[idx[k]]] = True
        sil, tin, tout, ent = silhouette_pixels(bx, pose, inc_desc, 'waymo', W_TOP, full=True)
        vis = visible_mask(sil, tin, npp, inbox)
        op, oi = pk[idx[k]], inten[idx[k]]
        med = float(np.median(oi)) if len(oi) else np.nan
        ov = oi / med if med > 0 else np.full(len(oi), np.nan)
        rr_ = row_records(sil, tin, tout, ent, vis, op, ov, npp, bx, W_TOP)
        vp = np.unique(sil[vis])
        nf = len(np.intersect1d(np.unique(op), vp))
        R.append(np.c_[np.full(len(rr_), gid), rr_])
        B.append([gid, float(np.hypot(*bx[:2])), len(np.unique(sil)), len(vp), nf / len(vp) if len(vp) else np.nan, med,
                  len(op), len(np.unique(op))])
        zb = bx[2] - bx[5] / 2
        Pt.append(np.c_[np.full(len(op), gid), np.clip((P[idx[k], 2] - zb) / bx[5], 0, 1), ov, oi])
    return fi, (np.concatenate(R), np.array(B, np.float64), np.concatenate(Pt))


def rows_cd(a):
    """§8 over all Waymo val Vehicle GT of columns C / D in the range bins (the §7 population and geometry)."""
    from multiprocessing import Pool
    S = {k: j for j, k in enumerate(SC_COLS)}
    p = pickle.load(open(PASS_PKL, 'rb'))
    infos = pickle.load(open(TOOLS.parent / 'data/waymo/waymo_infos_val.pkl', 'rb'))
    by_fid = {}
    for inf in infos:
        fid = inf['frame_id'] if 'frame_id' in inf else \
            inf['point_cloud']['lidar_sequence'] + '_%03d' % inf['point_cloud']['sample_idx']
        by_fid[fid] = inf
    res = pickle.load(open(p['runs'][0][1], 'rb'))
    fids = [r_['frame_id'] for r_ in res][:p['n_frames']]
    del res
    w = np.load(SC_DIR / 'waymo_val.npz'); wc = w['car']
    ok = (wc[:, S['n_pts']] >= 1) & (wc[:, S['length']] < LONG_M) & (_bin_of(wc[:, S['r']]) >= 0)
    frames, g0, n = [], [], 0
    for fid in fids:
        inf = by_fid[fid]; an = inf['annos']; names = np.asarray(an['name'])
        gv = np.asarray(an['gt_boxes_lidar'], np.float64).reshape(len(names), -1)[:, :7][names == 'Vehicle'] \
            if len(names) else np.zeros((0, 7))
        frames.append(dict(seq=inf['point_cloud']['lidar_sequence'], idx=int(inf['point_cloud']['sample_idx']), gv=gv,
                           sel=np.nonzero(ok[n:n + len(gv)])[0],
                           counts=np.asarray(inf['num_points_of_each_lidar'], np.int64)))
        g0.append(n); n += len(gv)
    assert n == len(wc)
    del infos, by_fid
    G.update(frames=frames, g0=g0, data_path=TOOLS.parent / 'data/waymo/waymo_processed_data',
             calib=pickle.load(open(TOP_CALIB, 'rb')))
    tasks = [k for k in range(len(frames)) if len(frames[k]['sel'])][:a.max_frames or None]
    R, B, Pt = [], [], []; t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (fi, out) in enumerate(pool.imap_unordered(_rows_cd_frame, tasks, chunksize=4)):
            if out is not None:
                R.append(out[0].astype(np.float32)); B.append(out[1]); Pt.append(out[2].astype(np.float32))
            if k % 1000 == 0:
                print(f'[rows_cd] {k + 1}/{len(tasks)} frames, {time.time() - t0:.0f} s', flush=True)
    B = np.concatenate(B); B = B[np.argsort(B[:, 0])]
    gi = B[:, 0].astype(np.int64)
    miss = (w['oracle_found5'].astype(bool) & ~w['ours27596_found5'].astype(bool))[gi]
    np.savez(Path(a.out) / 'rows_CD.npz', rows=np.concatenate(R), row_cols=np.array(('gid',) + ROW_COLS), boxes=B,
             box_cols=np.array(('gid', 'r', 'n_sil', 'n_vis', 'fill', 'med_int', 'n_pts', 'n_pix')), miss=miss,
             points=np.concatenate(Pt), point_cols=np.array(('gid', 'relh', 'rel_int', 'int_raw')))
    print(f'[rows_cd] {len(B)} boxes, {sum(len(x) for x in R)} rows, {sum(len(x) for x in Pt)} points, '
          f'{time.time() - t0:.0f} s', flush=True)


def holes_train(a):
    """§8 label-free per-beam check on Waymo TRAIN clouds: a pixel of beam b is a hole if it holds no TOP return while
    beams b-1 and b+1 in the same column both return within 1 m of each other in range. Frames: one random frame from each
    of a.n_frames random TRAIN segments (numpy default_rng(SEED)). Range bin = mean horizontal range of the two neighbours
    about the TOP. No label is read."""
    import glob
    from pcdet.datasets.waymo.waymo_rings import top_beam_ids
    dp = TOOLS.parent / 'data/waymo/waymo_processed_data'
    train = [ln.strip().replace('.tfrecord', '') for ln in open(TOOLS.parent / 'data/waymo/ImageSets/train.txt')
             if ln.strip()]
    rng = np.random.default_rng(SEED)
    segs = rng.choice(len(train), a.n_frames, replace=False)
    calib = pickle.load(open(TOP_CALIB, 'rb'))
    cand = np.zeros((64, len(BINS))); hole = np.zeros((64, len(BINS))); dup, used = [], []
    t0 = time.time()
    for n_done, s in enumerate(segs):
        seq = train[s]
        infos = pickle.load(open(glob.glob(str(dp / seq / '*.pkl'))[0], 'rb'))
        inf = infos[int(rng.integers(len(infos)))]
        sidx = int(inf['point_cloud']['sample_idx'])
        raw = np.load(dp / seq / ('%04d.npy' % sidx))
        n_top = int(inf['num_points_of_each_lidar'][0]); cal = calib[seq]
        beam, colf, _, inc_desc, diag = top_beam_ids(raw[:n_top, :3], cal['extrinsic'], cal['inclinations'])
        col = np.floor(colf).astype(np.int64) % W_TOP
        d = raw[:n_top, :3].astype(np.float64) - np.asarray(cal['extrinsic'])[:3, 3]
        r3, rh = np.linalg.norm(d, axis=1), np.hypot(d[:, 0], d[:, 1])
        img = np.full((64, W_TOP), np.inf); himg = np.full((64, W_TOP), np.nan)
        o = np.argsort(-r3)                                       # the nearest return of a pixel is written last
        img[beam[o], col[o]] = r3[o]; himg[beam[o], col[o]] = rh[o]
        dup.append(1.0 - np.isfinite(img).sum() / max(n_top, 1))
        up, mid, dn = img[:-2], img[1:-1], img[2:]
        okn = np.isfinite(up) & np.isfinite(dn) & (np.abs(up - dn) < 1.0)
        hl = okn & ~np.isfinite(mid)
        rb = _bin_of(0.5 * (himg[:-2] + himg[2:]).ravel()).reshape(okn.shape)
        for b in range(len(BINS)):
            cand[1:63, b] += (okn & (rb == b)).sum(1); hole[1:63, b] += (hl & (rb == b)).sum(1)
        used.append([s, sidx, diag['rows'], diag['max_resid_deg']])
        if n_done % 50 == 0:
            print(f'[holes_train] {n_done + 1}/{len(segs)} frames, {time.time() - t0:.0f} s', flush=True)
    np.savez(Path(a.out) / 'holes_train.npz', cand=cand, hole=hole, dup=np.array(dup), used=np.array(used, np.float64))
    print(f'[holes_train] {len(segs)} frames; pixels sharing a cell (duplicates) p50 {np.median(dup):.4f}', flush=True)


def donor_bands(a):
    """§8 donor side (H4), descriptive: per RR v2 donor instance (box frame, centred, 0.02 m centroid voxels), points per
    m^2 of box surface and intensity by relative height band (side faces split in four; the roof added to the top band;
    and ten bins). nuScenes intensity as stored in the database (0-255, voxel-averaged); relative = point / the instance's
    median (instances with median 0 skipped for the relative value)."""
    out = {}
    for key in NUS_KEYS:
        pts = np.load(DB_DIR / f'db_{key}_points.npy', mmap_mode='r')
        ix = np.load(DB_DIR / f'db_{key}_index.npz')
        off, cnt, size = ix['offset'], ix['count'], ix['size']
        dens, rel, ab, d10 = [], [], [], []
        n_skip = 0
        for i in range(len(off)):
            q = np.asarray(pts[off[i]:off[i] + cnt[i]], np.float64)
            dx, dy, dz = size[i]
            h = np.clip((q[:, 2] + dz / 2) / dz, 0, 1)
            band = np.minimum((h * 4).astype(np.int64), 3); b10 = np.minimum((h * 10).astype(np.int64), 9)
            area = np.full(4, 2 * (dx + dy) * dz / 4); area[3] += dx * dy
            a10 = np.full(10, 2 * (dx + dy) * dz / 10); a10[9] += dx * dy
            dens.append(np.bincount(band, minlength=4) / area); d10.append(np.bincount(b10, minlength=10) / a10)
            med = np.median(q[:, 3])
            ab.append([np.median(q[band == k, 3]) if (band == k).any() else np.nan for k in range(4)])
            if med > 0:
                rv = q[:, 3] / med
                rel.append([np.median(rv[band == k]) if (band == k).any() else np.nan for k in range(4)])
            else:
                n_skip += 1
        dens, d10, rel, ab = map(np.array, (dens, d10, rel, ab))
        out[key] = dict(n=len(off), n_rel_skipped=n_skip,
                        pts_per_m2_band_p25_50_75=np.nanpercentile(dens, (25, 50, 75), axis=0).round(0).T.tolist(),
                        pts_per_m2_10bins_p50=np.nanmedian(d10, 0).round(0).tolist(),
                        rel_int_band_p25_50_75=np.nanpercentile(rel, (25, 50, 75), axis=0).round(3).T.tolist(),
                        abs_int_band_p50=np.nanmedian(ab, 0).round(2).tolist())
        print(f'[donor_bands] {key}: {out[key]}', flush=True)
    json.dump(out, open(Path(a.out) / 'donor_bands.json', 'w'), indent=1)


def _spear(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return [np.nan, int(m.sum())]
    return [round(float(spearmanr(x[m], y[m])[0]), 3), int(m.sum())]


def _band_of(h, n=4):
    return np.minimum((np.asarray(h, np.float64) * n).astype(np.int64), n - 1)


def rows_report(a):
    """§8 tables (rows.json): per range bin and population (C unoccluded = visible fraction >= 0.9, primary; C all;
    D all; D unoccluded)."""
    out = Path(a.out)
    z = np.load(out / 'rows_CD.npz')
    Rw = z['rows'].astype(np.float64); Bx = z['boxes']; miss = z['miss']; Pt = z['points'].astype(np.float64)
    RC = {k: j for j, k in enumerate(z['row_cols'])}; BC = {k: j for j, k in enumerate(z['box_cols'])}
    pos = np.searchsorted(Bx[:, 0], Rw[:, RC['gid']]); ppos = np.searchsorted(Bx[:, 0], Pt[:, 0])
    assert np.array_equal(Bx[pos, 0], Rw[:, RC['gid']]) and np.array_equal(Bx[ppos, 0], Pt[:, 0])
    unocc = Bx[:, BC['n_vis']] / np.maximum(Bx[:, BC['n_sil']], 1)
    bbin = _bin_of(Bx[:, BC['r']])
    pops = {'C unoccluded': unocc >= 0.9, 'C all': np.ones(len(Bx), bool), 'D all': miss, 'D unoccluded': miss & (unocc >= 0.9)}
    hr = Rw[:, RC['n_hit']] / np.maximum(Rw[:, RC['n_vis']], 1)
    lost = Rw[:, RC['n_hit']] == 0
    band = _band_of(Rw[:, RC['relh']]); band10 = _band_of(Rw[:, RC['relh']], 10)
    pband = _band_of(Pt[:, 1])
    beam = Rw[:, RC['row']].astype(np.int64)
    res = {}
    dupl = 1 - Bx[:, BC['n_pix']] / np.maximum(Bx[:, BC['n_pts']], 1)
    res['check_points_sharing_a_pixel_p50_p95'] = np.round(np.percentile(dupl[Bx[:, BC['n_pts']] > 0], (50, 95)), 4).tolist()
    for pname, pm in pops.items():
        res[pname] = {}
        for b, (lo, hi) in enumerate(BINS):
            name = f'{lo:g}-{hi:g}'
            bm = pm & (bbin == b)
            rm = bm[pos]; pmk = bm[ppos]
            d = dict(n_cars=int(bm.sum()), n_rows=int(rm.sum()))
            tb = []
            for k in range(4):
                m = rm & (band == k); mp = pmk & (pband == k)
                lm = m & lost
                e = Rw[lm][:, [RC['n_none'], RC['n_behind'], RC['n_near']]].sum(0)
                tb.append(dict(band=f"{BANDS4[k]:.2f}-{BANDS4[k] + 0.25:.2f}", n_rows=int(m.sum()),
                               hit_pix=round(float(Rw[m, RC['n_hit']].sum() / max(Rw[m, RC['n_vis']].sum(), 1)), 3),
                               row_hit_mean=round(float(hr[m].mean()), 3) if m.any() else np.nan,
                               lost=round(float(lost[m].mean()), 3) if m.any() else np.nan,
                               row_rel_int_p50=round(float(np.nanmedian(Rw[m, RC['mean_val']])), 3) if m.any() else np.nan,
                               point_rel_int_p50=round(float(np.nanmedian(Pt[mp, 2])), 3) if mp.any() else np.nan,
                               point_raw_int_p50=round(float(np.nanmedian(Pt[mp, 3])), 4) if mp.any() else np.nan,
                               lost_px_none_behind_near=np.round(e / max(e.sum(), 1), 3).tolist()))
            d['bands'] = tb
            d['bins10_hit_pix'] = [round(float(Rw[rm & (band10 == k), RC['n_hit']].sum() /
                                               max(Rw[rm & (band10 == k), RC['n_vis']].sum(), 1)), 3) for k in range(10)]
            d['bins10_lost'] = [round(float(lost[rm & (band10 == k)].mean()), 3) if (rm & (band10 == k)).any() else np.nan
                                for k in range(10)]
            # beams
            bt = []
            for bb in range(64):
                m = rm & (beam == bb)
                if m.sum() >= 100:
                    bt.append([bb, int(m.sum()), Rw[m, RC['n_hit']].sum() / Rw[m, RC['n_vis']].sum(), lost[m].mean()])
            bt = np.array(bt).reshape(-1, 4)
            if len(bt):
                o = np.argsort(bt[:, 2])
                d['beams'] = dict(n_beams=len(bt), hit_pix_min_p10_p50_p90_max=np.round(
                    [bt[:, 2].min(), *np.percentile(bt[:, 2], (10, 50, 90)), bt[:, 2].max()], 3).tolist(),
                    worst3=[[int(bt[i, 0]), round(bt[i, 2], 3), int(bt[i, 1])] for i in o[:3]],
                    best3=[[int(bt[i, 0]), round(bt[i, 2], 3), int(bt[i, 1])] for i in o[::-1][:3]])
            wb = {}
            means = {}
            for k in range(4):
                m = rm & (band == k)
                rows_ = []
                for bb in np.unique(beam[m]):
                    mm = m & (beam == bb)
                    if mm.sum() >= 50:
                        v = hr[mm]; rows_.append([bb, len(v), v.mean(), v.std(ddof=1) / np.sqrt(len(v))])
                rows_ = np.array(rows_).reshape(-1, 4)
                if len(rows_) >= 3:
                    mu = hr[m].mean(); zz = (rows_[:, 2] - mu) / np.maximum(rows_[:, 3], 1e-9)
                    wb[tb[k]['band']] = dict(n_beams=len(rows_), beam_mean_p10_p90=np.round(
                        np.percentile(rows_[:, 2], (10, 90)), 3).tolist(), share_abs_z_gt3=round(float(np.mean(np.abs(zz) > 3)), 3),
                        worst=[int(rows_[np.argmin(rows_[:, 2]), 0]), round(float(rows_[:, 2].min()), 3)],
                        best=[int(rows_[np.argmax(rows_[:, 2]), 0]), round(float(rows_[:, 2].max()), 3)])
                    means[k] = dict(zip(rows_[:, 0].astype(int), rows_[:, 2]))
            cross = {}
            for k1, k2 in ((0, 1), (1, 2), (2, 3)):
                if k1 in means and k2 in means:
                    cb = sorted(set(means[k1]) & set(means[k2]))
                    cross[f'{k1}-{k2}'] = _spear([means[k1][x] for x in cb], [means[k2][x] for x in cb])
            d['beams_within_band'] = wb; d['beam_means_spearman_between_bands'] = cross
            # intensity
            m = rm & (Rw[:, RC['n_pts']] >= 1)
            d['rho_row_relint_vs_hit'] = _spear(Rw[m, RC['mean_val']], hr[m])
            d['rho_row_relint_vs_hit_within_band'] = [_spear(Rw[m & (band == k), RC['mean_val']], hr[m & (band == k)])
                                                      for k in range(4)]
            fi_, mi_ = Bx[bm, BC['fill']], Bx[bm, BC['med_int']]
            d['rho_car_fill_vs_median_int'] = _spear(mi_, fi_)
            okq = np.isfinite(mi_) & np.isfinite(fi_)
            if okq.sum() >= 40:
                qe = np.percentile(mi_[okq], (25, 50, 75)); qi = np.searchsorted(qe, mi_[okq])
                d['car_fill_p50_by_int_quartile'] = [round(float(np.median(fi_[okq][qi == k])), 3) for k in range(4)]
                d['int_quartile_edges'] = np.round(qe, 4).tolist()
            res[pname][name] = d
    # B rows
    res['B'] = {}
    for key in NUS_KEYS:
        f = out / f'rows_B_{key}.npz'
        if f.exists():
            zb = np.load(f); res.setdefault('_B_rows', []).append(zb['rows'].astype(np.float64)); bcols = list(zb['cols'])
    if '_B_rows' in res:
        Bc = {k: j for j, k in enumerate(bcols)}
        br = np.concatenate(res.pop('_B_rows'))
        bbn = _bin_of(br[:, Bc['r']]); bband = _band_of(br[:, Bc['relh']])
        for b, (lo, hi) in enumerate(BINS):
            name = f'{lo:g}-{hi:g}'
            res['B'][name] = {}
            for which, lab in ((0, 'render'), (1, 'kept')):
                m0 = (bbn == b) & (br[:, Bc['which']] == which) & (br[:, Bc['length']] < LONG_M)
                res['B'][name][lab] = [dict(n_rows=int((m0 & (bband == k)).sum()),
                                            hit_pix=round(float(br[m0 & (bband == k), Bc['n_hit']].sum() /
                                                                max(br[m0 & (bband == k), Bc['n_vis']].sum(), 1)), 3),
                                            lost=round(float((br[m0 & (bband == k), Bc['n_hit']] == 0).mean()), 3)
                                            if (m0 & (bband == k)).any() else np.nan) for k in range(4)]
    # train holes
    f = out / 'holes_train.npz'
    if f.exists():
        h = np.load(f); cand, hole = h['cand'], h['hole']
        res['holes_train'] = {'frames': int(len(h['used'])), 'dup_p50': round(float(np.median(h['dup'])), 4)}
        rates = {}
        for b, (lo, hi) in enumerate(BINS):
            name = f'{lo:g}-{hi:g}'
            ok = cand[:, b] >= 500
            rt = np.where(ok, hole[:, b] / np.maximum(cand[:, b], 1), np.nan)
            rates[b] = rt
            o = np.argsort(np.where(ok, rt, -1))[::-1]
            top5 = o[:5]
            res['holes_train'][name] = dict(
                candidates=int(cand[:, b].sum()), hole_rate=round(float(hole[:, b].sum() / max(cand[:, b].sum(), 1)), 4),
                beams=int(ok.sum()), beam_rate_p10_p50_p90_max=np.round(np.nanpercentile(rt, (10, 50, 90, 100)), 4).tolist(),
                top5=[[int(i), round(float(rt[i]), 4)] for i in top5],
                top5_share_of_holes=round(float(hole[top5, b].sum() / max(hole[:, b].sum(), 1)), 3),
                top5_share_of_candidates=round(float(cand[top5, b].sum() / max(cand[:, b].sum(), 1)), 3))
        res['holes_train']['beam_rate_spearman_40-50_vs_60-75'] = _spear(rates[2], rates[4])
        res['holes_train']['beam_rate_spearman_15-25_vs_50-60'] = _spear(rates[0], rates[3])
    f = out / 'donor_bands.json'
    if f.exists():
        res['donors'] = json.load(open(f))
    json.dump(res, open(out / 'rows.json', 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:20000], flush=True)


def fill_report(a):
    out = Path(a.out)
    B = [np.load(out / f'fill_B_{k}.npz') for k in NUS_KEYS]
    bc = {k: j for j, k in enumerate(B[0]['cols'])}
    b = np.concatenate([x['rows'] for x in B])
    for x, k in zip(B, NUS_KEYS):
        print(f'[fill_report] {k}: {len(x["rows"])} copies of {int(x["n_gate"])} gate rows, mismatches {int(x["n_mismatch"])}')
    cd = np.load(out / 'fill_CD.npz'); cc = {k: j for j, k in enumerate(cd['cols'])}
    w = cd['rows']; miss = cd['miss']

    def q(v):
        v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
        return [round(float(x), 3) for x in np.percentile(v, (25, 50, 75))] if len(v) else [np.nan] * 3
    table = {}
    for lo, hi in BINS:
        name = f'{lo:g}-{hi:g}'
        mb = (b[:, bc['r']] >= lo) & (b[:, bc['r']] < hi) & (b[:, bc['n_vis']] > 0) & (b[:, bc['length']] < LONG_M)
        mw = (w[:, cc['r']] >= lo) & (w[:, cc['r']] < hi) & (w[:, cc['n_vis']] > 0)
        uf = np.where(w[:, cc['n_sil']] > 0, w[:, cc['n_vis']] / np.maximum(w[:, cc['n_sil']], 1), np.nan)
        ub = b[:, bc['n_vis']] / np.maximum(b[:, bc['n_sil']], 1)
        bs = np.maximum(b[:, bc['n_sil']], 1); ws = np.maximum(w[:, cc['n_sil']], 1)
        rows = {      # full = distinct filled visible pixels / ALL silhouette pixels (occlusion included)
            'B render': dict(n=int(mb.sum()), fill=q(b[mb, bc['fill_render']]), rc=q(b[mb, bc['rc_render']]), unocc=q(ub[mb]),
                             full=q(b[mb, bc['nfill_render']] / bs[mb])),
            'B kept': dict(n=int(mb.sum()), fill=q(b[mb, bc['fill_kept']]), rc=q(b[mb, bc['rc_kept']]), unocc=q(ub[mb]),
                           full=q(b[mb, bc['nfill_kept']] / bs[mb])),
        }
        for lab, m in (('C all', mw), ('C unoccluded', mw & (uf >= 0.9)), ('D all', mw & miss),
                       ('D unoccluded', mw & miss & (uf >= 0.9))):
            rows[lab] = dict(n=int(m.sum()), fill=q(w[m, cc['fill']]), rc=q(w[m, cc['rc']]), unocc=q(uf[m]),
                             full=q(w[m, cc['nfill']] / ws[m]))
        br, cu = rows['B render']['fill'][1], rows['C unoccluded']['fill'][1]
        rows['ray_drop_1_minus_Cunocc_over_Brender_medians'] = round(1 - cu / br, 3) if br else np.nan
        fr, fu, fa = rows['B render']['full'][1], rows['C unoccluded']['full'][1], rows['C all']['full'][1]
        rows['full_chain_medians'] = dict(
            ray_drop=round(1 - fu / fr, 3), occlusion_partiality=round(1 - fa / fu, 3),
            sparsifier=round(1 - rows['B kept']['full'][1] / fr, 3), B_kept_over_C_all=round(rows['B kept']['full'][1] / fa, 3),
            share_C_unoccluded=round(rows['C unoccluded']['n'] / max(rows['C all']['n'], 1), 3))
        table[name] = rows
        for lab, v in rows.items():
            print(f'[fill_report] {name} m {lab}: {v}', flush=True)
    json.dump(table, open(out / 'fill.json', 'w'), indent=1)


# ---------------------------------------------------------------------------------------------------------- elev
def bottom_elevation_offset(bottom_z, r, sensor_h):
    """Elevation of a box bottom about a sensor sensor_h above the frame's z = 0, minus the elevation it would have on
    flat ground z = 0 at the same range (degrees). > 0: the box sits higher in the sensor's rows than on flat ground."""
    b, r = np.asarray(bottom_z, np.float64), np.asarray(r, np.float64)
    return np.degrees(np.arctan2(b - sensor_h, r) - np.arctan2(-sensor_h, r))


def _q(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [round(float(x), 2) for x in np.percentile(v, (10, 50, 90))] if len(v) else [np.nan] * 3


def elev(a):
    """Population check of what the panels' line 4 shows (descriptive): where the box bottoms sit in their sensor's
    elevation, against flat ground, per range bin, for ALL RR v2 copies of the gate (bottom = the gate's own row
    `off_plane` + the gate's least-squares plane; RR's virtual TOP at 2.184 m of the pitched nuScenes LIDAR_TOP frame,
    +y = forward) and ALL Waymo val Vehicle GT of column C / D (bottom from the infos, TOP at its extrinsic position);
    and how many of each copy's kept points the grid's z cut removes."""
    from rr_gate import V2_COPY_COLS, V2_FRAME_COLS
    C = {k: j for j, k in enumerate(V2_COPY_COLS)}; F = {k: j for j, k in enumerate(V2_FRAME_COLS)}
    S = {k: j for j, k in enumerate(SC_COLS)}
    out = dict(B={}, C={}, D={})
    cps = []
    for key in NUS_KEYS:
        g = pickle.load(open(GATE_DIR / f'gate_v2_{key}.pkl', 'rb'))
        cp, fr = g['copies'], g['frames_rows']
        row_of = {int(f[F['frame']]): f for f in fr}
        pl = np.array([row_of[int(f)] for f in cp[:, C['frame']]])
        x, y = cp[:, C['r_new']] * np.cos(cp[:, C['phi']]), cp[:, C['r_new']] * np.sin(cp[:, C['phi']])
        zb = cp[:, C['off_plane']] + np.tan(np.radians(pl[:, F['gp_sx']])) * x + \
            np.tan(np.radians(pl[:, F['gp_sy']])) * y + pl[:, F['gp_z0']]
        cps.append(np.c_[cp[:, C['plat']], cp[:, C['r_new']], y, zb, cp[:, C['n_final_g3']], cp[:, C['n_pts_grid']]])
    cp = np.concatenate(cps)
    dB = bottom_elevation_offset(cp[:, 3], cp[:, 1], 2.184)
    for b, (lo, hi) in enumerate(BINS):
        name = f'{lo:g}-{hi:g}'
        m = (cp[:, 1] >= lo) & (cp[:, 1] < hi) & (cp[:, 4] >= 1)
        cut = cp[m, 4] - cp[m, 5]
        out['B'][name] = dict(
            n=int(m.sum()), d_el_p10_50_90=_q(dB[m]), ahead=_q(dB[m & (cp[:, 2] > 0)]), behind=_q(dB[m & (cp[:, 2] < 0)]),
            boston=_q(dB[m & (cp[:, 0] == 0)]), singapore=_q(dB[m & (cp[:, 0] == 1)]),
            bottom_z_p10_50_90=_q(cp[m, 3]), share_copies_cut=round(float(np.mean(cut > 0)), 3),
            share_points_cut=round(float(cut.sum() / max(cp[m, 4].sum(), 1)), 3))
    # Waymo val GT bottoms in pass.pkl order (the frames of 20261018_01 P3)
    p = pickle.load(open(PASS_PKL, 'rb'))
    infos = pickle.load(open(TOOLS.parent / 'data/waymo/waymo_infos_val.pkl', 'rb'))
    by_fid = {}
    for inf in infos:
        fid = inf['frame_id'] if 'frame_id' in inf else \
            inf['point_cloud']['lidar_sequence'] + '_%03d' % inf['point_cloud']['sample_idx']
        by_fid[fid] = inf
    res = pickle.load(open(p['runs'][0][1], 'rb'))
    fids = [r_['frame_id'] for r_ in res][:p['n_frames']]
    del res
    calib = pickle.load(open(TOP_CALIB, 'rb'))
    boxes, tops = [], []
    for fid in fids:
        an = by_fid[fid]['annos']; names = np.asarray(an['name'])
        gv = np.asarray(an['gt_boxes_lidar'], np.float64).reshape(len(names), -1)[:, :7][names == 'Vehicle'] \
            if len(names) else np.zeros((0, 7))
        boxes.append(gv)
        tops.append(np.repeat(np.asarray(calib[by_fid[fid]['point_cloud']['lidar_sequence']]['extrinsic'])[:3, 3][None],
                              len(gv), 0))
    del infos, by_fid
    bx, tp = np.concatenate(boxes), np.concatenate(tops)
    w = np.load(SC_DIR / 'waymo_val.npz'); wc = w['car']
    assert len(bx) == len(wc) and np.allclose(np.hypot(bx[:, 0], bx[:, 1]), wc[:, S['r']], atol=1e-3)
    zb = bx[:, 2] - bx[:, 5] / 2 - tp[:, 2]                    # bottom relative to the TOP's height
    rt = np.hypot(bx[:, 0] - tp[:, 0], bx[:, 1] - tp[:, 1])
    dW = np.degrees(np.arctan2(zb, rt) - np.arctan2(-tp[:, 2], rt))
    okC = (wc[:, S['n_pts']] >= 1) & (wc[:, S['length']] < LONG_M)
    miss = w['oracle_found5'].astype(bool) & ~w['ours27596_found5'].astype(bool)
    for b, (lo, hi) in enumerate(BINS):
        name = f'{lo:g}-{hi:g}'
        m = okC & (wc[:, S['r']] >= lo) & (wc[:, S['r']] < hi)
        out['C'][name] = dict(n=int(m.sum()), d_el_p10_50_90=_q(dW[m]),
                              bottom_z_p10_50_90=_q(bx[m, 2] - bx[m, 5] / 2))
        out['D'][name] = dict(n=int((m & miss).sum()), d_el_p10_50_90=_q(dW[m & miss]),
                              bottom_z_p10_50_90=_q(bx[m & miss, 2] - bx[m & miss, 5] / 2))
    json.dump(out, open(Path(a.out) / 'elev.json', 'w'), indent=1)
    for col in ('B', 'C', 'D'):
        for name, v in out[col].items():
            print(f'[elev] {col} {name} m: {v}', flush=True)


# ---------------------------------------------------------------------------------------------------------- plot
SURF, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#b4b2ab'
C_CTX, C_ROW, C_HDL = '#d9d8d2', '#c3d3ea', '#f0a582'
C_KEY, C_LAG = '#eb6834', ('#1f5fae', '#5b95da', '#a9c8ef')
C_KEPT, C_UNSP = '#2a78d6', '#c2c0b8'
C_TOP, C_OTHER = '#2a78d6', '#1baf7a'
COL_HEAD = {
    'A': 'A. Source: nuScenes train car\n27471 loader, Boston 15 / Singapore 10 sweeps',
    'B': 'B. Source + RR v2\na relocated, re-rendered copy (gate v2 frames)',
    'C': 'C. Target: Waymo val Vehicle\n27596\'s input cloud',
    'D': 'D. Target miss (labelled analysis)\nthe oracle finds it at IoU 0.5, 27596 does not',
}
MARK = {0: 4.0, 1: 7.0, 2: 9.0, 3: 11.0, 4: 13.0}


SENSOR_H = {'A': 1.75, 'B': 2.184, 'C': 2.184, 'D': 2.184}     # sensor height above the frame's z = 0 (flat ground)


def _metrics(e):
    """Row statistics of the in-box points on the model grid against the panel's own lattice, the in-box points the
    grid's z cut removes, and where the box bottom sits in elevation (reported in titles and the CSV)."""
    g = np.asarray(e['ingrid'], bool)
    az, el = view_angles(e['pts'], e['pose'])
    r = float(np.hypot(*e['box'][:2]))
    zb = float(e['box'][2] - e['box'][5] / 2)
    _, elb = view_angles(np.array([[e['box'][0], e['box'][1], zb]]), e['pose'])
    m = dict(n_cut=int((~g).sum()), bottom_z=zb, bottom_el=float(elb[0]),
             flat_el=float(np.degrees(np.arctan2(-SENSOR_H[e['col']], r))))
    if e['col'] == 'A':
        key = (e['lag'] <= 1e-6) & g
        k, res = nearest_row(el, HDL32E_DEG)
        m.update(n_key=int(key.sum()), n_key_distinct=int(len(np.unique(np.round(e['pts'][key], 4), axis=0))),
                 rings_key=int(len(np.unique(k[key]))), rings_all=int(len(np.unique(k[g]))),
                 resid_key=float(np.median(res[key])) if key.any() else np.nan,
                 resid_old=float(np.median(res[~key & g])) if (~key & g).any() else np.nan,
                 n_lags=int(len(np.unique(np.round(e['lag'][g], 4)))))
    elif e['col'] == 'B':
        th = e['thetas_deg']
        _, elu = view_angles(e['unsp'], e['pose'])
        kk, res = nearest_row(el[g], th); ku, _ = nearest_row(elu, th)
        m.update(rows_kept=int(len(np.unique(kk))), rows_unsp=int(len(np.unique(ku))), n_unsp=int(len(e['unsp'])),
                 resid_kept=float(np.median(res)) if len(res) else np.nan)
    else:
        top = (e['lidar'] == 0) & g
        k, res = nearest_row(el, e['incl_deg'])
        m.update(n_top=int(top.sum()), n_other=int(((e['lidar'] != 0) & g).sum()), rows_top=int(len(np.unique(k[top]))),
                 resid_top=float(np.median(res[top])) if top.any() else np.nan)
    return m


def _short_frame(e):
    if e['col'] == 'A':
        mt = re.search(r'__LIDAR_TOP__(\d+)', e['frame_id'])
        return f"info {e['frame']} · ts {int(mt.group(1)) / 1e6:.2f}" if mt else f"info {e['frame']}"
    if e['col'] == 'B':
        return f"{e['frame_id']} · inst {e['instance']}"
    mt = re.match(r'segment-(\d+)_(\d+)_', e['segment'])
    return f"seg {mt.group(1)[:8]}…_{mt.group(2)} · frame {e['sample_idx']}" if mt else e['frame_id']


def _title(e, m):
    q = f"p{e['q']}"
    r = float(np.hypot(*e['box'][:2]))
    n, nv, nl, ppv = e['desc']
    l1 = f"{q} · {e['platform']} · {_short_frame(e)}"
    l2 = f"r {r:.1f} m · α {e['alpha']:+.0f}° · n_pts {int(n)} · n_vox {int(nv)} · n_layers {int(nl)} · ppv {ppv:.2f}"
    if e['col'] == 'A':
        dup = f" ({m['n_key_distinct']} distinct)" if m['n_key'] > m['n_key_distinct'] else ''
        l3 = f"keyframe {m['n_key']} pts{dup} on {m['rings_key']} rings · {m['n_lags']} sweep times"
    elif e['col'] == 'B':
        l3 = f"p {e['p_keep']:.2f} · kept {int(n)} of {m['n_unsp']} rendered · rows {m['rows_kept']} of {m['rows_unsp']}"
    else:
        l3 = f"TOP {m['n_top']} pts on {m['rows_top']} rows · other lidars {m['n_other']}"
    l4 = f"bottom z {m['bottom_z']:+.2f} m, {m['bottom_el']:+.2f}° (flat {m['flat_el']:+.2f}°)"
    if m['n_cut']:
        l4 += f" · {m['n_cut']} cut by z"
    if not e.get('match', True):
        l3 += ' · NOT REPRODUCED'
    return f'{l1}\n{l2}\n{l3}\n{l4}'


def _panel(ax, e, W, H, b):
    pose, box = e['pose'], e['box']
    az0 = view_angles(np.asarray(box[:3])[None], pose)[0][0]
    _, cel = view_angles(box_corners(box), pose)
    el0 = 0.5 * (cel.min() + cel.max())
    lo, hi = el0 - H / 2, el0 + H / 2

    def X(az):
        return -wrap_deg(az - az0)                 # image left = the sensor's left
    rows = e['incl_deg'] if e['col'] in ('C', 'D') else e.get('thetas_deg', None)
    if rows is None:
        rows = e['_train_rows']
    for rr_ in rows:
        if lo <= rr_ <= hi:
            ax.axhline(rr_, color=C_ROW, lw=0.8, zorder=0)
    if e['col'] == 'A':
        for rr_ in HDL32E_DEG:
            if lo <= rr_ <= hi:
                ax.axhline(rr_, color=C_HDL, lw=1.0, ls=(0, (4, 3)), zorder=0)
    s = MARK[b]
    if len(e['ctx']):
        caz, cel_ = view_angles(e['ctx'], pose)
        ax.scatter(X(caz), cel_, s=s * 0.35, c=C_CTX, lw=0, zorder=1, rasterized=True)
    az, el = view_angles(e['pts'], pose)
    g = np.asarray(e['ingrid'], bool)

    def dots(mask, colour, z, scale=1.0):
        """Points on the model grid filled; in-box points the grid's z cut removes as open rings."""
        ax.scatter(X(az[mask & g]), el[mask & g], s=s * scale, c=colour, lw=0, zorder=z, rasterized=True)
        if (mask & ~g).any():
            ax.scatter(X(az[mask & ~g]), el[mask & ~g], s=s * scale * 1.3, facecolors='none', edgecolors=colour,
                       linewidths=0.7, zorder=z, rasterized=True)
    if e['col'] == 'A':
        lag = e['lag']; key = lag <= 1e-6
        cls = np.digitize(lag, LAG_EDGES, right=True)
        for k in (2, 1, 0):
            dots(~key & (cls == k), C_LAG[k], 2)
        dots(key, C_KEY, 3, 1.2)
    elif e['col'] == 'B':
        uaz, uel = view_angles(e['unsp'], pose)
        ax.scatter(X(uaz), uel, s=s * 1.3, c=C_UNSP, lw=0, zorder=2, rasterized=True)
        dots(np.ones(len(az), bool), C_KEPT, 3)
    else:
        top = e['lidar'] == 0
        dots(~top, C_OTHER, 2)
        dots(top, C_TOP, 3)
    for pl in box_edge_polylines(box):
        paz, pel = view_angles(pl, pose)
        ax.plot(X(paz), pel, color=INK2, lw=0.8, alpha=0.85, zorder=4)
    ax.set_xlim(-W / 2, W / 2); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor(SURF)
    ax.tick_params(colors=INK2, labelsize=9, length=2)
    for sp in ax.spines.values():
        sp.set_color(MUTED)


def plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pcdet.datasets.augmentor import relocate_render as rr
    out = Path(a.out)
    ents = []
    for part in ('A', 'B', 'CD'):
        f = out / f'extract_{part}.pkl'
        if f.exists():
            ents += pickle.load(open(f, 'rb'))
    train_rows = np.sort(np.degrees(rr.waymo_top_train_lattice()['thetas']))
    for e in ents:
        e['_train_rows'] = train_rows
        e['metrics'] = _metrics(e)
    FW, AXW = 20.0, 4.3
    rows_csv = []
    for b, (lo_r, hi_r) in enumerate(BINS):
        W, H = sheet_window(lo_r)
        axh = AXW * H / W
        top_in, title_in, xl_in, leg_in = 2.2, 1.05, 0.45, 1.0
        FH = top_in + 3 * (title_in + axh + xl_in) + leg_in
        fig = plt.figure(figsize=(FW, FH), facecolor=SURF)
        fig.text(0.5, 1 - 0.22 / FH, f'{lo_r:g}-{hi_r:g} m: one box per panel, seen from its own sensor '
                 f'(azimuth x elevation about that cloud\'s sensor)', ha='center', va='top', fontsize=17, color=INK,
                 weight='bold')
        fig.text(0.5, 1 - 0.62 / FH,
                 f'Every panel {W:.1f} x {H:.1f} deg, {W / AXW:.2f} deg per inch on both axes. Rows: the boxes nearest the '
                 f'p25 / p50 / p75 of in-box points in a fixed-seed pool. Filled dots: in-box points on 27471\'s grid; '
                 f'open rings: in-box points the grid\'s z cut (-2 to 4 m) removes.\n'
                 f'Faint lines: Waymo TOP rows (A, B: the TRAIN-median lattice RR renders with; C, D: the segment\'s own '
                 f'inclinations). Dashed (A): nuScenes HDL-32E rings. Grey dots: scene within {CTX_M:g} m of the box.\n'
                 f'Title line 4: the box bottom\'s height in its frame and its elevation; "flat" = the elevation it would '
                 f'have on flat ground under the sensor (1.75 m for A, 2.184 m for B-D); "cut by z" = in-box points the '
                 f'grid\'s z cut removes.', ha='center', va='top', fontsize=10.5, color=INK2, linespacing=1.4)
        x0_in, colw_in = 0.75, 4.85
        for ci, col in enumerate(COLUMNS):
            fig.text((x0_in + ci * colw_in + AXW / 2) / FW, 1 - (top_in - 0.12) / FH, COL_HEAD[col], ha='center',
                     va='bottom', fontsize=12, color=INK, weight='bold', linespacing=1.3)
            for qi, q in enumerate(QS):
                ybot = FH - top_in - qi * (title_in + axh + xl_in) - title_in - axh
                ax = fig.add_axes([(x0_in + ci * colw_in) / FW, ybot / FH, AXW / FW, axh / FH])
                ee = [e for e in ents if e['col'] == col and e['bin'] == b and e['q'] == q]
                if not ee:
                    ax.text(0.5, 0.5, 'no box in this bin', ha='center', va='center', transform=ax.transAxes,
                            color=INK2)
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                e = ee[0]
                _panel(ax, e, W, H, b)
                ax.set_title(_title(e, e['metrics']), fontsize=9.6, color=INK, loc='left', pad=4, linespacing=1.25)
                if qi == 2:
                    ax.set_xlabel('azimuth from the box centre (deg; image left = sensor\'s left)', fontsize=9.5,
                                  color=INK2)
                if ci == 0:
                    ax.set_ylabel('elevation (deg)', fontsize=9.5, color=INK2)
                rows_csv.append(_csv_row(e))
        cut = Line2D([], [], ls='', marker='o', markerfacecolor='none', color=INK2, label='cut by the grid\'s z range')
        handles = {
            'A': [Line2D([], [], ls='', marker='o', color=C_KEY, label='keyframe (lag 0)'),
                  Line2D([], [], ls='', marker='o', color=C_LAG[0], label=f'sweeps, lag <= {LAG_EDGES[0]:g} s'),
                  Line2D([], [], ls='', marker='o', color=C_LAG[1], label=f'lag {LAG_EDGES[0]:g}-{LAG_EDGES[1]:g} s'),
                  Line2D([], [], ls='', marker='o', color=C_LAG[2], label=f'lag > {LAG_EDGES[1]:g} s'),
                  Line2D([], [], color=C_HDL, ls=(0, (4, 3)), label='HDL-32E ring'), cut],
            'B': [Line2D([], [], ls='', marker='o', color=C_KEPT, label='kept after the sparsifier (p)'),
                  Line2D([], [], ls='', marker='o', color=C_UNSP, label='rendered, before the sparsifier'),
                  Line2D([], [], color=C_ROW, lw=1.5, label='Waymo TOP row (TRAIN median)'), cut],
            'C': [Line2D([], [], ls='', marker='o', color=C_TOP, label='TOP lidar'),
                  Line2D([], [], ls='', marker='o', color=C_OTHER, label='other four lidars'),
                  Line2D([], [], color=C_ROW, lw=1.5, label='TOP row (segment\'s inclinations)')],
            'D': [Line2D([], [], ls='', marker='o', color=C_TOP, label='TOP lidar'),
                  Line2D([], [], ls='', marker='o', color=C_OTHER, label='other four lidars'),
                  Line2D([], [], color=INK2, label='box outline (projected)')],
        }
        for ci, col in enumerate(COLUMNS):
            lax = fig.add_axes([(x0_in + ci * colw_in) / FW, 0.05 / FH, AXW / FW, (leg_in - 0.2) / FH])
            lax.axis('off')
            lax.legend(handles=handles[col], loc='upper left', ncol=2 if col == 'A' else 1, fontsize=10,
                       frameon=False, labelcolor=INK2, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0)
        name = f'sheet_{lo_r:g}-{hi_r:g}m.png'
        fig.savefig(out / name, dpi=100, facecolor=SURF)
        plt.close(fig)
        print(f'[plot] {out / name}', flush=True)
    keys = []
    for r_ in rows_csv:
        keys += [k for k in r_ if k not in keys]
    with open(out / 'chosen_boxes.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys, restval=''); wr.writeheader(); wr.writerows(rows_csv)
    json.dump([dict(col=r['column'], bin=r['bin'], q=r['quantile'], **{k: r[k] for k in r if k.startswith('m_')})
               for r in rows_csv], open(out / 'metrics.json', 'w'), indent=1)
    print(f'[plot] {len(rows_csv)} panels; CSV {out / "chosen_boxes.csv"}', flush=True)


def _csv_row(e):
    r = float(np.hypot(*e['box'][:2]))
    row = dict(bin=f'{BINS[e["bin"]][0]:g}-{BINS[e["bin"]][1]:g}', column=e['col'], quantile=e['q'],
               pool_n_pts_quantile=round(e['q_value'], 2), pool_n=e['pool_n'], eligible_n=e['elig_n'],
               platform=e['platform'], frame=e['frame_id'], box_index=e['j'], r_m=round(r, 2),
               alpha_deg=round(e['alpha'], 1), n_pts=int(e['desc'][0]), n_vox=int(e['desc'][1]),
               n_layers=int(e['desc'][2]), ppv=round(float(e['desc'][3]), 3), length_m=round(float(e['box'][3]), 2),
               height_m=round(float(e['box'][5]), 2), reproduced=e.get('match', ''),
               p_keep=round(e['p_keep'], 3) if e['col'] == 'B' else '',
               instance=e.get('instance', ''), n_final_in_cloud=e.get('n_final', ''),
               sensor=('nuScenes LIDAR_TOP at SHIFT_COOR' if e['col'] == 'A' else
                       'virtual TOP (0, 0, 2.184)' if e['col'] == 'B' else 'Waymo TOP extrinsic'),
               sensor_origin=' '.join(f'{v:.3f}' for v in e['pose']['origin']))
    row.update({f'm_{k}': (round(v, 4) if isinstance(v, float) else v) for k, v in e['metrics'].items()})
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['select', 'extract_a', 'extract_b', 'extract_cd', 'elev', 'plot', 'fill_b', 'fill_cd',
                                     'fill_report', 'rows_cd', 'holes_train', 'donor_bands', 'rows_report'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--key', default='NUSCENES_N008'); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--rows', action='store_true', help='fill_b: write the §8 per-row records instead of the §7 table')
    ap.add_argument('--n_frames', type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    {'select': select, 'extract_a': extract_a, 'extract_b': extract_b, 'extract_cd': extract_cd, 'elev': elev,
     'plot': plot, 'fill_b': fill_b, 'fill_cd': fill_cd, 'fill_report': fill_report, 'rows_cd': rows_cd,
     'holes_train': holes_train, 'donor_bands': donor_bands, 'rows_report': rows_report}[a.mode](a)
    print(f'[{a.mode}] done in {time.time() - t0:.0f} s', flush=True)
