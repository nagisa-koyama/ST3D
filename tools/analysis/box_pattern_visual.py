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
        l3 = f"keyframe {m['n_key']} pts on {m['rings_key']} rings · {m['n_lags']} sweep times"
        if m['n_key'] > m['n_key_distinct']:
            l3 += f" · keyframe repeated ({m['n_key_distinct']} distinct)"
    elif e['col'] == 'B':
        l3 = f"p {e['p_keep']:.2f} · kept {int(n)} of {m['n_unsp']} rendered · rows {m['rows_kept']} of {m['rows_unsp']}"
    else:
        l3 = f"TOP {m['n_top']} pts on {m['rows_top']} rows · other lidars {m['n_other']}"
    l4 = f"bottom z {m['bottom_z']:+.2f} m at {m['bottom_el']:+.2f}° (flat ground {m['flat_el']:+.2f}°)"
    if m['n_cut']:
        l4 += f" · {m['n_cut']} cut by the grid's z range"
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
        top_in, title_in, xl_in, leg_in = 1.95, 1.05, 0.45, 1.0
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
                 f'inclinations). Dashed (A): nuScenes HDL-32E rings. Grey dots: scene within {CTX_M:g} m of the box.',
                 ha='center', va='top', fontsize=10.5, color=INK2, linespacing=1.4)
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
    ap.add_argument('mode', choices=['select', 'extract_a', 'extract_b', 'extract_cd', 'plot'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time()
    {'select': select, 'extract_a': extract_a, 'extract_b': extract_b, 'extract_cd': extract_cd, 'plot': plot}[a.mode](a)
    print(f'[{a.mode}] done in {time.time() - t0:.0f} s', flush=True)
