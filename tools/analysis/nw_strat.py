"""NW-STRAT: occlusion stratification of the nuScenes -> Waymo residual, and the S1 pitch evaluations (ANALYSIS).

experiments_md 20261027_03; design FIXED in 20261003_05 §5 (item 23, "the nuScenes -> Waymo residual", S0 and S1) at
2026-10-27 21:15 JST (experiments_md b6656df), before any number. It reads Waymo val labels to diagnose; nothing here
selects a method or feeds one (memory/repo/uda_legality_rule.md).

S0 inputs
- 20261017_13's pass.pkl (job 29382): per Vehicle GT the stored points, range and length, and the greedy matching of the
  oracle 27179, the teacher 27471 (scored as 27596) and 27704 (rescored as 27803);
- every model's result.pkl, the replication 28563's in-run epoch-20 scoring included;
- fill_CD_all.npz from `box_pattern_visual.py fill_cd --all_vehicle`: per Vehicle GT the unoccluded fraction
  u = visible / full silhouette (TOP first returns; a pixel without a return counts as visible).

Modes
- `pass`: per frame and model, 20261017_13's `frame_detections` (unchanged, no points): the greedy matching over all
  Vehicle GT at IoU 0.7 / 0.5; plus a re-matching inside the 50 m cap (GT and detections whose BEV centre range is
  < 50 m, the rest dropped). It must reproduce pass.pkl's matching exactly for the three models pass.pkl holds.
- `report`: checks first (27704's greedy AP 55.02; 20261018_03 §7.2's medians from fill_CD_all.npz; 20261007_03 §3's
  subset APs as the convention check); then M1 (carried AP per category and jointly), the 50 m capped evaluation, M2
  and the readings by the fixed bars, with the official KITTI evaluator re-checking every pooled M1 cell and the cap.
- `pitch_check`: S1's pre-GPU check through the real test loader (27704's config and its three twins): the hook's
  output equals `osens2_utils.pitch_points` of its input, GT boxes and frames are the base's, and the ground slope of
  the pitched cloud, fitted by the hook's own estimator, is Waymo's plus the arm's angle (the sign check).
- `pitch_collect`: S1's table (Car AP from each evaluation's log, W&B run by eval tag) and its reading.

M1 (20261007_03 §3 subset convention, greedy matching over ALL GT, BEV AP_R40 at IoU 0.7): with the GT of category X
set to don't-care, a detection matched to an X GT is dropped, an unmatched detection stays a false positive, and the GT
count excludes X. gap(X) = AP_oracle(X) - AP_model(X); C_X = gap(all) - gap(X).
M2: in each range x stored-points cell with >= 100 GT in each compared group, the rate at which the oracle finds a GT
that the model misses (IoU 0.5, any score: greedy matching over all detections) for u < 0.5 against u >= 0.9.

    python analysis/nw_strat.py pass --out <dir> oracle=<pkl> teacher=<pkl> m27704=<pkl> m28563=<pkl> [plgt=<pkl>]
    python analysis/nw_strat.py report --out <dir> [--fill_all <npz>] [--rings] [--repro_json <old nw_strat.json>]
    python analysis/nw_strat.py pitch_check --out <dir>
    python analysis/nw_strat.py pitch_collect --out <dir>

Opt-in report options (absent by default; added for the PL-GT beside items, experiments_md 20261027_05):
- `--rings`: 20261017_13 §5.2's per-ring AP (0-10 ... 50-75 m) from the same pooled matching (a TP binned by its GT's
  centre range, an FP by its own, n_GT by GT range), AP / PR / RL at IoU 0.7 and 0.5 per model and the gap to the
  oracle; it must reproduce 20261017_13's numbers.json for the oracle, the teacher and 27704, or the job stops.
- `--repro_json`: every number a previous nw_strat.json holds for the models it shares with this one (checks, u,
  categories, M1 greedy and official, M2, readings) must be equal, or the job exits 3 (it still writes its outputs,
  flagged in `repro_vs_previous`).
"""
import argparse
import copy
import glob
import json
import pickle
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS.parent, TOOLS, TOOLS / 'analysis'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _init_path  # noqa: F401,E402  (the checkout's pcdet, not the image's copy)

PASS_PKL = Path('/home/koyama/data/waymo_ranking_20261017/full/pass.pkl')
BPV_DIR = Path('/home/koyama/data/box_pattern_visual_20261018')            # 20261018_03 §7 outputs (fill_CD.npz, fill.json)
SC_DIR = Path('/home/koyama/data/source_coverage_20261018/full')            # waymo_val.npz (the §7 population and misses)
PASS_LABEL = {'oracle': 'oracle', 'teacher': 'ours27596', 'm27704': 'pl27704'}   # our labels -> pass.pkl's
CAP_M = 50.0
# the fixed partitions (design item 23, S0)
RANGE_BINS = ((0.0, 30.0), (30.0, 50.0), (50.0, 75.0))
PTS_BINS = ((0, 1), (1, 10), (10, 50), (50, 200), (200, 10 ** 9))
PTS_NAMES = ('0', '1-9', '10-49', '50-199', '>=200')
U_OCCL, U_VIS = 0.5, 0.9
M2_MIN = 100
# the fixed bars
BAR_C_OCCL, BAR_RATIO_FACTOR, BAR_RATIO_PROXY, BAR_DOMINANT, BAR_RANGE = 2.0, 1.5, 1.2, 0.5, 2.0
REPRO = {'m27704_greedy_ap': 55.02, 'oracle_greedy_ap': 65.21}
# 20261007_03 §3 (AP on GT subsets, IoU 0.7), the convention check: oracle / GBlobs + accumulation (= teacher 27596)
REPRO_SUBSETS = {'pts>=200': (94.1, 81.7), 'pts 50-199': (81.1, 60.1), 'pts 10-49': (38.3, 13.2), 'len<6': (65.3, 52.7),
                 'len<6 & pts>=200': (96.6, 87.4), 'len>=6': (44.6, 2.1)}
RANKING_NUMBERS = Path('/home/koyama/data/waymo_ranking_20261017/full/numbers.json')   # 20261017_13 §5 (job 29382)


# ------------------------------------------------------------------------------------------- per ring (--rings)
def ring_table(P):
    """20261017_13 §5.2's per-ring AP from the same pooled matching: a TP is binned by its GT's centre range, an FP by
    its own, n_GT by GT range (rings 0-10 ... 50-75 m; beyond 75 m in no ring). AP / PR / RL (AP_R40 and continuous) per
    model at IoU 0.7 and 0.5; gap = oracle AP - model AP per ring. Checked against 20261017_13's numbers.json."""
    from waymo_ranking_loss_fp_taxonomy import RINGS, RING_NAMES, ap_split
    gt = P['gt']
    edges = np.asarray(RINGS[1:-1], np.float64)
    gt_ring = np.where(gt['rng'] < RINGS[-1], np.searchsorted(edges, gt['rng'], side='right'), -1)
    nk = len(RINGS) - 1
    out = dict(rings=list(RING_NAMES), n_gt=[int((gt_ring == k).sum()) for k in range(nk)], models={})
    for lab in P['labels']:
        det = P['models'][lab]['det']
        d_ring = np.where(det['rng'] < RINGS[-1], np.searchsorted(edges, det['rng'], side='right'), -1)
        r = {}
        for thr, key in ((0.7, 'gt7'), (0.5, 'gt5')):
            tp = det[key] >= 0
            tp_ring = np.where(tp, gt_ring[np.maximum(det[key], 0)], -1)
            r[f'ring{thr}'] = []
            for k in range(nk):
                sel = (tp & (tp_ring == k)) | (~tp & (d_ring == k))
                r[f'ring{thr}'].append(ap_split(det['score'][sel], tp[sel], int((gt_ring == k).sum())))
        out['models'][lab] = r
    for lab in P['labels']:
        if lab != 'oracle':
            for thr in (0.7, 0.5):
                out['models'][lab][f'gap{thr}'] = [o['ap'] - m['ap'] for o, m in
                                                   zip(out['models']['oracle'][f'ring{thr}'], out['models'][lab][f'ring{thr}'])]
    ref = json.load(open(RANKING_NUMBERS))
    chk = {}
    for lab, rlab in PASS_LABEL.items():
        if lab not in out['models']:
            continue
        for thr in (0.7, 0.5):
            got, want = out['models'][lab][f'ring{thr}'], ref[rlab][f'ring{thr}']
            chk[f'{lab}_ring{thr}'] = bool(len(got) == len(want) and all(
                abs(g[k] - w[k]) <= 1e-9 for g, w in zip(got, want) for k in ('ap', 'pr', 'rl', 'ap_c', 'pr_c', 'rl_c', 'recall')))
    out['checks_vs_20261017_13'] = chk
    out['reproduces'] = bool(chk) and all(chk.values())
    return out


# --------------------------------------------------------------------------- reproduction of a previous report
def compare_previous(old, new, labels):
    """Paths at which `old` (a previous nw_strat.json) and `new` differ, over the models in `labels` (shared by both) and
    the model-free parts. Exact comparison (both sides through JSON); NaN equals NaN."""
    mism = []

    def walk(a, b, path):
        if isinstance(a, dict):
            if not isinstance(b, dict):
                mism.append(f'{path}: type'); return
            for k in a:
                if k not in b:
                    mism.append(f'{path}/{k}: missing'); continue
                walk(a[k], b[k], f'{path}/{k}')
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                mism.append(f'{path}: length'); return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f'{path}[{i}]')
        elif isinstance(a, float) or isinstance(b, float):
            if not (a == b or (a != a and b != b)):
                mism.append(f'{path}: {a} != {b}')
        elif a != b:
            mism.append(f'{path}: {a} != {b}')
    for k in ('n_frames', 'n_gt', 'checks', 'checks_failed', 'u', 'category_counts', 'both_seeds_same_side'):
        if k in old:
            walk(old[k], new.get(k), k)
    for k in ('m1_greedy', 'm1_official', 'm2', 'readings', 'official_raw'):
        for lab in labels:
            if k in old and lab in old[k]:
                walk(old[k][lab], new.get(k, {}).get(lab), f'{k}/{lab}')
    return mism


# ------------------------------------------------------------------------------------------------- AP conventions
def dc_ap(score, gt_idx, dc, n_gt_total=None):
    """AP_R40 (percent) under 20261007_03 §3's subset convention. score / gt_idx per detection (gt_idx = matched GT
    index over ALL GT, -1 if unmatched); dc = bool per GT, True = don't-care. A detection matched to a don't-care GT is
    dropped; an unmatched one stays FP; the GT count excludes don't-care."""
    from waymo_ranking_loss_fp_taxonomy import ap_stats
    dc = np.asarray(dc, bool)
    gt_idx = np.asarray(gt_idx, np.int64)
    matched = gt_idx >= 0
    drop = matched & dc[np.maximum(gt_idx, 0)]
    keep = ~drop
    tp = matched[keep]
    return ap_stats(np.asarray(score, np.float64)[keep], tp, int((~dc).sum()))[0]


def categories(gt, u):
    """The fixed categories (bool per GT). u is NaN where the silhouette is empty: such a GT is in no u category."""
    rng, npts, length = gt['rng'], gt['npts'], gt['length']
    with np.errstate(invalid='ignore'):
        occl = np.isfinite(u) & (u < U_OCCL)
        occl09 = np.isfinite(u) & (u < U_VIS)
    cat = {'far >= 50 m': rng >= 50.0, 'sparse < 50 points': npts < 50, 'long >= 6 m': length >= 6.0,
           'occluded u < 0.5': occl}
    cat['joint'] = cat['far >= 50 m'] | cat['sparse < 50 points'] | cat['long >= 6 m'] | occl
    cat['occluded u < 0.9 (beside)'] = occl09
    cat['joint with u < 0.9 (beside)'] = cat['far >= 50 m'] | cat['sparse < 50 points'] | cat['long >= 6 m'] | occl09
    return cat


def m2_cells(gt, u, oracle_found5, model_found5):
    """Per range x points cell: n and oracle-only-miss rate for u < 0.5, 0.5 <= u < 0.9, u >= 0.9; eligibility."""
    rng, npts = gt['rng'], gt['npts']
    miss = np.asarray(oracle_found5, bool) & ~np.asarray(model_found5, bool)
    fin = np.isfinite(u)
    with np.errstate(invalid='ignore'):
        groups = {'occl': fin & (u < U_OCCL), 'mid': fin & (u >= U_OCCL) & (u < U_VIS), 'vis': fin & (u >= U_VIS)}
    cells = []
    for (rlo, rhi) in RANGE_BINS:
        for (plo, phi), pname in zip(PTS_BINS, PTS_NAMES):
            m = (rng >= rlo) & (rng < rhi) & (npts >= plo) & (npts < phi)
            c = dict(range='%g-%g' % (rlo, rhi), points=pname, n_cell=int(m.sum()), n_nan_u=int((m & ~fin).sum()))
            for g, gm in groups.items():
                s = m & gm
                c['n_' + g] = int(s.sum())
                c['miss_' + g] = int((s & miss).sum())
                c['rate_' + g] = float((s & miss).sum() / s.sum()) if s.sum() else float('nan')
            c['eligible'] = bool(c['n_occl'] >= M2_MIN and c['n_vis'] >= M2_MIN)
            c['ratio'] = (c['rate_occl'] / c['rate_vis']) if (c['eligible'] and c['rate_vis'] > 0) else \
                (float('inf') if c['eligible'] and c['rate_occl'] > 0 else float('nan'))
            cells.append(c)
    return cells


def m2_aggregate(cells):
    """Two readings of 'the GT-weighted M2 ratio' over the eligible cells, weights w = GT of the two compared groups:
    (a) the weighted mean of the per-cell ratios; (b) the ratio of the weighted mean rates. Also with weights = all GT
    of the cell. Reported side by side; a reading needs every form on the same side of a bar."""
    el = [c for c in cells if c['eligible']]
    out = {'n_eligible': len(el), 'cells_eligible': ['%s m / %s' % (c['range'], c['points']) for c in el]}
    for wname, wf in (('w_groups', lambda c: c['n_occl'] + c['n_vis']), ('w_cell', lambda c: c['n_cell'])):
        w = np.array([wf(c) for c in el], np.float64)
        r = np.array([c['ratio'] for c in el], np.float64)
        ro = np.array([c['rate_occl'] for c in el], np.float64)
        rv = np.array([c['rate_vis'] for c in el], np.float64)
        ok = np.isfinite(r)
        out[wname] = dict(
            mean_of_ratios=float((w[ok] * r[ok]).sum() / w[ok].sum()) if ok.any() else float('nan'),
            n_cells_ratio_undefined=int((~ok).sum()),
            ratio_of_rates=float((w * ro).sum() / (w * rv).sum()) if len(el) and (w * rv).sum() > 0 else float('nan'))
    return out


def m2_forms(agg):
    return [agg[w][k] for w in ('w_groups', 'w_cell') for k in ('mean_of_ratios', 'ratio_of_rates')]


def readings(gap_all, c_occl, ratio_forms, capped_gap):
    """The fixed readings for one model. ratio_forms: every form of the M2 ratio (a reading needs all on one side)."""
    rf = np.asarray(ratio_forms, np.float64)
    ratio_ok = np.isfinite(rf).all() and len(rf) > 0
    hi = bool(ratio_ok and (rf >= BAR_RATIO_FACTOR).all())
    lo = bool(ratio_ok and (rf <= BAR_RATIO_PROXY).all())
    split = bool(ratio_ok and not hi and not lo and ((rf >= BAR_RATIO_FACTOR).any() or (rf <= BAR_RATIO_PROXY).any()))
    return dict(
        c_occl_ge_2=bool(c_occl >= BAR_C_OCCL),
        ratio_ge_1p5=hi, ratio_le_1p2=lo, ratio_forms_split=split,
        factor=bool(c_occl >= BAR_C_OCCL and hi),
        proxy=lo,
        dominant=bool(c_occl >= BAR_DOMINANT * gap_all),
        range_carries_2=bool(capped_gap <= gap_all - BAR_RANGE))


# --------------------------------------------------------------------------------------------------------- the pass
FRAMES, MODELS = None, None


def _pass_frame(fi):
    from waymo_ranking_loss_fp_taxonomy import frame_detections, bev_iou, greedy_match
    f = FRAMES[fi]
    gv = f['gv']
    rg = np.hypot(gv[:, 0], gv[:, 1])
    gm = np.nonzero(rg < CAP_M)[0]
    out = []
    for m in MODELS:
        db, ds = m[fi]
        o = frame_detections(db, ds, gv, f['go'], None)
        rd = np.hypot(db[:, 0], db[:, 1])
        dm = np.nonzero(rd < CAP_M)[0]
        cap7 = -np.ones(len(ds), np.int64); cap5 = -np.ones(len(ds), np.int64)
        if len(dm) and len(gm):
            iou = bev_iou(db[dm], gv[gm])
            for thr, arr in ((0.7, cap7), (0.5, cap5)):
                dt_gt = greedy_match(iou, ds[dm], thr)[0]
                arr[dm] = np.where(dt_gt >= 0, gm[np.maximum(dt_gt, 0)], -1)
        out.append(dict(gt7=o['dt_gt7'], gt5=o['dt_gt5'], gt_dt7=o['gt_dt7'], gt_dt5=o['gt_dt5'], cap7=cap7, cap5=cap5,
                        cap_in=rd < CAP_M))
    return fi, out


def run_pass(a):
    global FRAMES, MODELS
    from waymo_ranking_loss_fp_taxonomy import load_inputs
    runs = [tuple(r.split('=', 1)) for r in a.runs]
    assert runs[0][0] == 'oracle', 'the first model must be the oracle (pass.pkl frame order)'
    t0 = time.time()
    FRAMES, MODELS, names_seen = load_inputs(runs)
    print(f'[pass] loaded {len(FRAMES)} frames, {len(MODELS)} models in {time.time() - t0:.0f} s; GT names {names_seen}',
          flush=True)
    res = [None] * len(FRAMES)
    with Pool(a.workers) as pool:
        for k, (fi, o) in enumerate(pool.imap_unordered(_pass_frame, range(len(FRAMES)), chunksize=8)):
            res[fi] = o
            if k % 1000 == 0:
                print(f'[pass] {k}/{len(FRAMES)} frames, {time.time() - t0:.0f} s', flush=True)
    off = np.concatenate([[0], np.cumsum([len(f['gv']) for f in FRAMES])])
    gt = dict(frame=np.concatenate([np.full(len(f['gv']), i) for i, f in enumerate(FRAMES)]),
              rng=np.concatenate([np.hypot(f['gv'][:, 0], f['gv'][:, 1]) for f in FRAMES]),
              npts=np.concatenate([f['gnpts'] for f in FRAMES]),
              length=np.concatenate([f['gv'][:, 3] for f in FRAMES]))
    models = {}
    for mi, (label, _) in enumerate(runs):
        d = {k: [] for k in ('frame', 'score', 'rng', 'gt7', 'gt5', 'cap7', 'cap5', 'cap_in')}
        g = {'found7': [], 'found5': []}
        for fi, f in enumerate(FRAMES):
            db, ds = MODELS[mi][fi]
            o = res[fi][mi]
            d['frame'].append(np.full(len(ds), fi)); d['score'].append(ds); d['rng'].append(np.hypot(db[:, 0], db[:, 1]))
            for k in ('gt7', 'gt5', 'cap7', 'cap5'):
                d[k].append(np.where(o[k] >= 0, o[k] + off[fi], -1))
            d['cap_in'].append(o['cap_in'])
            g['found7'].append(o['gt_dt7'] >= 0); g['found5'].append(o['gt_dt5'] >= 0)
        models[label] = dict(det={k: np.concatenate(v) for k, v in d.items()}, gt={k: np.concatenate(v) for k, v in g.items()})
    out = dict(labels=[r[0] for r in runs], runs=runs, n_frames=len(FRAMES), gt=gt, models=models,
               n_gt_cap=int((gt['rng'] < CAP_M).sum()))
    # the matching must reproduce pass.pkl's for the models it holds
    p = pickle.load(open(PASS_PKL, 'rb'))
    checks = {}
    for k in ('frame', 'npts'):
        checks['gt_' + k] = bool(np.array_equal(p['gt'][k], gt[k]))
    for k in ('rng', 'length'):
        checks['gt_' + k] = bool(np.allclose(p['gt'][k], gt[k], atol=1e-4))
    for lab, plab in PASS_LABEL.items():
        if lab not in models:
            continue
        pd, md = p['models'][plab]['det'], models[lab]['det']
        checks[lab] = bool(all(np.array_equal(pd[k], md[k]) for k in ('frame', 'gt7', 'gt5')) and
                           np.allclose(pd['score'], md['score']))
        checks[lab + '_found5'] = bool(np.array_equal(p['models'][plab]['gt']['det5_score'] >= 0, models[lab]['gt']['found5']))
    out['checks_vs_pass_pkl'] = checks
    print(f'[pass] matching against pass.pkl: {checks}', flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(Path(a.out) / 'nw_pass.pkl', 'wb'))
    print(f'[pass] done in {time.time() - t0:.0f} s -> {Path(a.out) / "nw_pass.pkl"}', flush=True)
    if not all(checks.values()):
        print('[pass] CHECK FAILED: the matching does not reproduce pass.pkl', flush=True)
        sys.exit(3)


# ---------------------------------------------------------------------------------------------------- official
def official_ap(frames, dets, dc=None, cap=False, official=None):
    """Official KITTI evaluator (CPU IoU kernel, one partition per frame), Car BEV AP_R40 at IoU 0.7 and 0.5. GT with
    dc True are renamed 'Van' (ignored for Car: a detection assigned to them is dropped). cap: GT and detections with
    BEV centre range >= 50 m removed."""
    from kitti_eval_cpu import load_official_eval
    official = official or load_official_eval()
    from pcdet.datasets.kitti import kitti_utils
    mp = {'Vehicle': 'Car', 'Van': 'Van', 'Car': 'Car'}
    gts, dts, off = [], [], 0
    for f, (b, s) in zip(frames, dets):
        gv = f['gv'].astype(np.float64)
        names = np.array(['Vehicle'] * len(gv), dtype='<U7')
        if dc is not None:
            names[np.asarray(dc[off:off + len(gv)], bool)] = 'Van'
        off += len(gv)
        b = b.astype(np.float64); s = s.astype(np.float64)
        if cap:
            kg = np.hypot(gv[:, 0], gv[:, 1]) < CAP_M; kd = np.hypot(b[:, 0], b[:, 1]) < CAP_M
            gv, names, b, s = gv[kg], names[kg], b[kd], s[kd]
        gts.append(dict(name=names, gt_boxes_lidar=gv))
        dts.append(dict(name=np.array(['Car'] * len(s), dtype='<U7'), boxes_lidar=b, score=s))
    kitti_utils.transform_annotations_to_kitti_format(gts, map_name_to_kitti=mp, info_with_fakelidar=False)
    kitti_utils.transform_annotations_to_kitti_format(dts, map_name_to_kitti=mp)
    min_overlaps = np.array([[[0.7]] * 3, [[0.5]] * 3])
    ret = official.eval_class(gts, dts, [0], [0], 1, min_overlaps, num_parts=len(gts))
    ap = official.get_mAP_R40(ret['precision'])
    return float(ap[0, 0, 0]), float(ap[0, 0, 1])


# ------------------------------------------------------------------------------------------------------- checks
def _q3(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [round(float(x), 3) for x in np.percentile(v, (25, 50, 75))] if len(v) else [float('nan')] * 3


def fill_checks(fill_all_path):
    """fill_CD_all.npz covers every GT once; on the §7 population its rows equal fill_CD.npz's; 20261018_03 §7.2's
    C all / D all quartiles (fill, row coverage, unoccluded fraction, full fill) reproduce from it."""
    from box_pattern_visual import BINS
    new = np.load(fill_all_path); old = np.load(BPV_DIR / 'fill_CD.npz')
    cn = {k: j for j, k in enumerate(new['cols'])}
    R, O = new['rows'], old['rows']
    out = {}
    g = R[:, cn['g']].astype(np.int64)
    out['covers_every_gt_once'] = bool(np.array_equal(g, np.arange(len(g))))
    go = O[:, 0].astype(np.int64)
    out['rows_equal_on_s7_population'] = bool(len(go) and np.array_equal(R[go], O, equal_nan=True))
    out['max_abs_diff_on_s7_population'] = float(np.nanmax(np.abs(R[go] - O))) if len(go) else float('nan')
    fj = json.load(open(BPV_DIR / 'fill.json'))
    w = R[go]; miss = old['miss']
    uf = np.where(w[:, cn['n_sil']] > 0, w[:, cn['n_vis']] / np.maximum(w[:, cn['n_sil']], 1), np.nan)
    ws = np.maximum(w[:, cn['n_sil']], 1)
    mism = []
    for lo, hi in BINS:
        name = f'{lo:g}-{hi:g}'
        mw = (w[:, cn['r']] >= lo) & (w[:, cn['r']] < hi) & (w[:, cn['n_vis']] > 0)
        for lab, m in (('C all', mw), ('D all', mw & miss)):
            got = dict(n=int(m.sum()), fill=_q3(w[m, cn['fill']]), rc=_q3(w[m, cn['rc']]), unocc=_q3(uf[m]),
                       full=_q3(w[m, cn['nfill']] / ws[m]))
            ref = fj[name][lab]
            for k in ('n', 'fill', 'rc', 'unocc', 'full'):
                if got[k] != ref[k]:
                    mism.append(dict(bin=name, group=lab, key=k, got=got[k], ref=ref[k]))
    out['s72_quartiles_mismatches'] = mism
    out['s72_quartiles_reproduce'] = not mism
    return out


# ------------------------------------------------------------------------------------------------------- report
def run_report(a):
    from waymo_ranking_loss_fp_taxonomy import load_inputs
    out = Path(a.out)
    P = pickle.load(open(out / 'nw_pass.pkl', 'rb'))
    gt = P['gt']; n_gt = len(gt['rng'])
    res = dict(design='20261003_05 §5 item 23 S0 (experiments_md b6656df)', n_frames=P['n_frames'], n_gt=n_gt)
    # ---- checks first
    chk = dict(P['checks_vs_pass_pkl'])
    allgt = np.zeros(n_gt, bool)
    det = {lab: P['models'][lab]['det'] for lab in P['labels']}
    ap_all = {lab: dc_ap(det[lab]['score'], det[lab]['gt7'], allgt) for lab in P['labels']}
    chk['m27704_greedy_ap'] = round(ap_all['m27704'], 2)
    chk['m27704_greedy_ap_reproduces'] = bool(round(ap_all['m27704'], 2) == REPRO['m27704_greedy_ap'])
    chk['oracle_greedy_ap'] = round(ap_all['oracle'], 2)
    chk['oracle_greedy_ap_reproduces'] = bool(round(ap_all['oracle'], 2) == REPRO['oracle_greedy_ap'])
    fc = fill_checks(a.fill_all)
    chk.update({'fill_' + k: v for k, v in fc.items()})
    # convention check: 20261007_03 §3's subset APs (oracle, teacher)
    npts, length = gt['npts'], gt['length']
    subsets = {'pts>=200': npts >= 200, 'pts 50-199': (npts >= 50) & (npts < 200), 'pts 10-49': (npts >= 10) & (npts < 50),
               'len<6': length < 6.0, 'len<6 & pts>=200': (length < 6.0) & (npts >= 200), 'len>=6': length >= 6.0}
    sub = {}
    for name, keep in subsets.items():
        got = tuple(round(dc_ap(det[lab]['score'], det[lab]['gt7'], ~keep), 1) for lab in ('oracle', 'teacher'))
        sub[name] = dict(got=got, ref=REPRO_SUBSETS[name], ok=bool(got == REPRO_SUBSETS[name]))
    chk['s3_subset_convention'] = sub
    chk['s3_subset_convention_reproduces'] = bool(all(v['ok'] for v in sub.values()))
    res['checks'] = chk
    hard = ['gt_frame', 'gt_npts', 'gt_rng', 'gt_length', 'oracle', 'teacher', 'm27704', 'm27704_greedy_ap_reproduces',
            'fill_covers_every_gt_once', 'fill_rows_equal_on_s7_population', 'fill_s72_quartiles_reproduce']
    failed = [k for k in hard if not chk.get(k, False)]
    res['checks_failed'] = failed
    json.dump(res, open(out / 'nw_strat_checks.json', 'w'), indent=1, default=float)
    print(f'[report] checks: failed {failed}; convention check {chk["s3_subset_convention_reproduces"]}', flush=True)
    if failed:
        print('[report] STOP: a required check failed; no reading is computed', flush=True)
        sys.exit(3)
    # ---- u per GT
    F = np.load(a.fill_all); cn = {k: j for j, k in enumerate(F['cols'])}
    R = F['rows']
    u = np.where(R[:, cn['n_sil']] > 0, R[:, cn['n_vis']] / np.maximum(R[:, cn['n_sil']], 1), np.nan)
    res['u'] = dict(n_nan=int((~np.isfinite(u)).sum()), quartiles=_q3(u),
                    share_lt_0p5=float(np.mean(np.isfinite(u) & (u < U_OCCL))),
                    share_0p5_0p9=float(np.mean(np.isfinite(u) & (u >= U_OCCL) & (u < U_VIS))),
                    share_ge_0p9=float(np.mean(np.isfinite(u) & (u >= U_VIS))),
                    nan_by_range=[int((~np.isfinite(u) & (gt['rng'] >= lo) & (gt['rng'] < hi)).sum())
                                  for lo, hi in ((0, 5), (5, 30), (30, 75), (75, 1e9))])
    cat = categories(gt, u)
    res['category_counts'] = {k: int(v.sum()) for k, v in cat.items()}
    # ---- M1, greedy
    models = [lab for lab in P['labels'] if lab != 'oracle']
    m1 = {}
    for lab in models:
        r = {}
        gap_all = ap_all['oracle'] - ap_all[lab]
        r['all'] = dict(ap_oracle=ap_all['oracle'], ap_model=ap_all[lab], gap=gap_all)
        for cname, dcm in cat.items():
            ao = dc_ap(det['oracle']['score'], det['oracle']['gt7'], dcm)
            am = dc_ap(det[lab]['score'], det[lab]['gt7'], dcm)
            r[cname] = dict(ap_oracle=ao, ap_model=am, gap=ao - am, carried=gap_all - (ao - am), n_dc=int(dcm.sum()))
        # capped: re-matched inside 50 m
        caps = {}
        for l2 in ('oracle', lab):
            d = det[l2]
            ci = d['cap_in']
            caps[l2] = dc_ap(d['score'][ci], d['cap7'][ci], gt['rng'] >= CAP_M)
        r['capped_50m'] = dict(ap_oracle=caps['oracle'], ap_model=caps[lab], gap=caps['oracle'] - caps[lab],
                               n_gt=int((gt['rng'] < CAP_M).sum()))
        m1[lab] = r
    res['m1_greedy'] = m1
    # ---- official re-check
    if not a.no_official:
        from kitti_eval_cpu import load_official_eval
        official = load_official_eval()
        runs = P['runs']
        frames, dets, _ = load_inputs(runs)
        assert len(frames) == P['n_frames']
        lab_i = {lab: i for i, (lab, _) in enumerate(runs)}
        off_res = {}
        for lab in P['labels']:
            t0 = time.time()
            o = {'all': official_ap(frames, dets[lab_i[lab]], official=official)}
            for cname, dcm in cat.items():
                o[cname] = official_ap(frames, dets[lab_i[lab]], dc=dcm, official=official)
            o['capped_50m'] = official_ap(frames, dets[lab_i[lab]], cap=True, official=official)
            off_res[lab] = o
            print(f'[report] official {lab}: all {o["all"][0]:.4f}, capped {o["capped_50m"][0]:.4f} '
                  f'({time.time() - t0:.0f} s)', flush=True)
        m1o = {}
        for lab in models:
            ga = off_res['oracle']['all'][0] - off_res[lab]['all'][0]
            r = {'all': dict(ap_oracle=off_res['oracle']['all'][0], ap_model=off_res[lab]['all'][0], gap=ga)}
            for cname in list(cat) + ['capped_50m']:
                ao, am = off_res['oracle'][cname][0], off_res[lab][cname][0]
                r[cname] = dict(ap_oracle=ao, ap_model=am, gap=ao - am, carried=ga - (ao - am))
            m1o[lab] = r
        res['m1_official'] = m1o
        res['official_raw'] = off_res
    # ---- M2
    found5 = {lab: P['models'][lab]['gt']['found5'] for lab in P['labels']}
    m2 = {}
    for lab in models:
        cells = m2_cells(gt, u, found5['oracle'], found5[lab])
        m2[lab] = dict(cells=cells, aggregate=m2_aggregate(cells))
    res['m2'] = m2
    # ---- readings
    rd = {}
    for lab in models:
        r = m1[lab]
        agg = m2[lab]['aggregate']
        rd[lab] = readings(r['all']['gap'], r['occluded u < 0.5']['carried'], m2_forms(agg), r['capped_50m']['gap'])
        if 'm1_official' in res:
            ro = res['m1_official'][lab]
            rd[lab]['official'] = readings(ro['all']['gap'], ro['occluded u < 0.5']['carried'], m2_forms(agg),
                                           ro['capped_50m']['gap'])
    res['readings'] = rd
    if 'm27704' in rd and 'm28563' in rd:
        keys = ('c_occl_ge_2', 'ratio_ge_1p5', 'ratio_le_1p2', 'factor', 'proxy', 'dominant', 'range_carries_2')
        res['both_seeds_same_side'] = {k: bool(rd['m27704'][k] == rd['m28563'][k]) for k in keys}
    # ---- opt-in: per ring (20261017_13 §5.2), and the reproduction of a previous report
    fail = []
    if a.rings:
        res['rings'] = ring_table(P)
        print(f'[report] rings vs 20261017_13: {res["rings"]["checks_vs_20261017_13"]}', flush=True)
        if not res['rings']['reproduces']:
            fail.append('rings')
    if a.repro_json:
        old = json.load(open(a.repro_json))
        shared = [lab for lab in P['labels'] if lab in old.get('m1_greedy', {}) or lab == 'oracle']
        mism = compare_previous(old, json.loads(json.dumps(res, default=float)), shared)
        res['repro_vs_previous'] = dict(path=str(a.repro_json), models=shared, n_mismatches=len(mism),
                                        mismatches=mism[:200], reproduces=not mism)
        print(f'[report] reproduction of {a.repro_json} for {shared}: {len(mism)} mismatches', flush=True)
        if mism:
            fail.append('repro_vs_previous')
    json.dump(res, open(out / 'nw_strat.json', 'w'), indent=1, default=float)
    write_md(res, out / 'nw_strat.md')
    print(f'[report] -> {out / "nw_strat.json"}, {out / "nw_strat.md"}', flush=True)
    if fail:
        print(f'[report] STOP: {fail} did not reproduce (outputs written, flagged)', flush=True)
        sys.exit(3)


def write_md(res, path):
    L = []

    def table(head, rows):
        L.append('| ' + ' | '.join(head) + ' |'); L.append('|' + '---|' * len(head))
        L.extend('| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows); L.append('')
    L.append(f'NW-STRAT: {res["n_frames"]} frames, {res["n_gt"]} Vehicle GT. Checks failed: {res["checks_failed"]}')
    L.append(f'u: {res["u"]}'); L.append(f'category counts: {res["category_counts"]}'); L.append('')
    for key, title in (('m1_greedy', 'M1, greedy (20261007_03 §3 convention)'), ('m1_official', 'M1, official evaluator')):
        if key not in res:
            continue
        L.append(f'### {title}: BEV AP_R40 IoU 0.7, oracle / model / gap / carried')
        rows = []
        for lab, r in res[key].items():
            for c, v in r.items():
                rows.append([lab, c, '%.2f' % v['ap_oracle'], '%.2f' % v['ap_model'], '%.2f' % v['gap'],
                             '%.2f' % v['carried'] if 'carried' in v else '-', v.get('n_dc', v.get('n_gt', '-'))])
        table(['model', 'X (don\'t-care)', 'AP oracle', 'AP model', 'gap', 'C_X', 'n'], rows)
    for lab, m in res['m2'].items():
        L.append(f'### M2, {lab}: oracle-only miss rate at IoU 0.5 (any score), n / rate per u group')
        table(['range', 'points', 'n cell', 'u<0.5', '0.5-0.9', 'u>=0.9', 'NaN u', 'eligible', 'ratio'],
              [[c['range'], c['points'], c['n_cell'], '%d / %.3f' % (c['n_occl'], c['rate_occl']),
                '%d / %.3f' % (c['n_mid'], c['rate_mid']), '%d / %.3f' % (c['n_vis'], c['rate_vis']), c['n_nan_u'],
                c['eligible'], '%.2f' % c['ratio']] for c in m['cells']])
        L.append(f'aggregate: {m["aggregate"]}'); L.append('')
    L.append(f'### readings: {json.dumps(res["readings"])}')
    L.append(f'both seeds same side: {res.get("both_seeds_same_side")}')
    if 'rings' in res:
        rg = res['rings']
        for thr in (0.7, 0.5):
            L.append(f'### Per ring, IoU {thr} (20261017_13 §5.2 pooled matching): AP / PR / RL; gap to the oracle')
            rows = [[lab] + ['%.1f / %.1f / %.1f' % (s['ap'], s['pr'], s['rl']) for s in m[f'ring{thr}']]
                    for lab, m in rg['models'].items()]
            rows += [[lab + ' gap'] + ['%.2f' % g for g in m[f'gap{thr}']] for lab, m in rg['models'].items()
                     if f'gap{thr}' in m]
            table(['model'] + rg['rings'], rows + [['GT'] + rg['n_gt']])
        L.append(f'rings vs 20261017_13: {rg["checks_vs_20261017_13"]}')
    if 'repro_vs_previous' in res:
        r = res['repro_vs_previous']
        L.append(f'reproduction of {r["path"]} for {r["models"]}: {r["n_mismatches"]} mismatches {r["mismatches"][:20]}')
    Path(path).write_text('\n'.join(L) + '\n')


# ---------------------------------------------------------------------------------------------------------- S1
PITCH_EXTRA = '20261027_nwpitch'
BASE_CFG = 'cfgs/da-ieee-access/centerpoint-gblobs-accum-legaldepth-st3d-nuscenes2waymo.yaml'      # 27704's own config
TWIN_DIR = 'cfgs/da-ieee-access-analysis/nw_strat'
# forward ground slopes in the loader frame (20261004_02 §2, positive = ground rises ahead), and nuScenes TRAIN frames
SLOPE = {'boston': 2.29, 'singapore': 1.35, 'waymo': 0.11}
TRAIN_FRAMES = {'boston': 15695, 'singapore': 12435}      # 27704's own log: VEHICLE n008 / n015 kept of 28,130


def pitch_arms():
    """{tag: PITCH_DEG}: source slope minus Waymo's, so the pitched Waymo val ground rises ahead as the source's does."""
    mean = (TRAIN_FRAMES['boston'] * SLOPE['boston'] + TRAIN_FRAMES['singapore'] * SLOPE['singapore']) / \
        (TRAIN_FRAMES['boston'] + TRAIN_FRAMES['singapore'])
    return {'p218': round(SLOPE['boston'] - SLOPE['waymo'], 2), 'p124': round(SLOPE['singapore'] - SLOPE['waymo'], 2),
            'p176': round(mean - SLOPE['waymo'], 2)}


def twin_path(tag):
    return Path(TWIN_DIR) / f'centerpoint-gblobs-accum-legaldepth-st3d-nuscenes2waymo-nwpitch-{tag}.yaml'


def _forward_slope_deg(pts):
    from pcdet.datasets import osens2_utils as O
    fit = O.fit_plane(O.lowest_per_cell(np.asarray(pts[:, :3], np.float64)))
    return float('nan') if fit is None else float(np.degrees(np.arctan(fit[0][0])))


def run_pitch_check(a):
    import logging
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.datasets import osens2_utils as O
    import pcdet.datasets.osens2_utils as OU
    log = logging.getLogger('nw_pitch'); log.setLevel(logging.WARNING)

    def ds_of(path):
        cfg = cfg_from_yaml_file(str(path), EasyDict())
        ds, _, _ = build_dataloader(dataset_cfg=copy.deepcopy(cfg.DATA_CONFIG_TAR), class_names=cfg.CLASS_NAMES,
                                    batch_size=1, dist=False, workers=0, logger=log, training=False,
                                    model_ontology=cfg.get('ONTOLOGY', None))
        return cfg, ds
    cap = {}
    orig = OU.perturb

    def wrapped(points, gt_boxes, frame_id, cfg):
        o, s = orig(points, gt_boxes, frame_id, cfg)
        cap['in'], cap['out'], cap['gt'] = np.array(points, copy=True), np.array(o, copy=True), \
            (None if gt_boxes is None else np.array(gt_boxes, copy=True))
        return o, s
    OU.perturb = wrapped
    base_cfg, base = ds_of(BASE_CFG)
    assert base_cfg.DATA_CONFIG_TAR.get('EVAL_OSENS2', None) is None
    idxs = list(range(0, len(base), max(1, len(base) // a.frames)))[:a.frames]
    res = dict(n_val=len(base), frames=idxs, arms={})
    base_items = {}
    for i in [a.index] + idxs:
        cap.clear(); fr = base[i]
        assert not cap, 'the hook ran on the base config'
        base_items[i] = fr
    for tag, deg in pitch_arms().items():
        cfg, ds = ds_of(twin_path(tag))
        e = cfg.DATA_CONFIG_TAR.EVAL_OSENS2
        r = dict(pitch_deg=float(e.PITCH_DEG), origin=list(map(float, e.SENSOR_ORIGIN)), forward=e.FORWARD_AXIS,
                 expected_pitch_deg=deg)
        ok = [abs(float(e.PITCH_DEG) - deg) < 1e-9, list(map(float, e.SENSOR_ORIGIN)) == [1.43, 0.0, 2.184],
              e.FORWARD_AXIS == 'x']
        sl_in, sl_out = [], []
        for i in [a.index] + idxs:
            cap.clear(); fr = ds[i]
            same = np.allclose(cap['out'], O.pitch_points(cap['in'], deg, [1.43, 0.0, 2.184], 'x'))
            ok.append(bool(same))
            ok.append(bool(np.array_equal(fr['gt_boxes'], base_items[i]['gt_boxes'])))
            ok.append(fr['frame_id'] == base_items[i]['frame_id'])
            if i != a.index or not idxs:
                sl_in.append(_forward_slope_deg(cap['in'])); sl_out.append(_forward_slope_deg(cap['out']))
        r['all_checks_pass'] = bool(all(ok))
        r['n_checks'] = len(ok)
        r['forward_slope_before_median'] = float(np.nanmedian(sl_in))
        r['forward_slope_after_median'] = float(np.nanmedian(sl_out))
        r['slope_change_median'] = float(np.nanmedian(np.asarray(sl_out) - np.asarray(sl_in)))
        r['sign_ok'] = bool(abs(r['slope_change_median'] - deg) < 0.05)
        res['arms'][tag] = r
        print(f'[pitch_check] {tag}: {r}', flush=True)
    OU.perturb = orig
    res['all_pass'] = bool(all(v['all_checks_pass'] and v['sign_ok'] for v in res['arms'].values()))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(Path(a.out) / 'pitch_check.json', 'w'), indent=1)
    print(f'[pitch_check] all pass: {res["all_pass"]}', flush=True)
    if not res['all_pass']:
        sys.exit(3)


def run_pitch_collect(a):
    from osens2_collect import ap_block
    rows = {}
    for d in glob.glob(str(TOOLS.parent / 'output' / '**' / PITCH_EXTRA / 'eval' / 'epoch_20' / 'val' / 'nwpitch_*'),
                       recursive=True):
        logs = sorted(glob.glob(str(Path(d) / 'log_eval_*.txt')))
        if not logs:
            continue
        text = Path(logs[-1]).read_text()
        rows[Path(d).name] = dict(dir=d, bev_3d_07=ap_block(text, '0.70'), bev_3d_05=ap_block(text, '0.50'))
    wb = {}
    for f in glob.glob(str(TOOLS / 'wandb' / 'run-*' / 'files' / 'wandb-metadata.json')):
        try:
            args = json.load(open(f)).get('args', [])
        except Exception:
            continue
        if '--extra_tag' in args and args[args.index('--extra_tag') + 1] == PITCH_EXTRA and '--eval_tag' in args:
            wb.setdefault(args[args.index('--eval_tag') + 1], []).append(Path(f).parents[1].name.split('-')[-1])
    base = rows.get('nwpitch_asscored')
    out = dict(arms=pitch_arms(), rows={})
    for tag, r in sorted(rows.items()):
        b7 = r['bev_3d_07']
        r['bev'], r['3d'] = (b7[0], b7[3]) if b7 else (None, None)
        r['bev05'] = r['bev_3d_05'][0] if r['bev_3d_05'] else None
        r['d_bev'] = (r['bev'] - base['bev_3d_07'][0]) if (base and base['bev_3d_07'] and b7) else None
        r['wandb'] = wb.get(tag, [])
        out['rows'][tag] = r
    ds = [r['d_bev'] for t, r in out['rows'].items() if t != 'nwpitch_asscored' and r['d_bev'] is not None]
    out['pitch_term'] = bool(any(d >= 2.0 for d in ds)) if len(ds) == 3 else None
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.out) / 'pitch.json', 'w'), indent=1)
    for t, r in out['rows'].items():
        print(f'[pitch_collect] {t}: BEV {r["bev"]} 3D {r["3d"]} dBEV {r["d_bev"]} IoU0.5 BEV {r["bev05"]} W&B {r["wandb"]}')
    print(f'[pitch_collect] pitch term (any arm >= +2 BEV): {out["pitch_term"]}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['pass', 'report', 'pitch_check', 'pitch_collect'])
    ap.add_argument('runs', nargs='*', help='pass: <label>=<result.pkl>, oracle first')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--fill_all', default=None, help='report: fill_CD_all.npz (default <out>/fill_CD_all.npz)')
    ap.add_argument('--no_official', action='store_true')
    ap.add_argument('--rings', action='store_true', help='report: 20261017_13 §5.2 per-ring AP and gaps (opt-in)')
    ap.add_argument('--repro_json', default=None, help='report: a previous nw_strat.json that must be reproduced '
                                                        'for the models both hold (opt-in)')
    ap.add_argument('--frames', type=int, default=30, help='pitch_check: strided val frames for the slope statistics')
    ap.add_argument('--index', type=int, default=100, help='pitch_check: the dumped frame')
    a = ap.parse_intermixed_args()      # the runs may follow the options (30893 died on `pass --out X oracle=...`)
    if a.fill_all is None:
        a.fill_all = str(Path(a.out) / 'fill_CD_all.npz')
    t0 = time.time()
    {'pass': run_pass, 'report': run_report, 'pitch_check': run_pitch_check, 'pitch_collect': run_pitch_collect}[a.mode](a)
    print(f'[{a.mode}] done in {time.time() - t0:.0f} s', flush=True)
