"""Waymo residual: AP split into a recall bound and a ranking loss, and a taxonomy of false positives (CPU, ANALYSIS).

experiments_md 20261017_13 holds the definitions (§2) and the readings pre-declared before this ran (§3). Reads
Waymo-target eval `result.pkl` files, `waymo_infos_val.pkl` and the val clouds; no GPU, no retraining. ANALYSIS ONLY:
it reads Waymo val labels to diagnose, never to choose a setting (memory/repo/uda_legality_rule.md).

Protocol (20261007_03): Car BEV, Vehicle -> Car, "all objects" (no difficulty or range filter on GT). Per frame the Car
detections are taken in descending score, and each takes the unmatched Vehicle GT of highest BEV IoU above the
threshold (greedy). That fixes every detection as TP or FP, which the official evaluator does not (it re-matches at
every score threshold). 40-point interpolated AP (AP_R40).

- perfect-ranking AP (PR): the same detections and TP set, every TP ranked above every FP;
- ranking loss RL = PR - AP, with a continuous-recall version beside it (area under the precision envelope, PR_c = 100 R);
- per ring, from the same pooled matching: TP binned by its GT's centre range, FP by its own;
- FP taxonomy at IoU 0.7, by the FP's best-overlap Vehicle GT g*: duplicate (IoU >= 0.1 and g* matched by a
  higher-ranked box), localisation (IoU >= 0.1 otherwise), other-class (a Pedestrian / Cyclist / Sign GT at BEV IoU
  >= 0.1, or with its centre inside the box), background with / without points inside the 3D box. The cloud is the
  loader's own `WaymoDataset.get_lidar` frame (all lidars, no-label-zone points dropped) cropped to POINT_CLOUD_RANGE
  in x, y and z (the processor masks x, y; the voxel grid drops z outside [-2, 4]);
- removal gains: AP with one FP class removed (removing an FP changes no assignment);
- pairing: Vehicle GT matched (any score) by the first model and not by another, and the reverse, at IoU 0.7 / 0.5;
- `--official`: actual AP, PR and background-removed AP from the official KITTI evaluator, unmodified except for the
  CPU IoU kernel (`kitti_eval_cpu.load_official_eval`) and one partition per frame (same overlaps, less memory).

    python analysis/waymo_ranking_loss_fp_taxonomy.py --out <dir> [--workers 8] [--official] \\
        oracle=<result.pkl> ours=<result.pkl> [more=<result.pkl> ...]

The first model is the reference in the gap and pairing tables. `--report_only` rebuilds the tables from `<out>/pass.pkl`.
"""
import argparse
import copy
import gc
import itertools
import json
import pickle
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

INFOS = TOOLS.parent / 'data/waymo/waymo_infos_val.pkl'
DATA_PATH = TOOLS.parent / 'data/waymo/waymo_processed_data'
PC_RANGE = (-75.2, -75.2, -2.0, 75.2, 75.2, 4.0)
RINGS = (0, 10, 20, 30, 40, 50, 75)
RING_NAMES = tuple('%g-%g' % (RINGS[k], RINGS[k + 1]) for k in range(len(RINGS) - 1))
PTS_BINS = ((0, 1), (1, 10), (10, 50), (50, 200), (200, 10 ** 9))
PTS_NAMES = ('0', '1-9', '10-49', '50-199', '>=200')
OTHER_NAMES = ('Pedestrian', 'Cyclist', 'Sign')
OVERLAP_MIN = 0.1  # BEV IoU at which an FP "touches" a GT
TP, LOC, DUP, OTHER, BG_PTS, BG_EMPTY, BG = 0, 1, 2, 3, 4, 5, 6
FP_CODES = (LOC, DUP, OTHER, BG_PTS, BG_EMPTY)
CODE_NAMES = {LOC: 'localisation', DUP: 'duplicate', OTHER: 'other-class', BG_PTS: 'background, with points',
              BG_EMPTY: 'background, empty'}
MAP_NAME_TO_KITTI = {'Vehicle': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist', 'Sign': 'Sign', 'Car': 'Car'}


# ----------------------------------------------------------------------------------------------- geometry and matching
def bev_iou(a, b):
    """BEV IoU [len(a), len(b)] of (x, y, z, dx, dy, dz, heading) boxes, pcdet's CPU op."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
    return np.asarray(boxes_bev_iou_cpu(np.ascontiguousarray(a[:, :7], np.float32),
                                        np.ascontiguousarray(b[:, :7], np.float32)))


def greedy_match(iou, scores, thr):
    """iou [n_dt, n_gt]. Detections in descending score (stable sort), each takes the unmatched GT of highest IoU > thr.

    Returns dt_gt (GT index or -1 per detection), gt_dt (detection index or -1 per GT) and rank (each detection's
    position in the score order, 0 = highest)."""
    n_dt, n_gt = iou.shape
    order = np.argsort(-np.asarray(scores, np.float64), kind='stable')
    rank = np.empty(n_dt, np.int64)
    rank[order] = np.arange(n_dt)
    dt_gt = -np.ones(n_dt, np.int64)
    gt_dt = -np.ones(n_gt, np.int64)
    if n_gt == 0 or n_dt == 0:
        return dt_gt, gt_dt, rank
    reach = iou.max(1) > thr  # a detection that overlaps no GT above thr can never match
    free = np.ones(n_gt, bool)
    for i in order[reach[order]]:
        row = np.where(free, iou[i], -1.0)
        j = int(row.argmax())
        if row[j] > thr:
            dt_gt[i] = j
            gt_dt[j] = i
            free[j] = False
    return dt_gt, gt_dt, rank


def centres_in_boxes_bev(centres, boxes):
    """[len(boxes), len(centres)] True where the (x, y) centre lies inside the box's rotated BEV rectangle."""
    if len(centres) == 0 or len(boxes) == 0:
        return np.zeros((len(boxes), len(centres)), bool)
    dx = centres[None, :, 0] - boxes[:, None, 0]
    dy = centres[None, :, 1] - boxes[:, None, 1]
    c, s = np.cos(boxes[:, 6])[:, None], np.sin(boxes[:, 6])[:, None]
    lx, ly = dx * c + dy * s, -dx * s + dy * c
    return (np.abs(lx) <= boxes[:, None, 3] / 2) & (np.abs(ly) <= boxes[:, None, 4] / 2)


BOX_MARGIN = 1e-2  # pcdet's roiaware points_in_boxes_cpu: |local x, y| < half size + 1 cm, |dz| <= half height


def points_in_boxes_count(points, boxes, tree=None):
    """Number of points (x, y, z) inside each 3D box (centre z, heading about z), via a BEV k-d tree pre-filter.

    Same containment rule as pcdet's `points_in_boxes_cpu` (pinned by the unit test)."""
    out = np.zeros(len(boxes), np.int64)
    if len(boxes) == 0 or len(points) == 0:
        return out
    from scipy.spatial import cKDTree
    tree = tree if tree is not None else cKDTree(points[:, :2])
    lists = tree.query_ball_point(boxes[:, :2], 0.5 * np.hypot(boxes[:, 3], boxes[:, 4]) + 2 * BOX_MARGIN)
    lens = np.fromiter((len(x) for x in lists), np.int64, len(boxes))
    if lens.sum() == 0:
        return out
    flat = np.fromiter(itertools.chain.from_iterable(lists), np.int64, int(lens.sum()))
    bi = np.repeat(np.arange(len(boxes)), lens)
    b, p = boxes[bi], points[flat]
    dx, dy = p[:, 0] - b[:, 0], p[:, 1] - b[:, 1]
    c, s = np.cos(b[:, 6]), np.sin(b[:, 6])
    inside = (np.abs(dx * c + dy * s) < b[:, 3] / 2 + BOX_MARGIN) & \
             (np.abs(-dx * s + dy * c) < b[:, 4] / 2 + BOX_MARGIN) & (np.abs(p[:, 2] - b[:, 2]) <= b[:, 5] / 2)
    return np.bincount(bi[inside], minlength=len(boxes))


def classify(iou_v, dt_gt, gt_dt, rank, other_touch):
    """Code per detection: TP, or for an FP: DUP / LOC by its best-overlap Vehicle GT, else OTHER, else BG."""
    codes = np.full(len(dt_gt), BG, np.int64)
    codes[dt_gt >= 0] = TP
    fp = np.where(dt_gt < 0)[0]
    if len(fp) and iou_v.shape[1] > 0:
        g = iou_v[fp].argmax(1)
        touch = iou_v[fp, g] >= OVERLAP_MIN
        m = gt_dt[g]
        dup = touch & (m >= 0) & (rank[np.maximum(m, 0)] < rank[fp])
        codes[fp[dup]] = DUP
        codes[fp[touch & ~dup]] = LOC
    codes[(codes == BG) & np.asarray(other_touch, bool)] = OTHER
    return codes


def frame_detections(det_boxes, det_scores, veh_boxes, other_boxes, points=None, thresholds=(0.7, 0.5)):
    """Everything per detection and per Vehicle GT for one frame and one model (see the module docstring)."""
    iou_v = bev_iou(det_boxes, veh_boxes)
    other_touch = np.zeros(len(det_boxes), bool)
    if len(other_boxes) and len(det_boxes):
        other_touch = (bev_iou(det_boxes, other_boxes) >= OVERLAP_MIN).any(1) | \
            centres_in_boxes_bev(other_boxes[:, :2], det_boxes).any(1)
    match = {thr: greedy_match(iou_v, det_scores, thr) for thr in thresholds}
    codes = classify(iou_v, *match[0.7], other_touch)
    npts_box = -np.ones(len(det_boxes), np.int64)
    bg = codes == BG
    if bg.any() and points is not None:
        npts_box[bg] = points_in_boxes_count(points[0], det_boxes[bg], points[1])
        codes[bg] = np.where(npts_box[bg] > 0, BG_PTS, BG_EMPTY)
    # 20261007_03 §1's per-ring protocol (matching inside each ring), kept only to reproduce that table
    rd, rg = np.hypot(det_boxes[:, 0], det_boxes[:, 1]), np.hypot(veh_boxes[:, 0], veh_boxes[:, 1])
    tp_ring = np.zeros(len(det_boxes), bool)
    for k in range(len(RINGS) - 1):
        gs = np.where((rg >= RINGS[k]) & (rg < RINGS[k + 1]))[0]
        ds = np.where((rd >= RINGS[k]) & (rd < RINGS[k + 1]))[0]
        if len(gs) and len(ds):
            tp_ring[ds[greedy_match(iou_v[np.ix_(ds, gs)], det_scores[ds], 0.7)[0] >= 0]] = True
    return dict(dt_gt7=match[0.7][0], dt_gt5=match[0.5][0], gt_dt7=match[0.7][1], gt_dt5=match[0.5][1], codes=codes,
                npts_box=npts_box, tp_ring=tp_ring,
                gt_best_iou=iou_v.max(0) if len(det_boxes) else np.zeros(len(veh_boxes), np.float32))


# --------------------------------------------------------------------------------------------------------------- AP
def ap_stats(scores, is_tp, n_gt):
    """(AP_R40, AP_continuous, max recall) of detections with fixed TP flags; APs in percent.

    AP_R40 = mean over r = 1/40 ... 1 of the highest precision at recall >= r (0 beyond the reach), as
    waymo_target_residual.ap_r40; AP_continuous = sum over TPs of the precision envelope at their rank / n_gt."""
    if n_gt == 0:
        return float('nan'), float('nan'), float('nan')
    scores, is_tp = np.asarray(scores, np.float64), np.asarray(is_tp, bool)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    t = is_tp[np.argsort(-scores, kind='stable')]
    tp = np.cumsum(t)
    fp = np.cumsum(~t)
    rec = tp / n_gt
    env = np.maximum.accumulate((tp / np.maximum(tp + fp, 1))[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 41)[1:], side='left')  # first rank with recall >= r
    r40 = np.where(idx < len(rec), env[np.minimum(idx, len(rec) - 1)], 0.0)
    return 100 * float(r40.mean()), 100 * float(env[t].sum() / n_gt), float(rec[-1])


def ap_split(scores, is_tp, n_gt):
    """AP, perfect-ranking AP (every TP above every FP) and ranking loss, under AP_R40 and the continuous version."""
    ap, apc, r = ap_stats(scores, is_tp, n_gt)
    pr, _, _ = ap_stats(np.asarray(scores, np.float64) + 2.0 * np.asarray(is_tp, bool), is_tp, n_gt)
    return dict(ap=ap, pr=pr, rl=pr - ap, ap_c=apc, pr_c=100 * r, rl_c=100 * r - apc, recall=r)


# ------------------------------------------------------------------------------------------------------- the pass
FRAMES, MODELS = None, None  # filled in the parent before the pool forks


def load_points(seq, idx):
    """The evaluated cloud: the loader's own get_lidar, cropped to POINT_CLOUD_RANGE in x, y, z; (xyz, k-d tree)."""
    from scipy.spatial import cKDTree
    from pcdet.datasets.waymo.waymo_dataset import WaymoDataset
    pts = WaymoDataset.get_lidar(SimpleNamespace(data_path=DATA_PATH, dataset_cfg={}), seq, idx)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    keep = (x >= PC_RANGE[0]) & (x <= PC_RANGE[3]) & (y >= PC_RANGE[1]) & (y <= PC_RANGE[4]) & \
           (z >= PC_RANGE[2]) & (z <= PC_RANGE[5])
    xyz = np.ascontiguousarray(pts[keep, :3], np.float64)
    return xyz, cKDTree(xyz[:, :2])


def _pass_frame(fi):
    f = FRAMES[fi]
    pts = load_points(f['seq'], f['idx']) if f['need_points'] else None
    return fi, [frame_detections(m[fi][0], m[fi][1], f['gv'], f['go'], pts) for m in MODELS], \
        (0 if pts is None else len(pts[0]))


def load_inputs(runs, max_frames=None):
    """Compact per-frame arrays for the frames of the first result.pkl (all models must hold the same frames);
    `max_frames` keeps the first N of them (smoke tests only)."""
    infos = pickle.load(open(INFOS, 'rb'))
    by_fid, names_seen = {}, {}
    for inf in infos:
        fid = inf['frame_id'] if 'frame_id' in inf else \
            inf['point_cloud']['lidar_sequence'] + '_%03d' % inf['point_cloud']['sample_idx']
        by_fid[fid] = inf
    del infos
    frames, models, fids = [], [], None
    for label, path in runs:
        res = pickle.load(open(path, 'rb'))
        res_by = {r['frame_id']: r for r in res}
        if fids is None:
            all_fids = [r['frame_id'] for r in res]
            fids = all_fids[:max_frames] if max_frames else all_fids
            for fid in fids:
                inf = by_fid[fid]
                a = inf['annos']
                names = np.asarray(a['name'])
                boxes = np.asarray(a['gt_boxes_lidar'], np.float32).reshape(len(names), -1)[:, :7] if len(names) \
                    else np.zeros((0, 7), np.float32)
                for n in names:
                    names_seen[str(n)] = names_seen.get(str(n), 0) + 1
                v, o = names == 'Vehicle', np.isin(names, OTHER_NAMES)
                frames.append(dict(fid=fid, seq=inf['point_cloud']['lidar_sequence'],
                                   idx=int(inf['point_cloud']['sample_idx']), gv=boxes[v], go=boxes[o],
                                   gnpts=np.asarray(a['num_points_in_gt'])[v].astype(np.int64)))
        assert set(res_by) == set(all_fids) and len(res) == len(all_fids), f'{label}: frame set differs from the first model'
        per = []
        for fid in fids:
            r = res_by[fid]
            m = np.asarray(r['name']) == 'Car'
            b = np.asarray(r['boxes_lidar'], np.float32).reshape(len(m), -1)[:, :7] if len(m) \
                else np.zeros((0, 7), np.float32)
            per.append((b[m], np.asarray(r['score'], np.float64)[m]))
        models.append(per)
        del res, res_by
        gc.collect()
    return frames, models, names_seen


def run_pass(runs, workers, max_frames=None):
    global FRAMES, MODELS
    t0 = time.time()
    FRAMES, MODELS, names_seen = load_inputs(runs, max_frames)
    for f in FRAMES:
        f['need_points'] = True
    print(f'loaded {len(FRAMES)} frames, {len(MODELS)} models in {time.time() - t0:.0f} s; GT names {names_seen}',
          flush=True)
    results = [None] * len(FRAMES)
    npts = np.zeros(len(FRAMES), np.int64)
    with Pool(workers) as pool:
        for n, (fi, out, k) in enumerate(pool.imap_unordered(_pass_frame, range(len(FRAMES)), chunksize=8)):
            results[fi] = out
            npts[fi] = k
            if n % 500 == 0:
                print(f'  {n}/{len(FRAMES)} frames, {time.time() - t0:.0f} s', flush=True)
    labels = [r[0] for r in runs]
    gt = dict(frame=np.concatenate([np.full(len(f['gv']), i) for i, f in enumerate(FRAMES)]),
              rng=np.concatenate([np.hypot(f['gv'][:, 0], f['gv'][:, 1]) for f in FRAMES]),
              npts=np.concatenate([f['gnpts'] for f in FRAMES]),
              length=np.concatenate([f['gv'][:, 3] for f in FRAMES]))
    gt_off = np.concatenate([[0], np.cumsum([len(f['gv']) for f in FRAMES])])
    per_model = {}
    for mi, label in enumerate(labels):
        d = {k: [] for k in ('frame', 'score', 'rng', 'code', 'gt7', 'gt5', 'npts_box', 'tp_ring')}
        g = {k: [] for k in ('det7_score', 'det5_score', 'best_iou')}
        for fi, f in enumerate(FRAMES):
            o = results[fi][mi]
            db, ds = MODELS[mi][fi]
            d['frame'].append(np.full(len(ds), fi))
            d['score'].append(ds)
            d['rng'].append(np.hypot(db[:, 0], db[:, 1]))
            d['code'].append(o['codes'])
            d['gt7'].append(np.where(o['dt_gt7'] >= 0, o['dt_gt7'] + gt_off[fi], -1))
            d['gt5'].append(np.where(o['dt_gt5'] >= 0, o['dt_gt5'] + gt_off[fi], -1))
            d['npts_box'].append(o['npts_box'])
            d['tp_ring'].append(o['tp_ring'])
            g['det7_score'].append(np.where(o['gt_dt7'] >= 0, ds[np.maximum(o['gt_dt7'], 0)] if len(ds) else 0, -1.0))
            g['det5_score'].append(np.where(o['gt_dt5'] >= 0, ds[np.maximum(o['gt_dt5'], 0)] if len(ds) else 0, -1.0))
            g['best_iou'].append(o['gt_best_iou'])
        per_model[label] = dict(det={k: np.concatenate(v) for k, v in d.items()},
                                gt={k: np.concatenate(v).astype(np.float64) for k, v in g.items()})
    print(f'pass done in {time.time() - t0:.0f} s; median evaluated points per frame {np.median(npts):.0f}', flush=True)
    return dict(labels=labels, runs=runs, n_frames=len(FRAMES), gt=gt, models=per_model, names_seen=names_seen,
                points_per_frame=npts)


# ------------------------------------------------------------------------------------------------------- official
def official_aps(frames, dets, variants):
    """Official KITTI-evaluator Car BEV AP_R40 at IoU 0.7 and 0.5 for each detection variant {name: per-frame dets}."""
    from kitti_eval_cpu import load_official_eval
    official = load_official_eval()  # first: it stubs the CUDA kernel module before anything can import it
    from pcdet.datasets.kitti import kitti_utils
    gt0 = [dict(name=np.array(['Vehicle'] * len(f['gv'])), gt_boxes_lidar=f['gv'].astype(np.float64)) for f in frames]
    kitti_utils.transform_annotations_to_kitti_format(gt0, map_name_to_kitti=MAP_NAME_TO_KITTI, info_with_fakelidar=False)
    min_overlaps = np.array([[[0.7]] * 3, [[0.5]] * 3])  # [overlap, metric, class]
    out = {}
    for name, per_frame in variants.items():
        t0 = time.time()
        dt = [dict(name=np.array(['Car'] * len(s)), boxes_lidar=b.astype(np.float64), score=s.astype(np.float64))
              for b, s in per_frame]
        kitti_utils.transform_annotations_to_kitti_format(dt, map_name_to_kitti=MAP_NAME_TO_KITTI)
        ret = official.eval_class(copy.deepcopy(gt0), dt, [0], [0], 1, min_overlaps, num_parts=len(gt0))
        ap = official.get_mAP_R40(ret['precision'])
        out[name] = (float(ap[0, 0, 0]), float(ap[0, 0, 1]))
        print(f'  official {name}: BEV AP_R40 {out[name][0]:.4f} / {out[name][1]:.4f} ({time.time() - t0:.0f} s)',
              flush=True)
    return out


def run_official(runs, passed):
    frames, models, _ = load_inputs(runs, passed.get('max_frames'))
    assert len(frames) == passed['n_frames']
    res = {}
    for mi, label in enumerate(passed['labels']):
        det = passed['models'][label]['det']
        tp7, tp5, code = det['gt7'] >= 0, det['gt5'] >= 0, det['code']
        split = np.cumsum([len(s) for _, s in models[mi]])[:-1]
        tp7_f, tp5_f, code_f = np.split(tp7, split), np.split(tp5, split), np.split(code, split)
        variants = {
            'actual': models[mi],
            'perfect_0.7': [(b, s + 2.0 * t) for (b, s), t in zip(models[mi], tp7_f)],
            'perfect_0.5': [(b, s + 2.0 * t) for (b, s), t in zip(models[mi], tp5_f)],
            'no_background': [(b[~np.isin(c, (BG_PTS, BG_EMPTY))], s[~np.isin(c, (BG_PTS, BG_EMPTY))])
                              for (b, s), c in zip(models[mi], code_f)],
        }
        print(f'official evaluator, {label}', flush=True)
        o = official_aps(frames, models[mi], variants)
        res[label] = dict(ap7=o['actual'][0], ap5=o['actual'][1], pr7=o['perfect_0.7'][0], pr5=o['perfect_0.5'][1],
                          no_bg7=o['no_background'][0])
    return res


# -------------------------------------------------------------------------------------------------------- tables
def summarise(passed, official=None):
    """All tables of 20261017_13 §5 as markdown, plus the numbers as a dict."""
    labels, gt, nf = passed['labels'], passed['gt'], passed['n_frames']
    n_gt = len(gt['rng'])
    ring_of = lambda r: np.searchsorted(np.asarray(RINGS[1:-1], np.float64), r, side='right')  # noqa: E731
    gt_ring = np.where(gt['rng'] < RINGS[-1], ring_of(gt['rng']), -1)
    num, lines = {}, []

    def table(head, rows):
        lines.append('| ' + ' | '.join(head) + ' |')
        lines.append('|' + '---|' * len(head))
        lines.extend('| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows)
        lines.append('')

    for label in labels:
        det = passed['models'][label]['det']
        n = num[label] = {}
        d_ring = np.where(det['rng'] < RINGS[-1], ring_of(det['rng']), -1)
        for thr, key in ((0.7, 'gt7'), (0.5, 'gt5')):
            tp = det[key] >= 0
            n[f'split{thr}'] = ap_split(det['score'], tp, n_gt)
            tp_ring = np.where(tp, gt_ring[np.maximum(det[key], 0)], -1)
            n[f'ring{thr}'] = []
            for k in range(len(RINGS) - 1):
                sel = (tp & (tp_ring == k)) | (~tp & (d_ring == k))
                n[f'ring{thr}'].append(ap_split(det['score'][sel], tp[sel], int((gt_ring == k).sum())))
        n['ring_restricted0.7'] = [ap_stats(det['score'][d_ring == k], det['tp_ring'][d_ring == k],
                                            int((gt_ring == k).sum()))[0] for k in range(len(RINGS) - 1)]
        tp = det['gt7'] >= 0
        s_tp = det['score'][tp]
        n['tp_median'], n['tp_p25'] = float(np.median(s_tp)), float(np.percentile(s_tp, 25))
        n['fp_per_frame'] = {}
        for c in FP_CODES + ('background',):
            m = np.isin(det['code'], (BG_PTS, BG_EMPTY)) if c == 'background' else det['code'] == c
            n['fp_per_frame'][c] = dict(all=m.sum() / nf, p25=(m & (det['score'] > n['tp_p25'])).sum() / nf,
                                        med=(m & (det['score'] > n['tp_median'])).sum() / nf,
                                        ring_med=[(m & (det['score'] > n['tp_median']) & (d_ring == k)).sum() / nf
                                                  for k in range(len(RINGS) - 1)],
                                        ring_all=[(m & (d_ring == k)).sum() / nf for k in range(len(RINGS) - 1)])
        bgm = np.isin(det['code'], (BG_PTS, BG_EMPTY)) & (det['score'] > n['tp_median'])
        n['bg_med_points'] = dict(n=int(bgm.sum()), share_with_points=float((det['code'][bgm] == BG_PTS).mean()) if bgm.any() else float('nan'),
                                  median_points_if_any=float(np.median(det['npts_box'][bgm & (det['code'] == BG_PTS)])) if (bgm & (det['code'] == BG_PTS)).any() else float('nan'))
        n['removal'] = {}
        for c in FP_CODES + ('background', 'all'):
            if c == 'background':
                drop = np.isin(det['code'], (BG_PTS, BG_EMPTY))
            elif c == 'all':
                drop = ~tp
            else:
                drop = det['code'] == c
            s = ap_stats(det['score'][~drop], tp[~drop], n_gt)
            n['removal'][c] = dict(ap=s[0], gain=s[0] - n['split0.7']['ap'], gain_c=s[1] - n['split0.7']['ap_c'])
        n['tp_q_ring'], n['tp_q_pts'] = [], []
        tpr = gt_ring[det['gt7'][tp]]
        tpp = gt['npts'][det['gt7'][tp]]
        for k in range(len(RINGS) - 1):
            v = s_tp[tpr == k]
            n['tp_q_ring'].append([float(np.percentile(v, q)) for q in (10, 25, 50, 75, 90)] + [int(len(v))] if len(v) else None)
        for lo, hi in PTS_BINS:
            v = s_tp[(tpp >= lo) & (tpp < hi)]
            n['tp_q_pts'].append([float(np.percentile(v, q)) for q in (10, 25, 50, 75, 90)] + [int(len(v))] if len(v) else None)

    ref = labels[0]
    lines.append(f'frames {nf}, Vehicle GT {n_gt}; GT names seen {passed["names_seen"]}')
    lines.append(f'median evaluated points per frame {np.median(passed["points_per_frame"]):.0f}')
    lines.append('')
    lines.append('### Pooled AP split (greedy matching; AP_R40 and the continuous version)')
    table(['model', 'IoU', 'AP', 'PR', 'RL', 'max recall', 'AP_c', 'PR_c', 'RL_c'],
          [[lab, thr, '%.2f' % s['ap'], '%.2f' % s['pr'], '%.2f' % s['rl'], '%.4f' % s['recall'], '%.2f' % s['ap_c'],
            '%.2f' % s['pr_c'], '%.2f' % s['rl_c']] for lab in labels for thr in (0.7, 0.5)
           for s in [num[lab][f'split{thr}']]])
    lines.append(f'### Gap to {ref}: AP_ref - AP_x = recall term (PR_ref - PR_x) + ranking term (RL_x - RL_ref)')
    rows = []
    for lab in labels[1:]:
        for thr in (0.7, 0.5):
            a, b = num[ref][f'split{thr}'], num[lab][f'split{thr}']
            rows.append([lab, thr, '%.2f' % (a['ap'] - b['ap']), '%.2f' % (a['pr'] - b['pr']), '%.2f' % (b['rl'] - a['rl']),
                         '%.2f' % (a['ap_c'] - b['ap_c']), '%.2f' % (a['pr_c'] - b['pr_c']), '%.2f' % (b['rl_c'] - a['rl_c'])])
    table(['model', 'IoU', 'gap (R40)', 'recall term', 'ranking term', 'gap (cont.)', 'recall term', 'ranking term'], rows)
    for thr in (0.7, 0.5):
        lines.append(f'### Per ring, IoU {thr} (pooled matching; TP by GT ring, FP by own ring): AP / PR / RL')
        table(['model'] + list(RING_NAMES),
              [[lab] + ['%.1f / %.1f / %.1f' % (s['ap'], s['pr'], s['rl']) for s in num[lab][f'ring{thr}']] for lab in labels]
              + [['GT'] + [str(int((gt_ring == k).sum())) for k in range(len(RINGS) - 1)]])
        lines.append(f'Per ring, IoU {thr}, continuous: AP_c / PR_c / RL_c')
        table(['model'] + list(RING_NAMES),
              [[lab] + ['%.1f / %.1f / %.1f' % (s['ap_c'], s['pr_c'], s['rl_c']) for s in num[lab][f'ring{thr}']] for lab in labels])
    lines.append('### 20261007_03 §1 per-ring protocol (matching inside each ring), IoU 0.7: AP_R40 (reproduction check)')
    table(['model'] + list(RING_NAMES), [[lab] + ['%.1f' % v for v in num[lab]['ring_restricted0.7']] for lab in labels])
    lines.append('### FP taxonomy at IoU 0.7: FPs per frame (all / above own TP p25 / above own TP median)')
    table(['model', 'TP p25 / median'] + [CODE_NAMES[c] for c in FP_CODES] + ['background, both'],
          [[lab, '%.3f / %.3f' % (num[lab]['tp_p25'], num[lab]['tp_median'])] +
           ['%.2f / %.3f / %.3f' % (v['all'], v['p25'], v['med']) for c in FP_CODES + ('background',)
            for v in [num[lab]['fp_per_frame'][c]]] for lab in labels])
    for lab in labels[1:]:
        ratio = lambda a, b: 'n/a' if b == 0 else '%.2f' % (a / b)  # noqa: E731
        lines.append(f'ratio {lab} / {ref}, above own TP median: ' + ', '.join(
            '%s %s' % (CODE_NAMES.get(c, c), ratio(num[lab]['fp_per_frame'][c]['med'], num[ref]['fp_per_frame'][c]['med']))
            for c in FP_CODES + ('background',)))
    lines.append('')
    for lab in labels:
        b = num[lab]['bg_med_points']
        lines.append(f'{lab}: background FPs above TP median: {b["n"]}, share with points {b["share_with_points"]:.3f}, '
                     f'median points when non-empty {b["median_points_if_any"]:.0f}')
    lines.append('')
    lines.append('### FPs per frame above own TP median, by ring of the FP centre')
    table(['model', 'class'] + list(RING_NAMES),
          [[lab, CODE_NAMES.get(c, c)] + ['%.3f' % v for v in num[lab]['fp_per_frame'][c]['ring_med']]
           for lab in labels for c in FP_CODES + ('background',)])
    lines.append('### All FPs per frame, by ring of the FP centre')
    table(['model', 'class'] + list(RING_NAMES),
          [[lab, CODE_NAMES.get(c, c)] + ['%.2f' % v for v in num[lab]['fp_per_frame'][c]['ring_all']]
           for lab in labels for c in FP_CODES + ('background',)])
    lines.append('### Removal gains at IoU 0.7: AP with one FP class removed - AP (R40; continuous in brackets)')
    table(['model'] + [CODE_NAMES.get(c, c) for c in FP_CODES + ('background', 'all')],
          [[lab] + ['%+.2f (%+.2f)' % (num[lab]['removal'][c]['gain'], num[lab]['removal'][c]['gain_c'])
                    for c in FP_CODES + ('background', 'all')] for lab in labels])
    lines.append('### TP scores at IoU 0.7: p10 / p25 / p50 / p75 / p90 (n) per GT ring')
    fmt = lambda q: '-' if q is None else '%.2f / %.2f / %.2f / %.2f / %.2f (%d)' % tuple(q)  # noqa: E731
    table(['model'] + list(RING_NAMES), [[lab] + [fmt(q) for q in num[lab]['tp_q_ring']] for lab in labels])
    lines.append('### TP scores at IoU 0.7 per stored-points bin of the GT')
    table(['model'] + list(PTS_NAMES), [[lab] + [fmt(q) for q in num[lab]['tp_q_pts']] for lab in labels])
    # pairing
    pts_bin = np.digitize(gt['npts'], [b[1] for b in PTS_BINS[:-1]])
    for thr, key in ((0.7, 'det7_score'), (0.5, 'det5_score')):
        found_ref = passed['models'][ref]['gt'][key] >= 0
        for lab in labels[1:]:
            g = passed['models'][lab]['gt']
            found = g[key] >= 0
            only_ref, only_x = found_ref & ~found, found & ~found_ref
            num[lab][f'pair{thr}'] = dict(only_ref=int(only_ref.sum()), only_x=int(only_x.sum()),
                                          both=int((found & found_ref).sum()), neither=int((~found & ~found_ref).sum()))
            lines.append(f'### Pairing at IoU {thr}: {ref} finds and {lab} does not / {lab} finds and {ref} does not / both / GT')
            rows = []
            for k in range(len(RINGS) - 1):
                s = gt_ring == k
                rows.append([RING_NAMES[k] + ' m', int((only_ref & s).sum()), int((only_x & s).sum()), int((found & found_ref & s).sum()), int(s.sum())])
            for i, nm in enumerate(PTS_NAMES):
                s = pts_bin == i
                rows.append([nm + ' pts', int((only_ref & s).sum()), int((only_x & s).sum()), int((found & found_ref & s).sum()), int(s.sum())])
            rows.append(['all', int(only_ref.sum()), int(only_x.sum()), int((found & found_ref).sum()), n_gt])
            table(['bin', f'{ref} only', f'{lab} only', 'both', 'GT'], rows)
            bi = g['best_iou'][only_ref]
            hist = [int((bi < 1e-6).sum()), int(((bi >= 1e-6) & (bi < 0.1)).sum()), int(((bi >= 0.1) & (bi < 0.5)).sum()),
                    int(((bi >= 0.5) & (bi <= thr)).sum())]
            num[lab][f'pair{thr}']['only_ref_best_iou_hist'] = hist
            lines.append(f'{lab}: best BEV IoU of its own boxes to the {ref}-only GT: none {hist[0]}, (0, 0.1) {hist[1]}, '
                         f'[0.1, 0.5) {hist[2]}, [0.5, {thr}] {hist[3]}')
            lines.append('')
    if official:
        lines.append('### Official KITTI evaluator (CPU IoU kernel), BEV AP_R40')
        table(['model', 'AP 0.7', 'AP 0.5', 'PR 0.7', 'PR 0.5', 'RL 0.7', 'RL 0.5', 'AP 0.7 without background', 'gain'],
              [[lab, '%.4f' % o['ap7'], '%.4f' % o['ap5'], '%.2f' % o['pr7'], '%.2f' % o['pr5'], '%.2f' % (o['pr7'] - o['ap7']),
                '%.2f' % (o['pr5'] - o['ap5']), '%.2f' % o['no_bg7'], '%+.2f' % (o['no_bg7'] - o['ap7'])]
               for lab in labels for o in [official[lab]]])
        num['official'] = official
    return '\n'.join(lines), num


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='<label>=<result.pkl>; the first is the reference')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--official', action='store_true')
    ap.add_argument('--report_only', action='store_true')
    ap.add_argument('--max_frames', type=int, default=None, help='first N frames only (smoke test)')
    a = ap.parse_args()
    runs = [tuple(r.split('=', 1)) for r in a.runs]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.report_only:
        passed = pickle.load(open(out / 'pass.pkl', 'rb'))
    else:
        passed = run_pass(runs, a.workers, a.max_frames)
        passed['max_frames'] = a.max_frames
        pickle.dump(passed, open(out / 'pass.pkl', 'wb'))
    official = None
    if a.official:
        official = run_official(runs, passed)
        json.dump(official, open(out / 'official.json', 'w'), indent=1)
    elif (out / 'official.json').exists():
        official = json.load(open(out / 'official.json'))
    text, num = summarise(passed, official)
    (out / 'tables.md').write_text(text + '\n')
    json.dump(num, open(out / 'numbers.json', 'w'), indent=1, default=float)
    print(text, flush=True)


if __name__ == '__main__':
    main()
