"""Check PL-NEG: where each teacher's near-misses sit against its own pseudo-label cuts (CPU, ANALYSIS).

experiments_md 20261003_05 §5 ("Check PL-NEG", fixed 2026-10-22 21:36 JST) holds the design; 20261022_03 holds the
operational readings fixed before any number and the results. ANALYSIS ONLY: it reads target val labels to explain a
result, never to choose a setting or a cut (memory/repo/uda_legality_rule.md).

For each teacher's saved predictions on its target val (score floor 0.0001):
- GT set and ignores are the evaluation's own: annotations are built exactly as each dataset's KITTI-metric evaluation
  builds them (nuScenes / Waymo: `kitti_utils.transform_annotations_to_kitti_format` with the dataset's name map, which
  fabricates [0, 0, 50, 50] 2D boxes, so every Car GT is valid; KITTI: the info annotations as they are, with real 2D
  boxes, occlusion and truncation), and the official `clean_data` marks each GT valid (0), ignored (1: Van, or a Car
  outside the difficulty) or another class (-1), and each Car detection valid (0) or ignored (1: 2D height under the
  difficulty's minimum). Overlaps are the official BEV overlap in the camera frame (CPU kernel of `kitti_eval_cpu`).
- Matching is PL-SRC's greedy BEV matching at IoU 0.7 over the Car detections, extended with the evaluator's ignores:
  detections in descending score each take the still-unmatched non-(-1) GT of highest IoU above 0.7. A detection
  matched to a valid GT is a TP; matched to an ignored GT, or unmatched with ignored height, it is ignored; else an FP.
  With no ignores (nuScenes, Waymo) this is exactly PL-SRC's matching.
- tau: the lowest score whose detection set {score >= s} has pooled precision >= 0.7 over TP + FP (ignored left out).
- found: a valid GT matched by a detection with score >= tau. near-miss: a valid GT not found, with a Car detection at
  BEV IoU >= 0.5 at any score; its score is the highest score among those detections.
- zones against the student's Car cuts: >= SCORE, [NEG, SCORE), < NEG. C = share of near-misses scoring >= NEG.

    python analysis/pl_teacher_near_miss_cuts.py --out <dir> [--workers 8] [--max_frames N] [--no_official] \\
        <label>=<kitti|nuscenes|waymo>:<result.pkl>:<SCORE>:<NEG>:<group adding|level>[:<logged AP>] ...
"""
import argparse
import copy
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS.parent, TOOLS, TOOLS / 'analysis'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import waymo_ranking_loss_fp_taxonomy as W  # noqa: E402  (imports _init_path: the live pcdet)
import pl_student_new_detections_source as P  # noqa: E402
from kitti_eval_cpu import load_official_eval  # noqa: E402

KITTI_INFOS = TOOLS.parent / 'data/kitti/kitti_infos_val.pkl'
NUSC_INFOS = TOOLS.parent / 'data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl'
# nuscenes_dataset.kitti_eval's map, copied (it is local to that method)
NUSC_MAP = {'car': 'Car', 'Car': 'Car', 'pedestrian': 'Pedestrian', 'Pedestrian': 'Pedestrian', 'truck': 'Truck',
            'motorcycle': 'Cyclist', 'bicycle': 'Cyclist', 'Cyclist': 'Cyclist'}
WAYMO_MAP = {'Vehicle': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist', 'Sign': 'Sign', 'Car': 'Car'}
DIFFICULTY = {'kitti': 1, 'nuscenes': 1, 'waymo': 1}  # moderate; nuScenes / Waymo columns are identical by construction
TP_, FP_, IGN_ = 0, 1, 2
ZONES = ('>= SCORE', '[NEG, SCORE)', '< NEG')
PRECISION = 0.7
H_BAND_MIN = 0.10

OFFICIAL = None
FRAMES = None  # list of (gt_anno, car-only dt_anno, difficulty), set before the pool forks


def official():
    global OFFICIAL
    if OFFICIAL is None:
        OFFICIAL = load_official_eval()  # first: it stubs the CUDA kernel module
    return OFFICIAL


# ------------------------------------------------------------------------------------------------ annotations
def load_annos(kind, result_pkl, max_frames=None):
    """(gt_annos, dt_annos) in KITTI format, frame-aligned, exactly as the dataset's KITTI-metric evaluation builds
    them; dt_annos hold every class (the official check), the per-frame pass takes the Car ones."""
    official()
    from pcdet.datasets.kitti import kitti_utils
    dets = pickle.load(open(result_pkl, 'rb'))
    if kind == 'kitti':
        infos = pickle.load(open(KITTI_INFOS, 'rb'))
        by = {d['frame_id']: d for d in dets}
        assert len(by) == len(infos) == len(dets), 'KITTI: frame sets differ'
        gt = [copy.deepcopy(i['annos']) for i in infos]
        dt = [copy.deepcopy(by[i['point_cloud']['lidar_idx']]) for i in infos]
        fids = [i['point_cloud']['lidar_idx'] for i in infos]
    elif kind == 'nuscenes':
        infos = pickle.load(open(NUSC_INFOS, 'rb'))
        assert len(infos) == len(dets), 'nuScenes: frame counts differ'
        for i, d in zip(infos, dets):  # the evaluation pairs by order; check it
            tok = (d.get('metadata') or {}).get('token')
            assert tok is None or tok == i['token'], 'nuScenes: frame order differs from the infos'
        gt = [dict(name=copy.deepcopy(i['gt_names']), gt_boxes=np.asarray(i['gt_boxes'], np.float64)[:, :7].copy())
              for i in infos]
        dt = [copy.deepcopy(d) for d in dets]
        kitti_utils.transform_annotations_to_kitti_format(dt, map_name_to_kitti=NUSC_MAP)
        kitti_utils.transform_annotations_to_kitti_format(gt, map_name_to_kitti=NUSC_MAP, info_with_fakelidar=False)
        fids = [i['token'] for i in infos]
    elif kind == 'waymo':
        infos = pickle.load(open(W.INFOS, 'rb'))
        by_fid = {(i['frame_id'] if 'frame_id' in i else i['point_cloud']['lidar_sequence'] + '_%03d'
                   % i['point_cloud']['sample_idx']): i for i in infos}
        del infos
        fids = [d['frame_id'] for d in dets]
        gt = [copy.deepcopy(by_fid[f]['annos']) for f in fids]
        for g in gt:  # a frame without labels stores an empty array that cannot be reshaped
            g['gt_boxes_lidar'] = np.asarray(g['gt_boxes_lidar'], np.float64).reshape(len(g['name']), -1)[:, :7].copy() \
                if len(g['name']) else np.zeros((0, 7))
        dt = [copy.deepcopy(d) for d in dets]
        kitti_utils.transform_annotations_to_kitti_format(dt, map_name_to_kitti=WAYMO_MAP)
        kitti_utils.transform_annotations_to_kitti_format(gt, map_name_to_kitti=WAYMO_MAP, info_with_fakelidar=False)
    else:
        raise ValueError(kind)
    if max_frames:
        gt, dt, fids = gt[:max_frames], dt[:max_frames], fids[:max_frames]
    return gt, dt, fids


def car_only(dt):
    m = np.asarray(dt['name']) == 'Car'
    return {k: (np.asarray(v)[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in dt.items()}


def bev(anno):
    """The evaluator's camera-frame BEV boxes (x, z, l, w, ry)."""
    if len(anno['rotation_y']) == 0:
        return np.zeros((0, 5))
    return np.concatenate([anno['location'][:, [0, 2]], anno['dimensions'][:, [0, 2]],
                           np.asarray(anno['rotation_y'])[:, None]], 1)


# ------------------------------------------------------------------------------------------------ pure functions
def label_frame(iou, scores, ig_gt, ig_dt, thr=0.7, near=0.5):
    """iou [n_dt, n_gt] (camera BEV), ig_gt per GT (0 valid, 1 ignored, -1 other class), ig_dt per Car detection
    (0 valid, 1 ignored by 2D height). Returns per detection the label (TP_ / FP_ / IGN_) and per GT the matched
    detection's score (-1 if none) and the highest score among detections at IoU >= `near` (-1 if none)."""
    iou = np.asarray(iou, np.float64)
    scores = np.asarray(scores, np.float64)
    ig_gt, ig_dt = np.asarray(ig_gt, np.int64), np.asarray(ig_dt, np.int64)
    n_dt, n_gt = iou.shape
    lab = np.full(n_dt, FP_, np.int64)
    matched = -np.ones(n_gt)
    best_near = -np.ones(n_gt)
    if n_dt == 0:
        return lab, matched, best_near
    m = iou * (ig_gt != -1)[None, :] if n_gt else iou
    dt_gt, gt_dt, _ = W.greedy_match(m, scores, thr)
    for i in range(n_dt):
        j = dt_gt[i]
        if j >= 0:
            lab[i] = TP_ if ig_gt[j] == 0 else IGN_
        elif ig_dt[i] == 1:
            lab[i] = IGN_
    if n_gt:
        matched = np.where(gt_dt >= 0, scores[np.maximum(gt_dt, 0)], -1.0)
        s = np.where(iou >= near, scores[:, None], -1.0)
        best_near = s.max(0)
    return lab, matched, best_near


def zone_shares(scores, score_cut, neg_cut):
    s = np.asarray(scores, np.float64)
    n = max(len(s), 1)
    return dict(n=int(len(s)), ge_score=float((s >= score_cut).sum() / n),
                band=float(((s >= neg_cut) & (s < score_cut)).sum() / n), lt_neg=float((s < neg_cut).sum() / n),
                n_ge_score=int((s >= score_cut).sum()), n_band=int(((s >= neg_cut) & (s < score_cut)).sum()),
                n_lt_neg=int((s < neg_cut).sum()))


def h_band_reading(c_adding, c_level, margin=H_BAND_MIN):
    lo, hi = min(c_adding), max(c_level)
    if lo - hi >= margin - 1e-12:  # >= 0.10, robust to float rounding at the bar
        return 'supported'
    if hi >= lo:
        return 'contradicted'
    return 'inconclusive'


# ------------------------------------------------------------------------------------------------ the pass
def _frame(fi):
    gt, dt, diff = FRAMES[fi]
    off = official()
    _, ig_gt, ig_dt, _ = off.clean_data(gt, dt, 0, diff)
    iou = off.bev_box_overlap(bev(dt), bev(gt)) if len(dt['name']) and len(gt['name']) else \
        np.zeros((len(dt['name']), len(gt['name'])))
    lab, matched, best_near = label_frame(iou, dt['score'], ig_gt, ig_dt)
    return fi, lab, np.asarray(dt['score'], np.float64), matched, best_near, np.asarray(ig_gt)


def run_pair(label, kind, result_pkl, workers, max_frames=None, do_official=True):
    global FRAMES
    t0 = time.time()
    gt, dt, fids = load_annos(kind, result_pkl, max_frames)
    diff = DIFFICULTY[kind]
    FRAMES = [(g, car_only(d), diff) for g, d in zip(gt, dt)]
    out = [None] * len(FRAMES)
    with Pool(workers) as pool:
        for fi, *r in pool.imap_unordered(_frame, range(len(FRAMES)), chunksize=16):
            out[fi] = r
    lab = np.concatenate([o[0] for o in out])
    score = np.concatenate([o[1] for o in out])
    det_frame = np.concatenate([np.full(len(o[1]), i) for i, o in enumerate(out)])
    ig_gt = np.concatenate([o[4] for o in out])
    valid = ig_gt == 0
    matched = np.concatenate([o[2] for o in out])[valid]
    best_near = np.concatenate([o[3] for o in out])[valid]
    gt_frame = np.concatenate([np.full(len(o[4]), i) for i, o in enumerate(out)])[valid]
    res = dict(label=label, kind=kind, result_pkl=str(result_pkl), n_frames=len(FRAMES), difficulty=diff,
               n_valid_gt=int(valid.sum()), n_ignored_gt=int((ig_gt == 1).sum()), n_car_dets=int(len(score)),
               n_ignored_dets=int((lab == IGN_).sum()))
    keep = lab != IGN_
    res['greedy_ap7'], res['greedy_ap7_c'], res['greedy_max_recall'] = W.ap_stats(score[keep], lab[keep] == TP_,
                                                                                 res['n_valid_gt'])
    res['official_ap7'] = None
    if do_official:
        off = official()
        min_overlaps = np.array([[[0.7]] * 3, [[0.5]] * 3])
        ret = off.eval_class(copy.deepcopy(gt), copy.deepcopy(dt), [0], [diff], 1, min_overlaps, num_parts=len(gt))
        res['official_ap7'] = float(off.get_mAP_R40(ret['precision'])[0, 0, 0])
    print(f'{label}: {len(FRAMES)} frames in {time.time() - t0:.0f} s; greedy {res["greedy_ap7"]:.2f}, official '
          f'{res["official_ap7"]}', flush=True)
    arrays = dict(lab=lab, score=score, det_frame=det_frame, matched=matched, best_near=best_near, gt_frame=gt_frame,
                  fids=fids)
    return res, arrays


def read(res, arr, score_cut, neg_cut):
    lab, score = arr['lab'], arr['score']
    keep = lab != IGN_
    tau, prec, ntp, tau_first = P.tau_at_precision(score[keep], lab[keep] == TP_, PRECISION)
    found = (arr['matched'] >= 0) & (arr['matched'] >= tau)
    assert ntp == int(found.sum()), 'TPs above tau != valid GT found at tau'
    near = ~found & (arr['best_near'] >= 0)
    fp = lab == FP_
    nf = res['n_frames']
    r = dict(tau=tau, precision_at_tau=prec, recall_at_tau=ntp / max(res['n_valid_gt'], 1), tau_first=tau_first,
             score_cut=score_cut, neg_cut=neg_cut, found=int(found.sum()), near_miss=int(near.sum()),
             near_miss_per_frame=float(near.sum() / nf), missed_no_box=int((~found & (arr['best_near'] < 0)).sum()),
             zones_near=zone_shares(arr['best_near'][near], score_cut, neg_cut),
             zones_found=zone_shares(arr['matched'][found], score_cut, neg_cut),
             zones_fp=zone_shares(score[fp], score_cut, neg_cut), fp_per_frame=float(fp.sum() / nf))
    r['C'] = r['zones_near']['ge_score'] + r['zones_near']['band']
    r['fp_per_frame_zone'] = dict(ge_score=r['zones_fp']['n_ge_score'] / nf, band=r['zones_fp']['n_band'] / nf,
                                  lt_neg=r['zones_fp']['n_lt_neg'] / nf)
    r['near_miss_score_q'] = [float(np.percentile(arr['best_near'][near], q)) for q in (10, 25, 50, 75, 90)] \
        if near.any() else None
    # where tau sits against the cuts (reported)
    r['tau_vs_cuts'] = '>= SCORE' if tau >= score_cut else ('[NEG, SCORE)' if tau >= neg_cut else '< NEG')
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pairs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--no_official', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, groups = [], {}
    for spec in a.pairs:
        label, rest = spec.split('=', 1)
        f = rest.split(':')
        kind, pkl, score_cut, neg_cut, group = f[0], f[1], float(f[2]), float(f[3]), f[4]
        logged = float(f[5]) if len(f) > 5 else None
        res, arr = run_pair(label, kind, pkl, a.workers, a.max_frames, not a.no_official)
        res['logged_ap7'], res['group'] = logged, group
        res.update(read(res, arr, score_cut, neg_cut))
        np.savez_compressed(out / f'{label}_arrays.npz', **{k: np.asarray(v) for k, v in arr.items()})
        rows.append(res)
        groups.setdefault(group, []).append(res['C'])
    reading = None
    if groups.get('adding') and groups.get('level'):
        reading = dict(min_C_adding=min(groups['adding']), max_C_level=max(groups['level']),
                       diff=min(groups['adding']) - max(groups['level']),
                       outcome=h_band_reading(groups['adding'], groups['level']))
    json.dump(dict(pairs=rows, reading=reading), open(out / 'numbers.json', 'w'), indent=1, default=float)
    lines = ['| pair | group | frames | valid GT | logged AP | official AP | greedy AP | SCORE / NEG | tau (P, R) | found '
             '| near-miss (/frame) | C | near-miss >=SCORE / band / <NEG | found >=SCORE / band / <NEG | FP >=SCORE / '
             'band / <NEG | FP per frame by zone | ignored GT / dets |',
             '|' + '---|' * 17]
    for r in rows:
        zn, zf, zp = r['zones_near'], r['zones_found'], r['zones_fp']
        lines.append(
            f"| {r['label']} | {r['group']} | {r['n_frames']} | {r['n_valid_gt']} | {r['logged_ap7']} | "
            f"{'-' if r['official_ap7'] is None else '%.4f' % r['official_ap7']} | {r['greedy_ap7']:.2f} | "
            f"{r['score_cut']} / {r['neg_cut']} | {r['tau']:.4f} ({r['precision_at_tau']:.4f}, {r['recall_at_tau']:.4f}) | "
            f"{r['found']} | {r['near_miss']} ({r['near_miss_per_frame']:.3f}) | **{r['C']:.4f}** | "
            f"{zn['ge_score']:.4f} / {zn['band']:.4f} / {zn['lt_neg']:.4f} | {zf['ge_score']:.4f} / {zf['band']:.4f} / "
            f"{zf['lt_neg']:.4f} | {zp['ge_score']:.4f} / {zp['band']:.4f} / {zp['lt_neg']:.4f} | "
            f"{r['fp_per_frame_zone']['ge_score']:.3f} / {r['fp_per_frame_zone']['band']:.3f} / "
            f"{r['fp_per_frame_zone']['lt_neg']:.3f} | {r['n_ignored_gt']} / {r['n_ignored_dets']} |")
    lines.append('')
    for r in rows:
        lines.append(f"{r['label']}: tau {r['tau_vs_cuts']} (tau_first {r['tau_first']:.4f}); near-miss score "
                     f"p10/25/50/75/90 {r['near_miss_score_q']}; missed with no box at IoU >= 0.5: {r['missed_no_box']}; "
                     f"counts near {json.dumps({k: r['zones_near'][k] for k in ('n_ge_score', 'n_band', 'n_lt_neg')})}")
    lines.append('')
    lines.append('H-band reading: ' + json.dumps(reading))
    (out / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
