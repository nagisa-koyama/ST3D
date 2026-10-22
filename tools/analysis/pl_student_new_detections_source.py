"""Check PL-SRC: where the Car GT that a pseudo-labelling student finds and its teacher misses come from (CPU, ANALYSIS).

experiments_md 20261003_05 §5 ("Check PL-SRC", fixed 2026-10-22 21:01 JST) holds the design; 20261022_02 holds the
operational readings fixed before any number and the results. ANALYSIS ONLY: it reads Waymo val labels (boxes, object
IDs) to explain a result, never to choose a setting (memory/repo/uda_legality_rule.md).

Matching is 20261017_13's, reused from `waymo_ranking_loss_fp_taxonomy`: Car detections against Vehicle GT, no
difficulty or range filter, greedy BEV matching at IoU 0.7 (detections in descending score, each takes the unmatched GT
of highest BEV IoU above 0.7). The per-GT point count is the stored `num_points_in_gt`, as there.

- tau_m: each model's score at pooled precision p over all val frames: the lowest score s whose detection set
  {score >= s} has precision >= p (TP flags from the greedy matching over all detections; greedy runs in descending
  score, so the TP set above s is the same as matching only the detections above s).
- found_m(g): g is matched by a detection of model m with score >= tau_m.
- sets: both, S-only, T-only, neither.
- R-a (track link): g is linked if the same object ID has a teacher-found Vehicle GT in ANOTHER evaluated frame of the
  same sequence. L(set) = share linked.
- R-b (adjacency): bins range (0-10, ..., 50-75 m, GT centre BEV range) x stored points (1-9, 10-49, 50-199, >= 200);
  zero-point GT and GT at >= 75 m sit outside the bins. r_T = teacher found rate per bin; Q = bins with r_T >= 0.2;
  share_X = |X in Q| / |X in the bins|.
- localisation split of S-only: the teacher's best BEV IoU to g among its detections with score >= tau_T, in [0.5, 0.7]
  (localisation), > 0.7 (its box went to a neighbouring GT), else none (a new detection).

    python analysis/pl_student_new_detections_source.py --out <dir> [--workers 8] [--max_frames N] \\
        [--ref_pass <20261017_13 pass.pkl> --ref_labels teacher=ours,student=27704] \\
        teacher=<result.pkl> student=<result.pkl> [student2=<result.pkl> ...]

The first run is the teacher, the second the deciding student; any further run is a replication student, read against
the same teacher. `--report_only` rebuilds the tables from `<out>/pass.pkl`.
"""
import argparse
import gc
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

RINGS = W.RINGS                      # (0, 10, 20, 30, 40, 50, 75)
RING_NAMES = W.RING_NAMES
PTS_EDGES = (1, 10, 50, 200)         # bins 1-9, 10-49, 50-199, >= 200; 0 points sits outside
PTS_NAMES = ('1-9', '10-49', '50-199', '>=200')
PAIR_MIN_IOU = 0.1                   # sparse (detection, GT) overlaps kept for the localisation split
OPERATING_POINTS = (('p0.7', 'precision', 0.7), ('p0.5', 'precision', 0.5), ('s0.3', 'score', 0.3))
R_T_MIN = 0.2                        # R-b: a bin qualifies if the teacher's found rate there is >= 0.2
RA_L_MIN, RA_DIFF_MIN, RB_DIFF_MIN = 0.5, 0.10, 0.10


# ------------------------------------------------------------------------------------------------ pure functions
def tau_at_precision(scores, is_tp, p):
    """Lowest score s such that the detections with score >= s have pooled precision >= p.

    Returns (tau, precision at tau, number of TPs at tau, tau_first), where tau_first is the score threshold just
    before the precision first falls below p, walking down from the top (reported only; equal to tau when precision
    never recovers above p after its first fall). Thresholds are taken at unique scores, ties kept together."""
    scores = np.asarray(scores, np.float64)
    is_tp = np.asarray(is_tp, bool)
    if len(scores) == 0:
        return float('nan'), float('nan'), 0, float('nan')
    order = np.argsort(-scores, kind='stable')
    s, t = scores[order], is_tp[order]
    tp = np.cumsum(t)
    n = np.arange(1, len(s) + 1)
    last = np.r_[np.nonzero(s[:-1] != s[1:])[0], len(s) - 1]  # end of each tie group
    prec = tp[last] / n[last]
    ok = prec >= p
    if not ok.any():
        return float('nan'), float('nan'), 0, float('nan')
    k = last[np.nonzero(ok)[0][-1]]
    below = np.nonzero(~ok)[0]
    if len(below) == 0:
        tau_first = s[last[-1]]
    elif below[0] == 0:
        tau_first = float('nan')
    else:
        tau_first = s[last[below[0] - 1]]
    return float(s[k]), float(prec[np.nonzero(ok)[0][-1]]), int(tp[k]), float(tau_first)


def found_at(det_score_per_gt, tau):
    """A GT is found if the detection the greedy matching gave it (score >= 0; -1 = unmatched) scores >= tau."""
    d = np.asarray(det_score_per_gt, np.float64)
    return (d >= 0) & (d >= tau)


def four_sets(found_t, found_s):
    return dict(both=found_t & found_s, s_only=found_s & ~found_t, t_only=found_t & ~found_s,
                neither=~found_t & ~found_s)


def track_linked(key, frame, found_t):
    """True where another frame (not this GT's own) holds a teacher-found GT with the same key (sequence, object ID).

    Robust to a key repeated within a frame: it counts distinct (key, frame) pairs with a found GT."""
    key, frame, found_t = np.asarray(key), np.asarray(frame), np.asarray(found_t, bool)
    if len(key) == 0:
        return np.zeros(0, bool)
    kf = np.stack([key[found_t], frame[found_t]], 1)
    kf = np.unique(kf, axis=0) if len(kf) else kf.reshape(0, 2)
    n_frames_found = np.bincount(kf[:, 0], minlength=int(key.max()) + 1) if len(kf) else \
        np.zeros(int(key.max()) + 1, np.int64)
    own = np.zeros(len(key), bool)  # does this GT's own frame hold a found GT of the same key?
    if len(kf):
        own_set = set(map(tuple, kf.tolist()))
        own = np.fromiter(((int(k), int(f)) in own_set for k, f in zip(key, frame)), bool, len(key))
    return (n_frames_found[key] - own.astype(np.int64)) >= 1


def bin_index(rng, npts):
    """Bin id in 0..23 (ring-major: ring * 4 + points bin), or -1 for zero-point GT or GT at >= 75 m."""
    rng, npts = np.asarray(rng, np.float64), np.asarray(npts, np.int64)
    ring = np.searchsorted(np.asarray(RINGS[1:-1], np.float64), rng, side='right')
    pb = np.searchsorted(np.asarray(PTS_EDGES[1:], np.int64), npts, side='right')
    out = ring * len(PTS_NAMES) + pb
    out[(rng >= RINGS[-1]) | (npts < 1)] = -1
    return out


def adjacency(bins, found_t, sets, r_min=R_T_MIN):
    """R-b: teacher found rate per bin, the qualifying bins and each set's share in them (binned GT only)."""
    nb = (len(RINGS) - 1) * len(PTS_NAMES)
    inb = bins >= 0
    n = np.bincount(bins[inb], minlength=nb)
    nf = np.bincount(bins[inb & found_t], minlength=nb)
    r_t = np.where(n > 0, nf / np.maximum(n, 1), np.nan)
    q = np.nan_to_num(r_t, nan=-1.0) >= r_min
    share = {}
    for name, m in sets.items():
        mb = m & inb
        share[name] = float(q[bins[mb]].mean()) if mb.any() else float('nan')
    return r_t, q, share


# ------------------------------------------------------------------------------------------------------ the pass
FRAMES, MODELS = None, None  # set in the parent before the pool forks


def _pass_frame(fi):
    f = FRAMES[fi]
    out = []
    for m in MODELS:
        db, ds = m[fi]
        iou = W.bev_iou(db, f['gv'])
        dt7, gt7, _ = W.greedy_match(iou, ds, 0.7)
        dt5, gt5, _ = W.greedy_match(iou, ds, 0.5)
        di, gi = np.nonzero(iou >= PAIR_MIN_IOU)
        out.append(dict(dt_gt7=dt7, gt_dt7=gt7, gt_dt5=gt5, pd=di.astype(np.int32), pg=gi.astype(np.int32),
                        piou=iou[di, gi].astype(np.float32),
                        best=iou.max(0) if len(db) else np.zeros(len(f['gv']), np.float32)))
    return fi, out


def load_inputs(runs, max_frames=None):
    """Frames in the first result.pkl's order, with Vehicle GT boxes, stored points, object IDs and the pose, plus
    all-class (name, ID) per frame for the ID checks; each model's Car detections per frame."""
    infos = pickle.load(open(W.INFOS, 'rb'))
    by_fid = {}
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
                ids = np.asarray(a['obj_ids']).astype(str) if 'obj_ids' in a else np.full(len(names), '')
                v = names == 'Vehicle'
                frames.append(dict(fid=fid, seq=inf['point_cloud']['lidar_sequence'],
                                   idx=int(inf['point_cloud']['sample_idx']), gv=boxes[v],
                                   gnpts=np.asarray(a['num_points_in_gt'])[v].astype(np.int64),
                                   gid=ids[v], all_names=names.astype(str), all_ids=ids,
                                   pose=np.asarray(inf['pose'], np.float64)))
        assert set(res_by) == set(all_fids) and len(res) == len(all_fids), f'{label}: frame set differs from the first run'
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
    return frames, models


def run_pass(runs, workers, max_frames=None):
    global FRAMES, MODELS
    t0 = time.time()
    FRAMES, MODELS = load_inputs(runs, max_frames)
    print(f'loaded {len(FRAMES)} frames, {len(MODELS)} models in {time.time() - t0:.0f} s', flush=True)
    results = [None] * len(FRAMES)
    with Pool(workers) as pool:
        for n, (fi, out) in enumerate(pool.imap_unordered(_pass_frame, range(len(FRAMES)), chunksize=8)):
            results[fi] = out
            if n % 1000 == 0:
                print(f'  {n}/{len(FRAMES)} frames, {time.time() - t0:.0f} s', flush=True)
    gt_off = np.concatenate([[0], np.cumsum([len(f['gv']) for f in FRAMES])])
    det_off = [np.concatenate([[0], np.cumsum([len(m[fi][1]) for fi in range(len(FRAMES))])]) for m in MODELS]
    seqs = sorted({f['seq'] for f in FRAMES})
    seq_code = {s: i for i, s in enumerate(seqs)}
    gt = dict(frame=np.concatenate([np.full(len(f['gv']), i) for i, f in enumerate(FRAMES)]).astype(np.int64),
              seq=np.concatenate([np.full(len(f['gv']), seq_code[f['seq']]) for f in FRAMES]).astype(np.int64),
              sample_idx=np.concatenate([np.full(len(f['gv']), f['idx']) for f in FRAMES]).astype(np.int64),
              rng=np.concatenate([np.hypot(f['gv'][:, 0], f['gv'][:, 1]) for f in FRAMES]).astype(np.float64),
              npts=np.concatenate([f['gnpts'] for f in FRAMES]),
              obj_id=np.concatenate([f['gid'] for f in FRAMES]),
              boxes=np.concatenate([f['gv'] for f in FRAMES]),
              world_xy=np.concatenate([(f['gv'][:, :3].astype(np.float64) @ f['pose'][:3, :3].T + f['pose'][:3, 3])[:, :2]
                                       for f in FRAMES]))
    labels = [r[0] for r in runs]
    per_model = {}
    for mi, label in enumerate(labels):
        d = {k: [] for k in ('frame', 'score', 'gt7')}
        g = {k: [] for k in ('det7_score', 'det5_score', 'best_iou')}
        p = {k: [] for k in ('det', 'gt', 'iou')}
        for fi in range(len(FRAMES)):
            o = results[fi][mi]
            ds = MODELS[mi][fi][1]
            d['frame'].append(np.full(len(ds), fi))
            d['score'].append(ds)
            d['gt7'].append(np.where(o['dt_gt7'] >= 0, o['dt_gt7'] + gt_off[fi], -1))
            g['det7_score'].append(np.where(o['gt_dt7'] >= 0, ds[np.maximum(o['gt_dt7'], 0)] if len(ds) else 0, -1.0))
            g['det5_score'].append(np.where(o['gt_dt5'] >= 0, ds[np.maximum(o['gt_dt5'], 0)] if len(ds) else 0, -1.0))
            g['best_iou'].append(o['best'])
            p['det'].append(o['pd'].astype(np.int64) + det_off[mi][fi])
            p['gt'].append(o['pg'].astype(np.int64) + gt_off[fi])
            p['iou'].append(o['piou'])
        per_model[label] = dict(det={k: np.concatenate(v) for k, v in d.items()},
                                gt={k: np.concatenate(v).astype(np.float64) for k, v in g.items()},
                                pairs={k: np.concatenate(v) for k, v in p.items()})
    id_checks = check_ids(FRAMES)
    print(f'pass done in {time.time() - t0:.0f} s', flush=True)
    return dict(labels=labels, runs=runs, n_frames=len(FRAMES), fids=[f['fid'] for f in FRAMES], seqs=seqs, gt=gt,
                models=per_model, id_checks=id_checks, max_frames=max_frames)


# ------------------------------------------------------------------------------------------------------ ID checks
def check_ids(frames):
    """Are the object IDs present, unique within a frame, one class per (sequence, ID), and do they follow one object
    (stable box length, plausible world speed between evaluated frames)?"""
    n_gt = sum(len(f['gid']) for f in frames)
    empty = sum(int((f['gid'] == '').sum()) for f in frames)
    dup_frames_all = sum(int(len(f['all_ids']) != len(set(f['all_ids'].tolist()))) for f in frames)
    dup_frames_veh = sum(int(len(f['gid']) != len(set(f['gid'].tolist()))) for f in frames)
    cls = {}
    for f in frames:
        for n, i in zip(f['all_names'], f['all_ids']):
            cls.setdefault((f['seq'], i), set()).add(n)
    multi_class = sum(len(v) > 1 for v in cls.values())
    seq_frames = {}
    for f in frames:
        seq_frames.setdefault(f['seq'], []).append(f['idx'])
    gaps = np.concatenate([np.diff(sorted(v)) for v in seq_frames.values() if len(v) > 1]) if seq_frames else np.zeros(0)
    tracks = {}
    for f in frames:
        wxy = (f['gv'][:, :3].astype(np.float64) @ f['pose'][:3, :3].T + f['pose'][:3, 3])[:, :2] if len(f['gv']) \
            else np.zeros((0, 2))
        for j, i in enumerate(f['gid']):
            tracks.setdefault((f['seq'], i), []).append((f['idx'], float(f['gv'][j, 3]), wxy[j]))
    len_range, speeds, per_track = [], [], []
    for v in tracks.values():
        v.sort(key=lambda x: x[0])
        per_track.append(len(v))
        ls = np.array([x[1] for x in v])
        len_range.append(float(ls.max() - ls.min()))
        for a, b in zip(v[:-1], v[1:]):
            dt = 0.1 * (b[0] - a[0])  # Waymo frames are 10 Hz
            speeds.append(float(np.linalg.norm(b[2] - a[2]) / dt))
    len_range, speeds, per_track = np.asarray(len_range), np.asarray(speeds), np.asarray(per_track)
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float('nan')  # noqa: E731
    return dict(n_vehicle_gt=n_gt, empty_ids=empty, frames_with_duplicate_ids_all_classes=dup_frames_all,
                frames_with_duplicate_vehicle_ids=dup_frames_veh, ids_with_more_than_one_class=multi_class,
                n_ids_all_classes=len(cls), n_vehicle_tracks=len(tracks), n_sequences=len(seq_frames),
                frame_gap_values=sorted(set(gaps.astype(int).tolist()))[:10],
                frame_gap_share_5=float((gaps == 5).mean()) if len(gaps) else float('nan'),
                evaluated_frames_per_track=dict(p50=q(per_track, 50), p90=q(per_track, 90), max=int(per_track.max()) if len(per_track) else 0,
                                                share_single=float((per_track == 1).mean()) if len(per_track) else float('nan')),
                length_range_m=dict(p50=q(len_range, 50), p99=q(len_range, 99), max=float(len_range.max()) if len(len_range) else float('nan')),
                world_speed_mps=dict(n_steps=int(len(speeds)), p50=q(speeds, 50), p99=q(speeds, 99),
                                     max=float(speeds.max()) if len(speeds) else float('nan'),
                                     share_over_40=float((speeds > 40).mean()) if len(speeds) else float('nan')))


# ------------------------------------------------------------------------------------------------------ reference
def compare_with_reference(passed, ref_pass_path, ref_labels):
    """Per-GT equality with 20261017_13's pass.pkl (matched score at IoU 0.7 / 0.5, best BEV IoU), frames matched by ID."""
    ref = pickle.load(open(ref_pass_path, 'rb'))
    p0 = Path(ref['runs'][0][1])
    p0 = p0 if p0.is_absolute() else (TOOLS / p0)
    ref_fids = [r['frame_id'] for r in pickle.load(open(p0, 'rb'))]
    ref_fids = ref_fids[:ref['n_frames']]
    ref_off = np.concatenate([[0], np.cumsum(np.bincount(ref['gt']['frame'], minlength=ref['n_frames']))])
    pos = {fid: i for i, fid in enumerate(ref_fids)}
    my_frame = passed['gt']['frame']
    my_off = np.concatenate([[0], np.cumsum(np.bincount(my_frame, minlength=passed['n_frames']))])
    within = np.arange(len(my_frame)) - my_off[my_frame]
    to_ref = np.array([ref_off[pos[passed['fids'][f]]] for f in my_frame], np.int64) + within
    out = dict(ref_labels=ref['labels'], ref_n_gt=int(len(ref['gt']['rng'])), my_n_gt=int(len(my_frame)),
               rng_max_abs_diff=float(np.abs(ref['gt']['rng'][to_ref] - passed['gt']['rng']).max()),
               npts_equal=bool((ref['gt']['npts'][to_ref] == passed['gt']['npts']).all()))
    for mine, theirs in ref_labels.items():
        a, b = passed['models'][mine]['gt'], ref['models'][theirs]['gt']
        out[mine] = {k: dict(max_abs_diff=float(np.abs(a[k] - b[k][to_ref]).max()),
                             n_differ=int((a[k] != b[k][to_ref]).sum())) for k in ('det7_score', 'det5_score', 'best_iou')}
    return out


# ------------------------------------------------------------------------------------------------------ readings
def best_iou_above(pairs, det_score, n_gt, tau):
    """Per GT: the best BEV IoU among the model's detections with score >= tau (0 if none overlaps >= 0.1)."""
    out = np.zeros(n_gt, np.float64)
    keep = det_score[pairs['det']] >= tau
    np.maximum.at(out, pairs['gt'][keep], pairs['iou'][keep].astype(np.float64))
    return out


def read_pair(passed, teacher, student, op, numbers_t, numbers_s):
    """Sets, R-a, R-b, the localisation split and the per-bin table for one (teacher, student) pair at one operating
    point. Returns a dict of numbers."""
    gt = passed['gt']
    n_gt = len(gt['rng'])
    tau_t, tau_s = numbers_t['tau'][op], numbers_s['tau'][op]
    ft = found_at(passed['models'][teacher]['gt']['det7_score'], tau_t)
    fs = found_at(passed['models'][student]['gt']['det7_score'], tau_s)
    sets = four_sets(ft, fs)
    key = np.unique(np.char.add(np.char.add(gt['seq'].astype(str), '|'), gt['obj_id'].astype(str)), return_inverse=True)[1]
    linked = track_linked(key, gt['frame'], ft)
    has_id = gt['obj_id'] != ''
    bins = bin_index(gt['rng'], gt['npts'])
    r_t, q, share = adjacency(bins, ft, sets)
    out = dict(tau_t=tau_t, tau_s=tau_s, n_gt=n_gt, found_t=int(ft.sum()), found_s=int(fs.sum()),
               sizes={k: int(v.sum()) for k, v in sets.items()},
               L={k: float(linked[v].mean()) if v.any() else float('nan') for k, v in sets.items()},
               n_linked={k: int(linked[v].sum()) for k, v in sets.items()},
               ids_missing_in_sets={k: int((v & ~has_id).sum()) for k, v in sets.items()},
               share_Q=share, r_t=[None if np.isnan(x) else float(x) for x in r_t], Q=q.tolist())
    out['Ra'] = dict(L_s_only=out['L']['s_only'], L_neither=out['L']['neither'],
                     diff=out['L']['s_only'] - out['L']['neither'],
                     supported=bool(out['L']['s_only'] >= RA_L_MIN and out['L']['s_only'] - out['L']['neither'] >= RA_DIFF_MIN))
    out['Rb'] = dict(share_s_only=share['s_only'], share_neither=share['neither'],
                     diff=share['s_only'] - share['neither'], supported=bool(share['s_only'] - share['neither'] >= RB_DIFF_MIN))
    # sensitivity (reported only): the unbinned GT (zero points, >= 75 m) kept in the denominators, never in Q
    out['Rb_all_in_denominator'] = {k: float(((bins >= 0) & q[np.maximum(bins, 0)] & v).sum() / max(v.sum(), 1))
                                    for k, v in sets.items()}
    out['unbinned'] = {k: dict(zero_points=int((v & (gt['npts'] < 1)).sum()),
                               beyond_75m=int((v & (gt['rng'] >= RINGS[-1]) & (gt['npts'] >= 1)).sum()))
                       for k, v in sets.items()}
    # linked share per set, for GT in Q and outside Q (reported)
    inq = (bins >= 0) & q[np.maximum(bins, 0)]
    out['L_by_Q'] = {k: dict(in_Q=float(linked[v & inq].mean()) if (v & inq).any() else float('nan'),
                             outside_Q=float(linked[v & ~inq].mean()) if (v & ~inq).any() else float('nan'))
                     for k, v in sets.items()}
    # localisation split of S-only, at the teacher's operating point
    pt = passed['models'][teacher]
    best_t = best_iou_above(pt['pairs'], pt['det']['score'], n_gt, tau_t)
    so = sets['s_only']
    loc = so & (best_t >= 0.5) & (best_t <= 0.7)
    taken = so & (best_t > 0.7)
    none = so & (best_t < 0.5)
    d7 = pt['gt']['det7_score']
    best_any = pt['gt']['best_iou']
    out['split'] = dict(localisation=int(loc.sum()), taken_by_neighbour=int(taken.sum()), none=int(none.sum()),
                        none_teacher_matched_below_tau=int((none & (d7 >= 0) & (d7 < tau_t)).sum()),
                        none_teacher_box_05_07_below_tau=int((none & ~((d7 >= 0) & (d7 < tau_t)) & (best_any >= 0.5)).sum()),
                        none_no_teacher_box_05=int((none & ~((d7 >= 0) & (d7 < tau_t)) & (best_any < 0.5)).sum()))
    # per-bin table
    rows = []
    fs_rate = []
    for b in range(len(r_t)):
        m = bins == b
        rows.append(dict(bin=b, ring=RING_NAMES[b // len(PTS_NAMES)], pts=PTS_NAMES[b % len(PTS_NAMES)], n=int(m.sum()),
                         r_t=None if np.isnan(r_t[b]) else float(r_t[b]),
                         r_s=float(fs[m].mean()) if m.any() else None, Q=bool(q[b]),
                         **{k: int((v & m).sum()) for k, v in sets.items()},
                         L_s_only=float(linked[so & m].mean()) if (so & m).any() else None))
        fs_rate.append(rows[-1]['r_s'])
    for name, m in (('0 points', gt['npts'] < 1), ('>= 75 m, >= 1 point', (gt['rng'] >= RINGS[-1]) & (gt['npts'] >= 1))):
        rows.append(dict(bin=-1, ring=name, pts='', n=int(m.sum()), r_t=float(ft[m].mean()) if m.any() else None,
                         r_s=float(fs[m].mean()) if m.any() else None, Q=None,
                         **{k: int((v & m).sum()) for k, v in sets.items()},
                         L_s_only=float(linked[so & m].mean()) if (so & m).any() else None))
    out['bins'] = rows
    # T-only by ring (reported)
    ring = np.where(gt['rng'] < RINGS[-1], np.searchsorted(np.asarray(RINGS[1:-1], np.float64), gt['rng'], side='right'), -1)
    out['by_ring'] = {k: [int((v & (ring == r)).sum()) for r in range(len(RINGS) - 1)] + [int((v & (ring < 0)).sum())]
                      for k, v in sets.items()}
    return out


def model_numbers(passed, label):
    """Greedy AP_R40 / max recall (reproduction) and the operating points."""
    gt = passed['gt']
    n_gt = len(gt['rng'])
    m = passed['models'][label]
    tp7 = m['det']['gt7'] >= 0
    s7 = W.ap_split(m['det']['score'], tp7, n_gt)
    found5 = m['gt']['det5_score'] >= 0  # max recall at IoU 0.5 (a reproduction check only)
    n = dict(ap7=s7['ap'], ap7_c=s7['ap_c'], recall7=s7['recall'], recall5=float(found5.mean()),
             n_det=int(len(m['det']['score'])), tau={}, precision_at_tau={}, recall_at_tau={}, tau_first={})
    for op, kind, v in OPERATING_POINTS:
        if kind == 'precision':
            tau, prec, ntp, tau_first = tau_at_precision(m['det']['score'], tp7, v)
        else:
            tau = v
            sel = m['det']['score'] >= v
            prec, ntp, tau_first = float(tp7[sel].mean()) if sel.any() else float('nan'), int(tp7[sel].sum()), v
        n['tau'][op], n['precision_at_tau'][op], n['tau_first'][op] = tau, prec, tau_first
        n['recall_at_tau'][op] = ntp / n_gt
        assert ntp == int(found_at(m['gt']['det7_score'], tau).sum()), 'TPs above tau != GT found at tau'
    ring = np.where(gt['rng'] < RINGS[-1], np.searchsorted(np.asarray(RINGS[1:-1], np.float64), gt['rng'], side='right'), -1)
    f_any = m['gt']['det7_score'] >= 0
    n['found_any_by_ring'] = [int((f_any & (ring == r)).sum()) for r in range(len(RINGS) - 1)]
    n['found_any'] = int(f_any.sum())
    return n


# -------------------------------------------------------------------------------------------------------- tables
def summarise(passed, ref_check=None):
    labels = passed['labels']
    teacher, students = labels[0], labels[1:]
    num = dict(n_frames=passed['n_frames'], n_gt=int(len(passed['gt']['rng'])), id_checks=passed['id_checks'],
               reference=ref_check, models={}, pairs={})
    for lab in labels:
        num['models'][lab] = model_numbers(passed, lab)
    for st in students:
        num['pairs'][st] = {op: read_pair(passed, teacher, st, op, num['models'][teacher], num['models'][st])
                            for op, _, _ in OPERATING_POINTS}
    lines = []

    def table(head, rows):
        lines.append('| ' + ' | '.join(head) + ' |')
        lines.append('|' + '---|' * len(head))
        lines.extend('| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows)
        lines.append('')

    lines.append(f'frames {num["n_frames"]}, Vehicle GT {num["n_gt"]}')
    lines.append('ID checks: ' + json.dumps(passed['id_checks']))
    if ref_check:
        lines.append('reference (20261017_13 pass.pkl): ' + json.dumps(ref_check))
    lines.append('')
    lines.append('### Reproduction and operating points (greedy matching, IoU 0.7)')
    table(['model', 'AP_R40 0.7', 'AP_c 0.7', 'max recall 0.7', 'max recall 0.5', 'found (any score) by ring'] +
          [f'{op}: tau / precision / recall (tau_first)' for op, _, _ in OPERATING_POINTS],
          [[lab, '%.2f' % n['ap7'], '%.2f' % n['ap7_c'], '%.4f' % n['recall7'], '%.4f' % n['recall5'],
            ' / '.join(str(x) for x in n['found_any_by_ring'])] +
           ['%.4f / %.4f / %.4f (%.4f)' % (n['tau'][op], n['precision_at_tau'][op], n['recall_at_tau'][op], n['tau_first'][op])
            for op, _, _ in OPERATING_POINTS] for lab in labels for n in [num['models'][lab]]])
    for st in students:
        for op, _, _ in OPERATING_POINTS:
            r = num['pairs'][st][op]
            lines.append(f'### {teacher} vs {st} at {op}: tau_T {r["tau_t"]:.4f}, tau_S {r["tau_s"]:.4f}')
            table(['set', 'n', 'L (linked share)', 'linked', 'share in Q (binned)', 'share in Q (all in denominator)',
                   'zero-point', '>= 75 m', 'L in Q', 'L outside Q', 'IDs missing'],
                  [[k, r['sizes'][k], '%.4f' % r['L'][k], r['n_linked'][k], '%.4f' % r['share_Q'][k],
                    '%.4f' % r['Rb_all_in_denominator'][k], r['unbinned'][k]['zero_points'], r['unbinned'][k]['beyond_75m'],
                    '%.4f' % r['L_by_Q'][k]['in_Q'], '%.4f' % r['L_by_Q'][k]['outside_Q'], r['ids_missing_in_sets'][k]]
                   for k in ('both', 's_only', 't_only', 'neither')])
            lines.append(f'R-a: L(S-only) {r["Ra"]["L_s_only"]:.4f}, L(neither) {r["Ra"]["L_neither"]:.4f}, '
                         f'diff {r["Ra"]["diff"]:+.4f} -> supported {r["Ra"]["supported"]}')
            lines.append(f'R-b: share(S-only) {r["Rb"]["share_s_only"]:.4f}, share(neither) {r["Rb"]["share_neither"]:.4f}, '
                         f'diff {r["Rb"]["diff"]:+.4f} -> supported {r["Rb"]["supported"]}')
            lines.append('S-only split: ' + json.dumps(r['split']))
            lines.append('by ring (0-10 ... 50-75, >= 75): ' + json.dumps(r['by_ring']))
            lines.append('')
            table(['ring', 'points', 'GT', 'r_T', 'r_S', 'Q', 'both', 'S-only', 'T-only', 'neither', 'L(S-only)'],
                  [[b['ring'], b['pts'], b['n'], '-' if b['r_t'] is None else '%.3f' % b['r_t'],
                    '-' if b['r_s'] is None else '%.3f' % b['r_s'], '-' if b['Q'] is None else ('yes' if b['Q'] else 'no'),
                    b['both'], b['s_only'], b['t_only'], b['neither'],
                    '-' if b['L_s_only'] is None else '%.3f' % b['L_s_only']] for b in r['bins']])
    return '\n'.join(lines), num


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='<label>=<result.pkl>; the first is the teacher, the second the student')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_frames', type=int, default=None, help='first N frames only (smoke test)')
    ap.add_argument('--report_only', action='store_true')
    ap.add_argument('--ref_pass', default=None, help="20261017_13's pass.pkl, for the per-GT equality check")
    ap.add_argument('--ref_labels', default='', help='mine=theirs,... label map into --ref_pass')
    a = ap.parse_args()
    runs = [tuple(r.split('=', 1)) for r in a.runs]
    assert len(runs) >= 2, 'need a teacher and a student'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.report_only:
        passed = pickle.load(open(out / 'pass.pkl', 'rb'))
    else:
        passed = run_pass(runs, a.workers, a.max_frames)
        pickle.dump(passed, open(out / 'pass.pkl', 'wb'))
    ref_check = None
    if a.ref_pass and not a.max_frames:
        ref_check = compare_with_reference(passed, a.ref_pass, dict(x.split('=', 1) for x in a.ref_labels.split(',') if x))
        print('reference check: ' + json.dumps(ref_check), flush=True)
    text, num = summarise(passed, ref_check)
    (out / 'tables.md').write_text(text + '\n')
    json.dump(num, open(out / 'numbers.json', 'w'), indent=1, default=float)
    print(text, flush=True)


if __name__ == '__main__':
    main()
