"""Label-free Pedestrian / Cyclist pseudo-label cut from the teacher's precision on the labelled SOURCE val.

The rule was pre-declared in experiments_md/20261004_01 §5c (2026-10-05) before any target result was
read. It replaces the S2 rows' Pedestrian / Cyclist cut 0.6, which had been chosen against target labels.
Since the user's decision B (2026-10-05, 20261004_01 §5e) the SAME rule applies to Car in new method rows;
count balance is no longer the Car rule. `--classes` selects the classes (default: all three).

  1. Teacher predictions: the teacher AS TRAINED (source BatchNorm statistics), run on the labelled source
     val split with its training input (sweep count), eval-mode view, score floor 0.0001. The predictions
     come from generate_pseudo_labels.py on a flat config whose target slot is the source val split
     (make_teacher_on_source_val_cfg.py); this script only reads the resulting ps_label_e0.pkl.
  2. Ground truth: source val boxes of the class, with >= 1 lidar point and centre inside the detection
     range (|x|, |y| <= 75.2 m). Predictions are restricted to the same range.
  3. Matching: per frame and class, predictions in descending score, each to the nearest unmatched GT of
     the same class with BEV centre distance <= 1.0 m. Unmatched predictions are false positives.
  4. Precision P(t) = matched / kept over predictions with score >= t, on t = 0.10, 0.11, ..., 0.95.
  5. Cut = the smallest grid t with P(t') >= 0.50 for every grid t' >= t that keeps >= 50 predictions.
     If none exists, the class is gated off: cut 1.01 (no positive pseudo-labels; its boxes between
     NEG_THRESH and the cut fall into the ignore band).

    python source_precision_cut.py <ps_label_e0.pkl on source val> [--infos <nuScenes val infos pkl>]
"""
import argparse
import pickle

import numpy as np

ALL_CLASSES = {1: ('Car', 'car'), 2: ('Pedestrian', 'pedestrian'), 3: ('Cyclist', 'bicycle')}  # model index, nuScenes name
RANGE, MATCH_DIST, P_STAR, MIN_KEPT = 75.2, 1.0, 0.50, 50
GRID = np.round(np.arange(0.10, 0.951, 0.01), 2)


def frame_id(info):
    return info['lidar_path'].split('/')[-1][:-4]


def match(pred_xy, scores, gt_xy):
    """Greedy by score; returns a bool per prediction (matched to a distinct GT within MATCH_DIST)."""
    hit = np.zeros(len(pred_xy), bool)
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return hit
    used = np.zeros(len(gt_xy), bool)
    for i in np.argsort(-scores):
        d = np.hypot(*(gt_xy - pred_xy[i]).T)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= MATCH_DIST:
            used[j] = True
            hit[i] = True
    return hit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ps_label')
    ap.add_argument('--infos', default='../data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl')
    ap.add_argument('--classes', nargs='+', default=['Car', 'Pedestrian', 'Cyclist'])
    ap.add_argument('--extra_ps', nargs='*', default=[],
                    help='further ps_label files merged in (e.g. per-platform passes at different sweep counts)')
    args = ap.parse_args()
    CLASSES = {k: v for k, v in ALL_CLASSES.items() if v[0] in args.classes}
    ps = pickle.load(open(args.ps_label, 'rb'))
    for extra in args.extra_ps:
        more = pickle.load(open(extra, 'rb'))
        assert not set(more) & set(ps), 'the merged passes overlap'
        ps.update(more)
    infos = pickle.load(open(args.infos, 'rb'))
    rows = {c: [] for c in CLASSES}  # (score, matched)
    used_frames = 0
    for info in infos:
        fid = frame_id(info)
        if fid not in ps:
            continue
        used_frames += 1
        boxes = np.asarray(ps[fid]['gt_boxes']).reshape(-1, 9)
        names = np.asarray(info['gt_names'])
        gtb = np.asarray(info['gt_boxes'])
        npts = np.asarray(info.get('num_lidar_pts', np.ones(len(names))))
        for c, (_, nus) in CLASSES.items():
            g = gtb[(names == nus) & (npts >= 1)]
            g = g[(np.abs(g[:, 0]) <= RANGE) & (np.abs(g[:, 1]) <= RANGE)] if len(g) else g
            p = boxes[np.abs(boxes[:, 7]) == c]
            p = p[(np.abs(p[:, 0]) <= RANGE) & (np.abs(p[:, 1]) <= RANGE)]
            hit = match(p[:, :2], p[:, 8], g[:, :2] if len(g) else np.zeros((0, 2)))
            rows[c] += list(zip(p[:, 8], hit))
    print(f'source val frames read: {used_frames} of {len(infos)}')
    cuts = {}
    for c, (name, _) in CLASSES.items():
        a = np.array(rows[c], dtype=float).reshape(-1, 2)
        prec, kept = [], []
        for t in GRID:
            k = a[a[:, 0] >= t]
            kept.append(len(k))
            prec.append(k[:, 1].mean() if len(k) else np.nan)
        prec, kept = np.array(prec), np.array(kept)
        valid = kept >= MIN_KEPT
        cut = 1.01
        for i, t in enumerate(GRID):
            later = valid[i:]
            if valid[i] and np.all(prec[i:][later] >= P_STAR):
                cut = float(t)
                break
        cuts[name] = cut
        show = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        curve = '  '.join(f'{t:.2f}:{prec[GRID == t][0]:.2f}({kept[GRID == t][0]})' for t in show)
        print(f'{name:10s} precision(kept) at t: {curve}')
        print(f'{name:10s} cut = {cut:.2f}' + ('  (gated off: precision never holds >= 0.50)' if cut > 1 else ''))
    print('cuts:', cuts)


if __name__ == '__main__':
    main()
