"""Label-free NEG_THRESH per class: the upper half-maximum of the background plateau of the teacher's scores.

Pre-declared in experiments_md/20261004_01 §5e (user's decision A, 2026-10-05), before any score of the
teacher it is applied to was read. It replaces NEG_THRESH 0.1, which had been selected on KITTI-val results
in 2023. Background: 20260927_01 - at a 0.0001 floor the score marginal's dominant mode is the head's
background plateau (the sigmoid of its background logit), and true detections are a minority shoulder above.

  1. Input: the teacher's pseudo-labels on the target TRAIN split at score floor 0.0001 (MAX_OBJ_PER_SAMPLE
     as configured), every frame, boxes with centre inside the detection range |x|, |y| <= 75.2 m, per class.
  2. x = logit(score); histogram with bins 0.25 wide on [-10, 10].
  3. Search region: bins whose centre lies below the class's 90th percentile of x (true detections are a
     minority of a 0.0001-floor file, so the plateau lies in the lower 90%).
  4. Plateau mode m: the highest-count bin in the search region (tie: the lower bin).
  5. NEG_THRESH = sigmoid of the first point above m where the count falls to <= 0.5 x count(m), linearly
     interpolated in logit between the bracketing bin centres. If it never falls that far below the 90th
     percentile, NEG_THRESH = sigmoid(90th percentile).
  6. If a class's SCORE_THRESH from the cut rule is below its NEG_THRESH, SCORE_THRESH is raised to NEG_THRESH
     (no ignore band for that class).

    python plateau_neg_thresh.py <ps_label_e0.pkl on target train> [--classes Car Pedestrian Cyclist]
"""
import argparse
import pickle

import numpy as np

RANGE = 75.2
EDGES = np.arange(-10.0, 10.0 + 1e-9, 0.25)
CENTRES = (EDGES[:-1] + EDGES[1:]) / 2


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def plateau_neg(scores):
    x = logit(scores)
    counts, _ = np.histogram(x, EDGES)
    p90 = np.percentile(x, 90)
    region = np.where(CENTRES < p90)[0]
    m = region[np.argmax(counts[region])]  # argmax returns the first (lower) bin on a tie
    half = 0.5 * counts[m]
    for j in range(m + 1, len(CENTRES)):
        if CENTRES[j] >= p90:
            break
        if counts[j] <= half:
            c0, c1 = counts[j - 1], counts[j]
            t = (c0 - half) / (c0 - c1) if c0 != c1 else 0.0
            return float(sigmoid(CENTRES[j - 1] + t * (CENTRES[j] - CENTRES[j - 1]))), float(sigmoid(CENTRES[m])), counts, m
    return float(sigmoid(p90)), float(sigmoid(CENTRES[m])), counts, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ps_label')
    ap.add_argument('--classes', nargs='+', default=['Car', 'Pedestrian', 'Cyclist'])
    args = ap.parse_args()
    ps = pickle.load(open(args.ps_label, 'rb'))
    b = np.concatenate([np.asarray(v['gt_boxes']).reshape(-1, 9) for v in ps.values()])
    b = b[(np.abs(b[:, 0]) <= RANGE) & (np.abs(b[:, 1]) <= RANGE)]
    out = {}
    for ci, name in enumerate(args.classes, start=1):
        s = b[np.abs(b[:, 7]) == ci, 8]
        neg, mode, counts, m = plateau_neg(s)
        out[name] = round(neg, 3)
        print(f'{name:10s} boxes {len(s):9d}  plateau mode {mode:.3f} (bin count {counts[m]})  '
              f'upper half-maximum -> NEG_THRESH {neg:.3f}')
    print('NEG_THRESH:', out)


if __name__ == '__main__':
    main()
