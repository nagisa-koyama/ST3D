"""Apply a row's pseudo-label cuts to a score-floor (0.0001) teacher pass, as the training run will at its first pass.

Exists so the first-pass label counts a pseudo-labelling row will log (`generate_ps_e0` pos_ps_box / ign_ps_box) are
known BEFORE it is launched, and so `target_dump.py` can check the CenterHead targets of the real cut labels
(experiments_md 20261017_06 §3, §5). The arithmetic is self_training_utils.save_pseudo_label_batch's:
  - NEG_THRESH: a box with score < NEG_THRESH[class] is dropped;
  - SCORE_THRESH: a kept box with score < SCORE_THRESH[class] gets the NEGATIVE label (ignore band).
A frozen single-pass teacher has no memory at its first pass, so the stored labels are exactly these.

Cuts come either explicitly (--score / --neg) or from the rule outputs (--rule_b_log: the `cuts: {...}` line of
source_precision_cut.py; --rule_a_log: the `NEG_THRESH: {...}` line of plateau_neg_thresh.py). Rule A step 6 is applied
here: a SCORE_THRESH below its NEG_THRESH is raised to it. Values are rounded as the config will carry them (3 decimals).

    python apply_pseudo_label_cuts.py <ps_label_e0.pkl at 0.0001> --rule_b_log b.txt --rule_a_log a.txt --out cut.pkl
"""
import argparse
import ast
import pickle

import numpy as np

CLASSES = ['Car', 'Pedestrian', 'Cyclist']


def last_dict(path, prefix):
    found = None
    for line in open(path):
        if line.startswith(prefix):
            found = ast.literal_eval(line[len(prefix):].strip())
    assert found is not None, 'no line starting with %r in %s' % (prefix, path)
    return found


def apply_cuts(boxes, score, neg):
    """boxes (M, 9) [x y z dx dy dz yaw label score], labels 1..C -> the stored (M', 9) array."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 9).copy()
    if len(b) == 0:
        return b
    lab = b[:, 7].astype(np.int64)
    assert (lab >= 1).all() and (lab <= len(score)).all(), 'a score-floor pass must hold labels 1..C only, got %s' % (
        np.unique(lab),)
    keep = b[:, 8] >= np.asarray(neg)[lab - 1]
    b, lab = b[keep], lab[keep]
    ign = b[:, 8] < np.asarray(score)[lab - 1]
    b[ign, 7] = -lab[ign]
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ps_label')
    ap.add_argument('--score', type=float, nargs=3)
    ap.add_argument('--neg', type=float, nargs=3)
    ap.add_argument('--rule_b_log')
    ap.add_argument('--rule_a_log')
    ap.add_argument('--max_obj', type=int, default=500, help="the teacher head's MAX_OBJ_PER_SAMPLE (rank cap)")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    score = args.score or [last_dict(args.rule_b_log, 'cuts:')[c] for c in CLASSES]
    neg = args.neg or [last_dict(args.rule_a_log, 'NEG_THRESH:')[c] for c in CLASSES]
    raised = [s < n for s, n in zip(score, neg)]
    score = [round(max(s, n), 3) for s, n in zip(score, neg)]
    neg = [round(n, 3) for n in neg]
    print('rule B SCORE_THRESH (after rule A step 6):', score, '| raised to NEG:', [c for c, r in zip(CLASSES, raised) if r])
    print('rule A NEG_THRESH:', neg)
    print('config lines:  SCORE_THRESH: %s   NEG_THRESH: %s' % (score, neg))

    ps = pickle.load(open(args.ps_label, 'rb'))
    n_frames = len(ps)
    per_frame_raw = np.array([len(np.asarray(v['gt_boxes']).reshape(-1, 9)) for v in ps.values()])
    pos, ign, out = np.zeros(3), np.zeros(3), {}
    for fid, v in ps.items():
        b = apply_cuts(v['gt_boxes'], score, neg)
        lab = b[:, 7].astype(np.int64) if len(b) else np.zeros(0, np.int64)
        for c in range(3):
            pos[c] += (lab == c + 1).sum(); ign[c] += (lab == -(c + 1)).sum()
        if args.out:
            e = dict(v)
            e['gt_boxes'] = b.astype(np.asarray(v['gt_boxes']).dtype if len(v['gt_boxes']) else np.float32)
            e['memory_counter'] = np.zeros(len(b))
            for k in ('cls_scores', 'iou_scores'):
                if k in e:
                    e[k] = b[:, 8].copy()
            e.pop('teacher_classes', None)
            out[fid] = e
    print(f'teacher pass: {n_frames} frames, {per_frame_raw.mean():.1f} boxes per frame at the floor; '
          f'frames at the rank cap ({args.max_obj}): {int((per_frame_raw >= args.max_obj).sum())}')
    print('per frame after the cuts (the first-pass pos_ps_box / ign_ps_box the row should log):')
    for c, name in enumerate(CLASSES):
        print(f'  {name:10s} positives {pos[c] / n_frames:7.3f}   ignore band {ign[c] / n_frames:7.3f}')
    print('pos_ps_box=(%s) ign_ps_box=(%s)' % (','.join('%.3f' % (p / n_frames) for p in pos),
                                               ','.join('%.3f' % (i / n_frames) for i in ign)))
    if args.out:
        with open(args.out, 'wb') as f:
            pickle.dump(out, f)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
