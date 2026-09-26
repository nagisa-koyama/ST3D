"""Is a low-threshold pseudo-label file the superset of a higher-threshold one from the same teacher?

    python compare_ps_files.py <full file> <truncated file> <truncation threshold>

Restricts the full file to score >= threshold and compares frame by frame: box counts, and the
largest centre/score discrepancy after sorting both sets by score. Same teacher, same frames,
deterministic eval-mode inference -> should match to float precision; anything else means the
generation path differs from the training run's.
"""
import pickle, sys
import numpy as np

full = pickle.load(open(sys.argv[1], 'rb')); trunc = pickle.load(open(sys.argv[2], 'rb')); t = float(sys.argv[3])
missing = [k for k in trunc if k not in full]
print('frames: full %d, truncated %d, truncated frames missing from full: %d' % (len(full), len(trunc), len(missing)))
n_full = n_tr = n_eq = 0; worst_c = worst_s = 0.0; per_frame_all = []
for k, v in trunc.items():
    a = v['gt_boxes']; b = full[k]['gt_boxes']; b = b[b[:, 8] >= t]
    per_frame_all.append(len(full[k]['gt_boxes']))
    n_full += len(b); n_tr += len(a)
    if len(a) == len(b):
        a = a[np.argsort(-a[:, 8])]; b = b[np.argsort(-b[:, 8])]
        worst_c = max(worst_c, np.abs(a[:, :3] - b[:, :3]).max() if len(a) else 0)
        worst_s = max(worst_s, np.abs(a[:, 8] - b[:, 8]).max() if len(a) else 0)
        n_eq += 1
print('boxes >= %g: full %d, truncated %d; frames with equal counts %d / %d' % (t, n_full, n_tr, n_eq, len(trunc)))
print('worst centre |diff| %.5f m, worst score |diff| %.6f (over equal-count frames)' % (worst_c, worst_s))
b = np.concatenate([v['gt_boxes'] for v in full.values() if len(v['gt_boxes'])])
print('full file: %.1f boxes/frame (max %d), score min %.6f, share below 0.1: %.3f, share below 0.01: %.3f'
      % (np.mean(per_frame_all), max(per_frame_all), b[:, 8].min(), (b[:, 8] < 0.1).mean(), (b[:, 8] < 0.01).mean()))
for c, cn in [(1, 'Car'), (2, 'Ped'), (3, 'Cyc')]:
    s = b[b[:, 7] == c, 8]
    print('  %-4s %7d boxes, quantiles 1/10/50/90%%: %s' % (cn, len(s), np.round(np.quantile(s, [.01, .1, .5, .9]), 4)))
