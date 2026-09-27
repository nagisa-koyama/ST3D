"""Per-class COUNT-BALANCE cut from temporal persistence (label-free), with the GT check (valid.).

    python count_balance_from_persistence.py <persistence.pkl> [p_real_min=0.6] [fp_lo=0.10] [fp_hi=0.12]

For each class: p_real = persistence of boxes with score >= p_real_min, p_fp = persistence of boxes in
[fp_lo, fp_hi) (just above the plateau, taken as clutter); the real share at cut t is
pi(t) = (p(t) - p_fp) / (p_real - p_fp); N_est = pi(0.1) * kept(0.1) per frame; the count-balance cut is
where kept(t) per frame falls to N_est. Boxes within 70 m (persistence.py's RMAX). The GT column
uses the per-frame GT count persistence.py stored for validation only.
"""
import pickle
import sys

import numpy as np

CUTS = np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.22, 0.24, 0.26, 0.28,
                 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80])
NAMES = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}


def cross(x, y, target):
    for i in range(1, len(x)):
        a, b = y[i - 1], y[i]
        if (a - target) * (b - target) <= 0 and a != b:
            return x[i - 1] + (target - a) * (x[i] - x[i - 1]) / (b - a)
    return np.nan


PE = pickle.load(open(sys.argv[1], 'rb'))
p_real_min = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
fp_lo, fp_hi = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.10, 0.12)
gtm = PE[PE[:, 1] < 0]
frames = len(np.unique(gtm[:, 5]))
print('%d anchor frames; p_real from score >= %.2f, p_fp from [%.2f, %.2f)' % (frames, p_real_min, fp_lo, fp_hi))
print('%-10s | %6s %6s | %8s %8s | %8s | %8s %8s' % ('class', 'p_real', 'p_fp', 'N_est/fr', 'GT/fr', 'pi(0.1)', 't_count', 't_gt'))
res = {}
for c, name in NAMES.items():
    b = PE[(PE[:, 0] == c) & (PE[:, 1] >= 0)]
    if len(b) < 100:
        continue
    p_real = b[b[:, 1] >= p_real_min, 3].mean()
    p_fp = b[(b[:, 1] >= fp_lo) & (b[:, 1] < fp_hi), 3].mean()
    kept = np.array([(b[:, 1] >= t).sum() / frames for t in CUTS])
    pers = np.array([b[b[:, 1] >= t, 3].mean() if (b[:, 1] >= t).any() else np.nan for t in CUTS])
    pi = np.clip((pers - p_fp) / (p_real - p_fp), 0, 1) if p_real > p_fp else np.full(len(CUTS), np.nan)
    n_est = pi[0] * kept[0]
    gt = gtm[gtm[:, 0] == c][:, 4].mean()
    t_c, t_g = cross(CUTS, kept, n_est), cross(CUTS, kept, gt)
    res[name] = t_c
    print('%-10s | %6.3f %6.3f | %8.2f %8.2f | %8.2f | %8.3f %8.3f' % (name, p_real, p_fp, n_est, gt, pi[0], t_c, t_g))
print('count-balance cuts [Car, Pedestrian, Cyclist]:', [round(res.get(n, float('nan')), 3) for n in NAMES.values()])
