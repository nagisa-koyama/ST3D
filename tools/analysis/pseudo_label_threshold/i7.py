import os
import pickle, sys, numpy as np, importlib.util

spec = importlib.util.spec_from_file_location('h', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness.py')); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
R = [0, 10, 20, 30, 40, 50, 70]
for c in [1, 2]:
    pe = h.PE[(h.PE[:, 0] == c) & (h.PE[:, 1] >= 0)]; s, r, p, tp = pe[:, 1], pe[:, 2], pe[:, 3], pe[:, 4] > 0
    gcnt = h.PE[(h.PE[:, 0] == c) & (h.PE[:, 1] < 0), 4].sum()
    print('\n%s   ring: n  P0(lowest slice)  P1(top quintile of score in ring)  [VALID P(pers|FP) P(pers|TP)]  n_true est  [VALID TP count]' % h.NAMES[c])
    ntot = 0
    for lo, hi in zip(R[:-1], R[1:]):
        m = (r >= lo) & (r < hi); sm, pm = s[m], p[m]
        P0 = pm[sm < h.bins.BOTTOM[1]].mean() if (sm < h.bins.BOTTOM[1]).sum() > 20 else np.nan
        q = np.quantile(sm, 0.8); P1 = pm[sm >= max(q, 0.4)].mean() if (sm >= max(q, 0.4)).sum() > 20 else np.nan
        pi, *_ = h.persistence_unmix(sm, pm, lo=h.bins.BOTTOM, hi=(max(q, 0.4), 1.01)) if np.isfinite(P1) else (np.full(m.sum(), np.nan),)
        nt = np.nansum(pi); ntot += nt
        print('  %2d-%-2d %6d  %.3f  %.3f   [%.3f %.3f]   %7.0f  [%5d]' % (lo, hi, m.sum(), P0, P1, pm[~tp[m]].mean(), pm[tp[m]].mean() if tp[m].any() else np.nan, nt, tp[m].sum()))
    print('  total n_true %.0f -> I7 threshold %.2f  [VALID GT count %d -> t_cb %.2f, TP count %d]' % (ntot, h.count_balance(s, ntot), gcnt, h.count_balance(s, gcnt), tp.sum()))
