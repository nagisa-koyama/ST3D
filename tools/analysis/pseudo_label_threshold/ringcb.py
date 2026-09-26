import os
import pickle, sys, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('h', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness.py')); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
R = [0, 10, 20, 30, 40, 50, 70]
for c in [1, 2]:
    pe = h.PE[(h.PE[:, 0] == c) & (h.PE[:, 1] >= 0)]; s, r, p = pe[:, 1], pe[:, 2], pe[:, 3]
    nf = len(np.unique(pe[:, 5])); g = h.GT[h.GT[:, 1] == c]
    print('\n%s ring | VALID GT/fr  t_cb_ring | label-free n_true/fr  t_ring | kept/fr at 0.17 0.20' % h.NAMES[c])
    for lo, hi in zip(R[:-1], R[1:]):
        m = (r >= lo) & (r < hi); sm, pm = s[m], p[m]
        gcount = ((g[:, 3] >= lo) & (g[:, 3] < hi)).sum() / 1000 * nf
        q = np.quantile(sm, 0.8)
        pi, P0, P1 = h.persistence_unmix(sm, pm, hi=(max(q, 0.4), 1.01))
        print('  %2d-%-2d | %6.2f  %.2f | %6.2f  %.2f | %.2f %.2f' % (lo, hi, gcount / nf, h.count_balance(sm, gcount), pi.sum() / nf, h.count_balance(sm, pi.sum()), (sm >= .17).sum() / nf, (sm >= .20).sum() / nf))
