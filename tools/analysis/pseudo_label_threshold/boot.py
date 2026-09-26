import os
import pickle, sys, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('h', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness.py')); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
rng = np.random.default_rng(0)
for c in [1, 2]:
    pe = h.PE[(h.PE[:, 0] == c) & (h.PE[:, 1] >= 0)]
    fr = np.unique(pe[:, 5]); by = {f: np.where(pe[:, 5] == f)[0] for f in fr}
    i1, i3 = [], []
    for b in range(20 if h.bins.FULL else 100):   # full range: the exponential mixture (I3) is degenerate, keep only I1 and fewer draws
        idx = np.concatenate([by[f] for f in rng.choice(fr, len(fr))]); x = pe[idx]
        pi, _, _ = h.persistence_unmix(x[:, 1], x[:, 3]); i1.append(h.count_balance(x[:, 1], pi.sum()))
        if h.bins.FULL: i3.append(np.nan); continue
        w, *_ = h.trunc_exp_mix(x[:, 1]); i3.append(h.count_balance(x[:, 1], w * len(x)))
    print('%s  I1 5-95%%: %.2f-%.2f (median %.2f)   I3 5-95%%: %.2f-%.2f (median %.2f)' % (h.NAMES[c], *np.percentile(i1, [5, 95]), np.median(i1), *np.percentile(i3, [5, 95]), np.median(i3)))
