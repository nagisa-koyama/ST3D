import pickle, sys, numpy as np
a = pickle.load(open(sys.argv[1], 'rb'))   # class, score, npts, range, tp, frame
C = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
rank = lambda x: np.argsort(np.argsort(x, kind='stable'), kind='stable').astype(float)
def spear(x, y):
    return np.corrcoef(rank(x), rank(y))[0, 1] if len(x) > 2 else np.nan
def partial(x, y, z):   # spearman partial on ranks, controlling range
    rx, ry, rz = rank(x), rank(y), rank(z)
    r = np.corrcoef([rx, ry, rz]); xy, xz, yz = r[0, 1], r[0, 2], r[1, 2]
    return (xy - xz * yz) / np.sqrt((1 - xz**2) * (1 - yz**2))
R = [0, 10, 20, 30, 40, 50, 70]
for c, cn in C.items():
    b = a[a[:, 0] == c]; s, n, r, tp = b[:, 1], b[:, 2], b[:, 3], b[:, 4] > 0
    print('\n== %s  (n=%d, TP %d, empty boxes %.1f%%) ==' % (cn, len(b), tp.sum(), 100 * (n == 0).mean()))
    print('Spearman score~pts: all %.3f | TP %.3f | FP %.3f | partial on range %.3f | score~range %.3f | pts~range %.3f'
          % (spear(s, n), spear(s[tp], n[tp]), spear(s[~tp], n[~tp]), partial(s, n, r), spear(s, r), spear(n, r)))
    print('within ring  %-8s %-8s %-8s %-8s | median pts by score quartile (Q1..Q4)   | TP rate by quartile' % ('n', 'rho', 'rhoTP', 'rhoFP'))
    for lo, hi in zip(R[:-1], R[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() < 30: continue
        ss, nn, tt = s[m], n[m], tp[m]
        q = np.quantile(ss, [.25, .5, .75]); qi = np.searchsorted(q, ss)
        med = [np.median(nn[qi == i]) if (qi == i).any() else np.nan for i in range(4)]
        tpr = [tt[qi == i].mean() if (qi == i).any() else np.nan for i in range(4)]
        print('  %2d-%-2d m   %-8d %-8.3f %-8.3f %-8.3f | %s | %s' % (lo, hi, m.sum(), spear(ss, nn),
              spear(ss[tt], nn[tt]) if tt.sum() > 10 else np.nan, spear(ss[~tt], nn[~tt]) if (~tt).sum() > 10 else np.nan,
              ' '.join('%6.0f' % x for x in med), ' '.join('%.2f' % x for x in tpr)))
    # score threshold view: what a higher accept threshold keeps
    print('  accept >= thr:  thr  kept/frame  precision  median pts  (all ranges)')
    nf = len(np.unique(a[:, 5]))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k = s >= t
        if k.sum(): print('                 %.1f  %8.2f  %9.2f  %9.0f' % (t, k.sum() / nf, tp[k].mean(), np.median(n[k])))
