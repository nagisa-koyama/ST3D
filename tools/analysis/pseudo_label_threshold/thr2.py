import pickle, sys, numpy as np
a = pickle.load(open(sys.argv[1] + '/score_pts.pkl', 'rb'))
d = pickle.load(open(sys.argv[1] + '/audit_full.pkl', 'rb'))
g = np.array(d['target']['gt'], dtype=float).reshape(-1, 4); g = g[g[:, 2] >= 1]
R = [0, 10, 20, 30, 40, 50, 70]; T = [0.1, 0.15, 0.2]; nf = 1000
rank = lambda x: np.argsort(np.argsort(x, kind='stable'), kind='stable').astype(float)
sp = lambda x, y: np.corrcoef(rank(x), rank(y))[0, 1] if len(x) > 10 else np.nan
for c, cn in [(1, 'Car'), (2, 'Pedestrian'), (3, 'Cyclist')]:
    b = a[a[:, 0] == c]; gc = g[g[:, 1] == c]
    print('\n== %s ==  (real GT occupied boxes/frame %.2f)' % (cn, len(gc) / nf))
    print('thr   kept/fr  prec  recall  rho(score,pts)')
    for t in T:
        k = b[b[:, 1] >= t]
        print('%.2f  %7.2f  %.2f  %.2f    %.3f' % (t, len(k) / nf, k[:, 4].mean(), k[:, 4].sum() / len(gc), sp(k[:, 1], k[:, 2])))
    print('pts/box vs GT per ring: ' + '   '.join('%d-%d' % (lo, hi) for lo, hi in zip(R[:-1], R[1:])))
    for t in T:
        row = []
        for lo, hi in zip(R[:-1], R[1:]):
            gm = gc[(gc[:, 3] >= lo) & (gc[:, 3] < hi), 2]; k = b[(b[:, 1] >= t) & (b[:, 3] >= lo) & (b[:, 3] < hi), 2]
            row.append('%5.2f' % (k.mean() / gm.mean()) if len(k) >= 10 and len(gm) >= 10 else '   --')
        print('  thr %.2f               ' % t + '  '.join(row))
