import pickle, sys, numpy as np
d = pickle.load(open(sys.argv[1] + '/audit_full.pkl', 'rb'))
g = np.array(d['target']['gt'], dtype=float).reshape(-1, 4); g = g[g[:, 2] >= 1]   # ring, class, npts, r
R = [0, 10, 20, 30, 40, 50, 70]
for c, cn in [(1, 'Car'), (2, 'Pedestrian'), (3, 'Cyclist')]:
    print('%-11s' % cn + ''.join('%5.0f (%4d)   ' % (np.median(g[m, 2]), m.sum()) if (m := (g[:, 1] == c) & (g[:, 3] >= lo) & (g[:, 3] < hi)).sum() >= 5 else '   -- (%4d)   ' % m.sum() for lo, hi in zip(R[:-1], R[1:])))
