import pickle, sys, numpy as np
import bins
a = pickle.load(open(sys.argv[1] + '/score_pts.pkl', 'rb'))
d = pickle.load(open(sys.argv[1] + '/audit_full.pkl', 'rb'))
g = np.array(d['target']['gt'], dtype=float).reshape(-1, 4); g = g[g[:, 2] >= 1]
R = [0, 10, 20, 30, 40, 50, 70]
for c, cn in [(1, 'Car'), (2, 'Pedestrian')]:
    print('\n%s: mean pts/box per ring, pseudo kept at threshold / real GT (1.00 = unbiased)' % cn)
    print('ring     GT_mean   ' + '  '.join('thr%.1f' % t for t in bins.T))
    for lo, hi in zip(R[:-1], R[1:]):
        gm = g[(g[:, 1] == c) & (g[:, 3] >= lo) & (g[:, 3] < hi), 2].mean()
        b = a[(a[:, 0] == c) & (a[:, 3] >= lo) & (a[:, 3] < hi)]
        row = ['%5.2f' % (b[b[:, 1] >= t, 2].mean() / gm) if (b[:, 1] >= t).sum() >= 10 else '   --' for t in bins.T]
        print('%2d-%-2d  %8.1f   ' % (lo, hi, gm) + '  '.join('%6s' % x for x in row))
