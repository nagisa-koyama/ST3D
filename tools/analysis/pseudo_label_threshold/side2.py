import pickle, sys, numpy as np
import bins
a = pickle.load(open(sys.argv[1] + '/score_pts.pkl', 'rb'))
d = pickle.load(open(sys.argv[1] + '/audit_full.pkl', 'rb'))
g = np.array(d['target']['gt'], dtype=float).reshape(-1, 4); g = g[g[:, 2] >= 1]
SB = bins.SB
for c, cn in [(1, 'Car'), (2, 'Pedestrian'), (3, 'Cyclist')]:
    b = a[(a[:, 0] == c) & (a[:, 3] < 70)]; gc = g[(g[:, 1] == c) & (g[:, 3] < 70)]
    print('%s GT %.0f (n=%d)' % (cn, np.median(gc[:, 2]), len(gc)))
    for e0, e1 in zip(SB[:-1], SB[1:]):
        m = (b[:, 1] >= e0) & (b[:, 1] < e1)
        print('  %.2f-%.2f %.0f (n=%d)' % (e0, e1, np.median(b[m, 2]), m.sum()))
    print('  all>=0.1 %.0f (n=%d)' % (np.median(b[:, 2]), len(b)))
