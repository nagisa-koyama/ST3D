import pickle, sys, numpy as np
import bins
a = pickle.load(open(sys.argv[1] + '/score_pts.pkl', 'rb'))
d = pickle.load(open(sys.argv[1] + '/audit_full.pkl', 'rb'))
g = np.array(d['target']['gt'], dtype=float).reshape(-1, 4); g = g[g[:, 2] >= 1]
R = [0, 10, 20, 30, 40, 50, 70]
SB = bins.SB
def shares(r):
    h = np.histogram(r, bins=R)[0]; return ' | '.join('%.0f%%' % (100 * x / h.sum()) for x in h) + ' | %.1f' % np.median(r)
for c, cn in [(1, 'Car'), (2, 'Pedestrian'), (3, 'Cyclist')]:
    b = a[(a[:, 0] == c) & (a[:, 3] < 70)]; gc = g[(g[:, 1] == c) & (g[:, 3] < 70)]
    print('\n' + cn); print('GT | ' + shares(gc[:, 3]))
    for e0, e1 in zip(SB[:-1], SB[1:]):
        m = (b[:, 1] >= e0) & (b[:, 1] < e1)
        print('%.2f-%.2f | %s' % (e0, min(e1, 1), shares(b[m, 3])))
