import pickle, sys, numpy as np
a = pickle.load(open(sys.argv[1] + '/score_pts.pkl', 'rb'))
d = pickle.load(open(sys.argv[1] + '/audit_full.pkl', 'rb'))
g = np.array(d['target']['gt'], dtype=float).reshape(-1, 4); g = g[g[:, 2] >= 1]
R = [0, 10, 20, 30, 40, 50, 70]
SB = [.10, .12, .14, .16, .18, .20, .25, .30, .40, .50, 1.01]
for c, cn in [(1, 'Car'), (2, 'Pedestrian'), (3, 'Cyclist')]:
    b = a[a[:, 0] == c]
    print('\n%s | %s' % (cn, ' | '.join('%d-%d m' % r for r in zip(R[:-1], R[1:]))))
    print('GT | ' + ' | '.join('%.0f' % np.median(g[(g[:, 1] == c) & (g[:, 3] >= lo) & (g[:, 3] < hi), 2]) for lo, hi in zip(R[:-1], R[1:])))
    for e0, e1 in zip(SB[:-1], SB[1:]):
        cells = []
        for lo, hi in zip(R[:-1], R[1:]):
            m = (b[:, 1] >= e0) & (b[:, 1] < e1) & (b[:, 3] >= lo) & (b[:, 3] < hi)
            cells.append('%.0f' % np.median(b[m, 2]) if m.sum() >= 5 else '–')
        print('%.2f–%.2f | %s' % (e0, min(e1, 1), ' | '.join(cells)))
