import pickle, sys, numpy as np
a = pickle.load(open(sys.argv[1] + '/score_pts.pkl', 'rb'))
d = pickle.load(open(sys.argv[1] + '/audit_full.pkl', 'rb'))
g = np.array(d['target']['gt'], dtype=float).reshape(-1, 4); g = g[(g[:, 2] >= 1) & (g[:, 3] < 70)]
R = [0, 10, 20, 30, 40, 50, 70]; T = [.10, .12, .14, .16, .18, .20, .25, .30, .40, .50]
def tv(x, y):
    hx = np.histogram(x, R)[0] / len(x); hy = np.histogram(y, R)[0] / len(y); return 0.5 * np.abs(hx - hy).sum()
def w1(x, y):
    q = np.linspace(0.005, 0.995, 199); return np.abs(np.quantile(x, q) - np.quantile(y, q)).mean()
def pts_row(x):
    cells = [('%.0f' % np.median(x[(x[:, 3] >= lo) & (x[:, 3] < hi), 2])) if ((x[:, 3] >= lo) & (x[:, 3] < hi)).sum() >= 5 else '-' for lo, hi in zip(R[:-1], R[1:])]
    return cells + ['%.0f' % np.median(x[:, 2])]
def rng_row(x):
    h = np.histogram(x[:, 3], R)[0]; return ['%.0f' % (100 * v / h.sum()) for v in h] + ['%.1f' % np.median(x[:, 3])]
for c, cn in [(1, 'Car'), (2, 'Pedestrian'), (3, 'Cyclist')]:
    b = a[(a[:, 0] == c) & (a[:, 3] < 70)]; gc = g[g[:, 1] == c]
    print('\n## %s' % cn)
    print('PTS|GT|%.2f|%s' % (len(gc) / 1000, '|'.join(pts_row(gc))))
    for t in T:
        x = b[b[:, 1] >= t]
        print('PTS|>=%.2f|%.2f|%s' % (t, len(x) / 1000, '|'.join(pts_row(x))))
    print('RNG|GT|%s' % '|'.join(rng_row(gc)))
    for t in T:
        x = b[b[:, 1] >= t]
        print('RNG|>=%.2f|%s|%.3f|%.1f|%.2f' % (t, '|'.join(rng_row(x)), tv(x[:, 3], gc[:, 3]), w1(x[:, 3], gc[:, 3]), w1(np.log(x[:, 2]), np.log(gc[:, 2]))))
