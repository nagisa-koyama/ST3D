import pickle, sys, numpy as np, importlib.util
S = sys.argv[1]
spec = importlib.util.spec_from_file_location('h', S + '/harness.py'); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
GRID = np.round(np.arange(0.11, 0.60, 0.01), 2)
def knee(s, y):
    best = (np.inf, np.nan)
    for g in GRID:
        if (s > g).sum() < 30 or (s <= g).sum() < 30: continue
        X = np.column_stack([np.ones_like(s), np.maximum(s - g, 0)])
        b, *_ = np.linalg.lstsq(X, y, rcond=None); e = ((X @ b - y) ** 2).sum()
        if e < best[0]: best = (e, g)
    return best[1]
def i9(rows, c, rings=((10, 20), (20, 30)), boot=0):
    b = rows[rows[:, 0] == c]
    ks = []
    for lo, hi in rings:
        m = (b[:, 3] >= lo) & (b[:, 3] < hi)
        if m.sum() > 100: ks.append(knee(b[m, 1], np.log(b[m, 2])))
    return np.nanmean(ks), ks
K = pickle.load(open(S + '/kitti_oos.pkl', 'rb'))['rows']
for label, rows, cls in [('nuScenes', h.SP, {1: 'Car', 2: 'Ped'}), ('KITTI', K, {3: 'Car', 4: 'Ped'})]:
    for c, cn in cls.items():
        v, ks = i9(rows, c)
        v3, ks3 = i9(rows, c, rings=((0, 10), (10, 20), (20, 30), (30, 40)))
        print('%-8s %s  I9 knee 10-30m: %.2f  per ring %s | 0-40m: %.2f %s' % (label, cn, v, ks, v3, ks3))
