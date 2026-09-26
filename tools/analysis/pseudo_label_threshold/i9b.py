import pickle, sys, numpy as np, importlib.util
S = sys.argv[1]
spec = importlib.util.spec_from_file_location('h', S + '/harness.py'); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
def binned(s, y, w=0.01, top=0.6, minn=15):
    e = np.arange(0.10, top + 1e-9, w); xs, ys, ns = [], [], []
    for a, b in zip(e[:-1], e[1:]):
        m = (s >= a) & (s < b)
        if m.sum() >= minn: xs.append((a + b) / 2); ys.append(np.median(y[m])); ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)
def knee2(x, y, n):
    best = (np.inf, np.nan, None)
    for g in np.round(np.arange(0.12, 0.50, 0.01), 2):
        if (x < g).sum() < 3 or (x > g).sum() < 3: continue
        X = np.column_stack([np.ones_like(x), np.minimum(x - g, 0), np.maximum(x - g, 0)])
        W = np.sqrt(n)
        b, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None); e = (((X @ b - y) * W) ** 2).sum()
        if e < best[0]: best = (e, g, b[1:])
    return best[1], best[2]
K = pickle.load(open(S + '/kitti_oos.pkl', 'rb'))['rows']
for label, rows, cls in [('nuScenes', h.SP, {1: 'Car', 2: 'Ped'}), ('KITTI', K, {3: 'Car', 4: 'Ped'})]:
    for c, cn in cls.items():
        b = rows[rows[:, 0] == c]; out = []
        for lo, hi in [(10, 20), (20, 30)]:
            m = (b[:, 3] >= lo) & (b[:, 3] < hi)
            x, y, n = binned(b[m, 1], np.log(b[m, 2]))
            g, sl = knee2(x, y, n); out.append('%d-%d: %.2f (slopes %.1f -> %.1f, %d bins)' % (lo, hi, g, sl[0], sl[1], len(x)) if sl is not None else 'n/a')
        print('%-8s %s  %s' % (label, cn, ' | '.join(out)))
