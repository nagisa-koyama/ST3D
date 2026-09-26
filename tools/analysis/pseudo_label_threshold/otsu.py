import os
import pickle, sys, numpy as np, importlib.util
S = sys.argv[1]
spec = importlib.util.spec_from_file_location('h', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness.py')); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
from sklearn.mixture import GaussianMixture
import bins as B
def otsu(s, bins=90):
    hst, e = np.histogram(s, bins=bins, range=(B.LO, 1.0)); p = hst / hst.sum(); x = (e[:-1] + e[1:]) / 2
    best = (-1, None)
    for k in range(1, bins):
        w0, w1 = p[:k].sum(), p[k:].sum()
        if w0 == 0 or w1 == 0: continue
        m0, m1 = (p[:k] * x[:k]).sum() / w0, (p[k:] * x[k:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best[0]: best = (v, e[k])
    return best[1]
def gmm_logit(s):
    z = np.log(s / (1 - s)).reshape(-1, 1)
    g = GaussianMixture(2, random_state=0, n_init=20).fit(z); hi = np.argmax(g.means_.ravel())
    grid = np.linspace(B.LO, 0.9, 801); pz = g.predict_proba(np.log(grid / (1 - grid)).reshape(-1, 1))[:, hi]
    t = grid[np.argmax(pz >= 0.5)] if (pz >= .5).any() else np.nan
    w_true = g.weights_[hi]
    return t, h.count_balance(s, w_true * len(s))
K = pickle.load(open(S + '/kitti_oos.pkl', 'rb'))
refs = {('nuScenes', 1): (0.17, 0.20), ('nuScenes', 2): (0.10, 0.15), ('KITTI', 3): (0.47, 0.35), ('KITTI', 4): (0.60, 0.29)}
for label, rows, cls in [('nuScenes', h.PE[h.PE[:, 1] >= 0], {1: 'Car', 2: 'Ped'}), ('KITTI', K['rows'], {3: 'Car', 4: 'Ped'})]:
    for c, cn in cls.items():
        s = rows[rows[:, 0] == c, 1]
        t_post, t_cb = gmm_logit(s)
        print('%-8s %s [VALID t_dist %.2f t_cb %.2f]  Otsu %.2f | GMM-logit post=0.5 %.2f, count-balance %.2f' % (label, cn, *refs[(label, c)], otsu(s), t_post, t_cb))
