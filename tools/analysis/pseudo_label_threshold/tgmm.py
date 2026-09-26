import pickle, sys, numpy as np, importlib.util
from scipy.stats import norm
from scipy.optimize import minimize
S = sys.argv[1]
spec = importlib.util.spec_from_file_location('h', S + '/harness.py'); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
Z0 = np.log(0.1 / 0.9)
def fit(s):
    z = np.log(s / (1 - s))
    def unpack(th):
        return 1 / (1 + np.exp(-th[0])), th[1], np.exp(th[2]), th[3], np.exp(th[4])
    def nll(th):
        w, m1, s1, m2, s2 = unpack(th)
        f1 = norm.pdf(z, m1, s1) / norm.sf(Z0, m1, s1); f2 = norm.pdf(z, m2, s2) / norm.sf(Z0, m2, s2)
        return -np.log(w * f1 + (1 - w) * f2 + 1e-300).sum()
    best = None
    for m1 in [-4, -3, -2.5]:
        for m2 in [-1, 0, 1]:
            for w0 in [0.6, 0.85]:
                r = minimize(nll, [np.log(w0 / (1 - w0)), m1, np.log(0.7), m2, np.log(1.0)], method='Nelder-Mead', options={'maxiter': 6000, 'xatol': 1e-5, 'fatol': 1e-5})
                if best is None or r.fun < best.fun: best = r
    w, m1, s1, m2, s2 = unpack(best.x)
    if m1 > m2: w, m1, s1, m2, s2 = 1 - w, m2, s2, m1, s1
    # expected count of the HIGH component within the observed range
    n_hi = (1 - w) * len(z)
    grid = np.linspace(0.1, 0.95, 851); gz = np.log(grid / (1 - grid))
    f1 = w * norm.pdf(gz, m1, s1) / norm.sf(Z0, m1, s1); f2 = (1 - w) * norm.pdf(gz, m2, s2) / norm.sf(Z0, m2, s2)
    post = f2 / (f1 + f2); t_post = grid[np.argmax(post >= .5)] if (post >= .5).any() else np.nan
    return t_post, h.count_balance(s, n_hi), (w, m1, s1, m2, s2)
K = pickle.load(open(S + '/kitti_oos.pkl', 'rb')); rng = np.random.default_rng(3)
pe = h.PE
infos = pickle.load(open('/st3d/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl', 'rb'))
loc = np.array([i['lidar_path'].split('/')[-1].startswith('n008') for i in infos]); half = rng.random(len(infos)) < .5
sets = []
for c, cn in [(1, 'Car'), (2, 'Ped')]:
    for name, mask in [('all', np.ones(len(infos), bool)), ('Boston', loc), ('Singapore', ~loc), ('half A', half), ('half B', ~half)]:
        x = pe[mask[pe[:, 5].astype(int)]]; b = x[(x[:, 0] == c) & (x[:, 1] >= 0)]; ngt = x[(x[:, 0] == c) & (x[:, 1] < 0), 4].sum()
        sets.append(('nuScenes %s %s' % (cn, name), b[:, 1], h.count_balance(b[:, 1], ngt)))
r, g = K['rows'], K['gt']; fr = r[:, 5].astype(int); hk = rng.random(fr.max() + 1) < .5
for c, cn in [(3, 'Car'), (4, 'Ped')]:
    for name, sel in [('all', np.ones(fr.max() + 1, bool)), ('half A', hk), ('half B', ~hk)]:
        b = r[(r[:, 0] == c) & sel[fr]]; ngt = (g[:, 0] == c).sum() * sel.mean()
        sets.append(('KITTI %s %s' % (cn, name), b[:, 1], h.count_balance(b[:, 1], ngt)))
for name, s, tcb in sets:
    tp, tc, p = fit(s)
    print('%-28s [VALID t_cb %.2f]  tGMM post %.2f  cb %.2f   (w_lo %.2f, lo N(%.1f,%.2f), hi N(%.1f,%.2f))' % (name, tcb, tp, tc, *p))
