import pickle, sys, numpy as np
from sklearn.mixture import GaussianMixture
S = sys.argv[1]
PE = pickle.load(open(S + '/persistence.pkl', 'rb')); K = pickle.load(open(S + '/kitti_oos.pkl', 'rb'))['rows']
sets = [('nuScenes', PE[PE[:, 1] >= 0], {1: 'Car', 2: 'Ped', 3: 'Cyc'}), ('KITTI', K, {3: 'Car', 4: 'Ped'})]
print('%-14s %7s %9s %9s %8s %8s %8s  %s' % ('set', 'n', 'dBIC/n', 'w_hi', 'sep', 't_post', 'prec@t', '[VALID prec@t uses GT]'))
for lab, rows, cls in sets:
    for c, cn in cls.items():
        b = rows[rows[:, 0] == c]; s = b[:, 1]; z = np.log(s / (1 - s)).reshape(-1, 1)
        g1 = GaussianMixture(1, random_state=0).fit(z); g2 = GaussianMixture(2, random_state=0, n_init=10).fit(z)
        dbic = (g1.bic(z) - g2.bic(z)) / len(z)
        hi = int(np.argmax(g2.means_.ravel())); lo = 1 - hi
        sep = abs(g2.means_[hi, 0] - g2.means_[lo, 0]) / np.sqrt(0.5 * (g2.covariances_[hi, 0, 0] + g2.covariances_[lo, 0, 0]))
        grid = np.linspace(0.1, 0.95, 851); post = g2.predict_proba(np.log(grid / (1 - grid)).reshape(-1, 1))[:, hi]
        t = grid[np.argmax(post >= .5)] if (post >= .5).any() else np.nan
        prec = b[s >= t, 4].mean() if np.isfinite(t) else np.nan
        print('%-14s %7d %9.3f %9.2f %8.2f %8.2f %8.2f' % (lab + ' ' + cn, len(s), dbic, g2.weights_[hi], sep, t, prec))
