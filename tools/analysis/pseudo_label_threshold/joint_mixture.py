"""Label-free per-box P(real) from a 2-component GMM on (logit score, log points-in-box), per class
and range ring. The score marginal alone is a background plateau plus a monotone tail (no second
mode - 20260926_06 follow-up), but at 10-30 m real objects hold several times the points of false
positives at the same score, so the joint should separate where the marginal cannot.

    python joint_mixture.py <work dir with score_pts.pkl> [ring edges]

Selection uses only score, points and range. The TP column is read for VALIDATION columns only.
Reports, per class and ring: the fitted components, kept/frame and precision/recall at posterior
>= 0.5 (valid.), a reliability table (posterior decile -> observed TP share), and the
posterior-weighted mean log points against the GT mean and the TP-population mean (the quantity
the foreground calibration needs).
"""
import sys, pickle
import numpy as np
from sklearn.mixture import GaussianMixture

W = sys.argv[1]
SP = pickle.load(open(W + '/score_pts.pkl', 'rb'))          # class, score, npts, range, tp, frame
try:
    AU = pickle.load(open(W + '/audit_full.pkl', 'rb'))
    GT = np.array(AU['target']['gt'], dtype=float).reshape(-1, 4); GT = GT[GT[:, 2] >= 1]
except Exception:
    GT = None
NF = len(np.unique(SP[:, 5]))
RINGS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 70)]
NAMES = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}


def fit_ring(b):
    z = np.column_stack([np.log(b[:, 1] / (1 - b[:, 1])), np.log(b[:, 2])])
    g = GaussianMixture(2, covariance_type='full', random_state=0, n_init=8).fit(z)
    # the "real" component: higher mean log points (tie-break on score)
    hi = int(np.argmax(g.means_[:, 1] + 0.01 * g.means_[:, 0]))
    post = g.predict_proba(z)[:, hi]
    return g, hi, post


for c, cn in NAMES.items():
    print('\n==== %s ====' % cn)
    print('ring     n   | real comp: w    score  pts | bg comp: score  pts | post>=.5 kept/fr [VALID prec rec] | weighted mean pts (est / TP / GT)')
    all_post, all_tp, all_n = [], [], []
    for lo, hi_r in RINGS:
        b = SP[(SP[:, 0] == c) & (SP[:, 3] >= lo) & (SP[:, 3] < hi_r)]
        if len(b) < 200:
            continue
        g, hi, post = fit_ring(b); lo_c = 1 - hi
        tp = b[:, 4] > 0; keep = post >= 0.5
        sig = lambda m: 1 / (1 + np.exp(-m))
        est = np.exp((post * np.log(b[:, 2])).sum() / post.sum())
        tpm = np.exp(np.log(b[tp, 2]).mean()) if tp.any() else np.nan
        gtm = np.exp(np.log(GT[(GT[:, 1] == c) & (GT[:, 3] >= lo) & (GT[:, 3] < hi_r), 2]).mean()) if GT is not None else np.nan
        print('%2d-%-2d %7d | %.2f  %.3f  %6.1f | %.3f  %6.1f | %6.2f  [%.2f %.2f] | %6.1f / %6.1f / %6.1f' % (
            lo, hi_r, len(b), g.weights_[hi], sig(g.means_[hi, 0]), np.exp(g.means_[hi, 1]),
            sig(g.means_[lo_c, 0]), np.exp(g.means_[lo_c, 1]),
            keep.sum() / NF, tp[keep].mean() if keep.any() else np.nan, (tp & keep).sum() / max(tp.sum(), 1),
            est, tpm, gtm))
        all_post.append(post); all_tp.append(tp); all_n.append(b[:, 2])
    post = np.concatenate(all_post); tp = np.concatenate(all_tp)
    print('reliability (all rings pooled): posterior bin -> n, observed TP share [VALID]')
    edges = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
    for a, e in zip(edges[:-1], edges[1:]):
        m = (post >= a) & (post < e)
        if m.sum():
            print('   %.2f-%.2f %7d  %.2f' % (a, min(e, 1), m.sum(), tp[m].mean()))
    # score-only comparison at equal kept count
    s = np.concatenate([SP[(SP[:, 0] == c) & (SP[:, 3] >= lo) & (SP[:, 3] < hi_r), 1] for lo, hi_r in RINGS
                        if len(SP[(SP[:, 0] == c) & (SP[:, 3] >= lo) & (SP[:, 3] < hi_r)]) >= 200])
    k = (post >= 0.5).sum()
    if k:
        t_eq = np.sort(s)[::-1][k - 1]
        print('same kept count by SCORE alone: t=%.3f, precision %.2f  vs joint posterior>=0.5 precision %.2f'
              % (t_eq, tp[s >= t_eq].mean(), tp[post >= 0.5].mean()))
