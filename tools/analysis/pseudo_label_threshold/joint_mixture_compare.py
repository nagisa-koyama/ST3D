"""Joint (score, points) mixture vs score alone, PER RING at equal kept count, with an optional
label-free plateau exclusion (fit only boxes with score >= --min_score; the plateau is the
marginal's mode and carries no detections, so excluding it uses no labels).

    python joint_mixture_compare.py <work dir> [--min_score 0.1]
"""
import sys, pickle, argparse
import numpy as np
from sklearn.mixture import GaussianMixture

ap = argparse.ArgumentParser(); ap.add_argument('work'); ap.add_argument('--min_score', type=float, default=0.0)
args = ap.parse_args()
SP = pickle.load(open(args.work + '/score_pts.pkl', 'rb'))
AU = pickle.load(open(args.work + '/audit_full.pkl', 'rb'))
GT = np.array(AU['target']['gt'], dtype=float).reshape(-1, 4); GT = GT[GT[:, 2] >= 1]
NF = len(np.unique(SP[:, 5]))
RINGS = [(10, 20), (20, 30), (30, 40)]
print('fit restricted to score >= %g' % args.min_score)
for c, cn in [(1, 'Car'), (2, 'Pedestrian')]:
    print('\n==== %s ====' % cn)
    print('ring   GT/fr | JOINT post>=.5: kept/fr prec  rec  | SCORE at same count: t     prec  rec  | SCORE at t=0.20: kept prec rec | density est: post-weighted / >=0.18 cut / GT')
    for lo, hi in RINGS:
        b = SP[(SP[:, 0] == c) & (SP[:, 3] >= lo) & (SP[:, 3] < hi)]
        b = b[b[:, 1] >= args.min_score]
        tp = b[:, 4] > 0; ngt = ((GT[:, 1] == c) & (GT[:, 3] >= lo) & (GT[:, 3] < hi)).sum()
        z = np.column_stack([np.log(b[:, 1] / (1 - b[:, 1])), np.log(b[:, 2])])
        g = GaussianMixture(2, covariance_type='full', random_state=0, n_init=8).fit(z)
        hi_c = int(np.argmax(g.means_[:, 1])); post = g.predict_proba(z)[:, hi_c]
        keep = post >= 0.5; k = keep.sum()
        t_eq = np.sort(b[:, 1])[::-1][max(k - 1, 0)] if k else np.nan
        ks = b[:, 1] >= t_eq if k else np.zeros(len(b), bool)
        k20 = b[:, 1] >= 0.20
        gtm = np.exp(np.log(GT[(GT[:, 1] == c) & (GT[:, 3] >= lo) & (GT[:, 3] < hi), 2]).mean())
        est = np.exp((post * np.log(b[:, 2])).sum() / post.sum())
        cut18 = np.exp(np.log(b[b[:, 1] >= 0.18, 2]).mean())
        print('%2d-%-2d %5.2f | %6.2f  %.2f  %.2f | %.3f  %.2f  %.2f | %6.2f  %.2f  %.2f | %6.1f / %6.1f / %6.1f' % (
            lo, hi, ngt / NF, k / NF, tp[keep].mean() if k else np.nan, (tp & keep).sum() / max(tp.sum(), 1),
            t_eq, tp[ks].mean() if k else np.nan, (tp & ks).sum() / max(tp.sum(), 1),
            k20.sum() / NF, tp[k20].mean() if k20.any() else np.nan, (tp & k20).sum() / max(tp.sum(), 1),
            est, cut18, gtm))
        edges = [0, .1, .3, .5, .7, .9, 1.01]
        print('       reliability: ' + '  '.join('%.1f-%.1f:%.2f(n%d)' % (a, min(e, 1), tp[(post >= a) & (post < e)].mean(), ((post >= a) & (post < e)).sum())
                                            for a, e in zip(edges[:-1], edges[1:]) if ((post >= a) & (post < e)).sum() > 20))
