"""Label-free Car (and Pedestrian) count-balance cut for a KITTI target, from score_vs_points.py output.

Per-ring 2-component GMM on (logit score, log points) inside 30 m; the real-count estimate is the summed
posterior of the dense / high-score component, and the cut is the score at which the kept count inside
30 m equals it. This is the estimator behind the S2 Car cuts (0.301 for teacher 2da6oz6e, 0.275 for
zpndk3rh). The tp column of the input is read ONLY for the printed VALID check, and on a KITTI target it
counts unlabelled rear cars as false (score_vs_points does not restrict to the camera FOV), so that
column understates precision; the cut itself never reads it. Copied into the repo from a session
scratchpad (it existed only there).

    python kitti_car_count_balance_cut.py <score_vs_points output .pkl>
"""
import pickle, sys, numpy as np
from sklearn.mixture import GaussianMixture
SP = pickle.load(open(sys.argv[1], 'rb'))          # class, score, npts, range, tp, frame
NF = len(np.unique(SP[:, 5]))
for c, cn in ((1, 'Car'), (2, 'Pedestrian')):
    b = SP[(SP[:, 0] == c) & (SP[:, 3] < 30)]
    est = 0.0
    for lo, hi in ((0, 10), (10, 20), (20, 30)):
        r = b[(b[:, 3] >= lo) & (b[:, 3] < hi)]
        z = np.column_stack([np.log(r[:, 1] / (1 - r[:, 1])), np.log(r[:, 2])])
        g = GaussianMixture(2, covariance_type='full', random_state=0, n_init=8).fit(z)
        hi_c = int(np.argmax(g.means_[:, 1] + 0.01 * g.means_[:, 0]))
        est += g.predict_proba(z)[:, hi_c].sum()
    s = np.sort(b[:, 1])[::-1]; k = int(round(est)); cut = s[min(k, len(s)) - 1]
    kept = b[b[:, 1] >= cut]
    print('%-10s inside 30 m: boxes %d, est real %.0f (%.2f/frame), count-balance cut %.3f, kept %.2f/frame, VALID precision %.2f, real TP %.2f/frame'
          % (cn, len(b), est, est / NF, cut, len(kept) / NF, kept[:, 4].mean(), b[:, 4].sum() / NF))
