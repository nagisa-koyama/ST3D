import pickle, sys, numpy as np
a = pickle.load(open(sys.argv[1] + '/gaps_0.3_2.0.pkl', 'rb'))  # cls, range, npts, detected, score, gt_near, frame
R = [0, 10, 20, 30, 40, 50, 70]
for c, cn in [(1, 'Car'), (2, 'Ped')]:
    b = a[a[:, 0] == c]
    print('\n%s tracks %d  [VALID: share of track midpoints with a real GT box nearby %.2f]' % (cn, len(b), b[:, 5].mean()))
    print(' ring   n   label-free recall(>=0.10)  missed: median pts  detected: median pts, median score   [VALID GT-near among missed]')
    for lo, hi in zip(R[:-1], R[1:]):
        m = (b[:, 1] >= lo) & (b[:, 1] < hi); x = b[m]
        if len(x) < 10: continue
        d = x[:, 3] > 0
        print(' %2d-%-2d %5d   %.3f      %6.0f       %6.0f  %.2f      [%.2f]' % (lo, hi, len(x), d.mean(), np.median(x[~d, 2]) if (~d).any() else np.nan, np.median(x[d, 2]), np.median(x[d, 4]), x[~d, 5].mean() if (~d).any() else np.nan))
