import pickle, sys, numpy as np
import bins
a = pickle.load(open(sys.argv[1] + '/score_pts.pkl', 'rb'))   # class, score, npts, range, tp, frame
R = [0, 10, 20, 30, 40, 50, 70]
SB = bins.SB
for c, cn in [(1, 'Car'), (2, 'Pedestrian'), (3, 'Cyclist')]:
    b = a[a[:, 0] == c]
    print('\n== %s: median points in box  (n boxes)  by score bin x ring ==' % cn)
    print('score      ' + ''.join('%-15s' % ('%d-%d m' % (lo, hi)) for lo, hi in zip(R[:-1], R[1:])))
    for e0, e1 in zip(SB[:-1], SB[1:]):
        row = '%.2f-%.2f  ' % (e0, min(e1, 1))
        for lo, hi in zip(R[:-1], R[1:]):
            m = (b[:, 1] >= e0) & (b[:, 1] < e1) & (b[:, 3] >= lo) & (b[:, 3] < hi)
            row += ('%5.0f (%5d)    ' % (np.median(b[m, 2]), m.sum())) if m.sum() >= 5 else '   -- (%5d)    ' % m.sum()
        print(row)
