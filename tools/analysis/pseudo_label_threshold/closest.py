import pickle, sys, numpy as np
import bins
a = pickle.load(open(sys.argv[1] + '/score_pts.pkl', 'rb'))
d = pickle.load(open(sys.argv[1] + '/audit_full.pkl', 'rb'))
g = np.array(d['target']['gt'], dtype=float).reshape(-1, 4); g = g[(g[:, 2] >= 1) & (g[:, 1] == 1) & (g[:, 3] < 70)]
b = a[(a[:, 0] == 1) & (a[:, 3] < 70)]
R = [0, 10, 20, 30, 40, 50, 70]
def tv(x, y):
    hx = np.histogram(x, R)[0] / len(x); hy = np.histogram(y, R)[0] / len(y); return 0.5 * np.abs(hx - hy).sum()
def w1(x, y):   # 1-D Wasserstein via quantiles
    q = np.linspace(0.005, 0.995, 199); return np.abs(np.quantile(x, q) - np.quantile(y, q)).mean()
lp = lambda n: np.log(n)
print('SLICE        n     TV(range)  W1(range,m)  W1(log pts)  median pts (GT 10)')
SB = bins.SB
for e0, e1 in zip(SB[:-1], SB[1:]):
    m = (b[:, 1] >= e0) & (b[:, 1] < e1); x = b[m]
    if m.sum() < 10: continue
    print('%.2f-%.2f %6d   %.3f      %5.2f        %.2f         %.0f' % (e0, min(e1, 1), m.sum(), tv(x[:, 3], g[:, 3]), w1(x[:, 3], g[:, 3]), w1(lp(x[:, 2]), lp(g[:, 2])), np.median(x[:, 2])))
print('\nCUMULATIVE (score >= t)   kept/frame (GT 10.7)')
for t in bins.T:
    m = b[:, 1] >= t; x = b[m]
    print('>=%.2f  %6d  %5.2f   %.3f      %5.2f        %.2f         %.0f' % (t, m.sum(), m.sum() / 1000, tv(x[:, 3], g[:, 3]), w1(x[:, 3], g[:, 3]), w1(lp(x[:, 2]), lp(g[:, 2])), np.median(x[:, 2])))
