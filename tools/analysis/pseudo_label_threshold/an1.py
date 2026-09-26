import pickle, sys, numpy as np
a = pickle.load(open(sys.argv[1] + '/persistence.pkl', 'rb'))
box = a[a[:, 1] >= 0]; gtc = a[a[:, 1] < 0]
SB = [.10, .12, .14, .16, .18, .20, .25, .30, .40, .50, 1.01]
nf = len(np.unique(a[:, 5]))
for c, cn in [(1, 'Car'), (2, 'Ped'), (3, 'Cyc')]:
    b = box[box[:, 0] == c]; ngt = gtc[gtc[:, 0] == c, 4].sum()
    print('\n%s  boxes %d  GT/frame %.2f   [cols: slice n  persist  | VALIDATION prec, prec|persist, prec|not]' % (cn, len(b), ngt / nf))
    for e0, e1 in zip(SB[:-1], SB[1:]):
        m = (b[:, 1] >= e0) & (b[:, 1] < e1); x = b[m]; p = x[:, 3] > 0; t = x[:, 4] > 0
        print('%.2f-%.2f %6d  %.3f | %.3f %.3f %.3f' % (e0, e1, m.sum(), p.mean(), t.mean(), t[p].mean() if p.any() else np.nan, t[~p].mean()))
    p = b[:, 3] > 0; t = b[:, 4] > 0
    print('ALL      persist %.3f | P(persist|TP) %.3f  P(persist|FP) %.3f' % (p.mean(), p[t].mean(), p[~t].mean()))
