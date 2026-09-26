import pickle, sys, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('h', sys.argv[1] + '/harness.py'); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
STEP = 28
R = np.array([0, 10, 20, 30, 40, 50, 70]); LB = np.linspace(0, np.log(3000), 41)
def join(c):
    sp = h.SP[(h.SP[:, 0] == c) & (h.SP[:, 3] < 70)]; pe = h.PE[(h.PE[:, 0] == c) & (h.PE[:, 1] >= 0)]
    key = lambda fr, s: np.round(fr).astype(np.int64) * 10**7 + np.round(s * 1e6).astype(np.int64)
    kp = dict(zip(key(pe[:, 5], pe[:, 1]), pe[:, 3]))
    ks = key(sp[:, 5] * STEP, sp[:, 1])
    m = np.array([k in kp for k in ks])
    return sp[m], np.array([kp[k] for k in ks[m]])   # sp cols: class, score, npts, range, tp, frame
def dist(w, r, lp):
    hr = np.histogram(r, R, weights=w)[0]; hp = np.histogram(lp, LB, weights=w)[0]
    hr, hp = np.clip(hr, 0, None), np.clip(hp, 0, None)
    return hr / hr.sum(), np.cumsum(hp / hp.sum())
def cost(a, b):
    return 0.5 * np.abs(a[0] - b[0]).sum() + np.abs(a[1] - b[1]).sum() * (LB[1] - LB[0])
for c in [1, 2]:
    sp, pers = join(c)
    s, n, r, tp = sp[:, 1], sp[:, 2], sp[:, 3], sp[:, 4] > 0
    pi, P0, P1 = h.persistence_unmix(s, pers)
    w = (pers - P0) / (P1 - P0)
    est = dist(w, r, np.log(n))                                  # label-free estimate of real-object distribution
    g = h.GT[h.GT[:, 1] == c]; gtd = dist(np.ones(len(g)), g[:, 3], np.log(g[:, 2]))   # validation
    tpd = dist(tp.astype(float), r, np.log(n))                   # validation: true TP distribution
    ce = [cost(dist(np.ones((s >= t).sum()), r[s >= t], np.log(n[s >= t])), est) for t in h.GRID]
    cg = [cost(dist(np.ones((s >= t).sum()), r[s >= t], np.log(n[s >= t])), gtd) for t in h.GRID]
    ct = [cost(dist(np.ones((s >= t).sum()), r[s >= t], np.log(n[s >= t])), tpd) for t in h.GRID]
    print('%s joined %d | I6 (label-free) t=%.2f | VALID: vs GT t=%.2f, vs TP-population t=%.2f | est-vs-GT cost %.3f, est-vs-TP cost %.3f'
          % (h.NAMES[c], len(sp), h.GRID[np.argmin(ce)], h.GRID[np.argmin(cg)], h.GRID[np.argmin(ct)], cost(est, gtd), cost(est, tpd)))
    print('   est range shares ' + ' '.join('%.2f' % x for x in est[0]) + '  | GT ' + ' '.join('%.2f' % x for x in gtd[0]) + '  | TP ' + ' '.join('%.2f' % x for x in tpd[0]))
