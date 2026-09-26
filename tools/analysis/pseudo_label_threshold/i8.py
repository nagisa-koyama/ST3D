import pickle, sys, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('h', sys.argv[1] + '/harness.py'); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
spec2 = importlib.util.spec_from_file_location('j', sys.argv[1] + '/i6.py')
src = open(sys.argv[1] + '/i6.py').read().split('def dist')[0]; exec(src.split('STEP = 28')[1].replace('\nR = ', '\n_R = '), globals()) if False else None
STEP = 28
def join(c):
    sp = h.SP[(h.SP[:, 0] == c) & (h.SP[:, 3] < 70)]; pe = h.PE[(h.PE[:, 0] == c) & (h.PE[:, 1] >= 0)]
    key = lambda fr, s: np.round(fr).astype(np.int64) * 10**7 + np.round(s * 1e6).astype(np.int64)
    kp = dict(zip(key(pe[:, 5], pe[:, 1]), pe[:, 3])); ks = key(sp[:, 5] * STEP, sp[:, 1])
    m = np.array([k in kp for k in ks]); return sp[m], np.array([kp[k] for k in ks[m]])
RINGS = [(10, 20), (20, 30)]
for c in [1, 2]:
    sp, pers = join(c); s, n, r, tp = sp[:, 1], sp[:, 2], sp[:, 3], sp[:, 4] > 0
    g = h.GT[h.GT[:, 1] == c]
    est, gtm, tpm = {}, {}, {}
    for lo, hi in RINGS:
        m = (r >= lo) & (r < hi); sm, pm, lm = s[m], pers[m], np.log(n[m])
        P0 = pm[sm < 0.12].mean(); q = np.quantile(sm, 0.8); P1 = pm[sm >= max(q, 0.4)].mean()
        w = (pm - P0) / (P1 - P0)
        est[(lo, hi)] = (w * lm).sum() / w.sum()                        # label-free mean log pts of real objects
        gtm[(lo, hi)] = np.log(g[(g[:, 3] >= lo) & (g[:, 3] < hi), 2]).mean()   # VALIDATION
        tpm[(lo, hi)] = lm[tp[m]].mean()                                 # VALIDATION
    def crit(ref):
        out = []
        for t in h.GRID:
            e = 0
            for lo, hi in RINGS:
                k = (r >= lo) & (r < hi) & (s >= t)
                e += (np.log(n[k]).mean() - ref[(lo, hi)]) ** 2 if k.sum() > 20 else np.inf
            out.append(e)
        return h.GRID[int(np.argmin(out))]
    print('%s  I8 (label-free) t=%.2f | VALID vs GT-mean t=%.2f, vs TP-mean t=%.2f' % (h.NAMES[c], crit(est), crit(gtm), crit(tpm)))
    for k in RINGS:
        print('   ring %s  est exp(mean log pts) %.1f | GT %.1f | TP %.1f' % (k, np.exp(est[k]), np.exp(gtm[k]), np.exp(tpm[k])))
