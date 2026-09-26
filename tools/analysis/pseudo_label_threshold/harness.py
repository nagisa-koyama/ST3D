"""Label-free threshold indicators vs GT reference thresholds (validation only).

Inputs:
  score_pts.pkl   class, score, npts, range, tp, frame      (1000 frames, processed/range-masked)
  persistence.pkl class, score, range, persistent, tp, frame (964 anchors, raw boxes <70 m)
                  + rows with score=-1 carrying per-frame GT counts (validation)
  audit_full.pkl  GT records (validation)
Everything in `indicators()` reads only score / npts / range / persistence.
"""
import pickle, sys, numpy as np
from scipy.optimize import minimize
import bins

S = sys.argv[1]
SP = pickle.load(open(S + '/score_pts.pkl', 'rb'))
PE = pickle.load(open(S + '/persistence.pkl', 'rb'))
AU = pickle.load(open(S + '/audit_full.pkl', 'rb'))
GT = np.array(AU['target']['gt'], dtype=float).reshape(-1, 4)
GT = GT[(GT[:, 2] >= 1) & (GT[:, 3] < 70)]
GRID = bins.GRID
R = [0, 10, 20, 30, 40, 50, 70]
NAMES = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}


# ----------------------------------------------------------------- label-free indicators
def count_balance(scores, n_true):
    kept = np.array([(scores >= t).sum() for t in GRID])
    return GRID[np.argmin(np.abs(kept - n_true))]


def persistence_unmix(s, p, lo=bins.BOTTOM, hi=(0.50, 1.01)):
    P0 = p[(s >= lo[0]) & (s < lo[1])].mean()
    P1 = p[(s >= hi[0]) & (s < hi[1])].mean()
    edges = np.concatenate([bins.LOW_SB, np.arange(0.10, 0.50, 0.02), [1.01]])
    pi = np.zeros(len(s))
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (s >= e0) & (s < e1)
        if m.any():
            pi[m] = np.clip((p[m].mean() - P0) / max(P1 - P0, 1e-6), 0, 1)
    return pi, P0, P1


def trunc_exp_mix(s, lo=None):
    lo = bins.LO if lo is None else lo
    """Two truncated exponentials on [lo, 1]: fast (spurious) + slow (real). ML fit."""
    x = s - lo; L = 1.0 - lo

    def pdf(x, lam):
        return lam * np.exp(-lam * x) / (1 - np.exp(-lam * L))

    def nll(th):
        w = 1 / (1 + np.exp(-th[0])); l1, l2 = np.exp(th[1]), np.exp(th[2])
        return -np.log(w * pdf(x, l1) + (1 - w) * pdf(x, l2) + 1e-300).sum()
    best = None
    for w0 in [0.5, 0.8, 0.95]:
        r = minimize(nll, [np.log(w0 / (1 - w0)), np.log(30), np.log(2)], method='Nelder-Mead',
                     options={'maxiter': 4000, 'xatol': 1e-6, 'fatol': 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    w = 1 / (1 + np.exp(-best.x[0])); l1, l2 = sorted([np.exp(best.x[1]), np.exp(best.x[2])], reverse=True)
    post = (1 - w) * pdf(x, l2) / (w * pdf(x, l1) + (1 - w) * pdf(x, l2))
    return 1 - w, l1, l2, post


def kneedle(scores):
    kept = np.array([(scores >= t).sum() for t in GRID], dtype=float)
    y = np.log(kept); xn = (GRID - GRID[0]) / (GRID[-1] - GRID[0]); yn = (y - y[-1]) / (y[0] - y[-1])
    return GRID[np.argmax((1 - xn) - yn)]   # max distance below the chord of a convex decreasing curve


def indicators(c):
    out = {}
    pe = PE[(PE[:, 0] == c) & (PE[:, 1] >= 0)]
    s, p = pe[:, 1], pe[:, 3]
    pi, P0, P1 = persistence_unmix(s, p)
    out['I1 persist count-balance'] = count_balance(s, pi.sum())
    edges_pi = [(t, pi[(s >= t) & (s < t + 0.02)].mean()) for t in np.arange(0.10, 0.50, 0.02) if ((s >= t) & (s < t + 0.02)).any()]
    out['I2 persist pi=0.5'] = next((round(t, 2) for t, v in edges_pi if v >= 0.5), np.nan)
    w_true, l1, l2, post = trunc_exp_mix(s)
    out['I3 score-mix count-balance'] = count_balance(s, w_true * len(s))
    srt = np.argsort(s); ps_ = post[srt]
    out['I4 score-mix post=0.5'] = s[srt][np.argmax(ps_ >= 0.5)] if (ps_ >= 0.5).any() else np.nan
    out['I5 kneedle log-count'] = kneedle(s)
    out['_diag'] = 'P0 %.3f P1 %.3f n_true/fr %.2f | mix w_true %.3f lam %.1f/%.1f n_true/fr %.2f' % (
        P0, P1, pi.sum() / len(np.unique(pe[:, 5])), w_true, l1, l2, w_true * len(s) / len(np.unique(pe[:, 5])))
    return out


# ----------------------------------------------------------------- validation references (GT)
def references(c):
    b = SP[(SP[:, 0] == c) & (SP[:, 3] < 70)]; g = GT[GT[:, 1] == c]
    def tv(x):
        hx = np.histogram(x, R)[0] / len(x); hy = np.histogram(g[:, 3], R)[0] / len(g); return .5 * np.abs(hx - hy).sum()
    def w1(x, y):
        q = np.linspace(.005, .995, 199); return np.abs(np.quantile(x, q) - np.quantile(y, q)).mean()
    cost = [tv(b[b[:, 1] >= t, 3]) + w1(np.log(b[b[:, 1] >= t, 2]), np.log(g[:, 2])) if (b[:, 1] >= t).sum() > 50 else np.inf for t in GRID]
    pe = PE[PE[:, 0] == c]; ngt = pe[pe[:, 1] < 0, 4].sum()
    t_cb = count_balance(pe[pe[:, 1] >= 0, 1], ngt)
    return GRID[int(np.argmin(cost))], t_cb


if __name__ == '__main__':
    for c, cn in NAMES.items():
        t_dist, t_cb = references(c)
        ind = indicators(c)
        print('\n== %s ==  [VALIDATION refs: t_dist %.2f, t_cb %.2f]' % (cn, t_dist, t_cb))
        print('   ' + ind.pop('_diag'))
        for k, v in ind.items():
            print('   %-28s %.2f' % (k, v))
