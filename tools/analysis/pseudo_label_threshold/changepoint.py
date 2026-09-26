"""Label-free score change point for pseudo-labels: where does 'points inside the box' start to
depend on score, within range? Uses ONLY (class, score, npts, range, frame). The TP column is read
at the very end, for validation, never for selection."""
import pickle, sys, numpy as np

a = pickle.load(open(sys.argv[1] + '/score_pts.pkl', 'rb'))   # class, score, npts, range, tp, frame
C = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
RING = 5.0
GRID = np.round(np.arange(0.105, 0.45, 0.005), 3)
rng = np.random.default_rng(0)


def design(s, ring_idx, nr, sstar):
    X = np.zeros((len(s), nr + 2))
    X[np.arange(len(s)), ring_idx] = 1.0                 # ring fixed effects
    X[:, nr] = np.minimum(s - sstar, 0)                  # slope below the kink
    X[:, nr + 1] = np.maximum(s - sstar, 0)              # slope above
    return X


def fit(s, y, ring_idx, nr):
    best = (np.inf, None, None)
    for g in GRID:
        if (s > g).sum() < 30 or (s <= g).sum() < 30:
            continue
        X = design(s, ring_idx, nr, g)
        beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = ((X @ beta - y) ** 2).sum()
        if sse < best[0]:
            best = (sse, g, beta[nr:])
    # null: one slope, ring effects
    X0 = np.column_stack([np.eye(nr)[ring_idx], s])
    b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    sse0 = ((X0 @ b0 - y) ** 2).sum()
    return best, sse0


def hist_kink(s):
    """Second label-free signal: log-density of score, fit two lines, kink location."""
    edges = np.arange(0.1, 0.6001, 0.01)
    h, _ = np.histogram(s, bins=edges)
    x, ok = (edges[:-1] + edges[1:]) / 2, h > 0
    x, ly = x[ok], np.log(h[ok])
    best = (np.inf, None)
    for g in GRID:
        X = np.column_stack([np.ones_like(x), np.minimum(x - g, 0), np.maximum(x - g, 0)])
        b, *_ = np.linalg.lstsq(X, ly, rcond=None)
        sse = ((X @ b - ly) ** 2).sum()
        if sse < best[0]:
            best = (sse, g, b[1:])
    return best


for c, cn in C.items():
    b = a[(a[:, 0] == c) & (a[:, 3] < 70)]
    s, n, r, fr = b[:, 1], b[:, 2], b[:, 3], b[:, 5].astype(int)
    y = np.log(n)                                       # every box has >= 1 point
    ri = (r // RING).astype(int); nr = ri.max() + 1
    (sse, sstar, slopes), sse0 = fit(s, y, ri, nr)
    # bootstrap over frames
    frames = np.unique(fr); by = {f: np.where(fr == f)[0] for f in frames}
    boots = []
    for _ in range(200):
        idx = np.concatenate([by[f] for f in rng.choice(frames, len(frames))])
        (_, g, _), _ = fit(s[idx], y[idx], ri[idx], nr)
        boots.append(g)
    lo, hi = np.percentile([x for x in boots if x is not None], [5, 95])
    hk = hist_kink(s)
    print('\n== %s (n=%d) ==' % (cn, len(b)))
    print('points-vs-score hinge: s* = %.3f  (90%% bootstrap CI %.3f-%.3f)  slope below %.2f, above %.2f '
          '[d log(pts) per unit score];  SSE drop vs single slope %.1f%%'
          % (sstar, lo, hi, slopes[0], slopes[1], 100 * (sse0 - sse) / sse0))
    print('score-histogram log-density kink: %.3f  (slope below %.1f, above %.1f)' % (hk[1], hk[2][0], hk[2][1]))

    # the curve itself: range-adjusted log points by score bin, label-free
    Xr = np.eye(nr)[ri]; br, *_ = np.linalg.lstsq(Xr, y, rcond=None); resid = y - Xr @ br
    print('score bin    n     range-adjusted pts (x ring baseline)   share of boxes')
    for e0, e1 in zip([.10, .12, .14, .16, .18, .20, .25, .30, .40, .50], [.12, .14, .16, .18, .20, .25, .30, .40, .50, 1.01]):
        m = (s >= e0) & (s < e1)
        if m.sum() >= 20:
            print('%.2f-%.2f  %6d   %6.2f                                %5.1f%%' % (
                e0, min(e1, 1), m.sum(), np.exp(np.median(resid[m])), 100 * m.mean()))

    # validation only - target GT enters here and nowhere above
    tp = b[:, 4] > 0
    for t in sorted({round(sstar, 3), round(hk[1], 3)}):
        k = s >= t
        print('  [validation, uses GT] thr %.3f: kept %.2f/frame, precision %.2f, TP kept %.2f of TP>=0.1'
              % (t, k.sum() / 1000, tp[k].mean(), (tp & k).sum() / tp.sum()))
