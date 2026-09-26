import pickle, sys
import numpy as np
d = pickle.load(open(sys.argv[1], 'rb'))
edges = np.linspace(0, 75, 51)
def pb(fg, nb): return np.divide(fg, nb, out=np.zeros_like(fg), where=nb > 0)
tb = {k: pb(fg, nb) for k, (fg, nb, _) in d['target_bins'].items()}
tn = {k: nb for k, (fg, nb, _) in d['target_bins'].items()}
for s, (fg, nb, _) in d['source_bins'].items():
    fs = pb(fg, nb)
    print('\n== %s: fg keep rate min(F_t/F_s,1) per 1.5 m bin; boxes in bin (tgt PS / GT / src) ==' % s)
    print('bin         rate_PS  rate_GT  rate_TP    PS/GT   nPS   nGT  nsrc')
    for b in range(50):
        if fs[b] <= 0: continue
        r = lambda k: min(tb[k][b] / fs[b], 1) if tb[k][b] > 0 else float('nan')
        ratio = tb['pseudo'][b] / tb['gt'][b] if tb['gt'][b] > 0 else float('nan')
        print('%4.1f-%4.1f   %6.3f   %6.3f   %6.3f   %6.2f %5d %5d %5d' % (
            edges[b], edges[b+1], r('pseudo'), r('gt'), r('tp'), ratio,
            tn['pseudo'][b], tn['gt'][b], nb[b]))
    for k in ['pseudo', 'gt', 'tp']:
        live = (fs > 0) & (tb[k] > 0)
        top = np.argsort(-fs * live)[:3]
        print('%-6s logged-style ratio %.3f | top-3 source bins by pts/box: %s' % (
            k, tb[k][live].sum() / fs[live].sum(),
            ', '.join('%.1fm=%.0f(n=%d)' % (edges[i], fs[i], nb[i]) for i in top)))
