import pickle, sys, numpy as np
d = pickle.load(open(sys.argv[1] + '/audit_full.pkl', 'rb'))
for k in ['gt', 'pseudo', 'tp', 'fp']:
    fg, nb, fr = d['target_bins'][k]
    a = np.array(d['target'][k], dtype=float).reshape(-1, 4)
    print('%-7s sum of per-box counts / distinct fg points = %.2f' % (k, a[:, 2].sum() / fg.sum()))
for k, (fg, nb, fr) in d['source_bins'].items():
    a = np.array(d['source'][k], dtype=float).reshape(-1, 4)
    print('%-12s %.2f' % (k, a[:, 2].sum() / fg.sum()))
