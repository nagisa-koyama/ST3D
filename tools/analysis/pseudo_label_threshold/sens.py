import os
import pickle, sys, glob, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('h', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness.py')); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
for f in [sys.argv[1] + '/persistence.pkl'] + sorted(glob.glob(sys.argv[1] + '/persistence_*.pkl')):
    h.PE = pickle.load(open(f, 'rb'))
    row = f.split('/')[-1]
    for c in [1, 2]:
        pe = h.PE[(h.PE[:, 0] == c) & (h.PE[:, 1] >= 0)]
        pi, P0, P1 = h.persistence_unmix(pe[:, 1], pe[:, 3])
        nf = len(np.unique(pe[:, 5]))
        row += '  | %s I1 %.2f (P0 %.2f P1 %.2f n %.1f)' % (h.NAMES[c][:3], h.count_balance(pe[:, 1], pi.sum()), P0, P1, pi.sum() / nf)
    print(row)
