import os
import pickle, sys, numpy as np
sys.argv = [sys.argv[0], sys.argv[1]]
import importlib.util
spec = importlib.util.spec_from_file_location('h', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness.py')); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
infos = pickle.load(open('/st3d/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl', 'rb'))
loc = np.array([0 if i['lidar_path'].split('/')[-1].startswith('n008') else 1 for i in infos])
SP0, PE0, GT0 = h.SP.copy(), h.PE.copy(), h.GT.copy()
AU = h.AU
# GT records carry no frame -> rebuild GT per frame from score_pts frame ids is impossible; use per-frame GT rows in PE for counts,
# and for t_dist restrict GT via a parallel frame index saved? audit GT lacks frame -> approximate t_dist with all-frame GT
step = len(infos) // 1000
rng = np.random.default_rng(1); half = rng.random(len(infos)) < 0.5
subsets = {'Boston n008': loc == 0, 'Singapore n015': loc == 1, 'random half A': half, 'random half B': ~half}
for name, mask in subsets.items():
    h.SP = SP0[mask[(SP0[:, 5] * step).astype(int)]]
    h.PE = PE0[mask[PE0[:, 5].astype(int)]]
    print('\n#### %s' % name)
    for c in [1, 2]:
        pe = h.PE[h.PE[:, 0] == c]; nf = len(np.unique(pe[:, 5])); ngt = pe[pe[:, 1] < 0, 4].sum()
        t_cb = h.count_balance(pe[pe[:, 1] >= 0, 1], ngt)
        ind = h.indicators(c); diag = ind.pop('_diag')
        print('  %-10s GT/fr %.2f  [VALID t_cb %.2f]  I1 %.2f  I3 %.2f  | %s' % (h.NAMES[c], ngt / nf, t_cb, ind['I1 persist count-balance'], ind['I3 score-mix count-balance'], diag))
