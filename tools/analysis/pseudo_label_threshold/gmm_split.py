import pickle, sys, numpy as np, importlib.util
S = sys.argv[1]
spec = importlib.util.spec_from_file_location('o', S + '/otsu.py')
src = open(S + '/otsu.py').read().split('K = pickle')[0]; exec(src)
infos = pickle.load(open('/st3d/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl', 'rb'))
loc = np.array([i['lidar_path'].split('/')[-1].startswith('n008') for i in infos])
pe = h.PE; rng = np.random.default_rng(3)
half = rng.random(len(infos)) < .5
for name, mask in [('Boston', loc), ('Singapore', ~loc), ('half A', half), ('half B', ~half)]:
    x = pe[mask[pe[:, 5].astype(int)]]
    b = x[(x[:, 0] == 1) & (x[:, 1] >= 0)]; ngt = x[(x[:, 0] == 1) & (x[:, 1] < 0), 4].sum()
    tp, tc = gmm_logit(b[:, 1])
    print('nuScenes %-10s Car [VALID t_cb %.2f]  GMM post %.2f  cb %.2f' % (name, h.count_balance(b[:, 1], ngt), tp, tc))
K = pickle.load(open(S + '/kitti_oos.pkl', 'rb')); r, g = K['rows'], K['gt']
fr = r[:, 5]; nfr = int(fr.max()) + 1; hk = rng.random(nfr) < .5
for name, sel in [('KITTI half A', hk), ('KITTI half B', ~hk)]:
    b = r[(r[:, 0] == 3) & sel[fr.astype(int)]]
    ngt = len(g[g[:, 0] == 3]) * sel.mean()
    tp, tc = gmm_logit(b[:, 1])
    print('%-19s Car [VALID t_cb ~%.2f]  GMM post %.2f  cb %.2f' % (name, h.count_balance(b[:, 1], ngt), tp, tc))
boots = []
b = pe[(pe[:, 0] == 1) & (pe[:, 1] >= 0)]; frs = np.unique(b[:, 5]); by = {f: np.where(b[:, 5] == f)[0] for f in frs}
for _ in range(50):
    idx = np.concatenate([by[f] for f in rng.choice(frs, len(frs))]); boots.append(gmm_logit(b[idx, 1]))
boots = np.array(boots); print('nuScenes Car bootstrap 5-95%%: post %.2f-%.2f  cb %.2f-%.2f' % (*np.percentile(boots[:, 0], [5, 95]), *np.percentile(boots[:, 1], [5, 95])))
