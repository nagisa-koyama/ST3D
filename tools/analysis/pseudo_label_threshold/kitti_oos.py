import os
"""Out-of-sample: Lyft->KITTI SECOND teacher (p64jmfqs, ps_label_e0). Non-temporal indicators only.
Both pseudo-labels and GT restricted to the camera FOV (|azimuth| < 40 deg, x > 0) and < 70 m,
because KITTI annotates only there - a sensor/protocol restriction, not a label."""
import pickle, sys, numpy as np, importlib.util
from pathlib import Path
S = sys.argv[1]
spec = importlib.util.spec_from_file_location('h', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness.py')); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
sys.path.insert(0, '/st3d')
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

ps = pickle.load(open('/storage/wandb/run-20260726_163507-p64jmfqs/files/ps_label/ps_label_e0.pkl', 'rb'))
infos = {i['point_cloud']['lidar_idx']: i for i in pickle.load(open('/st3d/data/kitti/kitti_infos_train.pkl', 'rb'))}
SHIFT = np.array([0, 0, 1.7])
CL = {3: 'Car', 4: 'Pedestrian'}


def fov(xy):
    return (xy[:, 0] > 0) & (np.abs(np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))) < 40) & (np.linalg.norm(xy, axis=1) < 70)


rows, gtrows = [], []   # rows: cls, score, npts, range, tp, frame ; gtrows: cls, npts, range
for f, (fid, v) in enumerate(ps.items()):
    b = v['gt_boxes']; b = b[b[:, 7] > 0] if len(b) else b.reshape(0, 9)
    inf = infos[fid]; an = inf['annos']
    pts = np.fromfile('/st3d/data/kitti/training/velodyne/%s.bin' % fid, dtype=np.float32).reshape(-1, 4)[:, :3] + SHIFT
    g = an['gt_boxes_lidar'].copy(); g[:, :3] += SHIFT
    gn = an['name'][an['name'] != 'DontCare'][:len(g)]
    for c, name in CL.items():
        x = b[(b[:, 7] == c)]
        x = x[fov(x[:, :2])] if len(x) else x
        gg = g[(gn == name)]; gg = gg[fov(gg[:, :2])] if len(gg) else gg
        if len(gg):
            ng = roiaware_pool3d_utils.points_in_boxes_cpu(pts, gg[:, :7].astype(np.float32)).sum(1)
            for bx, n in zip(gg, ng):
                if n >= 1: gtrows.append((c, n, np.linalg.norm(bx[:2])))
        if not len(x): continue
        nb = roiaware_pool3d_utils.points_in_boxes_cpu(pts, x[:, :7].astype(np.float32)).sum(1)
        tp = np.zeros(len(x), bool); used = np.zeros(len(gg), bool)
        for k in np.argsort(-x[:, 8]):
            if not len(gg): break
            d = np.linalg.norm(gg[:, :2] - x[k, :2], axis=1); d[used] = np.inf; j = np.argmin(d)
            if d[j] <= 2.0: tp[k] = used[j] = True
        for bx, n, t in zip(x, nb, tp):
            if n >= 1: rows.append((c, bx[8], n, np.linalg.norm(bx[:2]), t, f))
rows, gtrows = np.array(rows, float), np.array(gtrows, float)
nf = len(ps)
R = [0, 10, 20, 30, 40, 50, 70]
for c, name in CL.items():
    b = rows[rows[:, 0] == c]; g = gtrows[gtrows[:, 0] == c]; s = b[:, 1]
    def tv(x):
        hx = np.histogram(x, R)[0] / len(x); hy = np.histogram(g[:, 2], R)[0] / len(g); return .5 * np.abs(hx - hy).sum()
    def w1(x, y):
        q = np.linspace(.005, .995, 199); return np.abs(np.quantile(x, q) - np.quantile(y, q)).mean()
    cost = [tv(b[s >= t, 3]) + w1(np.log(b[s >= t, 2]), np.log(g[:, 1])) if (s >= t).sum() > 50 else np.inf for t in h.GRID]
    t_dist = h.GRID[int(np.argmin(cost))]; t_cb = h.count_balance(s, len(g))
    w, l1, l2, post = h.trunc_exp_mix(s)
    srt = np.argsort(s)
    print('\n%s: pseudo %.1f/fr, GT %.2f/fr, prec@0.1 %.2f  [VALID refs t_dist %.2f, t_cb %.2f]' % (name, len(b) / nf, len(g) / nf, b[:, 4].mean(), t_dist, t_cb))
    print('   I3 score-mix count-balance %.2f (w_true %.2f, lam %.1f/%.1f, n_true/fr %.2f)' % (h.count_balance(s, w * len(s)), w, l1, l2, w * len(s) / nf))
    print('   I4 score-mix post=0.5      %.2f' % (s[srt][np.argmax(post[srt] >= .5)] if (post >= .5).any() else np.nan))
    print('   I5 kneedle                 %.2f' % h.kneedle(s))
pickle.dump({'rows': rows, 'gt': gtrows}, open(S + '/kitti_oos.pkl', 'wb'))
