"""Does a pseudo-label's INTENSITY say whether it is real, beyond its score? (S2: KITTI target)

Per pseudo-label box (Car, Pedestrian; positives AND ignored, so the whole score range is seen): score,
points inside, mean normalised intensity, share of zero-intensity returns, range. Matched to real KITTI
boxes for DIAGNOSIS only (same class, BEV centre <= 2 m, greedy by score; Van counts for Car and
Person_sitting for Pedestrian). Reports, per class and range ring, the held-out AUC of TP vs FP for:
score alone, intensity alone, points alone, and a logistic model on (logit score, intensity, zero share,
log points) fitted on even frames and scored on odd frames. If the joint model clearly beats score
alone, a label-free mixture over (score, intensity) has something to separate.

    python analysis/pseudo_label_threshold/score_intensity_joint.py <ps_label.pkl> [shift_z=1.70]
"""
import pickle, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PS = pickle.load(open(sys.argv[1], 'rb'))
SHIFT = float(sys.argv[2]) if len(sys.argv) > 2 else 1.70
infos = pickle.load(open('/home/koyama/code/ST3D/data/kitti/kitti_infos_train.pkl', 'rb'))
MATCH = {1: ('Car', 'Van'), 2: ('Pedestrian', 'Person_sitting')}

def in_box(p, b):
    d = p[:, :3] - b[:3]; c, s = np.cos(-b[6]), np.sin(-b[6])
    x = d[:, 0] * c - d[:, 1] * s; y = d[:, 0] * s + d[:, 1] * c
    return (np.abs(x) <= b[3] / 2) & (np.abs(y) <= b[4] / 2) & (np.abs(d[:, 2]) <= b[5] / 2)

rows = []
for j, i in enumerate(infos):
    fid = i['point_cloud']['lidar_idx']
    if fid not in PS: continue
    p = np.fromfile('/home/koyama/code/ST3D/data/kitti/training/velodyne/%s.bin' % fid, dtype=np.float32).reshape(-1, 4).copy()
    p[:, 2] += SHIFT
    gt = i['annos']['gt_boxes_lidar'].copy(); gt[:, 2] += SHIFT
    gn = i['annos']['name'][:len(gt)]
    b = np.asarray(PS[fid]['gt_boxes']).reshape(-1, 9)
    az = np.degrees(np.arctan2(b[:, 1], b[:, 0]))
    b = b[(b[:, 0] > 0) & (np.abs(az) < 40) & np.isin(np.abs(b[:, 7]), [1, 2])]
    near = p[np.linalg.norm(p[:, :2], axis=1) < 80]
    for cls in (1, 2):
        bc = b[np.abs(b[:, 7]) == cls]; bc = bc[np.argsort(-bc[:, 8])]
        g = gt[np.isin(gn, MATCH[cls])]; used = np.zeros(len(g), bool)
        for k in bc:
            m = in_box(near, k); n = m.sum()
            if n == 0: continue
            tp = 0
            if len(g):
                dd = np.linalg.norm(g[:, :2] - k[:2], axis=1); dd[used] = 1e9; a = dd.argmin()
                if dd[a] <= 2.0: tp = 1; used[a] = True
            it = near[m, 3]
            rows.append((cls, j % 2, k[8], n, it.mean(), (it == 0).mean(), np.linalg.norm(k[:2]), tp))
R = np.array(rows)
print('boxes with >= 1 point: Car %d (TP %.2f), Pedestrian %d (TP %.2f)' % (
    (R[:, 0] == 1).sum(), R[R[:, 0] == 1, 7].mean(), (R[:, 0] == 2).sum(), R[R[:, 0] == 2, 7].mean()))
print('class      ring   |  boxes  TP-share | held-out AUC TP vs FP: score  intensity  points  joint | median intensity TP / FP')
lg = lambda s: np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4)))
for cls, cname in ((1, 'Car'), (2, 'Pedestrian')):
    for lo, hi in ((0, 20), (20, 40), (0, 40)):
        X = R[(R[:, 0] == cls) & (R[:, 6] >= lo) & (R[:, 6] < hi)]
        if len(X) < 200 or X[:, 7].min() == X[:, 7].max(): continue
        F = np.stack([lg(X[:, 2]), X[:, 4], X[:, 5], np.log(X[:, 3])], 1)
        tr, te = X[:, 1] == 0, X[:, 1] == 1
        model = LogisticRegression(max_iter=1000).fit(F[tr], X[tr, 7])
        y = X[te, 7]
        auc = lambda v: roc_auc_score(y, v)
        print('%-10s %2d-%-3d | %6d   %.2f     |        %.3f   %.3f      %.3f   %.3f | %.3f / %.3f' % (
            cname, lo, hi, len(X), X[:, 7].mean(), auc(X[te, 2]), auc(X[te, 4]), auc(np.log(X[te, 3])),
            auc(model.predict_proba(F[te])[:, 1]), np.median(X[X[:, 7] == 1, 4]), np.median(X[X[:, 7] == 0, 4])))

# Selection bias of a STRICTER cut: among REAL objects (TP boxes), how strongly does the score track what
# the statistic measures? Points per box (the density correction) vs mean intensity (the intensity map).
from scipy.stats import spearmanr
print('\nAmong TP boxes, Spearman with score (a stricter cut keeps the high-score end):')
for cls, cname in ((1, 'Car'), (2, 'Pedestrian')):
    for lo, hi in ((0, 20), (20, 40)):
        X = R[(R[:, 0] == cls) & (R[:, 7] == 1) & (R[:, 6] >= lo) & (R[:, 6] < hi)]
        if len(X) < 100: continue
        print('  %-10s %2d-%-3d  TP boxes %5d | points per box %.2f | mean intensity %.2f'
              % (cname, lo, hi, len(X), spearmanr(X[:, 2], X[:, 3])[0], spearmanr(X[:, 2], X[:, 4])[0]))

# LABEL-FREE count estimate: a 2-component Gaussian mixture per class and ring, on score alone vs on
# (score, intensity, points). The component with the higher mean logit score is "real"; its summed
# posterior estimates the number of real objects, and the count-balance cut keeps that many top-scored
# boxes. Real labels are used only to score the estimate.
from sklearn.mixture import GaussianMixture
print('\nLabel-free count estimate (2-GMM) vs real TP count, and the resulting count-balance cut:')
print('class      ring   | real TP | score-only GMM: est  cut   precision | joint GMM: est  cut   precision')
for cls, cname in ((1, 'Car'), (2, 'Pedestrian')):
    for lo, hi in ((0, 20), (20, 40)):
        X = R[(R[:, 0] == cls) & (R[:, 6] >= lo) & (R[:, 6] < hi)]
        if len(X) < 200: continue
        out = []
        for feats in ([lg(X[:, 2])], [lg(X[:, 2]), X[:, 4], X[:, 5], np.log(X[:, 3])]):
            F = np.stack(feats, 1); F = (F - F.mean(0)) / (F.std(0) + 1e-9)
            g = GaussianMixture(2, random_state=0, n_init=3).fit(F)
            real = np.argmax(g.means_[:, 0]); est = g.predict_proba(F)[:, real].sum()
            order = np.argsort(-X[:, 2]); k = int(round(est)); kept = order[:k]
            out.append((est, X[order[k - 1], 2] if k > 0 else 1.0, X[kept, 7].mean() if k > 0 else np.nan))
        print('%-10s %2d-%-3d | %7d | %8.0f  %.2f  %.2f      | %8.0f  %.2f  %.2f' % (
            cname, lo, hi, X[:, 7].sum(), *out[0], *out[1]))
