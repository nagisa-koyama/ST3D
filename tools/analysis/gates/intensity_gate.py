"""Intensity-calibration gate, nuScenes -> KITTI, CPU, GT boxes on BOTH sides (diagnosis only).

Q1 Is there a car signal in intensity? AUC of intensity, Car-box points vs background, per range bin,
   in each dataset. ~0.5 = no signal.
Q2 Does foreground information change the mapping? Fit per-range quantile maps source -> target on
   half the frames, apply to the other half's source CAR points, and measure the distance (W1, KITTI
   units) to the held-out target CAR points: (a) raw/255, (b) one map per bin fitted on ALL points,
   (c) a map fitted on car points only. Floor = target car half A vs half B.
   A within-bin global map is monotone, so it keeps the SOURCE's car-vs-background contrast; the
   foreground-aware map imposes the TARGET's. If AUC_s ~ AUC_t and (b) ~ (c), foreground adds nothing.
"""
import pickle
import numpy as np

EDGES = [0, 10, 20, 30, 40, 50, 70]
N = 300
rng = np.random.RandomState(0)

def in_boxes(pts, boxes):
    m = np.zeros(len(pts), bool)
    for b in boxes:
        d = pts[:, :3] - b[:3]; c, s = np.cos(-b[6]), np.sin(-b[6])
        x = d[:, 0] * c - d[:, 1] * s; y = d[:, 0] * s + d[:, 1] * c
        m |= (np.abs(x) <= b[3] / 2) & (np.abs(y) <= b[4] / 2) & (np.abs(d[:, 2]) <= b[5] / 2)
    return m

def collect(frames):
    """list of (range, intensity, is_car, is_bg, above_ground, frame parity)"""
    out = []
    for fi, (pts, car_boxes, all_boxes) in enumerate(frames):
        r = np.linalg.norm(pts[:, :2], axis=1)
        car = in_boxes(pts, car_boxes); anyb = in_boxes(pts, all_boxes) | car
        keep = r < 70
        out.append(np.stack([r, pts[:, 3], car, ~anyb, pts[:, 2] > -1.3, np.full(len(pts), fi % 2)], 1)[keep])
    return np.concatenate(out)

def nuscenes():
    infos = pickle.load(open('/home/koyama/code/ST3D/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl', 'rb'))
    fr = []
    for i in infos[::len(infos) // N][:N]:
        p = np.fromfile('/home/koyama/code/ST3D/data/nuscenes/v1.0-trainval/' + i['lidar_path'], dtype=np.float32).reshape(-1, 5)[:, :4]
        p = p[np.linalg.norm(p[:, :2], axis=1) > 1.5]                     # ego points, as the loader removes
        names = np.asarray(i['gt_names']); b = i['gt_boxes'][:, :7] if len(names) else np.zeros((0, 7))
        fr.append((p, b[names == 'car'], b))
    return collect(fr)

def kitti():
    infos = pickle.load(open('/home/koyama/code/ST3D/data/kitti/kitti_infos_train.pkl', 'rb'))
    fr = []
    for i in infos[::len(infos) // N][:N]:
        p = np.fromfile('/home/koyama/code/ST3D/data/kitti/training/velodyne/%s.bin' % i['point_cloud']['lidar_idx'], dtype=np.float32).reshape(-1, 4)
        p = p[(p[:, 0] > 0) & (np.abs(np.degrees(np.arctan2(p[:, 1], p[:, 0]))) < 40)]   # annotated only in the camera FOV
        b = i['annos']['gt_boxes_lidar']; names = i['annos']['name'][:len(b)]
        fr.append((p, b[names == 'Car'], b))
    return collect(fr)

def auc(pos, neg):
    if len(pos) < 50 or len(neg) < 50: return np.nan
    neg = np.sort(rng.choice(neg, min(len(neg), 200000), replace=False))
    lo = np.searchsorted(neg, pos, 'left'); hi = np.searchsorted(neg, pos, 'right')
    return np.mean((lo + hi) / 2.0) / len(neg)

def qmap(fit_s, fit_t):
    s = np.sort(fit_s); q = np.linspace(0, 1, 201); tq = np.quantile(fit_t, q)
    return lambda x: np.interp((np.searchsorted(s, x, 'left') + np.searchsorted(s, x, 'right')) / 2.0 / len(s), q, tq)

def w1(a, b):
    q = np.linspace(0, 1, 201)
    return np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q)))

S, T = nuscenes(), kitti()
print('points kept: nuScenes %d (car %d), KITTI %d (car %d)' % (len(S), S[:, 2].sum(), len(T), T[:, 2].sum()))
print('\nring  | car pts s/t   | AUC car vs bg  s / t | AUC car vs above-ground bg s / t | W1 to KITTI car (held out): raw/255  global-map  car-map  floor')
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    s = S[(S[:, 0] >= lo) & (S[:, 0] < hi)]; t = T[(T[:, 0] >= lo) & (T[:, 0] < hi)]
    sc, sb, sg = s[s[:, 2] == 1], s[s[:, 3] == 1], s[(s[:, 3] == 1) & (s[:, 4] == 1)]
    tc, tb, tg = t[t[:, 2] == 1], t[t[:, 3] == 1], t[(t[:, 3] == 1) & (t[:, 4] == 1)]
    A = lambda x: x[x[:, 5] == 0, 1]; B = lambda x: x[x[:, 5] == 1, 1]
    if min(len(A(sc)), len(B(sc)), len(A(tc)), len(B(tc))) < 50:
        print('%2d-%-3d | too few car points' % (lo, hi)); continue
    g = qmap(A(s), A(t)); c = qmap(A(sc), A(tc))
    print('%2d-%-3d | %6d/%6d | %.3f / %.3f          | %.3f / %.3f                      | %.3f    %.3f       %.3f    %.3f'
          % (lo, hi, len(sc), len(tc), auc(sc[:, 1], sb[:, 1]), auc(tc[:, 1], tb[:, 1]),
             auc(sc[:, 1], sg[:, 1]), auc(tc[:, 1], tg[:, 1]),
             w1(B(sc) / 255.0, B(tc)), w1(g(B(sc)), B(tc)), w1(c(B(sc)), B(tc)), w1(A(tc), B(tc))))
for name, x in (('nuScenes', S), ('KITTI', T)):
    c, b = x[x[:, 2] == 1, 1], x[x[:, 3] == 1, 1]
    print('%-8s intensity car p25/p50/p75 %s | background %s' % (name, np.round(np.percentile(c, [25, 50, 75]), 3), np.round(np.percentile(b, [25, 50, 75]), 3)))
