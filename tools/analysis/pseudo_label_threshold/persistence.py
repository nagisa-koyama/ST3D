"""Label-free threshold indicators for pseudo-labels.

Selection uses ONLY: teacher boxes/scores (ps_label_e0.pkl) + ego poses/timestamps from the infos.
Target GT (info['gt_boxes']) is read into a separate array used for VALIDATION columns only.

Temporal persistence: a box at p (global frame) in keyframe i is 'persistent' if the previous and
next keyframes (0.5 s either side, same scene) hold same-class boxes q-, q+ with
|q- + q+ - 2p| < TOL (constant velocity, which includes static) and |q+ - q-| < VMAX*dt_total.
"""
import pickle, sys
import numpy as np

S = sys.argv[1]
ps = pickle.load(open('/storage/wandb/run-20260926_102950-uq83obp7/files/ps_label/ps_label_e0.pkl', 'rb'))
infos = pickle.load(open('/storage/../koyama_st3d_infos.pkl', 'rb')) if False else \
    pickle.load(open('/st3d/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl', 'rb'))
SHIFT_Z = 1.75
TOL, VMAX, RMAX = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5, float(sys.argv[3]) if len(sys.argv) > 3 else 20.0, 70.0
CLASSES = {1: 'car', 2: 'pedestrian', 3: 'bicycle'}


def fid(i):
    return infos[i]['lidar_path'].split('/')[-1][:-4]


def to_global(i, xy):
    inf = infos[i]
    T = np.linalg.inv(inf['car_from_global']) @ np.linalg.inv(inf['ref_from_car'])
    P = np.column_stack([xy, np.zeros(len(xy)), np.ones(len(xy))])
    return (T @ P.T).T[:, :2]


def boxes(i):
    k = fid(i)
    if k not in ps:
        return None
    b = ps[k]['gt_boxes']
    return b[b[:, 7] > 0] if len(b) else b.reshape(0, 9)


def neighbour_ok(i, j):
    return 0 <= j < len(infos) and 0.3 < abs(infos[j]['timestamp'] - infos[i]['timestamp']) < 0.7


step = len(infos) // 1000
rows = []   # cls, score, range, persistent, tp
for i in range(0, len(infos), step)[:1000]:
    if not (neighbour_ok(i, i - 1) and neighbour_ok(i, i + 1)):
        continue
    b0, bp, bn = boxes(i), boxes(i - 1), boxes(i + 1)
    if b0 is None or bp is None or bn is None or not len(b0):
        continue
    r = np.linalg.norm(b0[:, :2], axis=1)
    g0, gp, gn = to_global(i, b0[:, :2]), to_global(i - 1, bp[:, :2]), to_global(i + 1, bn[:, :2])
    dt = infos[i + 1]['timestamp'] - infos[i - 1]['timestamp']
    # validation only: real GT in the lidar frame (boxes above are shifted by SHIFT_COOR z only)
    inf = infos[i]
    gt_xy, gt_nm = inf['gt_boxes'][:, :2], inf['gt_names']
    gt_ok = inf['num_lidar_pts'] >= 1
    for c, name in CLASSES.items():
        m0 = (b0[:, 7] == c) & (r < RMAX)
        if not m0.any():
            continue
        qp, qn = gp[bp[:, 7] == c], gn[bn[:, 7] == c]
        # persistence (label-free)
        pers = np.zeros(m0.sum(), dtype=bool)
        for k, p in enumerate(g0[m0]):
            if len(qp) and len(qn):
                cp = qp[np.linalg.norm(qp - p, axis=1) < VMAX * dt / 2 + TOL]
                cn = qn[np.linalg.norm(qn - p, axis=1) < VMAX * dt / 2 + TOL]
                if len(cp) and len(cn):
                    mid = (cp[:, None, :] + cn[None, :, :]) / 2
                    pers[k] = (np.linalg.norm(mid - p, axis=2) < TOL).any()
        # validation: greedy by score, 2 m, same class
        sc = b0[m0, 8]; xy = b0[m0, :2]
        gsel = np.where((gt_nm == name) & gt_ok)[0]; used = np.zeros(len(gsel), bool)
        tp = np.zeros(m0.sum(), dtype=bool)
        for k in np.argsort(-sc):
            if not len(gsel):
                break
            d = np.linalg.norm(gt_xy[gsel] - xy[k], axis=1); d[used] = np.inf
            j = np.argmin(d)
            if d[j] <= 2.0:
                tp[k] = True; used[j] = True
        n_gt = ((gt_nm == name) & gt_ok & (np.linalg.norm(gt_xy, axis=1) < RMAX)).sum()
        rows += [(c, s, rr, pp, t, i) for s, rr, pp, t in zip(sc, r[m0], pers, tp)]
        rows.append((c, -1, -1, -1, n_gt, i))   # per-frame GT count marker, validation only
pickle.dump(np.array(rows, dtype=float), open(S + '/persistence%s.pkl' % ('' if len(sys.argv) < 3 else '_%s_%s' % (sys.argv[2], sys.argv[3])), 'wb'))
print('anchors used:', len(np.unique(np.array(rows)[:, 5])))
