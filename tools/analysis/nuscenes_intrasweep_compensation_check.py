"""Are nuScenes LIDAR_TOP points compensated for ego motion WITHIN a sweep? (experiments_md 20261007_03 §12)

Points are stored in firing order and one sweep covers ~368 deg, so at the seam each azimuth is fired twice, ~50 ms
apart. The sensor moves v*T between the two. For a matched (ring, azimuth) pair, the second hit minus the first,
projected on the motion direction perpendicular to the ray, is ~ v*T if the points are expressed in ONE frame
(compensated) and ~0 if each point is relative to the sensor at its own capture time (uncompensated). Motion direction
and speed from the previous sweep's transform. Label-free.

    python analysis/nuscenes_intrasweep_compensation_check.py <n_frames>
"""
import sys, pickle
import numpy as np

ROOT = '/home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval/'
infos = pickle.load(open(ROOT + 'nuscenes_infos_10sweeps_train.pkl', 'rb'))
n = int(sys.argv[1])
from scipy.spatial import cKDTree
SHIFTS = np.arange(-0.8, 0.8001, 0.02)
rows = []
for info in infos[::max(1, len(infos) // n)][:n]:
    sw = info['sweeps']
    if not sw or sw[0]['transform_matrix'] is None or sw[0]['time_lag'] <= 0:
        continue
    t = sw[0]['transform_matrix'][:3, 3]; v = np.linalg.norm(t[:2]) / sw[0]['time_lag']
    m = -t[:2] / max(np.linalg.norm(t[:2]), 1e-9); m = np.array([m[0], m[1], 0.0])
    p = np.fromfile(ROOT + info['lidar_path'], dtype=np.float32).reshape(-1, 5)
    S, E = [], []
    for r in np.unique(p[:, 4]).astype(int):
        q = p[p[:, 4] == r]
        if len(q) < 100:
            continue
        rel = np.degrees(np.unwrap(np.arctan2(q[:, 1], q[:, 0])))
        rel = np.abs(rel - rel[0]); sweep = rel[-1]
        if sweep < 361:
            continue
        over = sweep - 360
        S.append(q[rel < over, :3]); E.append(q[rel > 360, :3])
    if not S:
        continue
    S = np.concatenate(S); E = np.concatenate(E)
    keepS = (np.linalg.norm(S[:, :2], axis=1) > 2) & (np.linalg.norm(S[:, :2], axis=1) < 30)
    keepE = (np.linalg.norm(E[:, :2], axis=1) > 2) & (np.linalg.norm(E[:, :2], axis=1) < 30)
    S, E = S[keepS], E[keepE]
    if len(S) < 50 or len(E) < 50:
        continue
    tree = cKDTree(S)
    cost = [np.mean(np.minimum(tree.query(E + s * m)[0], 0.3)) for s in SHIFTS]
    k = int(np.argmin(cost)); depth = np.median(cost) - cost[k]
    rows.append((v, SHIFTS[k], depth))
rows = np.array(rows)
print(f'frames: {len(rows)}; best shift of the end-of-sweep copy along the motion direction (m) that aligns it with the start copy')
for lo, hi in [(0, 0.5), (0.5, 5), (5, 10), (10, 30)]:
    s = (rows[:, 0] >= lo) & (rows[:, 0] < hi) & (rows[:, 2] > 0.01)
    if s.any():
        print(f'speed {lo}-{hi} m/s: frames {s.sum()} (clear minimum), median speed {np.median(rows[s,0]):.1f}: best shift median {np.median(rows[s,1]):+.2f} m '
              f'(IQR {np.percentile(rows[s,1],25):+.2f} .. {np.percentile(rows[s,1],75):+.2f}); v*0.05 s = {np.median(rows[s,0])*0.05:.2f} m')
