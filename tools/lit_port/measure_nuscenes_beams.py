"""Beam elevation modes of nuScenes LIDAR_TOP, measured on UNLABELLED nuScenes TRAIN keyframes, and their comparison with
LiT's hard-coded table (lit/lidar.py::NuScenesLidarIntrinsics.vertical_degrees). For the legal LiT row, experiments_md
20261015_02 item 2. No label of any split is read; the keyframe list is the train infos' lidar paths only.

    python lit_port/measure_nuscenes_beams.py <nuscenes_infos_train.pkl> <nuscenes_root> <n_frames> <out.json>
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

LIT = [10.65520, 9.33780, 7.97498, 6.65758, 5.34018, 3.97735, 2.65996, 1.34256, -0.02027, -1.33767, -2.65507,
       -4.01789, -5.33529, -6.65269, -8.01552, -9.33292, -10.65032, -12.01314, -13.33054, -14.64794, -16.01077,
       -17.32817, -18.64557, -20.00839, -21.32579, -22.64319, -24.00602, -25.32341, -26.64081, -28.00364,
       -29.32104, -30.68386]


def main(infos_path, root, n, out):
    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)
    step = max(1, len(infos) // n)
    sel = infos[::step][:n]
    bw = 0.01                                                      # deg
    edges = np.arange(-40, 20 + bw, bw)
    hist = np.zeros(len(edges) - 1, np.int64)
    for k, inf in enumerate(sel):
        p = Path(root) / inf['lidar_path']
        pts = np.fromfile(p, np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
        rho = np.hypot(pts[:, 0], pts[:, 1])
        m = (rho > 3.0) & (rho < 70.0)                             # LiT's max_range is 70 m
        el = np.degrees(np.arctan2(pts[m, 2], rho[m]))
        hist += np.histogram(el, edges)[0]
    c = 0.5 * (edges[1:] + edges[:-1])
    sm = np.convolve(hist, np.ones(5) / 5, 'same')
    peaks = []
    order = np.argsort(-sm)
    for i in order:                                                # greedy non-maximum suppression, 0.6 deg apart
        if all(abs(c[i] - q) > 0.6 for q in peaks):
            peaks.append(c[i])
        if len(peaks) == 32:
            break
    peaks = np.array(sorted(peaks, reverse=True))
    # refine each mode: weighted mean within +-0.15 deg
    ref = []
    for q in peaks:
        w = np.abs(c - q) <= 0.15
        ref.append(float((c[w] * hist[w]).sum() / hist[w].sum()))
    ref = np.array(ref)
    lit = np.array(LIT)
    d = ref - lit
    res = dict(n_frames=len(sel), modes_deg=ref.tolist(), lit_deg=LIT, diff_deg=d.tolist(),
               max_abs_diff=float(np.abs(d).max()), median_abs_diff=float(np.median(np.abs(d))),
               mean_spacing=float(np.mean(-np.diff(ref))), lit_mean_spacing=float(np.mean(-np.diff(lit))))
    Path(out).write_text(json.dumps(res, indent=1))
    print(f'{len(sel)} frames; max |measured - LiT| {res["max_abs_diff"]:.3f} deg, median {res["median_abs_diff"]:.3f} deg')
    for a, b in zip(ref, lit):
        print(f'{a:9.3f} {b:9.3f} {a - b:+.3f}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
