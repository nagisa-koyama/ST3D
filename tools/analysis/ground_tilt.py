"""Per-frame ground plane in each loader's point frame, against what the extrinsic predicts.

Why: nuScenes' LIDAR_TOP calibration carries a 1.4-2.7 deg pitch relative to the vehicle and Lyft's
1.3 deg, and both loaders keep points in the SENSOR frame, so the ground the detector sees may slope
along the driving direction by a dataset-specific amount. SHIFT_COOR only fixes the height. The
earlier road-height estimates (20260922_03) were percentiles over an annulus, which average a tilt
out, so they cannot see this.

For each frame: horizontal range 4-40 m, the lowest return per 1 m BEV cell, RANSAC plane
z = a*x + b*y + c, least-squares refit on the inliers. Slopes are reported along the VEHICLE's
forward and left axes (from ref_from_car where the dataset has it; KITTI's velodyne x is forward),
in degrees, positive = ground rises in that direction. The prediction is the ego frame's z = 0
plane expressed in the sensor frame (third column of ref_from_car's rotation).

Raw points read straight from the files the infos point at, so no SHIFT_COOR is applied; nuScenes
gets remove_ego_points(1.5) as its loader does.

    python analysis/ground_tilt.py --dataset nuscenes --frames 600
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

ROOT = Path('/home/koyama/code/ST3D/data')


def lowest_per_cell(pts, rmin=4.0, rmax=40.0, cell=1.0, min_pts=3):
    r = np.hypot(pts[:, 0], pts[:, 1])
    p = pts[(r >= rmin) & (r <= rmax)]
    if len(p) == 0:
        return np.zeros((0, 3))
    ij = np.floor(p[:, :2] / cell).astype(np.int64)
    key = (ij[:, 0] + 10000) * 100000 + (ij[:, 1] + 10000)
    order = np.lexsort((p[:, 2], key))
    key, p = key[order], p[order]
    first = np.r_[True, key[1:] != key[:-1]]
    starts = np.flatnonzero(first)
    counts = np.diff(np.r_[starts, len(key)])
    # second-lowest return per cell (robust to a single low outlier)
    take = starts + np.minimum(1, counts - 1)
    keep = counts >= min_pts
    return p[take[keep], :3]


def fit_plane(g, iters=300, thr=0.12, rng=None):
    if len(g) < 30:
        return None
    rng = rng or np.random.default_rng(0)
    A = np.c_[g[:, 0], g[:, 1], np.ones(len(g))]
    best, best_n = None, -1
    for _ in range(iters):
        idx = rng.choice(len(g), 3, replace=False)
        try:
            coef = np.linalg.solve(A[idx], g[idx, 2])
        except np.linalg.LinAlgError:
            continue
        if abs(coef[0]) > 0.2 or abs(coef[1]) > 0.2:  # > 11 deg: not a road
            continue
        n = np.sum(np.abs(A @ coef - g[:, 2]) < thr)
        if n > best_n:
            best, best_n = coef, n
    if best is None:
        return None
    for t in (thr, 0.08):
        inl = np.abs(A @ best - g[:, 2]) < t
        best = np.linalg.lstsq(A[inl], g[inl, 2], rcond=None)[0]
    return best, int(inl.sum()), len(g)


def frames(dataset, n):
    if dataset == 'nuscenes':
        infos = pickle.load(open(ROOT / 'nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl', 'rb'))
        for i in np.linspace(0, len(infos) - 1, n).astype(int):
            info = infos[i]
            pts = np.fromfile(ROOT / 'nuscenes/v1.0-trainval' / info['lidar_path'], dtype=np.float32).reshape(-1, 5)
            pts = pts[~((np.abs(pts[:, 0]) < 1.5) & (np.abs(pts[:, 1]) < 1.5))]
            group = info['lidar_path'].split('/')[-1][:4]  # n008 Boston / n015 Singapore
            yield pts, info['ref_from_car'], group
    elif dataset == 'lyft':
        infos = pickle.load(open(ROOT / 'lyft/trainval/lyft_infos_val.pkl', 'rb'))
        for i in np.linspace(0, len(infos) - 1, n).astype(int):
            info = infos[i]
            raw = np.fromfile(ROOT / 'lyft/trainval' / info['lidar_path'], dtype=np.float32)
            pts = raw[: (len(raw) // 5) * 5].reshape(-1, 5)
            host = info['lidar_path'].split('/')[-1].split('_')[0]
            group = '64-beam' if host in ('host-a101', 'host-a102') else '40-beam'
            yield pts, info['ref_from_car'], group
    elif dataset == 'kitti':
        infos = pickle.load(open(ROOT / 'kitti/kitti_infos_val.pkl', 'rb'))
        for i in np.linspace(0, len(infos) - 1, n).astype(int):
            idx = infos[i]['point_cloud']['lidar_idx']
            pts = np.fromfile(ROOT / f'kitti/training/velodyne/{idx}.bin', dtype=np.float32).reshape(-1, 4)
            yield pts, None, 'KITTI'
    elif dataset == 'waymo':
        seqs = sorted((ROOT / 'waymo/waymo_processed_data').glob('segment-*'))
        val = set(l.strip().replace('.tfrecord', '') for l in open(ROOT / 'waymo/ImageSets/val.txt'))
        seqs = [s for s in seqs if s.name in val]
        per = max(1, n // len(seqs)) if seqs else 0
        rng = np.random.default_rng(0)
        for s in rng.choice(seqs, min(len(seqs), n), replace=False):
            f = sorted(s.glob('*.npy'))
            if not f:
                continue
            pts = np.load(f[len(f) // 2])[:, :3]
            yield pts, None, 'Waymo (vehicle frame)'
            if per < 1:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['nuscenes', 'lyft', 'kitti', 'waymo'])
    ap.add_argument('--frames', type=int, default=400)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = []
    for pts, rfc, group in frames(a.dataset, a.frames):
        res = fit_plane(lowest_per_cell(pts))
        if res is None:
            continue
        (sa, sb, c), ninl, ncell = res
        if rfc is not None:
            R, T = rfc[:3, :3], rfc[:3, 3]
            fwd, left, up = R[:, 0], R[:, 1], R[:, 2]  # vehicle axes in the sensor frame
            pred = (-up[0] / up[2], -up[1] / up[2], float(up @ T) / up[2])
        else:
            fwd, left = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
            pred = (np.nan, np.nan, np.nan)
        grad = np.array([sa, sb, 0.0])
        meas_f, meas_l = grad @ fwd, grad @ left
        pg = np.array([pred[0], pred[1], 0.0])
        pred_f, pred_l = (pg @ fwd, pg @ left) if rfc is not None else (np.nan, np.nan)
        rows.append((group, np.degrees(np.arctan(meas_f)), np.degrees(np.arctan(meas_l)), c,
                     np.degrees(np.arctan(pred_f)), np.degrees(np.arctan(pred_l)), pred[2], ninl / ncell))
    if a.out:
        np.save(a.out, np.array(rows, dtype=object))
    groups = sorted(set(r[0] for r in rows))
    print(f'{a.dataset}: {len(rows)} frames fitted')
    print(f'{"group":24s} {"n":>4s} | {"fwd slope deg (meas)":>22s} {"pred":>6s} | {"left slope deg (meas)":>22s} {"pred":>6s} | {"height c (meas)":>18s} {"pred":>6s} | inl')
    for gname in groups:
        g = np.array([r[1:] for r in rows if r[0] == gname], dtype=float)
        q = lambda v: f'{np.median(v):+6.2f} [{np.percentile(v, 25):+5.2f},{np.percentile(v, 75):+5.2f}]'
        print(f'{gname:24s} {len(g):4d} | {q(g[:, 0]):>22s} {np.nanmedian(g[:, 3]):+6.2f} | {q(g[:, 1]):>22s} '
              f'{np.nanmedian(g[:, 4]):+6.2f} | {q(g[:, 2]):>18s} {np.nanmedian(g[:, 5]):+6.2f} | {np.median(g[:, 6]):.2f}')


if __name__ == '__main__':
    main()
