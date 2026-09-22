"""End-to-end check that GT_BOXES_MOTION_COMPENSATION does what it claims, on the devkit datasets.

Covers nuScenes and Lyft - the two that annotate at 2 Hz keyframes with points arriving between
them, so their compensator has to INTERPOLATE a sweep's boxes. (Waymo and PandaSet annotate every
frame and need no interpolation; they have their own script, frame_accumulation_effect.py.)

Replicates <Dataset>.get_lidar_with_sweeps (same ego transform, same ego-point removal, same
compensator) and counts points inside each anchor Car box with the compensation on and off.

Two rules this obeys, both learned the hard way (experiments_md/20260922_05, Ablation 7):

  - Normalise each object against its OWN N=1 count. Scoring a moving object against its
    uncompensated ACCUMULATED count measures accidental overlap with the anchor box, not a
    density, and manufactured a 2.00-2.83x "overshoot" that survived four wrong explanations.
  - Compare moving against static PER OBJECT. A driving car returns ~39 points/frame against a
    parked car's 19 at the same range - parked cars shadow each other at the kerb - so the two
    are different populations and an unpaired comparison says nothing.

Expected shape of the result: static is untouched (compensation is a no-op when the box does not
move) and moving rises to meet it. Anything else means the timeline is wrong - which is exactly how
the unit bug in _timestamp_seconds was found, when moving sat at 0.51 of static and the median
displacement was identical at 0.25 s, 0.50 s and 0.75 s of lag.

  python3 analysis/motion_compensation_effect.py --dataset nuscenes [--sweeps 15]
  python3 analysis/motion_compensation_effect.py --dataset lyft --sweeps 5
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _init_path  # noqa: F401,E402
from pcdet.datasets.motion_compensation import (  # noqa: E402
    DevkitSweepCompensator, find_devkit_meta_dir)


# Per-dataset facts the shared code needs. The ego radii are not cosmetic: nuScenes strips 26% of
# its in-range cloud as roof self-returns and no other loader does it at all, so borrowing its
# numbers for Lyft would delete real returns.
DATASETS = {
    'nuscenes': dict(root='../data/nuscenes/v1.0-trainval', infos='nuscenes_infos_200sweeps_train.pkl',
                     version='v1.0-trainval', ego=(1.5, 1.0), car='car', width=5),
    'lyft': dict(root='../data/lyft/trainval', infos='lyft_infos_train.pkl',
                 version='trainval', ego=(0.0, 0.0), car='car', width=5),
}


def remove_ego_points(points, radius):
    if radius <= 0:
        return points
    return points[~((np.abs(points[:, 0]) < radius) & (np.abs(points[:, 1]) < radius))]


def _read(path, width):
    pts = np.fromfile(str(path), dtype=np.float32)
    # Lyft ships a handful of clouds whose byte count is not a whole number of points; its loader
    # truncates rather than failing, so this must too or the comparison stops at the first one.
    pts = pts[:len(pts) - (len(pts) % width)]
    return pts.reshape(-1, width)[:, :4]


def load_points(root, info, n, compensator, spec):
    """The anchor plus n-1 sweeps in the anchor frame, optionally per-object compensated."""
    pts = _read(root / info['lidar_path'], spec['width'])
    out = [remove_ego_points(pts, spec['ego'][0])]
    for k in range(n - 1):
        sweep = info['sweeps'][k]
        q = _read(root / sweep['lidar_path'], spec['width'])
        q = remove_ego_points(q, spec['ego'][1])
        if sweep['transform_matrix'] is not None:
            q[:, :3] = (sweep['transform_matrix'] @ np.vstack((q[:, :3].T, np.ones(len(q)))))[:3].T
        if compensator is not None:
            q = compensator.compensate_sweep(info, sweep, q)
        out.append(q)
    return np.concatenate(out)


def count_in_box(points, b):
    d = points[:, :3] - b[:3]
    c, s = np.cos(-b[6]), np.sin(-b[6])
    lx, ly = d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c
    return int(((np.abs(lx) <= b[3] / 2) & (np.abs(ly) <= b[4] / 2)
                & (np.abs(d[:, 2]) <= b[5] / 2)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=sorted(DATASETS), default='nuscenes')
    ap.add_argument('--root', default=None)
    ap.add_argument('--infos', default=None)
    ap.add_argument('--sweeps', type=int, default=15)
    ap.add_argument('--anchors', type=int, default=60)
    ap.add_argument('--moved', type=float, default=1.0, help='metres over the deepest lag')
    ap.add_argument('--seed', type=int, default=3)
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    root = Path(args.root or spec['root'])
    infos = pickle.load(open(root / (args.infos or spec['infos']), 'rb'))
    comp = DevkitSweepCompensator(infos, find_devkit_meta_dir(root, spec['version']),
                                  classes={spec['car']})
    n = args.sweeps

    # position within the scene, so anchors too near its start (where boxes_at clamps to the first
    # keyframe and every displacement is zero by construction) can be skipped
    pos = {j: r for order in comp.scene_order.values() for r, j in enumerate(order)}

    rows, used, lag = [], 0, 0.0
    for ii in np.random.default_rng(args.seed).permutation(len(infos)):
        info = infos[ii]
        i = comp.tok2frame[info['token']]
        if pos[i] < n or len(info.get('sweeps', [])) < n - 1:
            continue
        S = np.asarray(info['ref_from_car']) @ np.asarray(info['car_from_global'])
        t = comp.frames[i]['t']
        now = comp.boxes_at(info['token'], t, S)
        if not now:
            continue
        lag = float(info['sweeps'][n - 2]['time_lag'])
        then = comp.boxes_at(info['token'], t - lag, S)
        p1 = load_points(root, info, 1, None, spec)
        pu = load_points(root, info, n, None, spec)
        pc = load_points(root, info, n, comp, spec)
        for tid, b in now.items():
            if tid not in then:
                continue
            n1 = count_in_box(p1, b)
            if n1 < 5:
                continue        # too few points at N=1 for a ratio to mean anything
            rows.append((np.linalg.norm(b[:3] - then[tid][:3]), n1,
                         count_in_box(pu, b), count_in_box(pc, b)))
        used += 1
        if used >= args.anchors:
            break

    a = np.array(rows, dtype=float)
    moving = a[:, 0] > args.moved
    print('%s N=%d, %d anchors, %d Car boxes (%d moving >%.1f m over %.2f s)'
          % (args.dataset, n, used, len(a), moving.sum(), args.moved, lag))
    print('%-8s %7s %8s %10s %8s %10s %9s'
          % ('group', 'boxes', 'N=1', 'uncomp', 'comp', 'uncomp/N1', 'comp/N1'))
    for name, m in (('static', ~moving), ('moving', moving), ('all', np.ones(len(a), bool))):
        g = a[m]
        if not len(g):
            continue
        print('%-8s %7d %8.0f %10.0f %8.0f %10.2f %9.2f'
              % (name, len(g), np.median(g[:, 1]), np.median(g[:, 2]), np.median(g[:, 3]),
                 np.median(g[:, 2] / g[:, 1]), np.median(g[:, 3] / g[:, 1])))
    if moving.any() and (~moving).any():
        r = lambda m, c: np.median(a[m][:, c] / a[m][:, 1])  # noqa: E731
        print('\nmoving / static:  uncompensated %.2f  ->  compensated %.2f'
              % (r(moving, 2) / r(~moving, 2), r(moving, 3) / r(~moving, 3)))


if __name__ == '__main__':
    main()
