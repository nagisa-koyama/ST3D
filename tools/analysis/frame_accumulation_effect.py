"""What neighbouring-frame accumulation and compensation do on Waymo and PandaSet.

These two annotate EVERY frame, so a "sweep" is the previous frame of the same sequence and the
compensator needs no box interpolation. This drives the real loaders rather than re-implementing
them, so what it measures is what training sees.

Two rules it obeys, both from experiments_md/20260922_05 Ablation 7:

  - Normalise each object against its OWN N=1 count. Scoring a moving object against its
    uncompensated ACCUMULATED count measures accidental overlap with the anchor box, not a
    density, and once manufactured a 2.00-2.83x "overshoot" that survived four wrong explanations.
  - Compare moving against static PER OBJECT - they are different populations (a driving car sits
    on open roadway, a parked one is shadowed by its neighbours at the kerb).

PandaSet needs no motion threshold: its cuboids carry a `stationary` flag, which is the split
this measures. Waymo has none, so a box is "moving" if it displaced more than --moved metres
between the anchor and the deepest sweep.

  python3 analysis/frame_accumulation_effect.py --dataset pandaset [--sweeps 5] [--boxes 400]
  python3 analysis/frame_accumulation_effect.py --dataset waymo

PandaSet also needs `--bind /home/koyama/code/ST3D:/root/ST3D`; its infos bake absolute paths.
"""
import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _init_path  # noqa: F401,E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets.motion_compensation import preceding_frames  # noqa: E402

CFG = {'waymo': 'cfgs/da-ieee-access/centerpoint-sourceonly-waymo.yaml',
       'pandaset': 'cfgs/da-ieee-access/centerpoint-sourceonly-pandaset.yaml'}
VEHICLE = {'waymo': 'Vehicle', 'pandaset': 'Car'}


def count_in_box(points, b):
    d = points[:, :3] - b[:3]
    c, s = np.cos(-b[6]), np.sin(-b[6])
    lx, ly = d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c
    return int(((np.abs(lx) <= b[3] / 2) & (np.abs(ly) <= b[4] / 2)
                & (np.abs(d[:, 2]) <= b[5] / 2)).sum())


def build(name, cfg, sweeps, compensate):
    dc = copy.deepcopy(cfg.DATA_CONFIG)
    dc.MAX_SWEEPS = sweeps
    if not compensate:
        dc.GT_BOXES_MOTION_COMPENSATION = False
    log = logging.getLogger('quiet')
    log.setLevel(logging.ERROR)
    if name == 'waymo':
        from pcdet.datasets.waymo.waymo_dataset import WaymoDataset as D
    else:
        from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset as D
    return D(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, training=True, logger=log)


def rows_waymo(d1, du, dc, n, moved, seed, want):
    out = []
    for k in np.random.default_rng(seed).permutation(len(dc.infos)):
        info = dc.infos[k]
        pc = info['point_cloud']
        if pc['sample_idx'] < n:
            continue
        S = np.linalg.inv(np.asarray(info['pose']))
        now = dc._tracked_boxes(info, S)
        prev = preceding_frames(dc._frame_index, pc['lidar_sequence'], pc['sample_idx'], n)
        if not prev or not now:
            continue
        then = dc._tracked_boxes(dc._sweep_infos[prev[-1]], S)
        p1, pu, pc_ = (d.get_lidar_with_sweeps(info) for d in (d1, du, dc))
        names = dict(zip(info['annos']['obj_ids'], info['annos']['name']))
        for t, b in now.items():
            if t not in then or names.get(t) != VEHICLE['waymo']:
                continue
            n1 = count_in_box(p1, b)
            if n1 < 5:
                continue
            out.append((float(np.linalg.norm(b[:3] - then[t][:3]) > moved),
                        n1, count_in_box(pu, b), count_in_box(pc_, b)))
        if len(out) >= want:
            break
    return out


def rows_pandaset(d1, du, dc, n, _moved, seed, want):
    import pandas as pd
    out = []
    for k in np.random.default_rng(seed).permutation(len(dc.pandaset_infos)):
        info = dc.pandaset_infos[k]
        if info['frame_idx'] < n:
            continue
        pose = dc._get_pose(info)
        cub = pd.read_pickle(info['cuboids_path'])
        stationary = dict(zip(cub['uuid'], cub['stationary']))
        label = dict(zip(cub['uuid'], cub['label']))
        boxes = dc._tracked_boxes(info, pose)
        p1, pu, pc_ = (d._get_points_with_sweeps(info, pose) for d in (d1, du, dc))
        for u, b in boxes.items():
            if label.get(u) != VEHICLE['pandaset']:
                continue
            n1 = count_in_box(p1, b)
            if n1 < 5:
                continue
            out.append((0.0 if stationary.get(u, True) else 1.0,
                        n1, count_in_box(pu, b), count_in_box(pc_, b)))
        if len(out) >= want:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=sorted(CFG), required=True)
    ap.add_argument('--sweeps', type=int, default=5)
    ap.add_argument('--boxes', type=int, default=400)
    ap.add_argument('--moved', type=float, default=1.0, help='waymo only; pandaset has a flag')
    ap.add_argument('--seed', type=int, default=2)
    args = ap.parse_args()

    cfg = EasyDict()
    cfg_from_yaml_file(CFG[args.dataset], cfg)
    d1 = build(args.dataset, cfg, 1, False)
    du = build(args.dataset, cfg, args.sweeps, False)
    dc = build(args.dataset, cfg, args.sweeps, True)

    fn = rows_waymo if args.dataset == 'waymo' else rows_pandaset
    a = np.array(fn(d1, du, dc, args.sweeps, args.moved, args.seed, args.boxes), dtype=float)
    moving = a[:, 0] > 0
    split = ('PandaSet `stationary` flag' if args.dataset == 'pandaset'
             else 'displaced >%.1f m over %d frames' % (args.moved, args.sweeps - 1))
    print('%s, N=%d: %d %s boxes (%d moving by %s)'
          % (args.dataset, args.sweeps, len(a), VEHICLE[args.dataset], moving.sum(), split))
    print('%-8s %7s %8s %10s %8s %10s %9s'
          % ('group', 'boxes', 'N=1', 'uncomp', 'comp', 'uncomp/N1', 'comp/N1'))
    for name, m in (('static', ~moving), ('moving', moving)):
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
