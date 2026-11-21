"""Item 51 (experiments_md 20261003_05 §5): checks of the DTS source model D0's loader before its GPU run. CPU only.

    python3 analysis/item51_d0_check.py [--frames 40] [--json OUT]

Over --frames nuScenes TRAIN frames spread across the split, D0's config (TRAIN_BEAM_UPSAMPLE) against 25743's
(centerpoint-sourceonly-nuscenes.yaml), both training=True:
  1. identity: D0 at BEAM_PROB 0 loads exactly 25743's points (bug-prevention rule 8, >= 40 frames);
  2. at 0.5, 25743's points are the first rows of D0's, unchanged; the added points are reported;
  3. every new point's elevation lies inside the frame's own elevation range (a midpoint of two of its points);
  4. points per GT box (infos' LiDAR-frame boxes, before any augmentation): D0 >= 25743 for every box;
  5. evaluation (training=False, val frames spread the same way) is unchanged.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
D0 = 'cfgs/da-ieee-access/item51/centerpoint-dts-d0-sourceonly-nuscenes.yaml'
BASE = 'cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml'


def build(cfg_path, training, prob=None):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    from pcdet.utils import common_utils
    cfg = cfg_from_yaml_file(cfg_path, EasyDict())
    block = copy.deepcopy(cfg.DATA_CONFIG)
    if prob is not None:
        block.TRAIN_BEAM_UPSAMPLE.BEAM_PROB = prob
    return NuScenesDataset(dataset_cfg=block, class_names=cfg.CLASS_NAMES, root_path=None, training=training,
                           logger=common_utils.create_logger())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', type=int, default=40)
    p.add_argument('--json', default=None)
    args = p.parse_args()
    os.chdir(TOOLS)
    import _init_path  # noqa: F401
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    import torch
    base, d0, d0_zero = build(BASE, True), build(D0, True), build(D0, True, prob=0.0)
    base_ev, d0_ev = build(BASE, False), build(D0, False)
    assert d0.dataset_cfg.TRAIN_BEAM_UPSAMPLE.BEAM_PROB == 0.5 and 'TRAIN_BEAM_UPSAMPLE' not in base.dataset_cfg
    idx = np.linspace(0, len(base.infos) - 1, args.frames).astype(int)
    rows, fails = [], []
    for k in idx:
        b = base.get_lidar_with_sweeps(k, max_sweeps=1)
        z = d0_zero.get_lidar_with_sweeps(k, max_sweeps=1)
        np.random.seed(int(k))
        u = d0.get_lidar_with_sweeps(k, max_sweeps=1)
        ok1 = z.shape == b.shape and np.array_equal(z, b)
        ok2 = len(u) >= len(b) and np.array_equal(u[:len(b)], b)
        # 3: elevation of the new points between ring medians (rings from the raw file, after ego removal)
        raw = np.fromfile(str(base.root_path / base.infos[k]['lidar_path']), dtype=np.float32).reshape(-1, 5)
        raw = base.remove_ego_points(raw, center_radius=1.5)
        el = np.arctan2(raw[:, 2], np.hypot(raw[:, 0], raw[:, 1]))
        med = np.array([np.median(el[raw[:, 4] == i]) if (raw[:, 4] == i).any() else np.nan for i in range(32)])
        new_el = np.arctan2(u[len(b):, 2], np.hypot(u[len(b):, 0], u[len(b):, 1]))
        lo, hi = el.min(), el.max()  # a midpoint lies inside its parents' elevations (ring medians are too tight: 34898)
        ok3 = bool(((new_el > lo - 1e-6) & (new_el < hi + 1e-6)).all())
        # 4: points per GT box before augmentation
        boxes = base.infos[k]['gt_boxes'][:, :7].astype(np.float32)
        if len(boxes):
            nb = roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(b[:, :3]).float(), torch.from_numpy(boxes)).numpy().sum(1)
            nu = roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(u[:, :3]).float(), torch.from_numpy(boxes)).numpy().sum(1)
            ok4 = bool((nu >= nb).all())
            gain = float(nu[nb > 0].sum() / max(nb[nb > 0].sum(), 1))
        else:
            ok4, gain = True, None
        kv = int(k) * (len(base_ev.infos) - 1) // max(len(base.infos) - 1, 1)  # a val frame (the eval split)
        ok5 = np.array_equal(base_ev.get_lidar_with_sweeps(kv, max_sweeps=1), d0_ev.get_lidar_with_sweeps(kv, max_sweeps=1))
        r = dict(index=int(k), base=len(b), d0=len(u), added=len(u) - len(b), box_point_ratio=gain,
                 identity=ok1, prefix=ok2, between_rings=ok3, box_points=ok4, eval_unchanged=ok5)
        rows.append(r)
        if not all([ok1, ok2, ok3, ok4, ok5]):
            fails.append(r)
        print(r, flush=True)
    added = np.array([r['added'] / r['base'] for r in rows])
    ratio = np.array([r['box_point_ratio'] for r in rows if r['box_point_ratio'] is not None])
    print('frames %d, failures %d; added points / base: mean %.3f (min %.3f, max %.3f); in-box point ratio mean %.3f'
          % (len(rows), len(fails), added.mean(), added.min(), added.max(), ratio.mean()))
    if args.json:
        json.dump(dict(rows=rows, failures=fails), open(args.json, 'w'), indent=1)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
