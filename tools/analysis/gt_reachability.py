"""How much of each evaluator's Car ground truth can ANY detector in this family reach?

The KITTI-metric evaluators here score every annotated box (no range or point filter on GT; see
experiments_md/20261001_04). A box is out of reach if
  - its centre lies outside POST_CENTER_LIMIT_RANGE (|x|, |y| <= 75.2 m): no prediction can be placed
    there, so it is a guaranteed false negative; or
  - it holds no LiDAR point inside POINT_CLOUD_RANGE: the network has nothing to fire on.
The reachable fraction r caps recall, and with perfect precision on everything reachable AP_R40 is
floor(40 r) / 40 - so 100 r is the ceiling of any row scored against that target.

    python gt_reachability.py [--stride 1]

PandaSet: the evaluator's own GT path (`_get_annotations`, Car = raw label 'Car') and its own cone
(`fov_mask`); points from `_get_lidar_points`, counted inside the detection range. nuScenes: the eval
GT is the val infos as stored (`num_lidar_pts` = keyframe LIDAR_TOP points, which is what a MAX_SWEEPS 1
model sees). KITTI: moderate-eligible Cars only (difficulty 0-1); harder ones are ignored, not missed.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu  # noqa: E402

LIMIT = 75.2  # POST_CENTER_LIMIT_RANGE and POINT_CLOUD_RANGE in x and y, every da-ieee-access row


def summarise(name, n, far, empty, beyond_max=None):
    reach = n - far - empty
    r = reach / max(n, 1)
    extra = '' if beyond_max is None else f'  beyond the frame\'s farthest point {100 * beyond_max / n:4.1f}%'
    print(f'{name:38s} Car GT {n:7d}  centre beyond 75.2 m {100 * far / n:5.1f}%  '
          f'in range but 0 points {100 * empty / n:5.1f}%  reachable {100 * r:5.1f}%  '
          f'-> AP_R40 ceiling {100 * np.floor(40 * r) / 40:5.1f}{extra}', flush=True)


def pandaset(stride):
    from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset, fov_mask
    import logging
    logger = logging.getLogger('reach'); logger.addHandler(logging.NullHandler())
    for device, label in ((0, 'Pandar64 (spin)'), (1, 'PandarGT (flash)')):
        cfg = EasyDict(); cfg_from_yaml_file('cfgs/da-ieee-access/centerpoint-pandaset-spin2flash.yaml', cfg)
        dcfg = cfg.DATA_CONFIG_TAR; dcfg.LIDAR_DEVICE = device
        ds = PandasetDataset(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, training=False,
                             root_path=None, logger=logger)
        shift_z = float(dcfg.get('SHIFT_COOR', [0, 0, 0])[2])
        zlo, zhi = dcfg.POINT_CLOUD_RANGE[2] - shift_z, dcfg.POINT_CLOUD_RANGE[5] - shift_z
        acc = {k: [0, 0, 0, 0] for k in ('360', 'cone')}
        for i in range(0, len(ds.pandaset_infos), stride):
            info = ds.pandaset_infos[i]
            pose = ds._get_pose(info)
            boxes, labels, _ = ds._get_annotations(info, pose)
            boxes = boxes[labels == 'Car'][:, :7].astype(np.float32)
            if len(boxes) == 0:
                continue
            pts = ds._get_lidar_points(info, pose)[:, :3]
            r_max = np.hypot(pts[:, 0], pts[:, 1]).max()
            pts = pts[(np.abs(pts[:, 0]) <= LIMIT) & (np.abs(pts[:, 1]) <= LIMIT) & (pts[:, 2] >= zlo) & (pts[:, 2] <= zhi)]
            far = (np.abs(boxes[:, 0]) > LIMIT) | (np.abs(boxes[:, 1]) > LIMIT)
            n_in = points_in_boxes_cpu(pts, boxes).sum(1) if len(pts) else np.zeros(len(boxes))
            empty = (~far) & (n_in == 0)
            beyond = np.hypot(boxes[:, 0], boxes[:, 1]) > r_max
            for key, m in (('360', np.ones(len(boxes), bool)), ('cone', fov_mask(boxes, 60.0, 0.0))):
                a = acc[key]; a[0] += m.sum(); a[1] += (far & m).sum(); a[2] += (empty & m).sum(); a[3] += (beyond & m).sum()
        for key, a in acc.items():
            if device == 1 and key == '360':
                continue  # the flash target is only ever scored inside its cone
            summarise(f'PandaSet {label}, {key}', *a)


def nuscenes():
    infos = pickle.load(open(TOOLS.parent / 'data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl', 'rb'))
    n = far = empty = 0
    for info in infos:
        m = info['gt_names'] == 'car'
        b, p = info['gt_boxes'][m], info['num_lidar_pts'][m]
        f = (np.abs(b[:, 0]) > LIMIT) | (np.abs(b[:, 1]) > LIMIT)
        n += m.sum(); far += f.sum(); empty += ((~f) & (p == 0)).sum()
    summarise('nuScenes val', n, far, empty)


def kitti():
    infos = pickle.load(open(TOOLS.parent / 'data/kitti/kitti_infos_val.pkl', 'rb'))
    n = far = empty = 0
    for info in infos:
        a = info['annos']; k = len(a['gt_boxes_lidar'])
        m = (a['name'][:k] == 'Car') & (a['difficulty'][:k] >= 0) & (a['difficulty'][:k] <= 1)
        b, p = a['gt_boxes_lidar'][m], a['num_points_in_gt'][:k][m]
        f = (np.abs(b[:, 0]) > LIMIT) | (np.abs(b[:, 1]) > LIMIT)
        n += m.sum(); far += f.sum(); empty += ((~f) & (p == 0)).sum()
    summarise('KITTI val, moderate-eligible', n, far, empty)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()
    import os; os.chdir(TOOLS)
    kitti(); nuscenes(); pandaset(args.stride)
