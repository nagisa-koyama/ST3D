"""Re-score stored PandaSet predictions under GT protocols that ignore what no detector can reach.

PandaSet's KITTI-metric evaluation scores every labelled box - out past 150 m, and boxes no point
falls in - so 62-68% of its Car GT is unreachable for a +-75.2 m detector (experiments_md/20261001_04).
This tool counts LiDAR points inside every GT box over the FULL cloud of the target sensor, then
re-runs the official evaluator on CPU (the rotated-IoU kernel swapped as in kitti_eval_cpu.py) with
some GT marked IGNORED - occlusion set to 3, beyond every difficulty level - so an ignored box is
neither required nor penalised and a detection matching it is not a false positive. That is how KITTI
treats objects harder than the difficulty being scored; no box is deleted and no prediction filtered.

Protocols (applied to GT of every class; Car AP is reported):
  as_scored       nothing ignored (reproduces the logged AP: the validation of this tool)
  pts>=1          ignore GT holding no point of the target sensor
  in_range        ignore GT whose centre lies outside +-75.2 m (where no prediction can be placed)
  in_range,pts>=1 both
  in_range,pts>=5 both, with Waymo's LEVEL_1 bar (more than 5 points)
  rule_A          THE PROTOCOL ADOPTED (2026-10-02), the nuScenes devkit's filter_eval_boxes with our range:
                  GT REMOVED unless its centre is within RADIUS of the ego vehicle and it holds >= 1 point
                  of the target sensor; predictions REMOVED if their centre is beyond RADIUS. Removal, not
                  ignore, and a circle, not the detector's square - as nuScenes does it.

    python pandaset_eval_protocols.py [--protocols rule_A ...]   # counts (cached), then the table
"""
import copy
import os
import pickle
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent)); sys.path.insert(0, str(TOOLS / 'analysis'))
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets.kitti import kitti_utils  # noqa: E402
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu  # noqa: E402
from kitti_eval_cpu import load_official_eval  # noqa: E402

LIMIT = 75.2
RADIUS = 75.0  # rule_A evaluation range (centre distance in the ground plane)
CACHE = TOOLS.parent / 'output' / 'analysis'
W = '/home/koyama/data/wandb/'
O = str(TOOLS.parent / 'output' / 'da-ieee-access') + '/'
# the evaluator's own map (pandaset_dataset.kitti_eval)
MAP = {'Car': 'Car', 'Pedestrian': 'Pedestrian', 'Pedestrian with Object': 'Pedestrian',
       'Pickup Truck': 'Truck', 'Medium-sized Truck': 'Truck', 'Semi-truck': 'Truck',
       'Motorized Scooter': 'Cyclist', 'Bicycle': 'Cyclist', 'Cyclist': 'Cyclist'}
ROWS = [  # label, result.pkl, target device, cone
    ('S3 control, full (26389)', W + 'run-20260928_054827-u2i837pw/files/eval/eval_with_train/epoch_115/val/result.pkl', 1, True),
    ('S3 pooled correction, full (25932)', O + 'centerpoint-global-pandaset-spin2flash/20260922_sourceonly/eval/epoch_115/val/recover_25782/result.pkl', 1, True),
    ('S3 control, 15 ep (26510)', W + 'run-20260928_191624-qwbujvrh/files/eval/eval_with_train/epoch_15/val/result.pkl', 1, True),
    ('S3 accumulation N=4, 15 ep (26692)', W + 'run-20260929_215545-vjutu18e/files/eval/eval_with_train/epoch_15/val/result.pkl', 1, True),
    ('S3 accumulation + cone corr., 15 ep (26512)', W + 'run-20260928_231658-o0wrcnps/files/eval/eval_with_train/epoch_15/val/result.pkl', 1, True),
    ('S4 oracle spin->spin, full (26389), 360', O + 'centerpoint-pandaset-spin2spin/20261001_pandaset/eval/epoch_115/val/s4oracle_360/result.pkl', 0, False),
    ('S4 oracle spin->spin, full (26389), cone', O + 'centerpoint-pandaset-spin2spin/20261001_pandaset/eval/epoch_115/val/s4oracle_360/result.pkl', 0, True),
    ('S4 oracle, 15 ep (26510), 360', O + 'centerpoint-pandaset-spin2spin/20261001_pandaset/eval/epoch_15/val/s4oracle15_360/result.pkl', 0, False),
    ('S4 oracle, 15 ep (26510), cone', O + 'centerpoint-pandaset-spin2spin/20261001_pandaset/eval/epoch_15/val/s4oracle15_360/result.pkl', 0, True),
    ('S4 old flash->spin, accum+pooled (26314), 360', O + 'centerpoint-accum-global-pandaset-flash2spin/20260923_psaccum_w16_staged/eval/epoch_115/val/recover_25827_ep115/result.pkl', 0, False),
    ('S4 old flash->spin, accum+pooled (26314), cone', O + 'centerpoint-accum-global-pandaset-flash2spin/20260923_psaccum_w16_staged/eval/epoch_115/val/recover_25827_ep115/result.pkl', 0, True),
    ('S4 control flash->spin, 15 ep (26974), 360', W + 'run-20261001_222222-uzi23xbs/files/eval/eval_with_train/epoch_15/val/result.pkl', 0, False),
    ('S4 control flash->spin, 15 ep (26974), cone', W + 'run-20261001_222222-uzi23xbs/files/eval/eval_with_train/epoch_15/val/result.pkl', 0, True),
    ('S4 thinning in the cone, 15 ep (27001), 360', W + 'run-20261002_002220-3nbs4nyu/files/eval/eval_with_train/epoch_15/val/result.pkl', 0, False),
    ('S4 thinning in the cone, 15 ep (27001), cone', W + 'run-20261002_002220-3nbs4nyu/files/eval/eval_with_train/epoch_15/val/result.pkl', 0, True),
    ('S3 control seed 667, 15 ep (27003)', W + 'run-20261002_012748-r65hpn4e/files/eval/eval_with_train/epoch_15/val/result.pkl', 1, True),
    ('S3 accumulation seed 667, 15 ep (27004)', W + 'run-20261002_025227-oj1yf0mg/files/eval/eval_with_train/epoch_15/val/result.pkl', 1, True),
    ('S4 thinning FIXED, 15 ep (27005), 360', W + 'run-20261002_041932-xrk47zew/files/eval/eval_with_train/epoch_15/val/result.pkl', 0, False),
    ('S4 thinning FIXED, 15 ep (27005), cone', W + 'run-20261002_041932-xrk47zew/files/eval/eval_with_train/epoch_15/val/result.pkl', 0, True),
    ('S3 oracle flash->flash, 15 ep (26974 ckpt)', O + 'centerpoint-pandaset-flash2flash/20261001_pandaset/eval/epoch_15/val/s3oracle15/result.pkl', 1, True),
]
PROTOCOLS = ['as_scored', 'pts>=1', 'in_range', 'in_range,pts>=1', 'in_range,pts>=5', 'rule_A']


def gt_with_counts(device):
    """The evaluator's GT per frame (infos order) plus points of the target sensor in each box."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'pandaset_val_gt_points_dev{device}.pkl'
    if path.exists():
        return pickle.load(open(path, 'rb'))
    from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset
    import logging
    logger = logging.getLogger('protocols'); logger.addHandler(logging.NullHandler())
    cfg = EasyDict(); cfg_from_yaml_file('cfgs/da-ieee-access/centerpoint-pandaset-spin2flash.yaml', cfg)
    dcfg = cfg.DATA_CONFIG_TAR; dcfg.LIDAR_DEVICE = device
    ds = PandasetDataset(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, training=False, logger=logger)
    out = []
    for info in ds.pandaset_infos:
        pose = ds._get_pose(info)
        boxes, labels, _ = ds._get_annotations(info, pose)
        pts = ds._get_lidar_points(info, pose)[:, :3]
        n = points_in_boxes_cpu(pts, boxes[:, :7].astype(np.float32)).sum(1) if len(boxes) else np.zeros(0, int)
        out.append({'gt_boxes': boxes, 'gt_names': labels, 'n_pts': n, 'frame_idx': info['frame_idx']})
    pickle.dump(out, open(path, 'wb'))
    return out


def summarise_counts(gt, device):
    from pcdet.datasets.pandaset.pandaset_dataset import fov_mask
    for cone in ((False, True) if device == 0 else (True,)):
        n = []; inr = []
        for f in gt:
            m = f['gt_names'] == 'Car'
            b = f['gt_boxes'][m]
            if cone and len(b):
                c = fov_mask(b, 60.0, 0.0); b = b[c]; k = f['n_pts'][m][c]
            else:
                k = f['n_pts'][m]
            n.append(k); inr.append((np.abs(b[:, 0]) <= LIMIT) & (np.abs(b[:, 1]) <= LIMIT))
        n, inr = np.concatenate(n), np.concatenate(inr)
        tag = f"{'Pandar64 (spin)' if device == 0 else 'PandarGT (flash)'}, {'cone' if cone else '360'}"
        print(f'\n{tag}: {len(n)} Car GT; points of the target sensor in the box (full cloud)')
        for name, m in (('in range', inr), ('beyond 75.2 m', ~inr)):
            k = n[m]
            print(f'  {name:14s} {100 * m.mean():5.1f}% of GT | 0 pts {100 * (k == 0).mean():5.1f}%  1-4 {100 * ((k >= 1) & (k < 5)).mean():5.1f}%  '
                  f'5-19 {100 * ((k >= 5) & (k < 20)).mean():5.1f}%  20-99 {100 * ((k >= 20) & (k < 100)).mean():5.1f}%  '
                  f'>=100 {100 * (k >= 100).mean():5.1f}%  median {np.median(k) if len(k) else 0:.0f}')


def ignore_mask(frame, protocol):
    b, n = frame['gt_boxes'], frame['n_pts']
    far = (np.abs(b[:, 0]) > LIMIT) | (np.abs(b[:, 1]) > LIMIT)
    if protocol == 'as_scored':
        return np.zeros(len(b), bool)
    if protocol == 'pts>=1':
        return n < 1
    if protocol == 'in_range':
        return far
    if protocol == 'in_range,pts>=1':
        return far | (n < 1)
    if protocol == 'in_range,pts>=5':
        return far | (n <= 5)
    raise ValueError(protocol)


def score(gt, dets, cone, protocol, official, cls='Car'):
    from pcdet.datasets.pandaset.pandaset_dataset import fov_mask
    g_annos, d_annos = [], []
    for f, d in zip(gt, dets):
        assert d['frame_id'] == f['frame_idx'], 'prediction and GT lists are not in the same order'
        g = {'gt_boxes': f['gt_boxes'].copy(), 'gt_names': f['gt_names'].copy()}
        d = {k: copy.deepcopy(d[k]) for k in ('boxes_lidar', 'name', 'score')}
        if protocol == 'rule_A':  # nuScenes filter_eval_boxes: remove, both sides
            keep = (np.hypot(g['gt_boxes'][:, 0], g['gt_boxes'][:, 1]) <= RADIUS) & (f['n_pts'] >= 1)
            g = {k: v[keep] for k, v in g.items()}
            dk = np.hypot(d['boxes_lidar'][:, 0], d['boxes_lidar'][:, 1]) <= RADIUS
            d = {k: v[dk] for k, v in d.items()}
            ign = np.zeros(len(g['gt_boxes']), bool)
        else:
            ign = ignore_mask(f, protocol)
        if cone:
            gm = fov_mask(g['gt_boxes'], 60.0, 0.0) if len(g['gt_boxes']) else np.zeros(0, bool)
            g = {k: v[gm] for k, v in g.items()}; ign = ign[gm]
            dm = fov_mask(d['boxes_lidar'], 60.0, 0.0) if len(d['boxes_lidar']) else np.zeros(0, bool)
            d = {k: v[dm] for k, v in d.items()}
        g['ignore'] = ign
        g_annos.append(g); d_annos.append(d)
    kitti_utils.transform_annotations_to_kitti_format(d_annos, map_name_to_kitti=MAP)
    kitti_utils.transform_annotations_to_kitti_format(g_annos, map_name_to_kitti=MAP, info_with_fakelidar=False)
    for g in g_annos:
        g['occluded'] = np.where(g.pop('ignore'), 3, 0)  # beyond every difficulty level: ignored
    _, ap = official.get_official_eval_result(g_annos, d_annos, [cls])  # Car at IoU 0.7, Pedestrian at 0.5
    return ap[f'{cls}_bev/moderate_R40'], ap[f'{cls}_3d/moderate_R40']


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--protocols', nargs='+', default=PROTOCOLS, choices=PROTOCOLS)
    ap.add_argument('--only', nargs='*', default=None, help='score only rows whose label contains one of these')
    ap.add_argument('--cls', default='Car', choices=['Car', 'Pedestrian'])
    ap.add_argument('--rows_json', default=None, help='score these rows INSTEAD of the built-in list: a JSON list of '
                    '[label, result.pkl path, target device (0 spin / 1 flash), cone (true/false)]')
    args = ap.parse_args(); protocols = args.protocols
    rows = ROWS
    if args.rows_json:
        import json
        rows = [tuple(r) for r in json.load(open(args.rows_json))]
    os.chdir(TOOLS)
    official = load_official_eval()
    gts = {d: gt_with_counts(d) for d in (0, 1)}
    for d in (0, 1):
        summarise_counts(gts[d], d)
    print(f'\n{args.cls} AP_R40 BEV / 3D (the three difficulty columns are identical on this target)')
    print(f"{'row':48s} " + ' '.join(f'{p:>17s}' for p in protocols))
    for label, path, device, cone in rows:
        if args.only is not None and not any(t in label for t in args.only):
            continue
        if not os.path.exists(path):
            print(f'{label:48s} (no predictions yet: {path})'); continue
        dets = pickle.load(open(path, 'rb'))
        cells = [score(gts[device], dets, cone, p, official, args.cls) for p in protocols]
        print(f'{label:48s} ' + ' '.join(f'{b:7.2f} / {t:6.2f}' for b, t in cells), flush=True)


if __name__ == '__main__':
    main()
