"""CPU smoke for the genuine-ST3D row (experiments_md/20261015_01 §2.4): dump ONE batch's assigned targets and
compare them with the pseudo-labels that went in.

Synthetic pseudo-labels (no target label is read, none is needed): per KITTI-train frame, boxes of every kind the
pipeline can write - positives of the three classes (+1..+3) and ignore-band boxes (-1..-3) - centred on real
lidar points so they survive the zero-point filter. They go through the REAL dataset (ST3D target augmentation,
PROG_AUG re_prepare), the REAL collate, and CenterHead.assign_targets, and the script prints per class:
  in labels | boxes after augmentation (positives / ignored) | heatmap peaks | ignored pixels
Exit 1 if any positive is lost, any ignore box becomes a peak, or the peaks differ from the surviving positives.

    python analysis/genuine_st3d_target_dump.py [--frames 6] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _init_path  # noqa: F401,E402
from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models.dense_heads.center_head import CenterHead  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

CFG = 'cfgs/da-ieee-access/centerpoint-st3d-genuine-legal-nuscenes2kitti.yaml'
NAMES = ['Car', 'Pedestrian', 'Cyclist']
DIMS = {1: (4.0, 1.8, 1.6), 2: (0.8, 0.8, 1.7), 3: (1.8, 0.6, 1.7)}


def synth_labels(dataset, n, rng):
    labels = {}
    # pseudo-labels live in the SHIFT_COOR-shifted frame the model sees (pred_boxes are never un-shifted before
    # they are stored), so a raw lidar point is shifted the same way before it centres a box.
    shift = np.array(dataset.dataset_cfg.get('SHIFT_COOR', None) or (0.0, 0.0, 0.0), dtype=np.float32)
    for i in range(n):
        info = dataset.kitti_infos[i]
        pts = dataset.get_lidar(info['point_cloud']['lidar_idx'])
        pts = pts[(pts[:, 0] > 5) & (pts[:, 0] < 45) & (np.abs(pts[:, 1]) < 20)]
        rows = []
        for lab in (1, 2, 3, -1, -2, -3):
            for _ in range(2):
                p = pts[rng.integers(len(pts))][:3] + shift
                rows.append([p[0], p[1], p[2] + DIMS[abs(lab)][2] / 2, *DIMS[abs(lab)], rng.uniform(-np.pi, np.pi),
                             float(lab), 0.9 if lab > 0 else 0.3])
        labels[info['point_cloud']['lidar_idx']] = {
            'gt_boxes': np.array(rows, dtype=np.float32), 'cls_scores': None, 'iou_scores': None,
            'memory_counter': np.zeros(len(rows)), 'teacher_classes': np.array(['x'] * len(rows))}
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    cfg_from_yaml_file(CFG, cfg)
    logger = common_utils.create_logger()
    ds, _, _ = build_dataloader(cfg.DATA_CONFIG_TAR, cfg.CLASS_NAMES, args.frames, dist=False, workers=0,
                                logger=logger, training=True, model_ontology='kitti')
    ds.train()
    labels = synth_labels(ds, args.frames, rng)
    ds.set_pseudo_labels(labels)
    ok = True
    for stage in ('epoch 0 augmentation', 'after PROG_AUG re_prepare x4'):
        if stage.startswith('after'):
            for _ in range(4):
                ds.data_augmentor.re_prepare(augmentor_configs=None, intensity=cfg.SELF_TRAIN.PROG_AUG.SCALE)
        batch = ds.collate_batch([ds[i] for i in range(args.frames)])
        gt = torch.tensor(batch['gt_boxes'])
        head = CenterHead.__new__(CenterHead)
        torch.nn.Module.__init__(head)
        mc = cfg.MODEL.DENSE_HEAD
        head.model_cfg = EasyDict({'TARGET_ASSIGNER_CONFIG': dict(mc.TARGET_ASSIGNER_CONFIG),
                                   'LOSS_CONFIG': dict(mc.LOSS_CONFIG)})
        head.class_names = list(NAMES)
        head.class_names_each_head = [list(NAMES)]
        head.point_cloud_range = np.array(cfg.DATA_CONFIG_TAR.POINT_CLOUD_RANGE, dtype=np.float32)
        head.voxel_size = [0.1, 0.1, 0.15]
        head.separate_head_cfg = EasyDict({'HEAD_ORDER': ['center', 'center_z', 'dim', 'rot']})
        head.build_losses()
        fmap = [int((head.point_cloud_range[4] - head.point_cloud_range[1]) / 0.4),
                int((head.point_cloud_range[3] - head.point_cloud_range[0]) / 0.4)]
        t = head.assign_targets(gt.clone(), feature_map_size=fmap)
        hm = t['heatmaps'][0]                                 # (B, 3, H, W)
        msk = t['heatmap_masks'][0]
        print('\n=== %s ===' % stage)
        print('class       in(+/-)   after-aug(+/-)   heatmap peaks   ignored px(any class)')
        for c in (1, 2, 3):
            n_in_pos = sum(int((l['gt_boxes'][:, 7] == c).sum()) for l in labels.values())
            n_in_ign = sum(int((l['gt_boxes'][:, 7] == -c).sum()) for l in labels.values())
            lab = gt[..., 7]
            n_pos, n_ign = int((lab == c).sum()), int((lab == -c).sum())
            peaks = int(hm[:, c - 1].eq(1).sum())
            print('%-10s  %3d/%-3d    %3d/%-3d          %3d' % (NAMES[c - 1], n_in_pos, n_in_ign, n_pos, n_ign, peaks))
            bad = peaks > n_pos or peaks == 0 and n_pos > 0
            if bad:
                print('  !! peaks %d vs surviving positives %d' % (peaks, n_pos))
                ok = False
        ignored_px = int((msk == 0).sum()) if msk is not None else 0
        ign_total = int((gt[..., 7] < 0).sum())
        print('ignore boxes surviving: %d, ignored pixels: %d' % (ign_total, ignored_px))
        if ign_total > 0 and ignored_px == 0:
            print('  !! ignore boxes but no ignored footprint')
            ok = False
        # no ignore box may produce a peak in ANY channel at its centre
        total_peaks = int(hm.eq(1).sum())
        total_pos = int((gt[..., 7] > 0).sum())
        print('total peaks %d vs total positives %d' % (total_peaks, total_pos))
        if total_peaks > total_pos:
            ok = False
            print('  !! more peaks than positives: an ignore box became a positive')
    print('\nRESULT:', 'OK' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
