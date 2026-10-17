"""CPU smoke for the Car-only variant of the genuine ST3D row (experiments_md/20261017_04): from PREDICTIONS of all
three classes to ONE batch's assigned targets, with the real Car-only thresholds and memory config.

Unlike genuine_st3d_target_dump.py (which starts from stored labels), this starts one step earlier, because the variant
acts at storage: synthetic CenterPoint-shaped predictions (no target label is read, none is needed), per KITTI-train
frame, centred on real lidar points in the SHIFT_COOR frame:
  Car         3 kept at 0.9, 1 that the next two passes miss (-> voted ignore), 1 at 0.55 (under the 0.589 cut)
  Pedestrian  3 at 1.0 (the sigmoid maximum), 2 at 0.7
  Cyclist     2 at 1.0, 1 at 0.6
Three passes of the REAL save_pseudo_label_batch (pass 0 without, passes 1-2 with the consistency_ensemble memory and
voting; the GPU IoU kernel is replaced by an exact axis-aligned CPU IoU, all boxes have yaw 0 before augmentation),
then the REAL dataset (ST3D target augmentation, PROG_AUG re_prepare x4), collate and CenterHead.assign_targets.
Prints per class: stored (+/-) | after augmentation (+/-) | heatmap peaks | heatmap mass.
Exit 1 if any Pedestrian / Cyclist box is stored or reaches the heatmap, the stored Car labels differ from 3 positives
and 1 voted ignore per frame, Car peaks differ from the surviving Car positives, or ignore boxes leave no ignored pixel.

    python analysis/genuine_st3d_car_only_target_dump.py [--frames 6] [--seed 0]
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
from pcdet.utils import common_utils, memory_ensemble_utils, self_training_utils as stu  # noqa: E402

CFG = 'cfgs/da-ieee-access/centerpoint-st3d-genuine-legal-caronly-nuscenes2kitti.yaml'
NAMES = ['Car', 'Pedestrian', 'Cyclist']
DIMS = {1: (4.0, 1.8, 1.6), 2: (0.8, 0.8, 1.7), 3: (1.8, 0.6, 1.7)}
# (class, score, role) per frame; role 'miss' = absent from passes 1 and 2
PLAN = ([(1, 0.9, 'keep')] * 3 + [(1, 0.9, 'miss'), (1, 0.55, 'keep')]
        + [(2, 1.0, 'keep')] * 3 + [(2, 0.7, 'keep')] * 2 + [(3, 1.0, 'keep')] * 2 + [(3, 0.6, 'keep')])


def cpu_iou(a, b):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    lo = torch.maximum(a[:, None, :3] - a[:, None, 3:6] / 2, b[None, :, :3] - b[None, :, 3:6] / 2)
    hi = torch.minimum(a[:, None, :3] + a[:, None, 3:6] / 2, b[None, :, :3] + b[None, :, 3:6] / 2)
    inter = (hi - lo).clamp(min=0).prod(dim=2)
    va, vb = a[:, 3:6].prod(dim=1), b[:, 3:6].prod(dim=1)
    return (inter / (va[:, None] + vb[None, :] - inter)).float()


def synth_predictions(dataset, n, rng):
    """{frame_id: list of (box7, class, score, role)}; centres >= 6 m apart so no two boxes overlap."""
    shift = np.array(dataset.dataset_cfg.get('SHIFT_COOR', None) or (0.0, 0.0, 0.0), dtype=np.float32)
    preds = {}
    for i in range(n):
        info = dataset.kitti_infos[i]
        pts = dataset.get_lidar(info['point_cloud']['lidar_idx'])
        # near-ground points only (shifted z < 1 m), as a predicted object sits: a box placed on a tree or a facade can
        # be lifted out of the z range [-2, 4] by world scaling and dropped by the range mask, which is not a label bug
        pts = pts[(pts[:, 0] > 5) & (pts[:, 0] < 60) & (np.abs(pts[:, 1]) < 30) & (pts[:, 2] + shift[2] < 1.0)]
        centres, rows = [], []
        for c, s, role in PLAN:
            for _ in range(10000):
                p = pts[rng.integers(len(pts))][:3] + shift
                if all(np.hypot(*(p[:2] - q[:2])) >= 6.0 for q in centres):
                    break
            else:
                raise RuntimeError('could not place %d separated boxes in frame %s' % (len(PLAN), info))
            centres.append(p)
            d = DIMS[c]
            rows.append(([p[0], p[1], p[2] + d[2] / 2, *d, 0.0], c, s, role))
        preds[info['point_cloud']['lidar_idx']] = rows
    return preds


def pred_dict(rows):
    return {'pred_boxes': torch.tensor([r[0] for r in rows], dtype=torch.float32),
            'pred_scores': torch.tensor([r[2] for r in rows], dtype=torch.float32),
            'pred_labels': torch.tensor([r[1] for r in rows], dtype=torch.long)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    cfg_from_yaml_file(CFG, cfg)
    print('SELF_TRAIN.SCORE_THRESH', list(cfg.SELF_TRAIN.SCORE_THRESH), 'NEG_THRESH', list(cfg.SELF_TRAIN.NEG_THRESH))
    torch.Tensor.cuda = lambda self, *a, **k: self                       # CPU node
    memory_ensemble_utils.iou3d_nms_utils.boxes_iou3d_gpu = cpu_iou
    logger = common_utils.create_logger()
    ds, _, _ = build_dataloader(cfg.DATA_CONFIG_TAR, cfg.CLASS_NAMES, args.frames, dist=False, workers=0,
                                logger=logger, training=True, model_ontology='kitti')
    ds.train()
    preds = synth_predictions(ds, args.frames, rng)
    ids = list(preds)
    ok = True

    # ---- three generation passes through the real storage path
    stu.PSEUDO_LABELS.clear()
    stu.NEW_PSEUDO_LABELS.clear()
    for p in range(3):
        dicts = [pred_dict([r for r in preds[f] if p == 0 or r[3] != 'miss']) for f in ids]
        stu.save_pseudo_label_batch({'frame_id': ids}, pred_dicts=dicts, need_update=p > 0, student_ontology='kitti')
        stu.PSEUDO_LABELS.clear()
        stu.PSEUDO_LABELS.update(stu.NEW_PSEUDO_LABELS)
        stu.NEW_PSEUDO_LABELS.clear()
        labs = np.concatenate([stu.PSEUDO_LABELS[f]['gt_boxes'][:, 7] for f in ids])
        print('pass %d stored labels: %s' % (p, {int(k): int((labs == k).sum()) for k in np.unique(labs)}))
    stored = {f: stu.PSEUDO_LABELS[f]['gt_boxes'][:, 7] for f in ids}
    tops = np.concatenate([stu.PSEUDO_LABELS[f]['gt_boxes'][:, 2] + stu.PSEUDO_LABELS[f]['gt_boxes'][:, 5] / 2
                           for f in ids])
    print('stored box tops z (shifted frame): min %.2f max %.2f m' % (tops.min(), tops.max()))
    for f, lab in stored.items():
        if sorted(lab.tolist()) != [-1.0, 1.0, 1.0, 1.0]:
            print('  !! frame %s stored %s, expected three Car positives and one voted Car ignore' % (f, lab.tolist()))
            ok = False
    ds.set_pseudo_labels(stu.PSEUDO_LABELS)

    # ---- the real dataset, collate and target assignment
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
        hm = t['heatmaps'][0]                                            # (B, 3, H, W)
        msk = t['heatmap_masks'][0]
        lab = gt[..., 7]
        print('\n=== %s ===' % stage)
        print('class       stored(+/-)  after-aug(+/-)  heatmap peaks  heatmap mass')
        for c in (1, 2, 3):
            s_pos = sum(int((v == c).sum()) for v in stored.values())
            s_ign = sum(int((v == -c).sum()) for v in stored.values())
            n_pos, n_ign = int((lab == c).sum()), int((lab == -c).sum())
            peaks, mass = int(hm[:, c - 1].eq(1).sum()), float(hm[:, c - 1].sum())
            print('%-10s  %3d/%-3d      %3d/%-3d         %4d          %.1f' % (NAMES[c - 1], s_pos, s_ign, n_pos, n_ign,
                                                                            peaks, mass))
            if c > 1 and (s_pos or s_ign or n_pos or n_ign or peaks or mass):
                print('  !! %s reached the stored labels or the heatmap' % NAMES[c - 1])
                ok = False
            if c == 1 and (peaks != n_pos or n_pos == 0):
                print('  !! Car peaks %d vs surviving Car positives %d' % (peaks, n_pos))
                ok = False
        ignored_px = int((msk == 0).sum()) if msk is not None else 0
        ign_total = int((lab < 0).sum())
        print('Car ignore boxes surviving: %d, ignored pixels: %d' % (ign_total, ignored_px))
        if ign_total > 0 and ignored_px == 0:
            print('  !! ignore boxes but no ignored footprint')
            ok = False
    print('\nRESULT:', 'OK' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
