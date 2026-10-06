"""Item-4 target dump for an IA-SSD config (memory/repo/bug_prevention_checklist.md): one real TRAINING batch, the
head's own assignment, compared with the input labels. CPU only (the GPU point-in-box kernel is swapped for the
CPU one), so it runs on the master node before or beside a GPU run.

The backbone's sampled centres need CUDA ops, so the batch's input points stand in for them; the assignment code is
the head's own `assign_stack_targets_IASSD`, called the two ways the config uses it: centre targets (GT boxes, an
ignore band at GT_EXTRA_WIDTH) and the vote targets of ASSIGN_METHOD 'extend_gt' (boxes enlarged by EXTRA_WIDTH).

Printed per class: GT boxes in the batch, points inside them (independent count), points the head labels positive,
and whether the two label vectors agree point for point; plus the sentinel census (labels present in gt_boxes) and
the box-target / foreground-row pairing the losses rely on.

    python analysis/iassd_target_dump.py --cfg_file cfgs/da-ieee-access-pointrcnn/iassd-sourceonly-kitti2kitti.yaml [--frames 0 1]
"""
import argparse
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _init_path  # noqa: F401,E402

torch.Tensor.cuda = lambda self, *a, **k: self  # CPU-only: the IA-SSD box coder and losses call .cuda() in __init__

from easydict import EasyDict  # noqa: E402
import logging  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models.dense_heads import IASSD_head as head_mod  # noqa: E402
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils  # noqa: E402
from pcdet.utils import box_coder_utils, box_utils  # noqa: E402


def cpu_points_in_boxes(points, boxes):
    inside = roiaware_pool3d_utils.points_in_boxes_cpu(points[0].float(), boxes[0].float()) > 0  # (T, N)
    idx = torch.full((points.shape[1],), -1, dtype=torch.int)
    hit = inside.any(dim=0)
    idx[hit] = torch.argmax(inside.int(), dim=0)[hit].int()
    return idx[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--frames', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--no_aug', action='store_true', help='training view with augmentation off (e.g. no gt_sampling)')
    a = ap.parse_args()
    log = logging.getLogger('dump'); log.addHandler(logging.StreamHandler()); log.setLevel(logging.WARNING)
    cfg = cfg_from_yaml_file(a.cfg_file, EasyDict())
    dcfg = cfg.DATA_CONFIG if cfg.get('DATA_CONFIG', None) else list(cfg.DATA_CONFIGS.values())[0]
    ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=log, training=True, model_ontology=cfg.get('ONTOLOGY', None))
    if a.no_aug:
        from pcdet.datasets.point_calibration import _augmentation_off
        with _augmentation_off(ds):
            samples = [ds[i] for i in a.frames]
    else:
        samples = [ds[i] for i in a.frames]
    batch = ds.collate_batch(samples)
    pts = torch.from_numpy(batch['points']).float()[:, :4]  # [bs_idx, x, y, z]
    gt = torch.from_numpy(batch['gt_boxes']).float()        # (B, M, 8), zero-padded
    tcfg = cfg.MODEL.POINT_HEAD.TARGET_CONFIG
    head_mod.roiaware_pool3d_utils.points_in_boxes_gpu = cpu_points_in_boxes
    head = types.SimpleNamespace(num_class=len(cfg.CLASS_NAMES), box_coder=getattr(box_coder_utils, tcfg.BOX_CODER)(
        **tcfg.BOX_CODER_CONFIG))
    B = gt.shape[0]
    cls = gt[..., 7]
    print('%s | %s frames %s | points %d | gt_boxes %s' % (os.path.basename(a.cfg_file), type(ds).__name__, a.frames,
                                                          len(pts), tuple(gt.shape)))
    print('sentinel census, labels in gt_boxes:', {int(v): int((cls == v).sum()) for v in torch.unique(cls)},
          '(0 = padding rows)')
    per_box = {c: [] for c in range(1, len(cfg.CLASS_NAMES) + 1)}
    for k in range(B):
        m = pts[:, 0] == k
        valid = gt[k, :, 3:6].abs().sum(1) > 0
        inside = roiaware_pool3d_utils.points_in_boxes_cpu(pts[m, 1:4], gt[k][valid][:, :7]) > 0
        for t, c in enumerate(gt[k][valid][:, 7].long().tolist()):
            if c > 0:
                per_box[c].append(int(inside[t].sum()))
    print('points per frame:', [int((pts[:, 0] == k).sum()) for k in range(B)], '| augmentation', 'OFF' if a.no_aug else 'on')
    print('points per GT box, median (n boxes):', {cfg.CLASS_NAMES[c - 1]: (float(np.median(v)) if v else None, len(v))
                                                    for c, v in per_box.items()})

    for name, kw, ext_w in (('centre targets', dict(), tcfg.GT_EXTRA_WIDTH),
                            ('vote targets (extend_gt)', dict(use_ex_gt_assign=True, fg_pc_ignore=False),
                             tcfg.ASSIGN_METHOD.EXTRA_WIDTH)):
        ext = box_utils.enlarge_box3d(gt.view(-1, 8), extra_width=ext_w).view(B, -1, 8)
        out = head_mod.IASSD_Head.assign_stack_targets_IASSD(
            head, points=pts, gt_boxes=gt, extend_gt_boxes=ext, set_ignore_flag=True, ret_box_labels=True, **kw)
        lab = out['point_cls_labels']
        # independent labels: class of the containing box (inside the enlarged box for the vote path)
        ref = torch.zeros(len(pts), dtype=torch.long)
        multi = 0
        for k in range(B):
            m = pts[:, 0] == k
            valid = gt[k, :, 3:6].abs().sum(1) > 0
            boxes = (ext if kw else gt)[k][valid]
            inside = roiaware_pool3d_utils.points_in_boxes_cpu(pts[m, 1:4], boxes[:, :7]) > 0
            tight = roiaware_pool3d_utils.points_in_boxes_cpu(pts[m, 1:4], gt[k][valid][:, :7]) > 0
            r = torch.zeros(int(m.sum()), dtype=torch.long)
            if not kw:
                band = roiaware_pool3d_utils.points_in_boxes_cpu(pts[m, 1:4], ext[k][valid][:, :7]) > 0
                r[band.any(0) & ~tight.any(0)] = -1
            cover = inside if kw else tight
            for t in reversed(range(boxes.shape[0])):   # the FIRST containing box wins, as in the CPU stand-in
                r[cover[t]] = int(gt[k][valid][t, 7])
            if kw:  # extended path: a point inside a TIGHT box keeps that box ("instance points should keep unchanged")
                in_tight = tight.any(0)
                rt = torch.zeros_like(r)
                for t in reversed(range(boxes.shape[0])):
                    rt[tight[t]] = int(gt[k][valid][t, 7])
                r[in_tight] = rt[in_tight]
            ref[m] = r
            multi += int((cover.sum(0) > 1).sum())
        print('\n[%s] extra width %s' % (name, list(ext_w)))
        print('  %-11s %8s %14s %14s' % ('class', 'GT boxes', 'points (indep)', 'labelled pos'))
        for c, cname in enumerate(cfg.CLASS_NAMES, start=1):
            print('  %-11s %8d %14d %14d' % (cname, int((cls == c).sum()), int((ref == c).sum()), int((lab == c).sum())))
        print('  ignored (-1): %d | background: %d | labels agree point for point: %s (points inside 2+ boxes: %d)'
              % (int((lab == -1).sum()), int((lab == 0).sum()), bool(torch.equal(lab, ref)), multi))
        pos = lab > 0
        print('  gt_box_of_fg_points rows %d = positives %d: %s | box targets only on positives: %s'
              % (out['gt_box_of_fg_points'].shape[0], int(pos.sum()), out['gt_box_of_fg_points'].shape[0] == int(pos.sum()),
                 bool(out['point_box_labels'][~pos].abs().sum() == 0)))


if __name__ == '__main__':
    main()
