"""Padding check for a fixed-budget IA-SSD config whose frames hold FEWER points than its budget (experiments_md 20261011_07
§23, rows N2 / K2: 65,536 points by voxels on nuScenes single sweeps and on the camera-FOV crop of KITTI). CPU only.

`sample_points` fills such a frame up to the budget with COPIES of its points (drawn with replacement beyond half the
budget), so what the network receives is the frame's distinct points plus copies. This tool measures what the copies reach.

1. Census, per frame (N strided frames of each block: the training block in training mode with augmentation, and the
   evaluated block in evaluation mode, as test.py builds it): points reaching the budget step, distinct points leaving it
   (against min(16,384, points in), what the family's plain 16,384 sample keeps of the same frame), distinct / total, and
   the duplicates in SA layer 1's output. D-FPS (`pointnet2_batch_iassd`, whose near-origin skip is
   commented out) picks the point farthest from those already picked, and a copy of a picked point is at distance 0, so
   layer 1 keeps min(npoint, distinct) distinct points and max(0, npoint - distinct) duplicates; layers 2+ sample from it.
   On the first --fps_frames frames of each block this is checked by running D-FPS in numpy (the kernel's rule), and
   layer 1's ball query (the kernel's rule: the first nsample points in INDEX order within the radius, padded with the
   first) is compared between the padded cloud and the same frame without copies: distinct points per group.
2. Targets: one training batch (--target_frames) through the head's own assignment, both ways the config uses it (as
   `iassd_target_dump.py`). Every copy of a point must carry its original's class label and box target, so a GT box gains
   labelled positives only through the copies it received; the copies per box are printed against the frame's mean.

    python analysis/iassd_padding_check.py --cfg_file <cfg> [--frames 100] [--fps_frames 5] [--target_frames 0 1 2 3]
"""
import argparse
import logging
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.iassd_target_dump import cpu_points_in_boxes  # noqa: E402  (imports _init_path, patches .cuda for CPU)
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models.dense_heads import IASSD_head as head_mod  # noqa: E402
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils  # noqa: E402
from pcdet.utils import box_coder_utils, box_utils  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

BUDGET_STEPS = ('sample_points', 'sample_points_by_voxels')
PLAIN_BUDGET = 16384   # the family IA-SSD budget (rows N / K), for the comparison line


def hook_budget(ds):
    """Record the number of points entering the dataset's budget step (the first one with a fixed count)."""
    rec = {}
    q = ds.data_processor.data_processor_queue
    for i, f in enumerate(q):
        if getattr(f, 'func', None) is not None and f.func.__name__ in BUDGET_STEPS and \
                f.keywords['config'].NUM_POINTS[ds.data_processor.mode] != -1:
            def wrapped(data_dict=None, _f=f):
                rec['n_in'] = len(data_dict['points'])
                return _f(data_dict=data_dict)
            q[i] = wrapped
            return rec, f.keywords['config'].NUM_POINTS[ds.data_processor.mode]
    raise RuntimeError('no budget step in the processor queue')


def dfps(xyz, m):
    """furthest_point_sampling_kernel in numpy: start at index 0, then the point with the largest distance to the set."""
    idx = np.zeros(m, dtype=np.int64)
    dist = np.full(len(xyz), 1e10, dtype=np.float32)
    old = 0
    for j in range(1, m):
        d = ((xyz - xyz[old]) ** 2).sum(1)
        np.minimum(dist, d, out=dist)
        old = int(np.argmax(dist))
        idx[j] = old
    return idx


def ball_query_distinct(xyz, loc, centres, radius, nsample):
    """Distinct locations per group under ball_query_kernel_fast: the first nsample points in index order with d2 < r2."""
    tree = cKDTree(xyz)
    out = np.empty(len(centres), dtype=np.int64)
    for c, nb in enumerate(tree.query_ball_point(centres, radius, return_sorted=True)):
        nb = [k for k in nb if ((xyz[k] - centres[c]) ** 2).sum() < radius * radius][:nsample]
        out[c] = len(set(loc[nb].tolist())) if nb else 0
    return out


def census(ds, n, budget_rec, npoint1, fps_frames, radii, nsamples, label):
    rec, budget = budget_rec
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    rows, bq = [], []
    np.random.seed(0)
    for j, i in enumerate(idx):
        pts = ds[i]['points'][:, :3].astype(np.float32)
        uniq, first, loc = np.unique(pts, axis=0, return_index=True, return_inverse=True)
        n_uniq = len(uniq)
        rows.append((rec['n_in'], n_uniq, len(pts), max(0, npoint1 - n_uniq)))
        if j < fps_frames:
            sel = dfps(pts, npoint1)
            dup_fps = npoint1 - len(np.unique(loc[sel]))
            assert dup_fps == max(0, npoint1 - n_uniq), (dup_fps, npoint1, n_uniq)
            centres = pts[sel]
            order = np.sort(first)                       # the frame without copies, in first-appearance order
            xyz_u, loc_u = pts[order], loc[order]
            for r, ns in zip(radii, nsamples):
                bq.append((r, ns, ball_query_distinct(pts, loc, centres, r, ns).mean(),
                           ball_query_distinct(xyz_u, loc_u, centres, r, ns).mean()))
    a = np.array(rows, dtype=np.float64)
    q = lambda v: f'median {np.median(v):.0f} (p10 {np.percentile(v, 10):.0f}, p90 {np.percentile(v, 90):.0f})'
    print(f'[{label}] {len(idx)} frames, budget {budget}, SA layer 1 npoint {npoint1}')
    print(f'  points into the budget step: {q(a[:, 0])}')
    print(f'  distinct points out: {q(a[:, 1])}; kept by the voxel step: median {np.median(a[:, 1] / a[:, 0]):.3f}')
    plain = np.minimum(PLAIN_BUDGET, a[:, 0])   # a plain sample_points 16,384 of the same frame: distinct points it keeps
    print(f'  against the family budget, sample_points {PLAIN_BUDGET} of the same frame: {q(plain)} distinct; this config / '
          f'that: median {np.median(a[:, 1] / plain):.3f} (p10 {np.percentile(a[:, 1] / plain, 10):.3f}, p90 '
          f'{np.percentile(a[:, 1] / plain, 90):.3f}); this config holds more on {int((a[:, 1] > plain).sum())} of {len(a)} frames')
    print(f'  distinct / total: median {np.median(a[:, 1] / a[:, 2]):.3f} (p10 {np.percentile(a[:, 1] / a[:, 2], 10):.3f}, '
          f'p90 {np.percentile(a[:, 1] / a[:, 2], 90):.3f}); copies per distinct point: median {np.median(a[:, 2] / a[:, 1]):.2f}')
    print(f'  frames below half the budget (copies drawn WITH replacement): {int((a[:, 1] < budget / 2).sum())} of {len(a)}')
    print(f'  duplicates in SA layer 1 output (max(0, npoint - distinct)): {int((a[:, 3] > 0).sum())} of {len(a)} frames hold any; '
          f'mean {a[:, 3].mean():.1f}, max {a[:, 3].max():.0f} of {npoint1}')
    if fps_frames:
        print(f'  numpy D-FPS on {min(fps_frames, len(a))} frames: duplicates equal max(0, npoint - distinct) on every one')
        for r, ns in sorted({(r, ns) for r, ns, _, _ in bq}):
            p = np.mean([x for rr, nn, x, _ in bq if (rr, nn) == (r, ns)])
            u = np.mean([y for rr, nn, _, y in bq if (rr, nn) == (r, ns)])
            print(f'  layer-1 ball query r {r} m, nsample {ns}: distinct points per group {p:.1f} padded vs {u:.1f} '
                  f'without copies ({p / u:.2f})')
    return a


def targets(cfg, ds, frames):
    samples = [ds[i] for i in frames]
    batch = ds.collate_batch(samples)
    pts = torch.from_numpy(batch['points']).float()[:, :4]
    gt = torch.from_numpy(batch['gt_boxes']).float()
    tcfg = cfg.MODEL.POINT_HEAD.TARGET_CONFIG
    head_mod.roiaware_pool3d_utils.points_in_boxes_gpu = cpu_points_in_boxes
    head = types.SimpleNamespace(num_class=len(cfg.CLASS_NAMES), box_coder=getattr(box_coder_utils, tcfg.BOX_CODER)(
        **tcfg.BOX_CODER_CONFIG))
    B = gt.shape[0]
    _, first, loc = np.unique(pts.numpy(), axis=0, return_index=True, return_inverse=True)
    first, loc = torch.from_numpy(first), torch.from_numpy(loc.reshape(-1))
    print(f'[targets] frames {frames} | points {len(pts)}, distinct {len(first)} | copies per distinct point '
          + ', '.join(f'{int((pts[:, 0] == k).sum()) / len(np.unique(loc[pts[:, 0] == k])):.2f}' for k in range(B)))
    for name, kw, ext_w in (('centre targets', dict(), tcfg.GT_EXTRA_WIDTH),
                            ('vote targets (extend_gt)', dict(use_ex_gt_assign=True, fg_pc_ignore=False),
                             tcfg.ASSIGN_METHOD.EXTRA_WIDTH)):
        ext = box_utils.enlarge_box3d(gt.view(-1, 8), extra_width=ext_w).view(B, -1, 8)
        out = head_mod.IASSD_Head.assign_stack_targets_IASSD(
            head, points=pts, gt_boxes=gt, extend_gt_boxes=ext, set_ignore_flag=True, ret_box_labels=True, **kw)
        lab, box = out['point_cls_labels'], out['point_box_labels']
        same_lab = torch.equal(lab, lab[first[loc]])
        same_box = torch.equal(box, box[first[loc]])
        print(f'  [{name}] every copy carries its original\'s class label: {same_lab}; box target: {same_box}')
    per_cls = {}
    for k in range(B):
        m = pts[:, 0] == k
        frame_copies = int(m.sum()) / len(np.unique(loc[m]))
        valid = gt[k, :, 3:6].abs().sum(1) > 0
        inside = roiaware_pool3d_utils.points_in_boxes_cpu(pts[m, 1:4], gt[k][valid][:, :7]) > 0
        for t, c in enumerate(gt[k][valid][:, 7].long().tolist()):
            n_all = int(inside[t].sum())
            if c > 0 and n_all:
                per_cls.setdefault(cfg.CLASS_NAMES[c - 1], []).append(
                    (n_all / len(np.unique(loc[m][inside[t]])) / frame_copies, len(np.unique(loc[m][inside[t]]))))
    for c, v in per_cls.items():
        r = np.array([x for x, _ in v])
        print(f'  {c}: {len(v)} boxes with points; copies per box / the frame\'s mean copies: median {np.median(r):.2f} '
              f'(min {r.min():.2f}, max {r.max():.2f}); distinct points per box median {np.median([d for _, d in v]):.0f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--fps_frames', type=int, default=5)
    ap.add_argument('--target_frames', type=int, nargs='+', default=[0, 1, 2, 3])
    a = ap.parse_args()
    log = logging.getLogger('padding'); log.addHandler(logging.StreamHandler()); log.setLevel(logging.WARNING)
    cfg = cfg_from_yaml_file(a.cfg_file, EasyDict())
    sa = cfg.MODEL.BACKBONE_3D.SA_CONFIG
    npoint1, radii, nsamples = sa.NPOINT_LIST[0][0], sa.RADIUS_LIST[0], sa.NSAMPLE_LIST[0]
    assert sa.SAMPLE_METHOD_LIST[0] == ['D-FPS'], sa.SAMPLE_METHOD_LIST[0]
    print(f'{os.path.basename(a.cfg_file)}')
    for label, blk, training in (('training block, training mode, augmentation on', cfg.DATA_CONFIG, True),
                                 ('evaluated block, evaluation mode', cfg.DATA_CONFIG_TAR, False)):
        ds, _, _ = build_dataloader(dataset_cfg=blk, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                    logger=log, training=training, model_ontology=cfg.get('ONTOLOGY', None))
        census(ds, a.frames, hook_budget(ds), npoint1, a.fps_frames, radii, nsamples, label)
        if training:
            targets(cfg, ds, a.target_frames)


if __name__ == '__main__':
    main()
