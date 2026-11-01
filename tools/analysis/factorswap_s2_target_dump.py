"""Target dump for the FACTOR-SWAP S2 rows (experiments_md 20261003_05 §5 item 24; bug_prevention_checklist rule 4).

For each config, real KITTI TRAIN frames go through the TRAINING loader with augmentation switched off, beside the
26843 oracle config's loader on the same frames, and the script prints what the swap changed: points per frame,
points inside the GT boxes, box count / size / height / labels, and CenterHead's assigned targets on a collated
batch (positive peaks per class against positive-label boxes, ignored pixels). It also checks that the key is inert
when the same config is built for evaluation. ANALYSIS, CPU; run as a Slurm CPU job:

    python analysis/factorswap_s2_target_dump.py <cfg.yaml> [<cfg.yaml> ...]
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
os.chdir(TOOLS)
import _init_path  # noqa: E402,F401
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models.dense_heads.center_head import CenterHead  # noqa: E402
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

BASE = 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml'
FRAMES = [0, 700, 1500, 2300, 3100]
logger = common_utils.create_logger()


def build(cfg_path, training):
    cfg = cfg_from_yaml_file(cfg_path, EasyDict())
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=logger, training=training, model_ontology=cfg.get('ONTOLOGY'))
    if training:
        ds.data_augmentor.data_augmentor_queue = []      # deterministic comparison: no augmentation
    return cfg, ds


def item(ds, i):
    np.random.seed(0)
    return ds[i]


def head_for(cfg, ds):
    head = CenterHead.__new__(CenterHead)
    torch.nn.Module.__init__(head)
    head.model_cfg = cfg.MODEL.DENSE_HEAD
    head.class_names = list(cfg.CLASS_NAMES)
    head.class_names_each_head = [[c for c in names if c in cfg.CLASS_NAMES]
                                  for names in cfg.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD]
    head.point_cloud_range = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
    head.voxel_size = ds.voxel_size
    head.separate_head_cfg = cfg.MODEL.DENSE_HEAD.SEPARATE_HEAD_CFG
    head.build_losses()
    stride = cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
    fmap = (ds.grid_size[:2] // stride)[::-1]
    return head, fmap


def targets(cfg, ds, dicts):
    head, fmap = head_for(cfg, ds)
    batch = ds.collate_batch(dicts)
    t = head.assign_targets(torch.from_numpy(batch['gt_boxes']).float(), feature_map_size=list(fmap))
    gt = batch['gt_boxes']
    pos = {c: int(((gt[:, :, 7] == k + 1)).sum()) for k, c in enumerate(cfg.CLASS_NAMES)}
    peaks = {}
    for h, names in enumerate(head.class_names_each_head):
        hm = t['heatmaps'][h]
        for j, c in enumerate(names):
            peaks[c] = int(hm[:, j].eq(1).sum())
    ign = int((gt[:, :, 7] < 0).sum())
    masks = t['heatmap_masks'][0]
    ignored_px = 0 if masks is None else int((masks == 0).sum())
    return pos, peaks, ign, ignored_px


def in_box_points(points, boxes):
    if len(boxes) == 0:
        return 0
    m = roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(points[:, :3]).float(),
                                                  torch.from_numpy(boxes[:, :7]).float()).numpy()
    return int(m.any(axis=0).sum())


def dump(cfg_path, base):
    bcfg, bds = base
    cfg, ds = build(cfg_path, training=True)
    swap = cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP
    print('=' * 100)
    print(cfg_path, '\nTRAIN_FACTOR_SWAP:', dict(swap))
    b_dicts, s_dicts = [], []
    for i in FRAMES:
        b, s = item(bds, i), item(ds, i)
        b_dicts.append(b); s_dicts.append(s)
        bb, sb = b['gt_boxes'], s['gt_boxes']
        line = ('frame %s: points %6d -> %6d (x%.3f); in-box points %5d -> %5d; boxes %d -> %d; labels %s -> %s'
                % (b['frame_id'], len(b['points']), len(s['points']), len(s['points']) / max(len(b['points']), 1),
                   in_box_points(b['points'], bb), in_box_points(s['points'], sb), len(bb), len(sb),
                   bb[:, 7].astype(int).tolist(), sb[:, 7].astype(int).tolist()))
        print(line)
        if len(bb) == len(sb) and len(bb):
            dim = sb[:, 3:6] / bb[:, 3:6]
            bottom = (sb[:, 2] - sb[:, 5] / 2) - (bb[:, 2] - bb[:, 5] / 2)
            dz = sb[:, 2] - bb[:, 2]
            far = np.abs(bb[:, 0]) > 5
            pitch = np.degrees(np.arctan2(dz[far], bb[far, 0])) if far.any() else np.zeros(0)
            r = np.hypot(sb[:, 0], sb[:, 1])
            print('    dims ratio (per box, dx dy dz): %s' % np.round(dim, 3).tolist())
            print('    bottom shift (m): %s; centre dz (m): %s; implied pitch (deg): %s'
                  % (np.round(bottom, 3).tolist(), np.round(dz, 3).tolist(), np.round(pitch, 2).tolist()))
            print('    box range (m): %s' % np.round(r, 1).tolist())
    pos, peaks, ign, ign_px = targets(cfg, ds, s_dicts)
    bpos, bpeaks, bign, bign_px = targets(bcfg, bds, b_dicts)
    print('targets, base : positive-label boxes %s, peaks %s, ignore-label boxes %d, ignored px %d'
          % (bpos, bpeaks, bign, bign_px))
    print('targets, swap : positive-label boxes %s, peaks %s, ignore-label boxes %d, ignored px %d'
          % (pos, peaks, ign, ign_px))
    assert all(peaks[c] <= pos[c] for c in pos), 'a peak without a positive-label box: an ignore box became positive'
    # The key must be inert at evaluation.
    _, eds = build(cfg_path, training=False)
    _, bes = build(BASE, training=False)
    for i in FRAMES[:2]:
        e, be = item(eds, i), item(bes, i)
        assert len(e['points']) == len(be['points']), 'TRAIN_FACTOR_SWAP acted at evaluation'
        assert np.allclose(np.sort(e['points'][:, 0]), np.sort(be['points'][:, 0]))
    print('eval mode: unchanged on frames %s' % FRAMES[:2])


if __name__ == '__main__':
    base = build(BASE, training=True)
    for p in sys.argv[1:]:
        dump(p, base)
    print('ALL DUMPS DONE')
