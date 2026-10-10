"""Pre-launch check of PandaSet RING_PATTERN (experiments_md 20261010_05): what the pattern does to points and to Car
training boxes, against the source-only cloud of the same frames. Source TRAIN frames, training mode, as trained.

    python analysis/ring_pattern_pandaset_check.py <ring-pattern cfg> <control cfg> <frames>

Reports, over strided frames:
- points per frame into the voxeliser (after the processor: POINT_CLOUD_RANGE, so z in [-2, 4] m) for the pattern row
  and the control, overall and per 7.5-15 m ring (the target's figure comes from thinning_recount_check.py);
- the Pandar64 channels above the target's field of view (the 14.7 deg channel): share of points and of Car-box points;
- Car boxes at the loader stage (normative frame, before SHIFT_COOR): holding >= 1 point in the full cloud, and of
  those emptied by the pattern (PandaSet's zero-point filter then drops them as labels);
- laser lines (distinct channels) per Car box near 20 / 40 / 60 m, full cloud -> pattern, with the 1.33 deg lattice's
  geometric expectation on the same box (box height / range / spacing);
- Car boxes reaching the model with 0 points (must be 0).
"""
import sys

import numpy as np

sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.pandaset import pandaset_rings as R
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

cfg_p, cfg_c, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
logger = common_utils.create_logger()


def build(f):
    cfg = EasyDict()
    cfg_from_yaml_file(f, cfg)
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
    return cfg, ds


cfgP, P = build(cfg_p)
cfgC, C = build(cfg_c)
rp = cfgP.DATA_CONFIG.RING_PATTERN
RINGS = [0, 7.5, 15, 30, 45, 60, 75]


def in_range(p):
    r = np.hypot(p[:, 0], p[:, 1])
    return np.histogram(r[r < 75], bins=RINGS)[0]


def car_boxes(ds, info, pose):
    boxes, names, _ = ds._get_annotations(info, pose)
    return boxes[np.array([nm == 'Car' for nm in names], dtype=bool)] if len(boxes) else np.zeros((0, 7))


def point_box_ids(pts, boxes):
    if len(boxes) == 0 or len(pts) == 0:
        return np.zeros((len(boxes), len(pts)), dtype=bool)
    import torch
    return roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(pts[:, :3]).float(),
                                                     torch.from_numpy(boxes[:, :7]).float()).numpy() > 0


stats = dict(frames=0, pts_P=[], pts_C=[], ringP=np.zeros(6), ringC=np.zeros(6), ch0_pts=0, all_pts=0,
             ch0_box_pts=0, box_pts=0, car_hit=0, car_emptied=0, car_hit70=0, car_emptied70=0, zero_after=0,
             cars_after=0)
lines = {20: ([], [], []), 40: ([], [], []), 60: ([], [], [])}
step = max(1, len(P) // n)
for i in range(0, len(P), step)[:n]:
    info = P.pandaset_infos[i]
    pose = P._get_pose(info)
    # loader stage, full device-0 cloud with its channel labels (no ring cut): the control's path
    full = C._get_lidar_points(info, pose)
    labels = R.load_cached_labels(rp.LABEL_CACHE, info['sequence'], info['frame_idx'], len(full))
    np.random.seed(1000 + i)
    patt = P._get_lidar_points(info, pose)
    np.random.seed(1000 + i)
    import pandas as pd
    import pandaset as ps
    raw = pd.read_pickle(info['lidar_path'])
    raw = raw[raw.d == 0]
    ego = ps.geometry.lidar_points_to_ego(raw[['x', 'y', 'z']].to_numpy(), pose)
    keep = R.ring_pattern_points(ego, labels, rp)
    assert keep.sum() == len(patt), (keep.sum(), len(patt))
    boxes = car_boxes(C, info, pose)
    inb = point_box_ids(full, boxes)                                 # (boxes, points)
    stats['ch0_pts'] += int((labels == 0).sum())
    stats['all_pts'] += len(labels)
    stats['ch0_box_pts'] += int(inb[:, labels == 0].sum())
    stats['box_pts'] += int(inb.sum())
    hit = inb.sum(1) > 0
    emptied = hit & (inb[:, keep].sum(1) == 0)
    stats['car_hit'] += int(hit.sum())
    stats['car_emptied'] += int(emptied.sum())
    rng = np.hypot(boxes[:, 0], boxes[:, 1])
    stats['car_hit70'] += int((hit & (rng < 70)).sum())
    stats['car_emptied70'] += int((emptied & (rng < 70)).sum())
    for d, (raw_l, pat_l, exp_l) in lines.items():
        for b in np.nonzero(hit & (np.abs(rng - d) < 5))[0]:
            raw_l.append(len(np.unique(labels[inb[b]])))
            pat_l.append(len(np.unique(labels[inb[b] & keep])))
            exp_l.append(np.degrees(np.arctan2(boxes[b, 5], rng[b])) / rp.SPACING_DEG)
    # what reaches the model
    np.random.seed(2000 + i)
    dP = P[i]
    np.random.seed(2000 + i)
    dC = C[i]
    stats['pts_P'].append(int((np.hypot(dP['points'][:, 0], dP['points'][:, 1]) < 75).sum()))
    stats['pts_C'].append(int((np.hypot(dC['points'][:, 0], dC['points'][:, 1]) < 75).sum()))
    stats['ringP'] += in_range(dP['points'])
    stats['ringC'] += in_range(dC['points'])
    gb = dP['gt_boxes']
    cars = gb[gb[:, 7] == 1] if len(gb) else gb
    if len(cars):
        stats['cars_after'] += len(cars)
        stats['zero_after'] += int((point_box_ids(dP['points'], cars).sum(1) == 0).sum())
    stats['frames'] += 1

f = stats['frames']
print('frames %d' % f)
print('points/frame into the voxeliser (within 75 m): pattern %.0f (p10-p90 %.0f-%.0f), control %.0f' % (
    np.mean(stats['pts_P']), *np.percentile(stats['pts_P'], [10, 90]), np.mean(stats['pts_C'])))
print('per ring %s m, pattern / control: %s' % (RINGS, np.round(stats['ringP'] / np.maximum(stats['ringC'], 1), 3)))
print('per ring pattern pts/frame: %s' % np.round(stats['ringP'] / f).astype(int))
print('14.7 deg channel: %.2f%% of points, %.3f%% of Car-box points' % (
    100 * stats['ch0_pts'] / stats['all_pts'], 100 * stats['ch0_box_pts'] / max(stats['box_pts'], 1)))
print('Car boxes with >= 1 point in the full cloud: %d; emptied by the pattern: %d (%.1f%%)' % (
    stats['car_hit'], stats['car_emptied'], 100 * stats['car_emptied'] / max(stats['car_hit'], 1)))
print('  of them centred within 70 m: %d; emptied by the pattern: %d (%.1f%%)' % (
    stats['car_hit70'], stats['car_emptied70'], 100 * stats['car_emptied70'] / max(stats['car_hit70'], 1)))
for d, (raw_l, pat_l, exp_l) in lines.items():
    if raw_l:
        print('lines per Car box at %d m (n=%d): full %.0f -> pattern %.0f (median; lattice expectation %.2f)' % (
            d, len(raw_l), np.median(raw_l), np.median(pat_l), np.median(exp_l)))
print('Car boxes reaching the model: %d, with 0 points: %d' % (stats['cars_after'], stats['zero_after']))
