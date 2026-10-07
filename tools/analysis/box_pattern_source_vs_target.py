"""Per-box point ARRANGEMENT, accumulated source (augmentation off) vs target (eval), at matched point counts.

Why: the nuScenes -> Waymo residual is missed sparse cars although the accumulated source holds as many sparse cars
(experiments_md 20261007_03 §7), and the pattern-gap descriptors inside Car boxes differ strongly in nearest-neighbour
distance and points per voxel (§9). This says in which DIRECTION, per box, binned by points in the box: occupied
voxels, points per occupied voxel, occupied z-layers (0.15 m) and occupied BEV cells (0.1 m), median nearest-neighbour
distance. ANALYSIS: reads Car GT boxes on both sides.

    python analysis/box_pattern_source_vs_target.py <cfg> <frames per source> <target frames>
"""
import sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import _augmentation_off
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

VOX = np.array([0.1, 0.1, 0.15])
PTS_BINS = [(10, 50), (50, 200), (200, 10**9)]
RINGS = [(0, 20), (20, 40), (40, 75)]
cfg_file, n_src, n_tgt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
logger = common_utils.create_logger()


def per_box(ds, n):
    rows = []
    step = max(1, len(ds) // n)
    for i in range(0, len(ds), step)[:n]:
        d = ds[i]
        b = d.get('gt_boxes')
        if b is None or not len(b):
            continue
        b = b[b[:, 7] == 1]
        if not len(b):
            continue
        pts = d['points'][:, :3]
        m = roiaware_pool3d_utils.points_in_boxes_cpu(pts, b[:, :7])
        for j in range(len(b)):
            p = pts[m[j] > 0]
            if len(p) < 10:
                continue
            v = np.floor(p / VOX).astype(np.int64)
            occ = np.unique(v, axis=0)
            nn = cKDTree(p).query(p, k=2)[0][:, 1]
            rows.append((np.hypot(*b[j, :2]), len(p), len(occ), len(p) / len(occ), len(np.unique(v[:, 2])),
                         len(np.unique(v[:, :2], axis=0)), np.median(nn)))
    return np.array(rows)


def report(name, r):
    print(f'\n{name}: {len(r)} Car boxes with >= 10 points')
    print('| ring | points in box | boxes | occupied voxels | pts / occupied voxel | z-layers (0.15 m) | BEV cells (0.1 m) | median NN (m) |')
    print('|---|---|---|---|---|---|---|---|')
    for lo, hi in RINGS:
        for a, z in PTS_BINS:
            s = (r[:, 0] >= lo) & (r[:, 0] < hi) & (r[:, 1] >= a) & (r[:, 1] < z)
            if s.sum() < 5:
                continue
            q = np.median(r[s], axis=0)
            print(f'| {lo}-{hi} | {a}-{z - 1 if z < 10**9 else ""} | {s.sum()} | {q[2]:.0f} | {q[3]:.2f} | {q[4]:.0f} | {q[5]:.0f} | {q[6]:.3f} |')


src = []
for key, dc in cfg.DATA_CONFIGS.items():
    ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
    with _augmentation_off(ds):
        src.append(per_box(ds, n_src))
    del ds
report('SOURCE, accumulated (Boston 15 / Singapore 10, augmentation off)', np.concatenate(src))
dst, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                             workers=0, logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
report('TARGET, Waymo val', per_box(dst, n_tgt))
