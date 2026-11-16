"""nuScenes-lidarseg reader and dataset for the reproduction gate G0 (experiments_md episodes/20261116_01 §5), in
Pointcept's NuScenesDataset convention (pointcept/datasets/nuscenes.py, get_data):

- points: the keyframe LIDAR_TOP .pcd.bin of the detection tree, 5 float32 per point; x, y, z in the sensor frame,
  intensity / 255. Pointcept reads only `lidar_path`: no sweeps, no ego-point removal, no range clip. (pcdet's
  nuScenes loader removes ego points and is not used.)
- labels: lidarseg/<version>/<token>_lidarseg.bin of the separate lidarseg tree, one uint8 per point, mapped with
  the `nuscenes_lidarseg16` class map. A label count different from the point count raises.
- frames: the index written by tools/analysis/pcseg_nuscenes_index.py (sample.json order, the devkit's split).
"""
import json
import os

import numpy as np
import torch

from .class_map import build_class_map
from .pandaset_seg import PandasetSeg


def read_frame(root, lidarseg_root, lidar_rel, label_rel):
    """Returns (points [N, 4] float32 x, y, z, intensity / 255 in the LIDAR_TOP frame, raw [N] int64 lidarseg index)."""
    pts = np.fromfile(os.path.join(root, lidar_rel), dtype=np.float32)
    if pts.size % 5:
        raise ValueError('%s: %d floats, not a multiple of 5' % (lidar_rel, pts.size))
    pts = pts.reshape(-1, 5)
    raw = np.fromfile(os.path.join(lidarseg_root, label_rel), dtype=np.uint8).astype(np.int64)
    if len(raw) != len(pts):
        raise ValueError('%s: %d points but %d labels (%s)' % (lidar_rel, len(pts), len(raw), label_rel))
    points = np.concatenate([pts[:, :3], pts[:, 3:4] / 255], axis=1).astype(np.float32)
    return points, raw


def load_index(path):
    with open(path) as f:
        return json.load(f)


class NuScenesSeg(torch.utils.data.Dataset):
    """cfg keys: ROOT (detection tree holding samples/), LIDARSEG_ROOT, INDEX (from pcseg_nuscenes_index.py), SPLIT
    ('train' / 'val'), SHIFT_COOR, CLASS_MAP, GRID_SIZE, AUG (training only; None = no augmentation)."""

    def __init__(self, cfg, training, logger=None):
        self.cfg, self.training = cfg, training
        self.root, self.lidarseg_root = cfg['ROOT'], cfg['LIDARSEG_ROOT']
        self.shift = np.asarray(cfg.get('SHIFT_COOR', [0.0, 0.0, 0.0]), dtype=np.float32)
        self.class_map = build_class_map(cfg['CLASS_MAP'])
        self.grid_size = float(cfg['GRID_SIZE'])
        self.aug = cfg.get('AUG', None) if training else None
        index = load_index(cfg['INDEX'])
        self.split = cfg['SPLIT']
        self.frames = index[self.split]
        self.sequences = sorted({r[3] for r in self.frames})                  # scenes; train_seg checks the count
        self.skipped = []
        if logger is not None:
            logger.info('NuScenesSeg (%s, split %s): %d scenes, %d frames, index %s'
                        % ('train' if training else 'eval', self.split, len(self.sequences), len(self.frames),
                           cfg['INDEX']))

    def __len__(self):
        return len(self.frames)

    def load(self, index):
        """Points (shifted) and model-class labels of one frame, before augmentation and voxelization."""
        token, lidar_rel, label_rel, _ = self.frames[index]
        points, raw = read_frame(self.root, self.lidarseg_root, lidar_rel, label_rel)
        coord = points[:, :3] + self.shift
        return coord, points[:, 3:4], self.class_map(raw), token

    # Augmentation, voxelization and the sample format are PandasetSeg's (they only use load(), training, aug and
    # grid_size), so both datasets feed the same collate, loss and evaluate().
    __getitem__ = PandasetSeg.__getitem__
