"""SemanticKITTI reader and dataset for pair P2 (experiments_md episodes/20261128_01), in Pointcept's
SemanticKITTIDataset convention (pointcept/datasets/semantic_kitti.py, get_data):

- points: dataset/sequences/<seq>/velodyne/<frame>.bin, 4 float32 per point: x, y, z in the sensor frame and the
  remission, used as stored (multiples of 0.01 in [0, 0.99]; 20261128_01 §2). No range clip, no shift.
- labels: dataset/sequences/<seq>/labels/<frame>.label, one uint32 per point; the semantic id is the low 16 bits
  (`label & 0xFFFF`), mapped with a class map that raises on an undeclared id. A label count different from the point
  count raises.
- splits: the official ones, train 00-07, 09, 10 and val 08 (SemanticKITTI's semantic-kitti.yaml).
- FACTOR_OPS (optional, absent by default): an INTENSITY block of pcseg/factor_ops.py for sensor 'semantickitti'. It
  transforms the remission only, keyed by the frame key; points and labels are unchanged.
"""
import os

import numpy as np
import torch

from .class_map import build_class_map
from .pandaset_seg import PandasetSeg

SENSOR = 'semantickitti'      # the factor-ops sensor name (pcseg/factor_ops.py)
INTENSITY_STEP = 0.01         # stored remission step (20261128_01 §2; CPU 1 checks every frame)
SPLITS = {'train': ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'], 'val': ['08']}


def frame_key(seq, frame):
    return 'sk/%s/%06d' % (seq, frame)


def list_frames(root, seq):
    """Frame numbers of a sequence, from its velodyne directory, in order."""
    names = sorted(os.listdir(os.path.join(root, seq, 'velodyne')))
    frames = [int(n[:-4]) for n in names if n.endswith('.bin')]
    if frames != list(range(len(frames))):
        raise ValueError('%s: velodyne frames are not 0..%d' % (seq, len(frames) - 1))
    return frames


def read_points(root, seq, frame):
    """[N, 4] float32 x, y, z, remission of one scan, label-free (the .label file is not read)."""
    pts = np.fromfile(os.path.join(root, seq, 'velodyne', '%06d.bin' % frame), dtype=np.float32)
    if pts.size % 4:
        raise ValueError('%s/%06d: %d floats, not a multiple of 4' % (seq, frame, pts.size))
    return pts.reshape(-1, 4)


def read_labels(root, seq, frame):
    """[N] int64 semantic ids (label & 0xFFFF) of one scan."""
    lab = np.fromfile(os.path.join(root, seq, 'labels', '%06d.label' % frame), dtype=np.uint32)
    return (lab & 0xFFFF).astype(np.int64)


def read_frame(root, seq, frame):
    """Returns (points [N, 4] float32, raw [N] int64 semantic ids)."""
    points = read_points(root, seq, frame)
    raw = read_labels(root, seq, frame)
    if len(raw) != len(points):
        raise ValueError('%s/%06d: %d points but %d labels' % (seq, frame, len(points), len(raw)))
    return points, raw


class SemanticKittiSeg(torch.utils.data.Dataset):
    """cfg keys: ROOT (the dataset/sequences directory), SEQUENCES, SHIFT_COOR, CLASS_MAP, GRID_SIZE, AUG (training
    only; None = no augmentation), FACTOR_OPS (optional, INTENSITY only), FRAME_STEP (optional, default 1: keep every
    k-th frame of each sequence, for smokes)."""

    def __init__(self, cfg, training, logger=None):
        self.cfg, self.training = cfg, training
        self.root = cfg['ROOT']
        self.shift = np.asarray(cfg.get('SHIFT_COOR', [0.0, 0.0, 0.0]), dtype=np.float32)
        self.class_map = build_class_map(cfg['CLASS_MAP'])
        self.grid_size = float(cfg['GRID_SIZE'])
        self.aug = cfg.get('AUG', None) if training else None
        ops = cfg.get('FACTOR_OPS', None)
        if ops:
            from .factor_ops import FactorOps
            self.factor_ops = FactorOps(dict(ops), SENSOR, self.shift)
        else:
            self.factor_ops = None
        step = int(cfg.get('FRAME_STEP', 1) or 1)
        if step < 1:
            raise ValueError('FRAME_STEP %d' % step)
        self.sequences = list(cfg['SEQUENCES'])
        self.skipped = []
        self.frames = [(s, f) for s in self.sequences for f in list_frames(self.root, s)[::step]]
        if logger is not None:
            logger.info('SemanticKittiSeg (%s): sequences %s, %d frames (every %d)'
                        % ('train' if training else 'eval', self.sequences, len(self.frames), step))
            if self.factor_ops is not None:
                logger.info('SemanticKittiSeg (%s): factor ops, a fixed frame-keyed counterfactual set: %s'
                            % ('train' if training else 'eval', self.factor_ops.describe()))

    def __len__(self):
        return len(self.frames)

    def load(self, index):
        """Points (shifted) and model-class labels of one frame, before augmentation and voxelization."""
        seq, frame = self.frames[index]
        points, raw = read_frame(self.root, seq, frame)
        coord = points[:, :3] + self.shift
        key = frame_key(seq, frame)
        if self.factor_ops is None:
            return coord, points[:, 3:4], self.class_map(raw), key
        idx, intensity = self.factor_ops(coord, points[:, 3], key)
        if len(idx) != len(coord):
            raise ValueError('%s: P2 factor ops must keep every point' % key)
        return coord, intensity[:, None], self.class_map(raw), key

    # Augmentation, voxelization and the sample format are PandasetSeg's, as for NuScenesSeg.
    __getitem__ = PandasetSeg.__getitem__
