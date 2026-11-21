"""PandaSet semantic segmentation reader and dataset (both sensors; experiments_md episodes/20261116_01 §6, pair P1).

The point transform is ST3D's PandasetDataset._get_lidar_points without its training-only masks: world -> ego with the
frame's lidar pose (pandaset devkit), the axis swap to the normative frame (x forward, y left, z up), intensity / 255.
tests/test_pcseg_pandaset_reader.py checks the points are identical to ST3D's reader over >= 40 frames. Labels come
from annotations/semseg/<frame>.pkl.gz, one row per row of the lidar frame (both devices), and are cut by the SAME
index as the points.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from .class_map import build_class_map
from .grid_sample import grid_sample

N_FRAMES = 80


def _ego_from_world(xyz_world, pose):
    import pandaset as ps
    ego = ps.geometry.lidar_points_to_ego(xyz_world, pose)
    ego = ego[:, [1, 0, 2]]     # PandaSet ego (x right, y forward, z up) -> normative (x forward, y left, z up)
    ego[:, 1] = -ego[:, 1]
    return ego


def has_semseg(root, seq):
    return os.path.isdir(os.path.join(root, seq, 'annotations', 'semseg'))


def load_poses(root, seq):
    with open(os.path.join(root, seq, 'lidar', 'poses.json')) as f:
        return json.load(f)


def read_frame(root, seq, frame, device, pose):
    """One labelled frame. device 0 = Pandar64, 1 = PandarGT, -1 = both.

    Returns (points [N, 4] float32 x, y, z, intensity in the normative ego frame without SHIFT_COOR,
             raw_labels [N] int64 PandaSet class ids, keep [N] indices of these points in the stored frame).
    """
    lidar = pd.read_pickle(os.path.join(root, seq, 'lidar', '%02d.pkl.gz' % frame))
    sem = pd.read_pickle(os.path.join(root, seq, 'annotations', 'semseg', '%02d.pkl.gz' % frame))
    if len(lidar) != len(sem):
        raise ValueError('%s/%02d: %d points but %d labels' % (seq, frame, len(lidar), len(sem)))
    d = lidar['d'].to_numpy()
    keep = np.arange(len(lidar)) if device == -1 else np.flatnonzero(d == device)
    xyz = lidar[['x', 'y', 'z']].to_numpy()[keep]
    intensity = lidar['i'].to_numpy()[keep] / 255
    raw = sem['class'].to_numpy()[keep].astype(np.int64)
    ego = _ego_from_world(xyz, pose)
    points = np.concatenate([ego, intensity[:, None]], axis=1).astype(np.float32)
    return points, raw, keep


def augment(coord, rng, cfg):
    """Pointcept's nuScenes SpUNet training augmentation (semseg-spunet-v1m1-0-base), in its order. Moves points, never
    drops or reorders them, so the labels need no index."""
    coord = coord.copy()
    if cfg.get('ROTATE_Z_PROB', 0) and rng.random() < cfg['ROTATE_Z_PROB']:
        a = rng.uniform(-1, 1) * np.pi
        c, s = np.cos(a), np.sin(a)
        coord = coord @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=coord.dtype)
    if cfg.get('SCALE', None):
        coord *= rng.uniform(*cfg['SCALE'])
    if cfg.get('FLIP_PROB', 0):
        if rng.random() < cfg['FLIP_PROB']:
            coord[:, 0] = -coord[:, 0]
        if rng.random() < cfg['FLIP_PROB']:
            coord[:, 1] = -coord[:, 1]
    if cfg.get('JITTER', None):
        sigma, clip = cfg['JITTER']
        coord += np.clip(sigma * rng.standard_normal(coord.shape), -clip, clip).astype(coord.dtype)
    return coord


def frame_key(seq, frame):
    return '%s/%02d' % (seq, frame)


def key_seed(key, salt=0):
    """A stable 63-bit seed from a frame key (never from a dataset index)."""
    import hashlib
    return int.from_bytes(hashlib.sha256(('%s|%s' % (key, salt)).encode()).digest()[:8], 'little') >> 1


class PandasetSeg(torch.utils.data.Dataset):
    """cfg keys: ROOT (dir holding the sequence dirs), SEQUENCES, LIDAR_DEVICE, SHIFT_COOR, CLASS_MAP, GRID_SIZE,
    AUG (training only; None = no augmentation), FACTOR_OPS (optional: a factor-swap block, pcseg/factor_ops.py;
    train_seg.py passes DATA.FACTOR_SWAP to the train set and DATA.EVAL_PERTURB to the val set). A RENDER entry in that
    block (pcseg/render_cache.py) replaces each frame's cloud by its cached re-render first; the block's other
    operators then act on the rendered cloud."""

    def __init__(self, cfg, training, logger=None):
        self.cfg, self.training = cfg, training
        self.root = cfg['ROOT']
        self.device = int(cfg['LIDAR_DEVICE'])
        self.shift = np.asarray(cfg.get('SHIFT_COOR', [0.0, 0.0, 0.0]), dtype=np.float32)
        self.class_map = build_class_map(cfg['CLASS_MAP'])
        self.grid_size = float(cfg['GRID_SIZE'])
        self.aug = cfg.get('AUG', None) if training else None
        ops = cfg.get('FACTOR_OPS', None)
        ops = None if ops is None else dict(ops)
        render = None if ops is None else ops.pop('RENDER', None)
        if render is None:
            self.render = None
        else:
            from .render_cache import RenderCache
            self.render = RenderCache(render, self.device)
        if not ops:
            self.factor_ops = None
        else:
            from .factor_ops import FactorOps
            self.factor_ops = FactorOps(ops, self.device, self.shift)
        seqs = list(cfg['SEQUENCES'])
        self.sequences = [s for s in seqs if has_semseg(self.root, s)]
        self.skipped = [s for s in seqs if s not in self.sequences]
        self.frames = [(s, f) for s in self.sequences for f in range(N_FRAMES)]
        self._poses = {}
        if logger is not None:
            logger.info('PandasetSeg (%s, device %d): %d sequences with semseg, %d frames; skipped without semseg: %s'
                        % ('train' if training else 'eval', self.device, len(self.sequences), len(self.frames),
                           self.skipped))
            if self.render is not None:
                logger.info('PandasetSeg (%s, device %d): %s'
                            % ('train' if training else 'eval', self.device, self.render.describe()))
            if self.factor_ops is not None:
                logger.info('PandasetSeg (%s, device %d): factor ops, a fixed frame-keyed counterfactual set: %s'
                            % ('train' if training else 'eval', self.device, self.factor_ops.describe()))

    def __len__(self):
        return len(self.frames)

    def pose(self, seq, frame):
        if seq not in self._poses:
            self._poses[seq] = load_poses(self.root, seq)
        return self._poses[seq][frame]

    def load(self, index):
        """Points (shifted) and model-class labels of one frame, before augmentation and voxelization."""
        seq, frame = self.frames[index]
        if self.render is None:
            points, raw, _ = read_frame(self.root, seq, frame, self.device, self.pose(seq, frame))
        else:
            # The cached re-render (20261116_01 §8.9): new coordinates, and the intensity and raw label of the source
            # point each came from, read here by its (frame, index). The cache holds no label.
            def read(g):
                pts, lab, _ = read_frame(self.root, seq, g, self.device, self.pose(seq, g))
                return pts, lab
            points, raw = self.render.load(read, seq, frame)
        coord = points[:, :3] + self.shift
        key = frame_key(seq, frame)
        if self.factor_ops is None:
            return coord, points[:, 3:4], self.class_map(raw), key
        # Factor ops (20261116_01 §8.9): a keep index and the transformed intensity of the kept points; the labels
        # follow by the SAME index, here, before augmentation and voxelization. The ops never see a label.
        idx, intensity = self.factor_ops(coord, points[:, 3], key)
        return coord[idx], intensity[:, None], self.class_map(raw[idx]), key

    def __getitem__(self, index):
        coord, strength, segment, key = self.load(index)
        if self.training:
            # A fresh draw per sample and epoch from numpy's global RNG, which torch reseeds in every DataLoader worker
            # (base_seed + worker_id; tests/test_pcseg_engine.py guards it).
            rng = np.random.default_rng(np.random.randint(0, 2 ** 31 - 1))
            if self.aug is not None:
                coord = augment(coord, rng, self.aug)
        else:
            rng = np.random.default_rng(key_seed(key))                      # the same voxel draw in every evaluation
        idx, grid_coord, inverse = grid_sample(coord, self.grid_size, rng)
        sample = {
            'grid_coord': grid_coord,
            'feat': np.concatenate([coord[idx], strength[idx]], axis=1).astype(np.float32),
            'segment': segment[idx],
            'frame_key': key,
        }
        if not self.training:
            sample['inverse'] = inverse
            sample['origin_segment'] = segment
            sample['origin_coord'] = coord
        return sample


def collate(samples):
    """Concatenate a list of samples; 'offset' holds cumulative voxel counts (Pointcept's convention), the inverse maps
    are shifted to index the concatenated voxels, 'origin_offset' holds cumulative point counts."""
    out = {'frame_key': [s['frame_key'] for s in samples]}
    n_vox = np.cumsum([len(s['segment']) for s in samples])
    out['offset'] = torch.as_tensor(n_vox, dtype=torch.long)
    out['grid_coord'] = torch.as_tensor(np.concatenate([s['grid_coord'] for s in samples]), dtype=torch.int32)
    out['feat'] = torch.as_tensor(np.concatenate([s['feat'] for s in samples]))
    out['segment'] = torch.as_tensor(np.concatenate([s['segment'] for s in samples]), dtype=torch.long)
    if 'inverse' in samples[0]:
        starts = np.concatenate([[0], n_vox[:-1]])
        out['inverse'] = torch.as_tensor(np.concatenate([s['inverse'] + o for s, o in zip(samples, starts)]),
                                         dtype=torch.long)
        out['origin_segment'] = torch.as_tensor(np.concatenate([s['origin_segment'] for s in samples]),
                                                dtype=torch.long)
        out['origin_offset'] = torch.as_tensor(np.cumsum([len(s['origin_segment']) for s in samples]),
                                               dtype=torch.long)
        out['origin_coord'] = torch.as_tensor(np.concatenate([s['origin_coord'] for s in samples]))
    return out
