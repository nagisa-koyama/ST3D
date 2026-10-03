"""A learned per-point sampler: the per-bin density rule plus a small learned correction.

experiments_md/20261003_04. The processor step `sample_points_learned` keeps each point with
probability

    p_i = sigmoid( logit(rate[bin(r_i)]) + mlp(features_i) )

where `rate` is the per-bin keep rate the histogram rule would use (measured once by the trainer,
tools/analysis/train_point_sampler.py, and stored with the weights) and the MLP is a 2-layer network
over per-point local-geometry features whose LAST layer is zero at initialisation - so an untrained
sampler reproduces the rule exactly (the L0 ablation), and training can only move away from it where
the target's local geometry says so.

Everything is numpy: no torch state, so the object pickles into spawned DataLoader workers unchanged
and runs inside the CPU loader. Features per point: planar range, height about the sensor, elevation
angle, log point counts in 0.25 / 0.5 / 1.0 m grid cells, and azimuth (cos, sin) - so a target that
covers only a cone (PandarGT) is learned from its clouds, not declared by a key.
"""
import numpy as np

FEATURE_NAMES = ('range', 'z', 'elev_deg', 'log_n025', 'log_n05', 'log_n10', 'cos_az', 'sin_az')


def _cell_counts(xyz, size):
    """Points per occupied cubic cell of `size`, looked up per point: one integer key per cell, no tree."""
    cells = np.floor(xyz / size).astype(np.int64) + (1 << 20)  # non-negative within +-262 km
    key = (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    return counts[inv].astype(np.float32)


def point_features(points, shift_z=0.0):
    """(N, 8) float32 features; `shift_z` undoes the dataset's SHIFT_COOR so elevation is about the sensor.

    Local density is read from cubic grid cells at 0.25 / 0.5 / 1.0 m (log counts), not from
    radius queries: a KD-tree with two ball queries cost ~4 s per accumulated frame and made the
    first learned-sampler rows (27259 / 27260) loader-bound by ~8x. Three unique-with-counts hashes
    are ~0.1 s on a 360k-point cloud, and the sampler only needs density to within a cell.
    """
    xyz = points[:, :3].astype(np.float32)
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    z = xyz[:, 2] - shift_z
    elev = np.degrees(np.arctan2(z, np.maximum(r, 1e-3)))
    az = np.arctan2(xyz[:, 1], xyz[:, 0])  # the FOV lives here: a limited-FOV target is learned, never set by a key
    if len(xyz) == 0:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)
    n025, n05, n10 = (_cell_counts(xyz, s) for s in (0.25, 0.5, 1.0))
    return np.stack([r, z, elev, np.log(n025), np.log(n05), np.log(n10), np.cos(az), np.sin(az)],
                    axis=1).astype(np.float32)


class LearnedPointSampler:
    """Holds the rate table, feature normalisation and MLP weights; pure numpy at inference."""

    def __init__(self, rate, max_dist, shift_z, feat_mean, feat_std, w1, b1, w2, b2, w3, b3):
        self.rate = np.asarray(rate, np.float32)
        self.max_dist = float(max_dist)
        self.shift_z = float(shift_z)
        self.feat_mean = np.asarray(feat_mean, np.float32)
        self.feat_std = np.asarray(feat_std, np.float32)
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = [np.asarray(a, np.float32) for a in (w1, b1, w2, b2, w3, b3)]

    # ---- persistence
    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(**{k: d[k] for k in ('rate', 'max_dist', 'shift_z', 'feat_mean', 'feat_std', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3')})

    def save(self, path):
        np.savez(path, rate=self.rate, max_dist=self.max_dist, shift_z=self.shift_z, feat_mean=self.feat_mean,
                 feat_std=self.feat_std, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, w3=self.w3, b3=self.b3)

    @classmethod
    def from_rule(cls, rate, max_dist, shift_z, feat_mean, feat_std, hidden=32, seed=0):
        """The rule, plus an MLP whose last layer is zero: p == rate at initialisation."""
        rng = np.random.default_rng(seed)
        n_in = len(FEATURE_NAMES)
        w1 = rng.normal(0, 1 / np.sqrt(n_in), (n_in, hidden)); b1 = np.zeros(hidden)
        w2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden)); b2 = np.zeros(hidden)
        w3 = np.zeros((hidden, 1)); b3 = np.zeros(1)
        return cls(rate, max_dist, shift_z, feat_mean, feat_std, w1, b1, w2, b2, w3, b3)

    # ---- inference
    def rule_logit(self, points):
        r = np.hypot(points[:, 0], points[:, 1])
        idx = np.floor(np.clip(r, 0, self.max_dist - 1e-4) / self.max_dist * len(self.rate)).astype(np.int64)
        p = np.clip(self.rate[idx], 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).astype(np.float32)

    def mlp(self, feats):
        x = (feats - self.feat_mean) / self.feat_std
        h = np.maximum(x @ self.w1 + self.b1, 0)
        h = np.maximum(h @ self.w2 + self.b2, 0)
        return (h @ self.w3 + self.b3)[:, 0]

    def keep_probability(self, points, feats=None):
        if feats is None:
            feats = point_features(points, self.shift_z)
        logit = np.clip(self.rule_logit(points) + self.mlp(feats), -30, 30)  # exp overflow is harmless but noisy
        return 1.0 / (1.0 + np.exp(-logit))

    def sample(self, points, rng=np.random):
        if len(points) == 0:
            return points
        return points[rng.rand(len(points)) < self.keep_probability(points)]
