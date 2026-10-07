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
# (azimuth sector, range bin) cells: the trainer's count target and the gate of the learned correction.
N_SECTORS = 24  # 15 deg azimuth sectors
RANGE_EDGES = np.arange(0, 75.01, 2.5)  # 30 planar-range bins
N_CELLS = N_SECTORS * (len(RANGE_EDGES) - 1)


def cell_index(r, az):
    """Cell per point from planar range and azimuth (radians, as arctan2(y, x))."""
    sec = np.clip(np.floor((az + np.pi) / (2 * np.pi) * N_SECTORS).astype(np.int64), 0, N_SECTORS - 1)
    rb = np.clip(np.digitize(r, RANGE_EDGES) - 1, 0, len(RANGE_EDGES) - 2)
    return sec * (len(RANGE_EDGES) - 1) + rb


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


# ---- a virtual target range image (experiments_md/20261007_03 §12, 20261003_04 §38) --------------------------------------
LATTICE_FEATURE_NAMES = ('log_pix_cands', 'pix_rank', 'log_vox_pts')
DET_VOXEL = np.array([0.1, 0.1, 0.15]); DET_PCR = np.array([-75.2, -75.2, -2.0, 75.2, 75.2, 4.0])


def lattice_pixels(xyz, inclinations, n_cols, sensor_z):
    """Pixel of every point in a virtual spinning lidar at (0, 0, sensor_z): row = nearest of `inclinations` (radians,
    any order), column = azimuth bin of n_cols. Points outside the vertical field of view get pixel -1 (the sentinel).
    Returns (pixel, row, 3D range from the virtual sensor)."""
    d = xyz[:, :3].astype(np.float64) - np.array([0.0, 0.0, sensor_z])
    r = np.hypot(d[:, 0], d[:, 1]); rng = np.hypot(r, d[:, 2])
    el = np.arctan2(d[:, 2], np.maximum(r, 1e-6)); az = np.arctan2(d[:, 1], d[:, 0])
    inc = np.sort(np.asarray(inclinations, np.float64))                      # ascending
    k = np.clip(np.searchsorted(inc, el), 1, len(inc) - 1)
    k = np.where(np.abs(el - inc[k - 1]) <= np.abs(el - inc[k]), k - 1, k)  # nearest ascending index
    lo = inc[0] - (inc[1] - inc[0]) / 2; hi = inc[-1] + (inc[-1] - inc[-2]) / 2
    row = len(inc) - 1 - k                                                   # rows top -> bottom (row 0 = highest beam)
    col = np.clip(np.floor((az + np.pi) / (2 * np.pi) * n_cols).astype(np.int64), 0, n_cols - 1)
    valid = (el >= lo) & (el <= hi)
    pix = np.where(valid, row * n_cols + col, -1)
    return pix, np.where(valid, row, -1), rng


def zbuffer_keep(pix, rng):
    """True for the NEAREST point of every pixel (pixel -1 never kept): one return per pixel, as a range image holds."""
    keep = np.zeros(len(pix), bool); idx = np.nonzero(pix >= 0)[0]
    if len(idx):
        o = idx[np.lexsort((rng[idx], pix[idx]))]
        first = np.ones(len(o), bool); first[1:] = pix[o][1:] != pix[o][:-1]
        keep[o[first]] = True
    return keep


def lattice_features(xyz, pix, rng):
    """Per point: log candidates in its pixel, its range rank inside the pixel (0 = nearest, capped at 7), log points in
    its detector voxel. Points with pixel -1 get 0 candidates / rank 7."""
    n = len(pix); cand = np.zeros(n, np.float32); rank = np.full(n, 7.0, np.float32)
    idx = np.nonzero(pix >= 0)[0]
    if len(idx):
        _, inv, cnt = np.unique(pix[idx], return_inverse=True, return_counts=True); cand[idx] = cnt[inv]
        o = idx[np.lexsort((rng[idx], pix[idx]))]; grp = pix[o]
        start = np.r_[0, np.nonzero(grp[1:] != grp[:-1])[0] + 1]
        pos = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)]))
        rank[o] = np.minimum(pos, 7)
    vk = np.floor((xyz[:, :3] - DET_PCR[:3]) / DET_VOXEL).astype(np.int64)
    key = (vk[:, 0] * 2000 + vk[:, 1]) * 100 + vk[:, 2]
    _, vinv, vcnt = np.unique(key, return_inverse=True, return_counts=True)
    return np.stack([np.log1p(cand), rank, np.log(vcnt[vinv].astype(np.float32))], 1).astype(np.float32)


class LearnedPointSampler:
    """Holds the rate table, feature normalisation and MLP weights; pure numpy at inference."""

    def __init__(self, rate, max_dist, shift_z, feat_mean, feat_std, w1, b1, w2, b2, w3, b3, obs_mask=None,
                 rule_kind='rate', lattice_inc=None, lattice_cols=0, lattice_sensor_z=0.0, rule_margin=4.0):
        self.rate = np.asarray(rate, np.float32)
        self.max_dist = float(max_dist)
        self.shift_z = float(shift_z)
        self.feat_mean = np.asarray(feat_mean, np.float32)
        self.feat_std = np.asarray(feat_std, np.float32)
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = [np.asarray(a, np.float32) for a in (w1, b1, w2, b2, w3, b3)]
        # Cells the TARGET returned points in. The learned correction acts only there; elsewhere the keep
        # probability is the rule's (measured where the target observes). None = every cell (samplers
        # trained before 2026-10-04 carry no mask and behave exactly as before). 20261003_04 section 17.
        self.obs_mask = None if obs_mask is None else np.asarray(obs_mask, bool)
        # rule_kind 'rate' = the per-bin density rule (every sampler before 2026-10-08). 'zbuffer' = a virtual target range
        # image: the rule keeps the nearest point of every pixel (logit +rule_margin) and drops the rest (-rule_margin),
        # and the MLP also sees LATTICE_FEATURE_NAMES (20261003_04 §38).
        self.rule_kind = str(rule_kind); self.rule_margin = float(rule_margin)
        self.lattice_inc = None if lattice_inc is None else np.asarray(lattice_inc, np.float64)
        self.lattice_cols = int(lattice_cols); self.lattice_sensor_z = float(lattice_sensor_z)

    # ---- persistence
    @classmethod
    def load(cls, path):
        d = np.load(path)
        kw = {k: d[k] for k in ('rate', 'max_dist', 'shift_z', 'feat_mean', 'feat_std', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3')}
        if 'rule_kind' in d.files and str(d['rule_kind']) == 'zbuffer':
            kw.update(rule_kind='zbuffer', lattice_inc=d['lattice_inc'], lattice_cols=int(d['lattice_cols']),
                      lattice_sensor_z=float(d['lattice_sensor_z']), rule_margin=float(d['rule_margin']))
        return cls(**kw, obs_mask=d['obs_mask'] if 'obs_mask' in d.files else None)

    def save(self, path):
        extra = {} if self.obs_mask is None else {'obs_mask': self.obs_mask}
        if self.rule_kind == 'zbuffer':
            extra.update(rule_kind='zbuffer', lattice_inc=self.lattice_inc, lattice_cols=self.lattice_cols,
                         lattice_sensor_z=self.lattice_sensor_z, rule_margin=self.rule_margin)
        np.savez(path, rate=self.rate, max_dist=self.max_dist, shift_z=self.shift_z, feat_mean=self.feat_mean,
                 feat_std=self.feat_std, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, w3=self.w3, b3=self.b3, **extra)

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

    def lattice_parts(self, points):
        """(rule logit, extra features) of the z-buffer rule for one cloud."""
        pix, _, rng = lattice_pixels(points, self.lattice_inc, self.lattice_cols, self.lattice_sensor_z)
        rule = np.where(zbuffer_keep(pix, rng), self.rule_margin, -self.rule_margin).astype(np.float32)
        return rule, lattice_features(points, pix, rng)

    def keep_probability(self, points, feats=None):
        if self.rule_kind == 'zbuffer':
            rule, fx = self.lattice_parts(points)
            base = point_features(points, self.shift_z) if feats is None else feats[:, :len(FEATURE_NAMES)]
            corr = self.mlp(np.concatenate([base, fx], 1))
            if self.obs_mask is not None:
                corr = corr * self.obs_mask[cell_index(base[:, 0], np.arctan2(base[:, 7], base[:, 6]))]
            return 1.0 / (1.0 + np.exp(-np.clip(rule + corr, -30, 30)))
        if feats is None:
            feats = point_features(points, self.shift_z)
        corr = self.mlp(feats)
        if self.obs_mask is not None:
            corr = corr * self.obs_mask[cell_index(feats[:, 0], np.arctan2(feats[:, 7], feats[:, 6]))]
        logit = np.clip(self.rule_logit(points) + corr, -30, 30)  # exp overflow is harmless but noisy
        return 1.0 / (1.0 + np.exp(-logit))

    def sample(self, points, rng=np.random):
        if len(points) == 0:
            return points
        return points[rng.rand(len(points)) < self.keep_probability(points)]
