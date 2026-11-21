"""S3a factor-swap operators for P1 (experiments_md episodes/20261116_01 §8.9 stage S3a; rules §4, §7, §8.7).

Each operator moves ONE factor of a PandaSet sensor's frames to the other sensor's value. It returns a keep mask or a
transform of the intensity, never a new cloud, and it never sees a label: `PandasetSeg.load` moves the points and the
labels by the index this module returns, in the same function (§4). Two config keys, both absent by default:

  DATA.FACTOR_SWAP   the TRAINING set of an arm (term_k: retrain with factor k at the source's value);
  DATA.EVAL_PERTURB  the VAL set, by the same operator (the reliance form, §8.7 item 2: the unperturbed oracle scored
                     on val perturbed by the factor).

`tools/train_seg.py` hands either to `PandasetSeg` as its FACTOR_OPS. Without them every sample is today's.

Operators, applied in this fixed order whatever the order of the keys:

  CONE       {FOV_DEG, HEADING_DEG}: keep mask, the evaluation's cone (engine.cone_mask, about the sensor's xy origin,
             SHIFT_COOR's xy removed). FOV_DEG >= 360 keeps every point: the identity (engine.cone_mask itself also
             requires x > 0, so at 360 it would keep only the front half). Arm a3.
  DENSITY    {RATES: <json>}: keep mask. A point in planar range bin b is kept with probability rate[b], the bins and
             the rule of detection's `DataProcessor.sample_points_hist_based` (uniform bins on [0, MAX_DIST), points
             beyond in the last bin). The rates are min(1, detection's `per_bin_sample_rate`) of the in-cone
             points-per-frame histograms of the two TRAIN splits (tools/analysis/pcseg_factor_tables.py). Arm a1.
  INTENSITY  {TABLE: <npz>}: transform of the intensity. Per range ring, x -> F_to^-1(F_from(x)), detection's
             `DataProcessor.map_intensity_to_reference` with its tables' format; integer intensities are dequantised
             over one step first. Arms a2 / a4.

Draws (DENSITY's keep draw, INTENSITY's dequantisation) come from `key_seed(frame_key, salt)` and are taken over the
frame's FULL point count, then indexed by the points still kept, so a frame gets the same cloud in every epoch and
worker, and one operator's draw does not depend on another's mask. The arm is a fixed counterfactual dataset, not an
augmentation. numpy's global RNG is never used.

Files carry the device they apply to; a file for the other sensor raises.
"""
import hashlib
import json
import os

import numpy as np

from .engine import cone_mask

ORDER = ('CONE', 'DENSITY', 'INTENSITY')
SALTS = {'DENSITY': 'factor_ops/density', 'INTENSITY': 'factor_ops/intensity'}
DEVICE_NAMES = {0: 'Pandar64', 1: 'PandarGT'}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _key_seed(key, salt):
    from .pandaset_seg import key_seed
    return key_seed(key, salt)


# --- label-free reader (the measurement scripts read no annotation) ------------------------------------------------

def points_from_lidar(lidar, device, pose):
    """read_frame's points from an already loaded lidar DataFrame, without the semseg file.
    Returns (points [N, 4] float32, keep [N] indices into the stored frame)."""
    from .pandaset_seg import _ego_from_world
    d = lidar['d'].to_numpy()
    keep = np.arange(len(lidar)) if device == -1 else np.flatnonzero(d == device)
    xyz = lidar[['x', 'y', 'z']].to_numpy()[keep]
    intensity = lidar['i'].to_numpy()[keep] / 255
    ego = _ego_from_world(xyz, pose)
    return np.concatenate([ego, intensity[:, None]], axis=1).astype(np.float32), keep


def read_points(root, seq, frame, device, pose):
    """The points of `pandaset_seg.read_frame`, label-free (tests/test_pcseg_factor_ops.py: identical)."""
    import pandas as pd
    return points_from_lidar(pd.read_pickle(os.path.join(root, seq, 'lidar', '%02d.pkl.gz' % frame)), device, pose)


# --- pure operator functions (tested against the detection functions on the same inputs and draws) ---------------

def planar_range(coord, shift):
    """Planar range about the sensor's xy origin, float32 like detection's `np.linalg.norm(points[:, 0:2])`."""
    return np.linalg.norm(coord[:, :2] - np.asarray(shift, dtype=coord.dtype)[:2], axis=1)


def range_bins(r, max_dist, num_bins):
    """Detection's sample_points_hist_based binning, verbatim, then clipped so a bin is never out of [0, num_bins)."""
    b = np.floor(np.clip(r, 0, max_dist - 0.0001) / max_dist * num_bins).astype(np.int32)
    return np.clip(b, 0, num_bins - 1)


def density_keep(r, max_dist, rate, u):
    """Keep mask: u (uniform [0, 1), one per point) < rate of the point's range bin."""
    rate = np.asarray(rate, dtype=np.float64)
    return u < rate[range_bins(r, max_dist, len(rate))]


def cone_keep(coord, shift, fov_deg, heading_deg=0.0):
    """The evaluation's cone (engine.evaluate: cone_mask(origin_coord[:, :2] - float32 shift[:2], fov, heading)),
    except that FOV_DEG >= 360 keeps every point."""
    if float(fov_deg) >= 360.0:
        return np.ones(len(coord), dtype=bool)
    return cone_mask(coord[:, :2] - np.asarray(shift, dtype=np.float32)[:2], float(fov_deg), float(heading_deg))


def map_intensity(v, r, tables, u=None):
    """Detection's map_intensity_to_reference on intensities v [N] at planar ranges r [N], with the dequantisation
    draw u [N] (uniform on [-from_step / 2, from_step / 2)) given explicitly. Returns float64."""
    edges, from_q, to_q = tables['edges'], tables['from_q'], tables['to_q']
    levels = np.linspace(0.0, 1.0, from_q.shape[1])
    ring = np.clip(np.searchsorted(edges, r, side='right') - 1, 0, len(edges) - 2)
    v = np.asarray(v, dtype=np.float64)
    if tables['from_step'] > 0:
        if u is None or len(u) != len(v):
            raise ValueError('a dequantisation draw per point is needed (from_step %g)' % tables['from_step'])
        v = np.clip(v + u, 0.0, 1.0)
    out = np.empty_like(v)
    for k in np.unique(ring):
        m = ring == k
        xs = np.maximum.accumulate(from_q[k]) + np.arange(len(levels)) * 1e-12
        out[m] = np.interp(np.interp(v[m], xs, levels), levels, to_q[k])
    return out


# --- magnitudes from histograms (the derivations of tools/analysis/pcseg_factor_tables.py) -------------------------

def density_rates_from_histograms(rho_apply, rho_reference, min_fraction=0.01):
    """min(1, detection's DataProcessor.per_bin_sample_rate), with the sensor being thinned in detection's `src` slot
    and the reference sensor in its `tgt` slot. Under-populated bins (below min_fraction of either side's mean bin)
    keep rate 1, as there."""
    from types import SimpleNamespace
    from pcdet.datasets.processor.data_processor import DataProcessor
    stub = SimpleNamespace(hist_dist_src=np.asarray(rho_apply, dtype=np.float64),
                           hist_dist_tgt=np.asarray(rho_reference, dtype=np.float64),
                           hist_fg_src=None, hist_fg_tgt=None, hist_bg_src=None, hist_bg_tgt=None)
    rate = DataProcessor.per_bin_sample_rate(stub, {'MIN_HIST_BIN_FRACTION': float(min_fraction)})
    return np.minimum(rate, 1.0)


def intensity_tables_from_histograms(hist_from, hist_to, min_points=2000):
    """Detection's table construction (tools/analysis/intensity_testtime_tables.py main()): per ring, quantiles of the
    `from` sensor (mapped) and of the `to` sensor (reference); a ring with < min_points on either side takes the
    all-range quantiles. hist_* are (rings, NUM_LEVELS) counts of dequantised intensity.
    Returns (from_q, to_q, own) with own[k] True where ring k has its own map."""
    from pcdet.datasets.intensity_calibration import quantiles
    hist_from, hist_to = np.asarray(hist_from, np.float64), np.asarray(hist_to, np.float64)
    f_all, t_all = hist_from.sum(axis=0), hist_to.sum(axis=0)
    from_q, to_q, own = [], [], []
    for k in range(len(hist_from)):
        o = hist_to[k].sum() >= min_points and hist_from[k].sum() >= min_points
        from_q.append(quantiles(hist_from[k] if o else f_all))
        to_q.append(quantiles(hist_to[k] if o else t_all))
        own.append(bool(o))
    return np.array(from_q), np.array(to_q), np.array(own)


# --- the operators ----------------------------------------------------------------------------------------------------

class _Cone:
    def __init__(self, spec, device, shift):
        self.fov = float(spec['FOV_DEG'])
        self.heading = float(spec.get('HEADING_DEG', 0.0))
        self.shift = shift

    def keep(self, coord, intensity, key, n0, idx):
        return cone_keep(coord, self.shift, self.fov, self.heading)

    def describe(self):
        return 'CONE %g deg at heading %g%s' % (self.fov, self.heading, ' (identity)' if self.fov >= 360 else '')


class _Density:
    def __init__(self, spec, device, shift):
        self.path = spec['RATES']
        with open(self.path) as f:
            d = json.load(f)
        if d.get('kind') != 'pcseg_density_rate':
            raise ValueError('%s is not a pcseg density-rate file' % self.path)
        if int(d['apply_to_device']) != device:
            raise ValueError('%s thins device %d, but the dataset reads device %d'
                             % (self.path, int(d['apply_to_device']), device))
        self.max_dist = float(d['max_dist'])
        self.rate = np.asarray(d['rate'], dtype=np.float64)
        if len(self.rate) != int(d['num_bins']) or not np.all((self.rate >= 0) & (self.rate <= 1)):
            raise ValueError('%s: %d rates for %s bins, or a rate outside [0, 1]'
                             % (self.path, len(self.rate), d['num_bins']))
        self.sha = sha256_file(self.path)
        self.shift = shift

    def keep(self, coord, intensity, key, n0, idx):
        u = np.random.default_rng(_key_seed(key, SALTS['DENSITY'])).random(n0)[idx]
        return density_keep(planar_range(coord, self.shift), self.max_dist, self.rate, u)

    def describe(self):
        return 'DENSITY %s (sha256 %s): %d bins to %g m, rate %.3f..%.3f' % (
            self.path, self.sha[:12], len(self.rate), self.max_dist, self.rate.min(), self.rate.max())


class _Intensity:
    def __init__(self, spec, device, shift):
        self.path = spec['TABLE']
        t = np.load(self.path)
        if int(t['from_device']) != device:
            raise ValueError('%s maps device %d, but the dataset reads device %d'
                             % (self.path, int(t['from_device']), device))
        self.tables = {k: np.asarray(t[k], dtype=np.float64) for k in ('edges', 'from_q', 'to_q')}
        self.tables['from_step'] = float(t['from_step'])
        self.to_device = int(t['to_device'])
        self.sha = sha256_file(self.path)
        self.shift = shift

    def transform(self, coord, intensity, key, n0, idx):
        step = self.tables['from_step']
        u = None
        if step > 0:
            u = np.random.default_rng(_key_seed(key, SALTS['INTENSITY'])).uniform(-step / 2, step / 2, n0)[idx]
        out = map_intensity(intensity, planar_range(coord, self.shift), self.tables, u)
        return out.astype(intensity.dtype)

    def describe(self):
        return 'INTENSITY %s (sha256 %s): %d rings to %g m, onto device %d, step %g' % (
            self.path, self.sha[:12], len(self.tables['edges']) - 1, self.tables['edges'][-1], self.to_device,
            self.tables['from_step'])


_BUILD = {'CONE': _Cone, 'DENSITY': _Density, 'INTENSITY': _Intensity}


class FactorOps:
    """The operators of one FACTOR_SWAP / EVAL_PERTURB block for one dataset (device, SHIFT_COOR)."""

    def __init__(self, spec, device, shift):
        spec = dict(spec or {})
        unknown = sorted(set(spec) - set(ORDER))
        if unknown or not spec:
            raise ValueError('factor ops: unknown keys %s (known %s)' % (unknown, list(ORDER)) if unknown
                             else 'factor ops: an empty block (omit the key instead)')
        self.device = int(device)
        self.shift = np.asarray(shift, dtype=np.float32)
        self.ops = [(k, _BUILD[k](spec[k], self.device, self.shift)) for k in ORDER if spec.get(k) is not None]
        if not self.ops:
            raise ValueError('factor ops: every operator is null')

    def describe(self):
        return '; '.join(op.describe() for _, op in self.ops)

    def __call__(self, coord, intensity, key):
        """coord [N, 3] (shifted, as load() holds it), intensity [N] -> (idx [M] int64 indices into the input points,
        intensity [M] of those points). The caller moves every per-point array, the labels included, by idx."""
        coord = np.asarray(coord)
        if not np.isfinite(coord).all():
            raise ValueError('%s: non-finite coordinates reach the factor ops' % key)
        n0 = len(coord)
        idx = np.arange(n0, dtype=np.int64)
        inten = np.array(intensity, copy=True)
        for name, op in self.ops:
            if hasattr(op, 'keep'):
                k = op.keep(coord[idx], inten[idx], key, n0, idx)
                idx = idx[k]
            else:
                inten[idx] = op.transform(coord[idx], inten[idx], key, n0, idx)
        if len(idx) == 0:
            raise ValueError('%s: the factor ops kept no point (%s)' % (key, self.describe()))
        return idx, inten[idx]
