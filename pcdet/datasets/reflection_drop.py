"""REFLECTION_DROP, the reflection row R11 / R11r of the factor-swap oracles (ANALYSIS, training only; absent = unchanged;
experiments_md 20261003_05 §5 item 35g): drop the share e(b, c) of a target sensor's returns that a nuScenes-like
sensor would lose inside surfaces (M1, item 35f), per range bin b and height class c.

Classification of a return, as M1 (tools/analysis/reflection_raydrop_m1.py at 73f7af5, run with --ground plane):
  - r = the 3D range about the sensor origin; bins RANGE_EDGES (0-10, ..., 60-75 m); returns within EGO_M (3 m) and
    beyond 75 m are untouched;
  - dz = the height above the frame's label-free ground plane (ground_tilt.lowest_per_cell + fit_plane, 4-40 m, rng
    seeded with M1's SEED; the frame's z mode at 5-30 m when no plane fits); ground |dz| <= 0.3 m, above-ground
    0.3-3.0 m, anything else untouched.
The plane is fitted on the cloud handed in (the sensor's returns about its origin, axes with z up), as M1 fitted it.

Two modes (one per config, `MODE`):
  threshold  (R11)  in cell (b, c) with E > 0, a return with intensity < T is dropped, one with intensity == T is dropped
                    with probability Q; T / Q are the e-quantile of the target's TRAIN intensities in that cell and the
                    tie share that makes the pooled dropped share exactly E (tools/analysis/reflection_r11_stats.py).
  random     (R11r) in cell (b, c), a return is dropped with probability E, whatever its intensity.
E, T, Q are given per class as 7-entry lists over the bins; E = 0 (or null) drops nothing in that cell. Intensity is in
the scale the loader hands in (KITTI raw reflectance, Waymo tanh(raw), PandaSet raw / 255): T is measured in that scale.

Every draw is seeded by the frame key and the mode (crc32('factorswap/<frame>/REFLECTION_DROP_<MODE>')), so a frame is
cut the same way in every epoch and at every seed, as the other factor-swap arms.
"""
import numpy as np

RANGE_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0])
N_BINS = len(RANGE_EDGES) - 1
CLASSES = ('ground', 'above')
EGO_M = 3.0
GROUND_DZ, ABOVE_HI = 0.3, 3.0
GROUND_WIN, GROUND_BIN = (5.0, 30.0), 0.05
PLANE_SEED = 20261107               # M1's SEED: the plane fit's RANSAC draw, so the plane is M1's on the same cloud
MODES = ('threshold', 'random')
KEYS = ('MODE', 'E', 'T', 'Q', 'TOP_CALIB')


def range_bin(r):
    """Bin index of RANGE_EDGES, -1 outside [0, 75) m (M1's range_bin)."""
    r = np.asarray(r, dtype=np.float64)
    b = np.searchsorted(RANGE_EDGES, r, side='right') - 1
    b[(r < RANGE_EDGES[0]) | (r >= RANGE_EDGES[-1])] = -1
    return b


def height_class(dz):
    """0 = ground (|dz| <= 0.3 m), 1 = above-ground (0.3 < dz <= 3.0 m), -1 = neither (M1's height_class)."""
    dz = np.asarray(dz, dtype=np.float64)
    c = np.full(len(dz), -1, dtype=np.int64)
    c[np.abs(dz) <= GROUND_DZ] = 0
    c[(dz > GROUND_DZ) & (dz <= ABOVE_HI)] = 1
    return c


def frame_ground_mode(z, rh):
    """M1's frame_ground: the mode of z (0.05 m bins) over returns at 5-30 m horizontal range; nan if none."""
    m = (rh >= GROUND_WIN[0]) & (rh <= GROUND_WIN[1])
    if not np.any(m):
        return np.nan
    b = np.floor(z[m] / GROUND_BIN).astype(np.int64)
    u, c = np.unique(b, return_counts=True)
    return (u[np.argmax(c)] + 0.5) * GROUND_BIN


def ground_dz(xyz):
    """(dz per return, plane [a, b, c] or None): M1's ground_height(d, 'plane') on (n, 3) float64 sensor-frame points."""
    from .osens2_utils import fit_plane, lowest_per_cell
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    fit = fit_plane(lowest_per_cell(np.c_[x, y, z]), rng=np.random.default_rng(PLANE_SEED))
    if fit is None:
        return z - frame_ground_mode(z, np.hypot(x, y)), None
    a, b, c = fit[0]
    return z - (a * x + b * y + c), [float(a), float(b), float(c)]


def classify(xyz_sensor):
    """(cell per return: bin * 2 + class in [0, 14), -1 = untouched; the plane). `xyz_sensor` (n, 3) about the sensor."""
    xyz = np.asarray(xyz_sensor, dtype=np.float64)
    if len(xyz) == 0:
        return np.zeros(0, dtype=np.int64), None
    r = np.linalg.norm(xyz, axis=1)
    dz, plane = ground_dz(xyz)
    rb, cl = range_bin(r), height_class(dz)
    ok = (rb >= 0) & (cl >= 0) & (r >= EGO_M)
    return np.where(ok, rb * len(CLASSES) + cl, -1), plane


def _table(cfg, key):
    """(14,) float64 over cells bin * 2 + class from {ground: [7], above: [7]}; null -> nan."""
    sub = cfg[key]
    out = np.full(N_BINS * len(CLASSES), np.nan)
    for c, name in enumerate(CLASSES):
        vals = list(sub[name])
        assert len(vals) == N_BINS, 'REFLECTION_DROP.%s.%s needs %d values (bins %s)' % (key, name, N_BINS,
                                                                                      RANGE_EDGES.tolist())
        out[c::len(CLASSES)] = [np.nan if v is None else float(v) for v in vals]
    return out


def check_cfg(cfg):
    unknown = [k for k in cfg.keys() if k not in KEYS]
    assert not unknown, 'REFLECTION_DROP takes only %s, got %s' % (KEYS, unknown)
    assert cfg['MODE'] in MODES, 'REFLECTION_DROP.MODE is one of %s' % (MODES,)
    e = _table(cfg, 'E')
    ee = np.nan_to_num(e, nan=0.0)
    assert np.all((ee >= 0) & (ee < 1)), 'REFLECTION_DROP.E must lie in [0, 1)'
    if cfg['MODE'] == 'threshold':
        t, q = _table(cfg, 'T'), _table(cfg, 'Q')
        on = ee > 0
        assert np.all(np.isfinite(t[on])) and np.all(np.isfinite(q[on])), 'threshold mode needs T and Q where E > 0'
        assert np.all((q[on] >= 0) & (q[on] <= 1)), 'REFLECTION_DROP.Q must lie in [0, 1]'
    return ee


def frame_rng(frame_key, mode):
    from .kitti.kitti_factor_swap import frame_rng as fr
    return fr(frame_key, 'REFLECTION_DROP_%s' % mode.upper())


def drop_mask(cell, inten, cfg, rng):
    """Boolean drop mask over returns given their cells (classify) and intensities; one uniform draw per return."""
    ee = check_cfg(cfg)
    cell = np.asarray(cell, dtype=np.int64)
    inten = np.asarray(inten, dtype=np.float64)
    u = rng.random(len(cell))
    valid = cell >= 0
    ci = np.where(valid, cell, 0)
    e = np.where(valid, ee[ci], 0.0)
    if cfg['MODE'] == 'random':
        return valid & (u < e)
    t = np.where(valid, np.nan_to_num(_table(cfg, 'T'), nan=-np.inf)[ci], -np.inf)
    q = np.where(valid, np.nan_to_num(_table(cfg, 'Q'), nan=0.0)[ci], 0.0)
    on = valid & (e > 0)
    return on & ((inten < t) | ((inten == t) & (u < q)))


def keep_mask(xyz_sensor, inten, cfg, frame_key):
    """The arm on one frame: True = keep. `xyz_sensor` (n, 3) about the sensor origin, z up; `inten` in the loader's
    scale. Returns (keep, cell)."""
    cell, _ = classify(xyz_sensor)
    return ~drop_mask(cell, inten, cfg, frame_rng(frame_key, cfg['MODE'])), cell
