"""PandaSet Pandar64: laser channel of every point from the stored FIRING order, and RING_PATTERN thinning.

PandaSet ships no lidar -> ego extrinsic and stores points in WORLD coordinates, so a ring cannot be read off the
elevation about any origin the release gives: about the pose origin the rings are not cones at all, and even about
the fitted sensor origin a ~0.85 deg pitch of the sensor against the pose frame smears every ring by +-0.9 deg -
five times the 0.18 deg spacing of the fine band (KMeans elevation clustering agrees with the labels below on 3-5%
of points). What survives is ORDER and TIME. The raw `lidar/NN.pkl.gz` frames keep the Pandar64's firing order:
one firing block (~55.5 us, 0.2 deg of azimuth at 10 Hz) after another, each block's returns in channel order, top
laser first; and every point keeps its own firing timestamp `t`. Inside a block the firmware fires each channel at a
fixed offset (exact to the timestamp quantum, sd 0.000 us over 5,682 complete blocks) and places it in one of six
azimuth columns ~2.08 deg apart. Those three per-channel constants (`EL`, `OFF_US`, `DAZ`) are the template below.

`channel_labels` segments the stored order into blocks (a block ends where the elevation about the sensor steps up,
or its firing window would exceed 52.5 us) and assigns every block's points to strictly increasing channels with a
Viterbi whose transition costs compare CONSECUTIVE points - difference in firing time, in azimuth column and (both
beyond 15 m) in elevation - with the template's. Differences inside one block are free of the block's unknown
start time, its azimuth, the vehicle's motion and the sensor's tilt; a loose absolute-elevation term (+-1 deg free)
only anchors a block's first point. Measured on six TRAIN frames at 1-18.5 m/s (experiments_md 20261010_05): 64 of
64 channels on every frame; elevation-only and timing-only labelings (two independent cues) agree on 96.8-99.5% of
points; the combined labels are time-consistent to one timestamp quantum on 99.9-100%.

Template and origin are measured from PandaSet TRAIN clouds only (source data, no labels):
`tools/analysis/pandaset_ring_cache.py --template`. ORIGIN is the sensor in PandaSet's EGO axes (x right,
y forward, z up; the frame `ps.geometry.lidar_points_to_ego` returns, before the loader's axis swap and SHIFT_COOR):
1.86 m above the pose origin, the height at which in-block elevations become monotone (1,852 blocks against the
1,853-1,855 the firing period predicts; 1.6 or 2.2 m give 10,000+).

`RING_PATTERN` (PandasetDataset, training only; experiments_md 20261010_05) thins the Pandar64 cloud to a target
sensor's published scan pattern with the rule of pcdet/datasets/waymo/waymo_rings.py (27807): keep the channels
nearest a vertical lattice of the target's spacing with a random phase, and one point per kept channel per azimuth
bin of the target's step, also with a random phase. The labels are precomputed per frame (`LABEL_CACHE`), because
the Viterbi costs ~0.5 s a frame.
"""
import os

import numpy as np

N_CHANNELS = 64
# Per-channel template, channel 0 = top laser. Medians over complete 64-point blocks of 32 frames from 16 PandaSet
# TRAIN sequences. EL: elevation (deg) about ORIGIN, points beyond 10 m (2 m for the two bottom channels, which
# hit the road within 10 m), within ~0.15 deg of the published Pandar64 profile. OFF_US: firing time after the
# block's earliest channel (us). DAZ: azimuth offset from the block's median azimuth (deg).
EL = np.array([
    14.734, 10.888, 7.930, 4.916, 2.903, 1.896, 1.711, 1.527, 1.353, 1.261, 1.070, 0.878, 0.695, 0.511, 0.339, 0.249,
    0.060, -0.130, -0.314, -0.497, -0.675, -0.762, -0.953, -1.143, -1.330, -1.509, -1.690, -1.772, -1.965, -2.156,
    -2.338, -2.517, -2.699, -2.783, -2.970, -3.160, -3.340, -3.517, -3.696, -3.786, -3.969, -4.155, -4.345, -4.522,
    -4.704, -4.800, -4.983, -5.164, -5.352, -5.524, -5.706, -5.809, -5.991, -6.173, -7.165, -8.159, -9.167, -9.959,
    -10.978, -11.946, -12.886, -13.817, -18.986, -25.008])
OFF_US = np.array([
    19.55, 18.36, 16.93, 15.74, 14.31, 13.11, 51.98, 43.15, 4.05, 33.38, 47.21, 51.02, 51.98, 51.98, 27.42, 31.47,
    45.06, 49.11, 51.98, 51.98, 25.51, 29.33, 43.15, 21.46, 51.98, 47.21, 23.60, 51.98, 51.98, 27.42, 31.47, 45.06,
    21.46, 51.98, 51.98, 25.51, 29.33, 49.11, 2.62, 51.02, 11.68, 23.60, 6.44, 19.55, 1.43, 18.36, 10.49, 14.31,
    5.25, 15.74, 0.00, 16.93, 9.06, 13.11, 4.05, 7.87, 2.62, 11.68, 6.44, 1.43, 0.00, 10.49, 5.25, 9.06])
DAZ = np.array([
    -0.09, -0.06, -0.03, -0.01, -0.01, -0.02, -2.10, -4.17, -6.26, 4.17, 2.08, -0.01, -2.09, -4.18, -6.26, 4.18,
    2.09, -0.01, -2.08, -4.16, -6.24, 4.19, 2.10, 0.01, -2.07, -4.15, -6.23, 4.20, 2.12, 0.02, -2.06, -4.13, -6.21,
    4.22, 2.13, 0.04, -2.03, -4.12, -6.20, 4.23, 2.14, 0.05, -2.02, -4.10, -6.18, 4.25, 2.16, 0.07, -2.00, -4.09,
    -6.17, 4.27, 2.18, 0.09, 0.11, 0.12, 0.15, 0.18, 0.22, 0.29, 0.37, 0.43, 0.85, 1.33])
ORIGIN = np.array([0.0, 0.5, 1.86])   # sensor in PandaSet ego axes (x right, y forward, z up), metres

BLOCK_SPAN_US = 52.5   # the firing window: every complete block spans 51.98 us, the next starts ~55.5 us later
BLOCK_JUMP_DEG = 0.05  # a block ends where the elevation about the sensor steps UP by more than this


def sensor_angles(xyz_ego, origin=ORIGIN):
    """(horizontal range m, elevation deg, azimuth deg) of every point about the sensor, in PandaSet ego axes."""
    q = np.asarray(xyz_ego, dtype=np.float64)[:, :3] - np.asarray(origin, dtype=np.float64)
    r = np.hypot(q[:, 0], q[:, 1])
    return r, np.degrees(np.arctan2(q[:, 2], r)), np.degrees(np.arctan2(q[:, 1], q[:, 0]))


def _numba_kernels():
    import numba

    @numba.njit(cache=True)
    def segment(t, el, span, jump, n_ch):
        n = len(t)
        starts = [0]
        t0 = t[0]
        m = 1
        for i in range(1, n):
            if el[i] - el[i - 1] > jump or abs(t[i] - t0) > span or m >= n_ch:
                starts.append(i)
                t0 = t[i]
                m = 0
            elif t[i] < t0:
                t0 = t[i]
            m += 1
        starts.append(n)
        return np.array(starts)

    @numba.njit(cache=True)
    def viterbi(st, t, el, az, r, EL, OFF, DAZ, use_t, use_az, use_el):
        C = len(EL)
        lab = np.empty(len(t), np.int64)
        D = np.empty(C)
        Dn = np.empty(C)
        bp = np.empty((C + 1, C), np.int64)
        for b in range(len(st) - 1):
            s = st[b]
            n = st[b + 1] - s
            tol = 1.0 if r[s] > 5 else 3.0
            for c in range(C):
                x = abs(el[s] - EL[c]) - tol
                D[c] = x / 0.2 if (use_el and x > 0) else 0.0
            for i in range(1, n):
                k = s + i
                dt = t[k] - t[k - 1]
                da = (az[k] - az[k - 1] + 180.0) % 360.0 - 180.0
                de = el[k] - el[k - 1]
                far = r[k] > 15 and r[k - 1] > 15
                rmin = max(min(r[k], r[k - 1]), 0.1)
                tola = 0.3 + 2.0 / rmin    # near returns carry each laser's few-cm horizontal offset
                tol = 1.0 if r[k] > 5 else 3.0
                for c2 in range(C):
                    best = np.inf
                    arg = -1
                    for c1 in range(c2):       # channel order is strictly increasing inside a block
                        v = D[c1]
                        if v >= best:
                            continue
                        if use_t:
                            v += abs(dt - (OFF[c2] - OFF[c1])) / 0.3
                        if use_az:
                            x = abs(da - (DAZ[c2] - DAZ[c1])) - tola
                            if x > 0:
                                v += x / 0.3
                        if use_el and far:
                            x = abs(de - (EL[c2] - EL[c1])) - 0.08
                            if x > 0:
                                v += x / 0.1
                        if v < best:
                            best = v
                            arg = c1
                    x = abs(el[k] - EL[c2]) - tol
                    Dn[c2] = best + (x / 0.2 if (use_el and x > 0) else 0.0)
                    bp[i, c2] = arg
                for c in range(C):
                    D[c] = Dn[c]
            c = 0
            bv = np.inf
            for cc in range(C):
                if D[cc] < bv:
                    bv = D[cc]
                    c = cc
            for i in range(n - 1, -1, -1):
                lab[s + i] = c
                if i > 0:
                    c = bp[i, c]
        return lab

    return segment, viterbi


_KERNELS = None


def channel_labels(xyz_ego, t_us, origin=ORIGIN, use_time=True, use_azimuth=True, use_elevation=True):
    """Channel (0 = top laser) of every Pandar64 point of ONE frame, plus the number of blocks found.

    `xyz_ego`: the device-0 points of a raw `lidar/NN.pkl.gz` frame in STORED order, in PandaSet ego axes (world ->
    ego with that frame's pose; a rigid transform keeps the order). `t_us`: their timestamps in microseconds (any
    offset). The `use_*` switches drop one cue, for the independent-cue purity check only.
    """
    global _KERNELS
    xyz_ego = np.asarray(xyz_ego, dtype=np.float64)
    if len(xyz_ego) == 0:
        return np.zeros(0, dtype=np.int64), 0
    if _KERNELS is None:
        _KERNELS = _numba_kernels()
    segment, viterbi = _KERNELS
    t = np.ascontiguousarray(t_us, dtype=np.float64)
    r, el, az = sensor_angles(xyz_ego, origin)
    st = segment(t, el, BLOCK_SPAN_US, BLOCK_JUMP_DEG, N_CHANNELS)
    lab = viterbi(st, t, el, az, r, EL, OFF_US, DAZ, use_time, use_azimuth, use_elevation)
    return lab, len(st) - 1


def kept_channels(spacing_deg, phase_v, target_fov_deg=None, el_table=EL):
    """Channels nearest a vertical lattice of `spacing_deg` (offset `phase_v` in [0, 1) of a spacing), the lattice
    restricted to the source's field of view and, if given, to the target's published one [lo, hi] deg."""
    lo, hi = float(el_table.min()), float(el_table.max())
    if target_fov_deg is not None:
        lo, hi = max(lo, float(target_fov_deg[0])), min(hi, float(target_fov_deg[1]))
    j0 = int(np.floor(lo / spacing_deg - phase_v)) - 1
    j1 = int(np.ceil(hi / spacing_deg - phase_v)) + 1
    targets = (np.arange(j0, j1 + 1) + phase_v) * spacing_deg
    targets = targets[(targets >= lo) & (targets <= hi)]
    if len(targets) == 0:
        targets = np.array([(lo + hi) / 2.0])
    return np.unique(np.argmin(np.abs(el_table[None, :] - targets[:, None]), axis=1))


def channels_in_fov(target_fov_deg, spacing_deg, el_table=EL):
    """Channels within half a lattice spacing of the target's field of view [lo, hi] deg. For this template and the
    HDL-32E's field of view and spacing it is exactly the set `kept_channels` can return over all phases (pinned by a
    test), so a control row that applies only this cut differs from RING_PATTERN in the pattern and not in the field
    of view. It drops the 14.7 deg channel alone."""
    lo, hi = float(target_fov_deg[0]) - spacing_deg / 2.0, float(target_fov_deg[1]) + spacing_deg / 2.0
    return np.nonzero((el_table >= lo) & (el_table <= hi))[0]


def ring_pattern_mask(channel, az_deg, spacing_deg, az_res_deg, phase_v, phase_h, target_fov_deg=None):
    """Keep the points of `kept_channels(...)`, and of those one per channel per azimuth bin of `az_res_deg` (offset
    `phase_h` of a bin) - the first in stored order, which is firing order. `az_res_deg` None keeps every point of
    the kept channels. Points labelled < 0 are dropped. Returns (keep mask, kept channel ids)."""
    channel = np.asarray(channel)
    kept = kept_channels(spacing_deg, phase_v, target_fov_deg)
    keep = np.isin(channel, kept)
    if az_res_deg is None or len(channel) == 0:
        return keep, kept
    n_bins = int(round(360.0 / az_res_deg))
    b = np.floor(((np.asarray(az_deg) + 180.0) / 360.0 + phase_h / n_bins) * n_bins).astype(np.int64) % n_bins
    key = channel.astype(np.int64) * n_bins + b
    idx = np.nonzero(keep)[0]
    _, first = np.unique(key[idx], return_index=True)
    out = np.zeros(len(channel), dtype=bool)
    out[idx[first]] = True
    return out, kept


def cache_path(cache_dir, sequence, frame_idx):
    return os.path.join(cache_dir, str(sequence), '%02d.npy' % int(frame_idx))


def load_cached_labels(cache_dir, sequence, frame_idx, n_points):
    """Channel labels of one frame from the cache `tools/analysis/pandaset_ring_cache.py` writes; refuses a file
    whose length differs from the frame's device-0 point count (a stale cache or another device's points)."""
    path = cache_path(cache_dir, sequence, frame_idx)
    lab = np.load(path)
    assert len(lab) == n_points, 'ring cache %s holds %d labels for a frame of %d Pandar64 points' % (
        path, len(lab), n_points)
    return lab.astype(np.int64)


EVAL_THIN_MODES = ('rows', 'random', 'cols', 'random_cols', 'azbin', 'random_azbin', 'pattern', 'elmax', 'elmin')


def eval_thin_mask(xyz_ego, labels, cfg, rng):
    """ANALYSIS, evaluation only (EVAL_RING_THIN; experiments_md 20261011_02): which points of ONE Pandar64 scan (stored
    order, PandaSet ego axes) to keep under one scan-pattern manipulation, from the scan's channel labels.

    'rows' keeps every STRIDE-th channel (0 = top laser), so whole lines go. 'cols' keeps every STRIDE-th return of each
    channel in firing order (the azimuth step x STRIDE). 'azbin' keeps one point per channel per AZ_RES_DEG azimuth bin,
    the first in firing order. 'random' / 'random_cols' / 'random_azbin' keep the SAME NUMBER of points as their
    structured twin, drawn uniformly over the scan (the count-matched controls; counts match on the raw device-0 scan).
    'pattern' is RING_PATTERN's render (SPACING_DEG, AZ_RES_DEG, TARGET_FOV_DEG) with phases drawn from `rng`.
    'elmax' / 'elmin' drop the channels whose elevation in the sensor's own frame (EL) lies above / below EL_DEG - a
    vertical field-of-view cut by lines, so the sensor's pitch against the pose frame does not tilt it.
    """
    labels = np.asarray(labels).astype(np.int64)
    n = len(labels)
    mode = cfg['MODE']
    assert mode in EVAL_THIN_MODES, 'EVAL_RING_THIN.MODE must be one of %s, got %r' % (EVAL_THIN_MODES, mode)
    if n == 0:
        return np.zeros(0, dtype=bool)
    stride = int(cfg.get('STRIDE', 2))
    base_mode = mode[len('random_'):] if mode.startswith('random_') else ('rows' if mode == 'random' else mode)
    if base_mode == 'rows':
        base = (labels % stride) == 0
    elif base_mode == 'cols':
        order = np.argsort(labels, kind='stable')                       # stored (firing) order within each channel
        rank = np.empty(n, dtype=np.int64)
        starts = np.searchsorted(labels[order], labels[order], side='left')
        rank[order] = np.arange(n) - starts
        base = (rank % stride) == 0
    elif base_mode == 'azbin':
        _, _, az = sensor_angles(xyz_ego)
        n_bins = int(round(360.0 / float(cfg['AZ_RES_DEG'])))
        b = np.floor((az + 180.0) / 360.0 * n_bins).astype(np.int64) % n_bins
        _, first = np.unique(labels * n_bins + b, return_index=True)
        base = np.zeros(n, dtype=bool)
        base[first] = True
    elif base_mode == 'pattern':
        _, _, az = sensor_angles(xyz_ego)
        base, _ = ring_pattern_mask(labels, az, float(cfg['SPACING_DEG']), float(cfg['AZ_RES_DEG']), rng.random(),
                                    rng.random(), cfg.get('TARGET_FOV_DEG', None))
    elif base_mode == 'elmax':
        base = EL[labels] <= float(cfg['EL_DEG'])
    else:  # elmin
        base = EL[labels] >= float(cfg['EL_DEG'])
    if not (mode == 'random' or mode.startswith('random_')):
        return base
    keep = np.zeros(n, dtype=bool)
    keep[rng.choice(n, int(base.sum()), replace=False)] = True
    return keep


def ring_pattern_points(xyz_ego, labels, cfg, rng=None):
    """Keep mask of RING_PATTERN over one frame's device-0 points (stored order, PandaSet ego axes)."""
    rng = np.random if rng is None else rng
    draw = rng.random_sample if hasattr(rng, 'random_sample') else rng.random   # np.random or a Generator
    phase_v, phase_h = draw(), draw()
    if len(labels) == 0:
        return np.zeros(0, dtype=bool)
    _, _, az = sensor_angles(xyz_ego)
    keep, _ = ring_pattern_mask(labels, az, cfg['SPACING_DEG'], cfg['AZ_RES_DEG'], phase_v, phase_h,
                                cfg.get('TARGET_FOV_DEG', None))
    return keep
