"""Laser rings of a Lyft LIDAR_TOP scan and RING_PATTERN thinning (experiments_md 20261010_07).

What the files hold (measured 2026-10-10): no ring index (the .bin's 5th column is constant 1); the stored order is
laser-major (one laser's sweep after another, top to bottom, azimuth-ordered within a sweep); and the cloud is
MOTION-COMPENSATED - every point was moved to one reference time. So elevation and azimuth about the stored origin
drift with ego speed (the lowest laser swings 0.1 deg around the turn when stopped, 3.6 / 7.0 deg above 8 m/s on the
40- / 64-beam platform), which breaks ring recovery from geometry. This module undoes the compensation ONLY to identify
the lasers: each point is moved back to the sensor at its firing time, using the ego motion of the previous sweep
(source poses, label-free). Thinning keeps the ORIGINAL points.

`ring_pattern_mask` follows pcdet/datasets/waymo/waymo_rings.py's rule (27807, Waymo -> nuScenes): keep the rings
nearest a vertical lattice of the target's published spacing inside the source's field of view, and one point per kept
ring per azimuth bin of the target's azimuth step, both with a random phase.
"""
import numpy as np


def elevation_deg(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.degrees(np.arctan2(xyz[:, 2], np.maximum(np.hypot(xyz[:, 0], xyz[:, 1]), 1e-6)))


def azimuth_deg(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))


def ego_motion(info, yaw_only=False):
    """(velocity (3,) m/s, angular velocity (3,) rad/s), both in the keyframe's LIDAR_TOP frame and constant over the
    rotation, from the most recent previous sweep: its `transform_matrix` maps that sweep into the keyframe frame, so
    the sensor sat at T[:3, 3], rotated by T[:3, :3], `time_lag` seconds earlier (0.2 s in the Lyft infos). The
    rotation is small (< 2 deg), so its rotation vector is read off the skew part. `yaw_only` keeps only the z rate.
    None when the infos carry no usable sweep (a scene's first keyframe)."""
    for sw in info.get('sweeps', []) or []:
        T, lag = sw.get('transform_matrix', None), sw.get('time_lag', 0.0)
        if T is None or not lag or lag <= 0:
            continue
        T = np.asarray(T, dtype=np.float64)
        R = T[:3, :3]
        rotvec = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if yaw_only:
            rotvec[:2] = 0.0
        return -T[:3, 3] / lag, -rotvec / lag
    return None


def scan_direction(az_deg):
    """+1 if the stored azimuth increases within a laser sweep, -1 if it decreases."""
    step = (np.diff(az_deg) + 180.0) % 360.0 - 180.0
    return 1.0 if np.median(step) >= 0 else -1.0


def firing_offset(az_deg, sign, az_start_deg, period_s, ref_fraction):
    """Firing time of each point minus the compensation's reference time (s). A sweep starts at `az_start_deg` and
    turns once in direction `sign` over `period_s`; the reference time sits `ref_fraction` of the way through it. The
    compensated azimuth stands in for the firing azimuth: its error is d / r, i.e. milliseconds of time."""
    phase = (((np.asarray(az_deg) - az_start_deg) * sign) % 360.0) / 360.0
    return (phase - ref_fraction) * period_s


def deskew(xyz, velocity, ang_vel, dt):
    """Points in the sensor frame at their own firing time: undo the move to the reference time, assuming constant
    velocity and angular velocity (first order in the rotation, which stays below a few milliradians over one
    rotation). `dt` is firing time minus reference time per point."""
    xyz = np.asarray(xyz, dtype=np.float64)
    q = xyz - np.outer(dt, velocity)
    theta = -np.outer(dt, np.asarray(ang_vel, dtype=np.float64))   # per-point rotation vector
    return q + np.cross(theta, q)


def deskew_frame(xyz, motion, sign, az_start_deg, period_s, ref_fraction, iters=2):
    """De-skewed points (sensor at each point's firing time) of one stored scan. The firing time comes from the
    azimuth; iteration replaces the compensated azimuth with the de-skewed one (near points' azimuth is off by d / r
    after compensation, which misplaces their time by milliseconds). `motion` = ego_motion(info); None -> unchanged."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if motion is None or len(xyz) == 0:
        return xyz.copy()
    v, w = motion
    q = xyz
    for _ in range(max(1, iters)):
        dt = firing_offset(azimuth_deg(q), sign, az_start_deg, period_s, ref_fraction)
        q = deskew(xyz, v, w, dt)
    return q


def rings_from_deskewed(q, sign, az_start_deg, snap_deg=10.0, half=4, merge_deg=0.05):
    """Ring index per point (0 = first stored ring) and ring elevations, from the de-skewed stored scan `q`.

    A laser's sweep starts at `az_start_deg` and runs once round in direction `sign`, lasers one after another
    (laser-major order). A new ring starts where the azimuth phase about `az_start_deg` wraps; each start is then moved
    to the largest drop in de-skewed elevation (medians of `half` points either side) within `snap_deg` of travel,
    because lasers start their sweep a few degrees apart. Neighbouring rings whose median elevations differ by less
    than `merge_deg` are one laser and are merged."""
    q = np.asarray(q, dtype=np.float64)
    n = len(q)
    if n < 2 * half + 2:
        return np.zeros(n, dtype=np.int64), (np.array([np.median(elevation_deg(q))]) if n else np.zeros(0))
    az, el = azimuth_deg(q), elevation_deg(q)
    phase = ((az - az_start_deg) * sign) % 360.0
    wraps = np.nonzero(np.diff(phase) < -180.0)[0] + 1
    med = np.median(np.lib.stride_tricks.sliding_window_view(el, half), axis=1)
    drop = np.full(n, -np.inf)
    drop[half:n - half + 1] = med[:n - 2 * half + 1] - med[half:]
    starts = []
    for b in wraps:                                  # window: the end of one sweep and the start of the next
        lo, hi = b, b
        while lo - 1 > half and phase[lo - 1] > 360.0 - snap_deg:
            lo -= 1
        while hi + 1 < n - half and phase[hi + 1] < snap_deg:
            hi += 1
        starts.append(lo + int(np.argmax(drop[lo:hi + 1])))
    start = np.zeros(n, dtype=np.int64)
    start[np.unique([s for s in starts if 0 < s < n])] = 1
    ring = np.cumsum(start)
    lev = np.array([np.median(el[ring == k]) for k in range(ring[-1] + 1)])
    if merge_deg > 0 and len(lev) > 1:
        new = np.concatenate([[0], np.cumsum(np.abs(np.diff(lev)) >= merge_deg)])
        ring = new[ring]
        lev = np.array([np.median(el[ring == k]) for k in range(ring[-1] + 1)])
    return ring, lev


def unwrap_phase_by_ring(phase, ring):
    """Within each ring the firing phase rises in stored order. A point fired just after the sweep start whose
    compensated azimuth was pushed back across the start reads phase ~1 instead of ~0 (and the mirror case at the
    end): those are moved by one turn, so their firing time is not off by a whole rotation."""
    phase = np.array(phase, dtype=np.float64)
    for k in np.unique(ring):
        idx = np.nonzero(ring == k)[0]
        ph = phase[idx]
        low = np.nonzero(ph < 0.5)[0]
        if len(low) == 0 or len(low) == len(ph):
            continue
        head = np.arange(low[0])                              # leading points before the first low phase
        phase[idx[head[ph[head] > 0.5]]] -= 1.0
        high = np.nonzero(ph >= 0.5)[0]
        tail = np.arange(high[-1] + 1, len(ph))               # trailing points after the last high phase
        phase[idx[tail[ph[tail] < 0.5]]] += 1.0
    return phase


def deskew_and_rings(xyz, motion, sign, az_start_deg, period_s, ref_fraction, iters=2):
    """The full recovery: de-skew (`iters` passes), rings, then one more de-skew with the firing phase unwrapped per
    ring (`unwrap_phase_by_ring`) and the final rings. Returns (de-skewed xyz, ring per point, ring elevations)."""
    q = deskew_frame(xyz, motion, sign, az_start_deg, period_s, ref_fraction, iters=iters)
    ring, lev = rings_from_deskewed(q, sign, az_start_deg)
    if motion is None or len(xyz) == 0:
        return q, ring, lev
    v, w = motion
    phase = unwrap_phase_by_ring((((azimuth_deg(q) - az_start_deg) * sign) % 360.0) / 360.0, ring)
    q = deskew(xyz, v, w, (phase - ref_fraction) * period_s)
    ring, lev = rings_from_deskewed(q, sign, az_start_deg)
    return q, ring, lev


def ring_pattern_mask(xyz, ring, ring_el, spacing_deg, az_res_deg, phase_v, phase_h):
    """Keep the rings nearest a vertical lattice of `spacing_deg` (offset `phase_v` in [0, 1) of a spacing) inside the
    source's field of view, and one point per kept ring per azimuth bin of `az_res_deg` (offset `phase_h`). Points are
    in stored order, which is firing order within a ring, so the point kept per bin is the first one the scan reached.
    `ring_el[k]` is ring k's elevation (deg). Returns (keep mask, indices of the kept rings)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0 or len(ring_el) == 0:
        return np.zeros(len(xyz), dtype=bool), np.zeros(0, dtype=np.int64)
    ring_el = np.asarray(ring_el, dtype=np.float64)
    lo, hi = ring_el.min(), ring_el.max()
    j0 = int(np.floor((lo - phase_v * spacing_deg) / spacing_deg)) - 1
    j1 = int(np.ceil((hi - phase_v * spacing_deg) / spacing_deg)) + 1
    targets = (np.arange(j0, j1 + 1) + phase_v) * spacing_deg
    targets = targets[(targets >= lo) & (targets <= hi)]
    if len(targets) == 0:                                 # field of view narrower than one spacing
        targets = np.array([(lo + hi) / 2.0])
    kept_rings = np.unique(np.argmin(np.abs(ring_el[None, :] - targets[:, None]), axis=1))
    keep = np.isin(ring, kept_rings)
    n_bins = int(round(360.0 / az_res_deg))
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    b = np.floor(((az + np.pi) / (2 * np.pi) + phase_h / n_bins) * n_bins).astype(np.int64) % n_bins
    key = np.asarray(ring).astype(np.int64) * n_bins + b
    idx = np.nonzero(keep)[0]
    _, first = np.unique(key[idx], return_index=True)
    out = np.zeros(len(xyz), dtype=bool)
    out[idx[first]] = True
    return out, kept_rings
