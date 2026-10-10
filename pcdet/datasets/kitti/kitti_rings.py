"""Laser rings of a KITTI HDL-64E scan from the stored point ORDER, and evaluation-time scan-pattern cuts (ANALYSIS:
the KITTI arm of the target-oracle degradation table, experiments_md 20261011_02).

KITTI stores each velodyne scan laser-major: one laser's full turn after another, 64 turns, each starting at azimuth
~0 deg and increasing (measured 2026-10-11). The scans are raw, not motion-compensated, so a laser's azimuth is
monotone through the whole file, and a ring ends where the azimuth passes 0 deg again (see `rings_from_order`).
Rings are recovered on the scan as read from disk: before the FOV crop, SHIFT_COOR or any shuffle.

`eval_ring_thin_points` applies one cut, selected by `cfg.MODE`:
- `rows` / `random`: every STRIDE-th ring, or a count-matched random draw (STRIDE 2, 3, 4 = 64 -> 32, 21, 16 rings,
  LiDAR Distillation's integer decimations);
- `cols` / `random_cols`: every STRIDE-th return per ring in azimuth order, or its count-matched draw;
- `azbin` / `random_azbin`: one point per ring per AZ_RES_DEG bin (the first one stored), or its count-matched draw;
- `pattern`: render to a target's scan pattern (27807's rule) - the rings nearest a SPACING_DEG lattice and one point
  per kept ring per AZ_RES_DEG bin, phases drawn per frame.
"""
import numpy as np

from ...utils.beam_downsample_utils import ring_thin_mask
from ..lyft.lyft_rings import azimuth_deg, elevation_deg, ring_pattern_mask


def rings_from_order(xyz, far_m=20.0):
    """(ring per point in stored order, 0 = first stored ring; ring elevations in deg) for one raw KITTI scan.

    A ring ends where the azimuth phase passes 0 deg (the frame's start, facing forward): the phase drops from ~360 to
    ~0. Measured on real val frames (2026-10-11): this gives exactly 64 rings, the elevation jumps by about one ring gap
    (0.4 deg) exactly at each crossing, and the levels descend in 62-63 of 63 neighbouring pairs (the HDL-64E's laser
    blocks interleave once, near -6.3 deg). Elevation is NOT used to place boundaries: the lasers' vertical offsets
    make it change with range at every object edge. A ring's elevation is the median over its points beyond `far_m`
    (all its points if fewer than 20 are that far)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    az, el = azimuth_deg(xyz), elevation_deg(xyz)
    if n == 1:
        return np.zeros(1, dtype=np.int64), el.copy()
    sign = 1.0 if np.median((np.diff(az) + 180.0) % 360.0 - 180.0) >= 0 else -1.0
    phase = (az * sign) % 360.0
    start = np.zeros(n, dtype=np.int64)
    start[1:] = np.diff(phase) < -180.0
    ring = np.cumsum(start)
    far = np.hypot(xyz[:, 0], xyz[:, 1]) > far_m
    lev = np.empty(ring[-1] + 1)
    for k in range(len(lev)):
        m = ring == k
        lev[k] = np.median(el[m & far]) if np.sum(m & far) >= 20 else np.median(el[m])
    return ring, lev


def azbin_mask(xyz, ring, az_res_deg):
    """One point per ring per `az_res_deg` azimuth bin: the first one in stored order (bins start at -180 deg)."""
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)
    b = np.floor((azimuth_deg(xyz) + 180.0) / az_res_deg).astype(np.int64)
    key = np.asarray(ring).astype(np.int64) * (int(np.ceil(360.0 / az_res_deg)) + 1) + b
    keep = np.zeros(len(xyz), dtype=bool)
    keep[np.unique(key, return_index=True)[1]] = True
    return keep


def eval_ring_thin_mask(points, cfg, rng):
    """Keep mask for one raw KITTI scan under the cut `cfg` (MODE, STRIDE, AZ_RES_DEG, SPACING_DEG)."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)
    mode = cfg['MODE']
    ring, lev = rings_from_order(points[:, :3])
    if mode in ('rows', 'random', 'cols', 'random_cols'):
        return ring_thin_mask(points, ring, mode, stride=int(cfg.get('STRIDE', 2)), rng=rng,
                              num_beams=int(ring.max()) + 1)
    if mode in ('azbin', 'random_azbin'):
        keep = azbin_mask(points[:, :3], ring, float(cfg['AZ_RES_DEG']))
        if mode == 'azbin':
            return keep
        out = np.zeros(n, dtype=bool)
        out[rng.choice(n, int(keep.sum()), replace=False)] = True
        return out
    if mode == 'pattern':
        keep, _ = ring_pattern_mask(points[:, :3], ring, lev, float(cfg['SPACING_DEG']), float(cfg['AZ_RES_DEG']),
                                    rng.random(), rng.random())
        return keep
    raise ValueError("EVAL_RING_THIN.MODE must be rows, random, cols, random_cols, azbin, random_azbin or pattern, "
                     "got %r" % (mode,))


def eval_ring_thin_points(points, cfg, rng):
    return points[eval_ring_thin_mask(points, cfg, rng)]
