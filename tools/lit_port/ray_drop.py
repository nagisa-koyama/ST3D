"""Label-free ray-drop for the second legal LiT row (experiments_md 20261017_01). Plain numpy, no LiT import.

LiT's release drops rays with a learned keep-probability model (`lit.ray_keeper.GBMRayKeeper`, inputs: direction, distance,
incident angle; trained on paired simulated / real frames of nuScenes, which needs nuScenes boxes and is NOT in the release).
This module is the label-free replacement, pre-declared in the report: a keep-probability that depends on the RANGE only,
fitted once from unlabelled nuScenes TRAIN scans against translated Waymo frames:  p_b = min(1, real_b / translated_b)
per range bin b, with real_b / translated_b the mean points per frame in the bin (3-70 m about the sensor).

The drop is a post-filter on the ray caster's output (one point per hit ray), so applying it to stored translated frames is
identical to dropping rays during the cast. The random draw is deterministic per (sequence, frame).
"""
import zlib

import numpy as np

EDGES = np.array([3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], np.float64)
NBIN = len(EDGES) - 1


def bin_of(rng):
    """Range-bin index 0..NBIN-1 for 3 <= r < 70, -1 elsewhere."""
    b = np.digitize(rng, EDGES) - 1
    b[(rng < EDGES[0]) | (rng >= EDGES[-1])] = -1
    return b


def point_ranges(local_points, origin):
    d = np.asarray(local_points, np.float64) - np.asarray(origin, np.float64)
    return np.linalg.norm(d, axis=1)


def keep_mask(local_points, origin, p, seq, frame):
    """Boolean mask over the points: kept with probability p[bin]; points outside 3-70 m are always kept."""
    p = np.asarray(p, np.float64)
    assert p.shape == (NBIN,) and ((p >= 0) & (p <= 1)).all()
    b = bin_of(point_ranges(local_points, origin))
    pp = np.where(b >= 0, p[np.clip(b, 0, NBIN - 1)], 1.0)
    u = np.random.default_rng([zlib.crc32(seq.encode()), int(frame)]).random(len(pp))
    return u < pp


def bin_counts(rng):
    b = bin_of(rng)
    return np.bincount(b[b >= 0], minlength=NBIN).astype(np.float64)
