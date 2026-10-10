"""Beam-level downsampling of a LiDAR point cloud, for the LiDAR Distillation baseline.

Ported from LiDAR-Distillation (Wei et al., ECCV 2022, https://github.com/weiyithu/LiDAR-Distillation,
`pcdet/utils/downsample_utils.py`), whose method is: label every point with the laser ring it came
from, then keep only every `BEAM_RATIO`-th ring, and optionally only every `BIN_RATIO`-th return
within a kept ring. The first factor lowers the beam count; the second lowers the horizontal
sampling rate, which is what the paper writes as `32*` / `16*` versus `32` / `16`.

Three deliberate departures from the released code, each of which it would crash or mislead without:

1. **`np.bool` is gone.** The original writes `np.zeros(...).astype(np.bool)`, removed in NumPy 1.24;
   this container has 1.26.4, where it raises AttributeError. Uses the builtin `bool`.

2. **KMeans runs on a SUBSAMPLE, and the centroids are then reused for every point.** The original
   fits `KMeans(n_clusters=beam)` over every point of every frame. At ~100k points and 64 clusters
   that is seconds per frame, which is affordable once offline (the original precomputes the
   downsampled clouds to disk under `modes/<beam>/`) but not inside a DataLoader worker, and this
   port downsamples on the fly - see `data_processor.downsample_beams` for why. Fitting on
   `max_fit_points` elevations and assigning the rest by nearest centroid is the same partition
   whenever the rings are separated by more than the sampling noise, which is what a ring IS.

3. **Elevation is computed once and reused as the clustering feature.** Unchanged in substance, but
   note the original's `compute_angles` divides by the horizontal range, so a point at the sensor
   origin gives a division by zero; those points are dropped from the fit and assigned ring -1.

WHAT THIS CANNOT DO, and it matters for choosing a source dataset: it recovers rings from the
GEOMETRY of the processed cloud, so it needs that geometry to still carry the ring structure. It
does for KITTI (63 of 64 rings recoverable), Lyft (39 of 40, and 52 of 64) and nuScenes (32 of 32),
and it does NOT for Waymo, whose per-return motion compensation leaves only 35.3% of points within
0.05 deg of a declared beam inclination against a 32.1% null - i.e. no better than chance. Waymo is
the reference paper's own headline setting (Waymo -> nuScenes) and is the one pair this
implementation cannot honestly reproduce from the processed data in this checkout; it would need
re-extraction from the raw range images. See
experiments_md/memory/repo/dataset_sensor_and_platform_facts.md.
"""

import numpy as np


def compute_angles(points):
    """Spherical angles of each point, in DEGREES.

    Args:
        points: (N, 3+C). Only the first three columns are read.
    Returns:
        theta: (N,) elevation, negative below the sensor.
        phi: (N,) azimuth in [0, 360).
        valid: (N,) bool, False where the horizontal range is ~0 and the angles are undefined.
    """
    xy = np.linalg.norm(points[:, 0:2], axis=1)
    valid = xy > 1e-6

    theta = np.zeros(points.shape[0], dtype=np.float64)
    phi = np.zeros(points.shape[0], dtype=np.float64)

    theta[valid] = np.degrees(np.arctan(points[valid, 2] / xy[valid]))
    # atan2 over (y, x) gives (-180, 180]; shift to [0, 360) to match the original's branch on the
    # sign of sin(phi). Equivalent, minus the two inverse trig calls and the phi == 360 special case.
    phi[valid] = np.degrees(np.arctan2(points[valid, 1], points[valid, 0])) % 360.0

    return theta, phi, valid


def fit_beam_centroids(theta, beam, max_fit_points=20000, seed=0):
    """Elevation of each laser ring, as `beam` 1-D KMeans centroids, sorted ascending.

    Fitted on at most `max_fit_points` elevations - see departure 2 in the module docstring.
    """
    from sklearn.cluster import KMeans

    theta = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    if theta.shape[0] > max_fit_points:
        rng = np.random.RandomState(seed)
        theta = theta[rng.choice(theta.shape[0], max_fit_points, replace=False)]

    # n_init is pinned rather than left to sklearn's default, which changed to 'auto' in 1.4: the
    # centroids are a cached calibration, so they must not depend on the installed version.
    estimator = KMeans(n_clusters=beam, n_init=10, random_state=seed)
    estimator.fit(theta)
    return np.sort(estimator.cluster_centers_[:, 0])


def beam_label_from_centroids(theta, centroids, valid=None):
    """Assign each point to the nearest ring centroid.

    Returns:
        label: (N,) int, index into `centroids` (which is sorted, so the index IS the ring order),
            or -1 where `valid` is False.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    # Midpoints between adjacent centroids are the decision boundaries of a 1-D nearest-centroid
    # assignment, so one searchsorted replaces a full distance matrix.
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    label = np.searchsorted(boundaries, theta).astype(np.int64)

    if valid is not None:
        label = np.where(valid, label, -1)
    return label


def generate_mask(phi, label, num_beams, beam_ratio=1, bin_ratio=1):
    """Keep every `beam_ratio`-th ring, and every `bin_ratio`-th return within a kept ring.

    Mirrors the original `generate_mask`. `label` is already in ring order (ascending elevation),
    so the original's `idxs = np.argsort(centroids)` indirection is unnecessary here.

    Args:
        phi: (N,) azimuth in degrees.
        label: (N,) ring index, -1 for points with no ring.
        num_beams: number of rings the labelling used.
        beam_ratio: keep ring 0, ring `beam_ratio`, ... Discards the rest entirely.
        bin_ratio: within each kept ring, keep every `bin_ratio`-th point by azimuth.
    Returns:
        mask: (N,) bool.
    """
    mask = np.zeros(phi.shape[0], dtype=bool)

    for ring in range(0, num_beams, beam_ratio):
        in_ring = label == ring
        if not in_ring.any():
            continue
        if bin_ratio == 1:
            mask[in_ring] = True
            continue
        # Sorting by azimuth before decimating is what makes bin_ratio a HORIZONTAL resolution
        # reduction rather than an arbitrary thinning: consecutive returns of one ring are
        # consecutive in azimuth, so every bin_ratio-th one is an even angular subsample.
        ring_idx = np.flatnonzero(in_ring)
        ring_idx = ring_idx[np.argsort(phi[ring_idx])]
        mask[ring_idx[::bin_ratio]] = True

    return mask


def ring_labels(points, num_beams, sensor_origin=(0.0, 0.0, 0.0), centroids=None,
                max_fit_points=20000, seed=0):
    """Laser ring of every point, with elevation measured about the SENSOR.

    A ring is a cone of constant elevation about the sensor, so the angle must be measured from
    where the sensor is. Every loader in this repo adds SHIFT_COOR to the points at load time,
    which moves the origin to the ground (~1.7-2.0 m below the sensor); measured about that
    origin, one ring's points spread over several degrees with range and the KMeans clusters
    mix rings. Measured 2026-10-01 on six frames each: with the origin left at the shifted
    position, cluster purity against the true rings is 0.09 (KITTI), 0.13 (Lyft 40-beam) and
    0.11 (Lyft 64-beam), and a keep-every-other-ring mask agrees with the true one 50% of the
    time - chance - at every range. Pass `sensor_origin = SHIFT_COOR` for a shifted cloud.

    Must run BEFORE world augmentation: global scaling moves the sensor to `s * SHIFT_COOR` and
    is not recorded, which would bring the same error back at a few cm per percent of scale.

    Returns:
        label: (N,) int ring index in ascending elevation, -1 where the angle is undefined.
        centroids: the ring elevations used; pass them back to skip the fit on the next frame.
    """
    q = np.asarray(points[:, 0:3], dtype=np.float64) - np.asarray(sensor_origin, dtype=np.float64)
    theta, _, valid = compute_angles(q)
    if centroids is None:
        centroids = fit_beam_centroids(theta[valid], num_beams, max_fit_points=max_fit_points,
                                       seed=seed)
    return beam_label_from_centroids(theta, centroids, valid=valid), centroids


def random_ring_subset_mask(label, num_beams, keep_beams, rng=None):
    """Keep `keep_beams` of the `num_beams` rings, chosen uniformly at random for this frame.

    The beam-count-matching control for a dense -> sparse pair whose ratio is not an integer
    (Lyft 40-beam -> nuScenes 32 cannot be reached by keeping every k-th ring). Drawing the subset
    per frame, rather than fixing it, keeps every elevation represented over training, as random
    beam re-sampling does. Points with no ring (label -1) are dropped, as in `generate_mask`.
    """
    assert 0 < keep_beams <= num_beams, (keep_beams, num_beams)
    rng = np.random if rng is None else rng
    kept = rng.choice(num_beams, keep_beams, replace=False)
    return np.isin(label, kept)


def downsample_beams(points, num_beams, beam_ratio=1, bin_ratio=1, centroids=None,
                     max_fit_points=20000, seed=0):
    """One-call beam downsampling. Returns the kept points and the centroids used.

    Pass `centroids` back in on the next call to skip the KMeans fit; the ring elevations are a
    property of the sensor, not of the frame, so they are a calibration and not a per-frame quantity.
    Note that world-level augmentation (z rotation, uniform scaling, axis flips) leaves every point's
    elevation unchanged, so cached centroids stay valid after augmentation.
    """
    if beam_ratio == 1 and bin_ratio == 1:
        return points, centroids

    theta, phi, valid = compute_angles(points)
    if centroids is None:
        centroids = fit_beam_centroids(theta[valid], num_beams, max_fit_points=max_fit_points,
                                       seed=seed)
    label = beam_label_from_centroids(theta, centroids, valid=valid)
    mask = generate_mask(phi, label, num_beams, beam_ratio=beam_ratio, bin_ratio=bin_ratio)
    return points[mask], centroids


def ring_thin_mask(points, ring, mode, stride=2, rng=None, num_beams=32):
    """ANALYSIS, evaluation only (experiments_md 20261010_01 §3): which points of ONE scan to keep when its scan
    pattern is cut, given each point's laser ring (nuScenes stores it as the .bin's 5th column).

    'rows' keeps every STRIDE-th ring (whole scan lines removed); 'cols' keeps every STRIDE-th return of every ring
    in azimuth order (the azimuth step x STRIDE); 'random' / 'random_cols' keep the SAME NUMBER of points as
    'rows' / 'cols', drawn uniformly, so every line keeps points (the count-matched controls). Azimuth is taken about
    the points' own origin, so call it on the sensor-frame scan, before SHIFT_COOR.
    """
    ring = np.asarray(ring).astype(np.int64)
    if mode in ('rows', 'random'):
        base = (ring % stride) == 0
    elif mode in ('cols', 'random_cols'):
        phi = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        base = generate_mask(phi, ring, num_beams, beam_ratio=1, bin_ratio=stride)
    else:
        raise ValueError("MODE must be 'rows', 'cols', 'random' or 'random_cols', got %r" % (mode,))
    if mode in ('rows', 'cols'):
        return base
    rng = np.random.default_rng(0) if rng is None else rng
    keep = np.zeros(len(ring), dtype=bool)
    keep[rng.choice(len(ring), int(base.sum()), replace=False)] = True
    return keep
