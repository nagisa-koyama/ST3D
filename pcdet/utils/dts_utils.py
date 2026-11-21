"""DTS (Hu, Liu, Hu, CVPR 2023, "Density-insensitive Unsupervised Domain Adaption on 3D Object Detection") ported to
CenterPoint in this fork: experiments_md 20261003_05 §5 item 51 (DTS and ST3D++ re-run under our protocol).

Reference: the released code WoodwindHu/DTS @ 4aad196 (read-only clone /home/koyama/data/released_repos/DTS), cited
below as DTS/<path>:<line>. Every departure from it is named where it is made.

`ring_upsample` is RBRS's up-sampling half (DTS `random_beam_upsample`, pcdet/datasets/augmentor/data_augmentor.py:141-166),
used on the labelled source during source pretraining (the D0 model). Departure: the ring of each point is the sensor's
stored ring index (nuScenes column 4, label-free), and angles are taken about the sensor origin. DTS labels rings by a
per-frame KMeans of elevation about the SHIFT_COOR-shifted origin (DTS data_augmentor.py:92-111 after
nuscenes_dataset.py:125-126), which mixes rings near the sensor.
"""
import numpy as np


def polar(points):
    """DTS downsample_utils.compute_angles + get_polar_image: azimuth phi in [0, 2 pi), elevation theta = atan(z / rho),
    3D range r. The [0, 2 pi) convention fixes where the seam lies (+x), which decides which pairs DTS never forms."""
    x, y, z = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64), points[:, 2].astype(np.float64)
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    phi[phi >= 2.0 * np.pi] = 0.0
    theta = np.arctan2(z, np.hypot(x, y))
    r = np.sqrt(x * x + y * y + z * z)
    return phi, theta, r


def _nearest(sorted_vals, queries):
    """Index into sorted_vals of the nearest value to each query (absolute difference, no wrap) and that difference."""
    j = np.searchsorted(sorted_vals, queries)
    lo = np.clip(j - 1, 0, len(sorted_vals) - 1)
    hi = np.clip(j, 0, len(sorted_vals) - 1)
    d_lo, d_hi = np.abs(queries - sorted_vals[lo]), np.abs(queries - sorted_vals[hi])
    take_hi = d_hi < d_lo
    return np.where(take_hi, hi, lo), np.where(take_hi, d_hi, d_lo)


def ring_upsample(points, rings, beam_prob, phi_threshold, n_rings=None, rand=np.random.rand):
    """New points between adjacent rings, as DTS `random_beam_upsample`.

    For each ring pair (i, i + 1), i = 0 .. n_rings - 2, drawn with probability beam_prob[i] (one `rand()` per pair, in
    order, as DTS): every point of ring i takes the point of ring i + 1 nearest in azimuth; if that azimuth difference is
    below phi_threshold (rad, no wrap), a new point is added at the midpoint of the pair in (phi, theta, r), with every
    further column (intensity, ring, ...) averaged too. The input points are kept unchanged, first, in order.

    points: (N, C) with x, y, z in the sensor frame first. rings: (N,) integer ring index, adjacent indices adjacent in
    elevation (nuScenes: index 0 lowest, checked 2026-11-22 on three train frames). beam_prob: scalar or one value per
    pair. Returns (N + M, C) of points.dtype. An empty ring forms no pair (DTS's argmin would raise there).
    """
    rings = np.asarray(rings).astype(np.int64)
    if n_rings is None:
        n_rings = int(rings.max()) + 1 if len(rings) else 0
    probs = np.broadcast_to(np.asarray(beam_prob, dtype=np.float64), (max(n_rings - 1, 0),))
    if len(points) == 0 or n_rings < 2:
        return points
    phi, theta, r = polar(points)
    pol = points.astype(np.float64).copy()
    pol[:, 0], pol[:, 1], pol[:, 2] = phi, theta, r
    new = []
    for i in range(n_rings - 1):
        if not rand() < probs[i]:
            continue
        cur, nxt = np.flatnonzero(rings == i), np.flatnonzero(rings == i + 1)
        if len(cur) == 0 or len(nxt) == 0:
            continue
        order = nxt[np.argsort(phi[nxt], kind='stable')]
        k, d = _nearest(phi[order], phi[cur])
        ok = d < phi_threshold
        if not ok.any():
            continue
        mid = (pol[cur[ok]] + pol[order[k[ok]]]) / 2.0
        out = mid.copy()
        out[:, 0] = np.cos(mid[:, 1]) * np.cos(mid[:, 0]) * mid[:, 2]
        out[:, 1] = np.cos(mid[:, 1]) * np.sin(mid[:, 0]) * mid[:, 2]
        out[:, 2] = np.sin(mid[:, 1]) * mid[:, 2]
        new.append(out)
    if not new:
        return points
    return np.concatenate([points] + [n.astype(points.dtype) for n in new], axis=0)
