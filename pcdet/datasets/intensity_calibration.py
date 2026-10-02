"""Per-range, per-class intensity statistics, and the quantile map they define.

The intensity counterpart of `point_calibration`: where the density correction compares points per
radial bin, this compares the DISTRIBUTION of intensity per range ring and per channel (background,
and each class's box interior), so a source cloud's intensities can be mapped onto the target's.

Why per class and per ring (experiments_md/20261002_02):
  * the sensors trend in opposite directions with range (nuScenes brightens, KITTI darkens to 0 by
    40 m), so one map for every range is wrong everywhere;
  * cars are DARKER than background and pedestrians BRIGHTER, in every dataset measured, so one map
    per ring keeps the source's car/background contrast instead of the target's, and a single pooled
    "foreground" channel - dominated by car points - maps pedestrians worse than no split at all.

Units. Intensities are normalised to [0, 1] by the dataset's own `scale` (nuScenes 255, KITTI 1,
PandaSet 1 - its loader already divides by 255) and DEQUANTISED by adding uniform noise over one
quantisation `step` before they are histogrammed. Without that, an integer-valued source (nuScenes car
points sit on a handful of levels: p25 / p50 / p75 = 1 / 3 / 10) maps every tied value to a single
target quantile and the mapped histogram comes out as a comb.

Statistics are histograms over NUM_LEVELS bins of normalised intensity, so they can be summed over
frames, platforms or channel groups before quantiles are taken. Target boxes are whatever the dataset
returns as `gt_boxes`: real annotation for a source, PSEUDO-LABELS for a self-training target, which
is what keeps the target side UDA-legal. Points inside an IGNORED pseudo-label (negative class) go to
no channel: they are neither confident foreground nor background.
"""
import numpy as np

from .point_calibration import _augmentation_off, cone_mask

NUM_LEVELS = 1000
DEFAULT_RING_EDGES = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0)
DEFAULT_FRAMES = 300
QUANTILE_LEVELS = np.linspace(0.0, 1.0, 201)


def point_channels(points, boxes):
    """Channel of every point: 0 background, k >= 1 the class index (column 7) of the first confident
    box containing it, -1 inside an ignored pseudo-label (negative class index).

    Degenerate boxes are dropped first, as `DataProcessor.box_occupancy` does: a zero-extent box
    reaching the C++ kernel is the open `gt_sampling` segfault's failure mode.
    """
    channel = np.zeros(len(points), dtype=np.int64)
    if boxes is None or len(boxes) == 0 or len(points) == 0:
        return channel
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.shape[1] < 8:
        raise ValueError('point_channels needs the class column of gt_boxes (appended in prepare_data)')
    boxes = boxes[(boxes[:, 3:6] > 1e-3).all(axis=1)]
    if len(boxes) == 0:
        return channel
    from ..ops.roiaware_pool3d import roiaware_pool3d_utils
    inside = roiaware_pool3d_utils.points_in_boxes_cpu(
        np.ascontiguousarray(points[:, 0:3], dtype=np.float32),
        np.ascontiguousarray(boxes[:, 0:7])) > 0                       # (M boxes, N points)
    hit = inside.any(axis=0)
    first = np.argmax(inside, axis=0)
    cls = boxes[first, 7].astype(np.int64)
    channel[hit] = np.where(cls[hit] > 0, cls[hit], -1)
    return channel


def normalise(values, scale=1.0, step=0.0, rng=None):
    """Raw intensity -> [0, 1], dequantised over one `step` (raw units)."""
    v = np.asarray(values, dtype=np.float64)
    if step:
        rng = np.random if rng is None else rng
        v = v + rng.uniform(-step / 2.0, step / 2.0, len(v))
    return np.clip(v / float(scale), 0.0, 1.0)


def compute_intensity_statistics(dataset, num_frames=DEFAULT_FRAMES, ring_edges=DEFAULT_RING_EDGES,
                                 scale=1.0, step=0.0, intensity_index=3, indices=None,
                                 fov_degree=None, fov_heading=0.0, seed=0, logger=None, label=''):
    """Histogram of normalised intensity per (range ring, channel), summed over sampled frames.

    Goes through `dataset[idx]` with augmentation off, so the boxes are the ones training would see -
    real labels or pseudo-labels - and reads the points as they stand just before the point feature
    encoder (`keep_raw_points`), which is the last place intensity still exists in an x, y, z
    pipeline. Points beyond the last ring edge are ignored.

    Returns a dict: `hist` (rings, 1 + num_classes, NUM_LEVELS) counts, `edges`, `class_names`,
    `frames`, and the `scale` / `step` used.
    """
    base = dataset if hasattr(dataset, 'prepare_data') else getattr(dataset, 'dataset', dataset)
    positions = list(range(len(base))) if indices is None else list(indices)
    if not positions:
        raise ValueError('cannot measure intensity statistics from an empty dataset')
    stride = max(1, len(positions) // num_frames)
    edges = np.asarray(ring_edges, dtype=np.float64)
    num_classes = len(base.class_names)
    hist = np.zeros((len(edges) - 1, num_classes + 1, NUM_LEVELS), dtype=np.float64)
    rng = np.random.RandomState(seed)
    used = 0
    previous = getattr(base, 'keep_raw_points', False)
    base.keep_raw_points = True
    try:
        with _augmentation_off(base):
            for idx in positions[::stride]:
                if used >= num_frames:
                    break
                d = base[idx]
                pts = d.get('points_raw', None)
                if pts is None or not len(pts):
                    continue
                if fov_degree is not None:
                    pts = pts[cone_mask(pts, fov_degree, fov_heading)]
                r = np.linalg.norm(pts[:, 0:2], axis=1)
                keep = r < edges[-1]
                pts, r = pts[keep], r[keep]
                ch = point_channels(pts, d.get('gt_boxes', None))
                ring = np.clip(np.searchsorted(edges, r, side='right') - 1, 0, len(edges) - 2)
                lvl = np.minimum((normalise(pts[:, intensity_index], scale, step, rng) * NUM_LEVELS)
                                 .astype(np.int64), NUM_LEVELS - 1)
                ok = ch >= 0
                np.add.at(hist, (ring[ok], ch[ok], lvl[ok]), 1.0)
                used += 1
    finally:
        base.keep_raw_points = previous
    if used == 0:
        raise ValueError('no usable frames while measuring intensity statistics')
    if logger is not None:
        names = ['background'] + list(base.class_names)
        for r_i in range(len(edges) - 1):
            cells = ', '.join('%s %d pts p50 %.3f' % (names[c], hist[r_i, c].sum(), _median(hist[r_i, c]))
                              for c in range(num_classes + 1))
            logger.info('intensity statistics [%s] %g-%g m: %s' % (label, edges[r_i], edges[r_i + 1], cells))
    return {'hist': hist, 'edges': edges, 'class_names': list(base.class_names), 'frames': used,
            'scale': scale, 'step': step}


def _median(h):
    return float(quantiles(h, np.array([0.5]))[0]) if h.sum() else float('nan')


def quantiles(hist, levels=QUANTILE_LEVELS):
    """Inverse CDF of one histogram over [0, 1], linear within each bin. NaN if the histogram is empty."""
    h = np.asarray(hist, dtype=np.float64)
    total = h.sum()
    if total <= 0:
        return np.full(len(levels), np.nan)
    cdf = np.concatenate([[0.0], np.cumsum(h) / total])
    # Direct inversion: the bin whose CDF interval contains p, then linear within it. Empty bins are
    # never chosen (their interval has zero width), so leading, trailing and interior gaps all stay
    # empty - interpolating over the CDF knots instead spreads mass across them.
    p = np.clip(np.asarray(levels, dtype=np.float64), 1e-12, 1.0)
    k = np.minimum(np.searchsorted(cdf[1:], p, side='left'), len(h) - 1)
    width = np.maximum(cdf[k + 1] - cdf[k], 1e-15)
    return (k + np.clip((p - cdf[k]) / width, 0.0, 1.0)) / len(h)


def group_hist(hist, groups):
    """Sum channels into groups: `groups` is a list of channel-index lists, e.g. [[1], [2], [0, 3]]."""
    return np.stack([hist[:, g, :].sum(axis=1) for g in groups], axis=1)


def build_intensity_map(src_hist, tgt_hist, groups, min_points=2000):
    """Per (ring, group) quantile tables (src_q, tgt_q), shape (rings, groups, len(QUANTILE_LEVELS)).

    A group with fewer than `min_points` on either side falls back to the ring's POOLED distribution
    (every channel), which is the global per-ring map; a ring with too few points in total maps to
    itself. Both fallbacks are reported in `fallback`, (rings, groups) with 0 own / 1 pooled / 2 identity.
    """
    s, t = group_hist(src_hist, groups), group_hist(tgt_hist, groups)
    s_all, t_all = src_hist.sum(axis=1), tgt_hist.sum(axis=1)
    R, G = s.shape[:2]
    L = len(QUANTILE_LEVELS)
    src_q, tgt_q = np.zeros((R, G, L)), np.zeros((R, G, L))
    fallback = np.zeros((R, G), dtype=np.int64)
    for r in range(R):
        for g in range(G):
            if s[r, g].sum() >= min_points and t[r, g].sum() >= min_points:
                src_q[r, g], tgt_q[r, g] = quantiles(s[r, g]), quantiles(t[r, g])
            elif s_all[r].sum() >= min_points and t_all[r].sum() >= min_points:
                src_q[r, g], tgt_q[r, g] = quantiles(s_all[r]), quantiles(t_all[r])
                fallback[r, g] = 1
            else:
                src_q[r, g] = tgt_q[r, g] = QUANTILE_LEVELS
                fallback[r, g] = 2
    return src_q, tgt_q, fallback


def apply_intensity_map(values, ring, group, src_q, tgt_q):
    """Map normalised intensities through the (ring, group) tables: F_t^-1(F_s(x))."""
    out = np.empty(len(values), dtype=np.float64)
    for r in np.unique(ring):
        for g in np.unique(group[ring == r]):
            m = (ring == r) & (group == g)
            xs = np.maximum.accumulate(src_q[r, g]) + np.arange(len(QUANTILE_LEVELS)) * 1e-12
            out[m] = np.interp(np.interp(values[m], xs, QUANTILE_LEVELS), QUANTILE_LEVELS, tgt_q[r, g])
    return out


def w1(q_a, q_b):
    """Wasserstein-1 between two distributions given as quantile tables on the same levels."""
    return float(np.mean(np.abs(np.asarray(q_a) - np.asarray(q_b))))
