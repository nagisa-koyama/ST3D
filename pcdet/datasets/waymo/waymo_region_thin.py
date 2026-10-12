"""ANALYSIS: EVAL_TOP_THIN restricted to a REGION - the TOP block of an evaluation frame thinned only INSIDE, or only
OUTSIDE, a set of boxes given per frame (experiments_md 20261010_04 §5.0a). Asks whether the scan-line cue a Waymo
model uses sits on objects or on the background.

The boxes come from a file keyed by frame id ({frame_id: {'boxes': (M, 7+), 'labels': (M,)}}), written once by
tools/analysis/make_region_boxes.py from a legal model's predictions above a fixed score - no target label is read.
Boxes whose label is not positive (pseudo-label ignore -1..-C, IGNORE_CLASS_LABEL -99, background 0) never define a
region. `waymo_dataset.get_lidar_top_thin` calls this only when EVAL_TOP_THIN.REGION is set; absent = unchanged.
"""
import numpy as np

from .waymo_rings import top_beam_ids


def region_boxes(entry):
    """(M, 7) boxes [x, y, z, dx, dy, dz, heading] usable as regions from one box-file entry (None = no boxes)."""
    if entry is None:
        return np.zeros((0, 7))
    boxes = entry.get('boxes', None) if isinstance(entry, dict) else entry
    boxes = np.zeros((0, 7)) if boxes is None else np.asarray(boxes, dtype=np.float64)
    boxes = boxes.reshape(len(boxes), -1)[:, :7] if boxes.size else np.zeros((0, 7))
    labels = entry.get('labels', None) if isinstance(entry, dict) else None
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        assert len(labels) == len(boxes), '%d labels for %d boxes' % (len(labels), len(boxes))
        boxes = boxes[labels > 0]
    return boxes


def enlarge_boxes(boxes, margin_xy, margin_z):
    """Each box grown by `margin_xy` on every side in length and width and `margin_z` at top and bottom; centre and
    heading unchanged."""
    b = np.array(boxes, dtype=np.float64).reshape(-1, 7)
    b[:, 3:5] += 2.0 * margin_xy
    b[:, 5] += 2.0 * margin_z
    return b


def points_in_any_box(xyz, boxes):
    """(N,) bool: the point lies inside at least one box, bounds INCLUSIVE. Boxes (M, 7) with z at the box centre."""
    xyz = np.asarray(xyz, dtype=np.float64)
    inside = np.zeros(len(xyz), dtype=bool)
    for b in boxes:
        d = xyz[:, :3] - b[:3]
        c, s = np.cos(-b[6]), np.sin(-b[6])
        lx = d[:, 0] * c - d[:, 1] * s
        ly = d[:, 0] * s + d[:, 1] * c
        inside |= (np.abs(lx) <= b[3] / 2) & (np.abs(ly) <= b[4] / 2) & (np.abs(d[:, 2]) <= b[5] / 2)
    return inside


def region_top_thin_points(raw, num_points_of_each_lidar, calib, mode, boxes, side, margin_xy=0.5, margin_z=0.15,
                           stride=2, rng=None):
    """One stored frame with its TOP block thinned only in one region, returned as `get_lidar` returns it.

    `side` 'inside': TOP points inside any (enlarged) box are thinned, every other point is kept; 'outside': the
    reverse. `mode` 'rows' keeps the region's TOP points on beams whose index (0 = highest) is a multiple of `stride`
    (whole scan lines go, as eval_top_thin_points' 'rows'); 'random' keeps the SAME NUMBER of the region's TOP points,
    drawn uniformly (count-matched per frame). Side lidars are untouched. Rows are recovered on the full TOP block in
    stored order, before the no-label-zone filter; the filter and tanh(intensity) follow `get_lidar`.
    """
    if side not in ('inside', 'outside'):
        raise ValueError("REGION.SIDE must be 'inside' or 'outside', got %r" % (side,))
    if mode not in ('rows', 'random'):
        raise ValueError("MODE must be 'rows' or 'random' with REGION, got %r" % (mode,))
    counts = np.asarray(num_points_of_each_lidar).astype(np.int64)
    assert counts.sum() == len(raw), 'per-lidar counts %s do not add up to the %d stored points' % (counts, len(raw))
    n_top = int(counts[0])
    keep = np.ones(len(raw), dtype=bool)
    if n_top:
        beam = top_beam_ids(raw[:n_top, :3], calib['extrinsic'], calib['inclinations'])[0]
        inside = points_in_any_box(raw[:n_top, :3], enlarge_boxes(region_boxes(boxes), margin_xy, margin_z))
        region = inside if side == 'inside' else ~inside
        on_kept_rows = (beam % stride) == 0
        if mode == 'rows':
            keep[:n_top] = ~region | on_kept_rows
        else:
            rng = np.random.default_rng(0) if rng is None else rng
            idx = np.nonzero(region)[0]
            sel = ~region
            if len(idx):
                sel[rng.choice(idx, int((region & on_kept_rows).sum()), replace=False)] = True
            keep[:n_top] = sel
    pts = raw[keep]
    pts = pts[pts[:, 5] == -1]
    out = np.array(pts[:, 0:5], dtype=raw.dtype)
    out[:, 3] = np.tanh(out[:, 3])
    return out
