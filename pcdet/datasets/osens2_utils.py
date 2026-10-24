"""EVAL_OSENS2 (ANALYSIS, evaluation only; absent = unchanged): one perturbation of the EVALUATION cloud for the
target-oracle sensitivity analysis OSENS-2 (experiments_md 20261025_03; design fixed in 20261003_05 §5, 2026-10-25
01:34 JST, experiments_md f0bd7d0). Applied by DatasetTemplate.prepare_data to `points` only, at its start, i.e. in
the loader's frame (SHIFT_COOR applied, no augmentation at evaluation) and before the class filter, so the GT boxes
seen here are every box the loader keeps (all annotated classes; Waymo 'unknown' and KITTI 'DontCare' already dropped).
GT boxes and predictions are never moved. Exactly one arm per config:

  PITCH_DEG: d, SENSOR_ORIGIN: [x, y, z], FORWARD_AXIS: x | y
      A, mounting (frame part), pitch: rotate x, y, z about the lateral axis through SENSOR_ORIGIN (the sensor in the
      loader's frame: SHIFT_COOR plus the sensor's raw-frame position). FORWARD_AXIS is the vehicle's forward axis in
      that frame (x for Waymo's vehicle frame and KITTI's velodyne, y for nuScenes' LIDAR_TOP, whose x points right).
      d > 0 raises the points ahead: a flat ground then RISES ahead by d degrees, the sign of 20261004_02's forward
      ground slope and of LEVEL_COOR.PITCH_DEG.
  REGION_THIN: {SIDE: background | foreground, KEEP: f}, BOX_MARGIN: m
      C, background against foreground: keep round(f * n) of the n points OUTSIDE every GT box enlarged by m
      (background) or INSIDE any of them (foreground), drawn without replacement, seeded by the frame id; the other
      set is untouched. The enlargement is pcdet's `box_utils.enlarge_box3d(extra_width=(m, m, m))`: dx, dy and dz
      each grow by m. Containment is pcdet's `points_in_boxes_cpu`.
  GROUND_REMOVE_DIST: t, BOX_MARGIN: m
      C, ground: among the points outside every enlarged box, drop those within t (vertical residual) of the frame's
      ground plane z = a x + b y + c, fitted on those same outside points by ground_tilt.py's estimator (second-lowest
      return per 1 m BEV cell at 4-40 m horizontal range, RANSAC, least-squares refit on the inliers), copied below.
      A frame whose fit fails (fewer than 30 cells) is left whole and counted.

The height arm of A needs no code (a twin with SHIFT_COOR z offset: points and GT move together, predictions are
un-shifted by generate_prediction_dicts), nor does B's ego-points arm (nuScenes EVAL_KEEP_EGO_POINTS).
"""
import zlib

import numpy as np

ARMS = ('PITCH_DEG', 'REGION_THIN', 'GROUND_REMOVE_DIST')
BOX_CHUNK = 64   # boxes per points_in_boxes_cpu call: bounds the (boxes x points) mask's memory


def frame_rng(frame_id):
    """The arm's generator for one frame: seeded by the frame id, so two scorings see the same cloud."""
    return np.random.default_rng(zlib.crc32(str(frame_id).encode()))


def pitch_points(points, deg, origin, forward_axis):
    """Points rotated about the lateral axis through `origin`; d > 0 raises the points ahead (forward_axis 'x' / 'y')."""
    assert forward_axis in ('x', 'y'), forward_axis
    out = np.array(points, copy=True)
    f = 0 if forward_axis == 'x' else 1
    th = np.radians(float(deg))
    c, s = np.cos(th), np.sin(th)
    o = np.asarray(origin, dtype=np.float64)
    df = points[:, f].astype(np.float64) - o[f]
    dz = points[:, 2].astype(np.float64) - o[2]
    out[:, f] = (o[f] + c * df - s * dz).astype(points.dtype)
    out[:, 2] = (o[2] + s * df + c * dz).astype(points.dtype)
    return out


def enlarge_boxes(boxes, margin):
    """(n, 7) copy with dx, dy, dz each grown by `margin` (box_utils.enlarge_box3d's convention)."""
    b = np.array(boxes[:, :7], dtype=np.float32, copy=True)
    b[:, 3:6] += np.float32(margin)
    return b


def in_any_box(xyz, boxes):
    """Boolean per point: inside any of the (n, 7) boxes, by pcdet's points_in_boxes_cpu."""
    inside = np.zeros(len(xyz), dtype=bool)
    if len(xyz) == 0 or len(boxes) == 0:
        return inside
    import torch
    from ..ops.roiaware_pool3d import roiaware_pool3d_utils
    pts = torch.from_numpy(np.ascontiguousarray(xyz[:, :3], dtype=np.float32))
    for k in range(0, len(boxes), BOX_CHUNK):
        b = torch.from_numpy(np.ascontiguousarray(boxes[k:k + BOX_CHUNK, :7], dtype=np.float32))
        inside |= roiaware_pool3d_utils.points_in_boxes_cpu(pts, b).numpy().any(axis=0)
    return inside


# ---- ground_tilt.py's estimator (ST3D tools/analysis/ground_tilt.py at 448b4aa), verbatim but for the names ----------
def lowest_per_cell(pts, rmin=4.0, rmax=40.0, cell=1.0, min_pts=3):
    r = np.hypot(pts[:, 0], pts[:, 1])
    p = pts[(r >= rmin) & (r <= rmax)]
    if len(p) == 0:
        return np.zeros((0, 3))
    ij = np.floor(p[:, :2] / cell).astype(np.int64)
    key = (ij[:, 0] + 10000) * 100000 + (ij[:, 1] + 10000)
    order = np.lexsort((p[:, 2], key))
    key, p = key[order], p[order]
    first = np.r_[True, key[1:] != key[:-1]]
    starts = np.flatnonzero(first)
    counts = np.diff(np.r_[starts, len(key)])
    # second-lowest return per cell (robust to a single low outlier)
    take = starts + np.minimum(1, counts - 1)
    keep = counts >= min_pts
    return p[take[keep], :3]


def fit_plane(g, iters=300, thr=0.12, rng=None):
    if len(g) < 30:
        return None
    rng = rng or np.random.default_rng(0)
    A = np.c_[g[:, 0], g[:, 1], np.ones(len(g))]
    best, best_n = None, -1
    for _ in range(iters):
        idx = rng.choice(len(g), 3, replace=False)
        try:
            coef = np.linalg.solve(A[idx], g[idx, 2])
        except np.linalg.LinAlgError:
            continue
        if abs(coef[0]) > 0.2 or abs(coef[1]) > 0.2:  # > 11 deg: not a road
            continue
        n = np.sum(np.abs(A @ coef - g[:, 2]) < thr)
        if n > best_n:
            best, best_n = coef, n
    if best is None:
        return None
    for t in (thr, 0.08):
        inl = np.abs(A @ best - g[:, 2]) < t
        best = np.linalg.lstsq(A[inl], g[inl, 2], rcond=None)[0]
    return best, int(inl.sum()), len(g)
# -----------------------------------------------------------------------------------------------------------------------


def _check(cfg):
    set_arms = [k for k in ARMS if cfg.get(k, None) is not None]
    assert len(set_arms) == 1, 'EVAL_OSENS2 takes exactly one of %s, got %s' % (ARMS, set_arms)
    return set_arms[0]


def perturb(points, gt_boxes, frame_id, cfg):
    """(perturbed points, stats) for one evaluation frame. `gt_boxes` (n, >= 7) in the frame of `points`."""
    arm = _check(cfg)
    stats = {'arm': arm, 'n_in': int(len(points))}
    if arm == 'PITCH_DEG':
        out = pitch_points(points, cfg['PITCH_DEG'], cfg['SENSOR_ORIGIN'], cfg['FORWARD_AXIS'])
        stats['n_out'] = int(len(out))
        return out, stats
    margin = float(cfg['BOX_MARGIN'])
    boxes = np.zeros((0, 7), np.float32) if gt_boxes is None else enlarge_boxes(np.asarray(gt_boxes), margin)
    inside = in_any_box(points, boxes)
    stats.update(n_inside=int(inside.sum()), n_outside=int((~inside).sum()), n_boxes=int(len(boxes)))
    if arm == 'REGION_THIN':
        side, keep = cfg['REGION_THIN']['SIDE'], float(cfg['REGION_THIN']['KEEP'])
        assert side in ('background', 'foreground'), side
        assert 0.0 < keep < 1.0, keep
        pool = np.flatnonzero(~inside if side == 'background' else inside)
        n_keep = int(round(len(pool) * keep))
        kept = np.sort(frame_rng(frame_id).choice(pool, n_keep, replace=False)) if len(pool) else pool
        mask = inside.copy() if side == 'background' else ~inside
        mask[kept] = True
        out = points[mask]
        stats.update(n_pool=int(len(pool)), n_pool_kept=int(n_keep))
    else:
        dist = float(cfg['GROUND_REMOVE_DIST'])
        outside = np.flatnonzero(~inside)
        fit = fit_plane(lowest_per_cell(np.asarray(points[outside, :3], dtype=np.float64)))
        drop = np.zeros(len(points), dtype=bool)
        if fit is None:
            stats['fit_failed'] = 1
        else:
            a, b, c = fit[0]
            res = np.asarray(points[outside, 2], np.float64) - (a * points[outside, 0] + b * points[outside, 1] + c)
            drop[outside[np.abs(res) < dist]] = True
            stats.update(fit_failed=0, plane=[float(a), float(b), float(c)], fit_inliers=fit[1], fit_cells=fit[2])
        out = points[~drop]
        stats['n_dropped'] = int(drop.sum())
    stats['n_out'] = int(len(out))
    return out, stats


def apply_in_prepare_data(data_dict, cfg):
    """The hook DatasetTemplate.prepare_data calls at evaluation: replaces data_dict['points'] only."""
    data_dict['points'], _ = perturb(data_dict['points'], data_dict.get('gt_boxes', None), data_dict['frame_id'], cfg)
    return data_dict
