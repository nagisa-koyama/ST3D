"""TRAIN_FACTOR_SWAP (ANALYSIS, training only; absent = unchanged): set ONE generative factor of the KITTI TRAINING data
to its nuScenes value, for the factor-swap oracles of experiments_md 20261003_05 §5 item 24 (FACTOR-SWAP S2).

The contribution of a factor is AP(KITTI oracle) - AP(factor-swap oracle), both scored on UNPERTURBED KITTI val; the
key is read only when the dataset is in training mode, and the configs set it on DATA_CONFIG only. Every arm is
seeded by the frame (and the arm), so a frame is cut the same way in every epoch: the swapped training set is a fixed
dataset, like a recorded one, not a fresh draw per epoch. Exactly one arm per config:

  RING_THIN: {MODE: rows | random | azbin, STRIDE: s, AZ_RES_DEG: a}
      Point arm, on the raw scan as read (kitti_rings.eval_ring_thin_mask; rings from the stored order). rows = every
      s-th laser ring; random = the same count drawn uniformly per frame; azbin = one point per ring per a-degree bin.
  RANGE_KEEP: {EDGES: [...], RATE: [...]}
      Point arm: keep a point with probability RATE[b] of its horizontal-radius bin b (EDGES in m, len(RATE) + 1;
      radius beyond the last edge uses the last rate). The rates come from per-frame radial histograms of KITTI TRAIN
      and nuScenes TRAIN measured by tools/analysis/factorswap_s2_stats.py.
  PITCH_DEG: d | [d1, d2, ...], PITCH_PROB: [p1, p2, ...]
      Point AND box arm: rotate the points and the GT box centres about the lateral (y) axis through the sensor (the
      raw velodyne origin, x forward), d > 0 raising the points ahead (osens2_utils.pitch_points' sign: a flat ground
      then rises ahead by d degrees). Boxes stay upright; their yaw is unchanged. A list draws one angle per frame
      with PITCH_PROB (a pooled two-city source is a mixture of two mounting slopes); points and boxes of a frame
      draw the same angle.
  GT_DIM_SCALE: {Car: f, Pedestrian: f, ...}
      Label arm: multiply dx, dy, dz of the named classes' GT boxes by f, keeping each box's BOTTOM where it is (the
      centre rises by (f - 1) dz / 2). Points are untouched: the factor is the box definition on the same object.
  GT_IGNORE_BY_RANGE: {CLASS: name, EDGES: [...], KEEP: [...]}
      Label arm: a box of CLASS in horizontal-radius bin b stays supervised with probability KEEP[b]; otherwise its
      label becomes IGNORE_CLASS_LABEL (-99), which CenterHead turns into a zero-weight footprint (never a positive,
      never background). Other classes are untouched.
  REGION_THIN: {SIDE: foreground | background, KEEP: f}, BOX_MARGIN: m
      Point arm with boxes: OSENS-2's region thinning (osens2_utils.perturb) applied to the training cloud, with every
      box the loader keeps (all KITTI classes but DontCare).
"""
import zlib

import numpy as np

from ...utils import common_utils

POINT_ARMS = ('RING_THIN', 'RANGE_KEEP')
ARMS = POINT_ARMS + ('PITCH_DEG', 'GT_DIM_SCALE', 'GT_IGNORE_BY_RANGE', 'REGION_THIN')


def arm_of(cfg):
    set_arms = [k for k in ARMS if cfg.get(k, None) is not None]
    assert len(set_arms) == 1, 'TRAIN_FACTOR_SWAP takes exactly one of %s, got %s' % (ARMS, set_arms)
    return set_arms[0]


def frame_rng(frame_id, arm):
    return np.random.default_rng(zlib.crc32(('factorswap/%s/%s' % (frame_id, arm)).encode()))


def radius_bin(xy, edges):
    """Bin index per row of `xy` (n, 2) by horizontal radius; beyond the last edge -> the last bin."""
    edges = np.asarray(edges, dtype=np.float64)
    r = np.hypot(xy[:, 0], xy[:, 1])
    return np.clip(np.searchsorted(edges, r, side='right') - 1, 0, len(edges) - 2)


def pitch_xyz(xyz, deg):
    """(n, 3) rotated about the y axis through the origin; deg > 0 raises the points ahead (+x)."""
    th = np.radians(float(deg))
    c, s = np.cos(th), np.sin(th)
    out = np.array(xyz, copy=True)
    x, z = xyz[:, 0].astype(np.float64), xyz[:, 2].astype(np.float64)
    out[:, 0] = (c * x - s * z).astype(xyz.dtype)
    out[:, 2] = (s * x + c * z).astype(xyz.dtype)
    return out


def pitch_of(cfg, frame_id):
    """The frame's pitch angle: the scalar, or one draw from the PITCH_DEG list with PITCH_PROB."""
    deg = cfg['PITCH_DEG']
    if np.isscalar(deg):
        return float(deg)
    prob = np.asarray(cfg['PITCH_PROB'], dtype=np.float64)
    assert len(prob) == len(deg) and abs(prob.sum() - 1.0) < 1e-6, 'PITCH_PROB must match PITCH_DEG and sum to 1'
    return float(deg[int(frame_rng(frame_id, 'PITCH_DEG').choice(len(deg), p=prob))])


def apply_points(points, cfg, frame_id):
    """The point arms and the point half of PITCH_DEG, on the raw scan before any crop or shift."""
    arm = arm_of(cfg)
    if arm == 'RING_THIN':
        from .kitti_rings import eval_ring_thin_mask
        return points[eval_ring_thin_mask(points, cfg['RING_THIN'], frame_rng(frame_id, arm))]
    if arm == 'RANGE_KEEP':
        rate = np.asarray(cfg['RANGE_KEEP']['RATE'], dtype=np.float64)
        assert len(cfg['RANGE_KEEP']['EDGES']) == len(rate) + 1, 'RANGE_KEEP: len(EDGES) must be len(RATE) + 1'
        b = radius_bin(points[:, :2], cfg['RANGE_KEEP']['EDGES'])
        return points[frame_rng(frame_id, arm).random(len(points)) < rate[b]]
    if arm == 'PITCH_DEG':
        out = np.array(points, copy=True)
        out[:, :3] = pitch_xyz(points[:, :3], pitch_of(cfg, frame_id))
        return out
    return points


def apply_boxes(boxes, names, cfg, frame_id):
    """The box half of PITCH_DEG and GT_DIM_SCALE, on (n, 7) lidar boxes in the raw frame (before SHIFT_COOR)."""
    arm = arm_of(cfg)
    if arm == 'PITCH_DEG' and len(boxes):
        boxes = np.array(boxes, copy=True)
        boxes[:, :3] = pitch_xyz(boxes[:, :3], pitch_of(cfg, frame_id))
    elif arm == 'GT_DIM_SCALE' and len(boxes):
        boxes = np.array(boxes, copy=True)
        for name, f in cfg['GT_DIM_SCALE'].items():
            m = np.asarray(names) == name
            f = np.float32(f)
            boxes[m, 2] += (f - 1.0) * boxes[m, 5] / 2.0
            boxes[m, 3:6] *= f
    return boxes


def region_thin(points, boxes, frame_id, cfg):
    """REGION_THIN on the training cloud: points and boxes in the same frame (both after SHIFT_COOR)."""
    from ..osens2_utils import perturb
    sub = {'REGION_THIN': cfg['REGION_THIN'], 'BOX_MARGIN': cfg['BOX_MARGIN']}
    return perturb(points, boxes, frame_id, sub)[0]


def ignore_labels(boxes, names, class_names, cfg, frame_id):
    """(gt_classes, gt_scores) for prepare_data's pseudo-label route: model class index + 1, IGNORE_CLASS_LABEL for
    the boxes GT_IGNORE_BY_RANGE drops from supervision (and as a placeholder for names outside class_names, which
    prepare_data's class filter removes)."""
    sub = cfg['GT_IGNORE_BY_RANGE']
    keep = np.asarray(sub['KEEP'], dtype=np.float64)
    assert len(sub['EDGES']) == len(keep) + 1, 'GT_IGNORE_BY_RANGE: len(EDGES) must be len(KEEP) + 1'
    names = np.asarray(names)
    classes = np.array([class_names.index(n) + 1 if n in class_names else common_utils.IGNORE_CLASS_LABEL
                        for n in names], dtype=np.int32)
    target = names == sub['CLASS']
    if target.any():
        b = radius_bin(np.asarray(boxes)[:, :2], sub['EDGES'])
        u = frame_rng(frame_id, 'GT_IGNORE_BY_RANGE').random(len(names))
        classes[target & (u >= keep[b])] = common_utils.IGNORE_CLASS_LABEL
    return classes, np.ones(len(names), dtype=np.float32)
