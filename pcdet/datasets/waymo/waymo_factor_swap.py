"""TRAIN_FACTOR_SWAP on Waymo (ANALYSIS, training only; absent = unchanged): set ONE generative factor of the Waymo
TRAINING data to its nuScenes value, for the factor-swap oracles of the nuScenes -> Waymo direction (FACTOR-SWAP N->W;
the Waymo port of pcdet/datasets/kitti/kitti_factor_swap.py, experiments_md 20261003_05 §5 item 24).

The contribution of a factor is AP(Waymo oracle) - AP(factor-swap oracle), both scored on UNPERTURBED Waymo val; the key
is read only when the dataset is in training mode, and the configs set it on DATA_CONFIG only (evaluation reads
DATA_CONFIG_TAR). Every draw is seeded by the frame id (and the arm), so a frame is cut the same way in every epoch: the
swapped training set is a fixed dataset, like a recorded one. Exactly one arm per config:

  PITCH_DEG: d | [d1, d2, ...], PITCH_PROB: [p1, p2, ...], PIVOT: [x, y, z]
      Point AND box arm: rotate the points and the GT box centres about the lateral (y) axis through PIVOT, the Waymo
      TOP lidar's position in the vehicle frame (x forward, z up; from the TOP extrinsic), d > 0 raising the points
      ahead (kitti_factor_swap.pitch_xyz's sign: a flat ground then rises ahead by d degrees). Boxes stay upright;
      their yaw and size are unchanged. A list draws one angle per frame with PITCH_PROB; points and boxes of a frame
      draw the same angle. Applied on the stored cloud and boxes, before SHIFT_COOR.
  GT_DIM_SCALE: {Car: f, Pedestrian: f, ...}
      Label arm: multiply dx, dy, dz of the GT boxes whose MODEL class (the Waymo name through the dataset -> model
      ontology map, e.g. Vehicle -> Car) is named by f, keeping each box's BOTTOM where it is. Points are untouched.
  GT_IGNORE_BY_RANGE: {CLASS: <model class>, EDGES: [...], KEEP: [...]}
      Label arm: a box of CLASS (model name) in horizontal-radius bin b stays supervised with probability KEEP[b];
      otherwise its label becomes IGNORE_CLASS_LABEL (-99), which CenterHead turns into a zero-weight footprint. Labels
      reach prepare_data as gt_classes / gt_scores computed from the MODEL names (kitti_factor_swap.ignore_labels
      indexes class_names by name: Waymo's raw 'Vehicle' is not a model class, so raw names would make every Car -99).
  REGION_THIN: {SIDE: foreground | background, KEEP: f}, BOX_MARGIN: m
      Point arm with boxes: OSENS-2's region thinning (osens2_utils.perturb) on the training cloud, with every box the
      loader keeps (all Waymo classes but 'unknown'), after SHIFT_COOR, points and boxes in the same frame.

Waymo's stored num_points_in_gt (all returns, all five lidars) drives prepare_data's zero-point filter. An arm that can
EMPTY a box gets a differential recount (recount_emptied): a box is dropped exactly when the oracle drops it (stored
count < 1) or when the arm emptied it (>= 1 point of the loaded cloud inside it before the arm, none after). A plain
recount (None) would also drop the boxes that are empty in the first-return cloud already and that the oracle trains
on, a second change against the oracle. REGION_THIN gets it; PITCH_DEG (rigid: points move with their box) and
GT_DIM_SCALE (the box definition, on the objects the oracle trains on) keep the stored count (the target dump measures
how many boxes either would empty).
"""
import numpy as np

from ..kitti.kitti_factor_swap import frame_rng, ignore_labels, pitch_of, pitch_xyz, radius_bin, region_thin  # noqa: F401

ARMS = ('PITCH_DEG', 'GT_DIM_SCALE', 'GT_IGNORE_BY_RANGE', 'REGION_THIN')
ALLOWED_KEYS = ARMS + ('PITCH_PROB', 'PIVOT', 'BOX_MARGIN')
# Keys that change the Waymo training cloud or labels in another way; a factor-swap row changes one factor only.
EXCLUSIVE_DATASET_KEYS = ('RING_PATTERN', 'TRAIN_TOP_THIN', 'LIT_TRANSLATED', 'BEAM_DROP', 'BEAM_DISTILL',
                          'USE_PSEUDO_LABEL')


def arm_of(cfg):
    unknown = [k for k in cfg.keys() if k not in ALLOWED_KEYS]
    assert not unknown, 'TRAIN_FACTOR_SWAP on Waymo takes only %s, got %s' % (ALLOWED_KEYS, unknown)
    set_arms = [k for k in ARMS if cfg.get(k, None) is not None]
    assert len(set_arms) == 1, 'TRAIN_FACTOR_SWAP takes exactly one of %s, got %s' % (ARMS, set_arms)
    arm = set_arms[0]
    if arm == 'PITCH_DEG':
        assert cfg.get('PIVOT', None) is not None and len(cfg['PIVOT']) == 3, 'PITCH_DEG on Waymo needs PIVOT [x, y, z]'
    if arm == 'REGION_THIN':
        assert cfg.get('BOX_MARGIN', None) is not None, 'REGION_THIN needs BOX_MARGIN'
    return arm


def check_dataset_cfg(dataset_cfg, class_names):
    """The loader settings a factor-swap row may not combine with (asserted on the first training item)."""
    swap = dataset_cfg['TRAIN_FACTOR_SWAP']
    arm_of(swap)
    assert (dataset_cfg.get('MAX_SWEEPS', 1) or 1) <= 1, 'TRAIN_FACTOR_SWAP acts on one stored scan (MAX_SWEEPS <= 1)'
    both = [k for k in EXCLUSIVE_DATASET_KEYS if dataset_cfg.get(k, None)]
    assert not both, 'TRAIN_FACTOR_SWAP and %s both set' % both
    # __getitem__'s FOV crop filters gt_names / gt_boxes / num_points_in_gt but not the ignore arm's gt_classes.
    assert not dataset_cfg.get('FOV_POINTS_ONLY', False), 'TRAIN_FACTOR_SWAP is for the 360-degree Waymo cloud'
    assert not any(':' in c for c in class_names), 'TRAIN_FACTOR_SWAP expects plain model class names, not multi-head'
    named = list(swap.get('GT_DIM_SCALE', None) or {}) + \
        ([swap['GT_IGNORE_BY_RANGE']['CLASS']] if swap.get('GT_IGNORE_BY_RANGE', None) is not None else [])
    missing = [c for c in named if c not in class_names]
    assert not missing, 'TRAIN_FACTOR_SWAP names %s, which are not model classes %s' % (missing, class_names)


def model_names(names, map_dataset_to_model):
    """Dataset names (Waymo: Vehicle, Pedestrian, Cyclist, Sign) in the MODEL vocabulary, as prepare_data maps them."""
    names = np.asarray(names)
    if map_dataset_to_model is None:
        return names
    return np.array([map_dataset_to_model[n] for n in names]) if len(names) else names


def pitch_about(xyz, deg, pivot):
    """(n, 3) rotated about the lateral axis through `pivot`; deg > 0 raises the points ahead (+x)."""
    p = np.asarray(pivot, dtype=np.float64)
    out = pitch_xyz(np.asarray(xyz, dtype=np.float64) - p, deg) + p
    return out.astype(np.asarray(xyz).dtype)


def apply_points(points, cfg, frame_id):
    """The point half of PITCH_DEG, on the stored cloud before SHIFT_COOR. Every other arm leaves points here."""
    if arm_of(cfg) != 'PITCH_DEG':
        return points
    out = np.array(points, copy=True)
    out[:, :3] = pitch_about(points[:, :3], pitch_of(cfg, frame_id), cfg['PIVOT'])
    return out


def apply_boxes(boxes, names_model, cfg, frame_id):
    """The box half of PITCH_DEG and GT_DIM_SCALE on (n, >= 7) boxes before SHIFT_COOR; only centres / dims change."""
    arm = arm_of(cfg)
    if arm == 'PITCH_DEG' and len(boxes):
        boxes = np.array(boxes, copy=True)
        boxes[:, :3] = pitch_about(boxes[:, :3], pitch_of(cfg, frame_id), cfg['PIVOT'])
    elif arm == 'GT_DIM_SCALE' and len(boxes):
        boxes = np.array(boxes, copy=True)
        names_model = np.asarray(names_model)
        for name, f in cfg['GT_DIM_SCALE'].items():
            m = names_model == name
            f = np.float32(f)
            boxes[m, 2] += (f - 1.0) * boxes[m, 5] / 2.0
            boxes[m, 3:6] *= f
    return boxes


def points_per_box(points, boxes):
    """Points of `points` inside each (n, >= 7) box, by pcdet's points_in_boxes_cpu (degenerate boxes count 0)."""
    counts = np.zeros(len(boxes), dtype=np.int64)
    if len(boxes) == 0 or len(points) == 0:
        return counts
    import torch
    from ...ops.roiaware_pool3d import roiaware_pool3d_utils
    valid = (np.asarray(boxes)[:, 3:6] > 1e-3).all(axis=1)
    if valid.any():
        pts = torch.from_numpy(np.ascontiguousarray(points[:, :3], dtype=np.float32))
        b = torch.from_numpy(np.ascontiguousarray(np.asarray(boxes)[valid, :7], dtype=np.float32))
        counts[valid] = roiaware_pool3d_utils.points_in_boxes_cpu(pts, b).numpy().sum(axis=1)
    return counts


def recount_emptied(stored, before, after):
    """The zero-point filter's count after an arm that removes points: the stored count, except 0 for a box the arm
    emptied (>= 1 point before, none after). With no stored count, the count after the arm."""
    after = np.asarray(after, dtype=np.int64)
    if stored is None:
        return after
    out = np.array(stored, dtype=np.int64, copy=True)
    out[(np.asarray(before) > 0) & (after == 0)] = 0
    return out


def finish(input_dict, cfg, class_names, map_dataset_to_model, frame_id):
    """The arms that need the frame's points and boxes together (after SHIFT_COOR), on __getitem__'s input_dict:
    REGION_THIN (with the differential recount) and GT_IGNORE_BY_RANGE (gt_classes / gt_scores from model names)."""
    arm = arm_of(cfg)
    boxes = input_dict.get('gt_boxes', None)
    if boxes is None:
        return input_dict
    if arm == 'REGION_THIN':
        before = points_per_box(input_dict['points'], boxes)
        input_dict['points'] = region_thin(input_dict['points'], boxes, frame_id, cfg)
        after = points_per_box(input_dict['points'], boxes)
        input_dict['num_points_in_gt'] = recount_emptied(input_dict.get('num_points_in_gt', None), before, after)
    elif arm == 'GT_IGNORE_BY_RANGE':
        names = model_names(input_dict['gt_names'], map_dataset_to_model)
        input_dict['gt_classes'], input_dict['gt_scores'] = ignore_labels(boxes, names, list(class_names), cfg,
                                                                          frame_id)
    return input_dict
