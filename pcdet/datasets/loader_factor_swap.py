"""TRAIN_FACTOR_SWAP on PandaSet and Lyft (ANALYSIS, training only; absent = unchanged): set ONE generative factor of
the target's TRAINING data to its nuScenes value, for the factor-swap oracles on three more targets (PandaSet Pandar64
spin, Lyft 64-beam, Lyft 40-beam; experiments_md 20261003_05 §5 item 34, the port of items 24 / 26).

The contribution of a factor is AP(target oracle) - AP(factor-swap oracle), both scored on the target's UNPERTURBED val;
the key is read only in training mode and the configs set it on the training block only (evaluation reads
DATA_CONFIG_TAR). Exactly one arm per config, with the meaning of pcdet/datasets/waymo/waymo_factor_swap.py, whose
dataset-agnostic functions are reused here unchanged (imports only; the KITTI and Waymo paths are not touched):

  PITCH_DEG: d | [d1, d2, ...], PITCH_PROB: [...], PIVOT: [x, y, z]       (PandaSet; Lyft since item 37d)
      Points AND GT box centres rotated together about the lateral (y) axis through PIVOT, d > 0 raising the points
      ahead (+x; a flat ground then rises ahead by d degrees). Boxes stay upright (size, yaw unchanged). One angle per
      frame, drawn with PITCH_PROB from the frame key. PIVOT is in the LOADER frame AFTER SHIFT_COOR (this hook runs
      after the loader has shifted points and boxes). The angle is about +x: on Lyft, whose LIDAR_TOP x points BACKWARD,
      a config raises the ground ahead by d degrees with PITCH_DEG -d (experiments_md 20261003_05 §5 item 37d).
  GT_DIM_SCALE: {Car: f, Pedestrian: f}
      dx, dy, dz of the GT boxes whose MODEL class is named, x f, each box's BOTTOM kept. Points untouched.
  GT_IGNORE_BY_RANGE: {CLASS: <model class>, EDGES: [...], KEEP: [...]}
      A box of CLASS in horizontal-radius bin b stays supervised with probability KEEP[b], else its label becomes
      IGNORE_CLASS_LABEL (-99, a zero-weight CenterHead footprint). Labels reach prepare_data as gt_classes /
      gt_scores computed from MODEL names (raw 'car' / 'Pickup Truck' are not model classes).
  GT_IGNORE_RANDOM: {CLASS: <model class>, KEEP: k}                       (R7r, item 42c)
      GT_IGNORE_BY_RANGE's random twin: a box of CLASS stays supervised with probability KEEP whatever its range, else
      -99, by the same route (waymo_factor_swap.random_ignore_labels, one uniform per box from the frame key and arm).
  REGION_THIN: {SIDE: background | foreground, KEEP: f}, BOX_MARGIN: m
      OSENS-2's region thinning (osens2_utils.perturb) of the training cloud with every box the loader keeps (all
      annotated classes), points and boxes in the same (shifted) frame. KEEP >= 1 is the identity (no draw).
  REFLECTION_DROP: {MODE: threshold | random, E: {...}, T: {...}, Q: {...}}         (PandaSet only; item 35g)
      Applied in the PandaSet reader (pandaset_dataset._get_lidar_points), in ego axes about the fitted sensor origin
      before the axis swap (pcdet/datasets/reflection_drop.py); nothing to do here. Lyft has no intensity (constant 100).

Neither loader stores a per-box point count: prepare_data's zero-point filter counts the points of the cloud it is
handed. So:
  - PITCH_DEG and GT_DIM_SCALE (they move points or redraw boxes but remove no point) hand prepare_data the ORACLE's
    count, measured here with prepare_data's own expression on the unswapped cloud and boxes: the row trains on exactly
    the oracle's box set (the Waymo rows keep the stored count for the same reason). The count costs the same one
    points_in_boxes_cpu call prepare_data would otherwise make.
  - REGION_THIN background keeps every point inside the enlarged boxes, so prepare_data's recount equals the oracle's;
    foreground thinning's recount drops the oracle's empty boxes plus exactly those the arm emptied.
  - GT_IGNORE_BY_RANGE and GT_IGNORE_RANDOM move nothing.
  - REFLECTION_DROP removes points in the reader: prepare_data's recount drops the oracle's empty boxes plus exactly those
    the arm emptied (as foreground thinning).
The loader pops 'num_points_in_gt' after prepare_data (as waymo_dataset does): its length is the pre-filter box count.

Every draw is seeded by the FRAME KEY (and the arm), so a frame is cut the same way in every epoch and at every seed:
PandaSet '<sequence>/<frame:02d>' (its frame_id alone, 0-79, repeats across sequences), Lyft the lidar file stem.
"""
import numpy as np

from .kitti.kitti_factor_swap import ignore_labels, region_thin
from .waymo import waymo_factor_swap as wfs

ARMS = wfs.ARMS                     # ('PITCH_DEG', ..., 'REFLECTION_DROP', 'GT_IGNORE_RANDOM'): all on PandaSet
LOADER_ARMS = {
    'pandaset': ARMS,
    # Lyft had no pitch row in item 34 (its sensor x points backward); item 37d adds one, with the angle's sign set by the
    # config (PITCH_DEG about +x; see the docstring). Item 42c adds GT_IGNORE_RANDOM (R7r).
    'lyft': ('PITCH_DEG', 'GT_DIM_SCALE', 'GT_IGNORE_BY_RANGE', 'REGION_THIN', 'GT_IGNORE_RANDOM'),
}
# Keys that change the training cloud or labels in another way (or the frame the hook assumes); one factor per row.
EXCLUSIVE_DATASET_KEYS = {
    'pandaset': ('RING_PATTERN', 'RING_FOV_CUT', 'USE_PSEUDO_LABEL', 'ROW_EXTRUSION', 'BEAM_DROP', 'BEAM_DISTILL'),
    'lyft': ('RING_PATTERN', 'USE_PSEUDO_LABEL', 'LEVEL_COOR', 'ROW_EXTRUSION', 'BEAM_DROP', 'BEAM_DISTILL'),
}


def arm_of(cfg, loader):
    arm = wfs.arm_of(cfg)
    assert arm in LOADER_ARMS[loader], 'TRAIN_FACTOR_SWAP on %s takes %s, got %s' % (loader, LOADER_ARMS[loader], arm)
    return arm


def check_dataset_cfg(dataset_cfg, class_names, loader):
    """The loader settings a factor-swap row may not combine with (asserted on every training item; cheap)."""
    swap = dataset_cfg['TRAIN_FACTOR_SWAP']
    arm_of(swap, loader)
    assert (dataset_cfg.get('MAX_SWEEPS', 1) or 1) <= 1, 'TRAIN_FACTOR_SWAP acts on one stored scan (MAX_SWEEPS <= 1)'
    both = [k for k in EXCLUSIVE_DATASET_KEYS[loader] if dataset_cfg.get(k, None)]
    assert not both, 'TRAIN_FACTOR_SWAP and %s both set' % both
    assert not any(':' in c for c in class_names), 'TRAIN_FACTOR_SWAP expects plain model class names, not multi-head'
    named = wfs.label_arm_classes(swap)
    missing = [c for c in named if c not in class_names]
    assert not missing, 'TRAIN_FACTOR_SWAP names %s, which are not model classes %s' % (missing, class_names)


def oracle_counts(points, boxes):
    """prepare_data's zero-point count, by its own expression, on the cloud and boxes it would be handed."""
    import torch
    from ..ops.roiaware_pool3d import roiaware_pool3d_utils
    return roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, :3]), torch.from_numpy(boxes[:, :7])).numpy().sum(axis=1)


def apply(input_dict, cfg, class_names, map_dataset_to_model, frame_key, loader):
    """The swap on a loader's input_dict (points and gt_boxes already SHIFT_COOR-ed, gt_names in the dataset's
    vocabulary), before prepare_data. Returns input_dict."""
    arm = arm_of(cfg, loader)
    boxes = input_dict.get('gt_boxes', None)
    if boxes is None:
        return input_dict
    if arm in ('PITCH_DEG', 'GT_DIM_SCALE'):
        input_dict['num_points_in_gt'] = oracle_counts(input_dict['points'], boxes)
        names = wfs.model_names(input_dict['gt_names'], map_dataset_to_model)
        if arm == 'PITCH_DEG':
            input_dict['points'] = wfs.apply_points(input_dict['points'], cfg, frame_key)
        input_dict['gt_boxes'] = wfs.apply_boxes(boxes, names, cfg, frame_key)
    elif arm == 'GT_IGNORE_BY_RANGE':
        names = wfs.model_names(input_dict['gt_names'], map_dataset_to_model)
        input_dict['gt_classes'], input_dict['gt_scores'] = ignore_labels(boxes, names, list(class_names), cfg,
                                                                          frame_key)
    elif arm == 'GT_IGNORE_RANDOM':
        names = wfs.model_names(input_dict['gt_names'], map_dataset_to_model)
        input_dict['gt_classes'], input_dict['gt_scores'] = wfs.random_ignore_labels(names, list(class_names), cfg,
                                                                                     frame_key)
    elif arm == 'REGION_THIN':
        if float(cfg['REGION_THIN']['KEEP']) < 1.0:
            input_dict['points'] = region_thin(input_dict['points'], boxes, frame_key, cfg)
    # REFLECTION_DROP: applied by the PandaSet reader already (see the module docstring)
    return input_dict
