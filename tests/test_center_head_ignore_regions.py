"""CenterHead ignore regions: negative box labels are never positives, and their footprint is out of the loss.

Two kinds of box carry a negative label in gt_boxes[:, 7]:
- the pseudo-label IGNORE BAND (NEG_THRESH <= score < SCORE_THRESH), negated by self_training_utils
  to -1..-C so its class is still recoverable with abs();
- IGNORE_CLASS_LABEL, a SOURCE box of a class outside CLASS_NAMES kept by
  DATA_CONFIG.IGNORE_OTHER_CLASSES (e.g. nuScenes truck / bus under the KITTI ontology), which used to be
  dropped and so was learned as background.

Until 2026-10-06 assign_targets indexed `np.array(['bg', *class_names])` with the label directly. numpy
wraps negative indices, so -1 (an ignored Car) trained as a Cyclist POSITIVE, -2 as Pedestrian and -3
(an ignored Cyclist) as Car - in every CenterPoint pseudo-labelling row whose SCORE_THRESH exceeded its
NEG_THRESH (experiments_md/20261004_01 §5f). The first test is that regression.

CenterHead.__init__ calls .cuda(), so the head is built without __init__ and given only what
assign_targets / get_loss read. Runs on the CPU-only master node.
"""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.models.dense_heads.center_head import CenterHead  # noqa: E402
from pcdet.utils.common_utils import IGNORE_CLASS_LABEL  # noqa: E402

NAMES = ['Car', 'Pedestrian', 'Cyclist']
RANGE = [-10.0, -10.0, -2.0, 10.0, 10.0, 4.0]
VOXEL = [0.1, 0.1, 0.15]
STRIDE = 4
CELL = VOXEL[0] * STRIDE          # 0.4 m per heatmap pixel
FMAP = [50, 50]                   # [H, W] = 20 m / 0.4 m
DIMS = {1: (4.0, 1.8, 1.6), 2: (0.8, 0.8, 1.7), 3: (1.8, 0.6, 1.7)}


def _head():
    head = CenterHead.__new__(CenterHead)
    torch.nn.Module.__init__(head)
    head.model_cfg = EasyDict({
        'TARGET_ASSIGNER_CONFIG': {'FEATURE_MAP_STRIDE': STRIDE, 'NUM_MAX_OBJS': 50,
                                   'GAUSSIAN_OVERLAP': 0.1, 'MIN_RADIUS': 2},
        'LOSS_CONFIG': {'LOSS_WEIGHTS': {'cls_weight': 1.0, 'loc_weight': 2.0,
                                         'code_weights': [1.0] * 8}},
    })
    head.class_names = list(NAMES)
    head.class_names_each_head = [list(NAMES)]
    head.point_cloud_range = np.array(RANGE, dtype=np.float32)
    head.voxel_size = VOXEL
    head.separate_head_cfg = EasyDict({'HEAD_ORDER': ['center', 'center_z', 'dim', 'rot']})
    head.build_losses()
    return head


def _box(x, y, label, dims=None, heading=0.0):
    d = dims if dims is not None else DIMS[abs(label)] if abs(label) in DIMS else (8.0, 2.5, 3.0)
    return [x, y, 0.0, *d, heading, float(label)]


def _pix(x, y):
    """(row, col) of the heatmap pixel holding metric point (x, y)."""
    return int((y - RANGE[1]) / CELL), int((x - RANGE[0]) / CELL)


def _targets(rows):
    gt = torch.tensor([rows + [[0.0] * 8]], dtype=torch.float32)   # one zero row, as collate pads
    return _head().assign_targets(gt, feature_map_size=FMAP)


POSITIVES = [(-6, -6, 1), (-6, 0, 2), (-6, 6, 3)]
IGNORED = [(6, -6, -1), (6, 0, -2), (6, 6, -3), (0, -6, IGNORE_CLASS_LABEL)]


def test_negative_labels_are_never_positives():
    """The 2026-10-06 regression: before the fix this found 7 peaks, the ignored ones in permuted classes."""
    t = _targets([_box(*p) for p in POSITIVES + IGNORED])
    hm = t['heatmaps'][0][0]
    assert int(hm.eq(1).sum()) == 3, 'only the three positive-label boxes may produce a positive peak'
    for x, y, label in POSITIVES:
        r, c = _pix(x, y)
        assert hm[label - 1, r, c] == 1, f'label {label} must peak in its own channel'
    for x, y, label in IGNORED:
        r, c = _pix(x, y)
        assert hm[:, r, c].max() < 1, f'label {label} became a positive'
    assert int(t['masks'][0].sum()) == 3, 'regression targets come from positives only'


def test_ignore_mask_zeroes_ignored_footprints_only():
    t = _targets([_box(*p) for p in POSITIVES + IGNORED])
    mask = t['heatmap_masks'][0][0]
    for x, y, _ in IGNORED:
        assert mask[_pix(x, y)] == 0
    for x, y, _ in POSITIVES:
        assert mask[_pix(x, y)] == 1
    assert mask[_pix(-9.5, 9.5)] == 1, 'far from every box the loss must be untouched'


def test_rotated_ignore_box_covers_its_length_not_its_sides():
    """A 10 x 2 m box at 30 deg: a point 4.5 m along its axis is ignored, 3 m to its side is not.

    (2 m to the side is still inside the square its gaussian would cover, radius 4 px = 1.6 m.)"""
    heading = np.pi / 6
    t = _targets([_box(0, 0, IGNORE_CLASS_LABEL, dims=(10.0, 2.0, 3.0), heading=heading)])
    mask = t['heatmap_masks'][0][0]
    along = (4.5 * np.cos(heading), 4.5 * np.sin(heading))
    side = (-3.0 * np.sin(heading), 3.0 * np.cos(heading))
    assert mask[_pix(*along)] == 0
    assert mask[_pix(*side)] == 1


def test_no_ignore_boxes_means_no_mask():
    """Rows without ignore boxes must compute exactly the loss they always did (mask=None)."""
    t = _targets([_box(*p) for p in POSITIVES])
    assert t['heatmap_masks'] == [None]


def test_a_positive_inside_an_ignore_region_keeps_its_peak():
    """A car parked inside a truck's ignore footprint is still a car."""
    t = _targets([_box(0, 0, IGNORE_CLASS_LABEL, dims=(12.0, 3.0, 3.0)), _box(1.0, 0.0, 1)])
    hm, mask = t['heatmaps'][0][0], t['heatmap_masks'][0][0]
    r, c = _pix(1.0, 0.0)
    assert hm[0, r, c] == 1 and mask[r, c] == 1
    assert mask[_pix(-4.0, 0.0)] == 0


def _loss(head, hm_logits):
    pred = {'hm': hm_logits.clone(), 'center': torch.zeros(1, 2, *FMAP), 'center_z': torch.zeros(1, 1, *FMAP),
            'dim': torch.zeros(1, 3, *FMAP), 'rot': torch.zeros(1, 2, *FMAP)}
    head.forward_ret_dict = {'pred_dicts': [pred], 'target_dicts': head.forward_ret_dict['target_dicts']}
    loss, _ = head.get_loss()
    return float(loss)


def test_heatmap_loss_ignores_predictions_inside_ignore_regions_in_every_channel():
    head = _head()
    gt = torch.tensor([[_box(*p) for p in POSITIVES + IGNORED]], dtype=torch.float32)
    head.forward_ret_dict = {'target_dicts': head.assign_targets(gt, feature_map_size=FMAP)}
    base = torch.full((1, 3, *FMAP), -2.0)
    ref = _loss(head, base)
    r, c = _pix(6, 0)                                   # an ignored pedestrian
    inside = base.clone(); inside[0, :, r, c] = 4.0     # confident in EVERY channel there
    assert _loss(head, inside) == pytest.approx(ref, abs=1e-6), 'an ignored footprint must carry no loss'
    r, c = _pix(-9.5, 9.5)                              # background
    outside = base.clone(); outside[0, :, r, c] = 4.0
    assert _loss(head, outside) > ref + 1e-4, 'background must still be penalised'


# ------------------------------------------------------------------ DATA_CONFIG.IGNORE_OTHER_CLASSES

def _dataset(ignore_other, training=True):
    cfg = EasyDict({
        'POINT_CLOUD_RANGE': [-75.2, -75.2, -2, 75.2, 75.2, 4],
        'POINT_FEATURE_ENCODING': {'encoding_type': 'absolute_coordinates_encoding',
                                   'used_feature_list': ['x', 'y', 'z'],
                                   'src_feature_list': ['x', 'y', 'z', 'intensity']},
        'ONTOLOGY': 'kitti',
        'DATA_PROCESSOR': [],
        'DATA_AUGMENTOR': {'DISABLE_AUG_LIST': [],
                           'AUG_CONFIG_LIST': [{'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']}]},
    })
    if ignore_other:
        cfg['IGNORE_OTHER_CLASSES'] = True
    return DatasetTemplate(dataset_cfg=cfg, class_names=list(NAMES), training=training,
                           root_path=Path('/tmp'), logger=mock.MagicMock())


def _frame():
    names = np.array(['Car', 'Truck', 'Misc', 'Pedestrian'])
    boxes = np.array([[5, 0, 0, 4, 2, 1.5, 0], [10, 3, 0, 9, 2.6, 3.2, 0.3],
                      [15, -3, 0, 0.5, 0.5, 1, 0], [20, 5, 0, 0.8, 0.8, 1.7, 0]], dtype=np.float32)
    return {'points': np.random.rand(256, 4).astype(np.float32), 'gt_boxes': boxes, 'gt_names': names,
            'num_points_in_gt': np.array([30, 40, 5, 12], dtype=np.int32), 'frame_id': '0'}


def test_ignore_other_classes_keeps_them_with_the_ignore_label():
    out = _dataset(True).prepare_data(_frame())
    assert list(out['gt_names']) == ['Car', 'Truck', 'Misc', 'Pedestrian']
    assert list(out['gt_boxes'][:, 7]) == [1, IGNORE_CLASS_LABEL, IGNORE_CLASS_LABEL, 2]


def test_without_the_key_other_classes_are_dropped_as_before():
    out = _dataset(False).prepare_data(_frame())
    assert list(out['gt_names']) == ['Car', 'Pedestrian']
    assert list(out['gt_boxes'][:, 7]) == [1, 2]


def test_the_key_does_nothing_at_evaluation():
    out = _dataset(True, training=False).prepare_data(_frame())
    assert list(out['gt_names']) == ['Car', 'Pedestrian']


def test_ignore_label_stays_outside_the_pseudo_label_band():
    """-1..-C are recovered with abs() as a class; the class-less value must not collide with them."""
    assert IGNORE_CLASS_LABEL < -len(NAMES)
