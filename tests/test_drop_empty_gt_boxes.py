"""`DataProcessor.drop_empty_gt_boxes` (training-only recount after a point-dropping processor step,
experiments_md 20261005_01 §15.6): positive boxes left with too few points are dropped, ignore labels
(negative pseudo-label band, IGNORE_CLASS_LABEL -99) are kept whatever they hold, only `gt_boxes` is
touched, evaluation is a no-op, and the empty-cloud / no-box / degenerate-box sentinels behave."""
import sys
from pathlib import Path

import numpy as np
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import _init_path  # noqa: F401,E402
from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

RANGE = np.array([-75.2, -75.2, -2, 75.2, 75.2, 4], dtype=np.float32)


def _proc(training=True, **kw):
    cfg = [EasyDict(NAME='drop_empty_gt_boxes', **kw)]
    return DataProcessor(cfg, point_cloud_range=RANGE, training=training, num_point_features=3).forward


def _box(x, label):
    return [x, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0, label]


def _scene():
    # boxes centred at x = 0, 10, 20, 30, 40; points only in the boxes at 0 (3 points) and 10 (1 point)
    boxes = np.array([_box(0, 1), _box(10, 1), _box(20, 1), _box(30, -1), _box(40, common_utils.IGNORE_CLASS_LABEL)],
                     dtype=np.float32)
    pts = np.array([[0.0, 0.0, 1.0], [0.5, 0.2, 1.1], [-0.5, -0.2, 0.9], [10.0, 0.0, 1.0], [60.0, 0.0, 1.0]],
                   dtype=np.float32)
    return pts, boxes


def test_drops_empty_positive_keeps_ignore_labels():
    pts, boxes = _scene()
    names = np.array(['Car'] * 5)
    out = _proc()({'points': pts.copy(), 'gt_boxes': boxes.copy(), 'gt_names': names.copy()})
    assert out['gt_boxes'][:, 0].tolist() == [0.0, 10.0, 30.0, 40.0]       # x = 20 (positive, empty) dropped
    assert out['gt_boxes'][:, 7].tolist() == [1, 1, -1, common_utils.IGNORE_CLASS_LABEL]
    assert len(out['points']) == len(pts)                                    # points untouched
    assert len(out['gt_names']) == 5                                         # only gt_boxes is filtered


def test_min_points_threshold():
    pts, boxes = _scene()
    out = _proc(MIN_POINTS=2)({'points': pts.copy(), 'gt_boxes': boxes.copy()})
    assert out['gt_boxes'][:, 0].tolist() == [0.0, 30.0, 40.0]               # the 1-point box at 10 goes too


def test_eval_is_a_no_op():
    pts, boxes = _scene()
    out = _proc(training=False)({'points': pts.copy(), 'gt_boxes': boxes.copy()})
    assert out['gt_boxes'].shape == boxes.shape


def test_empty_cloud_drops_every_positive():
    _, boxes = _scene()
    out = _proc()({'points': np.zeros((0, 3), dtype=np.float32), 'gt_boxes': boxes.copy()})
    assert out['gt_boxes'][:, 7].tolist() == [-1, common_utils.IGNORE_CLASS_LABEL]


def test_no_boxes_and_none():
    pts, _ = _scene()
    out = _proc()({'points': pts.copy(), 'gt_boxes': np.zeros((0, 8), dtype=np.float32)})
    assert out['gt_boxes'].shape == (0, 8)
    out = _proc()({'points': pts.copy(), 'gt_boxes': None})
    assert out['gt_boxes'] is None
    out = _proc()({'points': pts.copy()})
    assert 'gt_boxes' not in out


def test_degenerate_positive_box_dropped_without_reaching_the_kernel():
    pts, boxes = _scene()
    boxes[0, 3:6] = 0.0                                                      # zero-extent positive box
    out = _proc()({'points': pts.copy(), 'gt_boxes': boxes.copy()})
    assert 0.0 not in out['gt_boxes'][:, 0].tolist()


def test_seven_column_boxes_are_treated_as_positive():
    pts, boxes = _scene()
    out = _proc()({'points': pts.copy(), 'gt_boxes': boxes[:, :7].copy()})
    assert out['gt_boxes'][:, 0].tolist() == [0.0, 10.0]
