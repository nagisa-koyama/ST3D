"""Empty boxes must not enter the foreground per-box mean.

The foreground channel is points per BOX. A box holding no points contributes nothing to the
numerator by definition, so counting it in the denominator can only drag the mean down - and what
it measures is not density but annotation policy, occlusion or sensor coverage.

PandarGT shows the scale: it labels the full 360 degrees while seeing about +-30, so 80.2% of its
Car boxes are empty. Including them would put the foreground estimate roughly a factor of five
below the density any detector actually sees, and since the correction applies
min(F_target/F_source, 1), an understated TARGET foreground over-thins the source.

This is the same confound as counting foreground per FRAME, one level down - see
experiments_md/20260922_08.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcdet.datasets.point_calibration import compute_foreground_histograms  # noqa: E402


def box(x, y=0.0, z=0.0, dx=4.0, dy=2.0, dz=2.0, yaw=0.0):
    return [x, y, z, dx, dy, dz, yaw]


class _Proc:
    """Stands in for DataProcessor.box_occupancy without the C++ kernel (axis-aligned is enough)."""

    @staticmethod
    def box_occupancy(points, boxes):
        if boxes is None or len(boxes) == 0:
            return (np.zeros(len(points), bool), np.zeros(0, np.int64), np.zeros((0, 7), np.float32))
        boxes = np.asarray(boxes, dtype=np.float32)[:, :7]
        boxes = boxes[(boxes[:, 3:6] > 1e-3).all(axis=1)]
        if len(boxes) == 0:
            return (np.zeros(len(points), bool), np.zeros(0, np.int64), np.zeros((0, 7), np.float32))
        inside = np.stack([
            (np.abs(points[:, 0] - b[0]) <= b[3] / 2) & (np.abs(points[:, 1] - b[1]) <= b[4] / 2)
            & (np.abs(points[:, 2] - b[2]) <= b[5] / 2) for b in boxes])
        return inside.any(axis=0), inside.sum(axis=1), boxes


class _Set:
    """Minimal dataset: one frame, given points and boxes."""

    def __init__(self, points, boxes, n=4):
        self._p, self._b, self._n = np.asarray(points, float), np.asarray(boxes, float), n
        self.data_processor = _Proc()

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return {'points': self._p, 'gt_boxes': self._b}


def _fg_at(points, boxes, min_points=1, num_bins=10, max_dist=50.0):
    fg, _, _ = compute_foreground_histograms(_Set(points, boxes), num_frames=2, num_bins=num_bins,
                                             max_dist=max_dist, min_points_in_box=min_points)
    return fg


def test_an_empty_box_does_not_halve_the_mean():
    """One box with 10 points and one empty box in the same bin: the mean is 10, not 5."""
    pts = np.column_stack([np.full(10, 10.0), np.zeros(10), np.zeros(10)])
    # the second box is at the SAME radius, 90 degrees round, so it shares the bin and holds
    # nothing - if it were merely far away it would land in another bin and prove nothing
    boxes = [box(10.0, 0.0), box(0.0, 10.0)]
    fg = _fg_at(pts, boxes, min_points=1)
    assert fg.max() == pytest.approx(10.0), fg


def test_counting_empty_boxes_is_what_the_old_behaviour_did():
    """min_points_in_box=0 keeps every box, reproducing the pre-fix mean - the control."""
    pts = np.column_stack([np.full(10, 10.0), np.zeros(10), np.zeros(10)])
    boxes = [box(10.0, 0.0), box(0.0, 10.0)]
    assert _fg_at(pts, boxes, min_points=0).max() == pytest.approx(5.0)


def test_the_exclusion_only_touches_the_denominator():
    """An empty box contributes no points, so the numerator is identical either way."""
    pts = np.column_stack([np.full(10, 10.0), np.zeros(10), np.zeros(10)])
    kept = _fg_at(pts, [box(10.0, 0.0)], min_points=1)
    with_empty = _fg_at(pts, [box(10.0, 0.0), box(0.0, 10.0)], min_points=1)
    assert np.allclose(kept, with_empty)


def test_boxes_in_a_different_bin_do_not_interfere():
    pts = np.column_stack([np.full(6, 5.0), np.zeros(6), np.zeros(6)])
    fg = _fg_at(pts, [box(5.0, 0.0), box(40.0, 0.0)], min_points=1, num_bins=10, max_dist=50.0)
    assert fg[1] == pytest.approx(6.0)     # 5 m falls in bin 1 of 10 over 50 m
    assert fg[8] == 0.0                    # the empty far box leaves its bin at 0, not negative


def test_a_higher_threshold_excludes_sparsely_sampled_boxes_too():
    pts = np.array([[10.0, 0.0, 0.0], [10.1, 0.0, 0.0], [10.2, 0.0, 0.0],
                    [30.0, 0.0, 0.0]])
    # the 30 m box holds a single point; at min_points=2 it stops counting at all
    at1 = _fg_at(pts, [box(10.0), box(30.0)], min_points=1)
    at2 = _fg_at(pts, [box(10.0), box(30.0)], min_points=2)
    assert at1[6] == pytest.approx(1.0) and at2[6] == 0.0


def test_all_boxes_empty_leaves_the_channel_at_zero_rather_than_nan():
    """np.divide's `where=` guard must hold: a nan rate would silently delete every point."""
    pts = np.column_stack([np.full(5, 10.0), np.zeros(5), np.zeros(5)])
    fg = _fg_at(pts, [box(40.0, 30.0)], min_points=1)
    assert np.all(np.isfinite(fg)) and fg.max() == 0.0


def test_degenerate_boxes_are_dropped_from_the_denominator_too():
    """They never reach the geometry kernel, so counting them would repeat the same error."""
    pts = np.column_stack([np.full(8, 10.0), np.zeros(8), np.zeros(8)])
    fg = _fg_at(pts, [box(10.0), box(10.0, 0.0, 0.0, 0.0, 0.0, 0.0)], min_points=1)
    assert fg.max() == pytest.approx(8.0)


def test_the_self_training_loop_passes_the_config_key_through():
    src = (Path(__file__).resolve().parent.parent
           / 'tools/train_utils/train_st_utils.py').read_text(encoding='utf-8')
    assert "HIST_DIST_MIN_POINTS_IN_BOX" in src, 'the refresh path must honour the threshold'
