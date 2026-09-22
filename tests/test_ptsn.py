"""Tests for PTSN, DALI's inference-time size normalization (IEEE T-RO 2024).

PTSN scales the input cloud by `s` and divides the predicted boxes by `s`. The whole method is
that pair of transforms, so the properties worth pinning down are the ones that make the pair
trustworthy rather than anything about accuracy:

  * SCALE 1.0 is a byte-exact no-op. Every run in this repo that does not ask for PTSN takes
    this path, so it has to be identity, not merely close to it.
  * scale-then-unscale is the identity on boxes, and heading is untouched - an isotropic scaling
    does not rotate anything.
  * the transform fires only in eval mode. Under self-training one dataset object serves a
    training loader and a generation loader, and the gate is what routes the scaling to the
    generation pass alone; if it leaked into training, the student would see scaled points
    against pseudo-labels that had already been divided by s.
  * the predicted size moves as 1/s. This is the claim the search rests on, so it gets an
    explicit test against a synthetic detector that reports a fixed box in the scaled frame.
  * select_scale picks the argmin, and ties resolve to the first candidate so a deliberately
    ordered sweep is reproducible.

See experiments_md/20260922_04 section 2 for the plan this implements, and section 2.3 for why a
PTSN row's UDA legality is inherited from whatever estimated the target size, not from PTSN.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402
from pcdet.utils import ptsn_utils  # noqa: E402

POINT_CLOUD_RANGE = np.array([-75.2, -75.2, -2.0, 75.2, 75.2, 4.0])


def _processor(training=False, configs=None):
    """A DataProcessor with an empty queue, so a test sees the PTSN step and nothing else."""
    return DataProcessor(
        processor_configs=configs if configs is not None else [],
        point_cloud_range=POINT_CLOUD_RANGE,
        training=training,
        num_point_features=4,
    )


def _points(n=64, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.empty((n, 4), dtype=np.float32)
    pts[:, 0:3] = rng.uniform(-40, 40, size=(n, 3))
    pts[:, 3] = rng.uniform(0, 1, size=n)      # intensity: must survive untouched
    return pts


def _boxes(n=8, seed=1):
    rng = np.random.default_rng(seed)
    boxes = np.empty((n, 7), dtype=np.float32)
    boxes[:, 0:3] = rng.uniform(-40, 40, size=(n, 3))
    boxes[:, 3:6] = rng.uniform(1.0, 5.0, size=(n, 3))
    boxes[:, 6] = rng.uniform(-np.pi, np.pi, size=n)
    return boxes


# --- the identity case, which is every non-PTSN run -------------------------------------------

def test_default_scale_is_one():
    assert _processor().ptsn_scale == 1.0


def test_scale_one_is_a_byte_exact_no_op_on_points():
    pts = _points()
    out = ptsn_utils.scale_points(pts, 1.0)
    assert np.array_equal(out, pts)


def test_scale_one_is_a_byte_exact_no_op_through_the_processor():
    proc = _processor(training=False)
    pts = _points()
    out = proc.forward({'points': pts.copy()})['points']
    assert np.array_equal(out, pts)


def test_unscale_at_one_is_a_byte_exact_no_op_on_boxes():
    boxes = _boxes()
    assert np.array_equal(ptsn_utils.unscale_boxes(boxes, 1.0), boxes)


# --- the forward transform --------------------------------------------------------------------

@pytest.mark.parametrize('scale', [0.8, 1.0, 1.25])
def test_only_xyz_is_scaled(scale):
    pts = _points()
    out = ptsn_utils.scale_points(pts, scale)
    assert np.allclose(out[:, 0:3], pts[:, 0:3] * scale)
    assert np.array_equal(out[:, 3], pts[:, 3])


def test_scale_points_does_not_mutate_its_input():
    pts = _points()
    before = pts.copy()
    ptsn_utils.scale_points(pts, 1.3)
    assert np.array_equal(pts, before)


@pytest.mark.parametrize('bad', [0.0, -1.0])
def test_non_positive_scale_is_rejected(bad):
    with pytest.raises(AssertionError):
        ptsn_utils.scale_points(_points(), bad)
    with pytest.raises(AssertionError):
        _processor().set_ptsn_scale(bad)


# --- the gate ---------------------------------------------------------------------------------

def test_processor_scales_in_eval_mode():
    proc = _processor(training=False)
    proc.set_ptsn_scale(1.2)
    pts = _points()
    out = proc.forward({'points': pts.copy()})['points']
    assert np.allclose(out[:, 0:3], pts[:, 0:3] * 1.2)


def test_processor_does_not_scale_in_train_mode():
    proc = _processor(training=True)
    proc.set_ptsn_scale(1.2)
    pts = _points()
    out = proc.forward({'points': pts.copy()})['points']
    assert np.array_equal(out, pts)


def test_mode_switch_flips_the_gate():
    """One dataset object serves both loaders under self-training; eval()/train() is the router."""
    proc = _processor(training=True)
    proc.set_ptsn_scale(1.2)
    pts = _points()
    assert np.array_equal(proc.forward({'points': pts.copy()})['points'], pts)
    proc.eval()
    assert not np.array_equal(proc.forward({'points': pts.copy()})['points'], pts)
    proc.train()
    assert np.array_equal(proc.forward({'points': pts.copy()})['points'], pts)


def test_gt_boxes_are_left_alone():
    """PTSN is inference-only and its inverse lands on predictions, never on GT."""
    proc = _processor(training=False)
    proc.set_ptsn_scale(1.2)
    boxes = _boxes()
    out = proc.forward({'points': _points(), 'gt_boxes': boxes.copy()})['gt_boxes']
    assert np.array_equal(out, boxes)


def test_scaling_runs_before_the_range_crop():
    """The crop must see scaled coordinates - it is the grid the network is built on.

    A consequence, stated here so it is not discovered in a log: at s > 1 the far field is cut
    harder than at s = 1, because POINT_CLOUD_RANGE is fixed while the scene grows.
    """
    crop = [EasyDict({'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': True})]
    pts = np.array([[70.0, 0.0, 0.0, 0.5]], dtype=np.float32)   # inside at s=1, outside at s=1.2
    assert len(_processor(configs=crop).forward({'points': pts.copy()})['points']) == 1
    proc = _processor(configs=crop)
    proc.set_ptsn_scale(1.2)
    assert len(proc.forward({'points': pts.copy()})['points']) == 0


# --- the inverse ------------------------------------------------------------------------------

@pytest.mark.parametrize('scale', [0.75, 0.9, 1.0, 1.1, 1.4])
def test_scale_then_unscale_is_the_identity_on_boxes(scale):
    boxes = _boxes()
    scaled = boxes.copy()
    scaled[:, 0:6] *= scale                       # what the network sees a box as, in scaled metres
    assert np.allclose(ptsn_utils.unscale_boxes(scaled, scale), boxes, atol=1e-6)


@pytest.mark.parametrize('scale', [0.75, 1.4])
def test_unscale_leaves_heading_untouched(scale):
    boxes = _boxes()
    out = ptsn_utils.unscale_boxes(boxes, scale)
    assert np.array_equal(out[:, 6], boxes[:, 6])


def test_unscale_handles_an_empty_prediction():
    empty = np.zeros((0, 7), dtype=np.float32)
    assert ptsn_utils.unscale_boxes(empty, 1.2).shape == (0, 7)


def test_unscale_does_not_mutate_its_input():
    boxes = _boxes()
    before = boxes.copy()
    ptsn_utils.unscale_boxes(boxes, 1.2)
    assert np.array_equal(boxes, before)


# --- the 1/s claim the search rests on ----------------------------------------------------------

@pytest.mark.parametrize('scale', [0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
def test_reported_size_moves_as_one_over_s(scale):
    """A detector whose size prior is frozen at the source reports the same box in the scaled
    frame whatever `s` is; dividing by `s` therefore yields source_size / s."""
    source_mean = np.array([4.60, 1.95, 1.72])
    predicted_in_scaled_frame = np.concatenate([np.zeros(3), source_mean, np.zeros(1)])[None, :]
    reported = ptsn_utils.unscale_boxes(predicted_in_scaled_frame, scale)[0, 3:6]
    assert np.allclose(reported, source_mean / scale)


def test_reported_size_is_strictly_decreasing_in_s():
    source_mean = np.array([4.60, 1.95, 1.72])
    box = np.concatenate([np.zeros(3), source_mean, np.zeros(1)])[None, :]
    sizes = [ptsn_utils.unscale_boxes(box, s)[0, 3] for s in [0.8, 0.9, 1.0, 1.1, 1.2]]
    assert all(a > b for a, b in zip(sizes, sizes[1:]))


# --- the search's decision rule -----------------------------------------------------------------

def test_select_scale_picks_the_argmin():
    source_mean = np.array([4.60, 1.95, 1.72])
    candidates = [0.9, 1.0, 1.1, 1.2]
    means = [source_mean / s for s in candidates]
    target = source_mean / 1.1                    # exactly reachable at one candidate
    best, rows = ptsn_utils.select_scale(candidates, means, target)
    assert best == 1.1
    assert len(rows) == len(candidates)
    assert rows[2][2] == pytest.approx(0.0, abs=1e-12)


def test_select_scale_gap_is_mean_absolute_metres():
    best, rows = ptsn_utils.select_scale([1.0], [[4.0, 2.0, 1.5]], [4.3, 2.0, 1.5])
    assert best == 1.0
    assert rows[0][2] == pytest.approx(0.1)


def test_select_scale_breaks_ties_towards_the_first_candidate():
    target = np.array([4.0, 2.0, 1.5])
    best, _ = ptsn_utils.select_scale([0.9, 1.1], [target + 0.1, target - 0.1], target)
    assert best == 0.9


def test_select_scale_rejects_an_empty_sweep():
    with pytest.raises(AssertionError):
        ptsn_utils.select_scale([], np.zeros((0, 3)), [4.0, 2.0, 1.5])


def test_select_scale_never_prefers_a_candidate_that_predicted_nothing():
    """A scale at which the detector found no boxes arrives as NaN; argmin would pick it."""
    target = np.array([4.0, 2.0, 1.5])
    means = [np.full(3, np.nan), target + 0.3]
    best, rows = ptsn_utils.select_scale([0.9, 1.1], means, target)
    assert best == 1.1
    assert not np.isfinite(rows[0][2])
