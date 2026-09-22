"""Tests for the two KITTI-eval behaviours this project's results depend on and cannot see.

Both are recorded as gotchas in the shared memory index, both silently change published numbers,
and neither had an executable guard before this file.

1. **AP is truncated, not merely reduced, when detections run out.** `eval_class` allocates a
   41-slot precision array and fills only `len(get_thresholds(...))` of them; `get_mAP_R40` then
   sums slots 1..40 and divides by 40 regardless. Unreachable recall points are therefore averaged
   in as ZERO precision. That is the mechanism behind "never RAISE SCORE_THRESH to relieve
   eval-time OOM" - a higher threshold discards low-confidence detections, which caps reachable
   recall, which leaves slots empty, which biases AP DOWN. The index's wording for this was once
   corrected because it had been written backwards; a test states the direction unambiguously.

2. **easy == moderate == hard is a signature, not a coincidence.** Difficulty is decided purely by
   the 2D image box's height, the occlusion flag and the truncation fraction. Every non-KITTI
   target fabricates `bbox = [0, 0, 50, 50]` with occlusion and truncation 0, which clears all
   three difficulty levels, so nothing is ever filtered and the three AP columns are identical.
   Those numbers are "all objects, no difficulty filter" - STRICTER than KITTI *hard*, and never
   "moderate". The tests below pin the thresholds that make that true.

See experiments_md/20260919_05 sections 6-8 and 20260920_02 for the audits these come from.

CPU-only: `eval.py` imports `rotate_iou`, which compiles a numba.cuda kernel at import time and
dies on a machine with no driver - even under `singularity exec --nv` (see the container notes'
Gotcha #4). Only that ONE module is stubbed here, so every function under test is the real one;
test_kitti_eval_class_mapping.py stubs the whole of `eval` instead because it needs a different
thing from it.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ROTATE_IOU = 'pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou'
if _ROTATE_IOU not in sys.modules:
    _stub = types.ModuleType(_ROTATE_IOU)
    _stub.rotate_iou_gpu_eval = lambda *args, **kwargs: None
    sys.modules[_ROTATE_IOU] = _stub

from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval  # noqa: E402

N_SAMPLE_PTS = 41


def _precision(filled, value=1.0):
    """A precision array with the first `filled` of 41 recall slots at `value`, rest at zero.

    This is the shape `eval_class` produces: it writes one slot per threshold returned by
    get_thresholds and leaves the remainder at the np.zeros it allocated.
    """
    prec = np.zeros((1, 1, 1, N_SAMPLE_PTS))
    prec[..., :filled] = value
    return prec


# --- how the 41 slots become an AP ---------------------------------------------------------------

def test_r40_averages_forty_slots_and_skips_the_first():
    """R40 deliberately drops recall point 0; R11 samples every fourth from 0."""
    assert kitti_eval.get_mAP_R40(_precision(N_SAMPLE_PTS)).ravel()[0] == pytest.approx(100.0)
    assert kitti_eval.get_mAP(_precision(N_SAMPLE_PTS)).ravel()[0] == pytest.approx(100.0)
    # Only slot 0 is perfect: R40 ignores it entirely, R11 counts it as one of its eleven.
    assert kitti_eval.get_mAP_R40(_precision(1)).ravel()[0] == pytest.approx(0.0)
    assert kitti_eval.get_mAP(_precision(1)).ravel()[0] == pytest.approx(100.0 / 11)


@pytest.mark.parametrize('filled', [1, 5, 11, 21, 31, 41])
def test_unfilled_slots_are_averaged_in_as_zero(filled):
    """THE mechanism: perfect precision everywhere it was measured still yields (k-1)/40."""
    expected = max(filled - 1, 0) / 40.0 * 100.0
    assert kitti_eval.get_mAP_R40(_precision(filled)).ravel()[0] == pytest.approx(expected)


def test_half_the_recall_points_costs_half_the_ap():
    """A detector with perfect precision over 21 of 41 points scores 50, not 100."""
    assert kitti_eval.get_mAP_R40(_precision(21)).ravel()[0] == pytest.approx(50.0)


# --- what decides how many slots get filled -------------------------------------------------------

def test_threshold_count_is_bounded_by_the_detections_available():
    """Five detections cannot sample forty-one recall points, however good they are."""
    scores = np.linspace(0.5, 0.9, 5)
    assert len(kitti_eval.get_thresholds(scores.copy(), 100)) <= len(scores)


def test_threshold_count_never_exceeds_the_forty_one_slots():
    scores = np.linspace(0.01, 0.99, 5000)
    assert len(kitti_eval.get_thresholds(scores.copy(), 5000)) <= N_SAMPLE_PTS


def test_get_thresholds_sorts_its_input_in_place():
    """`scores.sort()` mutates the caller's array. Pinned so a shared array is never passed."""
    scores = np.array([0.3, 0.9, 0.1])
    kitti_eval.get_thresholds(scores, 10)
    assert np.array_equal(scores, np.array([0.1, 0.3, 0.9]))


# --- the direction of the SCORE_THRESH bias, stated unambiguously ---------------------------------

def _best_case_ap(scores, num_gt):
    """The AP a PERFECT detector would score with this many usable detections.

    Precision is 1.0 at every recall point that could be reached, 0 at the rest - so anything
    this measures is purely the truncation, with detector quality held at its ceiling.
    """
    filled = len(kitti_eval.get_thresholds(np.asarray(scores, dtype=float).copy(), num_gt))
    return kitti_eval.get_mAP_R40(_precision(filled)).ravel()[0]


def test_raising_the_score_threshold_can_only_lower_the_attainable_ap():
    """Discarding low-confidence detections caps recall, which leaves 41-slot entries at zero.

    This is the executable form of "never RAISE SCORE_THRESH to relieve eval-time OOM".
    """
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.0, 1.0, size=500)
    num_gt = 500
    attainable = [_best_case_ap(scores[scores >= t], num_gt)
                  for t in [0.0001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]]
    assert all(a >= b for a, b in zip(attainable, attainable[1:])), attainable
    assert attainable[0] > attainable[-1]


def test_the_permissive_threshold_reaches_every_recall_point():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.0, 1.0, size=2000)
    assert _best_case_ap(scores, 2000) == pytest.approx(100.0)


def test_a_severe_threshold_truncates_even_a_perfect_detector():
    rng = np.random.default_rng(2)
    scores = rng.uniform(0.0, 1.0, size=2000)
    assert _best_case_ap(scores[scores >= 0.98], 2000) < 20.0


# --- difficulty: why three identical AP columns are a signature ------------------------------------

def _anno(names, bboxes, occluded, truncated):
    return {'name': np.array(names), 'bbox': np.array(bboxes, dtype=float),
            'occluded': np.array(occluded), 'truncated': np.array(truncated, dtype=float)}


FABRICATED = [0.0, 0.0, 50.0, 50.0]   # what every non-KITTI target's loader writes


def _valid_at(difficulty, anno):
    dt = _anno(['Car'], [FABRICATED], [0], [0.0])
    return kitti_eval.clean_data(anno, dt, 0, difficulty)[0]


@pytest.mark.parametrize('difficulty', [0, 1, 2])
def test_a_fabricated_box_is_valid_at_every_difficulty(difficulty):
    """Height 50 > 40, occlusion 0, truncation 0 - it clears easy, moderate and hard alike."""
    assert _valid_at(difficulty, _anno(['Car'], [FABRICATED], [0], [0.0])) == 1


def test_fabricated_boxes_make_the_three_levels_identical():
    """The telltale sign of a non-KITTI eval target: easy == moderate == hard, exactly."""
    anno = _anno(['Car'] * 4, [FABRICATED] * 4, [0] * 4, [0.0] * 4)
    assert [_valid_at(d, anno) for d in (0, 1, 2)] == [4, 4, 4]


def test_real_annotations_do_stratify_by_height():
    """A 30-pixel box fails easy (<= 40) and passes moderate and hard (> 25) - real KITTI."""
    anno = _anno(['Car'], [[0.0, 0.0, 50.0, 30.0]], [0], [0.0])
    assert [_valid_at(d, anno) for d in (0, 1, 2)] == [0, 1, 1]


def test_real_annotations_do_stratify_by_occlusion():
    anno = _anno(['Car'], [FABRICATED], [2], [0.0])
    assert [_valid_at(d, anno) for d in (0, 1, 2)] == [0, 0, 1]


def test_real_annotations_do_stratify_by_truncation():
    anno = _anno(['Car'], [FABRICATED], [0], [0.4])
    assert [_valid_at(d, anno) for d in (0, 1, 2)] == [0, 0, 1]


def test_the_fabricated_column_is_stricter_than_hard_not_equal_to_moderate():
    """Nothing is ever ignored, so the fabricated regime scores MORE objects than real hard does.

    That is why a fabricated-bbox number must never be tabled beside a real KITTI moderate one.
    """
    real = _anno(['Car'] * 3,
                 [[0.0, 0.0, 50.0, 30.0], FABRICATED, [0.0, 0.0, 50.0, 20.0]],
                 [0, 0, 0], [0.0, 0.0, 0.0])
    fabricated = _anno(['Car'] * 3, [FABRICATED] * 3, [0, 0, 0], [0.0, 0.0, 0.0])
    assert _valid_at(2, real) == 2            # the 20-pixel box is ignored even at hard
    assert _valid_at(2, fabricated) == 3      # under fabrication, all three are scored


def test_gt_and_detection_height_cutoffs_are_asymmetric_at_the_boundary():
    """GT uses `height <= MIN_HEIGHT`, detections use `height < MIN_HEIGHT`.

    So a GT box exactly 40 pixels tall is ignored at easy while a detection exactly 40 tall is
    not. Upstream KITTI behaviour, pinned because it is the kind of off-by-one that a reader of
    the two branches would assume away.
    """
    gt = _anno(['Car'], [[0.0, 0.0, 50.0, 40.0]], [0], [0.0])
    dt = _anno(['Car'], [[0.0, 0.0, 50.0, 40.0]], [0], [0.0])
    num_valid_gt, _, ignored_dt, _ = kitti_eval.clean_data(gt, dt, 0, 0)
    assert num_valid_gt == 0                  # GT at exactly 40 is ignored
    assert ignored_dt == [0]                  # the detection at exactly 40 is not
