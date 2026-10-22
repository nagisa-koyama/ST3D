"""tools/analysis/pl_teacher_near_miss_cuts.py: labels with the evaluator's ignores, near-miss scores, zones and the
H-band reading (CPU; experiments_md 20261022_03, check PL-NEG of 20261003_05 §5)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import pl_teacher_near_miss_cuts as N  # noqa: E402


def test_labels_with_ignored_gt_and_ignored_detections():
    # GT: g0 valid, g1 ignored (Van / other difficulty), g2 other class (never matchable)
    ig_gt = [0, 1, -1]
    iou = np.array([
        [0.80, 0.00, 0.00],   # d0 0.9 -> TP on g0
        [0.75, 0.00, 0.00],   # d1 0.8 -> g0 taken: unmatched, valid height -> FP
        [0.00, 0.90, 0.00],   # d2 0.7 -> matched to the ignored g1 -> ignored
        [0.00, 0.00, 0.95],   # d3 0.6 -> only overlaps the other-class g2 -> FP
        [0.00, 0.00, 0.00],   # d4 0.5 -> unmatched, small 2D height -> ignored
    ])
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    ig_dt = [0, 0, 0, 0, 1]
    lab, matched, best_near = N.label_frame(iou, scores, ig_gt, ig_dt)
    assert lab.tolist() == [N.TP_, N.FP_, N.IGN_, N.FP_, N.IGN_]
    assert matched.tolist() == [0.9, 0.7, -1.0]
    assert best_near.tolist() == [0.9, 0.7, 0.6]


def test_near_miss_score_is_the_highest_box_at_iou_half_or_more():
    ig_gt = [0]
    iou = np.array([[0.55], [0.65], [0.45], [0.72]])
    scores = [0.30, 0.20, 0.90, 0.05]  # the 0.45 box scores highest but is under 0.5; the 0.72 box matches at 0.05
    lab, matched, best_near = N.label_frame(iou, scores, ig_gt, [0, 0, 0, 0])
    assert matched.tolist() == [0.05] and best_near.tolist() == [0.30]
    assert lab.tolist() == [N.FP_, N.FP_, N.FP_, N.TP_]


def test_no_ignores_reduces_to_plain_greedy():
    rng = np.random.default_rng(1)
    for _ in range(100):
        nd, ng = rng.integers(1, 10), rng.integers(1, 6)
        iou = rng.uniform(0, 1, (nd, ng)) * (rng.uniform(0, 1, (nd, ng)) < 0.5)
        s = rng.uniform(0, 1, nd)
        lab, matched, _ = N.label_frame(iou, s, np.zeros(ng), np.zeros(nd))
        dt_gt, gt_dt, _ = N.W.greedy_match(iou, s, 0.7)
        assert ((lab == N.TP_) == (dt_gt >= 0)).all() and ((matched >= 0) == (gt_dt >= 0)).all()


def test_zones_and_reading():
    z = N.zone_shares([0.05, 0.10, 0.2, 0.394, 0.5], 0.394, 0.10)
    assert (z['n_lt_neg'], z['n_band'], z['n_ge_score']) == (1, 2, 2)
    z = N.zone_shares([0.17, 0.18, 0.5], 0.18, 0.18)  # no band when SCORE == NEG
    assert (z['n_lt_neg'], z['n_band'], z['n_ge_score']) == (1, 0, 2)
    assert N.h_band_reading([0.6, 0.7, 0.8], [0.4, 0.5]) == 'supported'
    assert N.h_band_reading([0.6, 0.7, 0.8], [0.4, 0.55]) == 'inconclusive'
    assert N.h_band_reading([0.6, 0.7, 0.8], [0.4, 0.6]) == 'contradicted'
