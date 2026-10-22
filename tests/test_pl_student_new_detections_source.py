"""tools/analysis/pl_student_new_detections_source.py: the operating point, the found sets, the track link and the bins
(CPU; experiments_md 20261022_02, check PL-SRC of 20261003_05 §5)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import pl_student_new_detections_source as P  # noqa: E402
import waymo_ranking_loss_fp_taxonomy as W  # noqa: E402


def test_tau_is_the_deepest_threshold_at_or_above_the_precision():
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    t = np.array([1, 1, 0, 1, 0, 0], bool)  # precision down the list: 1, 1, 0.667, 0.75, 0.6, 0.5
    tau, prec, ntp, tau_first = P.tau_at_precision(s, t, 0.7)
    assert tau == 0.6 and abs(prec - 0.75) < 1e-12 and ntp == 3
    assert tau_first == 0.8  # first fall below 0.7 is at 0.7; the threshold just before it is 0.8
    tau, prec, ntp, _ = P.tau_at_precision(s, t, 0.5)
    assert tau == 0.4 and ntp == 3
    assert np.isnan(P.tau_at_precision(s, np.zeros(6, bool), 0.5)[0])


def test_tau_keeps_ties_together():
    s = np.array([0.9, 0.9, 0.5])
    t = np.array([1, 0, 1], bool)  # the 0.9 group alone has precision 0.5; with 0.5 added, 0.667
    tau, prec, ntp, _ = P.tau_at_precision(s, t, 0.6)
    assert tau == 0.5 and ntp == 2
    tau, prec, ntp, _ = P.tau_at_precision(s[::-1].copy(), t[::-1].copy(), 0.5)  # input order does not matter
    assert tau == 0.5 and ntp == 2


def test_found_from_all_detections_equals_matching_only_those_above_tau():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_dt, n_gt = rng.integers(0, 12), rng.integers(0, 8)
        iou = rng.uniform(0, 1, (n_dt, n_gt)) * (rng.uniform(0, 1, (n_dt, n_gt)) < 0.4)
        scores = rng.uniform(0, 1, n_dt)
        tau = rng.uniform(0, 1)
        _, gt_dt, _ = W.greedy_match(iou, scores, 0.7)
        det_score = np.where(gt_dt >= 0, scores[np.maximum(gt_dt, 0)] if n_dt else 0, -1.0)
        found = P.found_at(det_score, tau)
        keep = scores >= tau
        _, gt_dt_k, _ = W.greedy_match(iou[keep], scores[keep], 0.7)
        assert (found == (gt_dt_k >= 0)).all()


def test_track_link_needs_a_found_gt_of_the_same_object_in_another_frame():
    key = np.array([0, 0, 0, 1, 1, 2])
    frame = np.array([0, 1, 2, 0, 1, 3])
    found = np.array([1, 0, 0, 0, 0, 1], bool)
    assert P.track_linked(key, frame, found).tolist() == [False, True, True, False, False, False]
    found = np.array([1, 1, 0, 0, 0, 0], bool)
    assert P.track_linked(key, frame, found).tolist() == [True, True, True, False, False, False]


def test_bins_ring_major_with_zero_points_and_far_gt_outside():
    rng = np.array([5.0, 15.0, 74.9, 75.0, 30.0, 0.0, 49.99, 50.0])
    npts = np.array([1, 10, 200, 5, 0, 9, 50, 199])
    assert P.bin_index(rng, npts).tolist() == [0, 5, 23, -1, -1, 0, 4 * 4 + 2, 5 * 4 + 2]


def test_adjacency_shares_over_binned_gt():
    bins = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1])
    found_t = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], bool)  # r_T bin 0 = 0.2 (qualifies), bin 1 = 0
    s_only = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1], bool)
    neither = ~found_t & ~s_only
    r_t, q, share = P.adjacency(bins, found_t, dict(s_only=s_only, neither=neither))
    assert abs(r_t[0] - 0.2) < 1e-12 and r_t[1] == 0 and q[0] and not q[1] and np.isnan(r_t[2])
    assert abs(share['s_only'] - 2 / 3) < 1e-12        # the unbinned S-only GT is not in the denominator
    assert abs(share['neither'] - 2 / 6) < 1e-12
