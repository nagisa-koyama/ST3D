"""Rule PB census tool (tools/analysis/iassd_pb_budget_census.py; experiments_md 20261024_05): the pure functions that turn
per-frame point counts into a PB budget and into what a budget keeps. CPU only, no dataset.
"""
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from analysis import iassd_pb_budget_census as pb  # noqa: E402


def test_pb_budget_is_the_smallest_multiple_of_4096_at_or_above_p99():
    assert pb.pb_budget(25000.0) == 28672
    assert pb.pb_budget(24576.0) == 24576          # exactly a multiple: itself
    assert pb.pb_budget(24576.01) == 28672
    assert pb.pb_budget(1.0) == 4096
    assert pb.pb_budget(0.0) == 4096
    with pytest.raises(ValueError):
        pb.pb_budget(float('nan'))
    with pytest.raises(ValueError):
        pb.pb_budget(-1.0)


def test_strided_indices_follow_the_calibration_stride():
    assert pb.strided_indices(28130, 1000)[:3] == [0, 28, 56]
    assert len(pb.strided_indices(28130, 1000)) == 1000
    assert pb.strided_indices(500, 1000) == list(range(500))   # fewer frames than asked: every frame
    assert len(pb.strided_indices(3712, 1000)) == 1000          # KITTI train: stride 3, first 1,000


def test_multiplicity_histogram_counts_positions_by_how_many_rows_hold_them():
    xyz = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 0], [2, 0, 0]], dtype=np.float32)
    assert pb.multiplicity_histogram(xyz) == {1: 1, 2: 1, 3: 1}
    assert pb.multiplicity_histogram(np.zeros((0, 3), np.float32)) == {}


def test_expected_distinct_out_matches_sample_points_exactly_at_or_below_the_budget():
    """At or below the budget every row is kept (padding adds copies only): distinct = distinct in."""
    assert pb.expected_distinct_out(1000, {1: 1000}, 4096) == 1000
    assert pb.expected_distinct_out(1000, {1: 990, 2: 5}, 1000) == 995


def test_expected_distinct_out_matches_a_simulation_above_the_budget():
    rng = np.random.RandomState(0)
    # 3,000 rows: 2,000 single positions, 400 doubled, 66 tripled (+2 rows of a 2-fold) -> realistic duplicates
    loc = np.r_[np.arange(2000), np.repeat(np.arange(2000, 2400), 2), np.repeat(np.arange(2400, 2466), 3),
                np.repeat([2466], 2)]
    n = len(loc)
    hist = {}
    for c in np.unique(loc, return_counts=True)[1]:
        hist[int(c)] = hist.get(int(c), 0) + 1
    for B in (500, 1500, 2900):
        sim = np.mean([len(np.unique(loc[rng.choice(n, B, replace=False)])) for _ in range(400)])
        assert abs(pb.expected_distinct_out(n, hist, B) - sim) < 0.01 * B, B
    assert pb.expected_distinct_out(n, {1: n}, 500) == pytest.approx(500)   # distinct input: exactly B


def test_occupied_voxels_counts_voxels_of_the_authors_step_inside_the_range():
    pcr = [-75.2, -75.2, -2, 75.2, 75.2, 4]
    xyz = np.array([[0.01, 0.01, 0.01], [0.02, 0.05, 0.10], [0.11, 0.0, 0.0], [0.0, 0.0, 0.16],
                    [80.0, 0.0, 0.0]], dtype=np.float32)          # two in one voxel; one out of range
    assert pb.occupied_voxels(xyz, pcr) == 3
    assert pb.occupied_voxels(np.zeros((0, 3), np.float32), pcr) == 0


def test_summarize_derives_budget_fractions_and_copies():
    rows = [dict(n_in=n, mult={1: n}, vox=n // 2) for n in range(20000, 30000, 10)]   # 1,000 frames
    s = pb.summarize(rows)
    p99 = float(np.percentile([r['n_in'] for r in rows], 99))
    assert s['pb_budget'] == pb.pb_budget(p99) == 32768
    assert s['n_in']['max'] == 29990
    assert s['pb']['subsampled_share'] == 0.0                     # nothing above 32,768
    assert s['old']['subsampled_share'] == 1.0                    # 16,384 subsamples every frame
    assert s['old']['real_point_fraction_median'] == pytest.approx(16384 / np.median([r['n_in'] for r in rows]),
                                                                  rel=1e-3)
    assert s['pb']['copies_per_distinct_median'] == pytest.approx(32768 / np.median([r['n_in'] for r in rows]),
                                                                 rel=1e-3)
    s2 = pb.summarize(rows, budget=24576)                         # an explicit budget: share above it
    assert s2['pb']['subsampled_share'] == pytest.approx(np.mean([r['n_in'] > 24576 for r in rows]))


def test_budget_step_is_found_and_must_follow_the_range_mask():
    def step(name):
        def f(data_dict=None, config=None):
            return data_dict
        f.__name__ = name
        return partial(f, config=EasyDict(NUM_POINTS=dict(train=16384, test=16384)))
    q = [step('mask_points_and_boxes_outside_range'), step('sample_points_hist_based'), step('sample_points'),
         step('drop_empty_gt_boxes'), step('shuffle_points')]
    at = pb.budget_step_index(q, 'train')
    assert at == 2
    assert pb.check_range_mask_first(q, at) == ['mask_points_and_boxes_outside_range', 'sample_points_hist_based']
    with pytest.raises(RuntimeError):
        pb.check_range_mask_first([step('shuffle_points'), step('sample_points')], 1)
    off = [step('mask_points_and_boxes_outside_range'),
           partial(step('sample_points').func, config=EasyDict(NUM_POINTS=dict(train=-1, test=-1)))]
    with pytest.raises(RuntimeError):
        pb.budget_step_index(off, 'train')
    vox = [step('mask_points_and_boxes_outside_range'), step('shuffle_points'), step('sample_points_by_voxels')]
    assert pb.budget_step_index(vox, 'test') == 2
