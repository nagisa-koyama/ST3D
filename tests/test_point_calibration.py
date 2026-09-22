"""Tests for measuring the density-correction histograms at load time.

The shipped hist_dist_*.npy files are raw counts over however many frames an old analysis run
processed, their shape disagrees with a direct measurement of the same data by up to 1.8x per bin,
and they go stale silently whenever MAX_SWEEPS, a platform subset or POINT_CLOUD_RANGE changes.
Measuring from the datasets actually being trained on removes all three problems.

See experiments_md/20260922_02_dataset_and_platform_domain_gap_analysis.md, defect 3.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.point_calibration import (compute_range_histogram,  # noqa: E402
                                              compute_foreground_histograms,
                                              link_foreground_calibration,
                                              link_point_calibration)
from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402


class FakeDataset:
    """Minimal stand-in: the histogram only needs len() and __getitem__()['points']."""

    def __init__(self, radii, num_frames=20, ontology='fake'):
        self.radii = np.asarray(radii, dtype=np.float64)
        self.num_frames = num_frames
        self.dataset_ontology = ontology
        self.data_processor = DataProcessor(
            [], point_cloud_range=np.array([-75.2, -75.2, -2, 75.2, 75.2, 4]),
            training=True, num_point_features=4
        )

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        pts = np.zeros((len(self.radii), 4))
        pts[:, 0] = self.radii                 # all on the +x axis, so |xy| == radius
        return {'points': pts}


def test_histogram_is_per_frame_not_a_raw_count():
    """Ten identical points per frame over any number of frames must give ten, not ten x frames."""
    ds = FakeDataset([10.0] * 10, num_frames=37)
    h = compute_range_histogram(ds, num_frames=37, num_bins=50)
    assert h.sum() == pytest.approx(10.0)


def test_histogram_length_matches_num_bins():
    ds = FakeDataset([5.0, 40.0])
    assert len(compute_range_histogram(ds, num_frames=5, num_bins=25)) == 25


def test_points_land_in_the_expected_bins():
    ds = FakeDataset([1.0, 74.0])
    h = compute_range_histogram(ds, num_frames=4, num_bins=50)       # 1.5 m per bin
    assert h[0] == pytest.approx(1.0)
    assert h[49] == pytest.approx(1.0)
    assert h[1:49].sum() == 0


def test_points_beyond_max_dist_are_clipped_not_dropped():
    """The correction clips distance into the last bin, so the histogram must agree."""
    ds = FakeDataset([200.0])
    h = compute_range_histogram(ds, num_frames=3, num_bins=50)
    assert h[-1] == pytest.approx(1.0)
    assert h.sum() == pytest.approx(1.0)


def test_measurement_is_deterministic():
    ds = FakeDataset([3.0, 20.0, 60.0], num_frames=50)
    a = compute_range_histogram(ds, num_frames=10, num_bins=50)
    b = compute_range_histogram(ds, num_frames=10, num_bins=50)
    assert np.array_equal(a, b)


def test_empty_dataset_raises():
    with pytest.raises(ValueError):
        compute_range_histogram(FakeDataset([1.0], num_frames=0), num_frames=5)


def test_link_installs_the_pair_on_the_source_only():
    src = FakeDataset([10.0] * 4, ontology='src')
    tgt = FakeDataset([10.0] * 12, ontology='tgt')
    hs, ht = link_point_calibration(src, tgt, num_frames=5, num_bins=50)
    assert np.array_equal(src.data_processor.hist_dist_src, hs)
    assert np.array_equal(src.data_processor.hist_dist_tgt, ht)
    # the target is never corrected - its own calibration target is itself
    assert tgt.data_processor.hist_dist_src is None
    assert tgt.data_processor.hist_dist_tgt is None


def test_link_measures_the_two_domains_separately():
    src = FakeDataset([10.0] * 4)
    tgt = FakeDataset([10.0] * 12)
    hs, ht = link_point_calibration(src, tgt, num_frames=5, num_bins=50)
    assert hs.sum() == pytest.approx(4.0)
    assert ht.sum() == pytest.approx(12.0)
    assert ht.sum() / hs.sum() == pytest.approx(3.0)      # target is 3x denser


def test_correction_is_a_noop_until_linked():
    """DataProcessor leaves points alone while either histogram is None - which is what keeps the
    measurement itself non-circular."""
    ds = FakeDataset([10.0] * 6)
    assert ds.data_processor.hist_dist_src is None
    out = ds.data_processor.sample_points_hist_based(data_dict={'points': np.zeros((6, 4))})
    assert len(out['points']) == 6


def _processor(src, tgt):
    p = DataProcessor([], point_cloud_range=np.array([-75.2, -75.2, -2, 75.2, 75.2, 4]),
                      training=True, num_point_features=4)
    p.set_hist_dist(np.asarray(src, dtype=float), np.asarray(tgt, dtype=float))
    return p


def test_rate_is_the_plain_ratio_for_well_populated_bins():
    p = _processor([10, 10, 10, 10], [5, 20, 10, 10])
    assert np.allclose(p.per_bin_sample_rate(), [0.5, 2.0, 1.0, 1.0])


def test_a_zero_source_bin_does_not_drop_every_point():
    """rand() < nan is False, so an unguarded zero source bin silently deletes the whole bin."""
    p = _processor([10, 0, 10, 10], [5, 5, 10, 10])
    rate = p.per_bin_sample_rate()
    assert np.isfinite(rate).all()
    assert rate[1] == 1.0


def test_under_populated_bins_are_left_uncorrected():
    """The 4th bin holds 0.5% of the mean, far too few points to estimate a ratio from."""
    p = _processor([100, 100, 100, 0.02], [50, 50, 50, 0.0001])
    rate = p.per_bin_sample_rate()
    assert np.allclose(rate[:3], 0.5)
    assert rate[3] == 1.0


def test_the_guard_threshold_is_configurable():
    p = _processor([100, 100, 100, 40], [50, 50, 50, 10])
    # mean source bin is 85. Default floor is 1% of that, 0.85, so bin 3 (40) is trusted.
    assert p.per_bin_sample_rate()[3] == pytest.approx(0.25)
    # raising the fraction to 0.5 lifts the floor to 42.5, which now excludes bin 3.
    cfg = EasyDict({'MIN_HIST_BIN_FRACTION': 0.5})
    assert p.per_bin_sample_rate(cfg)[3] == 1.0


def test_the_guard_is_scale_free():
    """Per-frame histograms and raw counts over N frames must give the same rates."""
    a = _processor([100, 100, 100, 0.02], [50, 50, 50, 0.0001]).per_bin_sample_rate()
    b = _processor([100e3, 100e3, 100e3, 20], [50e3, 50e3, 50e3, 0.1]).per_bin_sample_rate()
    assert np.allclose(a, b)


# --------------------------------------------------------------------------------------------
# Foreground-aware calibration
#
# A single per-bin rate scales the points on objects and the points on everything else by the same
# factor, so it cannot change a bin's foreground SHARE - matching the global radial profile leaves
# source objects at sigma_src/sigma_tgt of the target's object density (measured 0.42-0.76x). The
# fix is two channels. The target's boxes come from pseudo-labels, which keeps it UDA-legal.
# --------------------------------------------------------------------------------------------

class FakeBoxDataset(FakeDataset):
    """Adds boxes, and points placed to fall inside or outside them."""

    def __init__(self, fg_radii, bg_radii, box_label=1.0, num_frames=20, ontology='fake'):
        super().__init__(list(fg_radii) + list(bg_radii), num_frames=num_frames, ontology=ontology)
        self.fg_radii, self.bg_radii, self.box_label = list(fg_radii), list(bg_radii), box_label

    def __getitem__(self, index):
        pts = np.zeros((len(self.fg_radii) + len(self.bg_radii), 4))
        pts[:, 0] = self.fg_radii + self.bg_radii
        # one 2x2x2 m box centred on each foreground point; background points sit on the same
        # axis but far enough away in y to be outside every box
        pts[len(self.fg_radii):, 1] = 50.0
        boxes = np.zeros((len(self.fg_radii), 8))
        boxes[:, 0] = self.fg_radii
        boxes[:, 3:6] = 2.0
        boxes[:, 7] = self.box_label
        return {'points': pts, 'gt_boxes': boxes}


def _fg_processor(fg_s, bg_s, fg_t, bg_t):
    p = _processor(np.asarray(fg_s, float) + np.asarray(bg_s, float),
                   np.asarray(fg_t, float) + np.asarray(bg_t, float))
    p.set_foreground_hist(np.asarray(fg_s, float), np.asarray(bg_s, float),
                          np.asarray(fg_t, float), np.asarray(bg_t, float))
    return p


def test_points_in_any_box_flags_only_interior_points():
    pts = np.array([[10.0, 0, 0, 0], [10.0, 50.0, 0, 0]])
    boxes = np.array([[10.0, 0, 0, 2, 2, 2, 0]])
    assert list(DataProcessor.points_in_any_box(pts, boxes)) == [True, False]


def test_points_in_any_box_without_boxes_is_all_background():
    pts = np.zeros((5, 4))
    assert not DataProcessor.points_in_any_box(pts, None).any()
    assert not DataProcessor.points_in_any_box(pts, np.zeros((0, 7))).any()


def test_degenerate_boxes_are_dropped_before_the_geometry_kernel():
    """A zero-extent box is the same failure mode as the open gt_sampling segfault."""
    pts = np.array([[10.0, 0, 0, 0]])
    assert not DataProcessor.points_in_any_box(pts, np.array([[10.0, 0, 0, 0, 0, 0, 0]])).any()


def test_rate_is_computed_per_channel():
    p = _fg_processor(fg_s=[2, 2], bg_s=[10, 10], fg_t=[1, 4], bg_t=[5, 5])
    assert np.allclose(p.per_bin_sample_rate(None, 'fg'), [0.5, 2.0])
    assert np.allclose(p.per_bin_sample_rate(None, 'bg'), [0.5, 0.5])
    # 'all' still compares the whole cloud, and equals the sum of the channels
    assert np.allclose(p.per_bin_sample_rate(None, 'all'), [0.5, 0.75])


def test_foreground_and_background_get_different_rates():
    """fg kept outright, bg dropped outright - so the split is observable exactly."""
    bins = 50
    p = _fg_processor(fg_s=[1.0] * bins, bg_s=[1.0] * bins,
                      fg_t=[1.0] * bins, bg_t=[0.0] * bins)
    pts = np.zeros((2, 4))
    pts[:, 0] = 10.0
    pts[1, 1] = 50.0                                     # second point is outside the box
    boxes = np.array([[10.0, 0, 0, 2, 2, 2, 0]])
    out = p.sample_points_hist_based({'points': pts, 'gt_boxes': boxes})['points']
    assert len(out) == 1 and out[0][1] == 0.0            # the in-box point survived


def test_without_foreground_histograms_behaviour_is_unchanged():
    bins = 50
    p = _processor([1.0] * bins, [1.0] * bins)
    assert p.hist_fg_src is None
    pts = np.zeros((4, 4))
    pts[:, 0] = 10.0
    out = p.sample_points_hist_based({'points': pts, 'gt_boxes': np.zeros((0, 8))})['points']
    assert len(out) == 4                                 # rate 1 everywhere, nothing dropped


def test_foreground_histogram_splits_the_cloud_without_losing_points():
    ds = FakeBoxDataset(fg_radii=[10.0, 30.0], bg_radii=[10.0, 50.0, 60.0])
    fg, bg = compute_foreground_histograms(ds, num_frames=5, num_bins=15)
    assert fg.sum() == pytest.approx(2.0)                # per frame, not a raw count
    assert bg.sum() == pytest.approx(3.0)
    assert (fg + bg).sum() == pytest.approx(len(ds.radii))


def test_ignored_pseudo_labels_are_excluded_from_the_foreground_channel():
    """Memory voting demotes a box by negating its label; those must not count as foreground."""
    kept = FakeBoxDataset(fg_radii=[10.0], bg_radii=[50.0], box_label=1.0)
    demoted = FakeBoxDataset(fg_radii=[10.0], bg_radii=[50.0], box_label=-1.0)
    assert compute_foreground_histograms(kept, num_frames=3, num_bins=15)[0].sum() == 1.0
    assert compute_foreground_histograms(demoted, num_frames=3, num_bins=15)[0].sum() == 0.0


def test_link_foreground_installs_both_pairs_on_the_source_only():
    src = FakeBoxDataset(fg_radii=[10.0], bg_radii=[10.0, 20.0], ontology='src')
    tgt = FakeBoxDataset(fg_radii=[10.0, 10.0], bg_radii=[20.0], ontology='tgt')
    link_foreground_calibration(src, tgt, num_frames=5, num_bins=15)
    assert src.data_processor.hist_fg_src is not None
    assert tgt.data_processor.hist_fg_src is None
    # the whole-cloud pair is installed too, as the exact sum of the channels
    p = src.data_processor
    assert np.allclose(p.hist_dist_src, p.hist_fg_src + p.hist_bg_src)
    assert np.allclose(p.hist_dist_tgt, p.hist_fg_tgt + p.hist_bg_tgt)


def test_link_foreground_recovers_the_direction_a_uniform_rate_cannot():
    """Source foreground-poor vs target foreground-rich: the fg rate must exceed the bg rate."""
    src = FakeBoxDataset(fg_radii=[10.0], bg_radii=[10.0] * 9)      # 10% foreground
    tgt = FakeBoxDataset(fg_radii=[10.0] * 5, bg_radii=[10.0] * 5)  # 50% foreground
    link_foreground_calibration(src, tgt, num_frames=5, num_bins=15)
    p = src.data_processor
    b = int(10.0 / 75.0 * 15)
    assert p.per_bin_sample_rate(None, 'fg')[b] > p.per_bin_sample_rate(None, 'bg')[b]
