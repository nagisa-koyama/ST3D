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
