"""Pins the two opt-in calibration keys of the P->N count-matched control (experiments_md 20261003_05 §5 item 29).

HIST_DIST_IN_GRID counts both histograms inside the voxel grid, z included: the processor's range mask cuts x and y
only, so without it the calibration matches points the detector never sees (item 28). HIST_DIST_REFERENCE_CFG matches the
source to another training arm's SOURCE instead of the target view. With both absent, the measurement, the reference and
the rates are those of every earlier config.
"""
import os
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file
from pcdet.datasets.point_calibration import (calibration_reference, compute_range_histogram, link_point_calibration,
                                              points_in_grid)
from pcdet.datasets.processor.data_processor import DataProcessor

TOOLS_DIR = Path(__file__).resolve().parent.parent / 'tools'
PCR = np.array([-75.2, -75.2, -2, 75.2, 75.2, 4], dtype=np.float32)


class _Frames:
    """dataset[i] runs the real processor (the correction alone) over a fixed cloud."""

    def __init__(self, clouds, training=True, pcr=PCR):
        self.clouds = clouds
        self.point_cloud_range = pcr
        self.data_processor = DataProcessor([EasyDict(NAME='sample_points_hist_based', MIN_HIST_BIN_FRACTION=0.0)],
                                            point_cloud_range=pcr, training=training, num_point_features=3)

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, i):
        return self.data_processor.forward({'points': self.clouds[i].copy()})


def _ring(n, r, z, seed):
    az = np.random.default_rng(seed).uniform(-np.pi, np.pi, n)
    return np.stack([r * np.cos(az), r * np.sin(az), np.full(n, z)], axis=1).astype(np.float32)


def test_points_in_grid_cuts_z_as_the_voxel_generator():
    p = np.array([[50, 0, 0], [50, 0, 4.0], [50, 0, 9], [50, 0, -2.0], [50, 0, -2.5], [75.3, 0, 0], [-75.1, 0, 0]],
                 np.float32)
    assert points_in_grid(p, PCR)[:, 2].tolist() == [0.0, -2.0, 0.0]


def test_histogram_counts_every_point_by_default_and_the_grid_only_on_request():
    # 30 points per frame at 20 m inside the grid, 70 above it (z = 8 m), as nuScenes' upward beams far out
    ds = _Frames([np.concatenate([_ring(30, 20.0, 0.0, s), _ring(70, 20.0, 8.0, 10 + s)]) for s in range(3)])
    assert compute_range_histogram(ds, num_frames=3, num_bins=10, max_dist=75.0).sum() == pytest.approx(100)
    assert compute_range_histogram(ds, num_frames=3, num_bins=10, max_dist=75.0, pc_range=PCR).sum() == \
        pytest.approx(30)


def test_absent_keys_keep_the_target_view_and_every_point():
    src, tgt = _Frames([_ring(10, 20.0, 0.0, 0)]), _Frames([_ring(10, 20.0, 0.0, 1)], training=False)

    def never(*a, **k):
        raise AssertionError('no reference config was named')
    assert calibration_reference(EasyDict(HIST_DIST_ON_THE_FLY=True), src, tgt, build_reference=never) == (tgt, None)


def test_reference_cfg_replaces_the_target_view_and_in_grid_takes_the_source_range():
    src, tgt = _Frames([_ring(10, 20.0, 0.0, 0)]), _Frames([_ring(10, 20.0, 0.0, 1)], training=False)
    ref = _Frames([_ring(5, 20.0, 0.0, 2)])
    asked = []

    def build(cfg_file, logger=None):
        asked.append(cfg_file)
        return ref
    view, pc_range = calibration_reference(EasyDict(HIST_DIST_REFERENCE_CFG='cfgs/x.yaml', HIST_DIST_IN_GRID=True),
                                           src, tgt, build_reference=build)
    assert view is ref and asked == ['cfgs/x.yaml'] and np.array_equal(pc_range, PCR)


def test_reference_with_another_range_is_refused():
    src, tgt = _Frames([_ring(10, 20.0, 0.0, 0)]), _Frames([_ring(10, 20.0, 0.0, 1)], training=False)
    other = _Frames([_ring(5, 20.0, 0.0, 2)], pcr=np.array([0, -40, -3, 70.4, 40, 1], np.float32))
    with pytest.raises(AssertionError, match='POINT_CLOUD_RANGE'):
        calibration_reference(EasyDict(HIST_DIST_REFERENCE_CFG='cfgs/x.yaml'), src, tgt,
                              build_reference=lambda cfg_file, logger=None: other)


def test_in_grid_rate_is_the_ratio_of_points_the_detector_sees():
    # source: 100 in-grid + 100 above-grid points at 20 m; reference: 50 in-grid + 0 above. Over every return the rate
    # would be 50 / 200 = 0.25; inside the grid it is 50 / 100 = 0.5.
    src = _Frames([np.concatenate([_ring(100, 20.0, 0.0, s), _ring(100, 20.0, 8.0, 10 + s)]) for s in range(4)])
    ref = _Frames([_ring(50, 20.0, 0.0, 20 + s) for s in range(4)])
    link_point_calibration(src, ref, num_frames=4, num_bins=10, max_dist=75.0)
    assert src.data_processor.per_bin_sample_rate()[2] == pytest.approx(0.25)
    src.data_processor.set_hist_dist(None, None)
    link_point_calibration(src, ref, num_frames=4, num_bins=10, max_dist=75.0, pc_range=PCR)
    assert src.data_processor.per_bin_sample_rate()[2] == pytest.approx(0.5)


def test_control_config_is_the_density_twin_plus_the_two_keys():
    prev = os.getcwd()
    os.chdir(TOOLS_DIR)
    try:
        twin, ctrl = EasyDict(), EasyDict()
        cfg_from_yaml_file('cfgs/da-ieee-access/centerpoint-global-fovcut-recount-pandaset2nuscenes.yaml', twin)
        cfg_from_yaml_file('cfgs/da-ieee-access/centerpoint-global-fovcut-recount-latcount-pandaset2nuscenes.yaml', ctrl)
        lattice = EasyDict()
        cfg_from_yaml_file(ctrl.DATA_CONFIG.HIST_DIST_REFERENCE_CFG, lattice)
    finally:
        os.chdir(prev)
    assert ctrl.DATA_CONFIG.pop('HIST_DIST_REFERENCE_CFG') == \
        'cfgs/da-ieee-access/centerpoint-ringpattern-pandaset2nuscenes.yaml'
    assert ctrl.DATA_CONFIG.pop('HIST_DIST_IN_GRID') is True
    assert ctrl.pop('_BASE_CONFIG_') != twin.pop('_BASE_CONFIG_')   # the child points at the twin, the twin at its base
    assert ctrl == twin
    assert list(lattice.DATA_CONFIG.POINT_CLOUD_RANGE) == list(twin.DATA_CONFIG.POINT_CLOUD_RANGE)
    assert 'RING_PATTERN' in lattice.DATA_CONFIG and 'RING_PATTERN' not in twin.DATA_CONFIG
