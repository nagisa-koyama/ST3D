"""Pins the density calibration of a source whose processor ends in a FIXED point budget (`sample_points`, IA-SSD).

`compute_range_histogram` measures through `dataset[i]`, i.e. after the whole processor. With a 16-point budget on
both sides, both histograms would read 16 points per frame whatever the sensors deliver, and the correction would
compare two clouds the budget had already equalised. `HIST_DIST_BEFORE_POINT_BUDGET` (`skip_point_budget`) measures
with the budget removed; train.py refuses the combination without it. Also pins the padding fallback: a frame with
fewer than half the budget used to raise, and every other frame must be padded exactly as before
(experiments_md 20261005_01 §15.8).
"""
import numpy as np
import pytest
from easydict import EasyDict

from pcdet.datasets.point_calibration import (_point_budget_off, compute_range_histogram, link_point_calibration,
                                              point_budget_after_correction)
from pcdet.datasets.processor.data_processor import DataProcessor

PCR = np.array([-75.2, -75.2, -2, 75.2, 75.2, 4], dtype=np.float32)
BUDGET = 16


def _processor(training, with_correction):
    cfgs = []
    if with_correction:
        cfgs.append(EasyDict(NAME='sample_points_hist_based', MIN_HIST_BIN_FRACTION=0.0))
    cfgs.append(EasyDict(NAME='sample_points', NUM_POINTS={'train': BUDGET, 'test': BUDGET}))
    return DataProcessor(cfgs, point_cloud_range=PCR, training=training, num_point_features=3)


class _Frames:
    """dataset[i] runs the real processor over a fixed cloud, as DatasetTemplate.prepare_data does."""

    def __init__(self, clouds, training, with_correction):
        self.clouds = clouds
        self.data_processor = _processor(training, with_correction)

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, i):
        return self.data_processor.forward({'points': self.clouds[i].copy()})


def _cloud(n, seed, r_lo=5.0, r_hi=70.0):
    rng = np.random.default_rng(seed)
    az = rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(r_lo, r_hi, n)
    return np.stack([r * np.cos(az), r * np.sin(az), np.zeros(n)], axis=1).astype(np.float32)


def test_budget_equalises_the_measurement_unless_removed():
    dense = _Frames([_cloud(400, s) for s in range(3)], training=True, with_correction=True)
    hist_on = compute_range_histogram(dense, num_frames=3, num_bins=10, max_dist=75.0)
    assert hist_on.sum() == pytest.approx(BUDGET)
    with _point_budget_off(dense):
        hist_off = compute_range_histogram(dense, num_frames=3, num_bins=10, max_dist=75.0)
    assert hist_off.sum() == pytest.approx(400)


def test_queue_is_restored_and_only_the_budget_leaves_it():
    ds = _Frames([_cloud(100, 0)], training=True, with_correction=True)
    before = list(ds.data_processor.data_processor_queue)
    with _point_budget_off(ds):
        names = [p.func.__name__ for p in ds.data_processor.data_processor_queue]
        assert names == ['sample_points_hist_based']
    assert ds.data_processor.data_processor_queue == before
    assert len(ds[0]['points']) == BUDGET


def test_link_measures_the_sensors_and_thins_to_the_target_density():
    # source 4x denser than the target everywhere; with the budget measured the rate would be ~1
    src = _Frames([_cloud(800, s) for s in range(4)], training=True, with_correction=True)
    tgt = _Frames([_cloud(200, 10 + s) for s in range(4)], training=False, with_correction=False)
    hs, ht = link_point_calibration(src, tgt, num_frames=4, num_bins=10, max_dist=75.0, skip_point_budget=True)
    assert hs.sum() == pytest.approx(800) and ht.sum() == pytest.approx(200)
    rate = src.data_processor.per_bin_sample_rate()
    assert np.all(rate[hs > 0] < 0.5)
    # the installed correction then thins BEFORE the budget: ~200 points survive into a 16-point sample
    with _point_budget_off(src):
        kept = np.mean([len(src[i]['points']) for i in range(4)])
    assert 150 < kept < 250


def test_default_measurement_is_unchanged():
    src = _Frames([_cloud(800, s) for s in range(4)], training=True, with_correction=True)
    tgt = _Frames([_cloud(200, 10 + s) for s in range(4)], training=False, with_correction=False)
    hs, ht = link_point_calibration(src, tgt, num_frames=4, num_bins=10, max_dist=75.0)
    assert hs.sum() == pytest.approx(BUDGET) and ht.sum() == pytest.approx(BUDGET)


def test_guard_detects_a_budget_after_the_correction_only():
    hb = {'NAME': 'sample_points_hist_based'}
    sp = {'NAME': 'sample_points'}
    assert point_budget_after_correction([{'NAME': 'mask_points_and_boxes_outside_range'}, hb, sp])
    assert not point_budget_after_correction([sp, hb])
    assert not point_budget_after_correction([sp])
    assert not point_budget_after_correction([hb, {'NAME': 'transform_points_to_voxels'}])
    assert not point_budget_after_correction(None)


@pytest.mark.parametrize('n', [3, 7, 8, 12, 16, 40])
def test_padding_never_raises_and_keeps_every_point(n):
    proc = _processor(training=True, with_correction=False)
    pts = _cloud(n, n)
    out = proc.forward({'points': pts.copy()})['points']
    assert len(out) == BUDGET
    uniq = np.unique(out, axis=0)
    if n <= BUDGET:
        assert len(uniq) == n                                   # every original point is present
    if BUDGET // 2 <= n <= BUDGET:                               # upstream path: no point more than twice
        _, counts = np.unique(out, axis=0, return_counts=True)
        assert counts.max() <= 2


def test_iassd_waymo2nuscenes_configs_measure_before_the_budget_and_recount_after_it(monkeypatch):
    """The three IA-SSD replication arms (20261005_01 §15.8): every arm recounts after the budget; the thinning arm
    lists the budget after the correction and so must measure without it."""
    import os
    from pcdet.config import cfg_from_yaml_file
    tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
    monkeypatch.chdir(tools)
    for arm in ('sourceonly', 'global-toponly-recount', 'ringpattern'):
        cfg = cfg_from_yaml_file('cfgs/da-ieee-access-pointrcnn/iassd-%s-waymo2nuscenes.yaml' % arm, EasyDict())
        names = [p.NAME for p in cfg.DATA_CONFIG.DATA_PROCESSOR]
        assert names.index('drop_empty_gt_boxes') == names.index('sample_points') + 1, arm
        assert cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG.mean_size[0] == [4.80, 2.10, 1.78]  # Waymo source
        if point_budget_after_correction(cfg.DATA_CONFIG.DATA_PROCESSOR):
            assert cfg.DATA_CONFIG.get('HIST_DIST_BEFORE_POINT_BUDGET', False), arm
    c1 = cfg_from_yaml_file('cfgs/da-ieee-access-pointrcnn/iassd-global-toponly-recount-waymo2nuscenes.yaml', EasyDict())
    assert point_budget_after_correction(c1.DATA_CONFIG.DATA_PROCESSOR) and list(c1.DATA_CONFIG.LIDAR_INDICES) == [0]
