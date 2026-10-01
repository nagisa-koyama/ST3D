"""Pins the cone-restricted density histograms (DATA_CONFIG.HIST_DIST_FOV_DEGREE).

A radial histogram pooled over azimuth compares a cone-limited sensor against the whole 360 degrees
of the other. The fixture reproduces the PandaSet pair in miniature: a 'spin' cloud spread evenly
over 360 degrees and a 'flash' cloud with every point inside +-30 degrees but four times denser
there. Pooled, the flash side reads SPARSER than spin; inside the cone it is denser - the direction
of the correction flips, which is what experiments_md/20260928_02 found on the real sensors.
"""
import numpy as np
import pytest

from pcdet.datasets.point_calibration import compute_range_histogram, cone_mask


class _Frames:
    def __init__(self, clouds):
        self.clouds = clouds

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, i):
        return {'points': self.clouds[i]}


def _cloud(n, az_lo, az_hi, seed):
    rng = np.random.default_rng(seed)
    az = np.radians(rng.uniform(az_lo, az_hi, n))
    r = rng.uniform(5, 70, n)
    return np.stack([r * np.cos(az), r * np.sin(az), np.zeros(n), np.ones(n)], axis=1).astype(np.float32)


SPIN = _Frames([_cloud(36000, -180, 180, s) for s in range(4)])    # 100 pts/deg
FLASH = _Frames([_cloud(24000, -30, 30, 10 + s) for s in range(4)])  # 400 pts/deg, cone only


def test_cone_mask_wraps_across_180_degrees():
    pts = np.array([[-10, 0.1, 0], [-10, -0.1, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float32)
    assert cone_mask(pts, 60, heading_degree=180).tolist() == [True, True, False, False]
    assert cone_mask(pts, 60, heading_degree=0).tolist() == [False, False, True, False]


def test_pooled_histogram_calls_the_flash_side_sparser():
    spin = compute_range_histogram(SPIN, num_frames=4, num_bins=13, max_dist=70)
    flash = compute_range_histogram(FLASH, num_frames=4, num_bins=13, max_dist=70)
    assert flash.sum() / spin.sum() == pytest.approx(24000 / 36000, rel=0.02)


def test_cone_histogram_calls_it_denser_by_the_true_factor():
    kw = dict(num_frames=4, num_bins=13, max_dist=70, fov_degree=60, fov_heading=0.0)
    spin = compute_range_histogram(SPIN, **kw)
    flash = compute_range_histogram(FLASH, **kw)
    assert flash.sum() / spin.sum() == pytest.approx(4.0, rel=0.05)
    live = spin > 0
    assert np.all(flash[live] / spin[live] > 2.5)      # every populated bin, not just the total


def test_no_fov_is_the_pooled_measurement_unchanged():
    a = compute_range_histogram(SPIN, num_frames=4, num_bins=13, max_dist=70)
    b = compute_range_histogram(SPIN, num_frames=4, num_bins=13, max_dist=70, fov_degree=None)
    np.testing.assert_array_equal(a, b)


# --- calibration_target_config: which target split the density correction is measured on ---------
from easydict import EasyDict

from pcdet.datasets.point_calibration import calibration_target_config


def _target_cfg(**extra):
    return EasyDict(dict(DATA_SPLIT={'train': 'train', 'test': 'val'},
                         INFO_PATH={'train': ['x_train.pkl'], 'test': ['x_val.pkl']}, **extra))


def test_default_is_the_train_split():
    cfg, split = calibration_target_config(_target_cfg())
    assert split == 'train' and cfg.DATA_SPLIT.test == 'train' and cfg.INFO_PATH.test == ['x_train.pkl']


def test_test_split_reproduces_the_historical_behaviour():
    cfg, split = calibration_target_config(_target_cfg(HIST_DIST_TARGET_SPLIT='test'))
    assert split == 'test' and cfg.DATA_SPLIT.test == 'val' and cfg.INFO_PATH.test == ['x_val.pkl']


def test_train_split_points_the_eval_view_at_train_and_leaves_the_original_alone():
    original = _target_cfg(HIST_DIST_TARGET_SPLIT='train')
    cfg, split = calibration_target_config(original)
    assert split == 'train'
    assert cfg.DATA_SPLIT.test == 'train' and cfg.INFO_PATH.test == ['x_train.pkl']
    assert original.DATA_SPLIT.test == 'val' and original.INFO_PATH.test == ['x_val.pkl']


def test_unknown_split_refuses():
    with pytest.raises(ValueError):
        calibration_target_config(_target_cfg(HIST_DIST_TARGET_SPLIT='val'))


# ---------------------------------------------------------------- augmentation during the measurement

class _Augmentor:
    def __init__(self, fns):
        self.data_augmentor_queue = list(fns)


class _Processor:
    def set_hist_dist(self, src, tgt, max_dist):
        self.src, self.tgt = src, tgt

    def per_bin_sample_rate(self):
        return np.ones_like(self.src)


class _AugmentedFrames(_Frames):
    """A training-mode source: __getitem__ runs the augmentor's queue, as prepare_data does."""

    def __init__(self, clouds, fns):
        super().__init__(clouds)
        self.data_augmentor = _Augmentor(fns)
        self.data_processor = _Processor()
        self.calls = 0

    def __getitem__(self, i):
        d = {'points': self.clouds[i].copy()}
        for fn in self.data_augmentor.data_augmentor_queue:
            self.calls += 1
            d = fn(d)
        return d


def _turn_around(d):  # a world flip / large rotation: the forward cone now points backwards
    d['points'][:, :2] *= -1
    return d


def test_cone_calibration_measures_a_rotated_source_unrotated():
    """Job 27001: augmentation turned PandarGT's cone away from heading 0 before it was counted, so
    the flash source read ~1/3 of its points and the thinning was a no-op (20261001_03 section 5c)."""
    from pcdet.datasets.point_calibration import link_point_calibration
    src = _AugmentedFrames(FLASH.clouds, [_turn_around])
    link_point_calibration(src, SPIN, num_frames=4, fov_degree=60.0)
    expected = compute_range_histogram(FLASH, num_frames=4, fov_degree=60.0)
    np.testing.assert_allclose(src.data_processor.src, expected)
    assert src.calls == 0, 'the augmentation ran during a cone-restricted measurement'
    assert src.data_augmentor.data_augmentor_queue == [_turn_around], 'the queue was not restored'


def test_pooled_calibration_keeps_measuring_through_the_augmentation():
    """Without a cone every existing row measures exactly as before, augmentation included."""
    from pcdet.datasets.point_calibration import link_point_calibration
    src = _AugmentedFrames(FLASH.clouds, [_turn_around])
    link_point_calibration(src, SPIN, num_frames=4)
    assert src.calls == 4
    np.testing.assert_allclose(src.data_processor.src, compute_range_histogram(FLASH, num_frames=4))
