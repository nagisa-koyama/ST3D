"""Progressive augmentation must survive per-class and asymmetric intervals.

Job 25945 trained 8 epochs (10h58m) and died at the first `PROG_AUG.UPDATE_AUG` epoch:

    data_augmentor.py:199  assert np.isclose(flag - origin_intensity_list[0], ...)
    KeyError: 0

`SCALE_UNIFORM_NOISE` became a PER-CLASS dict when ROS and SN went per-class (577a3f8):
`{'Car': [0.85, 1.20], 'Pedestrian': [0.80, 1.25]}`. The old code indexed it as a list. With
exactly two classes `len(origin) == 2` passed BY COINCIDENCE, and the next line raised on `[0]`.

Two defects, not one. The symmetry assert is also wrong for these configs: every per-class ROS
interval derived from the data is deliberately asymmetric - Car is 0.15 below 1 and 0.20 above,
because every source in this study is a larger-vehicle domain (20260922_03). So even as a flat list
the assert fires, and if it did not, rebuilding the interval from its upper half alone would
silently symmetrise it and throw away the asymmetry the derivation exists to express.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.augmentor.data_augmentor import DataAugmentor  # noqa: E402


def _adjust(name, key, value, intensity):
    aug = DataAugmentor.__new__(DataAugmentor)          # no dataset needed for this method
    cfg = EasyDict({'NAME': name, key: value})
    return aug.adjust_augment_intensity(cfg, intensity)[key]


def test_per_class_dict_is_scaled_per_class():
    """The exact shape of job 25945."""
    out = _adjust('random_object_scaling', 'SCALE_UNIFORM_NOISE',
                  {'Car': [0.85, 1.20], 'Pedestrian': [0.80, 1.25]}, intensity=1.1)
    assert set(out) == {'Car', 'Pedestrian'}
    assert out['Car'] == pytest.approx([1 - 0.15 * 1.1, 1 + 0.20 * 1.1])
    assert out['Pedestrian'] == pytest.approx([1 - 0.20 * 1.1, 1 + 0.25 * 1.1])


def test_asymmetry_is_preserved_not_symmetrised():
    """Car is 0.15 below 1 and 0.20 above; the ratio must survive scaling."""
    out = _adjust('random_object_scaling', 'SCALE_UNIFORM_NOISE', {'Car': [0.85, 1.20]}, 2.0)['Car']
    below, above = 1 - out[0], out[1] - 1
    assert not np.isclose(below, above), 'the interval was symmetrised'
    assert above / below == pytest.approx(0.20 / 0.15)


def test_a_symmetric_flat_list_is_unchanged_in_behaviour():
    """No previously working config may change: scaling each side independently is identical when
    the interval is already symmetric."""
    out = _adjust('random_world_scaling', 'WORLD_SCALE_RANGE', [0.95, 1.05], 1.1)
    assert out == pytest.approx([1 - 0.05 * 1.1, 1 + 0.05 * 1.1])


def test_rotation_anchors_at_zero_not_one():
    out = _adjust('random_world_rotation', 'WORLD_ROT_ANGLE', [-0.7853, 0.7853], 0.5)
    assert out == pytest.approx([-0.7853 * 0.5, 0.7853 * 0.5])


def test_asymmetric_rotation_is_also_preserved():
    out = _adjust('random_object_rotation', 'ROT_UNIFORM_NOISE', [-0.2, 0.6], 0.5)
    assert out == pytest.approx([-0.1, 0.3])


def test_intensity_one_is_the_identity():
    for name, key, val in [('random_object_scaling', 'SCALE_UNIFORM_NOISE', {'Car': [0.85, 1.2]}),
                           ('random_world_scaling', 'WORLD_SCALE_RANGE', [0.95, 1.05])]:
        out = _adjust(name, key, val, 1.0)
        if isinstance(val, dict):
            assert out['Car'] == pytest.approx(val['Car'])
        else:
            assert out == pytest.approx(val)


def test_an_unlisted_augmentation_is_returned_untouched():
    aug = DataAugmentor.__new__(DataAugmentor)
    cfg = EasyDict({'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']})
    assert aug.adjust_augment_intensity(cfg, 1.5) is cfg


def test_a_malformed_interval_is_refused():
    with pytest.raises(AssertionError):
        _adjust('random_world_scaling', 'WORLD_SCALE_RANGE', [0.9, 1.0, 1.1], 1.1)
