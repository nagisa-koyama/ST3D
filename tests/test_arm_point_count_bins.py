"""tools/analysis/arm_point_count_bins.py helpers (experiments_md 20261003_05 §5 item 28).

Pins the 5-m binning (75 m and beyond not counted), the (z, y, x) voxel-index order of the voxel generator, the
0-70 m restriction of the pre-declared reading and its two bands, and that a sparse bin of the target does not decide.
"""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, '..', 'tools', 'analysis', 'arm_point_count_bins.py')
spec = importlib.util.spec_from_file_location('arm_point_count_bins', TOOL)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def test_bins_are_5m_and_stop_before_75():
    xy = np.array([[0.0, 0.0], [4.99, 0.0], [3.0, 4.0], [0.0, -69.9], [74.99, 0.0], [75.0, 0.0], [60.0, 60.0]])
    c = ab.bin_counts(xy)
    assert len(c) == 15
    assert c[0] == 2 and c[1] == 1 and c[13] == 1 and c[14] == 1 and c.sum() == 5


def test_voxel_index_order_is_zyx():
    vs, pcr = [0.1, 0.1, 0.15], np.array([-75.2, -75.2, -2.0, 75.2, 75.2, 4.0])
    coords = np.array([[3, 752, 1252]])          # z, y, x: x index 1252 -> 50.05 m, y index 752 -> 0.05 m
    np.testing.assert_allclose(ab.voxel_xy(coords, vs, pcr), [[50.05, 0.05]], atol=1e-4)


def _s(mean):
    return dict(mean=list(mean) + [999.0])     # the 70-75 m bin must never decide


def test_reading_matched_and_differ():
    t = np.full(14, 100.0)
    assert ab.reading(_s(t), _s(t * 1.05), _s(t))['matched']
    r = ab.reading(_s(t * 0.6), _s(t * 0.95), _s(t))
    assert not r['matched'] and abs(r['frame_ratio'] - 0.6 / 0.95) < 1e-9 and len(r['bins_outside']) == 14


def test_one_bin_outside_band_differs_even_at_matched_total():
    t = np.full(14, 100.0)
    lat = t.copy(); lat[3] = 70.0; lat[4] = 130.0  # total equal, two bins off by 30%
    r = ab.reading(_s(lat), _s(t), _s(t))
    assert abs(r['frame_ratio'] - 1.0) < 1e-9 and r['bins_outside'] == [3, 4] and not r['matched']


def test_sparse_target_bin_does_not_decide():
    t = np.full(14, 100.0); t[13] = 0.5        # < 1% of the target's 0-70 m points
    lat = t.copy(); lat[13] = 5.0
    assert ab.reading(_s(lat), _s(t), _s(t))['matched']
