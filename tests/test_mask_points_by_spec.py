"""`mask_points_by_spec` (eval-time sensor-spec cuts, experiments_md 20261008_05) and the Waymo
`LIDAR_INDICES` slice: elevation / radius geometry about the given sensor origin, the empty-cloud and
all-cut sentinels, boxes untouched, and the per-lidar slice taken BEFORE the NLZ filter."""
import sys
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import _init_path  # noqa: F401,E402
from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402

RANGE = np.array([-75.2, -75.2, -2, 75.2, 75.2, 4], dtype=np.float32)


def _proc(**spec):
    cfg = [EasyDict(NAME='mask_points_by_spec', **spec)]
    return DataProcessor(cfg, point_cloud_range=RANGE, training=False, num_point_features=3).forward


def _cloud():
    # points at 20 m horizontal from a sensor at (0, 0, 1.75): elevations -5, 0, +5 deg, and one at 1 m radius
    origin = np.array([0.0, 0.0, 1.75])
    rows = []
    for el in (-5.0, 0.0, 5.0):
        rows.append([20.0, 0.0, 1.75 + 20.0 * np.tan(np.radians(el))])
    rows.append([1.0, 0.0, 1.75])
    return np.asarray(rows, dtype=np.float32), origin


def test_elevation_max_cuts_only_above():
    pts, origin = _cloud()
    out = _proc(SENSOR_ORIGIN=origin.tolist(), ELEVATION_MAX_DEG=2.4)({'points': pts.copy()})
    assert len(out['points']) == 3 and not np.any(np.isclose(out['points'][:, 2], pts[2, 2]))


def test_elevation_min_cuts_only_below():
    pts, origin = _cloud()
    out = _proc(SENSOR_ORIGIN=origin.tolist(), ELEVATION_MIN_DEG=-2.0)({'points': pts.copy()})
    assert len(out['points']) == 3 and not np.any(np.isclose(out['points'][:, 2], pts[0, 2]))


def test_min_radius_cuts_near_point_and_leaves_boxes():
    pts, origin = _cloud()
    boxes = np.zeros((2, 8), dtype=np.float32)
    out = _proc(SENSOR_ORIGIN=origin.tolist(), MIN_RADIUS_M=1.5)({'points': pts.copy(), 'gt_boxes': boxes})
    assert len(out['points']) == 3 and out['gt_boxes'].shape == (2, 8)


def test_origin_matters():
    pts, _ = _cloud()
    # with the sensor 1 m higher every 0-deg point is below the horizon, so ELEVATION_MIN 0 cuts them
    out = _proc(SENSOR_ORIGIN=[0.0, 0.0, 2.75], ELEVATION_MIN_DEG=0.0)({'points': pts.copy()})
    assert len(out['points']) == 1


@pytest.mark.parametrize('spec', [dict(ELEVATION_MAX_DEG=2.4), dict(MIN_RADIUS_M=1.5)])
def test_empty_cloud_sentinel(spec):
    out = _proc(SENSOR_ORIGIN=[0.0, 0.0, 1.75], **spec)({'points': np.zeros((0, 3), np.float32)})
    assert out['points'].shape == (0, 3)


def test_all_points_cut_gives_empty_not_error():
    pts, origin = _cloud()
    out = _proc(SENSOR_ORIGIN=origin.tolist(), ELEVATION_MAX_DEG=-90.0)({'points': pts.copy()})
    assert out['points'].shape == (0, 3)


def test_no_key_is_identity():
    pts, origin = _cloud()
    out = _proc(SENSOR_ORIGIN=origin.tolist())({'points': pts.copy()})
    assert np.array_equal(out['points'], pts)


def test_waymo_lidar_indices_slices_before_nlz(tmp_path, monkeypatch):
    from pcdet.datasets.waymo.waymo_dataset import WaymoDataset
    # a fake frame file: 4 TOP points (one NLZ), 2 FRONT, 1 each for the other three lidars
    n = [4, 2, 1, 1, 1]
    feats = np.zeros((sum(n), 6), np.float32)
    feats[:, 0] = np.arange(sum(n))       # x encodes the stored index
    feats[:, 5] = -1
    feats[1, 5] = 1                       # a TOP point inside a no-label zone
    seq = tmp_path / 'seq'; seq.mkdir()
    np.save(seq / '0000.npy', feats)
    ds = WaymoDataset.__new__(WaymoDataset)
    ds.data_path = tmp_path
    ds.dataset_cfg = EasyDict(LIDAR_INDICES=[0])
    pts = ds.get_lidar('seq', 0, num_points_of_each_lidar=n)
    assert pts[:, 0].tolist() == [0.0, 2.0, 3.0]  # TOP only, NLZ point dropped after the slice
    ds.dataset_cfg = EasyDict()
    assert len(ds.get_lidar('seq', 0, num_points_of_each_lidar=n)) == sum(n) - 1
    ds.dataset_cfg = EasyDict(LIDAR_INDICES=[0])
    with pytest.raises(AssertionError):
        ds.get_lidar('seq', 0, num_points_of_each_lidar=[3, 2, 1, 1, 1])
