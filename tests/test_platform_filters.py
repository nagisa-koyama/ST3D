"""Tests for the per-platform subset filters on Lyft and nuScenes.

Both datasets mix two capture platforms, and neither was selectable before:

  * Lyft ships two vehicle configurations - BETA_V0 with 40-beam lidars and BETA_PLUS_PLUS with
    64-beam ones. The split is host-pure (`host-a101`/`host-a102` are the 64-beam vehicles) and
    also shows up as a bimodal point count, ~65k vs ~108k per frame, with nothing in between.
    Train is 82% 40-beam / 18% 64-beam, so an unfiltered Lyft source is predominantly 40-beam.
  * nuScenes was captured by two vehicles whose mapping to city is exact: n008 is every
    boston-seaport frame, n015 every singapore one. The LIDAR_TOP mounting is the same on both
    (1.84 m), so selecting one isolates a sensor-identical, geography-different subset.

Both filters are no-ops when their config key is absent, so existing configs are unaffected.
See experiments_md/20260922_01 for the measurements behind the host list.
"""
import json
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.lyft.lyft_dataset import LyftDataset  # noqa: E402
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset  # noqa: E402


class _Log:
    """Minimal stand-in that carries just the attributes the filters touch."""

    def info(self, *args, **kwargs):
        pass


def _lyft(cfg):
    ds = object.__new__(LyftDataset)
    ds.dataset_cfg = EasyDict(cfg)
    ds.logger = _Log()
    return ds


def _lyft_infos():
    return [{'lidar_path': 'lidar/%s_lidar1_%d.bin' % (host, i)}
            for i, host in enumerate(['host-a004', 'host-a101', 'host-a011', 'host-a102'])]


def test_lyft_filter_is_a_noop_without_the_key():
    infos = _lyft_infos()
    assert _lyft({}).filter_by_lidar_config(infos) == infos
    assert _lyft({'LIDAR_CONFIG': 'all'}).filter_by_lidar_config(infos) == infos


@pytest.mark.parametrize('cfg,expected_hosts', [
    (64, ['host-a101', 'host-a102']),
    (40, ['host-a004', 'host-a011']),
    ('64', ['host-a101', 'host-a102']),
])
def test_lyft_filter_selects_the_right_hosts(cfg, expected_hosts):
    kept = _lyft({'LIDAR_CONFIG': cfg}).filter_by_lidar_config(_lyft_infos())
    assert [Path(i['lidar_path']).name.split('_')[0] for i in kept] == expected_hosts


def test_lyft_filter_partitions_exactly():
    """The two subsets must be disjoint and together cover everything."""
    infos = _lyft_infos()
    a = _lyft({'LIDAR_CONFIG': 40}).filter_by_lidar_config(infos)
    b = _lyft({'LIDAR_CONFIG': 64}).filter_by_lidar_config(infos)
    assert len(a) + len(b) == len(infos)
    assert not ({i['lidar_path'] for i in a} & {i['lidar_path'] for i in b})


def test_lyft_filter_rejects_a_bad_value():
    with pytest.raises(AssertionError):
        _lyft({'LIDAR_CONFIG': 32}).filter_by_lidar_config(_lyft_infos())


def _nuscenes(cfg, tmp_path, mapping):
    """mapping: sample token -> (vehicle, location); written as real metadata json."""
    version = cfg.get('VERSION', 'v1.0-trainval')
    meta = tmp_path / version
    meta.mkdir(parents=True, exist_ok=True)
    logs, scenes, samples = [], [], []
    for n, (token, (vehicle, location)) in enumerate(mapping.items()):
        logs.append({'token': 'log%d' % n, 'vehicle': vehicle, 'location': location})
        scenes.append({'token': 'scene%d' % n, 'log_token': 'log%d' % n})
        samples.append({'token': token, 'scene_token': 'scene%d' % n})
    (meta / 'log.json').write_text(json.dumps(logs))
    (meta / 'scene.json').write_text(json.dumps(scenes))
    (meta / 'sample.json').write_text(json.dumps(samples))
    ds = object.__new__(NuScenesDataset)
    ds.dataset_cfg = EasyDict(dict(cfg, VERSION=version))
    ds.root_path = tmp_path
    ds.logger = _Log()
    return ds


MAPPING = {
    'sA': ('n008', 'boston-seaport'),
    'sB': ('n015', 'singapore-onenorth'),
    'sC': ('n015', 'singapore-queenstown'),
    'sD': ('n008', 'boston-seaport'),
}
INFOS = [{'token': t} for t in MAPPING]


def test_nuscenes_filter_is_a_noop_without_the_keys(tmp_path):
    ds = _nuscenes({}, tmp_path, MAPPING)
    assert ds.filter_by_platform(INFOS) == INFOS


def test_nuscenes_filter_by_vehicle(tmp_path):
    ds = _nuscenes({'VEHICLE': 'n015'}, tmp_path, MAPPING)
    assert [i['token'] for i in ds.filter_by_platform(INFOS)] == ['sB', 'sC']


def test_nuscenes_filter_by_location(tmp_path):
    ds = _nuscenes({'LOCATION': ['singapore-onenorth', 'singapore-queenstown']}, tmp_path, MAPPING)
    assert [i['token'] for i in ds.filter_by_platform(INFOS)] == ['sB', 'sC']


def test_nuscenes_vehicle_and_location_are_combined(tmp_path):
    ds = _nuscenes({'VEHICLE': 'n015', 'LOCATION': 'singapore-queenstown'}, tmp_path, MAPPING)
    assert [i['token'] for i in ds.filter_by_platform(INFOS)] == ['sC']


def test_nuscenes_vehicles_partition_exactly(tmp_path):
    a = _nuscenes({'VEHICLE': 'n008'}, tmp_path, MAPPING).filter_by_platform(INFOS)
    b = _nuscenes({'VEHICLE': 'n015'}, tmp_path, MAPPING).filter_by_platform(INFOS)
    assert len(a) + len(b) == len(INFOS)
    assert not ({i['token'] for i in a} & {i['token'] for i in b})


def test_nuscenes_empty_selection_raises(tmp_path):
    ds = _nuscenes({'VEHICLE': 'n099'}, tmp_path, MAPPING)
    with pytest.raises(AssertionError):
        ds.filter_by_platform(INFOS)
