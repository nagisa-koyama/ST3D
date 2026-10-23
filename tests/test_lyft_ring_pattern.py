"""CPU tests for RING_PATTERN / KEEP_FRAMES on the Lyft loader (pcdet/datasets/lyft/lyft_ring_pattern.py, lyft_dataset.py)
and the cache tool that feeds them (tools/analysis/lyft_ring_cache.py); experiments_md 20261024_01.

Pins, per the bug-prevention checklist:
- cache alignment: the tool's entry for a synthetic .bin is accepted by the loader's raw cloud, and its ring ids are
  the true lasers; a cloud that is not the cached one (row count, one coordinate, ring count) is refused;
- the thinning: one point per kept ring per azimuth bin (the first in stored order), and the kept rings are the rings
  nearest the 1.33 deg lattice inside the recovered range (brute force);
- order: the mask is taken on the RAW sensor-frame scan, before SHIFT_COOR, so the shift does not change it;
- 64-beam frames never enter (LIDAR_CONFIG + KEEP_FRAMES), and the key refuses 64-beam / accumulated / unlisted use;
- sentinels: labels pass the loader thinning untouched; a box the pattern empties is dropped by the zero-point filter
  (the recount) and never reaches the model as a positive; empty / degenerate cache entries are refused;
- absent keys and evaluation mode leave the loader unchanged.
"""
import copy
import os
import pickle
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))
sys.path.insert(0, str(ROOT / 'tests'))
import _init_path  # noqa: F401,E402
import lyft_ring_cache as T  # noqa: E402  (the cache tool)
import lyft_deskew_rings as ldr  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets.lyft import lyft_rings as lr  # noqa: E402
from pcdet.datasets.lyft import lyft_ring_pattern as RP  # noqa: E402
from pcdet.datasets.lyft.lyft_dataset import LyftDataset  # noqa: E402
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils  # noqa: E402
from test_lyft_rings import synthetic_scan  # noqa: E402

PRM = {'az_start': 6.435, 'period': 0.1, 'ref': 1.0}
# recovered levels of a real 40-beam gate frame (host-a011_lidar1_1232485964401715726, run 2), top laser first
LEV40 = np.array([7.3658, 6.3619, 5.3544, 4.3463, 3.3344, 2.3215, 1.9875, 1.6411, 1.3076, 0.9733, 0.6272, 0.2927,
                  -0.0464, -0.3838, -0.7226, -1.0613, -1.3968, -1.7353, -2.0735, -2.4087, -2.7466, -3.0846, -3.4172,
                  -3.7557, -4.0907, -4.4262, -4.7629, -5.0946, -5.4273, -5.7643, -6.7612, -7.7525, -8.7301, -9.7075,
                  -10.6765, -11.6368, -12.5727, -13.5114, -14.4411, -15.3582])
CFG_P = 'cfgs/da-ieee-access/centerpoint-ringpattern-lyft40-lyft2nuscenes.yaml'
RP_CFG = EasyDict(N_RINGS=40, SPACING_DEG=1.33, AZ_RES_DEG=0.332)


class _Draws:
    """An rng whose draws are given (phase_v, then phase_h)."""
    def __init__(self, *v):
        self.v = list(v)

    def random(self):
        return self.v.pop(0)


def wall_scan(levels, step=0.2, R=20.0, a0=6.0):
    """A standstill laser-major scan: laser k sweeps once from a0 at `levels[k]`, every point at horizontal range R
    (z = R tan el), as a Lyft .bin row [x, y, z, 100, 1]. Returns (rows float32 (N, 5), true ring per row)."""
    az = np.radians(a0 + np.arange(0.0, 360.0, step))
    rows, ring = [], []
    for k, e in enumerate(levels):
        z = R * np.tan(np.radians(e))
        rows.append(np.c_[R * np.cos(az), R * np.sin(az), np.full(len(az), z), np.full(len(az), 100.0),
                          np.ones(len(az))])
        ring.append(np.full(len(az), k))
    return np.concatenate(rows).astype(np.float32), np.concatenate(ring)


def tool_entry(tmp_path, monkeypatch, rows, info_extra=None, name='host-a004_lidar1_1000000000000000001'):
    """Write `rows` as a Lyft .bin and run the CACHE TOOL's own per-frame code on it."""
    (tmp_path / 'lidar').mkdir(exist_ok=True)
    rows.astype(np.float32).tofile(str(tmp_path / 'lidar' / (name + '.bin')))
    info = {'lidar_path': 'lidar/%s.bin' % name, 'token': 'tok_' + name, 'sweeps': []}
    info.update(info_extra or {})
    monkeypatch.setattr(ldr, 'ROOT', str(tmp_path) + '/')
    T.init_worker([info], PRM)
    stem, rec, entry = T.one_frame(0)
    return info, stem, rec, entry


def loader_points(tmp_path, info):
    """LyftDataset.get_lidar_with_sweeps on one info: the raw array the loader thins."""
    ds = object.__new__(LyftDataset)
    ds.root_path, ds.infos, ds._sweep_compensator = Path(tmp_path), [info], None
    return ds.get_lidar_with_sweeps(0, max_sweeps=1)


# ---------------------------------------------------------------- cache alignment


def test_tool_entry_is_accepted_by_the_loader_cloud_and_holds_the_true_rings(tmp_path, monkeypatch):
    stored, ring, _, info = synthetic_scan(n_lasers=40, el_top=7.0, gap=0.55, step=0.25)
    rows = np.c_[stored, np.full(len(stored), 100.0), np.ones(len(stored))]
    info, stem, rec, entry = tool_entry(tmp_path, monkeypatch, rows, info_extra={'sweeps': info['sweeps']})
    pts = loader_points(tmp_path, info)
    assert pts.dtype == np.float32 and pts.shape == (len(rows), 5)
    RP.check_entry(stem, entry, pts, 40)                                     # accepted
    assert T.xyz_crc32(rows.astype(np.float32)) == RP.xyz_crc32(pts) == entry['crc32']
    assert rec['rings'] == 40 and np.mean(RP.ring_ids(entry['counts']) == ring) > 0.999


def test_a_cloud_that_is_not_the_cached_one_is_refused(tmp_path, monkeypatch):
    rows, _ = wall_scan(LEV40)
    info, stem, _, entry = tool_entry(tmp_path, monkeypatch, rows)
    pts = loader_points(tmp_path, info)
    RP.check_entry(stem, entry, pts, 40)
    with pytest.raises(AssertionError, match='no entry'):
        RP.check_entry(stem, None, pts, 40)
    with pytest.raises(AssertionError, match='rows'):
        RP.check_entry(stem, entry, pts[:-1], 40)
    moved = pts.copy()
    moved[123, 1] += 1e-3
    with pytest.raises(AssertionError, match='different cloud'):
        RP.check_entry(stem, entry, moved, 40)
    with pytest.raises(AssertionError, match='recovered rings'):
        RP.check_entry(stem, entry, pts, 41)
    bad = dict(entry, counts=entry['counts'].copy())
    bad['counts'][0] += 1
    with pytest.raises(AssertionError, match='sum'):
        RP.check_entry(stem, bad, pts, 40)


def test_empty_and_degenerate_entries_are_refused():
    empty = np.zeros((0, 5), np.float32)
    e = {'n': 0, 'crc32': RP.xyz_crc32(empty), 'counts': np.zeros(0, np.int32), 'lev': np.zeros(0)}
    with pytest.raises(AssertionError):                     # an empty scan has no 40 rings: never a kept frame
        RP.ring_pattern_keep('x', empty, e, RP_CFG, rng=_Draws(0.3, 0.6))
    for counts in ([3, 0, 2], [3, -1, 4]):
        with pytest.raises(AssertionError, match='positive'):
            RP.ring_ids(counts)
    assert RP.ring_ids([2, 1, 3]).tolist() == [0, 0, 1, 2, 2, 2]


# ---------------------------------------------------------------- the thinning


@pytest.mark.parametrize('pv,ph', [(0.0, 0.0), (0.37, 0.81), (0.999, 0.5)])
def test_one_point_per_kept_ring_per_azimuth_bin_the_first_in_stored_order(pv, ph, tmp_path, monkeypatch):
    rows, true_ring = wall_scan(LEV40, step=0.1)
    info, stem, _, entry = tool_entry(tmp_path, monkeypatch, rows)
    pts = loader_points(tmp_path, info)
    keep = RP.ring_pattern_keep(stem, pts, entry, RP_CFG, rng=_Draws(pv, ph))
    ring = RP.ring_ids(entry['counts'])
    n_bins = int(round(360.0 / 0.332))
    az = np.arctan2(pts[:, 1].astype(np.float64), pts[:, 0].astype(np.float64))
    b = np.floor(((az + np.pi) / (2 * np.pi) + ph / n_bins) * n_bins).astype(np.int64) % n_bins
    kept_rings = np.unique(ring[keep])
    for k in kept_rings:
        idx = np.nonzero(ring == k)[0]
        occupied = np.unique(b[idx])
        kept_idx = idx[keep[idx]]
        assert len(kept_idx) == len(occupied)                                # one per occupied bin, none twice
        assert len(np.unique(b[kept_idx])) == len(kept_idx)
        first = {bb: i for i in idx[::-1] for bb in [b[i]]}                  # first stored index per bin
        assert sorted(first.values()) == sorted(kept_idx.tolist())
    assert not keep[~np.isin(ring, kept_rings)].any()


@pytest.mark.parametrize('pv', [0.0, 0.13, 0.5, 0.77, 0.999])
def test_kept_rings_are_the_nearest_to_the_lattice_inside_the_recovered_range(pv, tmp_path, monkeypatch):
    rows, _ = wall_scan(LEV40, step=0.5)
    info, stem, _, entry = tool_entry(tmp_path, monkeypatch, rows)
    pts = loader_points(tmp_path, info)
    keep = RP.ring_pattern_keep(stem, pts, entry, RP_CFG, rng=_Draws(pv, 0.25))
    lev = np.asarray(entry['lev'])
    lo, hi = lev.min(), lev.max()
    lines = [(j + pv) * 1.33 for j in range(-40, 40) if lo <= (j + pv) * 1.33 <= hi]
    expect = sorted({int(np.argmin(np.abs(lev - t))) for t in lines})
    got = np.unique(RP.ring_ids(entry['counts'])[keep]).tolist()
    assert got == expect
    assert 16 <= len(got) <= 18                       # 40-beam: the full pass keeps 16-18 rings (17 in 92%)
    gaps = np.abs(np.diff(np.sort(lev[got])))
    assert 1.0 < np.median(gaps) < 1.7


# ---------------------------------------------------------------- the loader (real config, synthetic data root)


def _infos_and_root(tmp_path, monkeypatch, seed, shift_ok=True):
    """A synthetic Lyft root: one 40-beam wall scan (listed and cached) and one 64-beam frame (cached by nobody).
    Two Car boxes on the 40-beam frame, each 1 x 1 m around azimuth 30 / 150 deg and 6 cm tall, so each holds the
    points of ONE ring: one ring RING_PATTERN keeps under the phase np.random.seed(seed) draws first, one it drops."""
    np.random.seed(seed)
    pv = np.random.random_sample()
    rows, _ = wall_scan(LEV40, step=0.2)
    _, kept_rings = lr.ring_pattern_mask(rows[:, :3].astype(np.float64), RP.ring_ids(np.full(40, len(rows) // 40)),
                                         LEV40, 1.33, 0.332, pv, 0.5)
    mid = [k for k in range(40) if -5.0 < LEV40[k] < 5.0]             # inside POINT_CLOUD_RANGE z after the shift
    k_keep = [k for k in mid if k in kept_rings][0]
    k_drop = [k for k in mid if k not in kept_rings][0]

    def box(k, az_deg):
        a = np.radians(az_deg)
        return [20 * np.cos(a), 20 * np.sin(a), 20 * np.tan(np.radians(LEV40[k])), 1.0, 1.0, 0.06, 0.0]
    stem40, stem64 = 'host-a004_lidar1_1000000000000000001', 'host-a101_lidar1_1000000000000000002'
    info40, _, rec, entry = tool_entry(tmp_path, monkeypatch, rows, name=stem40)
    assert rec['rings'] == 40
    rows64, _ = wall_scan(np.linspace(15.0, -25.0, 64), step=1.0)
    rows64.tofile(str(tmp_path / 'lidar' / (stem64 + '.bin')))
    infos = []
    for stem, extra in ((stem40, {'gt_boxes': np.array([box(k_keep, 30.0), box(k_drop, 150.0)], np.float32),
                                  'gt_names': np.array(['car', 'car'])}),
                        (stem64, {'gt_boxes': np.array([box(0, 30.0)], np.float32), 'gt_names': np.array(['car'])})):
        infos.append(dict({'lidar_path': 'lidar/%s.bin' % stem, 'token': 'tok_' + stem, 'sweeps': [],
                           'ref_from_car': np.eye(4)}, **extra))
    ver = tmp_path / 'trainval'
    ver.mkdir()
    (ver / 'lidar').symlink_to(tmp_path / 'lidar')
    for split in ('train', 'val'):
        pickle.dump(infos, open(ver / ('lyft_infos_%s.pkl' % split), 'wb'))
    cache = tmp_path / 'rings.pkl'
    pickle.dump({'meta': {}, 'frames': {stem40: entry}}, open(cache, 'wb'))
    lst = tmp_path / 'kept.txt'
    lst.write_text(stem40 + '\n')
    return dict(stem40=stem40, stem64=stem64, rows=rows, entry=entry, cache=cache, lst=lst, k_keep=k_keep,
                k_drop=k_drop, infos=infos)


def _cfg(env, **over):
    cfg = EasyDict()
    cwd = Path.cwd()
    try:
        os.chdir(ROOT / 'tools')
        cfg_from_yaml_file(CFG_P, cfg)
    finally:
        os.chdir(cwd)
    dc = copy.deepcopy(cfg.DATA_CONFIGS.LYFT_40BEAM)
    dc.DATA_PATH = str(env['cache'].parent)
    dc.KEEP_FRAMES = EasyDict(LIST=str(env['lst']), SHA256=RP.file_sha256(env['lst']))
    dc.RING_PATTERN.RING_CACHE = str(env['cache'])
    dc.RING_PATTERN.SHA256 = RP.file_sha256(env['cache'])
    for k, v in over.items():
        if v is None:
            dc.pop(k, None)
        else:
            dc[k] = v
    return cfg, dc


def _dataset(cfg, dc, training=True):
    return LyftDataset(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, training=training, root_path=None,
                       logger=mock.MagicMock(), model_ontology=cfg.get('ONTOLOGY'))


def test_loader_masks_the_raw_scan_before_the_shift_and_keeps_labels(tmp_path, monkeypatch):
    env = _infos_and_root(tmp_path, monkeypatch, seed=7)
    cfg, dc = _cfg(env)
    ds = _dataset(cfg, dc)
    assert len(ds) == 1 and Path(ds.infos[0]['lidar_path']).stem == env['stem40']   # the 64-beam frame never enters
    with mock.patch.object(LyftDataset, 'prepare_data', side_effect=lambda data_dict: data_dict):
        np.random.seed(7)
        seen = ds[0]                                                    # the input to prepare_data
        dc2 = copy.deepcopy(dc)
        dc2.SHIFT_COOR = [0.0, 0.0, 0.5]
        ds2 = _dataset(cfg, dc2)
        np.random.seed(7)
        out2 = ds2[0]
    np.random.seed(7)
    expect = RP.ring_pattern_keep(env['stem40'], env['rows'], env['entry'], dc.RING_PATTERN)
    shift = np.array(dc.SHIFT_COOR, np.float32)
    np.testing.assert_array_equal(seen['points'][:, :3], env['rows'][expect, :3] + shift)
    np.testing.assert_array_equal(out2['points'][:, :3], env['rows'][expect, :3] + np.float32([0, 0, 0.5]))
    # labels are untouched by the thinning: same boxes (shifted), same names, both still present before the recount
    np.testing.assert_allclose(seen['gt_boxes'][:, :3], env['infos'][0]['gt_boxes'][:, :3] + shift, atol=1e-6)
    assert seen['gt_names'].tolist() == ['car', 'car']


def test_a_box_the_pattern_empties_is_dropped_by_the_recount_and_never_reaches_the_model(tmp_path, monkeypatch):
    env = _infos_and_root(tmp_path, monkeypatch, seed=11)
    cfg, dc = _cfg(env)
    ds = _dataset(cfg, dc)
    np.random.seed(11)
    d = ds[0]
    boxes = d['gt_boxes']
    assert len(boxes) == 1 and boxes[0, 7] == 1                             # the kept ring's Car only
    inside = roiaware_pool3d_utils.points_in_boxes_cpu(d['points'][:, :3], boxes[:, :7]).sum(1)
    assert inside.min() >= 1
    # the control: without RING_PATTERN both boxes hold points and both reach the model
    _, dc0 = _cfg(env, RING_PATTERN=None)
    np.random.seed(11)
    d0 = _dataset(cfg, dc0)[0]
    assert len(d0['gt_boxes']) == 2 and (d0['gt_boxes'][:, 7] == 1).all()


def test_keep_frames_selects_and_refuses(tmp_path, monkeypatch):
    env = _infos_and_root(tmp_path, monkeypatch, seed=3)
    cfg, dc = _cfg(env, RING_PATTERN=None)
    assert [Path(i['lidar_path']).stem for i in _dataset(cfg, dc).infos] == [env['stem40']]
    # a list naming the 64-beam frame cannot be satisfied under LIDAR_CONFIG 40
    lst = tmp_path / 'with64.txt'
    lst.write_text(env['stem40'] + '\n' + env['stem64'] + '\n')
    _, dc64 = _cfg(env, RING_PATTERN=None, KEEP_FRAMES=EasyDict(LIST=str(lst)))
    with pytest.raises(AssertionError, match='listed frames found'):
        _dataset(cfg, dc64)
    # a list that is not the pinned file
    _, dcs = _cfg(env, RING_PATTERN=None, KEEP_FRAMES=EasyDict(LIST=str(env['lst']), SHA256='0' * 64))
    with pytest.raises(AssertionError, match='hashes to'):
        _dataset(cfg, dcs)
    # evaluation mode: the list is a training-split list, not applied; RING_PATTERN off
    ev = _dataset(cfg, _cfg(env)[1], training=False)
    assert len(ev.infos) == 1 and ev.ring_pattern_cfg is None          # val infos, LIDAR_CONFIG 40 only


@pytest.mark.parametrize('over,msg', [
    ({'LIDAR_CONFIG': 64}, '40-beam'),
    ({'LIDAR_CONFIG': None}, '40-beam'),
    ({'MAX_SWEEPS': 2}, 'one stored scan'),
    ({'KEEP_FRAMES': None}, 'KEEP_FRAMES'),
])
def test_ring_pattern_refuses_64beam_accumulated_or_unlisted_use(over, msg, tmp_path, monkeypatch):
    env = _infos_and_root(tmp_path, monkeypatch, seed=5)
    cfg, dc = _cfg(env, **over)
    with pytest.raises(AssertionError, match=msg):
        _dataset(cfg, dc)


def test_ring_pattern_refuses_a_cache_without_the_frame_or_with_the_wrong_hash(tmp_path, monkeypatch):
    env = _infos_and_root(tmp_path, monkeypatch, seed=5)
    cfg, dc = _cfg(env)
    dc.RING_PATTERN.SHA256 = '0' * 64
    with pytest.raises(AssertionError, match='hashes to'):
        _dataset(cfg, dc)
    other = tmp_path / 'other.pkl'
    pickle.dump({'meta': {}, 'frames': {}}, open(other, 'wb'))
    _, dc2 = _cfg(env)
    dc2.RING_PATTERN.RING_CACHE, dc2.RING_PATTERN.SHA256 = str(other), RP.file_sha256(other)
    with pytest.raises(AssertionError, match='no ring cache entry'):
        _dataset(cfg, dc2)


def test_absent_keys_leave_the_loader_unchanged(tmp_path, monkeypatch):
    env = _infos_and_root(tmp_path, monkeypatch, seed=9)
    cfg, dc = _cfg(env, RING_PATTERN=None, KEEP_FRAMES=None)
    ds = _dataset(cfg, dc)
    assert ds.ring_pattern_cfg is None and len(ds) == 1                  # LIDAR_CONFIG 40 alone
    seen = {}
    with mock.patch.object(LyftDataset, 'prepare_data', side_effect=lambda data_dict: seen.update(data_dict) or data_dict):
        ds[0]
    assert len(seen['points']) == len(env['rows'])
