"""Pandar64 channel labels from firing order + time (pcdet/datasets/pandaset/pandaset_rings.py) and the PandaSet
RING_PATTERN / RING_FOV_CUT loader path (experiments_md 20261010_05). CPU only."""
import numpy as np
import pytest

from pcdet.datasets.pandaset import pandaset_rings as R

QUANTUM_US = 2.0 ** -22 * 1e6   # PandaSet's float64 unix-second timestamps resolve 0.24 us


def synthetic_scan(n_blocks=150, drop=0.15, dead=(), tilt_deg=0.9, seed=0, only=None):
    """A Pandar64 frame as PandaSet stores it: blocks in firing order, each block's returns in channel order (top
    first), every return at block start + the channel's firing offset, quantised; the sensor pitched by `tilt_deg`
    (elevation about the ego frame varies as a sinusoid of azimuth, as measured). Returns (xyz in PandaSet ego axes,
    t in us, true channel)."""
    rng = np.random.default_rng(seed)
    xyz, t, ch = [], [], []
    for b in range(n_blocks):
        t_block = b * 55.5 + rng.normal(0, 3.0)      # the block clock jitters
        az_block = 30.0 - 0.2 * b
        chans = range(R.N_CHANNELS) if only is None else only
        for c in chans:
            if c in dead or (only is None and rng.random() < drop):
                continue
            az = az_block + R.DAZ[c]
            el = R.EL[c] + tilt_deg * np.sin(np.radians(az - 90.0))
            rng_m = rng.uniform(3.0, 12.0) if c >= 54 else rng.uniform(8.0, 80.0)
            a, e = np.radians(az), np.radians(el)
            xyz.append(R.ORIGIN + rng_m * np.array([np.cos(a), np.sin(a), np.tan(e)]))
            t.append(np.round((t_block + R.OFF_US[c]) / QUANTUM_US) * QUANTUM_US)
            ch.append(c)
    return np.array(xyz).reshape(-1, 3), np.array(t), np.array(ch, dtype=np.int64)


def test_labels_recover_channels_with_tilt_dropped_returns_and_a_dead_channel():
    xyz, t, ch = synthetic_scan(dead=(30,))
    lab, n_blocks = R.channel_labels(xyz, t)
    assert n_blocks >= 150
    assert np.mean(lab == ch) > 0.99
    assert np.mean(lab == 30) < 0.005     # a channel that never fired is (almost) never invented


@pytest.mark.parametrize('cues', [dict(use_time=False), dict(use_elevation=False)])
def test_each_independent_cue_alone_recovers_the_channels(cues):
    xyz, t, ch = synthetic_scan(seed=1)
    lab, _ = R.channel_labels(xyz, t, **cues)
    assert np.mean(lab == ch) > 0.97


def test_partial_blocks():
    only = (50, 55, 60)          # three channels per block, the rest absent (e.g. open sky and an occluder)
    xyz, t, ch = synthetic_scan(n_blocks=40, only=only)
    lab, _ = R.channel_labels(xyz, t)
    assert np.mean(lab == ch) > 0.99


def test_empty_frame_and_single_point():
    lab, nb = R.channel_labels(np.zeros((0, 3)), np.zeros(0))
    assert lab.shape == (0,) and nb == 0
    xyz, t, ch = synthetic_scan(n_blocks=1, only=(20,))
    lab, nb = R.channel_labels(xyz, t)
    assert nb == 1 and lab.tolist() == [20]


def test_fov_cut_is_the_set_the_lattice_can_reach():
    fov, sp = (-30.67, 10.67), 1.33
    reach = set()
    for p in np.linspace(0, 1, 400, endpoint=False):
        k = R.kept_channels(sp, p, fov)
        assert 0 not in k                           # the 14.7 deg channel is above the HDL-32E's field of view
        reach |= set(k.tolist())
    assert reach == set(R.channels_in_fov(fov, sp).tolist()) == set(range(1, R.N_CHANNELS))


def test_kept_channels_follow_the_lattice_spacing():
    for p in np.linspace(0, 1, 50, endpoint=False):
        k = R.kept_channels(1.33, p, (-30.67, 10.67))
        assert 17 <= len(k) <= 22
        el = np.sort(R.EL[k])
        interior = el[(el > -6.0) & (el < 1.6)]       # inside the 0.18 deg band, away from its edges
        assert np.all((np.diff(interior) > 1.1) & (np.diff(interior) < 1.6))
        # -19.0 and -25.0 deg: the hardware is sparser than the lattice there, so the lattice always reaches them
        assert 0 not in k and {62, 63} <= set(k.tolist())


def test_ring_pattern_mask_one_point_per_channel_per_bin():
    xyz, t, ch = synthetic_scan(n_blocks=200, drop=0.0, seed=2)
    _, _, az = R.sensor_angles(xyz)
    keep, kept = R.ring_pattern_mask(ch, az, 1.33, 0.332, 0.4, 0.6, (-30.67, 10.67))
    assert set(np.unique(ch[keep]).tolist()) <= set(kept.tolist())
    n_bins = int(round(360 / 0.332))
    b = np.floor(((az + 180) / 360 + 0.6 / n_bins) * n_bins).astype(int) % n_bins
    key = ch[keep] * n_bins + b[keep]
    assert len(np.unique(key)) == keep.sum()
    # 0.2 deg firing step -> 0.332 deg bins keep ~0.6 of each kept channel
    frac = keep.sum() / np.isin(ch, kept).sum()
    assert 0.5 < frac < 0.7
    keep_v, _ = R.ring_pattern_mask(ch, az, 1.33, None, 0.4, 0.6, (-30.67, 10.67))
    assert keep_v.sum() == np.isin(ch, kept).sum()
    keep0, _ = R.ring_pattern_mask(np.zeros(0, dtype=np.int64), np.zeros(0), 1.33, 0.332, 0.4, 0.6)
    assert keep0.shape == (0,)


def test_cache_refuses_a_length_mismatch(tmp_path):
    p = R.cache_path(str(tmp_path), '014', 7)
    import os
    os.makedirs(os.path.dirname(p))
    np.save(p, np.arange(10, dtype=np.int8))
    assert len(R.load_cached_labels(str(tmp_path), '014', 7, 10)) == 10
    with pytest.raises(AssertionError):
        R.load_cached_labels(str(tmp_path), '014', 7, 11)


def test_loader_cuts_before_axis_swap_and_shift_keyed_by_frame(tmp_path, monkeypatch):
    """_get_lidar_points thins the device-0 cloud in PandaSet ego axes, by the cache entry of (sequence, frame_idx)."""
    pd = pytest.importorskip('pandas')
    ps = pytest.importorskip('pandaset')
    from easydict import EasyDict
    from pcdet.datasets.pandaset import pandaset_dataset as D

    xyz, t, ch = synthetic_scan(n_blocks=20, drop=0.0, seed=3)
    frame = pd.DataFrame({'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2], 'i': np.full(len(t), 51.0),
                          't': t * 1e-6, 'd': np.zeros(len(t), dtype=np.int64)})
    side = pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [0.0], 'i': [0.0], 't': [0.0], 'd': [1]})
    raw = pd.concat([frame, side], ignore_index=True)
    monkeypatch.setattr(D.pd, 'read_pickle', lambda path: raw)
    monkeypatch.setattr(D.ps.geometry, 'lidar_points_to_ego', lambda pts, pose: np.asarray(pts, dtype=np.float64))
    labels = ch                              # channel 0 is present: the FOV cut must drop it
    p = R.cache_path(str(tmp_path), '021', 33)
    import os
    os.makedirs(os.path.dirname(p))
    np.save(p, labels.astype(np.int8))

    class Fake:
        training = True
        dataset_cfg = EasyDict(LIDAR_DEVICE=0)
        ring_pattern_cfg = None
        ring_fov_cut_cfg = EasyDict(LABEL_CACHE=str(tmp_path), TARGET_FOV_DEG=[-30.67, 10.67], SPACING_DEG=1.33)
        _ring_keep_mask = D.PandasetDataset._ring_keep_mask

    info = {'sequence': '021', 'frame_idx': 33, 'lidar_path': 'unused'}
    pts = D.PandasetDataset._get_lidar_points(Fake(), info, pose=None)
    keep = ch != 0
    assert len(pts) == keep.sum()
    # normative axes: x = ego y, y = -ego x; no SHIFT_COOR here (it is added later, in __getitem__)
    np.testing.assert_allclose(pts[:, 0], xyz[keep, 1], rtol=0, atol=1e-4)
    np.testing.assert_allclose(pts[:, 1], -xyz[keep, 0], rtol=0, atol=1e-4)
    np.testing.assert_allclose(pts[:, 3], 0.2, atol=1e-6)

    Fake.ring_fov_cut_cfg = None
    Fake.ring_pattern_cfg = EasyDict(LABEL_CACHE=str(tmp_path), SPACING_DEG=1.33, AZ_RES_DEG=0.332,
                                     TARGET_FOV_DEG=[-30.67, 10.67])
    np.random.seed(0)
    pts_p = D.PandasetDataset._get_lidar_points(Fake(), info, pose=None)
    assert 0 < len(pts_p) < keep.sum()

    Fake.training = False                       # evaluation: untouched (device 0 only, as LIDAR_DEVICE says)
    assert len(D.PandasetDataset._get_lidar_points(Fake(), info, pose=None)) == len(t)
