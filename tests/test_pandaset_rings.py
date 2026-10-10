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
        eval_ring_thin_cfg = None
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


# ---- EVAL_RING_THIN (evaluation-only cuts for the oracle-degradation table, experiments_md 20261011_03) ----

def _scan(seed=4, n_blocks=300):
    xyz, t, ch = synthetic_scan(n_blocks=n_blocks, drop=0.1, seed=seed)
    return xyz, ch


@pytest.mark.parametrize('mode', R.EVAL_THIN_MODES)
def test_eval_thin_empty_scan(mode):
    cfg = dict(MODE=mode, STRIDE=2, AZ_RES_DEG=0.332, EL_DEG=0.0, SPACING_DEG=1.33, TARGET_FOV_DEG=[-30.67, 10.67])
    keep = R.eval_thin_mask(np.zeros((0, 3)), np.zeros(0, dtype=np.int64), cfg, np.random.default_rng(0))
    assert keep.shape == (0,)


def test_eval_thin_unknown_mode_refused():
    xyz, ch = _scan(n_blocks=5)
    with pytest.raises(AssertionError):
        R.eval_thin_mask(xyz, ch, dict(MODE='lines'), np.random.default_rng(0))


@pytest.mark.parametrize('structured,par', [('rows', {}), ('cols', {}), ('cols', {'STRIDE': 4}),
                                            ('azbin', {'AZ_RES_DEG': 0.332})])
def test_eval_thin_random_twin_matches_the_count(structured, par):
    xyz, ch = _scan()
    s = R.eval_thin_mask(xyz, ch, dict(MODE=structured, **par), np.random.default_rng(1))
    rmode = 'random' if structured == 'rows' else 'random_' + structured
    r = R.eval_thin_mask(xyz, ch, dict(MODE=rmode, **par), np.random.default_rng(1))
    assert r.sum() == s.sum() and 0 < s.sum() < len(ch)
    if structured == 'rows':                          # the random control keeps (almost) every line
        assert len(np.unique(ch[r])) > len(np.unique(ch[s])) * 1.8


def test_eval_thin_rows_drop_whole_lines():
    xyz, ch = _scan()
    keep = R.eval_thin_mask(xyz, ch, dict(MODE='rows', STRIDE=2), np.random.default_rng(0))
    assert set(np.unique(ch[keep]).tolist()) == {c for c in np.unique(ch).tolist() if c % 2 == 0}
    assert np.all(keep == (ch % 2 == 0))


def test_eval_thin_cols_keep_every_kth_return_per_line_in_firing_order():
    xyz, ch = _scan()
    for stride in (2, 4):
        keep = R.eval_thin_mask(xyz, ch, dict(MODE='cols', STRIDE=stride), np.random.default_rng(0))
        for c in np.unique(ch):
            idx = np.nonzero(ch == c)[0]
            assert keep[idx].sum() == int(np.ceil(len(idx) / stride))
            assert keep[idx[0]] and np.all(keep[idx[::stride]])     # the 1st, (k+1)th, ... return of the line


def test_eval_thin_azbin_one_point_per_line_per_bin():
    xyz, ch = _scan(n_blocks=400)
    keep = R.eval_thin_mask(xyz, ch, dict(MODE='azbin', AZ_RES_DEG=0.332), np.random.default_rng(0))
    _, _, az = R.sensor_angles(xyz)
    n_bins = int(round(360 / 0.332))
    b = np.floor((az + 180) / 360 * n_bins).astype(int) % n_bins
    key = ch[keep] * n_bins + b[keep]
    assert len(np.unique(key)) == keep.sum()
    assert len(np.unique(key)) == len(np.unique(ch * n_bins + b))     # every occupied (line, bin) keeps one point
    assert 0.5 < keep.mean() < 0.7                                      # 0.2 deg firing step -> 0.332 deg bins


def test_eval_thin_elevation_cuts_are_by_channel():
    xyz, ch = _scan()
    for mode, deg in (('elmax', 0.0), ('elmax', -3.0), ('elmin', -14.0), ('elmin', -10.0)):
        keep = R.eval_thin_mask(xyz, ch, dict(MODE=mode, EL_DEG=deg), np.random.default_rng(0))
        el = R.EL[ch]
        assert np.all(keep == ((el <= deg) if mode == 'elmax' else (el >= deg)))


def test_eval_thin_pattern_is_the_training_render_and_reproducible():
    xyz, ch = _scan()
    cfg = dict(MODE='pattern', SPACING_DEG=1.33, AZ_RES_DEG=0.332, TARGET_FOV_DEG=[-30.67, 10.67])
    k1 = R.eval_thin_mask(xyz, ch, cfg, np.random.default_rng(7))
    k2 = R.eval_thin_mask(xyz, ch, cfg, np.random.default_rng(7))
    assert np.array_equal(k1, k2)
    assert set(np.unique(ch[k1]).tolist()) <= set(R.channels_in_fov([-30.67, 10.67], 1.33).tolist())
    rng = np.random.default_rng(7)
    _, _, az = R.sensor_angles(xyz)
    ref, _ = R.ring_pattern_mask(ch, az, 1.33, 0.332, rng.random(), rng.random(), [-30.67, 10.67])
    assert np.array_equal(k1, ref)


def test_loader_eval_thin_is_evaluation_only_and_seeded_by_frame(tmp_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    pytest.importorskip('pandaset')
    from easydict import EasyDict
    from pcdet.datasets.pandaset import pandaset_dataset as D

    xyz, t, ch = synthetic_scan(n_blocks=40, drop=0.0, seed=5)
    raw = pd.DataFrame({'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2], 'i': np.zeros(len(t)), 't': t * 1e-6,
                        'd': np.zeros(len(t), dtype=np.int64)})
    monkeypatch.setattr(D.pd, 'read_pickle', lambda path: raw)
    monkeypatch.setattr(D.ps.geometry, 'lidar_points_to_ego', lambda pts, pose: np.asarray(pts, dtype=np.float64))
    import os
    for frame in (3, 4):
        p = R.cache_path(str(tmp_path), '045', frame)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p, ch.astype(np.int8))

    class Fake:
        training = False
        dataset_cfg = EasyDict(LIDAR_DEVICE=0)
        ring_pattern_cfg = None
        ring_fov_cut_cfg = None
        eval_ring_thin_cfg = EasyDict(LABEL_CACHE=str(tmp_path), MODE='random', STRIDE=2)
        _eval_ring_thin_mask = D.PandasetDataset._eval_ring_thin_mask

    get = D.PandasetDataset._get_lidar_points
    a = get(Fake(), {'sequence': '045', 'frame_idx': 3, 'lidar_path': 'x'}, pose=None)
    b = get(Fake(), {'sequence': '045', 'frame_idx': 3, 'lidar_path': 'x'}, pose=None)
    c = get(Fake(), {'sequence': '045', 'frame_idx': 4, 'lidar_path': 'x'}, pose=None)
    assert len(a) == (ch % 2 == 0).sum() and np.array_equal(a, b)     # count of the rows cut; same draw on a rerun
    assert not np.array_equal(a, c)                                   # another frame, another draw
    Fake.training = True                                              # training: untouched
    assert len(get(Fake(), {'sequence': '045', 'frame_idx': 3, 'lidar_path': 'x'}, pose=None)) == len(ch)
