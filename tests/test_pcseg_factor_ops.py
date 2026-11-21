"""pcseg S3a factor-swap operators (pcseg/factor_ops.py; experiments_md episodes/20261116_01 §8.9, rules §4 / §7;
bug_prevention_checklist rules 1, 2, 8, 9).

- Absent keys: PandasetSeg's samples are today's (reader, labels, eval sample) over >= 40 real frames, and
  train_seg.build_datasets hands the datasets today's configs key for key.
- Identity settings (360 deg cone, rate 1, identity table, and all three at once): samples identical to the base over
  >= 40 real frames, both devices, training load and eval sample.
- Labels follow the points by the SAME index; the IGNORE label (-1) passes through unchanged; the operators never
  receive a label, so labels cannot change what they keep.
- Draws are frame-keyed: the same frame gives the same cloud across calls and instances, independent of numpy's global
  RNG, and one operator's draw does not depend on another's mask.
- Equality with the detection functions on the same inputs and draws: density keep mask vs
  DataProcessor.sample_points_hist_based, rates vs per_bin_sample_rate, intensity map vs map_intensity_to_reference,
  table construction vs tools/analysis/intensity_testtime_tables.py main(), the training cone vs the evaluation's cone
  (engine.evaluate's expression) and vs point_calibration.cone_mask (equal off the cone's boundary).
- Range bins and rings never leave their tables (rule 2); non-finite coordinates raise; a file for the other device
  raises.
- The arm configs differ from their oracle config only in DATA.FACTOR_SWAP / DATA.EVAL_PERTURB, and their files exist
  and carry provenance.
Real-data tests skip when PandaSet (or the measured factor_stats files) are absent.
"""
import functools
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
from easydict import EasyDict

from pcseg import IGNORE
from pcseg import engine
from pcseg import factor_ops as fo
from pcseg import pandaset_seg as ps_mod
from pcseg.class_map import build_class_map
from pcseg.grid_sample import grid_sample
from pcseg.pandaset_seg import PandasetSeg, has_semseg, key_seed, load_poses, read_frame

ROOT = '/home/koyama/data/pandaset/dataset'
STATS = '/home/koyama/data/pcseg/factor_stats'
TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
VAL = ['045', '059', '055', '051', '020', '097', '073', '043', '003', '101', '027', '056', '011', '078', '080', '109',
       '042', '021', '094', '057']
TRAIN_HEAD = ['014', '050', '079', '048', '093', '091', '063', '104', '100', '092']
SHIFT = [0.0, 0.0, 0.30]
RING_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0])
needs_data = pytest.mark.skipif(not os.path.isdir(ROOT), reason='PandaSet absent')
ARM_FILES = {'a1': os.path.join(STATS, 'a1_density_rate_pandargt_to_pandar64.json'),
             'a2': os.path.join(STATS, 'a2_intensity_pandargt_to_pandar64_rings.npz'),
             'a4': os.path.join(STATS, 'a4_intensity_pandar64_to_pandargt_rings.npz')}


# --- helpers ------------------------------------------------------------------------------------------------------------

def _rate_file(path, rate, device=1, max_dist=300.0):
    with open(path, 'w') as f:
        json.dump({'kind': 'pcseg_density_rate', 'apply_to_device': device, 'reference_device': 1 - device,
                   'max_dist': max_dist, 'num_bins': len(rate), 'rate': [float(x) for x in rate]}, f)
    return str(path)


def _table_file(path, from_q, to_q, step, from_device, edges=RING_EDGES):
    np.savez(path, edges=edges, from_q=from_q, to_q=to_q, from_step=step, from_device=from_device,
             to_device=1 - from_device)
    return str(path)


def _identity_table(path, device):
    q = np.tile(np.linspace(0.0, 1.0, 201), (len(RING_EDGES) - 1, 1))
    return _table_file(path, q, q, 0.0, device)


def _random_table(rng, rings=len(RING_EDGES) - 1):
    """Monotone quantile rows on [0, 1] with ties (flat runs), as real tables have."""
    def row():
        x = np.sort(rng.uniform(0, 1, 201))
        x[rng.integers(0, 190):][:8] = x[rng.integers(0, 190)]
        return np.sort(x)
    return np.array([row() for _ in range(rings)]), np.array([row() for _ in range(rings)])


def _synthetic(n=6000, seed=0, r_max=320.0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, r_max, n)
    az = rng.uniform(-np.pi, np.pi, n)
    coord = np.stack([r * np.cos(az), r * np.sin(az), rng.uniform(-2, 5, n)], 1).astype(np.float32)
    coord += np.asarray(SHIFT, np.float32)
    inten = (rng.integers(0, 256, n) / 255).astype(np.float32)
    return coord, inten


def _detection_processor(src, tgt, max_dist):
    from pcdet.datasets.processor.data_processor import DataProcessor
    dp = object.__new__(DataProcessor)
    dp.hist_dist_src, dp.hist_dist_tgt, dp.hist_max_dist = np.asarray(src), np.asarray(tgt), float(max_dist)
    dp.hist_fg_src = dp.hist_fg_tgt = dp.hist_bg_src = dp.hist_bg_tgt = None
    return dp


# --- spec, bins, sentinels ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('spec', [{}, {'DENSTY': {'RATES': 'x'}}, {'CONE': None}, {'CONE': {'FOV_DEG': 60}, 'RANGE': {}}])
def test_spec_rejects_unknown_empty_and_null_blocks(spec):
    with pytest.raises(ValueError):
        fo.FactorOps(spec, 0, SHIFT)


def test_operators_never_receive_labels():
    import inspect
    assert list(inspect.signature(fo.FactorOps.__call__).parameters) == ['self', 'coord', 'intensity', 'key']


def test_range_bins_and_rings_never_leave_their_tables():
    r = np.array([0.0, 1e-9, 1.4999, 1.5, 74.99, 299.9999, 300.0, 1e6], dtype=np.float32)
    b = fo.range_bins(r, 300.0, 200)
    assert b.min() >= 0 and b.max() <= 199 and b[-1] == 199 and b[0] == 0
    tables = {'edges': RING_EDGES, 'from_q': np.tile(np.linspace(0, 1, 201), (10, 1)),
              'to_q': np.tile(np.linspace(0, 1, 201), (10, 1)) ** 2, 'from_step': 0.0}
    v = np.full(len(r), 0.5)
    out = fo.map_intensity(v, r, tables)
    assert np.allclose(out, 0.25)                                 # every point, the 1e6 m one too, found a ring


def test_non_finite_coordinates_raise():
    coord, inten = _synthetic(100)
    coord[5, 0] = np.nan
    with pytest.raises(ValueError):
        fo.FactorOps({'CONE': {'FOV_DEG': 60.0}}, 0, SHIFT)(coord, inten, 'x/00')


def test_a_file_for_the_other_device_raises(tmp_path):
    rates = _rate_file(tmp_path / 'r.json', np.ones(200), device=1)
    table = _identity_table(tmp_path / 't.npz', device=0)
    with pytest.raises(ValueError):
        fo.FactorOps({'DENSITY': {'RATES': rates}}, 0, SHIFT)
    with pytest.raises(ValueError):
        fo.FactorOps({'INTENSITY': {'TABLE': table}}, 1, SHIFT)
    fo.FactorOps({'DENSITY': {'RATES': rates}}, 1, SHIFT)
    fo.FactorOps({'INTENSITY': {'TABLE': table}}, 0, SHIFT)


# --- equality with the detection functions -----------------------------------------------------------------------------

@pytest.mark.parametrize('max_dist,bins', [(300.0, 200), (75.0, 50)])
def test_density_keep_equals_detection_sample_points_hist_based(max_dist, bins):
    rng = np.random.default_rng(1)
    coord, inten = _synthetic(20000, seed=2, r_max=max_dist * 1.1)
    src = rng.uniform(50, 400, bins)                      # the sensor being thinned (detection's `src`)
    tgt = src * rng.uniform(0.05, 1.6, bins)              # the reference; some bins above 1
    src[3], tgt[7] = 0.1, 0.0                             # under-populated bins: detection's guard keeps rate 1
    pts = np.concatenate([coord, inten[:, None]], 1)
    for seed in (0, 1, 2):
        np.random.seed(seed)
        ref = _detection_processor(src, tgt, max_dist).sample_points_hist_based(
            data_dict={'points': pts.copy()}, config=EasyDict(MIN_HIST_BIN_FRACTION=0.01))['points']
        rate = fo.density_rates_from_histograms(src, tgt, 0.01)
        u = np.random.RandomState(seed).rand(len(pts))
        keep = fo.density_keep(fo.planar_range(coord, SHIFT), max_dist, rate, u)
        assert np.array_equal(ref, pts[keep])
        assert 0.2 < keep.mean() < 0.9


def test_density_rates_direction_and_guard():
    """The thinned sensor goes in detection's src slot: rate = min(1, reference / thinned); a bin the reference never
    reaches (0 points) is guarded to 1, the range factor's business (a6), not density's."""
    rate = fo.density_rates_from_histograms([4.0, 4.0, 1.0, 5.0], [1.0, 2.0, 2.0, 0.0], 0.01)
    assert rate.tolist() == [0.25, 0.5, 1.0, 1.0]


def test_density_rates_equal_detection_per_bin_sample_rate():
    rng = np.random.default_rng(3)
    src, tgt = rng.uniform(0, 100, 200), rng.uniform(0, 100, 200)
    src[:5] = 0.0
    ref = _detection_processor(src, tgt, 300.0).per_bin_sample_rate(EasyDict(MIN_HIST_BIN_FRACTION=0.01))
    assert np.array_equal(fo.density_rates_from_histograms(src, tgt, 0.01), np.minimum(ref, 1.0))


@pytest.mark.parametrize('step', [1 / 255.0, 0.0])
def test_map_intensity_equals_detection_map_intensity_to_reference(step):
    from pcdet.datasets.processor.data_processor import DataProcessor
    rng = np.random.default_rng(4)
    from_q, to_q = _random_table(rng)
    tables = {'edges': RING_EDGES, 'from_q': from_q, 'to_q': to_q, 'from_step': step}
    coord, inten = _synthetic(30000, seed=5)
    pts = np.concatenate([coord, inten[:, None]], 1)
    ref = DataProcessor.map_intensity_to_reference(None, data_dict={'points': pts.copy()},
                                                   config=EasyDict(INTENSITY_INDEX=3), tables=tables)['points']
    u = np.random.RandomState(len(pts)).uniform(-step / 2, step / 2, len(pts)) if step > 0 else None
    out = fo.map_intensity(inten, fo.planar_range(coord, SHIFT), tables, u).astype(np.float32)
    assert np.array_equal(out, ref[:, 3])
    assert not np.array_equal(out, inten)


def _load_detection_table_script():
    spec = importlib.util.spec_from_file_location('_det_intensity_tables',
                                                  os.path.join(TOOLS, 'analysis', 'intensity_testtime_tables.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_intensity_tables_equal_detection_table_script(monkeypatch, tmp_path):
    """Detection's construction is inline in the script's main(): run it on the same histograms (its dataset passes
    replaced by these) and compare the tables it writes."""
    rng = np.random.default_rng(6)
    n = len(RING_EDGES) - 1
    hist_from = rng.integers(0, 50, (n, 1000)).astype(np.float64)
    hist_to = rng.integers(0, 50, (n, 1000)).astype(np.float64)
    hist_from[2] = 0.0
    hist_from[2, 10:14] = 400.0                    # 1,600 < 2,000 points: the all-range fallback
    hist_to[5] = 0.0
    hist_to[5, 500:505] = 420.0                    # 2,100 >= 2,000: its own map
    hist_to[9] = 0.0                               # no point on one side
    det = _load_detection_table_script()
    monkeypatch.setattr(det, 'cfg_from_yaml_file', lambda *a, **k: EasyDict(
        DATA_CONFIG=EasyDict(DATASET='PandasetDataset'), DATA_CONFIG_TAR=EasyDict(DATASET='PandasetDataset')))
    monkeypatch.setattr(det, 'source_hist', lambda cfg, frames, log: (hist_to, RING_EDGES, 1 / 255.0))
    monkeypatch.setattr(det, 'pooled_hist', lambda dcfg, cfg, frames, log, **k: (hist_from, RING_EDGES, 1 / 255.0))
    monkeypatch.setattr(sys, 'argv', ['x', '--cfg', 'unused', '--out_prefix', str(tmp_path / 'det')])
    det.main()
    t = np.load(tmp_path / 'det_rings.npz')
    from_q, to_q, own = fo.intensity_tables_from_histograms(hist_from, hist_to, 2000)
    assert np.array_equal(t['from_q'], from_q) and np.array_equal(t['to_q'], to_q)
    assert own.tolist() == [True, True, False, True, True, True, True, True, True, False]


def test_identity_table_maps_every_intensity_to_itself(tmp_path):
    coord, _ = _synthetic(len(RING_EDGES) * 256 * 4, seed=7)
    inten = (np.tile(np.arange(256), len(coord) // 256) / 255).astype(np.float32)
    t = np.load(_identity_table(tmp_path / 't.npz', 0))
    tables = {k: np.asarray(t[k], np.float64) for k in ('edges', 'from_q', 'to_q')}
    tables['from_step'] = 0.0
    out = fo.map_intensity(inten, fo.planar_range(coord, SHIFT), tables).astype(np.float32)
    assert np.array_equal(out, inten)


def test_cone_keep_is_the_evaluation_cone_and_360_is_the_identity():
    from pcdet.datasets.point_calibration import cone_mask as calib_cone
    coord, inten = _synthetic(50000, seed=8)
    evaluation = engine.cone_mask(coord[:, :2] - np.asarray(SHIFT, dtype=np.float32)[:2], 60.0, 0.0)
    assert np.array_equal(fo.cone_keep(coord, SHIFT, 60.0, 0.0), evaluation)
    assert fo.cone_keep(coord, SHIFT, 360.0).all()
    assert not engine.cone_mask(coord[:, :2], 360.0, 0.0).all()        # why 360 is special-cased (x > 0)
    # detection's training-side cone (<= at the edge, no x > 0) agrees off the boundary
    calib = calib_cone(coord[:, :2].astype(np.float64), 60.0, 0.0)
    az = np.degrees(np.arctan2(coord[:, 1], coord[:, 0]))
    diff = calib != evaluation
    assert np.all(np.abs(np.abs(az[diff]) - 30.0) < 1e-4)
    edge = np.array([[np.cos(np.pi / 6), np.sin(np.pi / 6)]]) * 10
    assert calib_cone(edge, 60.0)[0] and not engine.cone_mask(edge, 60.0)[0]   # the boundary convention differs


# --- the dataset hook on synthetic frames ------------------------------------------------------------------------------

def _fake_reader(monkeypatch, n=4000, all_ignored=False):
    """Synthetic frames through PandasetSeg: raw ids 0..42 (0-4 map to IGNORE in pandaset_scalr17)."""
    def read(root, seq, frame, device, pose):
        rng = np.random.default_rng(1000 * int(seq) + frame)
        coord, inten = _synthetic(n, seed=1000 * int(seq) + frame)
        pts = np.concatenate([coord - np.asarray(SHIFT, np.float32), inten[:, None]], 1).astype(np.float32)
        raw = np.zeros(n, np.int64) if all_ignored else rng.integers(0, 43, n)
        return pts, raw, np.arange(n)
    monkeypatch.setattr(ps_mod, 'read_frame', read)
    monkeypatch.setattr(ps_mod, 'has_semseg', lambda root, seq: True)
    monkeypatch.setattr(ps_mod, 'load_poses', lambda root, seq: [None] * 80)
    return read


def _cfg(ops=None, device=1, seqs=('001', '002')):
    c = {'ROOT': '/nowhere', 'SEQUENCES': list(seqs), 'LIDAR_DEVICE': device, 'SHIFT_COOR': SHIFT,
         'CLASS_MAP': 'pandaset_scalr17', 'GRID_SIZE': 0.05}
    if ops is not None:
        c['FACTOR_OPS'] = ops
    return c


def _ops(tmp_path, device=1, rate=0.4):
    rng = np.random.default_rng(9)
    from_q, to_q = _random_table(rng)
    return {'CONE': {'FOV_DEG': 60.0, 'HEADING_DEG': 0.0},
            'DENSITY': {'RATES': _rate_file(tmp_path / ('r%d.json' % device), np.full(200, rate), device=device)},
            'INTENSITY': {'TABLE': _table_file(tmp_path / ('t%d.npz' % device), from_q, to_q, 1 / 255.0, device)}}


def test_labels_follow_the_index_and_ignore_passes_through(monkeypatch, tmp_path):
    read = _fake_reader(monkeypatch)
    cm = build_class_map('pandaset_scalr17')
    ops = _ops(tmp_path)
    ds = PandasetSeg(_cfg(ops), training=True)
    for i in (0, 3, 85):
        seq, frame = ds.frames[i]
        pts, raw, _ = read(None, seq, frame, 1, None)
        coord_full = pts[:, :3] + np.asarray(SHIFT, np.float32)
        seg_full = cm(raw)
        idx, inten = fo.FactorOps(ops, 1, SHIFT)(coord_full, pts[:, 3], ds.load(i)[3])
        coord, strength, seg, key = ds.load(i)
        assert np.array_equal(coord, coord_full[idx]) and np.array_equal(seg, seg_full[idx])
        assert np.array_equal(strength[:, 0], inten)
        assert (seg == IGNORE).sum() == (seg_full[idx] == IGNORE).sum() > 0
        assert seg.min() >= IGNORE and 0 < len(idx) < len(raw)
    # the operators see no label: an all-IGNORE frame keeps the same points and intensities
    a = ds.load(0)
    _fake_reader(monkeypatch, all_ignored=True)
    b = PandasetSeg(_cfg(ops), training=True).load(0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and (b[2] == IGNORE).all()


def test_eval_sample_is_built_from_the_perturbed_cloud(monkeypatch, tmp_path):
    _fake_reader(monkeypatch)
    ds = PandasetSeg(_cfg(_ops(tmp_path)), training=False)
    coord, strength, seg, key = ds.load(2)
    s = ds[2]
    idx, grid, inv = grid_sample(coord, 0.05, np.random.default_rng(key_seed(key)))
    assert np.array_equal(s['origin_coord'], coord) and np.array_equal(s['origin_segment'], seg)
    assert np.array_equal(s['inverse'], inv) and np.array_equal(s['segment'], seg[idx])
    assert np.array_equal(s['feat'], np.concatenate([coord[idx], strength[idx]], 1))


def test_same_frame_same_cloud_whatever_the_global_rng(monkeypatch, tmp_path):
    _fake_reader(monkeypatch)
    ops = _ops(tmp_path)
    np.random.seed(0)
    state = np.random.get_state()[1].copy()
    a = PandasetSeg(_cfg(ops), training=True).load(5)
    assert np.array_equal(np.random.get_state()[1], state)        # the global RNG is never drawn
    np.random.seed(123)
    ds = PandasetSeg(_cfg(ops), training=True)
    b, c = ds.load(5), ds.load(5)
    for x, y in ((a, b), (b, c)):
        assert all(np.array_equal(p, q) for p, q in zip(x[:3], y[:3]))
    other = ds.load(6)
    assert len(other[0]) != len(a[0]) or not np.array_equal(other[0], a[0])


def test_one_operators_draw_does_not_depend_on_another_mask(monkeypatch, tmp_path):
    ops = _ops(tmp_path)
    coord, inten = _synthetic(8000, seed=11)
    both, _ = fo.FactorOps({'CONE': ops['CONE'], 'DENSITY': ops['DENSITY']}, 1, SHIFT)(coord, inten, '001/04')
    dens, _ = fo.FactorOps({'DENSITY': ops['DENSITY']}, 1, SHIFT)(coord, inten, '001/04')
    cone = np.flatnonzero(fo.cone_keep(coord, SHIFT, 60.0))
    assert np.array_equal(both, np.intersect1d(dens, cone))
    i_all, v_all = fo.FactorOps({'INTENSITY': ops['INTENSITY']}, 1, SHIFT)(coord, inten, '001/04')
    i_c, v_c = fo.FactorOps({'CONE': ops['CONE'], 'INTENSITY': ops['INTENSITY']}, 1, SHIFT)(coord, inten, '001/04')
    assert np.array_equal(v_c, v_all[i_c])


def test_density_keeps_the_rate_and_draws_by_frame(tmp_path):
    coord, inten = _synthetic(200000, seed=12)
    ops = fo.FactorOps({'DENSITY': {'RATES': _rate_file(tmp_path / 'r.json', np.full(200, 0.3))}}, 1, SHIFT)
    a, _ = ops(coord, inten, '001/00')
    b, _ = ops(coord, inten, '001/01')
    assert abs(len(a) / len(coord) - 0.3) < 0.005 and not np.array_equal(a, b)


# --- routing in train_seg and the arm configs --------------------------------------------------------------------------

def _load_cfg(name):
    sys.path.insert(0, TOOLS)
    import train_seg
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        return train_seg.load_cfg('cfgs/pcseg/%s.yaml' % name)
    finally:
        os.chdir(cwd)


class _Recorder:
    made = []

    def __init__(self, cfg, training, logger=None):
        self.cfg, self.training = cfg, training
        self.sequences = list(cfg['SEQUENCES'])[:45 if training else 15]
        _Recorder.made.append(self)


def _routed(monkeypatch, cfg):
    sys.path.insert(0, TOOLS)
    import train_seg
    _Recorder.made = []
    monkeypatch.setattr(ps_mod, 'PandasetSeg', _Recorder)
    sets = train_seg.build_datasets(cfg)
    return sets['train'].cfg, sets['val'].cfg


BASE_KEYS = {'ROOT', 'LIDAR_DEVICE', 'SHIFT_COOR', 'CLASS_MAP', 'GRID_SIZE', 'SEQUENCES'}
ARMS = {  # config: (base config, block, device, operator keys)
    'spunet-p1-a1-density': ('spunet-pandasetgt', 'FACTOR_SWAP', 1, ['DENSITY']),
    'spunet-p1-a1-density-reliance': ('spunet-pandasetgt', 'EVAL_PERTURB', 1, ['DENSITY']),
    'spunet-p1-a2-intensity': ('spunet-pandasetgt', 'FACTOR_SWAP', 1, ['INTENSITY']),
    'spunet-p1-a2-intensity-reliance': ('spunet-pandasetgt', 'EVAL_PERTURB', 1, ['INTENSITY']),
    'spunet-p1-a3-cone': ('spunet-pandaset64', 'FACTOR_SWAP', 0, ['CONE']),
    'spunet-p1-a3-cone-reliance': ('spunet-pandaset64', 'EVAL_PERTURB', 0, ['CONE']),
    'spunet-p1-a4-intensity': ('spunet-pandaset64', 'FACTOR_SWAP', 0, ['INTENSITY']),
    'spunet-p1-a4-intensity-reliance': ('spunet-pandaset64', 'EVAL_PERTURB', 0, ['INTENSITY']),
}


@pytest.mark.parametrize('name', ['spunet-pandaset64', 'spunet-pandasetgt', 'spunet-pandaset64-smoke'])
def test_without_the_keys_build_datasets_passes_todays_configs(monkeypatch, name):
    cfg = _load_cfg(name)
    assert 'FACTOR_SWAP' not in cfg.DATA and 'EVAL_PERTURB' not in cfg.DATA
    tr, va = _routed(monkeypatch, cfg)
    assert set(tr) == BASE_KEYS | {'AUG'} and set(va) == BASE_KEYS
    assert tr['AUG'] == cfg.DATA.AUG


@pytest.mark.parametrize('name', sorted(ARMS))
def test_arm_blocks_reach_only_their_split(monkeypatch, name):
    base, block, device, keys = ARMS[name]
    cfg = _load_cfg(name)
    tr, va = _routed(monkeypatch, cfg)
    on, off = (tr, va) if block == 'FACTOR_SWAP' else (va, tr)
    assert 'FACTOR_OPS' not in off and dict(on['FACTOR_OPS']) == dict(cfg.DATA[block])
    assert sorted(on['FACTOR_OPS']) == keys


@pytest.mark.parametrize('name', sorted(ARMS))
def test_arm_config_differs_from_its_oracle_only_in_its_block(name):
    base, block, device, keys = ARMS[name]
    a, b = _load_cfg(name), _load_cfg(base)
    assert a.DATA.LIDAR_DEVICE == device
    other = 'EVAL_PERTURB' if block == 'FACTOR_SWAP' else 'FACTOR_SWAP'
    assert other not in a.DATA
    da, db = dict(a.DATA), dict(b.DATA)
    da.pop(block)
    assert da == db
    for x in (a, b):
        x.pop('TAG', None)
        x.pop('_BASE_CONFIG_', None)
    ra, rb = dict(a), dict(b)
    ra.pop('DATA'), rb.pop('DATA')
    assert ra == rb


@pytest.mark.parametrize('name', sorted(ARMS))
def test_arm_files_exist_match_the_device_and_carry_provenance(name):
    base, block, device, keys = ARMS[name]
    spec = _load_cfg(name).DATA[block]
    paths = [v for k in keys for kk, v in spec[k].items() if kk in ('RATES', 'TABLE')]
    for p in paths:
        assert p.startswith('/home/koyama/data/pcseg/factor_stats/'), p
    if not all(os.path.exists(p) for p in paths):
        pytest.skip('factor_stats files absent')
    fo.FactorOps(dict(spec), device, SHIFT)                       # loads, device checked
    for p in paths:
        prov = (json.load(open(p))['provenance'] if p.endswith('.json')
                else json.loads(str(np.load(p)['provenance'])))
        assert prov['split'] == 'train' and prov['labels_read'] == 'none' and prov['stats_sha256']
        assert prov['frames_per_device'] == 3600


def test_nuscenes_rejects_the_keys():
    sys.path.insert(0, TOOLS)
    import train_seg
    cfg = _load_cfg('spunet-nuscenes-smoke')
    cfg.DATA.FACTOR_SWAP = {'CONE': {'FOV_DEG': 60.0}}
    with pytest.raises(ValueError):
        train_seg.build_datasets(cfg)


# --- real frames -------------------------------------------------------------------------------------------------------

def _frames():
    seqs = [s for s in TRAIN_HEAD + VAL if has_semseg(ROOT, s)][:16]
    return [(s, f) for s in seqs for f in (0, 41, 79)]        # 48 frames


@functools.lru_cache(maxsize=None)
def _cached(seq, frame, device):
    return read_frame(ROOT, seq, frame, device, load_poses(ROOT, seq)[frame])


@pytest.fixture
def cached_reader(monkeypatch):
    """read_frame read once per frame for all variants of a test (copies, so no variant can alter another's)."""
    monkeypatch.setattr(ps_mod, 'read_frame',
                        lambda root, seq, frame, device, pose: tuple(a.copy() for a in _cached(seq, frame, device)))


def _real_set(device, ops=None, training=True):
    frames = _frames()
    seqs = sorted({s for s, _ in frames})
    ds = PandasetSeg(dict(_cfg(ops, device, seqs), ROOT=ROOT), training=training)
    return ds, [ds.frames.index(fr) for fr in frames]


@needs_data
@pytest.mark.parametrize('device', [0, 1])
def test_read_points_is_read_frame_without_labels(device):
    assert len(_frames()) >= 40
    for seq, frame in _frames():
        pose = load_poses(ROOT, seq)[frame]
        pts, keep = fo.read_points(ROOT, seq, frame, device, pose)
        ref, _, ref_keep = read_frame(ROOT, seq, frame, device, pose)
        assert np.array_equal(pts, ref) and np.array_equal(keep, ref_keep), (seq, frame)


@needs_data
@pytest.mark.parametrize('device', [0, 1])
def test_without_the_keys_samples_are_todays(device, cached_reader):
    """PandasetSeg without FACTOR_OPS against today's recipe written out here: read_frame, + SHIFT_COOR, class map;
    the eval sample's frame-keyed voxel draw."""
    cm = build_class_map('pandaset_scalr17')
    shift = np.asarray(SHIFT, np.float32)
    ds, idxs = _real_set(device)
    ev, _ = _real_set(device, training=False)
    assert ds.factor_ops is None and len(idxs) >= 40
    for i in idxs:
        seq, frame = ds.frames[i]
        pts, raw, _ = _cached(seq, frame, device)
        coord, strength, seg, key = ds.load(i)
        assert np.array_equal(coord, pts[:, :3] + shift) and np.array_equal(strength, pts[:, 3:4])
        assert np.array_equal(seg, cm(raw)) and coord.dtype == np.float32
        if i == idxs[0] or i == idxs[-1]:
            s = ev[i]
            vi, grid, inv = grid_sample(coord, 0.05, np.random.default_rng(key_seed(key)))
            assert np.array_equal(s['segment'], seg[vi]) and np.array_equal(s['inverse'], inv)
            assert np.array_equal(s['grid_coord'], grid) and np.array_equal(s['origin_segment'], seg)


def _identity_specs(tmp_path, device):
    rates = _rate_file(tmp_path / ('ones%d.json' % device), np.ones(200), device=device)
    table = _identity_table(tmp_path / ('id%d.npz' % device), device)
    cone = {'FOV_DEG': 360.0, 'HEADING_DEG': 0.0}
    return {'cone': {'CONE': cone}, 'density': {'DENSITY': {'RATES': rates}},
            'intensity': {'INTENSITY': {'TABLE': table}},
            'all': {'CONE': cone, 'DENSITY': {'RATES': rates}, 'INTENSITY': {'TABLE': table}}}


@needs_data
@pytest.mark.parametrize('device', [0, 1])
def test_identity_settings_give_todays_samples(device, tmp_path, cached_reader):
    base, idxs = _real_set(device)
    base_ev, _ = _real_set(device, training=False)
    for name, spec in _identity_specs(tmp_path, device).items():
        ds, _ = _real_set(device, spec)
        ev, _ = _real_set(device, spec, training=False)
        for i in idxs:
            a, b = base.load(i), ds.load(i)
            for x, y in zip(a[:3], b[:3]):
                assert x.dtype == y.dtype and np.array_equal(x, y), (name, base.frames[i])
        for i in (idxs[0], idxs[-1]):
            sa, sb = base_ev[i], ev[i]
            for k in ('grid_coord', 'feat', 'segment', 'inverse', 'origin_segment', 'origin_coord'):
                assert np.array_equal(sa[k], sb[k]), (name, k)


@needs_data
def test_real_frames_labels_follow_the_index(tmp_path, cached_reader):
    cm = build_class_map('pandaset_scalr17')
    shift = np.asarray(SHIFT, np.float32)
    for device in (0, 1):
        ops = _ops(tmp_path, device, rate=0.3)
        ds, idxs = _real_set(device, ops)
        n_ign = 0
        for i in idxs:
            seq, frame = ds.frames[i]
            pts, raw, _ = _cached(seq, frame, device)
            coord, strength, seg, key = ds.load(i)
            idx, inten = fo.FactorOps(ops, device, SHIFT)(pts[:, :3] + shift, pts[:, 3], key)
            assert np.array_equal(coord, pts[idx, :3] + shift) and np.array_equal(seg, cm(raw)[idx])
            assert np.array_equal(strength[:, 0], inten)
            n_ign += int((seg == IGNORE).sum())
        assert n_ign > 0                                          # the sentinel was exercised


@needs_data
def test_training_cone_is_the_evaluation_cone_on_real_frames(cached_reader):
    from pcdet.datasets.point_calibration import cone_mask as calib_cone
    base, idxs = _real_set(0, training=False)
    cone, _ = _real_set(0, {'CONE': {'FOV_DEG': 60.0, 'HEADING_DEG': 0.0}})
    shift = np.asarray(SHIFT, dtype=np.float32)
    n_diff = 0
    for i in idxs:
        s = base[i]
        m = engine.cone_mask(s['origin_coord'][:, :2] - shift[:2], 60.0, 0.0)     # engine.evaluate's expression
        coord, _, seg, _ = cone.load(i)
        assert np.array_equal(coord, s['origin_coord'][m]) and np.array_equal(seg, s['origin_segment'][m])
        c = calib_cone(s['origin_coord'][:, :2].astype(np.float64), 60.0, 0.0)
        n_diff += int((c != m).sum())
    assert n_diff <= 2                                            # boundary points only, if any


@needs_data
def test_stats_script_counts_what_the_operators_see():
    sys.path.insert(0, TOOLS)
    import pandas as pd
    from analysis import pcseg_factor_stats as st
    seq, frame = _frames()[0]
    pose = load_poses(ROOT, seq)[frame]
    lidar = pd.read_pickle(os.path.join(ROOT, seq, 'lidar', '%02d.pkl.gz' % frame))
    shift = np.asarray(SHIFT, np.float32)
    out = st.frame_stats(lidar, pose, '%s/%02d' % (seq, frame), shift, (60.0, 0.0))
    for dev in (0, 1):
        pts, _, _ = read_frame(ROOT, seq, frame, dev, pose)
        coord = pts[:, :3] + shift
        inc = fo.cone_keep(coord, shift, 60.0)
        d = out[dev]
        assert d['row'][0] == len(pts) and d['row'][1] == inc.sum() == d['hist_density'].sum()
        assert d['hist_int_cone'].sum() == inc.sum() and d['hist_int_all'].sum() == len(pts)
        r = fo.planar_range(coord, shift)
        assert np.array_equal(d['hist_density'], np.bincount(fo.range_bins(r[inc], 300.0, 200), minlength=200))


@needs_data
@pytest.mark.skipif(not os.path.exists(ARM_FILES['a2']), reason='factor_stats tables absent')
@pytest.mark.parametrize('arm,device', [('a2', 1), ('a4', 0)])
def test_measured_tables_map_as_detection_does_on_real_frames(arm, device, cached_reader):
    from pcdet.datasets.processor.data_processor import DataProcessor
    t = np.load(ARM_FILES[arm])
    assert int(t['from_device']) == device and int(t['to_device']) == 1 - device
    tables = {k: np.asarray(t[k], np.float64) for k in ('edges', 'from_q', 'to_q')}
    tables['from_step'] = float(t['from_step'])
    step = tables['from_step']
    shift = np.asarray(SHIFT, np.float32)
    for seq, frame in _frames()[:6]:
        pts, _, _ = _cached(seq, frame, device)
        p = np.concatenate([pts[:, :3] + shift, pts[:, 3:4]], 1)
        ref = DataProcessor.map_intensity_to_reference(None, data_dict={'points': p.copy()},
                                                       config=EasyDict(INTENSITY_INDEX=3), tables=tables)['points']
        u = np.random.RandomState(len(p)).uniform(-step / 2, step / 2, len(p))
        out = fo.map_intensity(pts[:, 3], fo.planar_range(p[:, :3], shift), tables, u).astype(np.float32)
        assert np.array_equal(out, ref[:, 3])


@pytest.mark.parametrize('name', ['spunet-p1-a3-cone', 'spunet-p1-a3-cone-reliance'])
def test_a3_cone_is_the_evaluation_cone(name):
    cfg = _load_cfg(name)
    block = cfg.DATA.get('FACTOR_SWAP', None) or cfg.DATA.EVAL_PERTURB
    assert dict(block.CONE) == dict(cfg.EVAL.CONE)


@pytest.mark.skipif(not os.path.exists(ARM_FILES['a1']), reason='factor_stats files absent')
def test_measured_a1_rates_bring_pandargt_to_pandar64s_in_cone_density():
    """Per bin: thinned PandarGT = Pandar64 wherever the rate is below 1; rate 1 only where PandarGT is the sparser
    sensor or a bin is guarded (beyond Pandar64's reach, or under the ego vehicle)."""
    d = json.load(open(ARM_FILES['a1']))
    rate, a, b = (np.asarray(d[k]) for k in ('rate', 'rho_apply_per_frame', 'rho_reference_per_frame'))
    assert d['apply_to_device'] == 1 and d['reference_device'] == 0 and len(rate) == d['num_bins'] == 200
    low = rate < 1
    assert np.allclose(rate[low] * a[low], b[low], rtol=1e-9)
    guarded = np.zeros(len(rate), bool)
    guarded[d['guarded_bins']] = True
    assert np.all(guarded[~low] | (a[~low] <= b[~low]))
    assert abs((rate * a).sum() / b.sum() - 1) < 0.03                 # 63k -> 16.3k in-cone points per frame


def test_operators_apply_the_documented_frame_keyed_draws(tmp_path):
    """FactorOps against the pure functions with the draws written out: DENSITY keeps u < rate with
    u = default_rng(key_seed(key, 'factor_ops/density')).random(N); INTENSITY maps with the dequantisation
    default_rng(key_seed(key, 'factor_ops/intensity')).uniform(-step / 2, step / 2, N); N = the frame's point count."""
    ops = _ops(tmp_path, device=1, rate=0.35)
    coord, inten = _synthetic(20000, seed=13)
    key = '014/07'
    n = len(coord)
    r = fo.planar_range(coord, SHIFT)
    idx, _ = fo.FactorOps({'DENSITY': ops['DENSITY']}, 1, SHIFT)(coord, inten, key)
    u = np.random.default_rng(key_seed(key, 'factor_ops/density')).random(n)
    assert np.array_equal(idx, np.flatnonzero(fo.density_keep(r, 300.0, np.full(200, 0.35), u)))
    t = np.load(ops['INTENSITY']['TABLE'])
    tables = {k: np.asarray(t[k], np.float64) for k in ('edges', 'from_q', 'to_q')}
    tables['from_step'] = step = float(t['from_step'])
    idx, out = fo.FactorOps({'INTENSITY': ops['INTENSITY']}, 1, SHIFT)(coord, inten, key)
    u = np.random.default_rng(key_seed(key, 'factor_ops/intensity')).uniform(-step / 2, step / 2, n)
    assert np.array_equal(idx, np.arange(n))
    assert np.array_equal(out, fo.map_intensity(inten, r, tables, u).astype(np.float32))
    assert not np.array_equal(out, inten)
