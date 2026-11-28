"""Pair P2, nuScenes <-> SemanticKITTI, for the intensity method row (experiments_md episodes/20261128_01).

- The two Complete & Label 10-class maps equal SALUDA's released dicts entry by entry (copied verbatim below), and raise
  on an undeclared id; ignored ids give pcseg's IGNORE sentinel.
- The SemanticKITTI reader (Pointcept's convention: x, y, z, remission as stored; semantic id = label & 0xFFFF).
- factor_ops with a sensor NAME (P2): INTENSITY only, from a table that names the sensor; PandaSet's int devices are
  unchanged.
- Rule 8, the identity: an identity INTENSITY table through the nuScenes and SemanticKITTI loaders over >= 40 frames
  each leaves coordinates and labels identical and intensities equal (no dequantisation) or within half a step.
- The configs: one recipe on both sides, the step budget, each method row = its oracle + one FACTOR_SWAP block that
  reaches the train set only, each smoke = its row + the smoke overrides.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

from pcseg import IGNORE
from pcseg.class_map import CL10_NAMES, build_class_map

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, 'tools')
SK_ROOT = '/home/koyama/data/semantic_kitti/dataset/sequences'
NUSCENES = '/home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval'
TABLE_DIR = '/home/koyama/data/pcseg/factor_stats'

# --- SALUDA (github.com/valeoai/SALUDA at e4579fe), verbatim -----------------------------------------------------------
# datasets/nuscenes.py class_mapping_da, the first dict ("Official class mappings from Complet&Label (CVPR 2021)")
SALUDA_NS = {
    0: 0, 1: 0, 2: 6, 3: 6, 4: 6, 5: 6, 6: 6, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 2, 15: 5, 16: 5, 17: 1,
    18: 5, 19: 5, 20: 5, 21: 3, 22: 5, 23: 4, 24: 7, 25: 0, 26: 8, 27: 9, 28: 0, 29: 0, 30: 10, 31: 0,
}
# datasets/semantic_kitti.py class_mapping_da, learning_map ("Original class mapping from Complete&Label paper")
SALUDA_SK = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 0, 18: 4, 20: 5, 30: 6, 31: 0, 32: 0, 40: 7, 44: 7, 48: 8, 49: 0, 50: 0,
    51: 0, 52: 0, 60: 7, 70: 10, 71: 10, 72: 9, 80: 0, 81: 0, 99: 0, 252: 1, 253: 0, 254: 6, 255: 0, 256: 0, 257: 5,
    258: 4, 259: 5,
}
# SALUDA's learning_map_inv names its train ids 1..10 (car, bicycle, motorcycle, truck, other-vehicle, person, road,
# sidewalk, terrain, vegetation); ours are the same classes in the same order.
SALUDA_ORDER = ['car', 'bicycle', 'motorcycle', 'truck', 'other_vehicle', 'pedestrian', 'driveable_surface', 'sidewalk',
                'terrain', 'vegetation']


@pytest.mark.parametrize('name,ref', [('nuscenes_cl10', SALUDA_NS), ('semantickitti_cl10', SALUDA_SK)])
def test_cl10_equals_saluda_entry_by_entry(name, ref):
    cm = build_class_map(name)
    assert cm.names == CL10_NAMES == SALUDA_ORDER
    raw = np.array(sorted(ref))
    want = np.array([ref[r] - 1 if ref[r] > 0 else IGNORE for r in raw])
    assert np.array_equal(cm(raw), want)
    declared = {int(r) for r in np.flatnonzero(cm.lut != -2)}
    assert declared == set(ref)                                    # nothing declared beyond SALUDA's keys
    for r in [ref[k] for k in ref]:
        assert 0 <= r <= 10


def test_cl10_raises_on_undeclared_ids():
    ns, sk = build_class_map('nuscenes_cl10'), build_class_map('semantickitti_cl10')
    for cm, bad in ((ns, [32]), (ns, [-1]), (sk, [2]), (sk, [260]), (sk, [65536])):
        with pytest.raises(ValueError):
            cm(np.array(bad))
    assert sk(np.array([31, 32, 49, 80])).tolist() == [IGNORE] * 4          # bicyclist, motorcyclist, other-ground, pole
    assert ns(np.array([9, 12, 28])).tolist() == [IGNORE] * 3               # barrier, traffic cone, manmade


# --- the SemanticKITTI reader ------------------------------------------------------------------------------------------

def _fake_sk(root, seq, n=5, pts=50, seed=0):
    rs = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, seq, 'velodyne'))
    os.makedirs(os.path.join(root, seq, 'labels'))
    sem = np.array(sorted(SALUDA_SK), dtype=np.uint32)
    out = []
    for f in range(n):
        p = np.c_[rs.uniform(-50, 50, (pts, 3)), np.round(rs.uniform(0, 0.99, pts), 2)].astype(np.float32)
        s = sem[rs.integers(0, len(sem), pts)]
        inst = rs.integers(0, 3, pts).astype(np.uint32) << 16           # instance ids in the upper 16 bits
        p.tofile(os.path.join(root, seq, 'velodyne', '%06d.bin' % f))
        (s | inst).astype(np.uint32).tofile(os.path.join(root, seq, 'labels', '%06d.label' % f))
        out.append((p, s.astype(np.int64)))
    return out


def test_semantickitti_reader_masks_the_instance_bits_and_checks_counts(tmp_path):
    from pcseg import semantickitti_seg as sk
    root = str(tmp_path)
    ref = _fake_sk(root, '04')
    for f, (p, s) in enumerate(ref):
        pts, raw = sk.read_frame(root, '04', f)
        assert np.array_equal(pts, p) and np.array_equal(raw, s)
    assert sk.list_frames(root, '04') == list(range(5))
    np.arange(3, dtype=np.uint32).tofile(os.path.join(root, '04', 'labels', '000002.label'))
    with pytest.raises(ValueError):
        sk.read_frame(root, '04', 2)
    os.remove(os.path.join(root, '04', 'velodyne', '000001.bin'))
    with pytest.raises(ValueError):
        sk.list_frames(root, '04')


def test_semantickitti_dataset_loads_samples_with_frame_step(tmp_path):
    from pcseg.semantickitti_seg import SemanticKittiSeg
    root = str(tmp_path)
    ref = _fake_sk(root, '08', n=6)
    cfg = {'ROOT': root, 'SEQUENCES': ['08'], 'SHIFT_COOR': [0.0, 0.0, 0.0], 'CLASS_MAP': 'semantickitti_cl10',
           'GRID_SIZE': 0.05, 'FRAME_STEP': 2}
    ds = SemanticKittiSeg(cfg, False)
    assert ds.frames == [('08', 0), ('08', 2), ('08', 4)] and ds.sequences == ['08']
    coord, strength, seg, key = ds.load(1)
    p, s = ref[2]
    assert key == 'sk/08/000002' and np.array_equal(coord, p[:, :3]) and np.array_equal(strength[:, 0], p[:, 3])
    assert np.array_equal(seg, build_class_map('semantickitti_cl10')(s))
    sample = ds[1]
    assert sample['feat'].shape[1] == 4 and len(sample['origin_segment']) == len(p)
    assert set(np.unique(sample['segment'])) <= set(np.unique(seg))


@pytest.mark.skipif(not os.path.isdir(SK_ROOT), reason='SemanticKITTI absent')
def test_semantickitti_real_scans_over_44_frames():
    from pcseg import semantickitti_seg as sk
    cm = build_class_map('semantickitti_cl10')
    for seq in ['%02d' % i for i in range(11)]:
        frames = sk.list_frames(SK_ROOT, seq)
        for f in (frames[0], frames[len(frames) // 3], frames[2 * len(frames) // 3], frames[-1]):
            pts, raw = sk.read_frame(SK_ROOT, seq, f)
            assert len(pts) > 50000 and np.isfinite(pts).all()
            v = pts[:, 3].astype(np.float64)
            assert v.min() >= 0 and v.max() <= 1 and np.all(np.abs(v * 100 - np.round(v * 100)) < 1e-3)
            y = cm(raw)                                                     # raises on an undeclared id
            assert y.min() >= IGNORE and y.max() < 10


# --- factor ops with a sensor name -------------------------------------------------------------------------------------

def _write_identity(path, frm, to, step, ints=False):
    edges = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0])
    q = np.tile(np.linspace(0.0, 1.0, 201), (len(edges) - 1, 1))
    if ints:
        np.savez(path, edges=edges, from_q=q, to_q=q, from_step=step, from_device=frm, to_device=to)
    else:
        np.savez(path, edges=edges, from_q=q, to_q=q, from_step=step, from_sensor=frm, to_sensor=to)
    return path


def test_factor_ops_sensor_names_intensity_only(tmp_path):
    from pcseg.factor_ops import FactorOps
    t = _write_identity(str(tmp_path / 't.npz'), 'nuscenes', 'semantickitti', 0.0)
    ops = FactorOps({'INTENSITY': {'TABLE': t}}, 'nuscenes', [0, 0, 0])
    coord = np.random.default_rng(0).uniform(-40, 40, (500, 3)).astype(np.float32)
    v = np.random.default_rng(1).uniform(0, 1, 500).astype(np.float32)
    idx, out = ops(coord, v, 'tok')
    assert np.array_equal(idx, np.arange(500)) and np.allclose(out, v, atol=1e-6)
    assert 'onto device semantickitti' in ops.describe()
    with pytest.raises(ValueError):
        FactorOps({'INTENSITY': {'TABLE': t}}, 'semantickitti', [0, 0, 0])      # the table maps another sensor
    with pytest.raises(ValueError):
        FactorOps({'CONE': {'FOV_DEG': 60.0}}, 'nuscenes', [0, 0, 0])            # P2: INTENSITY only
    ti = _write_identity(str(tmp_path / 'ti.npz'), 0, 1, 0.0, ints=True)
    with pytest.raises(ValueError):
        FactorOps({'INTENSITY': {'TABLE': ti}}, 'nuscenes', [0, 0, 0])           # no from_sensor
    with pytest.raises(ValueError):
        FactorOps({'INTENSITY': {'TABLE': t}}, 0, [0, 0, 0])                     # PandaSet device: no from_device
    assert FactorOps({'INTENSITY': {'TABLE': ti}}, 0, [0, 0, 0]).describe().endswith('onto device 1, step 0')


# --- rule 8: the identity table through the real loaders, >= 40 frames each ----------------------------------------

def _p2_cfg(name):
    sys.path.insert(0, TOOLS)
    import train_seg
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        return train_seg.load_cfg('cfgs/pcseg/%s.yaml' % name)
    finally:
        os.chdir(cwd)


def _datasets(cfg, block=None):
    sys.path.insert(0, TOOLS)
    import train_seg
    import copy
    c = copy.deepcopy(cfg)
    if block is not None:
        c.DATA.EVAL_PERTURB = block                                      # on the VAL set: no augmentation, fixed order
    return train_seg.build_datasets(c)['val']


@pytest.mark.parametrize('cfg_name,sensor,step', [('spunet-p2-nuscenes', 'nuscenes', 1 / 255.0),
                                                  ('spunet-p2-semantickitti', 'semantickitti', 0.01)])
def test_identity_table_leaves_40_frames_unchanged(tmp_path, cfg_name, sensor, step):
    root = NUSCENES if sensor == 'nuscenes' else SK_ROOT
    if not os.path.isdir(root):
        pytest.skip('%s absent' % sensor)
    cfg = _p2_cfg(cfg_name)
    base = _datasets(cfg)
    exact = _datasets(cfg, {'INTENSITY': {'TABLE': _write_identity(str(tmp_path / 'e.npz'), sensor, 'x', 0.0)}})
    deq = _datasets(cfg, {'INTENSITY': {'TABLE': _write_identity(str(tmp_path / 'd.npz'), sensor, 'x', step)}})
    idxs = np.linspace(0, len(base) - 1, 40).round().astype(int)
    for i in idxs:
        cb, ib, sb, kb = base.load(int(i))
        ce, ie, se, ke = exact.load(int(i))
        cd, id_, sd, kd = deq.load(int(i))
        assert kb == ke == kd
        assert np.array_equal(cb, ce) and np.array_equal(cb, cd)
        assert np.array_equal(sb, se) and np.array_equal(sb, sd)
        assert np.allclose(ie, ib, rtol=0, atol=1e-6)
        assert np.all(np.abs(id_ - ib) <= step / 2 + 1e-6)
    # the dequantisation draw is keyed by the frame: the same frame twice gives the same intensities
    assert np.array_equal(deq.load(int(idxs[3]))[1], deq.load(int(idxs[3]))[1])


# --- configs -------------------------------------------------------------------------------------------------------------

def _strip(cfg):
    cfg.pop('TAG', None)
    cfg.pop('_BASE_CONFIG_', None)
    return cfg


def test_p2_nuscenes_is_g0_but_the_class_map():
    a, b = _strip(_p2_cfg('spunet-p2-nuscenes')), _strip(_p2_cfg('spunet-nuscenes'))
    assert a.DATA.CLASS_MAP == 'nuscenes_cl10' and b.DATA.CLASS_MAP == 'nuscenes_lidarseg16'
    da, db = dict(a.DATA), dict(b.DATA)
    da.pop('CLASS_MAP'), db.pop('CLASS_MAP')
    assert da == db
    ra, rb = dict(a), dict(b)
    ra.pop('DATA'), rb.pop('DATA')
    assert ra == rb


def test_one_recipe_on_both_sides():
    n, k = _p2_cfg('spunet-p2-nuscenes'), _p2_cfg('spunet-p2-semantickitti')
    assert n.MODEL == k.MODEL
    for key in ('GRID_SIZE', 'SHIFT_COOR', 'AUG'):
        assert n.DATA[key] == k.DATA[key]
    on, ok = dict(n.OPTIMIZATION), dict(k.OPTIMIZATION)
    for key in ('NUM_EPOCHS', 'SAVE_EVERY'):
        on.pop(key), ok.pop(key)
    assert on == ok
    assert n.TARGET_DUMP_FRAMES == k.TARGET_DUMP_FRAMES
    assert k.DATA.CLASS_MAP == 'semantickitti_cl10' and k.DATA.SPLITS.val == ['08']
    assert k.DATA.SPLITS.train == ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']


def test_step_budget():
    n, k = _p2_cfg('spunet-p2-nuscenes'), _p2_cfg('spunet-p2-semantickitti')
    assert n.OPTIMIZATION.NUM_EPOCHS * (28130 // 12) == 117200
    it = 19130 // 12
    assert it == 1594 and k.OPTIMIZATION.NUM_EPOCHS == round(117200 / it) == 74
    assert k.EVAL.EVERY == round(11720 / it) == 7
    if os.path.isdir(SK_ROOT):
        from pcseg.semantickitti_seg import list_frames
        assert sum(len(list_frames(SK_ROOT, s)) for s in k.DATA.SPLITS.train) == 19130
        assert sum(len(list_frames(SK_ROOT, s)) for s in k.DATA.SPLITS.val) == 4071


@pytest.mark.parametrize('arm,oracle,table', [
    ('spunet-p2-m-ns2sk', 'spunet-p2-nuscenes', 'p2_intensity_nuscenes_to_semantickitti_rings.npz'),
    ('spunet-p2-m-sk2ns', 'spunet-p2-semantickitti', 'p2_intensity_semantickitti_to_nuscenes_rings.npz')])
def test_method_row_is_its_oracle_plus_one_intensity_block(arm, oracle, table):
    a, b = _strip(_p2_cfg(arm)), _strip(_p2_cfg(oracle))
    assert dict(a.DATA.FACTOR_SWAP) == {'INTENSITY': {'TABLE': os.path.join(TABLE_DIR, table)}}
    assert 'EVAL_PERTURB' not in a.DATA
    da, db = dict(a.DATA), dict(b.DATA)
    da.pop('FACTOR_SWAP')
    assert da == db
    ra, rb = dict(a), dict(b)
    ra.pop('DATA'), rb.pop('DATA')
    assert ra == rb


class _Recorder:
    def __init__(self, cfg, training, logger=None):
        self.cfg, self.training = cfg, training
        self.sequences = (list(cfg['SEQUENCES']) if 'SEQUENCES' in cfg
                          else ['s%d' % i for i in range(700 if training else 150)])
        self.skipped = []


@pytest.mark.parametrize('arm,module,cls', [('spunet-p2-m-ns2sk', 'pcseg.nuscenes_seg', 'NuScenesSeg'),
                                            ('spunet-p2-m-sk2ns', 'pcseg.semantickitti_seg', 'SemanticKittiSeg')])
def test_factor_swap_reaches_the_train_set_only(monkeypatch, arm, module, cls):
    sys.path.insert(0, TOOLS)
    import importlib
    import train_seg
    monkeypatch.setattr(importlib.import_module(module), cls, _Recorder)
    sets = train_seg.build_datasets(_p2_cfg(arm))
    assert 'FACTOR_OPS' in sets['train'].cfg and 'FACTOR_OPS' not in sets['val'].cfg
    assert dict(sets['train'].cfg['FACTOR_OPS']) == dict(_p2_cfg(arm).DATA.FACTOR_SWAP)
    sets = train_seg.build_datasets(_p2_cfg(arm.replace('m-ns2sk', 'nuscenes').replace('m-sk2ns', 'semantickitti')))
    assert 'FACTOR_OPS' not in sets['train'].cfg and 'FACTOR_OPS' not in sets['val'].cfg


SMOKE_OPT = {'NUM_EPOCHS': 2, 'SAVE_EVERY': 1, 'MAX_CKPT': 2, 'LOG_EVERY': 2}


@pytest.mark.parametrize('row', ['spunet-p2-nuscenes', 'spunet-p2-semantickitti', 'spunet-p2-m-ns2sk',
                                 'spunet-p2-m-sk2ns'])
def test_smoke_equals_its_row_but_the_smoke_overrides(row):
    s, a = _strip(_p2_cfg(row + '-smoke')), _strip(_p2_cfg(row))
    nus = a.DATA.NAME == 'NuScenesSeg'
    over = ('SCENES', 'EXPECT_SEQUENCES') if nus else ('SPLITS', 'FRAME_STEP', 'EXPECT_SEQUENCES')
    if nus:
        assert s.DATA.SCENES == {'train': ['scene-0001', 'scene-0002'], 'val': ['scene-0003']}
    else:
        assert s.DATA.SPLITS == {'train': ['04'], 'val': ['08']} and s.DATA.FRAME_STEP == {'train': 1, 'val': 100}
    ds, da = dict(s.DATA), dict(a.DATA)
    for k in over:
        ds.pop(k, None), da.pop(k, None)
    assert ds == da
    os_, oa = dict(s.OPTIMIZATION), dict(a.OPTIMIZATION)
    for k, v in SMOKE_OPT.items():
        assert os_.pop(k) == v
        oa.pop(k)
    assert os_ == oa and s.OPTIMIZATION.BATCH_SIZE == 12
    assert s.EVAL.EVERY == 1 and s.MODEL == a.MODEL


# --- the tables script ---------------------------------------------------------------------------------------------------

def _tables_mod():
    spec = importlib.util.spec_from_file_location(
        'pcseg_p2_intensity_tables', os.path.join(TOOLS, 'analysis', 'pcseg_p2_intensity_tables.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_tables_script_format_check_and_histogram():
    m = _tables_mod()
    assert m.format_ok('nuscenes', np.array([0, 1, 37, 255]) / 255.0)
    assert not m.format_ok('nuscenes', np.array([0.5 / 255]))
    assert m.format_ok('semantickitti', np.array([0.0, 0.29, 0.99], np.float32))
    assert not m.format_ok('semantickitti', np.array([0.295]))
    assert not m.format_ok('semantickitti', np.array([1.01]))
    p = np.array([[5, 0, 0, 0.5], [0, 15, 0, 0.5], [400, 0, 0, 0.5]], np.float32)
    h, r = m.frame_hist('semantickitti', p, 'k', 0.01)
    assert h.shape[0] == 10 and h.sum() == 3
    assert h[0].sum() == 1 and h[1].sum() == 1 and h[9].sum() == 1          # 5 m, 15 m, and 400 m in the last ring
    assert m.TABLES[('nuscenes', 'semantickitti')] == os.path.basename(
        _p2_cfg('spunet-p2-m-ns2sk').DATA.FACTOR_SWAP.INTENSITY.TABLE)
    assert m.TABLES[('semantickitti', 'nuscenes')] == os.path.basename(
        _p2_cfg('spunet-p2-m-sk2ns').DATA.FACTOR_SWAP.INTENSITY.TABLE)
