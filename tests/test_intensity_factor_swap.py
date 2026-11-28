"""TRAIN_FACTOR_SWAP.INTENSITY_MAP (pcdet/datasets/intensity_swap.py; experiments_md 20261003_05 §5 item 72, design fixed
2026-11-29 00:15 JST). The arm must be the evaluation-time stage's map (same table, same draw -> same values), change
the intensity column only, be fixed per frame, sit in all three loaders' training-only factor-swap path and nowhere else
(EVAL_FACTOR_SWAP and Lyft refuse it), and its configs must differ from their oracles by the key only. CPU only; run
from the repo root."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'tools'))
from pcdet.datasets import intensity_swap as isw  # noqa: E402
from pcdet.datasets import loader_factor_swap as lfs  # noqa: E402
from pcdet.datasets.kitti import kitti_factor_swap as kfs  # noqa: E402
from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402
from pcdet.datasets.waymo import waymo_factor_swap as wfs  # noqa: E402

REAL = Path('/home/koyama/data/intensity_tables')


def table(tmp_path, step, name='t.npz', identity=False):
    """A 3-ring table with plateaus (repeated quantiles, as quantized intensities give) and a skewed target."""
    edges = np.array([0.0, 10.0, 30.0, 75.0])
    lv = np.linspace(0.0, 1.0, 201)
    from_q = np.stack([np.round(lv ** 2, 2), np.round(lv, 2), np.minimum(lv * 2, 1.0)])
    to_q = from_q.copy() if identity else np.stack([lv ** 0.5, lv * 0.3, np.sqrt(lv) * 0.8])
    p = tmp_path / name
    np.savez(p, edges=edges, from_q=from_q, to_q=to_q, from_step=step)
    return p


def cloud(n=3000, cols=5, seed=0):
    rng = np.random.RandomState(seed)
    pts = np.zeros((n, cols), dtype=np.float32)
    pts[:, 0:2] = rng.uniform(-80, 80, (n, 2))
    pts[:, 2] = rng.uniform(-2, 2, n)
    pts[:, 3] = np.round(rng.uniform(0, 1, n), 2)
    if cols > 4:
        pts[:, 4] = rng.uniform(0, 1, n)                       # elongation / timestamp: must stay
    return pts


@pytest.mark.parametrize('step', [0.0, 0.01])
def test_map_equals_the_evaluation_time_stage_for_the_same_draw(tmp_path, step):
    p = table(tmp_path, step)
    pts = cloud()
    t = isw.load_tables(p)
    ref = DataProcessor.map_intensity_to_reference(None, data_dict={'points': pts.copy()},
                                                   config=EasyDict(INTENSITY_INDEX=3), tables=t)['points']
    mine = isw.map_values(pts[:, 3], np.linalg.norm(pts[:, 0:2], axis=1), t, np.random.RandomState(len(pts)))
    assert np.array_equal(mine.astype(np.float32), ref[:, 3])


@pytest.mark.skipif(not (REAL / 'nuscenes2kitti_rings.npz').exists(), reason='no real tables here')
def test_real_kitti_table_matches_the_stage_and_moves_toward_nuscenes():
    t = isw.load_tables(REAL / 'nuscenes2kitti_rings.npz')
    pts = cloud(cols=4)
    ref = DataProcessor.map_intensity_to_reference(None, data_dict={'points': pts.copy()},
                                                   config=EasyDict(INTENSITY_INDEX=3), tables=t)['points']
    mine = isw.map_values(pts[:, 3], np.linalg.norm(pts[:, 0:2], axis=1), t, np.random.RandomState(len(pts)))
    assert np.array_equal(mine.astype(np.float32), ref[:, 3])
    assert t['from_step'] == pytest.approx(0.01)               # the direction: from = KITTI (step 0.01), to = nuScenes
    w = isw.load_tables(REAL / 'nuscenes2waymo_rings.npz')
    assert np.array_equal(t['to_q'], w['to_q']) and w['from_step'] == 0.0   # one nuScenes side for both targets


def test_only_the_intensity_column_changes_and_the_frame_fixes_the_draw(tmp_path):
    cfg = EasyDict(INTENSITY_MAP=EasyDict(TABLES=str(table(tmp_path, 0.01))))
    pts = cloud()
    a = isw.apply_points(pts, cfg, '000123')
    assert a is not pts and a.dtype == pts.dtype and np.array_equal(pts, cloud())      # input untouched
    assert np.array_equal(a[:, [0, 1, 2, 4]], pts[:, [0, 1, 2, 4]])
    assert not np.array_equal(a[:, 3], pts[:, 3]) and a[:, 3].min() >= 0 and a[:, 3].max() <= 1
    assert np.array_equal(a, isw.apply_points(pts, cfg, '000123'))                       # every epoch, every seed
    assert not np.array_equal(a, isw.apply_points(pts, cfg, '000124'))                   # frame-seeded dequantization
    assert isw.apply_points(pts[:0], cfg, 'x').shape == (0, 5)


def test_identity_table_reproduces_the_intensities(tmp_path):
    cfg = EasyDict(INTENSITY_MAP=EasyDict(TABLES=str(table(tmp_path, 0.0, 'id.npz', identity=True))))
    pts = cloud()
    a = isw.apply_points(pts, cfg, 'f')
    assert np.array_equal(a[:, [0, 1, 2, 4]], pts[:, [0, 1, 2, 4]])
    assert np.abs(a[:, 3] - pts[:, 3]).max() <= 1e-6


def test_ring_is_horizontal_radius_and_beyond_the_last_edge_takes_the_last_ring(tmp_path):
    p = table(tmp_path, 0.0)
    t = isw.load_tables(p)
    v = np.array([0.5, 0.5, 0.5, 0.5])
    out = isw.map_values(v, np.array([5.0, 20.0, 50.0, 120.0]), t, None)
    lv = np.linspace(0.0, 1.0, 201)
    xs0 = np.maximum.accumulate(np.round(lv ** 2, 2)) + np.arange(201) * 1e-12            # ring 0's own map
    assert out[0] == pytest.approx(np.interp(np.interp(0.5, xs0, lv), lv, lv ** 0.5), abs=1e-6)
    xs1 = np.maximum.accumulate(np.round(lv, 2)) + np.arange(201) * 1e-12
    assert out[1] == pytest.approx(np.interp(np.interp(0.5, xs1, lv), lv, lv * 0.3), abs=1e-6)   # ring 1's map
    assert out[2] == pytest.approx(out[3])                                                 # beyond 75 m = ring 2
    assert len({round(float(x), 4) for x in out[:3]}) == 3                                  # three rings, three maps
    pts = np.array([[3.0, 4.0, 100.0, 0.5]], dtype=np.float32)                            # z never enters the radius
    cfg = EasyDict(INTENSITY_MAP=EasyDict(TABLES=str(p)))
    assert isw.apply_points(pts, cfg, 'f')[0, 3] == pytest.approx(out[0], abs=1e-6)


def test_the_three_loader_paths_apply_it(tmp_path):
    cfg = EasyDict(INTENSITY_MAP=EasyDict(TABLES=str(table(tmp_path, 0.01))))
    pts = cloud()
    want = isw.apply_points(pts, cfg, 'k')
    assert kfs.arm_of(cfg) == 'INTENSITY_MAP' and np.array_equal(kfs.apply_points(pts, cfg, 'k'), want)
    assert wfs.arm_of(cfg) == 'INTENSITY_MAP' and np.array_equal(wfs.apply_points(pts, cfg, 'k'), want)
    d = lfs.apply({'points': pts, 'gt_boxes': np.zeros((1, 8), dtype=np.float32), 'gt_names': np.array(['Car'])},
                  cfg, ['Car'], {}, 'k', 'pandaset')
    assert np.array_equal(d['points'], want)
    assert np.array_equal(lfs.apply({'points': pts}, cfg, ['Car'], {}, 'k', 'pandaset')['points'], want)  # no boxes
    assert np.array_equal(kfs.apply_boxes(np.ones((2, 7), np.float32), ['Car', 'Car'], cfg, 'k'), np.ones((2, 7)))
    assert np.array_equal(wfs.apply_boxes(np.ones((2, 7), np.float32), ['Car', 'Car'], cfg, 'k'), np.ones((2, 7)))


def test_refused_where_it_does_not_belong(tmp_path):
    cfg = EasyDict(INTENSITY_MAP=EasyDict(TABLES=str(table(tmp_path, 0.0))))
    assert 'INTENSITY_MAP' not in kfs.POINT_ARMS                 # so EVAL_FACTOR_SWAP refuses it (kitti_dataset)
    with pytest.raises(AssertionError):
        lfs.arm_of(cfg, 'lyft')                                  # Lyft: constant intensity
    with pytest.raises(AssertionError):
        kfs.arm_of(EasyDict(INTENSITY_MAP=cfg.INTENSITY_MAP, RANGE_KEEP=EasyDict(EDGES=[0, 1], RATE=[1.0])))
    with pytest.raises(AssertionError):
        wfs.arm_of(EasyDict(INTENSITY_MAP=EasyDict(TABLES='x', SCALE=2)))
    with pytest.raises(AssertionError):
        wfs.arm_of(EasyDict(INTENSITY_MAP=EasyDict()))


def test_loader_hooks_are_training_gated():
    """INTENSITY_MAP rides the existing TRAIN_FACTOR_SWAP read in each loader, which is gated on self.training."""
    for f, needle in (('pcdet/datasets/kitti/kitti_dataset.py', 'kitti_factor_swap.apply_points(points, swap'),
                      ('pcdet/datasets/waymo/waymo_dataset.py', 'waymo_factor_swap.apply_points(points, swap'),
                      ('pcdet/datasets/pandaset/pandaset_dataset.py', 'loader_factor_swap.apply(input_dict, swap')):
        src = (REPO / f).read_text()
        gate = "swap = self.dataset_cfg.get('TRAIN_FACTOR_SWAP', None) if self.training else None"
        assert gate in src and src.index(gate) < src.index(needle), f


def _flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, pre + k + '.'))
        else:
            out[pre + k] = json.dumps(v, sort_keys=True, default=str)
    return out


def _diff(a, b):
    fa, fb = _flat(a), _flat(b)                                  # the _BASE_CONFIG_ paths themselves are not settings
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k) and not k.endswith('_BASE_CONFIG_'))


ROWS = [  # swap config, its oracle config, the oracle's Fig. 1 base (None: the oracle config already exists)
    ('centerpoint-item72-kitti-imap.yaml', 'cfgs/da-ieee-access-analysis/item72/centerpoint-item72-kitti-oracle-xyzi.yaml',
     'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml'),
    ('centerpoint-item72-waymo-imap.yaml', 'cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo-xyzi.yaml',
     'cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml'),
    ('centerpoint-item72-pandaspin-imap.yaml', 'cfgs/da-ieee-access/centerpoint-pandaset-spin2spin-xyzi.yaml',
     'cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml'),
]


@pytest.mark.parametrize('row', ROWS, ids=[r[0] for r in ROWS])
def test_configs_differ_from_their_oracles_by_the_key_only(row):
    from pcdet.config import cfg_from_yaml_file
    os.chdir(REPO / 'tools')
    swap = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/item72/' + row[0], EasyDict())
    oracle = cfg_from_yaml_file(row[1], EasyDict())
    base = cfg_from_yaml_file(row[2], EasyDict())
    assert _diff(swap, oracle) == ['DATA_CONFIG.TRAIN_FACTOR_SWAP.INTENSITY_MAP.TABLES']
    assert {k.rsplit('.', 1)[0] for k in _diff(oracle, base)} == {'DATA_CONFIG.POINT_FEATURE_ENCODING',
                                                                  'DATA_CONFIG_TAR.POINT_FEATURE_ENCODING'}
    for blk in ('DATA_CONFIG', 'DATA_CONFIG_TAR'):
        assert list(oracle[blk].POINT_FEATURE_ENCODING.used_feature_list) == ['x', 'y', 'z', 'intensity']
    assert swap.DATA_CONFIG.TRAIN_FACTOR_SWAP.INTENSITY_MAP.TABLES.startswith('/storage/intensity_tables/nuscenes2')
    assert (swap.DATA_CONFIG.get('MAX_SWEEPS', 1) or 1) <= 1 and 'TRAIN_FACTOR_SWAP' not in swap.DATA_CONFIG_TAR
