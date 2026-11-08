"""Item 43 (experiments_md 20261003_05 §5 item 43, design fixed 2026-11-09 01:18 JST): dense -> sparse training-time swaps on
nuScenes by accumulation. CPU only, no dataset files.

- MAX_SWEEPS_BY_VEHICLE (nuscenes_dataset.sweeps_for_vehicle): absent = MAX_SWEEPS for every frame; present = the frame's
  capture vehicle's depth, an unknown vehicle or a depth above MAX_SWEEPS raises, and SWEEP_COUNT_MIX is refused beside it.
- The generated configs (analysis/make_item43_configs.py) are on disk as generated, and each is its parent plus ONLY the
  intended change: L'_K = 27468's config + the count-matching arm on both platform blocks + the oracle's DATA_CONFIG_TAR;
  the reference configs = the oracle recipe + VEHICLE; the own-input twins = the oracle + an accumulated DATA_CONFIG_TAR at
  the training config's per-platform depths.
- The arm's processor at rate 1 is EXACTLY the teacher block's processor (points, order and labels; bug rule 8), and at a
  rate below 1 it thins (the test fails without the arm), also in a bin the reference barely reaches (amendment A7 of
  experiments_md 20261109_01: with the default MIN_HIST_BIN_FRACTION 0.01 the 0-1.5 m bin kept every accumulated point, and
  the first gate, job 32416, read n008 DIFFER at 0-5 m).
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'analysis'))
import _init_path  # noqa: E402,F401
import make_item43_configs as m43  # noqa: E402
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset, sweeps_for_vehicle  # noqa: E402

_CACHE = {}


def resolved(path):
    if path not in _CACHE:
        _CACHE[path] = m43.plain(m43.load(path))
    return copy.deepcopy(_CACHE[path])


# ---- MAX_SWEEPS_BY_VEHICLE ----------------------------------------------------------------------------------------------

def test_sweeps_for_vehicle_absent_key_is_max_sweeps():
    assert sweeps_for_vehicle(EasyDict(MAX_SWEEPS=15), 'n008') == 15
    assert sweeps_for_vehicle(EasyDict(MAX_SWEEPS=1), None) == 1


def test_sweeps_for_vehicle_per_platform_and_its_guards():
    cfg = EasyDict(MAX_SWEEPS=15, MAX_SWEEPS_BY_VEHICLE={'n008': 15, 'n015': 10})
    assert (sweeps_for_vehicle(cfg, 'n008'), sweeps_for_vehicle(cfg, 'n015')) == (15, 10)
    with pytest.raises(KeyError):
        sweeps_for_vehicle(cfg, 'n009')
    with pytest.raises(AssertionError):
        sweeps_for_vehicle(EasyDict(MAX_SWEEPS=10, MAX_SWEEPS_BY_VEHICLE={'n008': 15, 'n015': 10}), 'n008')


def _shell(cfg, training=False):
    """A NuScenesDataset shell: only what __getitem__ reads before get_lidar_with_sweeps (as test_sweep_selection)."""
    ds = NuScenesDataset.__new__(NuScenesDataset)
    ds._merge_all_iters_to_one_epoch = False
    ds.infos = [{'lidar_path': 'x/a.bin', 'token': 'boston'}, {'lidar_path': 'x/b.bin', 'token': 'singapore'}]
    ds._vehicle_by_token = {'boston': 'n008', 'singapore': 'n015'}
    ds.training = training
    ds.dataset_cfg = EasyDict(cfg)
    seen = []

    class _Stop(Exception):
        pass

    def fake(index, max_sweeps=1):
        seen.append(max_sweeps)
        raise _Stop()
    ds.get_lidar_with_sweeps = fake
    return ds, seen, _Stop


def _depths(ds, stop):
    for i in range(len(ds.infos)):
        try:
            ds[i]
        except stop:
            pass


def test_getitem_reads_the_depth_of_the_frame_vehicle():
    ds, seen, stop = _shell({'MAX_SWEEPS': 15, 'MAX_SWEEPS_BY_VEHICLE': {'n008': 15, 'n015': 10}})
    _depths(ds, stop)
    assert seen == [15, 10]
    ds, seen, stop = _shell({'MAX_SWEEPS': 15})                      # absent key: unchanged
    _depths(ds, stop)
    assert seen == [15, 15]


def test_sweep_count_mix_is_refused_beside_the_key():
    ds, _, stop = _shell({'MAX_SWEEPS': 15, 'MAX_SWEEPS_BY_VEHICLE': {'n008': 15, 'n015': 10},
                          'SWEEP_COUNT_MIX': 'uniform'}, training=True)
    with pytest.raises(AssertionError, match='SWEEP_COUNT_MIX'):
        ds[0]


# ---- generated configs --------------------------------------------------------------------------------------------------

def test_configs_on_disk_are_the_generated_ones():
    for path, (header, body) in m43.outputs().items():
        assert (TOOLS / path).read_text() == m43.render(header, body), path


def _strip_arm(block):
    for k in list(m43.ARM) + ['HIST_DIST_REFERENCE_CFG']:
        block.pop(k)
    block['DATA_PROCESSOR'] = [p for p in block['DATA_PROCESSOR']
                               if p['NAME'] not in ('sample_points_hist_based', 'drop_empty_gt_boxes')]
    return block


def test_lprime_is_the_teacher_plus_the_arm_plus_the_oracle_eval_block():
    lp, teacher, oracle = resolved(m43.LPRIME), resolved(m43.TEACHER), resolved(m43.ORACLE)
    assert lp['DATA_CONFIG_TAR'] == oracle['DATA_CONFIG_TAR']
    assert list(lp['DATA_CONFIGS']) == list(teacher['DATA_CONFIGS']) == ['NUSCENES_N015', 'NUSCENES_N008']  # as 27468 logged
    for v, b in m43.BLOCK.items():
        blk = lp['DATA_CONFIGS'][b]
        assert blk['VEHICLE'] == v and blk['HIST_DIST_REFERENCE_CFG'] == m43.REF[v]
        assert blk['HIST_DIST_IN_GRID'] is True and blk['HIST_DIST_ON_THE_FLY'] is True
        assert (blk['HIST_DIST_FRAMES'], blk['HIST_DIST_BINS'], blk['HIST_DIST_MAX_DIST']) == (1000, 50, 75.0)  # 1.5-m bins
        assert [p['NAME'] for p in blk['DATA_PROCESSOR']] == [
            'mask_points_and_boxes_outside_range', 'shuffle_points', 'sample_points_hist_based', 'drop_empty_gt_boxes',
            'transform_points_to_voxels']
        assert _strip_arm(blk) == teacher['DATA_CONFIGS'][b]
    lp.pop('DATA_CONFIGS'), teacher.pop('DATA_CONFIGS'), lp.pop('DATA_CONFIG_TAR'), teacher.pop('DATA_CONFIG_TAR')
    assert lp == teacher                                                     # MODEL, OPTIMIZATION, CLASS_NAMES, ONTOLOGY
    assert (resolved(m43.TEACHER)['DATA_CONFIGS']['NUSCENES_N008']['MAX_SWEEPS'],
            resolved(m43.TEACHER)['DATA_CONFIGS']['NUSCENES_N015']['MAX_SWEEPS']) == (15, 10)


def test_reference_configs_are_the_oracle_on_one_platform():
    oracle = resolved(m43.ORACLE)
    for v in m43.BLOCK:
        ref = resolved(m43.REF[v])
        assert ref['DATA_CONFIG'].pop('VEHICLE') == v
        assert ref == oracle
        assert ref['DATA_CONFIG']['MAX_SWEEPS'] == 1


def test_gate_views_are_lprime_with_one_block():
    lp = resolved(m43.LPRIME)
    for v, b in m43.BLOCK.items():
        g = resolved(m43.GATE[v])
        assert list(g['DATA_CONFIGS']) == [b] and g['DATA_CONFIGS'][b] == lp['DATA_CONFIGS'][b]
        g.pop('DATA_CONFIGS')
        assert g == {k: x for k, x in lp.items() if k != 'DATA_CONFIGS'}


@pytest.mark.parametrize('train_cfg', list(m43.OWN))
def test_own_input_twin_is_the_oracle_with_the_training_depths(train_cfg):
    twin, oracle, tr = resolved(m43.OWN[train_cfg]), resolved(m43.ORACLE), resolved(train_cfg)
    t = twin['DATA_CONFIG_TAR']
    want = {v: tr['DATA_CONFIGS'][b]['MAX_SWEEPS'] for v, b in m43.BLOCK.items()}
    assert t.pop('MAX_SWEEPS_BY_VEHICLE') == want and t.pop('MAX_SWEEPS') == max(want.values())
    assert t.pop('INFO_PATH') == tr['DATA_CONFIGS']['NUSCENES_N008']['INFO_PATH']
    assert t.pop('GT_BOXES_MOTION_COMPENSATION') is True and t.pop('COMPENSATE_AT_EVAL') is True
    sh = [p for p in t['DATA_PROCESSOR'] if p['NAME'] == 'shuffle_points'][0]
    assert sh['SHUFFLE_ENABLED'] == {'train': True, 'test': True}
    sh['SHUFFLE_ENABLED'] = {'train': True, 'test': False}
    o = oracle['DATA_CONFIG_TAR']
    o.pop('MAX_SWEEPS'), o.pop('INFO_PATH')
    assert t == o
    twin.pop('DATA_CONFIG_TAR'), oracle.pop('DATA_CONFIG_TAR')
    assert twin == oracle
    assert tr['MODEL'] == oracle['MODEL'] and tr['CLASS_NAMES'] == oracle['CLASS_NAMES']  # the checkpoint's model


# ---- the arm itself: identity at rate 1, thinning below it --------------------------------------------------------------

def _processor(block, training=True):
    from pcdet.datasets.processor.data_processor import DataProcessor
    cfgs = [EasyDict(p) for p in block['DATA_PROCESSOR'] if p['NAME'] != 'transform_points_to_voxels']
    return DataProcessor(cfgs, point_cloud_range=np.array(block['POINT_CLOUD_RANGE'], dtype=np.float32),
                         training=training, num_point_features=3)


def _frame(seed):
    rng = np.random.default_rng(seed)
    r, az = rng.uniform(1, 74, 4000), rng.uniform(-np.pi, np.pi, 4000)
    pts = np.stack([r * np.cos(az), r * np.sin(az), rng.uniform(-1.5, 3.5, 4000)], 1).astype(np.float32)
    boxes = np.array([[10, 0, 0.5, 4, 2, 1.5, 0, 1], [30, 5, 0.5, 4, 2, 1.5, 0, 1], [-50, 20, 0.5, 0.8, 0.8, 1.7, 0, 2]],
                     dtype=np.float32)
    inside = (boxes[:, None, :3] + rng.uniform(-0.3, 0.3, (3, 20, 3))).reshape(-1, 3).astype(np.float32)
    return np.concatenate([pts, inside]), boxes


@pytest.mark.parametrize('vehicle', list(m43.BLOCK))
def test_arm_at_rate_one_is_exactly_the_teacher_block(vehicle):
    b = m43.BLOCK[vehicle]
    arm = _processor(resolved(m43.LPRIME)['DATA_CONFIGS'][b])
    base = _processor(resolved(m43.TEACHER)['DATA_CONFIGS'][b])
    ones = np.ones(50)
    arm.set_hist_dist(ones, ones, max_dist=75.0)                             # rate exactly 1 in every bin
    for s in range(40):
        pts, boxes = _frame(s)
        np.random.seed(100 + s)
        a = arm.forward({'points': pts.copy(), 'gt_boxes': boxes.copy()})
        np.random.seed(100 + s)
        c = base.forward({'points': pts.copy(), 'gt_boxes': boxes.copy()})
        assert np.array_equal(a['points'], c['points']) and np.array_equal(a['gt_boxes'], c['gt_boxes'])


@pytest.mark.parametrize('vehicle', list(m43.BLOCK))
def test_arm_below_rate_one_thins_and_recounts(vehicle):
    arm = _processor(resolved(m43.LPRIME)['DATA_CONFIGS'][m43.BLOCK[vehicle]])
    arm.set_hist_dist(np.full(50, 10.0), np.full(50, 1.0), max_dist=75.0)  # rate 0.1 everywhere
    np.random.seed(0)
    pts, boxes = _frame(0)
    out = arm.forward({'points': pts.copy(), 'gt_boxes': boxes.copy()})
    assert 0.05 < len(out['points']) / len(pts) < 0.15
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    held = roiaware_pool3d_utils.points_in_boxes_cpu(out['points'][:, :3], out['gt_boxes'][:, :7]).sum(1)
    assert (held >= 1).all()                                                 # no positive left on empty space


@pytest.mark.parametrize('vehicle', list(m43.BLOCK))
def test_arm_thins_a_bin_the_reference_barely_reaches(vehicle):
    blk = resolved(m43.LPRIME)['DATA_CONFIGS'][m43.BLOCK[vehicle]]
    step = [p for p in blk['DATA_PROCESSOR'] if p['NAME'] == 'sample_points_hist_based'][0]
    assert step['MIN_HIST_BIN_FRACTION'] == 0.0
    arm = _processor(blk)
    src, ref = np.full(50, 6500.0), np.full(50, 430.0)   # job 32416, n008: 325,525 / 21,547 in-grid points per frame
    src[0], ref[0] = 2204.0, 1.0      # 0-1.5 m: accumulated 2,204, single sweep ~1, under 1% of the reference's mean bin
    arm.set_hist_dist(src, ref, max_dist=75.0)
    np.random.seed(0)
    az = np.random.uniform(-np.pi, np.pi, 4000)
    near = np.stack([1.2 * np.cos(az), 1.2 * np.sin(az), np.zeros(4000)], 1).astype(np.float32)
    out = arm.forward({'points': near, 'gt_boxes': np.zeros((0, 8), dtype=np.float32)})
    assert len(out['points']) < 20    # rate 1 / 2204, not 1 (the guard would keep all 4,000)
