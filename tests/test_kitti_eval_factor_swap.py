"""EVAL_FACTOR_SWAP, the evaluation-time twin of a KITTI TRAIN_FACTOR_SWAP point arm (pcdet/datasets/kitti/kitti_dataset.py;
experiments_md 20261003_05 §5 item 33e: score the R3 density oracles on val cut the way they were trained). CPU only,
synthetic frames, no dataset files.

The loader is run on a bare KittiDataset instance and stopped at get_calib, right after the point hooks, and checked for:
  - key absent: the scan is unchanged in evaluation;
  - evaluation: the scan equals kitti_factor_swap.apply_points on the same frame (same function, same frame-seeded draw);
  - training: the evaluation key does nothing;
  - RATE 1 everywhere is the identity;
  - a box or label arm is refused;
  - the two item-33e configs carry R3's RANGE_KEEP exactly and its square root per bin.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import _init_path  # noqa: E402,F401
from pcdet.datasets.kitti import kitti_factor_swap  # noqa: E402

SAMPLE_IDX = '000042'
EDGES = [0, 1.5, 3, 4.5, 6, 30, 75]
RATE = [1, 0.53, 0.38, 0.19, 0.2, 0.25]
S2 = TOOLS / 'cfgs/da-ieee-access-analysis/factorswap-s2'


def scan(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    r, az = rng.uniform(0.5, 80.0, n), rng.uniform(-np.pi, np.pi, n)
    return np.stack([r * np.cos(az), r * np.sin(az), rng.uniform(-2, 1, n), rng.random(n)], 1).astype(np.float32)


class _Stop(Exception):
    pass


def loaded_points(monkeypatch, points, cfg, training=False):
    """The points KittiDataset.__getitem__ holds when it reaches get_calib, plus how often apply_points ran."""
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    calls, seen = [], {}
    real = kitti_factor_swap.apply_points

    def spy(pts, swap, frame_id):
        out = real(pts, swap, frame_id)
        calls.append(frame_id)
        seen['points'] = out
        return out
    monkeypatch.setattr(kitti_factor_swap, 'apply_points', spy)
    ds = object.__new__(KittiDataset)
    ds._merge_all_iters_to_one_epoch, ds.training = False, training
    ds.kitti_infos = [{'point_cloud': {'lidar_idx': SAMPLE_IDX}}]
    ds.get_lidar = lambda idx: points

    def stop(idx):
        raise _Stop()
    ds.get_calib = stop
    ds.dataset_cfg = EasyDict(cfg)
    with pytest.raises(_Stop):
        ds[0]
    return seen.get('points', points), len(calls)


def range_keep(rate=RATE):
    return {'EVAL_FACTOR_SWAP': {'RANGE_KEEP': {'EDGES': EDGES, 'RATE': list(rate)}}}


def test_absent_key_leaves_the_eval_scan_unchanged(monkeypatch):
    pts = scan()
    out, n = loaded_points(monkeypatch, pts, {})
    assert n == 0 and out is pts


def test_eval_applies_the_training_cut_with_the_same_draw(monkeypatch):
    pts = scan()
    cfg = range_keep()
    out, n = loaded_points(monkeypatch, pts, cfg)
    want = kitti_factor_swap.apply_points(pts, EasyDict(cfg['EVAL_FACTOR_SWAP']), SAMPLE_IDX)
    assert n == 1 and np.array_equal(out, want)
    assert 0.15 < len(out) / len(pts) < 0.45
    again, _ = loaded_points(monkeypatch, pts, cfg)
    assert np.array_equal(again, out)


def test_training_ignores_the_eval_key(monkeypatch):
    pts = scan()
    out, n = loaded_points(monkeypatch, pts, range_keep(), training=True)
    assert n == 0 and out is pts


def test_rate_one_is_the_identity(monkeypatch):
    pts = scan()
    out, n = loaded_points(monkeypatch, pts, range_keep([1] * len(RATE)))
    assert n == 1 and np.array_equal(out, pts)


def test_box_and_label_arms_are_refused(monkeypatch):
    for arm in ({'GT_DIM_SCALE': {'Car': 1.165}}, {'GT_IGNORE_BY_RANGE': {'EDGES': EDGES, 'KEEP': RATE}}):
        with pytest.raises(AssertionError, match='point arm'):
            loaded_points(monkeypatch, scan(), {'EVAL_FACTOR_SWAP': arm})


def test_halfway_cut_contains_the_training_cut(monkeypatch):
    pts = scan()
    full, _ = loaded_points(monkeypatch, pts, range_keep())
    half, _ = loaded_points(monkeypatch, pts, range_keep([math.sqrt(r) for r in RATE]))
    as_set = lambda a: {tuple(p) for p in a}
    assert len(full) < len(half) < len(pts) and as_set(full) <= as_set(half)


def test_item_33e_configs_carry_r3s_cut():
    r3 = yaml.safe_load(open(S2 / 'centerpoint-factorswap-s2-r3-density.yaml'))['DATA_CONFIG']['TRAIN_FACTOR_SWAP']
    own = yaml.safe_load(open(S2 / 'centerpoint-factorswap-s2-eval-r3-density.yaml'))['DATA_CONFIG_TAR']['EVAL_FACTOR_SWAP']
    half = yaml.safe_load(open(S2 / 'centerpoint-factorswap-s2-eval-r3-halfway.yaml'))['DATA_CONFIG_TAR']['EVAL_FACTOR_SWAP']
    assert own == r3
    assert half['RANGE_KEEP']['EDGES'] == r3['RANGE_KEEP']['EDGES']
    assert np.allclose(half['RANGE_KEEP']['RATE'], np.sqrt(r3['RANGE_KEEP']['RATE']), atol=5e-5)


def without_base_keys(d):
    """The resolved config without the `_BASE_CONFIG_` bookkeeping keys, which name the file a block came from."""
    if isinstance(d, dict):
        return {k: without_base_keys(v) for k, v in d.items() if k != '_BASE_CONFIG_'}
    if isinstance(d, list):
        return [without_base_keys(v) for v in d]
    return d


def test_item_33e_configs_resolve_to_the_oracle_base_plus_the_key(monkeypatch):
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(TOOLS)
    base = cfg_from_yaml_file('cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml', EasyDict())
    for name in ('eval-r3-density', 'eval-r3-halfway'):
        twin = cfg_from_yaml_file(str(S2 / ('centerpoint-factorswap-s2-%s.yaml' % name)), EasyDict())
        assert 'EVAL_FACTOR_SWAP' in twin.DATA_CONFIG_TAR and 'EVAL_FACTOR_SWAP' not in base.DATA_CONFIG_TAR
        twin.DATA_CONFIG_TAR.pop('EVAL_FACTOR_SWAP')
        assert without_base_keys(twin) == without_base_keys(base)
