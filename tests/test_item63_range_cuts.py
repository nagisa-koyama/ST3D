"""Item 63's per-range rule A and leaf configs (tools/analysis/item63_range_cuts.py; experiments_md 20261003_05 §5 item 63,
design fixed 2026-11-27 13:02 JST). Run from the repo root."""
import os
import sys
from pathlib import Path

import numpy as np
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item63_range_cuts as m  # noqa: E402


def test_bins_follow_item48_rings():
    assert m.bins_of(np.array([0.0, 9.99, 10.0, 25.0, 39.9, 40.0, 70.0])).tolist() == [0, 0, 1, 2, 3, 4, 4]


def test_per_bin_rule_a_is_the_plateau_rule_on_that_bin_and_falls_back_when_thin():
    sys.path.insert(0, str(TOOLS / 'analysis' / 'pseudo_label_threshold'))
    from plateau_neg_thresh import plateau_neg
    rng = np.random.default_rng(0)
    rows = []
    for k, (lo, hi) in enumerate([(1, 9), (11, 19), (21, 29), (31, 39), (45, 70)]):
        n = 5000 if k < 4 else 300                                   # the last bin is thin for every class
        s = 1 / (1 + np.exp(-(rng.normal(-3 + 0.3 * k, 1.0, n))))
        r = rng.uniform(lo, hi, n)
        for ci in (1, 2, 3):
            b = np.zeros((n, 9)); b[:, 0] = r; b[:, 7] = ci; b[:, 8] = s
            rows.append(b)
    boxes = np.concatenate(rows)
    out = m.range_rule_a(boxes, {'Car': 0.5, 'Pedestrian': 0.6, 'Cyclist': 0.7})
    for k in range(4):
        want = round(float(plateau_neg(boxes[(boxes[:, 7] == 1) & (m.bins_of(boxes[:, 0]) == k), 8])[0]), 3)
        assert out['Car']['cuts'][k] == want and not out['Car']['fallback'][k]
    assert out['Car']['cuts'][4] == 0.5 and out['Cyclist']['cuts'][4] == 0.7 and out['Car']['fallback'][4]
    assert out['Car']['boxes'] == [5000, 5000, 5000, 5000, 300]


def test_leaf_resolves_to_59b_plus_thresh_by_range(tmp_path, monkeypatch):
    from pcdet.config import cfg_from_yaml_file
    res = {'666': {'rule_a': {c: dict(cuts=[0.1, 0.12, 0.15, 0.2, 0.3], fallback=[False] * 4 + [True]) for c in m.CLASSES}}}
    txt = m.leaf_text(666, res)
    p = TOOLS / m.CFG_DIR / '_test_leaf.yaml'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt)
    cwd = os.getcwd(); os.chdir(TOOLS)
    try:
        leaf = cfg_from_yaml_file(str(p.relative_to(TOOLS)), EasyDict())
        base = cfg_from_yaml_file(m.LEAF_59B % 666, EasyDict())
    finally:
        os.chdir(cwd); p.unlink()
    assert leaf.SELF_TRAIN.THRESH_BY_RANGE.EDGES == list(m.EDGES)
    assert leaf.SELF_TRAIN.THRESH_BY_RANGE.CUTS == [[0.1, 0.12, 0.15, 0.2, 0.3]] * 3
    st = dict(leaf.SELF_TRAIN); st.pop('THRESH_BY_RANGE')
    assert st == dict(base.SELF_TRAIN)
    for k in set(leaf) | set(base):
        if k not in ('SELF_TRAIN', '_BASE_CONFIG_'):
            assert leaf.get(k) == base.get(k), k
    assert leaf.OPTIMIZATION.TRAIN_ONLY_BN is True


# ---------------------------------------------------------------- item 63′, count balance by range
def test_count_cut_keeps_the_k_highest():
    s = np.array([0.9, 0.1, 0.5, 0.3, 0.7])
    assert m.count_cut(s, 2) == 0.7 and (s >= m.count_cut(s, 2)).sum() == 2
    assert m.count_cut(s, 9) == m.FLOOR and m.count_cut(s, 0) > 1.0


def test_counts_and_balanced_cuts_on_synthetic_frames():
    # two frames; class 1 at 5 m (scores .9, .2) and 35 m (.8, .6, .4); class 2 at 15 m (.5)
    f = {'a': np.array([[5, 0, 0, 0, 0, 0, 0, 1, .9], [35, 0, 0, 0, 0, 0, 0, 1, .8], [15, 0, 0, 0, 0, 0, 0, 2, .5]]),
         'b': np.array([[5, 0, 0, 0, 0, 0, 0, 1, .2], [0, 35, 0, 0, 0, 0, 0, 1, .6], [35, 1, 0, 0, 0, 0, 0, 1, .4]])}
    assert m.per_frame_counts(f, 2)['Car'] == [1.0, 0.0, 0.0, 1.5, 0.0]
    dens = {'Car': [0.5, 0.0, 0.0, 1.0, 0.0], 'Pedestrian': [0.0, 1.0, 0.0, 0.0, 0.0], 'Cyclist': [0.0] * 5}
    cb = m.count_balance_cuts(f, 2, dens)
    assert cb['Car']['cuts'][0] == 0.9 and cb['Car']['cuts'][3] == 0.6            # keep 1 of 2 near, 2 of 3 at 30-40 m
    assert cb['Car']['target_boxes'] == [1, 0, 0, 2, 0] and cb['Car']['fov_boxes'] == [2, 0, 0, 3, 0]
    assert cb['Pedestrian']['cuts'][1] == m.FLOOR                                 # wants 2, has 1: keep all
    assert m.kept_per_frame(f, 2, cb)['Car'] == [0.5, 0.0, 0.0, 1.0, 0.0]


def test_fov_mask_is_the_evaluations_camera_test():
    cal = m.calib_of('training', '000000')
    c = np.array([[10.0, 0.0, -1.0], [-10.0, 0.0, -1.0], [10.0, 30.0, -1.0]])
    assert m.fov_mask(c, cal, np.array([375, 1242])).tolist() == [True, False, False]


def test_cb_leaf_resolves_to_59b_plus_thresh_by_range():
    from pcdet.config import cfg_from_yaml_file
    res = {'667': {'cuts': {c: dict(cuts=[0.3, 0.2, 0.15, 0.1, 0.5]) for c in m.CLASSES}}}
    p = TOOLS / m.CFG_DIR / '_test_cb.yaml'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(m.leaf_text_cb(667, res))
    cwd = os.getcwd(); os.chdir(TOOLS)
    try:
        leaf = cfg_from_yaml_file(str(p.relative_to(TOOLS)), EasyDict())
        base = cfg_from_yaml_file(m.LEAF_59B % 667, EasyDict())
    finally:
        os.chdir(cwd); p.unlink()
    assert leaf.SELF_TRAIN.THRESH_BY_RANGE.CUTS == [[0.3, 0.2, 0.15, 0.1, 0.5]] * 3
    st = dict(leaf.SELF_TRAIN); st.pop('THRESH_BY_RANGE')
    assert st == dict(base.SELF_TRAIN) and leaf.OPTIMIZATION == base.OPTIMIZATION


def test_cb_readings():
    def row(name, seed, rings):
        return dict(row=name, seed=seed, reproduced=True, pooled={'bev': 70.0, '3d': 50.0}, rings=[{'bev': v} for v in rings])
    rows = []
    for s in m.SEEDS:
        rows += [row('L_ada', s, [95, 95, 84, 66, 35]), row('P_RS1', s, [92, 92, 74, 56, 30]),
                 row('P_PL_BN', s, [95, 95, 83, 62, 35]), row('P_PL_BN_CB', s, [95, 95, 83.5, 64.5, 35])]
    seeds, word, d, w2 = m.readings_cb(rows)
    assert word == 'a normalization-level correction can close most of the far residual'
    assert d == {666: 2.5, 667: 2.5} and w2 == 'count balance by range helps where 59b fell short'
    assert m.parse_jobs(['s666=1', 's667=2']) == {666: 1, 667: 2}
