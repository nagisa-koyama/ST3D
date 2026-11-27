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
