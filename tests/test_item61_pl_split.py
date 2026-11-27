"""Item 61's configs, rescoring arguments and readings (tools/analysis/item61_pl_split.py; experiments_md 20261003_05 §5 item
61, design fixed 2026-11-27 12:53 JST). Run from the repo root."""
import os
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item61_pl_split as m  # noqa: E402


def load_cfg(path):
    from pcdet.config import cfg_from_yaml_file
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        return cfg_from_yaml_file(path, EasyDict())
    finally:
        os.chdir(cwd)


def test_gsrc_is_g_on_the_thinned_clouds_only():
    g = load_cfg('cfgs/da-ieee-access-analysis/item58/centerpoint-item58-p-bnonly-g.yaml')
    gs = load_cfg(m.GSRC_CFG)
    pk = load_cfg(m.EVAL_CFG)
    assert gs.OPTIMIZATION.NUM_EPOCHS == 20 and gs.OPTIMIZATION.TRAIN_ONLY_BN is True
    assert gs.DATA_CONFIG.TRAIN_FACTOR_SWAP == pk.DATA_CONFIG.TRAIN_FACTOR_SWAP and 'TRAIN_FACTOR_SWAP' not in g.DATA_CONFIG
    dgs = dict(gs.DATA_CONFIG); dgs.pop('TRAIN_FACTOR_SWAP')
    assert dgs == dict(g.DATA_CONFIG)
    for k in set(g) | set(gs):
        if k not in ('DATA_CONFIG', '_BASE_CONFIG_'):
            assert gs.get(k) == g.get(k), k


def test_rescore_args_and_dirs():
    assert m.rescore_args() == [('i61_plbn_s666', 37209, 10), ('i61_plbn_s667', 37210, 10),
                                ('i61_plall_s666', 37211, 10), ('i61_plall_s667', 37212, 10)]
    assert m.rescore_args({666: 5, 667: 6}) == [('i61_gsrc_s666', 5, 20), ('i61_gsrc_s667', 6, 20)]
    d = m.result_dir('gsrc', 667)
    assert d.parts[-4:] == ('i61_gsrc_s667', 'eval', 'epoch_20', 'i61_gsrc_s667')
    assert m.result_dir('plbn', 666).parts[-2] == 'epoch_10'
    assert m.parse_jobs(['gsrc666=1', 'gsrc667=2']) == {666: 1, 667: 2}


def row(name, seed, rings):
    return dict(row=name, seed=seed, reproduced=True, pooled={'bev': 70.0, '3d': 50.0}, rings=[{'bev': v} for v in rings])


def test_readings_use_item58_thresholds_and_the_declared_words():
    rows = []
    for s in m.SEEDS:
        rows += [row('L_ada', s, [95, 95, 84, 66, 35]), row('P_RS1', s, [92, 92, 74, 56, 30]),
                 row('P_PL_BN_ADA', s, [92, 92, 81, 63, 30]), row('P_GSRC_ADA', s, [92, 92, 74.3, 56.2, 30])]
    out = m.readings(rows)
    assert out['plbn'][1] == 'a normalization-level correction can close most of the far residual'
    assert m.WORDS['plbn'][out['plbn'][1]] == "59b's correction lies in its gamma / beta, as G's does"
    assert out['gsrc'][1] == 'it cannot'
    assert m.WORDS['gsrc']['it cannot'].startswith('continued training alone does not')
    assert out['plall'][1] == 'not read'


def test_check_flags_a_moved_weight():
    pk = {'backbone_3d.conv_input.0.weight': torch.ones(2), 'backbone_3d.conv_input.1.weight': torch.ones(2),
          'backbone_3d.conv_input.1.bias': torch.zeros(2), 'backbone_3d.conv_input.1.running_mean': torch.zeros(2),
          'backbone_3d.conv_input.1.running_var': torch.ones(2),
          'backbone_3d.conv_input.1.num_batches_tracked': torch.tensor(1), 'global_step': torch.tensor([1.])}
    ok = dict(pk, **{'backbone_3d.conv_input.1.weight': torch.full((2,), 2.0), 'global_step': torch.tensor([9.])})
    bad = dict(ok, **{'backbone_3d.conv_input.0.weight': torch.full((2,), 3.0)})
    st = {(m.PK[666], m.PK_EPOCH): pk, (1, m.GSRC_EPOCH): ok, (2, m.GSRC_EPOCH): bad}
    assert m.check({666: 1}, lambda j, e: st[(j, e)])['P_GSRC s666']['ok'] is True
    assert m.check({666: 2}, lambda j, e: st[(j, e)])['P_GSRC s666']['ok'] is False
