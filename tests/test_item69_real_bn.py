"""Item 69's readings and gates (tools/analysis/item69_real_bn.py; experiments_md 20261003_05 §5 item 69, design fixed
2026-11-28 17:51 JST). Run from the repo root."""
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item69_real_bn as m  # noqa: E402

O_RINGS = [95.0, 95.0, 84.0, 68.0, 38.0]                       # the oracle (both seeds equal here), far mean 76


def row(name, seed, rings, pooled):
    return dict(row=name, seed=seed, reproduced=True, logged_bev=pooled, pooled={'bev': pooled, '3d': pooled / 2},
                rings=[{'bev': v} for v in rings])


def rows_with(g_far_gain, aff3d_far_gain=0.0, rs1=None):
    rs1 = rs1 or m.B1
    out = [row('O_ada', 666, O_RINGS, 80.0), row('O_ada', 667, O_RINGS, 80.0)]
    for s in m.SEEDS:
        base = [93.0, 93.0, 74.0, 58.0, 30.0]                   # far gap 10 against the oracle
        out.append(row('rs1', s, base, rs1[s]))
        for arm, gain in (('G_r', g_far_gain), ('aff_all', g_far_gain), ('aff_3d', aff3d_far_gain), ('aff_2d', 0.0)):
            out.append(row(arm, s, [93.0, 93.0, 74.0 + gain, 58.0 + gain, 30.0], rs1[s] + gain / 2))
    return out


OK = {s: {'ok': True} for s in m.SEEDS}


def test_far_words():
    r = m.readings(rows_with(6.0, 0.5), OK)['readings']
    assert r['G_r_far']['word'] == "a normalization-level correction closes most of the real pair's far residual"
    assert r['aff_all_far']['word'] == 'the correction lies in gamma / beta'
    assert r['aff_3d_far']['word'] == 'cannot'
    s0 = r['G_r_far']['seeds'][0]
    assert s0['gap_rs1'] == pytest.approx(10.0) and s0['delta'] == pytest.approx(6.0) and s0['R'] == pytest.approx(0.6)
    assert m.readings(rows_with(3.0, 3.0), OK)['readings']['aff_3d_far']['word'] == 'partly'


def test_pooled_reading_uses_the_pooled_gap():
    r = m.readings(rows_with(6.0), OK)['readings']
    x = r['G_r_pooled']['seeds'][0]
    assert x['gap_rs1'] == pytest.approx(80.0 - m.B1[666]) and x['delta'] == pytest.approx(3.0)


def test_gates():
    off = dict(m.B1)
    off[668] = m.B1[668] + 0.2
    assert m.readings(rows_with(6.0, rs1=off), OK)['readings'] == 'not read'
    bad = dict(OK)
    bad[667] = {'ok': False}
    assert m.readings(rows_with(6.0), bad)['readings'] == 'not read'
    rows = rows_with(6.0)
    rows[3]['reproduced'] = False
    out = m.readings(rows, OK)
    assert out['readings'] == 'not read' and out['not_reproduced']


def test_names_and_paths():
    assert m.tag('G_r', 667) == 'i69_G_r_s667' and m.tag('aff_3d', 668) == 'i69_aff_3d_s668'
    assert str(m.hybrid_path('aff_2d', 666)).endswith('item69/ckpt/aff_2d_s666/checkpoint_epoch_20.pth')
    assert m.BN41A == {666: 32373, 667: 32411, 668: 32434} and m.EPOCH == 20


@pytest.mark.skipif(not (TOOLS / 'cfgs/da-ieee-access-analysis/item69/centerpoint-item69-n2k-bnonly-g.yaml').exists(),
                    reason='no config')
def test_g_r_config_is_item58_g_and_shares_the_controls_model():
    import os
    import json
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    os.chdir(TOOLS)
    g = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/item69/centerpoint-item69-n2k-bnonly-g.yaml', EasyDict())
    g58 = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/item58/centerpoint-item58-p-bnonly-g.yaml', EasyDict())
    ctrl = cfg_from_yaml_file('cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti.yaml', EasyDict())
    J = lambda x: json.dumps(x, sort_keys=True, default=str)                                  # noqa: E731
    assert J(g.MODEL) == J(g58.MODEL) == J(ctrl.MODEL)
    assert g.OPTIMIZATION.TRAIN_ONLY_BN and g.OPTIMIZATION.NUM_EPOCHS == 20
    for k in ('POINT_CLOUD_RANGE', 'POINT_FEATURE_ENCODING', 'DATA_PROCESSOR', 'SHIFT_COOR', 'FOV_POINTS_ONLY', 'DATA_SPLIT',
              'INFO_PATH'):
        assert J(g.DATA_CONFIG.get(k)) == J(ctrl.DATA_CONFIG_TAR.get(k)), k
