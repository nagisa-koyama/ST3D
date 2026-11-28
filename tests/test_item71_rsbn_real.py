"""Item 71's spec, readings and checks (tools/analysis/item71_rsbn_real.py; experiments_md 20261003_05 §5 item 71, design fixed
2026-11-28 21:31 JST). Run from the repo root."""
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item71_rsbn_real as m  # noqa: E402

O_RINGS = [95.0, 95.0, 84.0, 70.0, 40.0]


def row(name, seed, rings, pooled):
    return dict(row=name, seed=seed, reproduced=True, logged_bev=pooled, pooled={'bev': pooled, '3d': pooled / 2},
                rings=[{'bev': v} for v in rings])


BASE_ADA = [75.0, 85.0, 76.0, 52.0, 20.0]                     # near gap 15, far gap 13
BASE_NONE = [86.0, 79.0, 30.0, 2.0, 0.2]


def rows_with(gain=None):
    """gain: {(name, seed): (near, far, pooled)} added to the BN_ada base for that row."""
    gain = gain or {}
    out = [row('O_ada', 666, O_RINGS, 80.0), row('O_ada', 667, O_RINGS, 80.0)]
    for s in m.SEEDS:
        out.append(row('BN_ada', s, BASE_ADA, m.B1[s]))
        out.append(row('BN_none', s, BASE_NONE, m.B0[s]))
        for v in m.VARIANTS:
            for arm in m.ARMS:
                n, f, p = gain.get((m.row_name(v, arm), s), (0.0, 0.0, 0.0))
                base = BASE_NONE if arm.startswith('none') else BASE_ADA
                ref = m.B0[s] if arm.startswith('none') else m.B1[s]
                out.append(row(m.row_name(v, arm), s, [base[0] + n, base[1] + n, base[2] + f, base[3] + f, base[4]], ref + p))
    return out


SRC = [dict(row='%s_src' % v, seed=s, logged_bev=m.BN_SRC[s] + 0.3) for v in m.VARIANTS for s in m.SEEDS]


def test_words_including_split():
    g = {('st_ada', s): (0.0, 7.0, 4.0) for s in m.SEEDS}
    r = m.readings(rows_with(g), SRC)['readings']
    assert r['H_A_st']['far']['word'] == 'most'                # 7 of 13, R 0.54
    assert r['H_A_st']['near']['word'] == 'cannot' and r['H_A_stg']['far']['word'] == 'cannot'
    g = {('st_ada', 666): (0.0, 4.0, 0.0), ('st_ada', 667): (0.0, 3.0, 0.0), ('st_ada', 668): (0.0, 1.5, 0.0)}
    assert m.readings(rows_with(g), SRC)['readings']['H_A_st']['far']['word'] == 'partly'
    g = {('st_ada', 666): (0.0, 8.0, 0.0), ('st_ada', 667): (0.0, -40.0, 0.0), ('st_ada', 668): (0.0, -8.0, 0.0)}
    assert m.readings(rows_with(g), SRC)['readings']['H_A_st']['far']['word'] == 'split'   # item 70's case
    g = {('stg_ada_s', s): (-3.0, -2.0, -1.0) for s in m.SEEDS}
    r = m.readings(rows_with(g), SRC)['readings']
    assert r['H_S_stg']['near']['word'] == 'widens' and r['H_S_stg']['far']['word'] == 'widens'


def test_row_step_and_gamma_beta_differences():
    g = {('stg_ada', s): (0.0, 3.0, 2.5) for s in m.SEEDS}
    g.update({('stg_ada_s', s): (0.0, 0.0, 0.0) for s in m.SEEDS})
    r = m.readings(rows_with(g), SRC)['readings']
    assert r['H_gb_ada_pooled']['word'] == 'per-range gamma / beta add' and r['H_gb_ada_far']['d'] == pytest.approx([3.0] * 3)
    assert r['H_gb_ada_s_pooled']['word'] == 'no difference'
    assert r['D_r_stg_far']['word'] == 'it costs' and r['D_r_st_far']['word'] == 'adds nothing'


def test_no_target_data_reads_against_bn_none():
    g = {('st_none_s', s): (0.0, 35.0, 10.0) for s in m.SEEDS}                # far gap 61 -> 26, R 0.57
    r = m.readings(rows_with(g), SRC)['readings']
    x = r['H_0_st_none_s']['far']
    assert x['word'] == 'most' and x['seeds'][0]['gap_ref'] == pytest.approx(77.0 - 16.0)


def test_gates_and_source():
    rows = rows_with()
    [r for r in rows if r['row'] == 'BN_ada' and r['seed'] == 667][0]['pooled']['bev'] += 0.06
    out = m.readings(rows, SRC)
    assert out['readings'] == 'not read' and out['references_off'] == [('BN_ada', 667, pytest.approx(66.79))]
    rows = rows_with()
    rows[5]['reproduced'] = False
    assert m.readings(rows, SRC)['readings'] == 'not read'
    assert m.readings(rows_with(), SRC)['H_src']['st']['word'] == 'no effect on the source'
    low = [dict(r, logged_bev=r['logged_bev'] - 3.0) for r in SRC]
    assert m.readings(rows_with(), low)['H_src']['stg']['word'] == 'costs the source'
    assert m.readings(rows_with(), SRC[:2])['H_src']['st']['word'] == 'not read'


def test_spec_jobs_and_paths():
    sp = list(m.spec())
    assert len(sp) == 30 and len({x[5] for x in sp}) == 30 and len({x[6] for x in sp}) == 30
    assert ('stg', 'ada_s', 667, 'ada', 0.301) == sp[[x[6] for x in sp].index('i71_stg_ada_s_s667')][:5]
    assert sp[0][7].endswith('item71/centerpoint-item71-nuscenes2kitti-st.yaml')
    j = m.parse_jobs(['st666=1', 'st667=2', 'st668=3', 'stg666=4', 'stg667=5', 'stg668=6'])
    assert j[('stg', 667)] == 5
    with pytest.raises(AssertionError):
        m.parse_jobs(['st666=1'])
    assert m.S == pytest.approx(0.4 / 1.33, abs=0.001)


def test_ckptcheck(tmp_path):
    import torch
    sd = {}
    for i in range(34):
        n = 'backbone_3d.l%d' % i
        sd[n + '.running_mean'] = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
        sd[n + '.affine.weight'] = torch.tensor([[1.0, 1.0], [1.0, 1.2]])
    sd['other.running_mean'] = torch.zeros(2)                  # a plain BN buffer (1-D) is not a RangeBN layer
    torch.save({'model_state': sd}, tmp_path / 'c.pth')
    out = m.ckptcheck(tmp_path / 'c.pth')
    assert len(out) == 34 and out[0]['gamma_spread'] == pytest.approx(0.2) and not out[0]['has_bn_weight']
    sd['backbone_3d.l0.affine.weight'] = torch.ones(2, 2)      # gamma never moved
    torch.save({'model_state': sd}, tmp_path / 'c.pth')
    assert m.ckptcheck(tmp_path / 'c.pth')[0]['gamma_spread'] == 0.0
