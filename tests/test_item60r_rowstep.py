"""Item 60r's rows and readings (tools/analysis/item60r_rowstep.py; experiments_md 20261003_05 §5 item 60r, design fixed
2026-11-28 15:32 JST). Run from the repo root."""
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item60r_rowstep as m  # noqa: E402


def row(name, seed, rings, pooled=70.0, logged=None):
    return dict(row=name, seed=seed, reproduced=True, logged_bev=pooled if logged is None else logged,
                pooled=dict(bev=pooled, **{'3d': 50.0}), rings=[dict(bev=v, **{'3d': v / 2}) for v in rings])


def rows_with(none_far, ada_s_far, ada_far=4.0, ident=5.12):
    out = [row('P_RS_none_s1', 666, [1, 1, 1, 1, 1], pooled=ident), row('P_RS_none', 666, [1, 1, 1, 1, 1], 5.12),
           row('P_RS_none', 667, [1, 1, 1, 1, 1], 1.49)]
    for s in m.SEEDS:
        out += [row('L_ada', s, [95, 95, 84, 66, 35]), row('P_RS1', s, [92, 92, 74, 56, 30]),        # s far 10 / 10
                row('P_RS_ada', s, [92, 92, 74 + ada_far, 56 + ada_far, 30]),
                row('P_RS_none_s', s, [92, 92, 74 + none_far, 56 + none_far, 30]),
                row('P_RS_ada_s', s, [92, 92, 74 + ada_s_far, 56 + ada_s_far, 30])]
    return out


def test_scales_dirs_and_spec():
    assert m.S == pytest.approx(0.4 / 1.33, abs=5e-4) and m.S_UPPER == pytest.approx((1 / 3) / 1.33, abs=5e-4)
    sp = list(m.spec())
    assert [(a, s) for a, s, *_ in sp] == [('none_s', 666), ('none_s', 667), ('ada_s', 666), ('ada_s', 667),
                                           ('none_s025', 666), ('none_s025', 667), ('ident', 666)]
    assert str(m.eval_dir('ada_s', 667)).endswith('item60/eval/ada_x0301_s667')
    assert str(m.eval_dir('ident', 666)).endswith('item60/eval/none_x1000_s666')
    assert len({str(d) for *_, d, _ in sp}) == len(sp)                 # no two rows share an output dir
    assert all(not str(d).endswith(('none_s666', 'ada_s666', 'glob_s666')) for *_, d, _ in sp)   # item 60 / 60g's kept


def test_h_r0_words():
    r = m.readings(rows_with(6.0, 0.0))
    assert r['H_r0'] == m.WORDS_R0['a normalization-level correction can close most of the far residual']
    assert m.readings(rows_with(3.0, 0.0))['H_r0'] == 'partly'
    assert m.readings(rows_with(0.5, 0.0))['H_r0'] == 'they do not'
    assert m.readings(rows_with(-2.0, 0.0))['H_r0'] == 'they widen the far residual'
    assert r['H_r0_seeds'][0]['delta'] == pytest.approx(6.0) and r['H_r0_seeds'][0]['R'] == pytest.approx(0.6)


def test_d_r_words():
    assert m.readings(rows_with(0.0, 6.5))['D_r'] == m.WORDS_D[0]            # D 2.5
    assert m.readings(rows_with(0.0, 4.5))['D_r'] == m.WORDS_D[1]            # D 0.5
    assert m.readings(rows_with(0.0, 1.5))['D_r'] == m.WORDS_D[2]            # D -2.5
    assert m.readings(rows_with(0.0, 5.5))['D_r'] == m.WORDS_D[3]            # D 1.5
    assert m.readings(rows_with(0.0, 6.5))['D_r_seeds'][666] == pytest.approx(2.5)


def test_nothing_is_read_without_the_identity_or_a_reproduction():
    r = m.readings(rows_with(6.0, 6.0, ident=5.20))
    assert not r['identity']['ok'] and r['H_r0'] == r['H_ra'] == r['D_r'] == 'not read'
    assert m.readings(rows_with(6.0, 6.0, ident=5.16))['identity']['ok']          # within 0.05
    rows = rows_with(6.0, 6.0)
    rows[-1]['reproduced'] = False
    r = m.readings(rows)
    assert r['not_reproduced'] == [('P_RS_ada_s', 667)] and r['H_r0'] == 'not read'


@pytest.mark.skipif(not m.ITEM60G_MANIFEST.exists(), reason='no item 60g manifest')
def test_manifest_carries_item60g_rows():
    rows = m.manifest({666: 37227, 667: 37228})
    names = {(r['row'], r['seed']) for r in rows}
    assert {('P_RS_none_s', 666), ('P_RS_ada_s', 667), ('P_RS_none_s1', 666), ('P_RS_none', 666), ('P_RS_ada', 667),
            ('P_RS_glob', 666), ('P_RS1', 667), ('L_ada', 666)} <= names
    assert all(r['group'] == 'item60r' for r in rows)
