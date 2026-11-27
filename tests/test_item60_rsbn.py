"""Item 60's manifest and readings (tools/analysis/item60_rsbn.py; experiments_md 20261003_05 §5 item 60, design fixed
2026-11-27 12:09 JST). Run from the repo root."""
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item60_rsbn as m  # noqa: E402


def row(name, seed, rings, pooled=70.0):
    return dict(row=name, seed=seed, reproduced=True, pooled=dict(bev=pooled, **{'3d': 50.0}),
                rings=[dict(bev=v) for v in rings])


def base(seed):
    return [row('L_ada', seed, [95, 95, 84, 66, 35]), row('P_RS1', seed, [92, 92, 74, 56, 30])]   # s far 10 / 10


def test_h1_words():
    for diffs, word in (((2.5, 3.0), 'range-stratified BN helps on the source'), ((-2.0, -4.0), 'it costs on the source'),
                        ((0.5, -0.9), 'no effect on the source'), ((2.5, 0.5), 'unclear')):
        rows = []
        for s, dlt in zip(m.SEEDS, diffs):
            rows += [row('P_src', s, [0] * 5, 68.0), row('P_RS_src', s, [0] * 5, 68.0 + dlt)]
        d, w = m.h1(rows)
        assert w == word and d[666] == pytest.approx(diffs[0])
    assert m.h1([])[1] == 'not read'


def test_h2_most_partly_widens():
    def rows_with(far):
        out = []
        for s in m.SEEDS:
            out += base(s) + [row('P_RS_ada', s, [92, 92, 74 + far, 56 + far, 30])]
        return out
    assert m.h2(rows_with(6.0))[1] == 'a normalization-level correction can close most of the far residual'
    assert m.h2(rows_with(3.0))[1] == 'partly'
    assert m.h2(rows_with(0.5))[1] == 'it cannot'
    assert m.h2(rows_with(-2.0))[1] == 'it widens the far residual'
    seeds, _ = m.h2(rows_with(6.0))
    assert seeds[0]['delta'] == pytest.approx(6.0) and seeds[0]['R'] == pytest.approx(0.6)


def test_modes_dirs_and_jobs():
    assert set(m.MODES) == {'src', 'none', 'ada'} and m.MODES['src'][1] == m.MODES['none'][1] == 'none'
    assert m.MODES['ada'] == (m.CFG, 'ada') and m.MODES['src'][0] == m.CFG_SRC
    assert str(m.eval_dir('ada', 667)).endswith('item60/eval/ada_s667')
    assert m.parse_jobs(['s666=1', 's667=2']) == {666: 1, 667: 2}


@pytest.mark.skipif(not Path(m.REF_57).exists(), reason='no item 57 manifest')
def test_references_are_the_declared_rows():
    refs = m.references()
    names = sorted((r['row'], r['seed']) for r in refs)
    want = sorted((n, s) for n in ('P_RS1', 'P_RS5', 'L_ada', 'O_ada', 'P_none', 'P_src') for s in m.SEEDS)
    assert names == want
    by = {(r['row'], r['seed']): r for r in refs}
    assert by[('P_none', 666)]['job'] == 32283 and abs(by[('P_none', 666)]['logged_bev'] - 19.7653) < 1e-3
    assert by[('P_src', 667)]['job'] == 32284 and 'lattice' in by[('P_src', 667)]['result']
