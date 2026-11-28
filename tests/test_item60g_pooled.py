"""Item 60g's manifest and readings (tools/analysis/item60g_pooled.py; experiments_md 20261003_05 §5 item 60g, design fixed
2026-11-28 10:38 JST). Run from the repo root."""
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item60g_pooled as m  # noqa: E402


def row(name, seed, rings, pooled=70.0):
    return dict(row=name, seed=seed, reproduced=True, pooled=dict(bev=pooled, **{'3d': 50.0}),
                rings=[dict(bev=v) for v in rings])


def rows_with(glob_far, ada_far=4.0):
    out = []
    for s in m.SEEDS:
        out += [row('L_ada', s, [95, 95, 84, 66, 35]), row('P_RS1', s, [92, 92, 74, 56, 30]),        # s far 10 / 10
                row('P_RS_ada', s, [92, 92, 74 + ada_far, 56 + ada_far, 30]),
                row('P_RS_glob', s, [92, 92, 74 + glob_far, 56 + glob_far, 30])]
    return out


def test_h_g_words():
    assert m.h_g(rows_with(6.0))[1] == 'a normalization-level correction can close most of the far residual'
    assert m.h_g(rows_with(3.0))[1] == 'partly'
    assert m.h_g(rows_with(0.5))[1] == 'it cannot'
    assert m.h_g(rows_with(-2.0))[1] == 'it widens the far residual'
    seeds, _ = m.h_g(rows_with(3.0))
    assert seeds[0]['delta'] == pytest.approx(3.0) and seeds[0]['R'] == pytest.approx(0.3)


def test_split_words():
    d, w = m.split(rows_with(0.5, ada_far=4.0))                    # glob "cannot", D = 3.5
    assert w == m.WORDS_SPLIT[0] and d[666] == pytest.approx(3.5)
    assert m.split(rows_with(-2.0, ada_far=4.0))[1] == m.WORDS_SPLIT[0]          # "widens" counts as "cannot"
    assert m.split(rows_with(3.5, ada_far=4.0))[1] == m.WORDS_SPLIT[1]           # |D| 0.5
    assert m.split(rows_with(7.0, ada_far=4.0))[1] == m.WORDS_SPLIT[2]           # D -3
    assert m.split(rows_with(2.5, ada_far=4.0))[1] == m.WORDS_SPLIT[3]           # glob partly, D 1.5
    assert m.split([])[1] == 'not read'


def test_dirs():
    assert str(m.eval_dir(667)).endswith('item60/eval/glob_s667') and m.ROW == 'P_RS_glob'


@pytest.mark.skipif(not m.ITEM60_MANIFEST.exists(), reason='no item 60 manifest')
def test_manifest_carries_item60_rows():
    rows = m.manifest({666: 37227, 667: 37228})
    names = {(r['row'], r['seed']) for r in rows}
    assert {('P_RS_glob', 666), ('P_RS_glob', 667), ('P_RS_ada', 666), ('P_RS1', 667), ('L_ada', 666)} <= names
    assert all(r['group'] == 'item60g' for r in rows)
