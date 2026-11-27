"""Item 62's manifest and readings (tools/analysis/item62_pop.py; experiments_md 20261003_05 §5 item 62). Run from the repo root."""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item62_pop as m  # noqa: E402


def row(name, seed, rings, pooled=70.0):
    return dict(row=name, seed=seed, reproduced=True, pooled={'bev': pooled, '3d': 50.0}, rings=[{'bev': v} for v in rings])


def test_arms_dirs_and_checkpoints():
    assert m.ARMS == ('all', 'far20', 'tbox', 'gtbox') and m.row_name('tbox') == 'P_POP_TBOX'
    assert str(m.arm_dir('gtbox', 667)).endswith('item62/gtbox_s667')
    assert '0i5ai4hn' in m.PK_CKPT[666] and '1ya9lozy' in m.PK_CKPT[667]


def test_all_check_and_readings():
    rows = []
    for s in m.SEEDS:
        rows += [row('L_ada', s, [95, 95, 84, 66, 35]), row('P_RS1', s, [92, 92, 74, 56, 30], 69.36),
                 row('P_POP_ALL', s, [92, 92, 74, 56, 30], 69.40), row('P_POP_FAR20', s, [92, 92, 80, 62, 30]),
                 row('P_POP_TBOX', s, [92, 92, 72, 54, 30])]
    assert m.all_check(rows, 666) and m.all_check(rows, 667)
    assert m.reading(rows, 'far20')[1] == 'a normalization-level correction can close most of the far residual'
    assert m.reading(rows, 'tbox')[1] == 'it widens the far residual'
    assert m.reading(rows, 'gtbox')[1] == 'not read'
    bad = [dict(r, pooled={'bev': 70.0, '3d': 50.0}) if r['row'] == 'P_POP_ALL' else r for r in rows]
    assert not m.all_check(bad, 666)
