"""Item 52's readings (experiments_md 20261003_05 §5 item 52, fixed before any per-ring number). CPU only."""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS / 'analysis'))
import item52_uada_rings as m  # noqa: E402


def ap(u, mm, reproduced=True):
    ring = lambda xs: [{'bev': x, 'n_gt': 1} for x in xs]  # noqa: E731
    return {'rows': [dict(row='U', rings=ring(u), reproduced=reproduced), dict(row='M', rings=ring(mm), reproduced=True)]}


def test_primary_reading_uses_20_to_40_m():
    rd = m.readings(ap([80, 70, 60, 50, 30], [80, 70, 56, 47, 30]))
    assert rd['d'] == [0, 0, 4, 3, 0] and rd['res'] == (3.5, 'UADA3D ahead where the residual sits')
    assert rd['near'][1] == 'level near' and rd['far'] == (1.5, 'level far')


def test_bar_is_inclusive_both_ways_and_level_between():
    assert m.classify(3.0, 'x') == 'UADA3D ahead x' and m.classify(-3.0, 'x') == 'UADA3D behind x'
    assert m.classify(2.99, 'x') == 'level x' and m.classify(-2.99, 'x') == 'level x'


def test_not_read_when_a_compared_row_is_not_reproduced():
    assert m.readings(ap([1] * 5, [1] * 5, reproduced=False)) is None


def test_ckpt_line_of_an_adabn_log(tmp_path):
    log = tmp_path / 'log.txt'
    log.write_text('[2026-10-04 14:40:00,000  adabn_eval.py 90  INFO]  ckpt             '
                   '/storage/wandb/run-20260922_155434-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth\n')
    assert m.ckpt_of_log(log).endswith('-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth')


def test_rows_cover_the_six_arms_once():
    assert [r[0] for r in m.ROWS] == ['U', 'M', 'S0', 'S1', 'Usrc', 'Utgt'] and set(m.LOGGED) == {r[0] for r in m.ROWS}
