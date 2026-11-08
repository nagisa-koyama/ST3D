"""tools/analysis/item42_rule_a.py: IoU 0.5 BEV read from the evaluator's own result string, beside the unchanged IoU 0.7 values
of pandaset_eval_protocols.score (experiments_md 20261108_05). CPU only, no dataset files."""
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item42_rule_a as r42  # noqa: E402
import make_item42_configs as m42  # noqa: E402

RESULT = """Car AP@0.70, 0.70, 0.70:
bbox AP:1.0000, 2.0000, 3.0000
bev  AP:11.0000, 12.0000, 13.0000
3d   AP:21.0000, 22.0000, 23.0000
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:1.0000, 2.0000, 3.0000
bev  AP:31.0000, 32.5000, 33.0000
3d   AP:41.0000, 42.0000, 43.0000
aos  AP:0.0, 0.0, 0.0
Car AP@0.70, 0.50, 0.50:
bbox AP:1.0000, 2.0000, 3.0000
bev  AP:51.0000, 52.0000, 53.0000
3d   AP:61.0000, 62.0000, 63.0000
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:1.0000, 2.0000, 3.0000
bev  AP:71.0000, 72.2500, 73.0000
3d   AP:81.0000, 82.0000, 83.0000
"""


def test_bev05_reads_the_r40_iou05_block_moderate_column():
    assert r42.bev05_moderate(RESULT) == 72.25


def test_bev05_refuses_a_string_without_the_block():
    with pytest.raises(ValueError):
        r42.bev05_moderate(RESULT.split('Car AP_R40@0.70, 0.50, 0.50:')[0])


class FakeOfficial:
    def __init__(self):
        self.calls = 0

    def get_official_eval_result(self, g, d, classes):
        self.calls += 1
        return RESULT, {'Car_bev/moderate_R40': 32.5, 'Car_3d/moderate_R40': 42.0}


def fake_score(gt, dets, cone, protocol, official, cls='Car'):
    """Stands in for pandaset_eval_protocols.score: it calls the evaluator once and returns its IoU-0.7 dict entries."""
    _, ap = official.get_official_eval_result(gt, dets, [cls])
    return ap['Car_bev/moderate_R40'], ap['Car_3d/moderate_R40']


def test_score_both_keeps_the_iou07_values_and_adds_iou05_from_the_same_call():
    off = FakeOfficial()
    assert r42.score_both(fake_score, [], [], False, 'rule_A', off) == (32.5, 42.0, 72.25)
    assert off.calls == 1


def test_wrapper_passes_the_evaluator_result_through_unchanged():
    keep = r42.KeepResultString(FakeOfficial())
    out = keep.get_official_eval_result([], [], ['Car'])
    assert out == (RESULT, {'Car_bev/moderate_R40': 32.5, 'Car_3d/moderate_R40': 42.0}) and keep.last == RESULT


def test_rows_pair_every_pandar64_adabn_row_with_its_banked_evaluation():
    rows = r42.rows()
    p_rows = {(r[1], r[2], r[3]) for r in m42.ROWS if r[0] == 'P'}
    assert set(r42.NO_ADABN_PKL) == p_rows
    assert len(rows) == 16 and sum(r['adabn'] for r in rows) == 8
    assert len({r['pkl'] for r in rows}) == 16
