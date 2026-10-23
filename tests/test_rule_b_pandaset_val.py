"""CPU tests for tools/analysis/pseudo_label_threshold/rule_b_pandaset_val.py (experiments_md 20261022_07 §3.1, §9).

Job 30220 died after the model pass at `frames[extras['frame_key'][b]]`: torch's pin_memory turns plain tuples into lists,
so the (sequence, frame_idx) key arrived unhashable. These tests pin the key normalisation, the batch-to-frames path with
fake predictions in the CenterHead output format (1-based labels), and the rule-B cut on a synthetic case with a known
answer. No GPU, no data.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools' / 'analysis' / 'pseudo_label_threshold'


@pytest.fixture(scope='module')
def rb():
    sys.path.insert(0, str(ROOT / 'tools'))
    sys.path.insert(0, str(TOOL))
    spec = importlib.util.spec_from_file_location('_rule_b_pandaset_val_under_test', TOOL / 'rule_b_pandaset_val.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pin_memory_turns_tuples_into_lists_and_frame_key_fixes_it(rb):
    from torch.utils.data._utils.pin_memory import pin_memory
    if not torch.cuda.is_available():
        pytest.skip('pin_memory needs CUDA; the conversion is read from torch source in 20261022_07 §9')
    pinned = pin_memory({'frame_key': [('045', 3)]})
    assert isinstance(pinned['frame_key'][0], list)
    assert rb.frame_key(pinned['frame_key'][0]) == ('045', 3)


def test_frame_key_accepts_tuple_and_list(rb):
    assert rb.frame_key(('045', 3)) == ('045', 3)
    assert rb.frame_key(['045', 3]) == ('045', 3)
    assert rb.frame_key([np.str_('045'), np.int64(3)]) == ('045', 3)
    hash(rb.frame_key(['045', 3]))


def _gt(cls, x, y, npts=10):
    return [x, y, 0.0, 4.0, 2.0, 1.5, 0.0, cls, npts]


def test_collect_stores_hashable_keys_and_rejects_duplicates(rb):
    frames = {}
    extras = {'frame_key': [['045', 0], ['045', 1]],  # lists, as pin_memory delivers them
              'gt_pack': [np.array([_gt(1, 10, 0)], np.float32), np.zeros((0, 9), np.float32)],
              'n_raw': [100, 90], 'n_thin': [17, 15]}
    preds = [{'pred_boxes': torch.tensor([[10.0, 0, 0, 4, 2, 1.5, 0]]), 'pred_labels': torch.tensor([1]),
              'pred_scores': torch.tensor([0.9])},
             {'pred_boxes': torch.zeros((0, 7)), 'pred_labels': torch.zeros(0, dtype=torch.long),
              'pred_scores': torch.zeros(0)}]
    r, t = rb.collect(frames, extras, preds)
    assert (r, t) == (190, 32)
    assert set(frames) == {('045', 0), ('045', 1)}
    assert frames[('045', 0)]['pred'].shape == (1, 9) and frames[('045', 0)]['pred'][0, 7] == 1
    assert frames[('045', 1)]['pred'].shape == (0, 9) and frames[('045', 1)]['gt'].shape == (0, 9)
    with pytest.raises(AssertionError):
        rb.collect(frames, extras, preds)


def test_rule_b_cut_on_a_known_case(rb):
    # 100 frames: one Car GT matched at score 0.8, two unmatched Car predictions at 0.15 (precision 1/3 below 0.16, 1 above);
    # one Pedestrian GT with 0 raw points (not matchable) and a prediction on it at 0.9 (precision 0 everywhere);
    # one Car GT outside the range with a prediction on it (both excluded).
    frames = {}
    for i in range(100):
        gt = np.array([_gt(1, 10, 0), _gt(2, 5, 5, npts=0), _gt(1, 80, 0)], np.float32)
        pred = np.array([[10.2, 0, 0, 4, 2, 1.5, 0, 1, 0.8], [30, 30, 0, 4, 2, 1.5, 0, 1, 0.15],
                         [-30, 30, 0, 4, 2, 1.5, 0, 1, 0.15], [5, 5, 0, 1, 1, 1.7, 0, 2, 0.9],
                         [80, 0, 0, 4, 2, 1.5, 0, 1, 0.99]], np.float32)
        frames[('s', i)] = {'pred': pred, 'gt': gt}
    cuts, curves = rb.rule_b_cuts(frames, {1: 'Car', 2: 'Pedestrian'})
    assert cuts == {'Car': 0.16, 'Pedestrian': 1.01}
    prec, kept, n_gt = curves[1]
    assert n_gt == 100
    assert kept[rb.GRID == 0.10][0] == 300 and abs(prec[rb.GRID == 0.10][0] - 1 / 3) < 1e-9
    assert kept[rb.GRID == 0.50][0] == 100 and prec[rb.GRID == 0.50][0] == 1.0
    assert curves[2][2] == 0
