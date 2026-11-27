"""SELF_TRAIN.THRESH_BY_RANGE (pcdet/utils/self_training_utils.py; experiments_md 20261003_05 §5 item 63, design fixed
2026-11-27 13:02 JST): one pseudo-label cut per class and x-y range bin, no ignore band. Rule 8: a one-bin table equal to
NEG = SCORE reproduces the existing filter exactly. Run from the repo root."""
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))
import _init_path  # noqa: F401,E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.utils import self_training_utils as stu  # noqa: E402

LEAF = 'cfgs/da-ieee-access-analysis/item59/centerpoint-item59-b-bn-s666.yaml'      # NEG = SCORE = [0.077, 0.089, 0.089]
DIMS = {1: (4.0, 1.8, 1.6), 2: (0.8, 0.8, 1.7), 3: (1.8, 0.6, 1.7)}


@pytest.fixture
def st_cfg(monkeypatch):
    old = os.getcwd()
    os.chdir(ROOT / 'tools')
    try:
        c = cfg_from_yaml_file(LEAF, EasyDict())
    finally:
        os.chdir(old)
    monkeypatch.setattr(stu, 'cfg', c)
    stu.PSEUDO_LABELS.clear(); stu.NEW_PSEUDO_LABELS.clear()
    yield c
    stu.PSEUDO_LABELS.clear(); stu.NEW_PSEUDO_LABELS.clear()


def pred(rows):
    boxes = torch.tensor([[x, y, 0.0, *DIMS[c], 0.0] for x, y, c, s in rows], dtype=torch.float32)
    return {'pred_boxes': boxes, 'pred_scores': torch.tensor([s for *_, s in rows], dtype=torch.float32),
            'pred_labels': torch.tensor([c for _, _, c, _ in rows])}


def run(rows):
    pos, ign = stu.save_pseudo_label_batch({'frame_id': ['f']}, pred_dicts=[pred(rows)], need_update=False,
                                           student_ontology='kitti')
    out = stu.NEW_PSEUDO_LABELS['f']['gt_boxes'].copy()
    stu.NEW_PSEUDO_LABELS.clear()
    return out, pos, ign


ROWS = [(5, 0, 1, 0.08), (5, 1, 1, 0.07), (25, 0, 1, 0.20), (25, 3, 1, 0.10), (50, 0, 1, 0.30), (50, 4, 1, 0.50),
        (8, 2, 2, 0.09), (8, 3, 2, 0.05), (30, 0, 3, 0.10)]


def test_one_bin_equal_to_neg_and_score_is_the_existing_filter(st_cfg):
    assert list(st_cfg.SELF_TRAIN.NEG_THRESH) == list(st_cfg.SELF_TRAIN.SCORE_THRESH)
    base, _, _ = run(ROWS)
    st_cfg.SELF_TRAIN.THRESH_BY_RANGE = EasyDict(EDGES=[0.0], CUTS=[[v] for v in st_cfg.SELF_TRAIN.NEG_THRESH])
    got, _, ign = run(ROWS)
    assert np.array_equal(base, got) and base.shape[0] == 7                 # 5 cars >= 0.077, 1 ped and 1 cyc >= 0.089
    assert all(m.val == 0 for m in ign.meters)


def test_per_bin_cuts_keep_the_right_boxes_and_write_no_ignore_label(st_cfg):
    st_cfg.SELF_TRAIN.THRESH_BY_RANGE = EasyDict(EDGES=[0.0, 10.0, 20.0, 30.0, 40.0],
                                                 CUTS=[[0.075, 0.1, 0.15, 0.2, 0.4], [0.06, 0.1, 0.1, 0.1, 0.1],
                                                       [0.2, 0.2, 0.2, 0.05, 0.2]])
    got, _, ign = run(ROWS)
    kept = sorted((round(float(b[0]), 1), round(float(b[1]), 1), int(b[7])) for b in got)
    assert kept == [(5.0, 0.0, 1), (8.0, 2.0, 2), (25.0, 0.0, 1), (30.0, 0.0, 3), (50.0, 4.0, 1)]
    assert (got[:, 7] > 0).all() and all(m.val == 0 for m in ign.meters)


def test_range_cuts_refuses_bad_labels_and_tables():
    by = EasyDict(EDGES=[0.0, 20.0], CUTS=[[0.1, 0.2]])
    boxes = np.array([[5.0, 0, 0], [25.0, 0, 0]])
    assert stu.range_cuts(boxes, np.array([1, 1]), by).tolist() == [0.1, 0.2]
    with pytest.raises(ValueError):
        stu.range_cuts(boxes, np.array([-1, 1]), by)                 # an ignore label must never reach it
    with pytest.raises(ValueError):
        stu.range_cuts(boxes, np.array([1, 2]), by)                  # no row for class 2
    with pytest.raises(ValueError):
        stu.range_cuts(boxes, np.array([1, 1]), EasyDict(EDGES=[5.0, 20.0], CUTS=[[0.1, 0.2]]))
