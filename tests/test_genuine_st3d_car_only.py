"""The Car-only variant of the genuine ST3D row 28650 (experiments_md/20261017_04): CPU checks before its GPU step.

The variant changes no code: Pedestrian / Cyclist SCORE_THRESH = NEG_THRESH = 1.01 makes the existing NEG_THRESH
removal of `save_pseudo_label_batch` drop every prediction of those classes (scores are sigmoids, max 1.0). Covered,
with the sentinel values of memory/repo/bug_prevention_checklist.md:
1. The config differs from 28650's in exactly SELF_TRAIN.SCORE_THRESH / NEG_THRESH, and only in the Pedestrian and
   Cyclist entries (Car keeps 0.589).
2. Pedestrian / Cyclist predictions at any score, 1.0 included, never reach the stored pseudo-labels; a frame with
   only such predictions stores an empty (0, 9) array that the consumer accepts.
3. Memory voting over passes with a Car-only memory (consistency_ensemble, KEEP_CLASS, UNION_CLASSES, VOTE_ON_EMPTY
   as in 28650): every stored label is +1 or -1; with a non-empty current prediction the three opt-in keys give the
   released single-class result bit for bit; on an empty current prediction VOTE_ON_EMPTY forgets (the released
   early return never does) - the one disclosed difference.
4. The stored labels through `fill_pseudo_labels` and CenterHead's `assign_targets` (with -99 appended): Car
   positives peak in channel 0, the Pedestrian / Cyclist channels hold no peak and no gaussian, Car ignores leave
   the loss.

`.cuda()` and the GPU IoU kernel are replaced by CPU equivalents (this host has no GPU); the code under test is
otherwise the real code.
"""
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
from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.models.dense_heads.center_head import CenterHead  # noqa: E402
from pcdet.utils import memory_ensemble_utils as meu  # noqa: E402
from pcdet.utils import self_training_utils as stu  # noqa: E402
from pcdet.utils.common_utils import IGNORE_CLASS_LABEL  # noqa: E402

CAR_ONLY = 'cfgs/da-ieee-access/centerpoint-st3d-genuine-legal-caronly-nuscenes2kitti.yaml'
GENUINE = 'cfgs/da-ieee-access/centerpoint-st3d-genuine-legal-nuscenes2kitti.yaml'
NAMES = ['Car', 'Pedestrian', 'Cyclist']
DIMS = {1: (4.0, 1.8, 1.6), 2: (0.8, 0.8, 1.7), 3: (1.8, 0.6, 1.7)}


def _load(path):
    old = os.getcwd()
    os.chdir(ROOT / 'tools')
    try:
        return cfg_from_yaml_file(path, EasyDict())
    finally:
        os.chdir(old)


@pytest.fixture(scope='module')
def car_cfg():
    return _load(CAR_ONLY)


@pytest.fixture(scope='module')
def genuine_cfg():
    return _load(GENUINE)


@pytest.fixture(autouse=True)
def cpu_iou(monkeypatch):
    """CPU stand-ins for .cuda() and boxes_iou3d_gpu: exact 3D IoU of yaw-0 boxes (all the fixtures use yaw 0)."""
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)

    def iou(a, b):
        a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
        lo = torch.maximum(a[:, None, :3] - a[:, None, 3:6] / 2, b[None, :, :3] - b[None, :, 3:6] / 2)
        hi = torch.minimum(a[:, None, :3] + a[:, None, 3:6] / 2, b[None, :, :3] + b[None, :, 3:6] / 2)
        inter = (hi - lo).clamp(min=0).prod(dim=2)
        va, vb = a[:, 3:6].prod(dim=1), b[:, 3:6].prod(dim=1)
        return (inter / (va[:, None] + vb[None, :] - inter)).float()

    monkeypatch.setattr(meu.iou3d_nms_utils, 'boxes_iou3d_gpu', iou)


def _flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = prefix + str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key + '.'))
        else:
            out[key] = v
    return out


# ------------------------------------------------------------------ 1. the config

def test_config_differs_from_28650_only_in_the_pedestrian_and_cyclist_thresholds(car_cfg, genuine_cfg):
    a, b = _flatten(car_cfg), _flatten(genuine_cfg)
    assert a.pop('_BASE_CONFIG_') == GENUINE, 'the variant must inherit 28650 itself'
    b.pop('_BASE_CONFIG_')
    assert set(a) == set(b), 'the variant must resolve to the same keys as 28650'
    differ = {k for k in a if a[k] != b[k]}
    assert differ == {'SELF_TRAIN.SCORE_THRESH', 'SELF_TRAIN.NEG_THRESH'}, differ
    for key in ('SCORE_THRESH', 'NEG_THRESH'):
        new, old = list(car_cfg.SELF_TRAIN[key]), list(genuine_cfg.SELF_TRAIN[key])
        assert new[0] == old[0] == 0.589, 'Car keeps the rule-derived cut of job 28641'
        assert new[1] > 1.0 and new[2] > 1.0, 'Pedestrian / Cyclist must be above any sigmoid score'
    assert car_cfg.CLASS_NAMES == NAMES, 'the network stays 3-class so 25743 loads unchanged'
    assert car_cfg.SELF_TRAIN.MEMORY_ENSEMBLE.MEMORY_VOTING.VOTE_ON_EMPTY is True
    assert car_cfg.SELF_TRAIN.TEACHER_IS_STUDENT is True and car_cfg.OPTIMIZATION.NUM_EPOCHS == 30


# ------------------------------------------------------------------ 2. what is stored

class _Pred:
    @staticmethod
    def make(rows):
        # rows: (x, y, class 1..3, score); CenterPoint-shaped pred_dicts for one frame
        if not rows:
            return {'pred_boxes': torch.zeros(0, 7), 'pred_scores': torch.zeros(0),
                    'pred_labels': torch.zeros(0, dtype=torch.long)}
        boxes = torch.tensor([[x, y, 0.0, *DIMS[c], 0.0] for x, y, c, s in rows])
        return {'pred_boxes': boxes, 'pred_scores': torch.tensor([s for *_, s in rows], dtype=torch.float32),
                'pred_labels': torch.tensor([c for _, _, c, _ in rows])}


@pytest.fixture
def st_cfg(monkeypatch, car_cfg):
    """The REAL Car-only thresholds and memory config (nothing overridden)."""
    c = copy.deepcopy(car_cfg)
    monkeypatch.setattr(stu, 'cfg', c)
    stu.PSEUDO_LABELS.clear()
    stu.NEW_PSEUDO_LABELS.clear()
    yield c
    stu.PSEUDO_LABELS.clear()
    stu.NEW_PSEUDO_LABELS.clear()


def _pass(rows, need_update, frame='f0'):
    pos, ign = stu.save_pseudo_label_batch({'frame_id': [frame]}, pred_dicts=[_Pred.make(rows)],
                                           need_update=need_update, student_ontology='kitti')
    stu.PSEUDO_LABELS.update(stu.NEW_PSEUDO_LABELS)
    stu.NEW_PSEUDO_LABELS.clear()
    return stu.PSEUDO_LABELS[frame]['gt_boxes'], pos, ign


def test_pedestrian_and_cyclist_never_stored_at_any_score(st_cfg):
    sig_max = float(torch.tensor([1e4]).sigmoid())               # the largest score CenterHead can emit
    assert sig_max == 1.0
    # Scores are float32 and the cut float64: float32(0.589) = 0.58899999 < 0.589, so a score equal to the cut is
    # removed (pre-existing, identical in 28650). The Car cases straddle it.
    rows = [(0, 0, 1, 0.95), (8, 0, 1, 0.5891), (16, 0, 1, 0.5889),
            (0, 8, 2, sig_max), (8, 8, 2, 0.9999), (16, 8, 2, 0.6),
            (0, -8, 3, sig_max), (8, -8, 3, 0.7)]
    gt, pos, ign = _pass(rows, need_update=False)
    assert gt.shape == (2, 9)
    assert gt[:, 7].tolist() == [1.0, 1.0], 'only the two Cars above 0.589 are stored, both positive'
    assert sorted(np.round(gt[:, 8], 4).tolist()) == [0.5891, 0.95]
    assert [m.val for m in pos.meters] == [2, 0, 0] and [m.val for m in ign.meters] == [0, 0, 0]
    assert list(stu.PSEUDO_LABELS['f0']['teacher_classes']) == ['Car', 'Car']


def test_a_frame_with_only_pedestrian_predictions_stores_an_empty_array(st_cfg):
    gt, pos, ign = _pass([(0, 0, 2, 1.0), (5, 0, 3, 0.99)], need_update=False)
    assert gt.shape == (0, 9)
    ds = DatasetTemplate.__new__(DatasetTemplate)
    ds.class_names = list(NAMES)
    ds.pseudo_labels = stu.PSEUDO_LABELS
    inp = {'frame_id': 'f0'}
    ds.fill_pseudo_labels(inp)
    assert inp['gt_boxes'].shape == (0, 7) and len(inp['gt_names']) == 0
    assert list(inp['pos_ps_bbox']) == [0, 0, 0] and list(inp['ign_ps_bbox']) == [0, 0, 0]


# ------------------------------------------------------------------ 3. memory voting with a Car-only memory

JUNK = [(4, 0, 2, 0.95), (30, 30, 3, 0.9)]                       # removed before the memory every pass


def test_memory_voting_over_passes_writes_only_plus_and_minus_one(st_cfg):
    g0, _, _ = _pass([(0, 0, 1, 0.9), (10, 0, 1, 0.7)] + JUNK, need_update=False)
    assert g0[:, 7].tolist() == [1.0, 1.0]
    # the Car at x=10 is missed from here on; a Pedestrian sits where it was (it would match across classes if stored)
    seq = [(0, 0, 1, 0.9), (10, 0, 2, 0.99)] + JUNK
    g1, _, _ = _pass(seq, need_update=True)
    assert sorted(g1[:, 7].tolist()) == [1.0, 1.0]
    g2, pos, ign = _pass(seq, need_update=True)
    assert sorted(g2[:, 7].tolist()) == [-1.0, 1.0], 'second miss: voted ignore, written -1 (KEEP_CLASS: -|1|)'
    assert [m.val for m in pos.meters] == [1, 0, 0] and [m.val for m in ign.meters] == [1, 0, 0]
    g3, _, _ = _pass(seq, need_update=True)
    assert g3[:, 7].tolist() == [1.0], 'third miss: removed'
    for g in (g0, g1, g2, g3):
        assert set(g[:, 7].tolist()) <= {1.0, -1.0}


def _seq_under(cfg_mem, passes):
    """Run passes through the real save_pseudo_label_batch with the given MEMORY_ENSEMBLE block."""
    stu.cfg.SELF_TRAIN.MEMORY_ENSEMBLE = cfg_mem
    stu.PSEUDO_LABELS.clear()
    out = []
    for i, rows in enumerate(passes):
        g, _, _ = _pass(rows, need_update=i > 0)
        out.append(g.copy())
    return out


def _released(mem):
    m = copy.deepcopy(mem)
    m.pop('UNION_CLASSES', None)
    m.MEMORY_VOTING.pop('KEEP_CLASS', None)
    m.MEMORY_VOTING.pop('VOTE_ON_EMPTY', None)
    return m


def test_opt_in_keys_equal_the_released_function_whenever_cars_are_predicted(st_cfg):
    mem = copy.deepcopy(st_cfg.SELF_TRAIN.MEMORY_ENSEMBLE)
    passes = [[(0, 0, 1, 0.7), (10, 0, 1, 0.8), (20, 0, 1, 0.6)] + JUNK,
              [(0.2, 0, 1, 0.9), (20, 0, 1, 0.65)] + JUNK,
              [(0.2, 0, 1, 0.6), (30, 0, 1, 0.75)] + JUNK,
              [(30, 0, 1, 0.8)] + JUNK,
              [(0, 0, 1, 0.95), (30, 0, 1, 0.7)] + JUNK]
    ours = _seq_under(copy.deepcopy(mem), passes)
    released = _seq_under(_released(mem), passes)
    for a, b in zip(ours, released):
        np.testing.assert_array_equal(a, b)
    assert any(-1.0 in g[:, 7] for g in ours), 'the sequence must exercise a voted ignore'


def test_vote_on_empty_is_the_one_difference_from_the_released_single_class_code(st_cfg):
    mem = copy.deepcopy(st_cfg.SELF_TRAIN.MEMORY_ENSEMBLE)
    passes = [[(0, 0, 1, 0.9)] + JUNK] + [list(JUNK)] * 3       # after pass 0 no Car is predicted in this frame
    ours = _seq_under(copy.deepcopy(mem), passes)
    released = _seq_under(_released(mem), passes)
    assert [g[:, 7].tolist() for g in ours] == [[1.0], [1.0], [-1.0], []], 'VOTE_ON_EMPTY: ignore at 2, removal at 3'
    assert [g[:, 7].tolist() for g in released] == [[1.0]] * 4, 'released early return: the Car is kept for good'


# ------------------------------------------------------------------ 4. into CenterHead

def _head():
    head = CenterHead.__new__(CenterHead)
    torch.nn.Module.__init__(head)
    head.model_cfg = EasyDict({
        'TARGET_ASSIGNER_CONFIG': {'FEATURE_MAP_STRIDE': 4, 'NUM_MAX_OBJS': 50, 'GAUSSIAN_OVERLAP': 0.1, 'MIN_RADIUS': 2},
        'LOSS_CONFIG': {'LOSS_WEIGHTS': {'cls_weight': 1.0, 'loc_weight': 2.0, 'code_weights': [1.0] * 8}},
    })
    head.class_names = list(NAMES)
    head.class_names_each_head = [list(NAMES)]
    head.point_cloud_range = np.array([-10.0, -10.0, -2.0, 10.0, 10.0, 4.0], dtype=np.float32)
    head.voxel_size = [0.1, 0.1, 0.15]
    head.separate_head_cfg = EasyDict({'HEAD_ORDER': ['center', 'center_z', 'dim', 'rot']})
    head.build_losses()
    return head


def test_car_only_labels_reach_centerhead_with_no_pedestrian_or_cyclist_target(st_cfg):
    keep, lose = (-6, -6, 1, 0.9), (6, 6, 1, 0.8)
    junk = [(-6, 6, 2, 1.0), (6, -6, 3, 0.95), (0, 0, 2, 0.9)]
    _pass([keep, lose] + junk, need_update=False)
    _pass([keep] + junk, need_update=True)
    g, _, _ = _pass([keep] + junk, need_update=True)
    assert sorted(g[:, 7].tolist()) == [-1.0, 1.0]

    ds = DatasetTemplate.__new__(DatasetTemplate)
    ds.class_names = list(NAMES)
    ds.pseudo_labels = stu.PSEUDO_LABELS
    inp = {'frame_id': 'f0'}
    ds.fill_pseudo_labels(inp)
    assert list(inp['pos_ps_bbox']) == [1, 0, 0] and list(inp['ign_ps_bbox']) == [1, 0, 0]
    assert list(inp['gt_names']) == ['Car', 'Car']
    rows = [[*inp['gt_boxes'][i], float(inp['gt_classes'][i])] for i in range(len(inp['gt_classes']))]
    rows.append([0.0, 9.0, 0.0, 8.0, 2.5, 3.0, 0.0, float(IGNORE_CLASS_LABEL)])     # -99, class-less
    t = _head().assign_targets(torch.tensor([rows + [[0.0] * 8]], dtype=torch.float32), feature_map_size=[50, 50])
    hm = t['heatmaps'][0][0]
    assert int(hm[0].eq(1).sum()) == 1 and hm[0, int((-6 + 10) / 0.4), int((-6 + 10) / 0.4)] == 1
    assert float(hm[1:].abs().sum()) == 0.0, 'Pedestrian / Cyclist channels: no peak and no gaussian'
    mask = t['heatmap_masks'][0][0]
    for x, y in [(6, 6), (0, 9)]:
        assert mask[int((y + 10) / 0.4), int((x + 10) / 0.4)] == 0, 'ignored footprint must be out of the loss'
    assert int(t['masks'][0].sum()) == 1
