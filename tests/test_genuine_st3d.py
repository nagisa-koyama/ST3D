"""The genuine ST3D row (experiments_md/20261015_01 §2.4): CPU checks before its first GPU step.

Covered, with the sentinel values of memory/repo/bug_prevention_checklist.md:
1. `consistency_ensemble` on CenterPoint-shaped predictions (three classes, class-agnostic nothing): negative
   ignore-band labels (-1..-3) survive a pass, a voted ignore keeps its class under MEMORY_VOTING.KEEP_CLASS
   (absent = the published -1), a box that reappears with a higher score is restored, and UNION_CLASSES stops a
   Car-only memory being matched against a Pedestrian-only prediction.
2. The whole pseudo-label path `save_pseudo_label_batch` (score bands -> negative labels -> memory ensemble)
   over several passes, then `fill_pseudo_labels` and CenterHead's `assign_targets`: every label the path can
   produce is in {+-1..+-3}, no ignored box becomes a positive peak, an ignored footprint is out of the loss, and
   the class-less sentinels (-99, 0) are handled by the consumer.
3. SRC.USE_DATA False with BACKWARD_TOGETHER: `train_one_epoch_st` trains on the target alone.
4. The config itself: keys, and that it differs from the frozen single-pass S2 row in the teacher policy.

`.cuda()` and the GPU IoU kernel are replaced by CPU equivalents (this host has no GPU); the code under test is
otherwise the real code.
"""
import copy
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

CFG_PATH = 'cfgs/da-ieee-access/centerpoint-st3d-genuine-legal-nuscenes2kitti.yaml'
NAMES = ['Car', 'Pedestrian', 'Cyclist']


def _load_cfg():
    import os
    old = os.getcwd()
    os.chdir(ROOT / 'tools')
    try:
        return cfg_from_yaml_file(CFG_PATH, EasyDict())
    finally:
        os.chdir(old)


@pytest.fixture(scope='module')
def real_cfg():
    return _load_cfg()


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


DIMS = {1: (4.0, 1.8, 1.6), 2: (0.8, 0.8, 1.7), 3: (1.8, 0.6, 1.7)}


def _row(x, y, label, score):
    d = DIMS[abs(label)]
    return [x, y, 0.0, *d, 0.0, float(label), score]


def _infos(rows, classes=None):
    gt = np.array(rows, dtype=np.float32).reshape(-1, 9)
    names = np.array(classes if classes is not None
                     else [NAMES[abs(int(r[7])) - 1] if r[7] > 0 else 'Ignored' for r in gt])
    # CenterPoint writes cls_scores / iou_scores = the peak score when the head has neither
    return {'gt_boxes': gt, 'cls_scores': gt[:, 8].copy(), 'iou_scores': gt[:, 8].copy(),
            'memory_counter': np.zeros(len(gt)), 'teacher_classes': names}


def _cfg(keep_class=True, union=True, empty=True):
    voting = {'ENABLED': True, 'IGNORE_THRESH': 2, 'RM_THRESH': 3}
    if keep_class:
        voting['KEEP_CLASS'] = True
    if empty:
        voting['VOTE_ON_EMPTY'] = True
    c = {'ENABLED': True, 'NAME': 'consistency_ensemble', 'IOU_THRESH': 0.1, 'MEMORY_VOTING': voting}
    if union:
        c['UNION_CLASSES'] = True
    return EasyDict(c)


def _merge(a, b, ens_cfg):
    return meu.memory_ensemble(copy.deepcopy(a), copy.deepcopy(b), ens_cfg, meu.consistency_ensemble)


def _labels(info):
    return sorted(info['gt_boxes'][:, 7].tolist())


# ------------------------------------------------------------------ 1. consistency_ensemble

def test_higher_score_wins_and_negative_ignore_labels_survive():
    a = _infos([_row(0, 0, 1, 0.6), _row(10, 0, 2, 0.5), _row(20, 0, -3, 0.2)])
    b = _infos([_row(0.1, 0, 1, 0.8), _row(10, 0, -2, 0.3), _row(20, 0, -3, 0.25)])
    out = _merge(a, b, _cfg())
    got = {(round(float(r[0])), int(r[7])): float(r[8]) for r in out['gt_boxes']}
    # Car: b's box and score (0.8 > 0.6). Pedestrian a (+2, 0.5) beats b's ignore-band (-2, 0.3): stays positive.
    # Cyclist ignore band: b (0.25) > a (0.2) replaces it, still -3.
    assert got == {(0, 1): pytest.approx(0.8), (10, 2): pytest.approx(0.5), (20, -3): pytest.approx(0.25)}
    assert set(out['memory_counter'].tolist()) == {0.0}
    assert len(out['teacher_classes']) == 3 and len(out['cls_scores']) == 3


def test_voting_keeps_the_class_of_an_ignored_box_only_with_keep_class():
    a = _infos([_row(0, 0, 1, 0.6), _row(10, 0, 2, 0.5)])
    gone = _infos([_row(30, 0, 1, 0.7)])           # neither a-box is seen again
    kept = a
    for _ in range(2):                              # IGNORE_THRESH 2: second miss turns the box into an ignore
        kept = _merge(kept, gone, _cfg(keep_class=True))
    assert sorted(r[7] for r in kept['gt_boxes'] if abs(r[0] - 10) < 1) == [-2.0], 'Pedestrian must stay -2'
    published = a
    for _ in range(2):
        published = _merge(published, gone, _cfg(keep_class=False))
    assert sorted(r[7] for r in published['gt_boxes'] if abs(r[0] - 10) < 1) == [-1.0], 'published: -1'
    # RM_THRESH 3: the third miss removes both
    kept = _merge(kept, gone, _cfg())
    assert not any(abs(r[0]) < 1 or abs(r[0] - 10) < 1 for r in kept['gt_boxes'])


def test_a_voted_ignore_is_restored_by_a_higher_score_prediction():
    a = _infos([_row(0, 0, 1, 0.4)])
    gone = _infos([_row(30, 0, 1, 0.7)])
    for _ in range(2):
        a = _merge(a, gone, _cfg())
    assert -1.0 in a['gt_boxes'][:, 7]
    back = _merge(a, _infos([_row(0, 0, 1, 0.9)]), _cfg())
    r = back['gt_boxes'][np.abs(back['gt_boxes'][:, 0]) < 1][0]
    assert r[7] == 1.0 and r[8] == pytest.approx(0.9)


def test_empty_prediction_for_one_class_does_not_corrupt_the_others():
    a = _infos([_row(0, 0, 1, 0.6), _row(10, 0, 3, 0.5)])
    b = _infos([_row(0, 0, 1, 0.7)])               # no Cyclist this pass
    out = _merge(a, b, _cfg())
    assert _labels(out) == [1.0, 3.0]
    assert float(out['memory_counter'][out['gt_boxes'][:, 7] == 3][0]) == 1.0   # counted as missed, not dropped


def test_union_classes_stops_cross_class_matching():
    a = _infos([_row(0, 0, 1, 0.6)])               # Car only
    b = _infos([_row(0, 0, 2, 0.9)])               # a Pedestrian at the same place, much smaller
    # make them overlap enough to match if (wrongly) compared: same centre, IoU of the volumes is small, use a low IoU_THRESH
    cfg = _cfg(union=False)
    cfg.IOU_THRESH = 0.001
    wrong = _merge(a, b, cfg)
    right_cfg = _cfg(union=True)
    right_cfg.IOU_THRESH = 0.001
    right = _merge(a, b, right_cfg)
    assert _labels(wrong) == [2.0], 'without UNION_CLASSES the Pedestrian replaced the Car (the hazard)'
    assert _labels(right) == [-1.0 + 0 if False else 1.0, 2.0]


def test_ensemble_with_no_kwargs_matches_the_published_function_on_one_class():
    """Absent keys change nothing for a single class (KEEP_CLASS: -1 == -|1|)."""
    a = _infos([_row(0, 0, 1, 0.6), _row(8, 0, 1, 0.5)])
    b = _infos([_row(0, 0, 1, 0.7)])
    x = _merge(a, b, _cfg(keep_class=False, union=False, empty=False))
    y = _merge(a, b, _cfg(keep_class=True, union=True, empty=True))
    np.testing.assert_array_equal(x['gt_boxes'], y['gt_boxes'])


# ------------------------------------------------------------------ 2. the whole pseudo-label path

class _Pred:
    """CenterPoint-shaped pred_dicts for one frame."""

    @staticmethod
    def make(rows):
        # rows: (x, y, class 1..3, score)
        if not rows:
            return {'pred_boxes': torch.zeros(0, 7), 'pred_scores': torch.zeros(0), 'pred_labels': torch.zeros(0, dtype=torch.long)}
        boxes = torch.tensor([[x, y, 0.0, *DIMS[c], 0.0] for x, y, c, s in rows])
        return {'pred_boxes': boxes, 'pred_scores': torch.tensor([s for *_, s in rows]),
                'pred_labels': torch.tensor([c for _, _, c, _ in rows])}


@pytest.fixture
def st_cfg(monkeypatch, real_cfg):
    c = copy.deepcopy(real_cfg)
    c.SELF_TRAIN.SCORE_THRESH = [0.5, 0.4, 0.4]
    c.SELF_TRAIN.NEG_THRESH = [0.2, 0.2, 0.2]
    monkeypatch.setattr(stu, 'cfg', c)
    stu.PSEUDO_LABELS.clear()
    stu.NEW_PSEUDO_LABELS.clear()
    yield c
    stu.PSEUDO_LABELS.clear()
    stu.NEW_PSEUDO_LABELS.clear()


def _pass(rows, need_update):
    stu.save_pseudo_label_batch({'frame_id': ['f0']}, pred_dicts=[_Pred.make(rows)], need_update=need_update,
                                student_ontology='kitti')
    stu.PSEUDO_LABELS.clear()
    stu.PSEUDO_LABELS.update(stu.NEW_PSEUDO_LABELS)
    stu.NEW_PSEUDO_LABELS.clear()
    return stu.PSEUDO_LABELS['f0']['gt_boxes']


def test_score_bands_become_negative_labels_and_memory_votes_over_passes(st_cfg):
    p0 = _pass([(0, 0, 1, 0.9), (10, 0, 2, 0.45), (20, 0, 3, 0.25), (30, 0, 1, 0.1)], need_update=False)
    # thresholds: Car 0.5/0.2, Ped 0.4/0.2, Cyc 0.4/0.2.  0.1 is under NEG -> dropped; 0.25 Cyc is the ignore band.
    assert sorted(p0[:, 7].tolist()) == [-3.0, 1.0, 2.0]
    # pass 2: Car still seen, the Pedestrian and Cyclist vanish
    p1 = _pass([(0, 0, 1, 0.9)], need_update=True)
    assert sorted(p1[:, 7].tolist()) == [-3.0, 1.0, 2.0] and len(p1) == 3
    # pass 3: second miss -> voted ignore, class kept (-2), the Cyclist (already -3) stays -3
    p2 = _pass([(0, 0, 1, 0.9)], need_update=True)
    assert sorted(p2[:, 7].tolist()) == [-3.0, -2.0, 1.0]
    # pass 4: third miss -> removed
    p3 = _pass([(0, 0, 1, 0.9)], need_update=True)
    assert p3[:, 7].tolist() == [1.0]
    # every label the path can write is one a consumer knows
    for p in (p0, p1, p2, p3):
        assert set(np.abs(p[:, 7]).tolist()) <= {1.0, 2.0, 3.0}


def test_pseudo_labels_reach_centerhead_as_positives_and_ignore_regions(st_cfg):
    p0 = _pass([(-6, -6, 1, 0.9), (-6, 0, 2, 0.8), (-6, 6, 3, 0.7), (6, -6, 1, 0.3), (6, 0, 2, 0.3), (6, 6, 3, 0.3)],
               need_update=False)
    assert sorted(p0[:, 7].tolist()) == [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]

    ds = DatasetTemplate.__new__(DatasetTemplate)
    ds.class_names = list(NAMES)
    ds.pseudo_labels = stu.PSEUDO_LABELS
    inp = {'frame_id': 'f0'}
    ds.fill_pseudo_labels(inp)
    assert list(inp['pos_ps_bbox']) == [1, 1, 1] and list(inp['ign_ps_bbox']) == [1, 1, 1]
    labels = inp['gt_classes']
    rows = [[*inp['gt_boxes'][i], float(labels[i])] for i in range(len(labels))]
    rows.append([0.0, 9.0, 0.0, 8.0, 2.5, 3.0, 0.0, float(IGNORE_CLASS_LABEL)])     # -99, class-less
    head = _head()
    t = head.assign_targets(torch.tensor([rows + [[0.0] * 8]], dtype=torch.float32), feature_map_size=[50, 50])
    hm = t['heatmaps'][0][0]
    assert int(hm.eq(1).sum()) == 3, 'only the three positive pseudo-labels may peak'
    for x, y, c in [(-6, -6, 1), (-6, 0, 2), (-6, 6, 3)]:
        assert hm[c - 1, int((y + 10) / 0.4), int((x + 10) / 0.4)] == 1
    mask = t['heatmap_masks'][0][0]
    for x, y in [(6, -6), (6, 0), (6, 6), (0, 9)]:
        assert mask[int((y + 10) / 0.4), int((x + 10) / 0.4)] == 0, 'ignored footprint must be out of the loss'
    assert int(t['masks'][0].sum()) == 3


def test_label_minus_99_never_reaches_fill_pseudo_labels_but_would_fail_loudly(st_cfg):
    """-99 is a SOURCE sentinel. Fed to the pseudo-label consumer it indexes class_names out of range and raises,
    rather than wrapping to a permuted class: nothing in the ensemble can write it (test above)."""
    stu.PSEUDO_LABELS['f0'] = {'gt_boxes': np.array([_row(0, 0, 1, 0.5)], dtype=np.float32)}
    stu.PSEUDO_LABELS['f0']['gt_boxes'][0, 7] = IGNORE_CLASS_LABEL
    ds = DatasetTemplate.__new__(DatasetTemplate)
    ds.class_names = list(NAMES)
    ds.pseudo_labels = stu.PSEUDO_LABELS
    with pytest.raises(IndexError):
        ds.fill_pseudo_labels({'frame_id': 'f0'})


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


# ------------------------------------------------------------------ 3. SRC.USE_DATA False + BACKWARD_TOGETHER

def test_target_only_iteration_trains_with_backward_together(monkeypatch, real_cfg):
    import train_utils.train_st_utils as tsu
    c = copy.deepcopy(real_cfg)
    monkeypatch.setattr(tsu, 'cfg', c)
    monkeypatch.setattr(tsu.wandb, 'log', lambda *a, **k: None)
    assert c.SELF_TRAIN.SRC.USE_DATA is False and c.SELF_TRAIN.SRC.BACKWARD_TOGETHER
    assert c.SELF_TRAIN.TAR.USE_DATA and c.SELF_TRAIN.TAR.BACKWARD_TOGETHER

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    w0 = model.weight.detach().clone()

    class _Sched:
        def step(self, it):
            pass

    batches = [{'x': torch.randn(3, 4), 'pos_ps_bbox': torch.ones(3, 3), 'ign_ps_bbox': torch.zeros(3, 3)}
               for _ in range(4)]

    def model_func(m, batch):
        loss = (m(batch['x']) ** 2).mean()
        return loss, {}, {}, None

    class _DS:
        class_names = NAMES

    class _Loader:
        class dataset:                      # noqa: N801
            dataset = _DS()

        def __len__(self):
            return len(batches)

        def __iter__(self):
            return iter(batches)

    class _Tbar:
        def set_postfix(self, *a, **k):
            pass

        def refresh(self):
            pass

    n = tsu.train_one_epoch_st(model, opt, [], _Loader(), model_func, _Sched(), 0, EasyDict({'GRAD_NORM_CLIP': 10}),
                               rank=0, tbar=_Tbar(), total_it_each_epoch=4, dataloader_iter=iter(batches))
    assert n == 4
    assert not torch.equal(model.weight.detach(), w0), 'the target loss alone must move the weights'


# ------------------------------------------------------------------ 4. the config

def test_config_is_the_unfrozen_consistency_ensemble_target_only_row(real_cfg):
    s = real_cfg.SELF_TRAIN
    assert s.TEACHER_IS_STUDENT is True and s.FROZEN_TEACHER_SINGLE_PASS is False
    assert s.SRC.USE_DATA is False and s.TAR.USE_DATA is True
    assert s.MEMORY_ENSEMBLE.NAME == 'consistency_ensemble' and s.MEMORY_ENSEMBLE.IOU_THRESH == 0.1
    assert s.MEMORY_ENSEMBLE.MEMORY_VOTING.IGNORE_THRESH == 2 and s.MEMORY_ENSEMBLE.MEMORY_VOTING.RM_THRESH == 3
    assert s.PROG_AUG.ENABLED and list(s.PROG_AUG.UPDATE_AUG) == [5, 10, 20, 25]
    assert s.UPDATE_PSEUDO_LABEL_INTERVAL == 2 and list(s.UPDATE_PSEUDO_LABEL) == [0]
    assert real_cfg.OPTIMIZATION.NUM_EPOCHS == 30 and real_cfg.OPTIMIZATION.LR == 0.0015
    assert real_cfg.DATA_CONFIG_TAR.USE_PSEUDO_LABEL is True
    assert real_cfg.DATA_CONFIG.MAX_SWEEPS == 1
    names = [a['NAME'] for a in real_cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.AUG_CONFIG_LIST]
    assert names == ['random_object_scaling', 'random_object_rotation', 'random_world_flip',
                     'random_world_rotation', 'random_world_scaling']
    assert 'random_object_rotation' not in real_cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST
    assert len(s.SCORE_THRESH) == len(s.NEG_THRESH) == 3
    assert all(n < sc or n == sc for n, sc in zip(s.NEG_THRESH, s.SCORE_THRESH))


def test_vote_on_empty_is_opt_in_and_the_published_early_return_never_forgets():
    a = _infos([_row(0, 0, 2, 0.5)])
    empty = _infos([])
    published = a
    for _ in range(5):
        published = _merge(published, empty, _cfg(empty=False))
    assert len(published['gt_boxes']) == 1 and published['gt_boxes'][0, 7] == 2.0, 'published: never ignored or removed'
    voted = a
    for _ in range(2):
        voted = _merge(voted, empty, _cfg())
    assert voted['gt_boxes'][0, 7] == -2.0
    voted = _merge(voted, empty, _cfg())
    assert len(voted['gt_boxes']) == 0
