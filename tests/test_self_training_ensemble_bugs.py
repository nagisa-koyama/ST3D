"""
Regression tests for pcdet/utils/common_utils.py::mask_dict and
pcdet/utils/memory_ensemble_utils.py::bipartite_ensemble.

Covers two bugs found 2026-08-23 while auditing ST3D's original (pre-UADA3D-migration)
self-training / pseudo-label ensembling code:

1. `mask_dict()` indexed every value in the dict with `value[mask]` unconditionally, crashing
   with `TypeError: 'NoneType' object is not subscriptable` whenever a dict value was `None`
   (e.g. `cls_scores`/`iou_scores`, which are legitimately `None` for detectors that don't
   produce those scores). This is reachable in practice: `memory_ensemble_utils.memory_ensemble()`
   calls `mask_dict()` directly on the raw pseudo-label info dicts whenever self-training runs
   with more than one class (e.g. any `*_car_ped*`/`*_car_ped_cyc*` config with
   `SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED: True`).

2. `bipartite_ensemble()` (one of three interchangeable `MEMORY_ENSEMBLE.NAME` implementations)
   never read, updated, or returned the `teacher_classes` field, unlike its two siblings
   `consistency_ensemble()`/`nms_ensemble()` which both handle it. Selecting
   `MEMORY_ENSEMBLE.NAME: bipartite_ensemble` would silently drop `teacher_classes` from the
   merged pseudo-label dict.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.utils.common_utils import mask_dict  # noqa: E402


def test_mask_dict_skips_none_values():
    result_dict = {
        'gt_boxes': np.arange(12).reshape(4, 3),
        'cls_scores': None,
        'iou_scores': None,
        'memory_counter': np.array([0, 1, 2, 3]),
    }
    mask = np.array([True, False, True, False])

    masked = mask_dict(result_dict, mask)

    assert masked['cls_scores'] is None
    assert masked['iou_scores'] is None
    assert masked['gt_boxes'].shape == (2, 3)
    np.testing.assert_array_equal(masked['memory_counter'], np.array([0, 2]))


def _make_gt_infos(gt_boxes, teacher_classes, memory_counter=None):
    return {
        'gt_boxes': gt_boxes.astype(np.float32),
        'cls_scores': None,
        'iou_scores': None,
        'memory_counter': memory_counter if memory_counter is not None else np.zeros(gt_boxes.shape[0]),
        'teacher_classes': np.array(teacher_classes),
    }


def test_bipartite_ensemble_preserves_teacher_classes(monkeypatch):
    import pcdet.utils.memory_ensemble_utils as meu

    # This host/container has no CUDA device (no GPU on this machine). Stub out the CUDA-only
    # calls (`.cuda()`, `boxes_iou3d_gpu`, and `linear_sum_assignment`'s exact result) with
    # CPU-only/trivial equivalents so the test focuses purely on whether `teacher_classes`
    # survives the merge, matching a trivial 1-to-1 identity assignment.
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)

    class _FakeIouMatrix:
        def cpu(self):
            return self

        def numpy(self):
            return np.eye(1, dtype=np.float32)

    monkeypatch.setattr(
        meu.iou3d_nms_utils, 'boxes_iou3d_gpu', lambda a, b: _FakeIouMatrix()
    )
    monkeypatch.setattr(
        meu, 'linear_sum_assignment', lambda cost: (np.array([0]), np.array([0]))
    )

    gt_infos_a = _make_gt_infos(
        np.array([[0, 0, 0, 1, 1, 1, 0, 1, 0.9]]), teacher_classes=['Car'],
    )
    gt_infos_b = _make_gt_infos(
        np.array([[0, 0, 0, 1, 1, 1, 0, 1, 0.95]]), teacher_classes=['Car'],
    )

    merged = meu.bipartite_ensemble(gt_infos_a, gt_infos_b, EasyDict({'IOU_THRESH': 0.5}))

    assert 'teacher_classes' in merged
    assert len(merged['teacher_classes']) == merged['gt_boxes'].shape[0]
