"""IA-SSD target assignment against the pipeline's sentinel labels (memory/repo/bug_prevention_checklist.md, rule 1).

`IASSD_Head.assign_stack_targets_IASSD` is called unbound with a minimal `self` (num_class + the IA-SSD box coder),
and its GPU point-in-box kernel is swapped for the CPU one, so this runs on the master node.

Pinned:
  * positive labels only (every source-only row): labels equal an independent point-in-box computation, the
    enlarged-but-not-inside band is -1, and gt_box_of_fg_points has exactly one row per positive point - the pairing
    the corner, centre-ness and vote losses rely on;
  * a NEGATIVE label (pseudo-label ignore band) and -99 (IGNORE_CLASS_LABEL) are ignore regions: their points are -1,
    carry no box target, and do not lengthen gt_box_of_fg_points; -99 no longer reaches mean_size[label - 1];
  * all-zero padding rows assign nothing;
  * the same on the extended-box path (use_ex_gt_assign) the config's vote assignment takes.
"""
import types

import numpy as np
import pytest
import torch

pytest.importorskip('pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils')
from pcdet.models.dense_heads import IASSD_head as head_mod  # noqa: E402
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils  # noqa: E402
from pcdet.utils import box_coder_utils, box_utils  # noqa: E402

MEAN = [[4.63, 1.96, 1.74], [0.73, 0.67, 1.77], [1.70, 0.61, 1.30]]


def _cpu_points_in_boxes(points, boxes):
    """CPU stand-in for points_in_boxes_gpu: (1, N, 3), (1, T, 7) -> (1, N) index of the first containing box, or -1."""
    inside = roiaware_pool3d_utils.points_in_boxes_cpu(points[0].float(), boxes[0].float()) > 0  # (T, N)
    idx = torch.full((points.shape[1],), -1, dtype=torch.int)
    hit = inside.any(dim=0)
    idx[hit] = torch.argmax(inside.int(), dim=0)[hit].int()
    return idx[None]


@pytest.fixture
def head(monkeypatch):
    monkeypatch.setattr(head_mod.roiaware_pool3d_utils, 'points_in_boxes_gpu', _cpu_points_in_boxes)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)  # the coder moves mean_size to CUDA in __init__
    coder = box_coder_utils.PointResidual_BinOri_Coder(angle_bin_num=12, use_mean_size=True, mean_size=MEAN)
    return types.SimpleNamespace(num_class=3, box_coder=coder)


def _scene(labels):
    """Three separated unit-ish boxes along x, points inside each, in the 0.5 m band around each, and far away."""
    rng = np.random.RandomState(0)
    boxes, pts = [], []
    for i, lab in enumerate(labels):
        cx = 10.0 * i
        boxes.append([cx, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0, lab])
        pts.append(np.c_[rng.uniform(cx - 1.8, cx + 1.8, 40), rng.uniform(-0.9, 0.9, 40), rng.uniform(-0.7, 0.7, 40)])
        pts.append(np.c_[rng.uniform(cx + 2.05, cx + 2.15, 10), rng.uniform(-0.5, 0.5, 10), np.zeros(10)])  # band
    pts.append(np.c_[rng.uniform(100, 110, 30), rng.uniform(-5, 5, 30), np.zeros(30)])  # background
    boxes.append([0.0] * 8)  # padding row, as the collate pads gt_boxes
    pts = np.concatenate(pts).astype(np.float32)
    points = torch.from_numpy(np.c_[np.zeros(len(pts)), pts].astype(np.float32))
    gt = torch.tensor([boxes], dtype=torch.float32)
    return points, gt


def _assign(head, points, gt, **kw):
    ext = box_utils.enlarge_box3d(gt.view(-1, 8), extra_width=[0.2, 0.2, 0.2]).view(1, -1, 8)
    return head_mod.IASSD_Head.assign_stack_targets_IASSD(
        head, points=points, gt_boxes=gt, extend_gt_boxes=ext, set_ignore_flag=True, ret_box_labels=True, **kw)


def _expected_labels(points, gt):
    inside = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 1:4], gt[0, :, :7]) > 0  # (T, N)
    ext = box_utils.enlarge_box3d(gt[0], extra_width=[0.2, 0.2, 0.2])
    inside_ext = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 1:4], ext[:, :7]) > 0
    lab = torch.zeros(points.shape[0], dtype=torch.long)
    lab[inside_ext.any(0) & ~inside.any(0)] = -1
    for t in range(gt.shape[1]):
        c = int(gt[0, t, 7])
        if c != 0:
            lab[inside[t]] = c if c > 0 else -1
    return lab


def test_positive_labels_match_an_independent_assignment(head):
    points, gt = _scene([1, 2, 3])
    out = _assign(head, points, gt)
    assert torch.equal(out['point_cls_labels'], _expected_labels(points, gt))
    pos = out['point_cls_labels'] > 0
    assert out['gt_box_of_fg_points'].shape[0] == int(pos.sum()) == 120
    assert (out['point_box_labels'][~pos].abs().sum() == 0) and (out['point_box_labels'][pos].abs().sum(1) > 0).all()
    assert (out['point_cls_labels'] == -1).sum() > 0  # the enlarged band is ignored


@pytest.mark.parametrize('ignore_label', [-1, -2, -3, -99])
def test_negative_label_is_an_ignore_region(head, ignore_label):
    points, gt = _scene([1, ignore_label, 3])
    out = _assign(head, points, gt)
    lab = out['point_cls_labels']
    assert torch.equal(lab, _expected_labels(points, gt))
    in_ignored = (points[:, 1] > 8) & (points[:, 1] < 12) & (points[:, 2].abs() < 1)
    assert (lab[in_ignored] == -1).all()
    pos = lab > 0
    assert out['gt_box_of_fg_points'].shape[0] == int(pos.sum()) == 80  # only the two real boxes' points
    assert (out['gt_box_of_fg_points'][:, 7] > 0).all()
    assert out['point_box_labels'][in_ignored].abs().sum() == 0


@pytest.mark.parametrize('ignore_label', [-1, -99])
def test_extended_box_path_also_ignores(head, ignore_label):
    points, gt = _scene([2, ignore_label])
    ext = box_utils.enlarge_box3d(gt.view(-1, 8), extra_width=[1.0, 1.0, 1.0]).view(1, -1, 8)
    out = head_mod.IASSD_Head.assign_stack_targets_IASSD(
        head, points=points, gt_boxes=gt, extend_gt_boxes=ext, set_ignore_flag=True, ret_box_labels=True,
        use_ex_gt_assign=True, fg_pc_ignore=False)
    lab = out['point_cls_labels']
    near_ignored = (points[:, 1] > 7) & (points[:, 1] < 13)
    assert (lab[near_ignored] <= 0).all()
    assert out['gt_box_of_fg_points'].shape[0] == int((lab > 0).sum())
    assert (out['gt_box_of_fg_points'][:, 7] > 0).all()


def test_padding_rows_assign_nothing(head):
    points, gt = _scene([1])
    out = _assign(head, points, gt)
    far = points[:, 1] > 50
    assert (out['point_cls_labels'][far] == 0).all()
