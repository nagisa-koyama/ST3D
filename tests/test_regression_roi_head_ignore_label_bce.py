"""PointRCNN ROI head: an ignored ROI (label -1) must never reach binary_cross_entropy as a target.

Defect: `RoIHeadTemplate.get_box_cls_layer_loss` handed `rcnn_cls_labels.float()` straight to
`F.binary_cross_entropy`. The ROI target assigner marks ignored ROIs with -1 (the sentinel in
memory/repo/bug_prevention_checklist.md, rule 1) and torch >= 2.x asserts that a BCE target lies in
[0, 1], so the first batch holding an ignored ROI killed the run: job 27466 (PointRCNN S2, 2026-10-06).
Window: the UADA3D port until ST3D 4ac22ee (2026-10-06), which clamps the target FOR THE CALL ONLY and
keeps the `labels >= 0` mask for the reduction, so ignored rows still contribute nothing.
Reports: experiments_md/20261003_13 §3.M (bug prevention), memory/repo/bug_prevention_checklist.md.

CPU only: the method is called unbound with a minimal `self`; torch's CPU kernel raises the same
"between 0 and 1" error as the GPU assert, so the pre-fix formula (copied from `git show 4ac22ee^`)
is shown to die on the very input the fixed method accepts.
"""
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools')); sys.path.insert(0, str(ROOT))
import _init_path  # noqa: F401,E402
from pcdet.models.roi_heads import roi_head_template as rht  # noqa: E402


def _head(cls_loss='BinaryCrossEntropy', weight=1.0):
    return types.SimpleNamespace(model_cfg=EasyDict(LOSS_CONFIG=EasyDict(
        CLS_LOSS=cls_loss, LOSS_WEIGHTS={'rcnn_cls_weight': weight})))


def _loss(head, rcnn_cls, labels):
    return rht.RoIHeadTemplate.get_box_cls_layer_loss(
        head, {'rcnn_cls': rcnn_cls, 'rcnn_cls_labels': labels})


def _pre_fix_reference(head, forward_ret_dict):
    """The BinaryCrossEntropy branch exactly as it stood before 4ac22ee (git show 4ac22ee^:pcdet/models/
    roi_heads/roi_head_template.py, lines 197-205)."""
    loss_cfgs = head.model_cfg.LOSS_CONFIG
    rcnn_cls = forward_ret_dict['rcnn_cls']
    rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
    rcnn_cls_flat = rcnn_cls.view(-1)
    batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
    cls_valid_mask = (rcnn_cls_labels >= 0).float()
    rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
    return rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']


# (batch * rois, 1) logits and their labels: positive, negative, IGNORED (-1), positive.
LOGITS = torch.tensor([[2.0], [-1.0], [0.5], [-3.0]])
LABELS = torch.tensor([[1, 0, -1, 1]], dtype=torch.long)


def test_pre_fix_reference_dies_on_an_ignored_roi():
    with pytest.raises(RuntimeError, match='between 0 and 1'):
        _pre_fix_reference(_head(), {'rcnn_cls': LOGITS, 'rcnn_cls_labels': LABELS})


def test_fixed_loss_is_the_mean_bce_over_the_valid_rows_only():
    loss, tb = _loss(_head(), LOGITS, LABELS)
    valid = LABELS.view(-1) >= 0
    expected = F.binary_cross_entropy(torch.sigmoid(LOGITS.view(-1)[valid]),
                                      LABELS.view(-1)[valid].float(), reduction='mean')
    assert torch.isfinite(loss)
    assert torch.allclose(loss, expected), (loss, expected)
    assert tb == {'rcnn_loss_cls': pytest.approx(expected.item())}


def test_ignored_rows_receive_no_gradient():
    logits = LOGITS.clone().requires_grad_(True)
    loss, _ = _loss(_head(), logits, LABELS)
    loss.backward()
    grad = logits.grad.view(-1)
    assert grad[2] == 0.0, 'the ignored ROI must not train the classifier'
    assert (grad[[0, 1, 3]] != 0).all()


def test_an_all_ignored_batch_gives_zero_loss_without_nan():
    loss, _ = _loss(_head(), LOGITS, torch.full((1, 4), -1, dtype=torch.long))
    assert loss.item() == 0.0 and torch.isfinite(loss)


def test_without_ignored_rois_the_fix_changes_nothing():
    labels = torch.tensor([[1, 0, 0, 1]], dtype=torch.long)
    fixed, _ = _loss(_head(weight=2.0), LOGITS, labels)
    ref = _pre_fix_reference(_head(weight=2.0), {'rcnn_cls': LOGITS, 'rcnn_cls_labels': labels})
    assert torch.allclose(fixed, ref)


def test_the_target_handed_to_bce_lies_in_the_unit_interval(monkeypatch):
    """The GPU assert that killed 27466, pinned without a GPU: capture what the method passes to BCE."""
    seen = []
    real = rht.F.binary_cross_entropy

    def spy(inp, target, *a, **k):
        seen.append(target.detach().clone())
        return real(inp, target, *a, **k)

    monkeypatch.setattr(rht.F, 'binary_cross_entropy', spy)
    _loss(_head(), LOGITS, LABELS)
    assert len(seen) == 1
    assert seen[0].min() >= 0.0 and seen[0].max() <= 1.0, seen[0]
    # ... and the positive / negative rows are passed through unchanged.
    assert torch.equal(seen[0][[0, 1, 3]], torch.tensor([1.0, 0.0, 1.0]))


def test_cross_entropy_branch_also_ignores_minus_one():
    """The other CLS_LOSS takes `ignore_index=-1`; both branches must agree that -1 means ignore."""
    logits = torch.tensor([[2.0, -1.0], [0.3, 0.9], [-1.0, 1.0], [0.0, 0.0]])
    labels = torch.tensor([0, 1, -1, 1], dtype=torch.long)
    loss, _ = _loss(_head('CrossEntropy'), logits, labels)
    expected = F.cross_entropy(logits[[0, 1, 3]], labels[[0, 1, 3]], reduction='mean')
    assert torch.allclose(loss, expected)
