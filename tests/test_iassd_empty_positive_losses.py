"""IA-SSD head losses on a batch with NO positive (experiments_md 20261024_03 §5.5): S2R A2R (job 30375) died at
iteration 24 in `get_contextual_vote_loss` (torch.cat of an empty list) once `drop_empty_gt_boxes` removed the boxes the
16,384-point budget empties in sparse accumulated frames; `get_corner_layer_loss` would have returned NaN (the mean of an
empty tensor) on the same batch. Both now return a graph-connected zero there, and are unchanged whenever a positive
exists (checked against the upstream formulas, copied here)."""
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import _init_path  # noqa: F401,E402

pytest.importorskip('pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils')
from pcdet.models.dense_heads.IASSD_head import IASSD_Head  # noqa: E402
from pcdet.utils import loss_utils  # noqa: E402

CFG = EasyDict(LOSS_CONFIG=EasyDict(LOSS_WEIGHTS={'vote_weight': 1.0, 'corner_weight': 1.0}))


def _head(n=12, labels=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.zeros(n, dtype=torch.long) if labels is None else torch.tensor(labels)
    pos = labels > 0
    ctr_offsets = torch.cat([torch.zeros(n, 1), torch.randn(n, 3, generator=g)], dim=1).requires_grad_(True)  # (batch, dx, dy, dz)
    box_preds = torch.cat([torch.randn(n, 3, generator=g), torch.rand(n, 3, generator=g) + 1, torch.randn(n, 1, generator=g)],
                          dim=1).requires_grad_(True)
    gt = torch.cat([torch.randn(int(pos.sum()), 3, generator=g), torch.rand(int(pos.sum()), 3, generator=g) + 1,
                    torch.randn(int(pos.sum()), 2, generator=g)], dim=1)
    ret = {
        'center_origin_cls_labels': labels, 'center_cls_labels': labels,
        'center_origin_gt_box_of_fg_points': gt, 'center_gt_box_of_fg_points': gt,
        'centers_origin': torch.cat([torch.zeros(n, 1), torch.randn(n, 3, generator=g)], dim=1),
        'ctr_offsets': ctr_offsets, 'point_box_preds': box_preds,
    }
    return types.SimpleNamespace(forward_ret_dict=ret, model_cfg=CFG), ctr_offsets, box_preds


def _vote_upstream(h):
    d = h.forward_ret_dict
    pos_mask = d['center_origin_cls_labels'] > 0
    out = []
    for i in d['center_origin_cls_labels'].unique():
        if i <= 0:
            continue
        m = d['center_origin_cls_labels'] == i
        lab = d['center_origin_gt_box_of_fg_points'][:, 0:3][(pos_mask & m)[pos_mask == 1]]
        pred = (d['centers_origin'] + d['ctr_offsets'])[m][:, 1:4]
        out.append(F.smooth_l1_loss(pred, lab).unsqueeze(-1))
    return torch.cat(out, dim=-1).mean()


def _corner_upstream(h):
    d = h.forward_ret_dict
    pos = d['center_cls_labels'] > 0
    return loss_utils.get_corner_loss_lidar(d['point_box_preds'][pos][:, 0:7], d['center_gt_box_of_fg_points'][:, 0:7]).mean()


@pytest.mark.parametrize('labels', [[0] * 12, [0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0]])
def test_no_positive_gives_a_graph_connected_zero(labels):
    h, ctr, box = _head(labels=labels)
    vote, tb_v = IASSD_Head.get_contextual_vote_loss(h)
    corner, tb_c = IASSD_Head.get_corner_layer_loss(h)
    assert torch.isfinite(vote) and torch.isfinite(corner)
    assert float(vote) == 0.0 and float(corner) == 0.0
    assert tb_v['center_origin_loss_reg'] == 0.0 and tb_c['corner_loss_reg'] == 0.0
    (vote + corner).backward()
    assert ctr.grad is not None and box.grad is not None          # DDP: every parameter still receives a gradient
    assert ctr.grad.abs().sum() == 0 and box.grad.abs().sum() == 0


@pytest.mark.parametrize('labels', [[1, 0, 2, 0, 1, 0, 0, 3, 0, 0, -1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]])
def test_with_positives_the_losses_equal_upstream(labels):
    h, _, _ = _head(labels=labels, seed=1)
    vote, _ = IASSD_Head.get_contextual_vote_loss(h)
    corner, _ = IASSD_Head.get_corner_layer_loss(h)
    assert torch.equal(vote, _vote_upstream(h))
    assert torch.equal(corner, _corner_upstream(h))


def test_upstream_formulas_fail_exactly_where_the_guard_acts():
    h, _, _ = _head(labels=[0] * 12)
    with pytest.raises(RuntimeError, match='non-empty list'):
        _vote_upstream(h)
    assert torch.isnan(_corner_upstream(h))
