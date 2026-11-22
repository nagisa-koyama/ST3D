"""DTS adaptation pieces on CenterPoint (pcdet/utils/dts_utils.py; experiments_md 20261003_05 §5 item 51, row D). CPU only.

boxes_iou3d_gpu and nms_gpu are CUDA ops; here they are replaced by CPU stand-ins (axis-aligned BEV IoU, keep-all NMS), so
these tests check the port's arithmetic and plumbing. The real ops run in the GPU smoke.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from easydict import EasyDict

from pcdet.utils import dts_utils as d

CFG = EasyDict({'CONSISTENCY_LOSS': {'inter_graph_loss_weight': 0.3, 'GLR_loss_weight': 0.4, 'object_loss_weight': 0.05},
                'INTER_GRAPH_THRESHOLD': 0.1})


def aa_iou(a, b):
    """Axis-aligned BEV IoU (x, y, l, w), a CPU stand-in for boxes_iou3d_gpu on unrotated boxes."""
    ax1, ax2 = a[:, 0] - a[:, 3] / 2, a[:, 0] + a[:, 3] / 2
    ay1, ay2 = a[:, 1] - a[:, 4] / 2, a[:, 1] + a[:, 4] / 2
    bx1, bx2 = b[:, 0] - b[:, 3] / 2, b[:, 0] + b[:, 3] / 2
    by1, by2 = b[:, 1] - b[:, 4] / 2, b[:, 1] + b[:, 4] / 2
    iw = (torch.min(ax2[:, None], bx2[None]) - torch.max(ax1[:, None], bx1[None])).clamp(min=0)
    ih = (torch.min(ay2[:, None], by2[None]) - torch.max(ay1[:, None], by1[None])).clamp(min=0)
    inter = iw * ih
    return inter / (a[:, 3:4] * a[:, 4:5] + (b[:, 3] * b[:, 4])[None] - inter)


@pytest.fixture(autouse=True)
def cpu_ops(monkeypatch):
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    monkeypatch.setattr(iou3d_nms_utils, 'boxes_iou3d_gpu', aa_iou)


def box(x, y, l=4.0, w=2.0):
    return [x, y, 0.0, l, w, 1.5, 0.0]


def test_beam_keep_mask_one_draw_per_beam_and_keeps_points_without_a_beam():
    ring = np.array([0, 0, 1, 2, -1, 3])
    calls = []
    keep = d.beam_keep_mask(ring, 4, 0.75, rand=lambda n: calls.append(n) or np.array([0.1, 0.9, 0.5, 0.8]))
    assert calls == [4] and keep.tolist() == [True, True, False, True, True, False]
    with pytest.raises(AssertionError):
        d.beam_keep_mask(np.array([4]), 4, 0.75)
    rng = np.random.default_rng(0)
    frac = np.mean([d.beam_keep_mask(np.arange(64), 64, 0.75, rand=rng.random).mean() for _ in range(400)])
    assert abs(frac - 0.75) < 0.01


def reference_pool(features, boxes, pc_range, voxel, stride, g=7):
    """The released form (SECONDHead roi_grid_pool): expand the map once per box."""
    c, h, w = features.shape
    sx, sy = voxel[0] * stride, voxel[1] * stride
    x1 = (boxes[:, 0] - boxes[:, 3] / 2 - pc_range[0]) / sx
    x2 = (boxes[:, 0] + boxes[:, 3] / 2 - pc_range[0]) / sx
    y1 = (boxes[:, 1] - boxes[:, 4] / 2 - pc_range[1]) / sy
    y2 = (boxes[:, 1] + boxes[:, 4] / 2 - pc_range[1]) / sy
    ca, sa = torch.cos(boxes[:, 6]), torch.sin(boxes[:, 6])
    theta = torch.stack(((x2 - x1) / (w - 1) * ca, (x2 - x1) / (w - 1) * (-sa), (x1 + x2 - w + 1) / (w - 1),
                         (y2 - y1) / (h - 1) * sa, (y2 - y1) / (h - 1) * ca, (y1 + y2 - h + 1) / (h - 1)), 1).view(-1, 2, 3)
    grid = F.affine_grid(theta, torch.Size((len(boxes), c, g, g)), align_corners=True)
    return F.grid_sample(features.unsqueeze(0).expand(len(boxes), c, h, w), grid, align_corners=True).reshape(len(boxes), -1)


def test_bev_box_pool_equals_the_released_form_and_passes_gradient():
    torch.manual_seed(0)
    feat = torch.randn(5, 40, 50, requires_grad=True)
    boxes = torch.tensor([box(3.0, 4.0), [8.0, 2.0, 0.0, 3.0, 1.5, 1.0, 0.7], box(1.2, 7.5, 2.0, 1.0)])
    args = ([0.0, 0.0, -2.0, 20.0, 16.0, 4.0], [0.1, 0.1, 0.15], 4)
    got = d.bev_box_pool(feat, boxes, *args)
    assert got.shape == (3, 5 * 49) and torch.allclose(got, reference_pool(feat, boxes, *args), atol=1e-6)
    got.sum().backward()
    assert feat.grad is not None and feat.grad.abs().sum() > 0
    assert d.bev_box_pool(feat, boxes[:0], *args).shape == (0, 5 * 49)


def test_bev_box_pool_reads_the_box_footprint():
    feat = torch.zeros(1, 40, 50)
    feat[0, 8:13, 5:10] = 3.0  # cells y 8-12, x 5-9 = metres x 2.0-3.6, y 3.2-4.8 at 0.4 m cells
    out = d.bev_box_pool(feat, torch.tensor([box(2.8, 4.0, 1.2, 1.2)]), [0.0, 0.0, -2, 20, 16, 4], [0.1, 0.1, 0.15], 4)
    assert torch.allclose(out, torch.full_like(out, 3.0))


def nodes(boxes, logits, feats):
    return (torch.tensor(boxes, dtype=torch.float32), torch.tensor(logits, dtype=torch.float32), feats)


def test_oga_no_pairs_is_zero_and_identical_views_have_no_graph_terms():
    f = torch.randn(2, 6, requires_grad=True)
    far = nodes([box(50, 50), box(60, 60)], [1.0, 2.0], f)
    near = nodes([box(0, 0), box(10, 0)], [1.0, 2.0], f.detach())
    loss, tb = d.oga_loss([far], [near], CFG)
    assert loss == 0.0 and tb['oga_pairs'] == 0
    same = nodes([box(0, 0), box(10, 0)], [1.0, 2.0], f)
    loss, tb = d.oga_loss([same], [near], CFG)
    assert tb['oga_pairs'] == 2 and tb['oga_inter'] == pytest.approx(0.0) and tb['oga_GLR'] == pytest.approx(0.0, abs=1e-6)
    fn = F.normalize(f.detach(), dim=1) * torch.tensor([[1.0], [2.0]])
    assert tb['oga_object'] == pytest.approx(float(torch.exp(-(fn * fn).sum(1)).mean()), rel=1e-5)
    assert float(loss) == pytest.approx(0.05 * tb['oga_object'], rel=1e-5)


def test_oga_matches_by_iou_many_to_one_and_averages_over_the_batch():
    torch.manual_seed(1)
    fs, ft = torch.randn(3, 4, requires_grad=True), torch.randn(2, 4)
    s = nodes([box(0, 0), box(0.3, 0), box(20, 0)], [0.5, -1.0, 2.0], fs)   # two students on teacher 0, one on 1
    t = nodes([box(0.1, 0), box(20.2, 0)], [1.5, 0.7], ft)
    loss, tb = d.oga_loss([s, nodes([], [], torch.zeros(0, 4))], [t, t], CFG)  # second sample empty: skipped, still B = 2
    assert tb['oga_pairs'] == 3
    a, b = torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1])
    fa = F.normalize(fs.detach()[a], dim=1) * torch.tensor([0.5, -1.0, 2.0])[:, None]
    fb = F.normalize(ft[b], dim=1) * torch.tensor([1.5, 1.5, 0.7])[:, None]
    assert tb['oga_object'] == pytest.approx(float(torch.exp(-(fa * fb).sum(1)).mean()) / 2, rel=1e-5)
    loss.backward()
    assert fs.grad.abs().sum() > 0


def test_ema_update_parameters_only():
    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    ema = d.make_ema_teacher(m)
    assert not ema.training and all(not p.requires_grad for p in ema.parameters())
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
        m[1].running_mean.add_(5.0)
    before = [p.clone() for p in ema.parameters()]
    d.ema_update(ema, m, 0.9)
    for e, b0, p in zip(ema.parameters(), before, m.parameters()):
        assert torch.allclose(e, 0.9 * b0 + 0.1 * p)
    assert torch.allclose(ema[1].running_mean, torch.zeros(2))  # buffers are not averaged


def test_decode_nodes_returns_raw_logits_of_the_peaks(monkeypatch):
    from pcdet.models.model_utils import model_nms_utils
    monkeypatch.setattr(model_nms_utils, 'class_agnostic_nms',
                        lambda box_scores, box_preds, nms_config, score_thresh=None: (torch.arange(len(box_scores)), box_scores))
    head = EasyDict(model_cfg={'POST_PROCESSING': {'POST_CENTER_LIMIT_RANGE': [-50, -50, -5, 50, 50, 5]}},
                    point_cloud_range=[0.0, 0.0, -2.0, 20.0, 16.0, 4.0], voxel_size=[0.1, 0.1, 0.15], feature_map_stride=4)
    hm = torch.full((1, 3, 40, 50), -6.0)
    hm[0, 1, 10, 20] = 2.0
    pd = {'hm': hm, 'center': torch.zeros(1, 2, 40, 50), 'center_z': torch.zeros(1, 1, 40, 50),
          'dim': torch.zeros(1, 3, 40, 50), 'rot': torch.cat([torch.ones(1, 1, 40, 50), torch.zeros(1, 1, 40, 50)], 1)}
    (boxes, logits), = d.decode_nodes(head, [pd], topk=5, nms_thresh=0.8, post_max=512, hm_is_prob=False)
    assert boxes.shape == (5, 7) and logits[0] == pytest.approx(2.0, abs=1e-4)
    assert boxes[0, 0] == pytest.approx(20 * 0.4) and boxes[0, 1] == pytest.approx(10 * 0.4) and boxes[0, 3] == pytest.approx(1.0)
    pd['hm'] = torch.clamp(hm.sigmoid(), 1e-4, 1 - 1e-4)
    (_, logits2), = d.decode_nodes(head, [pd], topk=5, nms_thresh=0.8, post_max=512, hm_is_prob=True)
    assert torch.allclose(logits2, logits, atol=1e-4)


def test_oga_reads_the_student_map_from_the_head_not_the_callers_batch(monkeypatch):
    """Smoke 34914: under DDP the model's forward runs on a COPY of the batch dict, so the student's
    spatial_features_2d is never in the caller's dict. The pre-fix oga_from_batches read it from there (KeyError)."""
    from pcdet.models import load_data_to_gpu  # noqa: F401  (the function oga_from_batches imports)
    feat = torch.randn(1, 4, 40, 50, requires_grad=True)
    nodes_ = [(torch.tensor([box(3.0, 4.0)]), torch.tensor([1.0]))]
    monkeypatch.setattr(d, 'decode_nodes', lambda *a, **k: nodes_)
    head = EasyDict(point_cloud_range=[0.0, 0.0, -2.0, 20.0, 16.0, 4.0], voxel_size=[0.1, 0.1, 0.15], feature_map_stride=4,
                    forward_ret_dict={'pred_dicts': [], 'spatial_features_2d': feat})
    model = EasyDict(dense_head=head)
    ema = EasyDict(module_list=[], dense_head=EasyDict(forward_ret_dict={'pred_dicts': []}))
    student_batch = {'batch_size': 1}                      # what the caller holds after a DDP forward
    teacher_batch = {'batch_size': 1, 'spatial_features_2d': feat.detach() + 0.1}
    cfg = EasyDict(CFG, STUDENT_NODES={'TOPK': 9, 'NMS_THRESH': 0.8, 'POST_MAX': 5},
                   TEACHER_NODES={'TOPK': 9, 'NMS_THRESH': 0.7, 'POST_MAX': 5})
    loss, tb = d.oga_from_batches(model, ema, student_batch, teacher_batch, cfg)
    assert tb['oga_pairs'] == 1 and 'spatial_features_2d' not in head.forward_ret_dict  # popped: not kept to the next step
    loss.backward()
    assert feat.grad is not None and feat.grad.abs().sum() > 0
    with pytest.raises(AssertionError):
        d.oga_from_batches(model, ema, student_batch, teacher_batch, cfg)  # no map kept: refused, not silent


def test_center_head_keeps_the_bev_map_only_when_the_batch_asks():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_center_head_ignore_regions import FMAP, _head
    for ask in (True, False):
        head = _head()
        head.forward_ret_dict = {'spatial_features_2d': 'stale'}
        head.shared_conv = torch.nn.Identity()
        head.domain_discriminator = None
        head.heads_list = []
        head.training = True
        m = torch.zeros(1, 1, *FMAP)
        try:
            head.forward({'spatial_features_2d': m, 'DTS_KEEP_BEV': ask} if ask else {'spatial_features_2d': m})
        except (IndexError, KeyError):
            pass  # the stub has no heads; the map is kept before they are needed
        assert (head.forward_ret_dict.get('spatial_features_2d') is m) == ask


def test_split_for_dts_keeps_ignore_labels_identical_in_both_views():
    """Sentinels through the DTS split (bug-prevention rule 1): D0's cuts give Pedestrian / Cyclist an ignore band, so the
    target's pseudo-labels carry NEGATIVE class labels (self_training_utils) and IGNORE_CLASS_LABEL -99. Both views must
    keep every box and its label exactly; neither view may drop, sign-flip or re-index one."""
    from pcdet.datasets.dataset import DatasetTemplate
    from pcdet.utils import common_utils

    class _Identity:
        def forward(self, data_dict=None, **kw):
            return data_dict if data_dict is not None else kw

    ds = DatasetTemplate.__new__(DatasetTemplate)
    ds.dts_views_cfg = EasyDict({'NUM_BEAMS': 64, 'KEEP_PROB': 0.75, 'TEACHER_KEEP_PROB': 0.75})
    ds.point_feature_encoder = _Identity()
    ds.data_processor = _Identity()
    rng = np.random.RandomState(0)
    pts = np.concatenate([rng.rand(2000, 4).astype(np.float32), rng.randint(-1, 64, (2000, 1)).astype(np.float32)], 1)
    labels = np.array([1, -1, 2, -2, 3, -3, common_utils.IGNORE_CLASS_LABEL], dtype=np.int32)
    boxes = np.concatenate([rng.rand(7, 7).astype(np.float32), labels.reshape(-1, 1).astype(np.float32)], 1)
    dd = {'points': pts, 'gt_boxes': boxes.copy(), 'gt_classes': labels.copy(), 'gt_scores': np.linspace(0.2, 0.9, 7),
          'gt_names': np.array(['Car', 'Car', 'Pedestrian', 'Pedestrian', 'Cyclist', 'Cyclist', 'Misc'])}
    np.random.seed(0)
    s, t = ds._split_for_dts(dd)
    for view in (s, t):
        assert np.array_equal(view['gt_boxes'], boxes)
        assert np.array_equal(view['gt_boxes'][:, 7].astype(np.int32), labels)
        assert np.array_equal(view['gt_classes'], labels)
        assert view['points'].shape[1] == 4  # the ring column is stripped
    view_rows = lambda v: set(map(tuple, v['points'].tolist()))
    assert view_rows(s) != view_rows(t)  # two independent beam masks
    assert view_rows(s) <= set(map(tuple, pts[:, :4].tolist()))
    s['gt_boxes'][0, 7] = 99  # the views own their arrays: one cannot change the other or the caller's dict
    assert t['gt_boxes'][0, 7] == 1 and dd['gt_boxes'][0, 7] == 1
