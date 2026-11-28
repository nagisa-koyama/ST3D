"""RangeBN with gamma / beta per knot (`MODEL.RANGE_BN.AFFINE_PER_KNOT`; experiments_md 20261003_05 §5 item 71, design fixed
2026-11-28 21:31 JST). The identity setting (one knot) must be BatchNorm exactly; each knot's gamma / beta must get the
gradient of its own elements; OpenPCDet's optimizer must hold the per-knot parameters (its layer groups take LEAF modules
only, so parameters registered on RangeBN itself would never train); the default path keeps its state keys. CPU only; run
from the repo root.
"""
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'tools'))
from pcdet.models.model_utils import range_bn as rb  # noqa: E402

TOL = dict(atol=1e-5, rtol=1e-4)


class Pos:
    def __init__(self, rho):
        self.rho = torch.as_tensor(rho, dtype=torch.float64)

    def voxels(self, n):
        assert n == self.rho.numel()
        return self.rho

    def bev(self, h, w, b, device):
        assert b * h * w == self.rho.numel()
        return self.rho


def _bn(dim, c=4):
    m = torch.nn.BatchNorm1d(c, eps=1e-3, momentum=0.01) if dim == 1 else torch.nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        m.weight.copy_(torch.rand(c, generator=g) + 0.5)
        m.bias.copy_(torch.randn(c, generator=g))
    return m


@pytest.mark.parametrize('dim', [1, 2])
def test_one_knot_with_affine_per_knot_is_batchnorm_exactly(dim):
    torch.manual_seed(0)
    C, n = 4, 200
    ref, twin = _bn(dim, C), _bn(dim, C)
    shape = (n, C) if dim == 1 else (2, C, 10, 10)
    m = rb.RangeBN(twin, Pos(torch.rand(n) * 70), knots=(25.0,), affine_per_knot=True)
    assert twin.weight is None and twin.bias is None
    assert torch.allclose(m.affine.weight[0], ref.weight) and torch.allclose(m.affine.bias[0], ref.bias)
    for _ in range(3):
        x = torch.randn(*shape) * 3 + 1
        x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
        y1, y2 = ref(x1), m(x2)
        assert torch.allclose(y1, y2, **TOL)
        g = torch.randn_like(y1)
        y1.backward(g)
        y2.backward(g)
        assert torch.allclose(x1.grad, x2.grad, **TOL)
        assert torch.allclose(ref.weight.grad, m.affine.weight.grad[0], **TOL)
        assert torch.allclose(ref.bias.grad, m.affine.bias.grad[0], **TOL)
        ref.weight.grad = ref.bias.grad = m.affine.weight.grad = m.affine.bias.grad = None
    ref.eval(); m.eval()
    x = torch.randn(*shape)
    assert torch.allclose(ref(x), m(x), **TOL)


def test_each_knot_gets_the_gradient_of_its_own_elements():
    torch.manual_seed(1)
    rho = torch.cat([torch.full((100,), 5.0), torch.full((100,), 55.0)])   # all weight on knot 0, then on knot 2
    m = rb.RangeBN(_bn(1), Pos(rho), knots=(5.0, 25.0, 55.0), thin=1, affine_per_knot=True)
    x = torch.randn(200, 4)
    y = m(x)
    g = torch.zeros_like(y)
    g[:100] = 1.0                                                           # loss from the near elements only
    y.backward(g)
    assert m.affine.bias.grad[0].abs().min() > 0                            # d/d beta_0 = sum of the near upstream grads
    assert torch.all(m.affine.bias.grad[1:] == 0) and torch.all(m.affine.weight.grad[1:] == 0)
    with torch.no_grad():                                                   # a far-only change moves far outputs only
        m.affine.bias[2] += 1.0
    m.eval()
    y0 = m(x)
    with torch.no_grad():
        m.affine.bias[2] -= 1.0
    y1 = m(x)
    assert torch.allclose(y0[:100], y1[:100]) and torch.allclose(y0[100:] - y1[100:], torch.ones(100, 4))


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(4, 4)
        self.norm = torch.nn.BatchNorm1d(4)
        self.head = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))


def _wrap(model, apk):
    pos = Pos(torch.rand(8) * 70)
    for parent, leaf in ((model, 'norm'), (model.head, '1')):
        setattr(parent, leaf, rb.RangeBN(getattr(parent, leaf), pos, knots=(5.0, 25.0, 55.0), affine_per_knot=apk))
    return model


@pytest.mark.parametrize('apk', [False, True])
def test_openpcdet_optimizer_holds_every_trainable_parameter(apk):
    from train_utils.optimization import build_optimizer
    model = _wrap(Toy(), apk)
    opt = build_optimizer(model, EasyDict(OPTIMIZER='adam_onecycle', LR=0.003, WEIGHT_DECAY=0.01, MOMENTUM=0.9))
    n, missing = rb.optimizer_coverage(model, opt)
    assert missing == [] and n == sum(1 for p in model.parameters() if p.requires_grad)
    names = {k for k, _ in model.named_parameters()}
    if apk:
        assert {'norm.affine.weight', 'norm.affine.bias', 'head.1.affine.weight', 'head.1.affine.bias'} <= names
        assert 'norm.bn.weight' not in names
    else:
        assert 'norm.bn.weight' in names and not any('affine' in k for k in names)


def test_coverage_catches_a_parameter_on_a_non_leaf_module():
    """The trap the leaf KnotAffine avoids: a parameter registered on RangeBN itself is missed by the layer groups."""
    from train_utils.optimization import build_optimizer
    model = _wrap(Toy(), True)
    model.norm.register_parameter('stray', torch.nn.Parameter(torch.ones(3)))
    opt = build_optimizer(model, EasyDict(OPTIMIZER='adam_onecycle', LR=0.003, WEIGHT_DECAY=0.01, MOMENTUM=0.9))
    assert rb.optimizer_coverage(model, opt)[1] == ['norm.stray']


def test_default_state_keys_unchanged_and_affine_keys_added():
    a = rb.RangeBN(_bn(1), Pos(torch.zeros(1)), knots=(5.0, 15.0)).state_dict().keys()
    assert set(a) == {'bn.weight', 'bn.bias', 'running_mean', 'running_var', 'num_batches_tracked', 'knot_batches',
                      'thin_steps'}
    b = rb.RangeBN(_bn(1), Pos(torch.zeros(1)), knots=(5.0, 15.0), affine_per_knot=True).state_dict().keys()
    assert set(b) == {'affine.weight', 'affine.bias', 'running_mean', 'running_var', 'num_batches_tracked', 'knot_batches',
                      'thin_steps'}


def test_reestimation_leaves_the_per_knot_affine_alone():
    torch.manual_seed(2)
    rho = torch.cat([torch.full((60,), 5.0), torch.full((60,), 55.0)])
    model = torch.nn.Sequential()
    model.add_module('n', rb.RangeBN(_bn(1), Pos(rho), knots=(5.0, 55.0), thin=1, affine_per_knot=True))
    with torch.no_grad():
        model.n.affine.weight[1] *= 2.0
    w0 = model.n.affine.weight.clone()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, batch):
            return self.m(batch['x'])
    loader = [{'x': torch.randn(120, 4) * 2 + 3} for _ in range(3)]
    rb.reestimate_range_bn(Wrap(model), loader, to_gpu=False)
    assert torch.equal(model.n.affine.weight, w0)
    assert model.n.running_mean[0].sub(3).abs().max() < 1.0                 # re-estimated toward the new data's mean


@pytest.mark.parametrize('name', ['st', 'stg'])
def test_item71_configs_share_the_controls_model_and_set_the_variant(name):
    import json
    import os
    from pcdet.config import cfg_from_yaml_file
    d = REPO / 'tools/cfgs/da-ieee-access-analysis/item71'
    if not d.exists():
        pytest.skip('no item 71 configs')
    os.chdir(REPO / 'tools')
    tr = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/item71/centerpoint-item71-nuscenes-%s.yaml' % name, EasyDict())
    ev = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/item71/centerpoint-item71-nuscenes2kitti-%s.yaml' % name, EasyDict())
    ctrl = cfg_from_yaml_file('cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml', EasyDict())
    J = lambda x: json.dumps(x, sort_keys=True, default=str)                 # noqa: E731
    assert J(tr.MODEL) == J(ev.MODEL)
    rbc = tr.MODEL.pop('RANGE_BN')
    assert J(tr.MODEL) == J(ctrl.MODEL) and J(tr.OPTIMIZATION) == J(ctrl.OPTIMIZATION)
    assert list(rbc.KNOTS) == [5.0, 15.0, 25.0, 35.0, 55.0] and rbc.THIN == 32
    assert bool(rbc.get('AFFINE_PER_KNOT', False)) == (name == 'stg')
    assert J(tr.DATA_CONFIG) == J(ctrl.DATA_CONFIG) and ev.DATA_CONFIG_TAR.DATASET == 'KittiDataset'
