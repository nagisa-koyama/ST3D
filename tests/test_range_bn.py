"""Range-stratified BatchNorm for training (pcdet/models/model_utils/range_bn.py; experiments_md 20261003_05 §5 item 60, design
fixed 2026-11-27 12:09 JST). Rule 8 of the bug-prevention checklist: the new layer at its identity setting (one knot) must be
BatchNorm exactly - output, gradients, running statistics, evaluation. CPU only; run from the repo root.
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
    """Stands in for adabn_range._Pos: fixed ranges per element."""

    def __init__(self, rho):
        self.rho = torch.as_tensor(rho, dtype=torch.float64)

    def voxels(self, n):
        assert n == self.rho.numel()
        return self.rho

    def bev(self, h, w, b, device):
        assert b * h * w == self.rho.numel()
        return self.rho


def bn1d(c=4):
    return torch.nn.BatchNorm1d(c, eps=1e-3, momentum=0.01)


def init_affine(*bns):
    g = torch.Generator().manual_seed(0)
    w, b = torch.rand(bns[0].num_features, generator=g) + 0.5, torch.randn(bns[0].num_features, generator=g)
    for m in bns:
        with torch.no_grad():
            m.weight.copy_(w)
            m.bias.copy_(b)


@pytest.mark.parametrize('dim', [1, 2])
def test_one_knot_is_batchnorm_exactly(dim):
    torch.manual_seed(0)
    C = 4
    ref = bn1d(C) if dim == 1 else torch.nn.BatchNorm2d(C, eps=1e-3, momentum=0.01)
    twin = bn1d(C) if dim == 1 else torch.nn.BatchNorm2d(C, eps=1e-3, momentum=0.01)
    init_affine(ref, twin)
    shape = (200, C) if dim == 1 else (2, C, 10, 10)
    n = 200
    m = rb.RangeBN(twin, Pos(torch.rand(n) * 70), knots=(25.0,))
    for step in range(3):
        x = torch.randn(*shape) * 3 + 1
        x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
        y1, y2 = ref(x1), m(x2)
        assert torch.allclose(y1, y2, **TOL)
        g = torch.randn_like(y1)
        y1.backward(g)
        y2.backward(g)
        assert torch.allclose(x1.grad, x2.grad, **TOL)
        assert torch.allclose(ref.weight.grad, twin.weight.grad, **TOL)
        assert torch.allclose(ref.bias.grad, twin.bias.grad, **TOL)
        ref.weight.grad = ref.bias.grad = twin.weight.grad = twin.bias.grad = None
        assert torch.allclose(ref.running_mean, m.running_mean[0], **TOL)
        assert torch.allclose(ref.running_var, m.running_var[0], **TOL)
    ref.eval(); m.eval()
    x = torch.randn(*shape)
    assert torch.allclose(ref(x), m(x), **TOL)


def test_the_wrapped_layer_loses_its_own_statistics_and_keeps_its_affine():
    bn = bn1d()
    m = rb.RangeBN(bn, Pos(torch.zeros(1)), knots=(5.0, 15.0))
    assert bn.running_mean is None and bn.running_var is None and not bn.track_running_stats
    assert m.running_mean.shape == (2, 4) and m.bn.weight is bn.weight
    keys = set(m.state_dict())
    assert {'running_mean', 'running_var', 'num_batches_tracked', 'knot_batches', 'thin_steps', 'bn.weight', 'bn.bias'} == keys


def test_each_knot_normalizes_its_own_elements():
    torch.manual_seed(1)
    n = 300
    rho = torch.cat([torch.full((150,), 5.0), torch.full((150,), 15.0)])      # exactly on the knots: one-hot weights
    x = torch.cat([torch.randn(150, 4) * 2 + 3, torch.randn(150, 4) * 0.5 - 1])
    bn = bn1d()
    with torch.no_grad():
        bn.weight.fill_(1.0); bn.bias.zero_()
    m = rb.RangeBN(bn, Pos(rho), knots=(5.0, 15.0))
    m.momentum = None                                                           # cumulative: running = this batch
    y = m(x)
    for sl in (slice(0, 150), slice(150, 300)):
        assert torch.allclose(y[sl].mean(0), torch.zeros(4), atol=1e-4)
        assert torch.allclose(y[sl].var(0, unbiased=False), torch.ones(4), atol=1e-2)
        k = 0 if sl.start == 0 else 1
        assert torch.allclose(m.running_mean[k], x[sl].mean(0), atol=1e-5)
        assert torch.allclose(m.running_var[k], x[sl].var(0, unbiased=True), atol=1e-4)
    m.eval()                                          # eval: the running (unbiased) variance, 150 / 149 of the batch's
    assert torch.allclose(m(x), y, atol=2e-2) and not torch.allclose(m(x), y, atol=1e-5)
    assert n == x.shape[0]


def test_a_thin_knot_takes_pooled_moments_and_keeps_its_statistics():
    torch.manual_seed(2)
    rho = torch.cat([torch.full((100,), 5.0), torch.full((10,), 55.0)])         # 10 < thin 32 at the far knot
    x = torch.randn(110, 4)
    m = rb.RangeBN(bn1d(), Pos(rho), knots=(5.0, 55.0), thin=32)
    before = m.running_mean[1].clone()
    y = m(x)
    pooled = (x - x.mean(0)) / torch.sqrt(x.var(0, unbiased=False) + 1e-3) * m.bn.weight + m.bn.bias
    assert torch.allclose(y[100:], pooled[100:], **TOL)
    assert torch.equal(m.running_mean[1], before)
    assert m.thin_steps.tolist() == [0, 1] and m.knot_batches.tolist() == [1, 0]


class Toy(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, batch):
        return self.layer(batch)


def test_reestimation_is_a_per_knot_cumulative_average_and_restores_the_layer():
    torch.manual_seed(3)
    rho = torch.cat([torch.full((60,), 5.0), torch.full((60,), 15.0)])
    m = rb.RangeBN(bn1d(), Pos(rho), knots=(5.0, 15.0, 55.0))                  # the 55 m knot is never seen
    with torch.no_grad():
        m.running_mean.fill_(7.0); m.running_var.fill_(2.0)
    m.knot_batches.fill_(9); m.thin_steps.fill_(4)
    model = Toy(m).eval()
    batches = [torch.randn(120, 4) + i for i in range(3)]
    never = rb.reestimate_range_bn(model, batches, to_gpu=False)
    assert never == {'layer': 1}
    want0 = torch.stack([b[:60].mean(0) for b in batches]).mean(0)
    want1 = torch.stack([b[60:].mean(0) for b in batches]).mean(0)
    assert torch.allclose(m.running_mean[0], want0, atol=1e-5) and torch.allclose(m.running_mean[1], want1, atol=1e-5)
    assert torch.all(m.running_mean[2] == 7.0) and torch.all(m.running_var[2] == 2.0)          # kept the source statistics
    assert m.momentum == 0.01 and m.knot_batches.tolist() == [9, 9, 9] and m.thin_steps.tolist() == [4, 4, 4]
    assert model.training is False and m.training is False                      # the mode it had before
    model.train()
    rb.reestimate_range_bn(model, batches, to_gpu=False)
    assert model.training and m.training


def test_build_network_wraps_only_when_the_key_is_set(monkeypatch):
    import pcdet.models as models
    import spconv.pytorch as spconv

    def toy(model_cfg, num_class, dataset):
        return torch.nn.Sequential(spconv.SubMConv3d(4, 4, 3, indice_key='a'), torch.nn.BatchNorm1d(4),
                                   torch.nn.Conv2d(4, 4, 1), torch.nn.BatchNorm2d(4))
    monkeypatch.setattr(models, 'build_detector', toy)
    ds = EasyDict(point_cloud_range=[0, -40, -3, 70.4, 40, 1])
    plain = models.build_network(EasyDict(NAME='x'), 3, ds)
    assert not rb.range_bn_layers(plain) and not hasattr(plain, 'n_range_bn')
    assert set(plain.state_dict()) == set(toy(None, 3, None).state_dict())
    wrapped = models.build_network(EasyDict(NAME='x', RANGE_BN=EasyDict(KNOTS=[5.0, 15.0, 25.0, 35.0, 55.0])), 3, ds)
    assert wrapped.n_range_bn == 2 and len(rb.range_bn_layers(wrapped)) == 2
    assert all(m.thin == 32 and len(m.knots) == 5 for _, m in rb.range_bn_layers(wrapped))


def test_the_optimizer_sees_the_same_parameters():
    """adam_onecycle collects leaf modules: the wrapped BN stays a leaf, so gamma / beta are optimized as before."""
    sys.path.insert(0, str(REPO / 'tools'))
    from train_utils.optimization import build_optimizer
    cfgo = EasyDict(OPTIMIZER='adam_onecycle', LR=0.003, WEIGHT_DECAY=0.01, MOMENTUM=0.9)
    plain = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    wrapped = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    wrapped[1] = rb.RangeBN(wrapped[1], Pos(torch.zeros(1)), knots=(5.0, 15.0))
    n = lambda opt: sum(p.numel() for g in opt.opt.param_groups for p in g['params'])     # noqa: E731
    assert n(build_optimizer(plain, cfgo)) == n(build_optimizer(wrapped, cfgo)) == 4 * 4 + 4 + 4 + 4


def test_item60_config_is_p_k_plus_range_bn_only():
    import os
    from pcdet.config import cfg_from_yaml_file
    cwd = os.getcwd()
    os.chdir(REPO / 'tools')
    try:
        pk = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml', EasyDict())
        rs = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/item60/centerpoint-item60-pk-rsbn.yaml', EasyDict())
        sp = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/item60/centerpoint-item60-pk-rsbn-on-sprime.yaml', EasyDict())
    finally:
        os.chdir(cwd)
    assert rs.MODEL.RANGE_BN.KNOTS == [5.0, 15.0, 25.0, 35.0, 55.0] and rs.MODEL.RANGE_BN.THIN == 32
    rs_model = dict(rs.MODEL); rs_model.pop('RANGE_BN')
    assert rs_model == dict(pk.MODEL)
    for k in set(pk) | set(rs):
        if k not in ('MODEL', '_BASE_CONFIG_'):
            assert rs.get(k) == pk.get(k), k
    assert pk.OPTIMIZATION.NUM_EPOCHS == 152 and pk.DATA_CONFIG.TRAIN_FACTOR_SWAP.RING_THIN.MODE == 'pattern'
    assert sp.DATA_CONFIG_TAR.EVAL_FACTOR_SWAP.RING_THIN == pk.DATA_CONFIG.TRAIN_FACTOR_SWAP.RING_THIN
    assert sp.MODEL == rs.MODEL


# ---- item 60g (experiments_md 20261003_05 §5 item 60g, design fixed 2026-11-28 10:38 JST): pooled re-estimation ----

def test_pooled_reestimation_with_one_knot_equals_reestimate_bn():
    from pcdet.utils import adabn_utils
    torch.manual_seed(4)
    ref, twin = bn1d(), bn1d()
    init_affine(ref, twin)
    m = rb.RangeBN(twin, Pos(torch.rand(120) * 70), knots=(25.0,))
    batches = [torch.randn(120, 4) * (1 + i) + i for i in range(4)]
    adabn_utils.reestimate_bn(Toy(ref).eval(), batches, to_gpu=False)
    rb.reestimate_range_bn(Toy(m).eval(), batches, to_gpu=False, pooled=True)
    assert torch.allclose(ref.running_mean, m.running_mean[0], **TOL)
    assert torch.allclose(ref.running_var, m.running_var[0], **TOL)
    x = torch.randn(120, 4)
    assert torch.allclose(ref(x), m(x), **TOL)


def test_pooled_reestimation_gives_every_knot_the_pooled_statistics():
    torch.manual_seed(5)
    rho = torch.cat([torch.full((60,), 5.0), torch.full((60,), 15.0), torch.full((60,), 40.0)])
    m = rb.RangeBN(bn1d(), Pos(rho), knots=(5.0, 15.0, 55.0))
    with torch.no_grad():
        m.running_mean.fill_(7.0); m.running_var.fill_(2.0)
    m.knot_batches.fill_(9); m.thin_steps.fill_(4)
    batches = [torch.cat([torch.randn(60, 4) + 3, torch.randn(60, 4) * 2, torch.randn(60, 4) - 4]) + i for i in range(3)]
    never = rb.reestimate_range_bn(Toy(m).eval(), batches, to_gpu=False, pooled=True)
    assert never == {'layer': 0} and m.pooled is False
    want_mean = torch.stack([b.mean(0) for b in batches]).mean(0)
    want_var = torch.stack([b.var(0, unbiased=True) for b in batches]).mean(0)
    for k in range(3):
        assert torch.allclose(m.running_mean[k], want_mean, atol=1e-5)
        assert torch.allclose(m.running_var[k], want_var, atol=1e-4)
    assert rb.knot_spread(m) == (0.0, 0.0)
    assert m.momentum == 0.01 and m.knot_batches.tolist() == [9, 9, 9] and m.thin_steps.tolist() == [4, 4, 4]
    plain = bn1d()                                       # evaluation equals a plain BatchNorm holding the pooled statistics
    with torch.no_grad():
        plain.weight.copy_(m.bn.weight); plain.bias.copy_(m.bn.bias)
        plain.running_mean.copy_(want_mean); plain.running_var.copy_(want_var)
    plain.eval()
    x = torch.randn(180, 4) * 3
    assert torch.allclose(m(x), plain(x), **TOL)


def test_pooled_differs_from_per_knot_reestimation_and_is_off_by_default():
    torch.manual_seed(6)
    rho = torch.cat([torch.full((60,), 5.0), torch.full((60,), 15.0)])
    batches = [torch.cat([torch.randn(60, 4) + 3, torch.randn(60, 4) - 3]) for _ in range(2)]
    a = rb.RangeBN(bn1d(), Pos(rho), knots=(5.0, 15.0))
    b = rb.RangeBN(bn1d(), Pos(rho), knots=(5.0, 15.0))
    assert a.pooled is False and 'pooled' not in a.state_dict()
    rb.reestimate_range_bn(Toy(a).eval(), batches, to_gpu=False)
    rb.reestimate_range_bn(Toy(b).eval(), batches, to_gpu=False, pooled=True)
    assert rb.knot_spread(a)[0] > 5.0 and rb.knot_spread(b) == (0.0, 0.0)
    assert torch.allclose(a.running_mean[0], torch.full((4,), 3.0), atol=0.5)


def test_pooled_thin_batch_updates_no_knot():
    m = rb.RangeBN(bn1d(), Pos(torch.zeros(10)), knots=(5.0, 15.0), thin=32)
    before = m.running_mean.clone()
    m.pooled = True
    m.train()
    x = torch.randn(10, 4)
    y = m(x)
    pooled = (x - x.mean(0)) / torch.sqrt(x.var(0, unbiased=False) + 1e-3) * m.bn.weight + m.bn.bias
    assert torch.allclose(y, pooled, **TOL) and torch.equal(m.running_mean, before)
    assert m.knot_batches.tolist() == [0, 0] and m.thin_steps.tolist() == [1, 1]
