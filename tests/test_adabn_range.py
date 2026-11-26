"""Range-stratified AdaBN (pcdet/utils/adabn_range.py; experiments_md 20261003_05 §5 item 57, design fixed 2026-11-27 00:10 JST).
CPU only. One knot must be exactly global AdaBN (adabn_utils.reestimate_bn); with knots, each range must come out with the
normalized offset and spread it had on the source."""
import copy

import numpy as np
import pytest
import torch

from pcdet.utils import adabn_range as ar
from pcdet.utils import adabn_utils


def test_knot_weights_piecewise_linear_and_constant_beyond_the_ends():
    w = ar.knot_weights(torch.tensor([0.0, 5.0, 10.0, 15.0, 30.0, 55.0, 90.0], dtype=torch.float64), ar.KNOTS_ITEM57)
    assert torch.allclose(w.sum(1), torch.ones(7, dtype=torch.float64))
    assert w[0].tolist() == [1, 0, 0, 0, 0] and w[1].tolist() == [1, 0, 0, 0, 0]
    assert torch.allclose(w[2], torch.tensor([0.5, 0.5, 0, 0, 0], dtype=torch.float64))
    assert w[3].tolist() == [0, 1, 0, 0, 0]
    assert torch.allclose(w[4], torch.tensor([0, 0, 0.5, 0.5, 0], dtype=torch.float64))
    assert w[5].tolist() == [0, 0, 0, 0, 1] and w[6].tolist() == [0, 0, 0, 0, 1]
    assert ar.knot_weights(torch.tensor([3.0, 70.0]), (0.0,)).tolist() == [[1.0], [1.0]]


def test_weighted_moments():
    x = torch.randn(500, 3, dtype=torch.float64) * 2 + 1
    s, m, v = ar.weighted_moments(x, torch.ones(500, 1))
    assert s.item() == 500 and torch.allclose(m[0], x.mean(0)) and torch.allclose(v[0], x.var(0, unbiased=False))
    w = torch.zeros(500, 2); w[:200, 0] = 1; w[200:, 1] = 1
    s, m, v = ar.weighted_moments(x, w)
    assert torch.allclose(m[1], x[200:].mean(0)) and torch.allclose(v[0], x[:200].var(0, unbiased=False))


class Dense(torch.nn.Module):
    """A BEV-like two-layer model: conv -> BN2d -> ReLU -> conv -> BN2d, input batch['x'] (B, 2, 8, 8)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.c1, self.b1 = torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.BatchNorm2d(4)
        self.c2, self.b2 = torch.nn.Conv2d(4, 3, 1), torch.nn.BatchNorm2d(3)
        for b in (self.b1, self.b2):
            b.weight.data.uniform_(0.5, 1.5); b.bias.data.uniform_(-0.5, 0.5)
            b.running_mean.uniform_(-1, 1); b.running_var.uniform_(0.5, 2)

    def forward(self, batch):
        return self.b2(self.c2(torch.relu(self.b1(self.c1(batch['x'])))))


def batches(seed, n=6, shift=0.0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(4, 2, 8, 8, generator=g) * scale + shift
        r = torch.linspace(0, 1, 8)
        x = x + 3.0 * r[None, None, None, :]                    # range-dependent content along x
        out.append({'x': x})
    return out


PC = (0.0, -4.0, -1.0, 8.0, 4.0, 1.0)


def test_one_knot_is_global_adabn_exactly():
    a, b = Dense(), Dense()
    data = batches(1, shift=0.7, scale=1.8)
    adabn_utils.reestimate_bn(a, data, to_gpu=False)
    rs = ar.RangeStratifiedBN(b, PC)
    rs.estimate(data, (0.0,), to_gpu=False)
    for (na, ma), (nb, mb) in zip(adabn_utils.bn_layers(a), adabn_utils.bn_layers(b)):
        assert torch.allclose(ma.running_mean, mb.running_mean, atol=1e-5), na
        assert torch.allclose(ma.running_var, mb.running_var, atol=1e-5), na
    rs.apply()
    a.eval()
    x = {'x': torch.randn(2, 2, 8, 8)}
    assert torch.allclose(a(x), b(x), atol=1e-5)
    rs.restore()
    fresh = Dense()
    assert torch.equal(b.b1.running_mean, fresh.b1.running_mean)           # saved statistics back
    b.eval(); fresh.eval()
    assert torch.allclose(b(x), fresh(x), atol=1e-6)                       # and no override left


class Sparse(torch.nn.Module):
    """One BatchNorm1d on (N, 1) features; positions supplied by the test."""

    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(1, eps=1e-3)
        self.bn.weight.data.fill_(1.0); self.bn.bias.data.fill_(0.0)

    def forward(self, batch):
        return self.bn(batch['x'])


def two_range_batches(seed, near, far, n=8, k=400):
    g = torch.Generator().manual_seed(seed)
    out, rho = [], torch.cat([torch.full((k,), 5.0), torch.full((k,), 55.0)]).double()
    for _ in range(n):
        out.append({'x': torch.cat([torch.randn(k, 1, generator=g) * near[1] + near[0],
                                    torch.randn(k, 1, generator=g) * far[1] + far[0]])})
    return out, rho


def test_knots_give_each_range_its_source_offset_and_spread():
    src, rho = two_range_batches(0, near=(0.0, 1.0), far=(2.0, 0.5))
    tgt, _ = two_range_batches(1, near=(1.0, 2.0), far=(5.0, 3.0))
    m = Sparse()
    rs = ar.RangeStratifiedBN(m, (-80, -80, -2, 80, 80, 4))
    rs.pos.voxels = lambda n: rho                                          # positions: first half at 5 m, second at 55 m
    rs.collect(src, (5.0, 55.0), to_gpu=False)
    rs.estimate(tgt, (5.0, 55.0), to_gpu=False)
    rs.apply()
    xs = torch.cat([b['x'] for b in src]); mu, sd = xs.mean(), xs.std(unbiased=False)
    want_near = ((0.0 - mu) / sd, 1.0 / sd)                                 # the source's own normalized moments per range
    want_far = ((2.0 - mu) / sd, 0.5 / sd)
    with torch.no_grad():                                                   # on the clouds the statistics came from
        y = torch.cat([m(b) for b in tgt]).reshape(8, 800)
    near, far = y[:, :400].reshape(-1), y[:, 400:].reshape(-1)
    assert abs(near.mean() - want_near[0]) < 0.02 and abs(near.std() / want_near[1] - 1) < 0.02
    assert abs(far.mean() - want_far[0]) < 0.02 and abs(far.std() / want_far[1] - 1) < 0.02
    # global AdaBN (one knot) cannot: it standardizes the pooled target, so the far range keeps the target's own offset
    rs.restore()
    rs.estimate(tgt, (0.0,), to_gpu=False)
    rs.apply()
    with torch.no_grad():
        y1 = torch.cat([m(b) for b in tgt]).reshape(8, 800)
    assert abs(y1[:, 400:].std() / want_far[1] - 1) > 0.5


def test_stage0_mismatch_is_zero_when_target_equals_source():
    src, rho = two_range_batches(0, near=(0.0, 1.0), far=(2.0, 0.5))
    m = Sparse()
    rs = ar.RangeStratifiedBN(m, (-80, -80, -2, 80, 80, 4))
    rs.pos.voxels = lambda n: rho
    s = rs.collect(src, (5.0, 55.0), to_gpu=False)
    d = rs.estimate(src, (0.0,), diag_knots=(5.0, 55.0), to_gpu=False)
    mm = ar.mismatch(s['bn'], d['bn'])
    assert max(mm['d_mean']) < 1e-6 and max(mm['d_logsd']) < 1e-6 and abs(mm['share_src'][0] - 0.5) < 1e-9


def test_voxel_positions_from_the_last_sparse_output():
    p = ar._Pos((-75.2, -75.2, -2, 75.2, 75.2, 4))

    class Out:
        indices = torch.tensor([[0, 0, 752, 752], [0, 0, 752, 1052]], dtype=torch.int32)
        spatial_shape = [41, 1504, 1504]
    p.hook(None, None, Out())
    r = p.voxels(2)
    assert abs(r[0] - np.hypot(0.05, 0.05)) < 1e-6 and abs(r[1] - np.hypot(30.05, 0.05)) < 1e-6
    with pytest.raises(RuntimeError):
        p.voxels(3)


class TwoSparse(torch.nn.Module):
    """Two BatchNorm1d layers on (N, 1) features: `first.bn` then `second.bn`; positions supplied by the test."""

    def __init__(self):
        super().__init__()
        self.first, self.second = Sparse(), Sparse()

    def forward(self, batch):
        return self.second({'x': self.first(batch)})


def test_layer_filter_gives_knots_to_matching_layers_only():
    src, rho = two_range_batches(0, near=(0.0, 1.0), far=(2.0, 0.5))
    tgt, _ = two_range_batches(1, near=(1.0, 2.0), far=(5.0, 3.0))
    m = TwoSparse()
    rs = ar.RangeStratifiedBN(m, (-80, -80, -2, 80, 80, 4))
    rs.pos.voxels = lambda n: rho
    rs.collect(src, (5.0, 55.0), to_gpu=False)
    rs.estimate(tgt, (5.0, 55.0), to_gpu=False, layers=r'^first\.')
    st = rs.tgt[1]
    assert st['first.bn']['knots'] == (5.0, 55.0) and st['second.bn']['knots'] == (0.0,)
    assert torch.equal(st['second.bn']['o'], torch.zeros(1, 1, dtype=torch.float64))
    rs.apply()
    with torch.no_grad():
        y = torch.cat([m(b) for b in tgt]).reshape(8, 800)
    assert abs(y.mean()) < 0.02 and abs(y.std() - 1) < 0.02             # the last layer is global AdaBN: standardized
    # a filter matching nothing is global AdaBN everywhere
    rs.restore()
    a = TwoSparse()
    ra = ar.RangeStratifiedBN(a, (-80, -80, -2, 80, 80, 4)); ra.pos.voxels = lambda n: rho
    ra.estimate(tgt, (0.0,), to_gpu=False)
    rs.estimate(tgt, (5.0, 55.0), to_gpu=False, layers=r'^nothing$')
    for n in ('first.bn', 'second.bn'):
        assert torch.allclose(rs.tgt[1][n]['mean'], ra.tgt[1][n]['mean']) and torch.allclose(rs.tgt[1][n]['var'], ra.tgt[1][n]['var'])
