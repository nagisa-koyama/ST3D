"""Item 60r: RS-BN knots looked up at s * range (pcdet/models/model_utils/range_bn.set_range_scale, adabn_range._Pos.scale;
experiments_md 20261003_05 §5 item 60r, design fixed 2026-11-28 15:32 JST). Rule 8 of the bug-prevention checklist: s = 1 is
the default and leaves everything bitwise unchanged. CPU only; run from the repo root.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'tools'))
from pcdet.models.model_utils import range_bn as rb  # noqa: E402
from pcdet.utils.adabn_range import _Pos, knot_weights, weighted_moments  # noqa: E402

PC = [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]
KNOTS = (5.0, 15.0, 25.0, 35.0, 55.0)


def _voxel_pos(pos, n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    nz, ny, nx = 4, 160, 176
    idx = torch.stack([torch.zeros(n, dtype=torch.long), torch.randint(0, nz, (n,), generator=g),
                       torch.randint(0, ny, (n,), generator=g), torch.randint(0, nx, (n,), generator=g)], 1)
    pos.last = (idx, [nz, ny, nx])
    x = PC[0] + (idx[:, 3].double() + 0.5) * (PC[3] - PC[0]) / nx
    y = PC[1] + (idx[:, 2].double() + 0.5) * (PC[4] - PC[1]) / ny
    return torch.hypot(x, y)


def _bev_ranges(h, w):
    xs = PC[0] + (torch.arange(w, dtype=torch.float64) + 0.5) * (PC[3] - PC[0]) / w
    ys = PC[1] + (torch.arange(h, dtype=torch.float64) + 0.5) * (PC[4] - PC[1]) / h
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.hypot(xx, yy).reshape(-1)


def test_scale_one_is_the_default_and_bitwise_unchanged():
    pos = _Pos(PC)
    assert pos.scale == 1.0
    r = _voxel_pos(pos)
    assert torch.equal(pos.voxels(300), r)
    assert torch.equal(pos.bev(20, 22, 2, 'cpu'), _bev_ranges(20, 22).repeat(2))


def test_scale_multiplies_voxel_and_bev_ranges_and_caches_do_not_leak():
    pos = _Pos(PC)
    r = _voxel_pos(pos)
    b1 = pos.bev(20, 22, 1, 'cpu').clone()
    pos.scale = 0.301
    assert torch.equal(pos.voxels(300), r * 0.301)
    assert torch.equal(pos.bev(20, 22, 1, 'cpu'), _bev_ranges(20, 22) * 0.301)
    pos.scale = 1.0
    assert torch.equal(pos.bev(20, 22, 1, 'cpu'), b1)


class Net(torch.nn.Module):
    """One RangeBN on a BEV map and one on voxels; forward(batch) as reestimate_range_bn calls it."""

    def __init__(self, C=3):
        super().__init__()
        self.pos = _Pos(PC)
        self.bev_bn = rb.RangeBN(torch.nn.BatchNorm2d(C, eps=1e-3, momentum=0.01), self.pos, KNOTS)
        self.vox_bn = rb.RangeBN(torch.nn.BatchNorm1d(C, eps=1e-3, momentum=0.01), self.pos, KNOTS)

    def forward(self, batch):
        _voxel_pos(self.pos, n=batch['v'].shape[0], seed=int(batch['seed']))
        return self.bev_bn(batch['x']), self.vox_bn(batch['v'])


def test_set_range_scale_moves_the_knot_weights_of_every_layer():
    net = Net()
    x = torch.randn(2, 3, 20, 22)
    w1 = net.bev_bn.weights(x).clone()
    assert rb.set_range_scale(net, 0.301) == 0.301
    assert net.bev_bn._pos[0].scale == net.vox_bn._pos[0].scale == 0.301
    torch.testing.assert_close(net.bev_bn.weights(x), knot_weights(_bev_ranges(20, 22).repeat(2) * 0.301, KNOTS).float())
    r = _voxel_pos(net.pos, n=50)
    torch.testing.assert_close(net.vox_bn.weights(torch.randn(50, 3)), knot_weights(r * 0.301, KNOTS).float())
    rb.set_range_scale(net, 1.0)
    assert torch.equal(net.bev_bn.weights(x), w1)              # the weight cache is keyed by the scale
    with pytest.raises(ValueError):
        rb.set_range_scale(torch.nn.Linear(2, 2), 0.3)
    with pytest.raises(AssertionError):
        rb.set_range_scale(net, 0.0)


def test_evaluation_output_at_scale_one_is_unchanged():
    torch.manual_seed(0)
    net = Net().eval()
    for m in (net.bev_bn, net.vox_bn):
        m.running_mean.copy_(torch.randn_like(m.running_mean))
        m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
    batch = {'x': torch.randn(2, 3, 20, 22), 'v': torch.randn(300, 3), 'seed': 3}
    y0 = net(batch)
    rb.set_range_scale(net, 1.0)
    y1 = net(batch)
    assert torch.equal(y0[0], y1[0]) and torch.equal(y0[1], y1[1])
    rb.set_range_scale(net, 0.301)
    assert not torch.allclose(net(batch)[0], y0[0])


def test_reestimation_uses_the_scaled_weights():
    torch.manual_seed(1)
    net = Net()
    net.bev_bn.running_mean.copy_(torch.randn_like(net.bev_bn.running_mean))
    src_mean = net.bev_bn.running_mean.clone()
    rb.set_range_scale(net, 0.301)
    batch = {'x': torch.randn(2, 3, 20, 22) * 2 + 1, 'v': torch.randn(300, 3), 'seed': 5}
    never = rb.reestimate_range_bn(net, [batch], to_gpu=False)
    xf = batch['x'].permute(0, 2, 3, 1).reshape(-1, 3)
    w = knot_weights(_bev_ranges(20, 22).repeat(2) * 0.301, KNOTS)
    s, mean, var = weighted_moments(xf, w)
    seen = s >= net.bev_bn.thin
    assert seen[:2].all() and not seen[3:].any()              # 0-70.4 m maps to 0-21.2 m: knots 35 / 55 never used
    torch.testing.assert_close(net.bev_bn.running_mean[seen].double(), mean[seen], atol=1e-5, rtol=1e-4)
    unb = var * (s / (s - 1.0))[:, None]
    torch.testing.assert_close(net.bev_bn.running_var[seen].double(), unb[seen], atol=1e-5, rtol=1e-4)
    assert torch.equal(net.bev_bn.running_mean[~seen], src_mean[~seen])   # unseen knots keep their source statistics
    assert never['bev_bn'] == int((~seen).sum())
