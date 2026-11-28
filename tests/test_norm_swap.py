"""LayerNorm / GroupNorm in place of BatchNorm (pcdet/models/model_utils/norm_swap.py, `MODEL.NORM_SWAP`; experiments_md
20261003_05 §5 item 68, design fixed 2026-11-28 16:42 JST). CPU only; run from the repo root."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'tools'))
from pcdet.models.model_utils import norm_swap as ns  # noqa: E402


def _rows(batch_idx):
    r = ns._Rows()
    b = torch.as_tensor(batch_idx, dtype=torch.int32)
    r.last = torch.stack([b, torch.zeros_like(b), torch.zeros_like(b), torch.arange(len(b), dtype=torch.int32)], 1)
    return r


def _sparse_gn_reference(x, b, G, eps, w, bias):
    y = torch.empty_like(x)
    C = x.shape[1]
    for f in b.unique():
        m = b == f
        for g in range(G):
            sl = slice(g * C // G, (g + 1) * C // G)
            v = x[m][:, sl]
            mu, var = v.mean(), v.var(unbiased=False)
            y[m.nonzero()[:, 0][:, None], torch.arange(C)[sl][None]] = (v - mu) / torch.sqrt(var + eps)
    return y * w + bias


def test_sparse_group_norm_equals_a_per_frame_loop_and_frames_never_mix():
    torch.manual_seed(0)
    b = torch.tensor([0] * 7 + [1] * 13 + [2] * 3)
    x = torch.randn(len(b), 16, dtype=torch.float64) * 3 + torch.arange(len(b), dtype=torch.float64)[:, None]
    m = ns.SparseGroupNorm(4, 16, _rows(b), eps=1e-3).double()
    with torch.no_grad():
        m.weight.copy_(torch.rand(16) + 0.5)
        m.bias.copy_(torch.randn(16))
    y = m(x)
    torch.testing.assert_close(y, _sparse_gn_reference(x, b, 4, 1e-3, m.weight, m.bias))
    x2 = x.clone()
    x2[b == 1] *= 10.0
    y2 = m(x2)
    assert torch.equal(y2[b == 0], y[b == 0]) and torch.equal(y2[b == 2], y[b == 2])


def test_sparse_group_norm_gradients():
    torch.manual_seed(1)
    b = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    m = ns.SparseGroupNorm(2, 4, _rows(b), eps=1e-3).double()
    x = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(m, (x,))


def test_sparse_group_norm_needs_a_recorded_convolution():
    m = ns.SparseGroupNorm(2, 4, ns._Rows())
    with pytest.raises(RuntimeError):
        m(torch.randn(5, 4))
    m = ns.SparseGroupNorm(2, 4, _rows([0, 0, 1]))
    with pytest.raises(RuntimeError):
        m(torch.randn(5, 4))                                       # 5 rows after a 3-row convolution output


def test_element_layer_norm():
    torch.manual_seed(2)
    m = ns.ElementLayerNorm(8, eps=1e-3)
    with torch.no_grad():
        m.weight.copy_(torch.rand(8) + 0.5)
        m.bias.copy_(torch.randn(8))
    x = torch.randn(30, 8) * 4 + 2
    torch.testing.assert_close(m(x), F.layer_norm(x, (8,), m.weight, m.bias, 1e-3))
    x4 = torch.randn(2, 8, 5, 6) * 4 + 2
    ref = ((x4 - x4.mean(1, keepdim=True)) / torch.sqrt(x4.var(1, unbiased=False, keepdim=True) + 1e-3)
           * m.weight[None, :, None, None] + m.bias[None, :, None, None])
    torch.testing.assert_close(m(x4), ref, atol=1e-5, rtol=1e-5)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.BatchNorm1d(16, eps=1e-3, momentum=0.01))
        self.b = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 1), torch.nn.BatchNorm2d(64, eps=1e-3, momentum=0.01))
        self.c = torch.nn.BatchNorm2d(256, eps=1e-3)


@pytest.mark.parametrize('kind', ['layer', 'group'])
def test_installer_replaces_every_batchnorm_and_keeps_no_running_buffer(kind):
    if kind == 'group':
        pytest.importorskip('spconv.pytorch')
    t = Toy()
    n = ns.install_norm_swap(t, {'TYPE': kind, 'MAX_GROUPS': 32})
    assert n == 3
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in t.modules())
    keys = list(t.state_dict())
    assert not any(k.endswith(('running_mean', 'running_var', 'num_batches_tracked')) for k in keys)
    if kind == 'layer':
        assert all(isinstance(m, ns.ElementLayerNorm) for m in (t.a[1], t.b[1], t.c))
    else:
        assert isinstance(t.a[1], ns.SparseGroupNorm) and t.a[1].num_groups == 16          # min(32, 16)
        assert isinstance(t.b[1], torch.nn.GroupNorm) and t.b[1].num_groups == 32 and t.b[1].eps == 1e-3
        assert t.c.num_groups == 32
    assert len(ns.norm_swap_layers(t)) == 3
    with pytest.raises(AssertionError):
        ns.install_norm_swap(Toy(), {'TYPE': 'instance'})


@pytest.mark.parametrize('v', ['ln', 'gn'])
def test_item68_configs_change_only_the_normalization(v):
    from pcdet.config import cfg_from_yaml_file
    from easydict import EasyDict
    import os
    os.chdir(REPO / 'tools')
    plain = lambda c: {k: v for k, v in c.items()}                                          # noqa: E731
    for base, mine in (('cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml', 'nuscenes-%s' % v),
                       ('cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti.yaml', 'nuscenes2kitti-%s' % v)):
        b, m = cfg_from_yaml_file(base, EasyDict()), cfg_from_yaml_file(
            'cfgs/da-ieee-access-analysis/item68/centerpoint-item68-%s.yaml' % mine, EasyDict())
        assert 'NORM_SWAP' not in b.MODEL
        swap = m.MODEL.pop('NORM_SWAP')
        assert swap == ({'TYPE': 'layer'} if v == 'ln' else {'TYPE': 'group', 'MAX_GROUPS': 32})
        m.pop('_BASE_CONFIG_', None); b.pop('_BASE_CONFIG_', None)
        assert plain(m) == plain(b)
