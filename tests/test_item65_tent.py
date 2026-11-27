"""Item 65's TENT-style adaptation (tools/analysis/item65_tent.py; experiments_md 20261003_05 §5 item 65, design fixed
2026-11-27 13:02 JST): only BN gamma / beta receive gradients, BN in train mode and the rest in eval mode, the top-k entropy.
CPU only; run from the repo root."""
import math
import sys
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item65_tent as m  # noqa: E402


def toy():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv2d(2, 3, 1), torch.nn.BatchNorm2d(3), torch.nn.Dropout(0.5),
                               torch.nn.Conv2d(3, 3, 1, bias=False), torch.nn.BatchNorm2d(3))


def test_only_bn_affine_gets_gradients_and_the_modes():
    net = toy()
    params = m.tent_params(net)
    assert len(params) == 4 and all(p.requires_grad for p in params)
    m.set_modes(net)
    assert not net.training and not net[2].training and net[1].training and net[4].training
    before = {n: p.clone() for n, p in net.named_parameters()}
    opt = torch.optim.Adam(params, lr=0.01)
    loss = m.topk_entropy(net(torch.randn(2, 2, 5, 7)), k=10)
    loss.backward()
    opt.step()
    for n, p in net.named_parameters():
        if n.startswith(('1.', '4.')):
            assert p.grad is not None and not torch.equal(p, before[n]), n
        else:
            assert p.grad is None and torch.equal(p, before[n]), n


def test_topk_entropy_values():
    z = torch.full((2, 3, 4, 5), -10.0)
    z[0, 0, 0, :2] = 0.0                                           # two cells at p = 0.5 in frame 0
    h = m.topk_entropy(z, k=2)
    other = (torch.sigmoid(torch.tensor(-10.0)) * torch.nn.functional.softplus(torch.tensor(10.0))
             + (1 - torch.sigmoid(torch.tensor(-10.0))) * torch.nn.functional.softplus(torch.tensor(-10.0)))
    assert abs(float(h) - (math.log(2) + float(other)) / 2) < 1e-5
    assert m.topk_entropy(torch.zeros(1, 1, 2, 2), k=500).item() - math.log(2) < 1e-6   # k larger than the map


def test_paths_and_rows():
    assert str(m.ckpt_path(667, 10)).endswith('item65/s667/ckpt/checkpoint_epoch_10.pth')
    assert m.result_dir(666, 1).parts[-4:] == ('i65_e1_s666', 'eval', 'epoch_1', 'i65_e1_s666')
    assert m.row_name(10) == 'P_TENT_E10' and m.EPOCHS == (1, 10)
