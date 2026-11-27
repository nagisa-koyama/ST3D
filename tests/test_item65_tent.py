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


# ---------------------------------------------------------------- item 65′ (`--variant prior`, design 2026-11-27 15:47 JST)
def test_prior_topk_is_per_class():
    z = torch.full((1, 3, 4, 5), -10.0)
    z[0, 0, :, :] = 5.0                                            # class 0 holds every high cell
    top = m.topk_cells(z, k=3)
    assert top.shape == (1, 3, 3)
    assert torch.all(top[0, 0] == 5.0) and torch.all(top[0, 1:] == -10.0)   # a joint top-k would give class 0 all three
    assert m.topk_cells(z, k=500).shape == (1, 3, 20)


def test_prior_kl_zero_at_the_prior_and_rising_towards_collapse():
    pi = torch.tensor([0.05, 0.01, 0.02])
    assert abs(float(m.bern_kl(pi, pi.clone()))) < 1e-7
    vals = [float(m.bern_kl(pi, pi * f)) for f in (1.0, 0.5, 0.1, 1e-3)]
    assert all(b > a for a, b in zip(vals, vals[1:])) and vals[-1] > 0.1
    assert float(m.bern_kl(pi, torch.zeros(3))) > 0.1               # p -> 0 is finite (clamped) but penalized


def test_prior_loss_on_a_hand_built_heatmap():
    z = torch.full((2, 3, 4, 5), -3.0)
    z[:, 0, 0, 0] = 0.0                                             # one cell at p = 0.5 per frame in class 0
    pi = torch.tensor([0.2, 0.1, 0.1])
    loss, h, kl, pbar = m.prior_loss(z, pi, k=2)
    p3 = torch.sigmoid(torch.tensor(-3.0))
    assert torch.allclose(pbar, torch.tensor([(0.5 + float(p3)) / 2, float(p3), float(p3)]), atol=1e-6)
    e3 = float(m.bin_entropy(torch.tensor(-3.0)))
    assert abs(float(h) - (2 * (math.log(2) + e3) + 4 * 2 * e3) / 12) < 1e-6
    assert abs(float(kl) - float(m.bern_kl(pi, pbar))) < 1e-7 and abs(float(loss) - float(h) - float(kl)) < 1e-6


def test_prior_loss_gradients_reach_only_bn_affine_and_resist_collapse():
    net = toy()
    params = m.tent_params(net)
    m.set_modes(net)
    x = torch.randn(2, 2, 5, 7)
    with torch.no_grad():
        pi = m.class_marginal(net(x), k=10)
    before = {n: p.clone() for n, p in net.named_parameters()}
    loss, _, kl, _ = m.prior_loss(net(x), pi, k=10)
    assert float(kl) < 1e-6                                         # at the start the prior holds exactly
    loss.backward()
    for n, p in net.named_parameters():
        assert (p.grad is not None) == n.startswith(('1.', '4.')), n
    # pushing every logit down raises the loss through KL (the collapse 65 reached is penalized)
    with torch.no_grad():
        down = net(x) - 6.0
    assert float(m.prior_loss(down, pi, k=10)[2]) > 0.5
    assert all(torch.equal(p, before[n]) for n, p in net.named_parameters())


def test_prior_variant_paths_and_rows():
    assert str(m.ckpt_path(666, 1, 'prior')).endswith('item65p/s666/ckpt/checkpoint_epoch_1.pth')
    assert m.result_dir(667, 10, 'prior').parts[-4:] == ('i65p_e10_s667', 'eval', 'epoch_10', 'i65p_e10_s667')
    assert m.row_name(1, 'prior') == 'P_TENTP_E1' and m.tag(666, 10, 'prior') == 'i65p_e10_s666'
    assert m.PI_BATCHES == 200 and (m.GUARD_AT, m.GUARD_MIN) == (50, 0.5)
    assert str(m.ckpt_path(667, 10)).endswith('item65/s667/ckpt/checkpoint_epoch_10.pth')     # 65 unchanged
