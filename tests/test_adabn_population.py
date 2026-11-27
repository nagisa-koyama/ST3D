"""Population AdaBN (pcdet/utils/adabn_population.py; experiments_md 20261003_05 §5 item 62, design fixed 2026-11-27 12:53 JST).
Identity: with every element a member it equals adabn_utils.reestimate_bn. CPU only; run from the repo root."""
import copy
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pcdet.utils import adabn_population as ap, adabn_utils  # noqa: E402

PC = [0, -40, -3, 70.4, 40, 1]


def toy():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4, eps=1e-3), torch.nn.ReLU(),
                            torch.nn.Conv2d(4, 4, 1), torch.nn.BatchNorm2d(4, eps=1e-3))
    for b in (m[1], m[4]):
        with torch.no_grad():
            b.weight.uniform_(0.5, 1.5); b.bias.normal_()
    return m


def batches():
    g = torch.Generator().manual_seed(1)
    return [torch.randn(2, 3, 8, 11, generator=g) * (1 + i) + i for i in range(3)]


def test_all_members_equal_global_adabn():
    a, b = toy(), toy()
    adabn_utils.reestimate_bn(a, batches(), to_gpu=False)
    _, d = ap.reestimate_bn_population(b, batches(), ap.member_all, PC, to_gpu=False)
    for (na, ma), (nb, mb) in zip(a.named_modules(), b.named_modules()):
        if isinstance(ma, torch.nn.BatchNorm2d):
            assert torch.allclose(ma.running_mean, mb.running_mean, atol=1e-5)
            assert torch.allclose(ma.running_var, mb.running_var, atol=1e-4)
            assert mb.momentum == ma.momentum and 'forward' not in mb.__dict__
    assert all(v['counted'] == 3 and v['skipped'] == 0 and abs(v['share'] - 1.0) < 1e-6 for v in d['layers'].values())
    x = torch.randn(2, 3, 8, 11)
    a.eval(); b.eval()
    assert torch.allclose(a(x), b(x), atol=1e-4)


def test_footprint_of_a_rotated_box():
    box = torch.tensor([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, math.pi / 2]])      # long side along y after the rotation
    x = torch.tensor([10.0, 11.5, 10.9, 9.2, 10.0])
    y = torch.tensor([1.8, 0.0, 0.0, -1.9, 2.2])
    assert ap.in_footprints(x, y, box).tolist() == [True, False, True, True, False]
    assert not ap.in_footprints(x, y, torch.zeros(0, 7)).any()


def test_members_far_and_per_frame_boxes():
    bi = torch.tensor([0, 0, 1, 1])
    x = torch.tensor([5.0, 30.0, 10.0, 10.0])
    y = torch.zeros(4)
    assert ap.member_far(20.0)('bev', bi, x, y, None).tolist() == [False, True, False, False]
    ctx = {'boxes': [torch.zeros(0, 7), torch.tensor([[10.0, 0, 0, 2, 2, 2, 0]])]}
    assert ap.member_boxes('bev', bi, x, y, ctx).tolist() == [False, False, True, True]


def test_a_layer_without_members_keeps_its_statistics():
    m = toy()
    before = [(b.running_mean.clone(), b.running_var.clone()) for b in (m[1], m[4])]
    _, d = ap.reestimate_bn_population(m, batches(), ap.member_far(1e9), PC, to_gpu=False)
    for (mu, var), b in zip(before, (m[1], m[4])):
        assert torch.equal(b.running_mean, mu) and torch.equal(b.running_var, var)
    assert sorted(d['never']) == ['1', '4'] and all(v['skipped'] == 3 for v in d['layers'].values())


def test_gt_boxes_of_drops_padding():
    g = torch.zeros(2, 3, 8)
    g[0, 0] = torch.tensor([1, 2, 0, 4, 2, 1.5, 0, 1])
    g[1, :2] = torch.tensor([1, 2, 0, 4, 2, 1.5, 0, 2])
    assert [b.shape[0] for b in ap.gt_boxes_of({'gt_boxes': g})] == [1, 2]
