"""OPTIMIZATION.TRAIN_ONLY_BN and MODEL.RANGE_AFFINE_BN (experiments_md 20261003_05 §5 item 58, design fixed 2026-11-27 02:56
JST). CPU only. Frozen weights must stay bitwise equal through training steps, with the optimizer built over the trainable
modules (as train.py does) and also over the whole model."""
import sys
from pathlib import Path

import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from train_utils.optimization import build_optimizer  # noqa: E402

from pcdet.utils import adabn_range as ar  # noqa: E402
from pcdet.utils.bn_only import freeze_all_but_bn  # noqa: E402

OPT = EasyDict(OPTIMIZER='adam_onecycle', LR=0.003, WEIGHT_DECAY=0.01, MOMENTUM=0.9, MOMS=[0.95, 0.85], PCT_START=0.4,
               DIV_FACTOR=10, DECAY_STEP_LIST=[35, 45], LR_DECAY=0.1, LR_CLIP=1e-7, LR_WARMUP=False, WARMUP_EPOCH=1,
               GRAD_NORM_CLIP=10)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.head = torch.nn.Conv2d(4, 1, 1)

    def forward(self, x):
        return self.head(torch.relu(self.bn(self.conv(x))))


def step(model, opt):
    model.train()
    loss = model(torch.randn(2, 2, 8, 8)).pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()


def test_only_bn_trains_and_frozen_weights_stay_bitwise_equal():
    m = Net()
    conv0, head0, bn0 = m.conv.weight.clone(), m.head.weight.clone(), m.bn.weight.clone()
    trainable, n_train, n_frozen = freeze_all_but_bn(m)
    assert n_train == 2 and n_frozen == 4                                   # bn.weight, bn.bias | conv, head weights + biases
    assert [p.requires_grad for p in m.parameters()] == [False, False, True, True, False, False]
    opt = build_optimizer(trainable, OPT)
    for _ in range(3):
        step(m, opt)
    assert torch.equal(m.conv.weight, conv0) and torch.equal(m.head.weight, head0)
    assert not torch.equal(m.bn.weight, bn0)


def test_frozen_weights_stay_fixed_even_with_an_optimizer_over_the_whole_model():
    m = Net()
    conv0 = m.conv.weight.clone()
    freeze_all_but_bn(m)
    opt = build_optimizer(m, OPT)                    # OptimWrapper skips requires_grad False in its decoupled weight decay
    for _ in range(3):
        step(m, opt)
    assert torch.equal(m.conv.weight, conv0)


def test_range_affine_starts_at_bn_and_trains():
    m = Net()
    m.eval()
    x = torch.randn(2, 2, 8, 8)
    with torch.no_grad():
        y0 = m(x)
    pc = (0.0, -4.0, -1.0, 8.0, 4.0, 1.0)
    pos = ar._Pos(pc)
    m.bn = ar.RangeAffineBN(m.bn, pos, (2.0, 6.0))
    with torch.no_grad():
        assert torch.equal(m(x), y0)                                            # a = b = 0: exactly BN
    trainable, n_train, _ = freeze_all_but_bn(m, keep_types=(ar.RangeAffine,))
    assert n_train == 4                                                     # bn weight / bias + a / b
    opt = build_optimizer(trainable, OPT)
    a0 = m.bn.mod.a.clone()
    step(m, opt)
    assert not torch.equal(m.bn.mod.a, a0)                                  # the optimizer found the leaf's parameters
    m.eval()
    assert m.bn.bn.weight.shape == (4,) and set(dict(m.named_parameters())) >= {'bn.bn.weight', 'bn.mod.a', 'bn.mod.b'}


def test_range_affine_on_sparse_features_uses_the_recorded_positions():
    bn = torch.nn.BatchNorm1d(3)
    pos = ar._Pos((-75.2, -75.2, -2, 75.2, 75.2, 4))
    w = ar.RangeAffineBN(bn, pos, (5.0, 55.0))
    with torch.no_grad():
        w.mod.b[1] = torch.tensor([1.0, 2.0, 3.0])                         # only the 55 m knot shifts

    class Out:
        indices = torch.tensor([[0, 0, 752, 752], [0, 0, 752, 1302]], dtype=torch.int32)   # ~0 m and ~55 m
        spatial_shape = [41, 1504, 1504]
    pos.hook(None, None, Out())
    w.eval()
    x = torch.zeros(2, 3)
    y = w(x) - bn(x)
    assert torch.allclose(y[0], torch.zeros(3), atol=1e-6) and torch.allclose(y[1], torch.tensor([1.0, 2.0, 3.0]), atol=1e-3)
