"""AdaBN with BEV statistics restricted to the target sensor's cone (pcdet/utils/adabn_masked.py, T05 item 45).

Pins: an all-True mask reproduces adabn_utils.reestimate_bn exactly (the identity setting, bug-prevention rule 8);
a partial mask gives each BatchNorm2d the statistics of the masked cells only while other BatchNorm layers are
re-estimated as usual; the cone follows DatasetTemplate.extract_fov_gt's convention with rows = y and columns = x;
the model is restored (no patched forward left, momentum, counters, train/eval flag, weights).
"""
import numpy as np
import pytest
import torch

from pcdet.utils import adabn_masked, adabn_utils

PC_RANGE = [-75.2, -40.0, -2.0, 75.2, 40.0, 4.0]  # H != W, so a transposed layout cannot pass


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(3, momentum=0.01)
        self.lin = torch.nn.Linear(3, 4)
        self.bn1 = torch.nn.BatchNorm1d(4, momentum=0.01)

    def forward(self, batch):
        y = torch.relu(self.bn2(self.conv(batch['x'])))
        return self.bn1(self.lin(y.mean((2, 3))))


class FirstBN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn2 = torch.nn.BatchNorm2d(2)

    def forward(self, batch):
        return self.bn2(batch['x'])


def _model(cls=Toy):
    torch.manual_seed(0)
    return cls().eval()


def _loader(n=12, h=8, w=10, shift=3.0):
    g = torch.Generator().manual_seed(1)
    return [{'x': torch.randn(5, 2, h, w, generator=g) * 2.0 + shift} for _ in range(n)]


def _all_true(h, w, device):
    return torch.ones(h, w, dtype=torch.bool, device=device)


def test_all_true_mask_reproduces_plain_adabn():
    a, b = _model(), _model()
    adabn_utils.reestimate_bn(a, _loader(), to_gpu=False)
    adabn_masked.reestimate_bn_bev_masked(b, _loader(), _all_true, to_gpu=False)
    for (na, ma), (nb, mb) in zip(adabn_utils.bn_layers(a), adabn_utils.bn_layers(b)):
        assert na == nb
        assert torch.allclose(ma.running_mean, mb.running_mean, atol=1e-5), na
        assert torch.allclose(ma.running_var, mb.running_var, atol=1e-5), na
    x = _loader(1)[0]
    assert torch.allclose(a(x), b(x), atol=1e-5)


def test_partial_mask_takes_the_masked_cells_only():
    m = _model(FirstBN)
    batch = _loader(1, h=6, w=6)[0]
    mask = torch.zeros(6, 6, dtype=torch.bool)
    mask[:, :2] = True
    batch['x'][:, :, :, 2:] += 50.0  # outside the mask: would move the statistics if it were read
    adabn_masked.reestimate_bn_bev_masked(m, [batch], lambda h, w, d: mask.to(d), to_gpu=False)
    inside = batch['x'][:, :, :, :2].transpose(0, 1).reshape(2, -1)
    assert torch.allclose(m.bn2.running_mean, inside.mean(1), atol=1e-5)
    assert torch.allclose(m.bn2.running_var, inside.var(1, unbiased=True), atol=1e-4)


def test_other_batchnorm_layers_are_reestimated_as_usual():
    m = _model()
    mask = torch.zeros(8, 10, dtype=torch.bool)
    mask[2:6, 3:9] = True
    adabn_masked.reestimate_bn_bev_masked(m, _loader(), lambda h, w, d: mask.to(d), to_gpu=False)
    # reference by hand: each batch normalised by its masked BN2d statistics on the way through, then
    # BN1d's cumulative average of batch means and unbiased batch variances (torch's momentum=None)
    ref = _model()
    means2, vars2, means1, vars1 = [], [], [], []
    with torch.no_grad():
        for batch in _loader():
            x = ref.conv(batch['x'])
            xm = x.transpose(0, 1)[:, :, mask].reshape(3, -1)
            mu, var = xm.mean(1), xm.var(1, unbiased=False)
            means2.append(mu)
            vars2.append(xm.var(1, unbiased=True))
            y = (x - mu[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + ref.bn2.eps)
            y = torch.relu(y * ref.bn2.weight[None, :, None, None] + ref.bn2.bias[None, :, None, None])
            z = ref.lin(y.mean((2, 3)))
            means1.append(z.mean(0))
            vars1.append(z.var(0, unbiased=True))
    assert torch.allclose(m.bn2.running_mean, torch.stack(means2).mean(0), atol=1e-5)
    assert torch.allclose(m.bn2.running_var, torch.stack(vars2).mean(0), atol=1e-4)
    assert torch.allclose(m.bn1.running_mean, torch.stack(means1).mean(0), atol=1e-5)
    assert torch.allclose(m.bn1.running_var, torch.stack(vars1).mean(0), atol=1e-4)
    assert m.bn1.momentum == 0.01 and m.bn2.momentum == 0.01


def test_model_is_restored_and_weights_do_not_move():
    m = _model()
    w = {k: v.clone() for k, v in m.state_dict().items() if 'running' not in k and 'num_batches' not in k}
    tracked = {n: l.num_batches_tracked.clone() for n, l in adabn_utils.bn_layers(m)}
    _, diag = adabn_masked.reestimate_bn_bev_masked(m, _loader(), _all_true, to_gpu=False)
    assert not m.training
    for n, l in adabn_utils.bn_layers(m):
        assert 'forward' not in l.__dict__, n
        assert l.momentum == 0.01
        assert torch.equal(l.num_batches_tracked, tracked[n])
    for k, v in m.state_dict().items():
        if k in w:
            assert torch.equal(v, w[k]), k
    assert set(diag) == {'bn2'} and diag['bn2']['content_in_mask'] == pytest.approx(1.0)


def test_empty_mask_raises():
    m = _model()
    with pytest.raises(ValueError):
        adabn_masked.reestimate_bn_bev_masked(m, _loader(1), lambda h, w, d: torch.zeros(h, w, dtype=torch.bool),
                                              to_gpu=False)
    assert 'forward' not in m.bn2.__dict__  # restored even when the pass fails


@pytest.mark.parametrize('heading', [0.0, 90.0])
def test_cone_matches_extract_fov_gt_on_cell_centres(heading):
    from pcdet.datasets.dataset import DatasetTemplate
    h, w = 40, 75
    mask = adabn_masked.bev_cone_mask(h, w, PC_RANGE, 60.0, heading).numpy()
    xs = PC_RANGE[0] + (np.arange(w) + 0.5) * (PC_RANGE[3] - PC_RANGE[0]) / w
    ys = PC_RANGE[1] + (np.arange(h) + 0.5) * (PC_RANGE[4] - PC_RANGE[1]) / h
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    boxes = np.zeros((h * w, 7))
    boxes[:, 0], boxes[:, 1] = xx.ravel(), yy.ravel()
    boxes[:, 3:6] = 1.0
    ref = DatasetTemplate.extract_fov_gt(boxes, 60.0, heading).reshape(h, w)
    assert (mask == ref).all()
    assert 0.0 < mask.mean() < 0.5


def test_cone_layout_rows_are_y_columns_are_x():
    h, w = 40, 75
    mask = adabn_masked.bev_cone_mask(h, w, PC_RANGE, 60.0, 0.0)
    row_y0 = h // 2  # y ~ 0
    col = lambda x: int((x - PC_RANGE[0]) / (PC_RANGE[3] - PC_RANGE[0]) * w)
    assert mask[row_y0, col(20.0)]          # straight ahead
    assert not mask[row_y0, col(-20.0)]     # behind
    assert not mask[0, col(10.0)]           # far to the side (y = -39 at x = 10)
