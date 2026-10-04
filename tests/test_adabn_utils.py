"""AdaBN statistics re-estimation (pcdet/utils/adabn_utils.py, experiments_md/20261004_01).

Pins what the teacher path relies on: only BN buffers change (no weight), the statistics become the
data's, MIX blends with the saved ones, momentum and the train/eval flag are restored, and a BN layer is
re-estimated even though the rest of the model stays in eval mode.
"""
import torch

from pcdet.utils import adabn_utils


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)
        self.bn = torch.nn.BatchNorm1d(3, momentum=0.01)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, batch):
        return self.drop(self.bn(self.lin(batch['x'])))


def _model():
    torch.manual_seed(0)
    m = Toy()
    with torch.no_grad():
        m.lin.weight.copy_(torch.eye(3))
        m.lin.bias.zero_()
    return m


def _loader(mean, n=40):
    g = torch.Generator().manual_seed(1)
    return [{'x': torch.randn(16, 3, generator=g) * 2.0 + mean} for _ in range(n)]


def test_statistics_become_the_datas_and_weights_do_not_move():
    m = _model().eval()  # a frozen teacher is in eval mode
    w = m.lin.weight.clone()
    adabn_utils.reestimate_bn(m, _loader(5.0), to_gpu=False)
    assert torch.allclose(m.bn.running_mean, torch.full((3,), 5.0), atol=0.3)
    assert torch.allclose(m.bn.running_var, torch.full((3,), 4.0), atol=1.0)
    assert torch.equal(m.lin.weight, w)
    assert m.bn.momentum == 0.01 and not m.training and not m.bn.training


def test_mix_blends_with_the_saved_statistics():
    m = _model()  # saved: mean 0, var 1
    adabn_utils.reestimate_bn(m, _loader(4.0), mix=0.5, to_gpu=False)
    assert torch.allclose(m.bn.running_mean, torch.full((3,), 2.0), atol=0.2)


def test_training_flag_is_restored():
    m = _model().train()
    adabn_utils.reestimate_bn(m, _loader(1.0, n=2), to_gpu=False)
    assert m.training and m.bn.training


def test_gap_is_zero_for_unchanged_statistics_and_reports_a_shift():
    m = _model()
    saved = {n: (b.running_mean.clone(), b.running_var.clone()) for n, b in adabn_utils.bn_layers(m)}
    assert adabn_utils.bn_gap(m, saved)['bn']['mean_shift'] == 0.0
    adabn_utils.reestimate_bn(m, _loader(3.0), to_gpu=False)
    g = adabn_utils.summarise_gap(adabn_utils.bn_gap(m, saved))
    assert g['bn']['mean_shift'] > 2.5


def _ds_model():
    from pcdet.models.model_utils.dsnorm import DSNorm
    m = _model()
    with torch.no_grad():
        m.bn.running_mean.fill_(1.0)
    return DSNorm.convert_dsnorm(m).eval()


def test_dsnorm_domains_are_separate_copies_after_conversion():
    m = _ds_model()
    assert m.bn.running_mean_source.data_ptr() != m.bn.running_mean_target.data_ptr()
    assert torch.equal(m.bn.running_mean_source, m.bn.running_mean_target)


def test_per_domain_bn_only_the_target_statistics_are_reestimated():
    from pcdet.models.model_utils.dsnorm import set_ds_source
    m = _ds_model()
    m.apply(set_ds_source)
    src_mean = m.bn.running_mean_source.clone()
    tracked = m.bn.num_batches_tracked.clone()
    adabn_utils.reestimate_bn(m, _loader(6.0), to_gpu=False)
    assert torch.equal(m.bn.running_mean_source, src_mean)          # source branch untouched
    assert torch.allclose(m.bn.running_mean_target, torch.full((3,), 6.0), atol=0.3)
    assert m.bn.domain_label == 0                                    # domain restored
    assert torch.equal(m.bn.num_batches_tracked, tracked)
    gap = adabn_utils.bn_gap(m, {'bn': (src_mean, torch.ones(3))})   # reports the TARGET set
    assert gap['bn']['mean_shift'] > 4.0
