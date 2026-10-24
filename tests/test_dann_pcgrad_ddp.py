"""DANN-M + PCGrad under DistributedDataParallel: the torchjd path with the in-head DANN, a GRL at alpha != 1 and BatchNorm
(experiments_md 20261022_06 §10; design T05 §5, fixed 2026-10-24 10:13 JST).

train_one_epoch_st with SELF_TRAIN.USE_TORCHJD under DDP (`ddp_manual_sync`): DDP's sync is switched OFF for the whole
iteration (both forwards), `torchjd.backward([loss_src, dann, st], PCGrad(), parallel_chunk_size=1)` fills `.grad` itself
(the reducer never fires), then sync is switched back on and `commu_utils.all_reduce_grads` averages by hand. This file
runs exactly that sequence on two CPU processes over gloo, float64, with the real dann_utils (GRL at alpha = 37) and a
BatchNorm backbone, and checks:

  * with the manual all-reduce, every parameter (discriminator included) agrees across ranks and equals the mean of the
    per-rank torchjd results computed without DDP;
  * without it (negative control), the ranks disagree - the hazard the all-reduce exists for is real here too;
  * both forwards unsynced with BatchNorm raise nothing (no in-place buffer-broadcast error, cf. job 26404);
  * the discriminator gets a non-zero gradient through PCGrad, and the backbone's domain row is reversed and scaled by
    alpha (the row the aggregator sees is -alpha x the GRL-bypassed one on the backbone).
"""
import multiprocessing as mp
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))
from pcdet.models.model_utils import dann_utils  # noqa: E402
from pcdet.utils import commu_utils  # noqa: E402

torchjd = pytest.importorskip('torchjd')
from torchjd.aggregation import PCGrad  # noqa: E402

WORLD = 2
ALPHA = 37.0
DANN_WEIGHT = 0.1


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.backbone = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.head = nn.Conv2d(8, 2, 1)
        self.domain_discriminator = dann_utils.build_domain_discriminator(
            {'LOSS_WEIGHTS': {'dann_weight': DANN_WEIGHT}}, in_channels=8)

    def forward(self, x, domain_label, target):
        f = self.backbone(x)
        det = ((self.head(f) - target) ** 2).mean()
        dann, _ = dann_utils.domain_adversarial_loss(
            dann_utils.domain_predictions(self.domain_discriminator, f, ALPHA), domain_label, DANN_WEIGHT)
        return det, dann


def _model():
    return _Toy().double()


def _batches(rank):
    g = torch.Generator().manual_seed(100 + rank)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)  # noqa: E731
    return r(3, 3, 6, 6), r(3, 2, 6, 6), r(3, 3, 6, 6), r(3, 2, 6, 6)   # source x / y, target x / pseudo-label y


def _losses(m, rank):
    xs, ys, xt, yt = _batches(rank)
    ds, dns = m(xs, 0, ys)
    dt, dnt = m(xt, 1, yt)
    return [ds, 0.5 * dns + 0.5 * dnt, dt]   # train_st_utils: [loss_src_sum, dann_loss_sum, st_loss_sum]


def _grads(m):
    return {n.replace('module.', '', 1): (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in m.named_parameters()}


def _local(rank):
    m = _model()
    torchjd.backward(_losses(m, rank), PCGrad(), parallel_chunk_size=1)
    return _grads(m)


def _domain_row(rank, grl):
    """The domain row of the Jacobian on the backbone, with the GRL at ALPHA or bypassed."""
    m = _model()
    if not grl:
        m.forward = lambda x, dl, y, m=m: (
            ((m.head(m.backbone(x)) - y) ** 2).mean(),
            dann_utils.domain_adversarial_loss(m.domain_discriminator(m.backbone(x)), dl, DANN_WEIGHT)[0])
    dom = _losses(m, rank)[1]
    return torch.autograd.grad(dom, list(m.backbone.parameters()))


def _worker(rank, init_file, out_dir):
    dist.init_process_group('gloo', init_method='file://%s' % init_file, rank=rank, world_size=WORLD)
    try:
        res = {}
        for mode in ('synced', 'unsynced'):
            ddp = nn.parallel.DistributedDataParallel(_model())
            try:
                ddp.require_backward_grad_sync = False          # whole iteration, as ddp_manual_sync does
                losses = _losses(ddp, rank)                     # both forwards unsynced (BatchNorm inside)
                torchjd.backward(losses, PCGrad(), parallel_chunk_size=1)
                if mode == 'synced':
                    ddp.require_backward_grad_sync = True
                    commu_utils.all_reduce_grads(ddp)
                res[mode] = _grads(ddp)
            except RuntimeError as e:
                res[mode] = str(e)
        res['local'] = _local(rank)
        res['row_grl'] = _domain_row(rank, True)
        res['row_plain'] = _domain_row(rank, False)
        torch.save(res, Path(out_dir) / ('rank%d.pt' % rank))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope='module')
def rank_results(tmp_path_factory):
    if not dist.is_gloo_available():
        pytest.skip('gloo backend not available')
    d = tmp_path_factory.mktemp('ddp_pcgrad')
    ctx = mp.get_context('spawn')
    procs = [ctx.Process(target=_worker, args=(r, str(d / 'init'), str(d))) for r in range(WORLD)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    return [torch.load(d / ('rank%d.pt' % r), weights_only=False) for r in range(WORLD)]


def test_both_forwards_unsynced_with_batchnorm_raise_nothing(rank_results):
    for r in rank_results:
        for mode in ('synced', 'unsynced'):
            assert not isinstance(r[mode], str), r[mode]


def test_manual_all_reduce_agrees_across_ranks_and_equals_the_mean_of_local_pcgrad(rank_results):
    a, b = rank_results[0]['synced'], rank_results[1]['synced']
    assert any(n.startswith('domain_discriminator.') for n in a)
    for n in a:
        mean_local = (rank_results[0]['local'][n] + rank_results[1]['local'][n]) / 2
        assert torch.allclose(a[n], b[n], rtol=0, atol=1e-12), ('ranks disagree', n)
        assert torch.allclose(a[n], mean_local, rtol=1e-9, atol=1e-12), ('not the mean of local PCGrad', n)


def test_without_the_manual_all_reduce_the_ranks_drift_apart(rank_results):
    a, b = rank_results[0]['unsynced'], rank_results[1]['unsynced']
    assert any(not torch.allclose(a[n], b[n], atol=1e-9) for n in a), 'negative control: torchjd must bypass DDP'
    for r in rank_results:
        for n in r['unsynced']:
            assert torch.allclose(r['unsynced'][n], r['local'][n], rtol=1e-9, atol=1e-12), n


def test_discriminator_gets_a_gradient_through_pcgrad(rank_results):
    for n, g in rank_results[0]['synced'].items():
        if n.startswith('domain_discriminator.'):
            assert g.abs().max() > 0, n


def test_domain_row_on_the_backbone_is_reversed_and_scaled_by_alpha(rank_results):
    for r in rank_results:
        for g, gp in zip(r['row_grl'], r['row_plain']):
            assert gp.abs().max() > 0
            assert torch.allclose(g, -ALPHA * gp, rtol=1e-9, atol=1e-15)
