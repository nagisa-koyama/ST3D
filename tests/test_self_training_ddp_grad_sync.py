"""Gradient synchronisation for the two self-training backward paths under DistributedDataParallel.

Two CPU processes over gloo, a tiny linear model, one forward per "domain" (two forwards per
iteration, exactly as train_one_epoch_st does), then:

  * the PLAIN path - one backward over the summed loss - must come out of DDP's own reducer
    equal to the mean of the per-rank gradients (two forwards then one backward is fine for the
    reducer: each parameter's hook fires once);
  * the PCGrad path - `torchjd.backward` - computes with torch.autograd.grad and writes .grad
    itself, so DDP's reducer never fires. Left alone, each rank keeps its OWN aggregated gradient
    and the replicas drift apart silently. With DDP sync switched off for the iteration
    (`require_backward_grad_sync = False`, what `no_sync()` toggles) and
    `commu_utils.all_reduce_grads` afterwards, the ranks agree and match the mean of what each
    would have produced alone.

The negative control (torchjd without the manual all-reduce disagrees across ranks) is asserted
too, so the test fails if the thing it guards against stops being a hazard for a reason nobody
understands.
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
import _init_path  # noqa: F401,E402
from pcdet.utils import commu_utils  # noqa: E402

torchjd = pytest.importorskip('torchjd')
from torchjd.aggregation import PCGrad  # noqa: E402

WORLD = 2


def _model(bn=False):
    torch.manual_seed(0)
    if bn:
        return nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
    return nn.Linear(4, 2)


def _batches(rank):
    g = torch.Generator().manual_seed(100 + rank)
    return torch.randn(3, 4, generator=g), torch.randn(3, 4, generator=g)   # "source", "target"


def _flat_grad(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


def _local_reference(rank, mode, bn=False):
    """What this rank's gradient is WITHOUT any DDP: the quantity DDP must average."""
    m = _model(bn)
    xs, xt = _batches(rank)
    ls, lt = (m(xs) ** 2).mean(), (m(xt) ** 2).mean() * 0.5
    if mode == 'plain':
        (ls + lt).backward()
    else:
        torchjd.backward([ls, lt], PCGrad(), parallel_chunk_size=1)
    return _flat_grad(m)


def _worker(rank, init_file, out_dir):
    dist.init_process_group('gloo', init_method='file://%s' % init_file, rank=rank, world_size=WORLD)
    try:
        results = {}
        for mode in ('plain', 'torchjd_unsynced', 'torchjd_synced'):
            ddp = nn.parallel.DistributedDataParallel(_model())
            xs, xt = _batches(rank)
            if mode != 'plain':
                ddp.require_backward_grad_sync = False        # what train_one_epoch_st does
            ls = (ddp(xs) ** 2).mean()                         # forward 1: source
            lt = (ddp(xt) ** 2).mean() * 0.5                   # forward 2: target
            if mode == 'plain':
                (ls + lt).backward()
            else:
                torchjd.backward([ls, lt], PCGrad(), parallel_chunk_size=1)
                if mode == 'torchjd_synced':
                    ddp.require_backward_grad_sync = True
                    commu_utils.all_reduce_grads(ddp)
            results[mode] = _flat_grad(ddp).clone()
            results[mode + '_local'] = _local_reference(rank, 'plain' if mode == 'plain' else 'torchjd')

        # BatchNorm + broadcast_buffers: DDP rewrites running_mean/var in place at every synced
        # forward, and the source forward's graph already saved them (job 26404). The first forward
        # unsynced is the fix; the naive form is recorded as raised-or-not.
        for mode, first_unsynced in (('bn_naive', False), ('bn_first_forward_unsynced', True)):
            ddp = nn.parallel.DistributedDataParallel(_model(bn=True))
            xs, xt = _batches(rank)
            try:
                if first_unsynced:
                    ddp.require_backward_grad_sync = False
                ls = (ddp(xs) ** 2).mean()
                ddp.require_backward_grad_sync = True
                lt = (ddp(xt) ** 2).mean() * 0.5
                (ls + lt).backward()
                results[mode] = _flat_grad(ddp).clone()
            except RuntimeError as e:
                results[mode] = str(e)
        results['bn_local'] = _local_reference(rank, 'plain', bn=True)
        torch.save(results, Path(out_dir) / ('rank%d.pt' % rank))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope='module')
def rank_results(tmp_path_factory):
    if not dist.is_gloo_available():
        pytest.skip('gloo backend not available')
    d = tmp_path_factory.mktemp('ddp')
    ctx = mp.get_context('spawn')
    procs = [ctx.Process(target=_worker, args=(r, str(d / 'init'), str(d))) for r in range(WORLD)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    return [torch.load(d / ('rank%d.pt' % r)) for r in range(WORLD)]


def _mean_local(res, key):
    return torch.stack([r[key] for r in res]).mean(0)


def test_plain_summed_backward_is_reduced_by_ddp(rank_results):
    for r in rank_results:
        assert torch.allclose(r['plain'], _mean_local(rank_results, 'plain_local'), atol=1e-6)


def test_torchjd_without_manual_sync_leaves_the_ranks_apart(rank_results):
    a, b = (r['torchjd_unsynced'] for r in rank_results)
    assert not torch.allclose(a, b, atol=1e-6), 'the hazard this file guards against has vanished'
    for r in rank_results:   # each rank silently kept its own local result
        assert torch.allclose(r['torchjd_unsynced'], r['torchjd_unsynced_local'], atol=1e-6)


def test_torchjd_with_manual_all_reduce_agrees_across_ranks_and_equals_the_mean(rank_results):
    a, b = (r['torchjd_synced'] for r in rank_results)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a, _mean_local(rank_results, 'torchjd_synced_local'), atol=1e-6)


def test_all_reduce_grads_is_a_no_op_outside_a_process_group():
    """World size 1 (every single-GPU run): nothing is touched, a missing grad stays missing."""
    m = _model()
    (m(torch.ones(1, 4)).sum()).backward()
    before = m.weight.grad.clone()
    m.bias.grad = None
    commu_utils.all_reduce_grads(m)
    assert m.bias.grad is None
    assert torch.equal(m.weight.grad, before)


def test_bn_buffers_two_forwards_naive_raises_the_inplace_version_error(rank_results):
    """The exact failure of job 26404, reproduced on CPU. If this stops raising, the toggle in
    train_one_epoch_st is no longer load-bearing and should be re-examined, not deleted blindly."""
    for r in rank_results:
        assert isinstance(r['bn_naive'], str) and 'inplace operation' in r['bn_naive'], r['bn_naive']


def test_bn_buffers_first_forward_unsynced_reduces_correctly(rank_results):
    a, b = (r['bn_first_forward_unsynced'] for r in rank_results)
    assert not isinstance(a, str), a
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a, _mean_local(rank_results, 'bn_local'), atol=1e-6)
