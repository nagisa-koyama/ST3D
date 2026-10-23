"""The in-head DANN under DistributedDataParallel, on the PLAIN self-training path (experiments_md/20261022_06 §9).

Until the PL S->B + DANN row, the CenterPoint in-head DANN (pcdet/models/model_utils/dann_utils.py) had only run on ONE
GPU (26376, 26382). The row runs on two, through train_one_epoch_st's plain summed backward with the first forward
unsynced (`ddp_two_forwards`). The GRL is a custom autograd.Function (ReverseLayerF): its backward is a node INSIDE the
graph, so DDP's per-parameter hooks still fire during `loss.backward()` and no manual all-reduce is needed - unlike
torchjd, which differentiates with torch.autograd.grad and bypasses the reducer (tests/test_self_training_ddp_grad_sync.py).
This file pins that claim on CPU over gloo, with the real dann_utils module and a BatchNorm backbone:

  * the gradients after the two forwards and one backward agree across ranks and equal the mean of the per-rank
    references computed without DDP - backbone, detection head and discriminator alike;
  * the domain loss's own gradient into the backbone is non-zero and the NEGATIVE of the same loss's gradient with the
    GRL bypassed, under DDP as well (the 2026-09-26 UADA3D conditional GRL sent zero; ST3D 1247338);
  * the discriminator's gradient is not reversed, and the detection loss sends it nothing.

The domain-loss part of a DDP-reduced gradient is isolated as (summed loss) - (detection loss), the latter's reduced
value being the mean of the per-rank detection gradients (float64, so the subtraction is exact enough). A DDP backward
of the domain loss ALONE is not a situation the row meets and is not tested: the detection head then gets no gradient,
and with find_unused_parameters=False (train.py) DDP never finishes that bucket and skips its all-reduce.
"""
import multiprocessing as mp
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcdet.models.model_utils import dann_utils  # noqa: E402

WORLD = 2
DANN_WEIGHT = 0.1


class _Toy(nn.Module):
    """backbone (conv + BN) -> features; detection head on the features; in-head DANN on the SAME features, as
    CenterHead taps spatial_features_2d."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.backbone = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.head = nn.Conv2d(8, 2, 1)
        self.domain_discriminator = dann_utils.build_domain_discriminator(
            {'LOSS_WEIGHTS': {'dann_weight': DANN_WEIGHT}}, in_channels=8)

    def forward(self, x, domain_label, grl=True):
        f = self.backbone(x)
        det = (self.head(f) ** 2).mean()
        preds = dann_utils.domain_predictions(self.domain_discriminator, f) if grl else self.domain_discriminator(f)
        dann, _ = dann_utils.domain_adversarial_loss(preds, domain_label, DANN_WEIGHT)
        return det, dann


def _model():
    return _Toy().double()


def _batches(rank):
    g = torch.Generator().manual_seed(100 + rank)
    return (torch.randn(3, 3, 6, 6, generator=g, dtype=torch.float64),
            torch.randn(3, 3, 6, 6, generator=g, dtype=torch.float64))   # source, target


def _total(m, xs, xt, sync_toggle=None, grl=True, which='all'):
    """train_one_epoch_st's plain path: source forward (unsynced under DDP), target forward (synced), one backward of
    src + tar + 0.5 dann_src + 0.5 dann_tar."""
    if sync_toggle:
        sync_toggle(False)
    ds, dns = m(xs, 0, grl)
    if sync_toggle:
        sync_toggle(True)
    dt, dnt = m(xt, 1, grl)
    dann = 0.5 * dns + 0.5 * dnt
    return {'all': ds + dt + dann, 'dann': dann, 'det': ds + dt}[which]


def _grads(m):
    return {n: (p.grad.clone() if p.grad is not None else None) for n, p in m.named_parameters()}


def _local(rank, grl=True, which='all'):
    m = _model()
    xs, xt = _batches(rank)
    _total(m, xs, xt, grl=grl, which=which).backward()
    return _grads(m)


def _worker(rank, init_file, out_dir):
    dist.init_process_group('gloo', init_method='file://%s' % init_file, rank=rank, world_size=WORLD)
    try:
        res = {}
        for name, grl in (('all', True), ('all_nogrl', False)):
            ddp = nn.parallel.DistributedDataParallel(_model())

            def toggle(on, ddp=ddp):
                ddp.require_backward_grad_sync = bool(on)   # what train_st_utils._set_ddp_grad_sync does

            xs, xt = _batches(rank)
            _total(ddp, xs, xt, sync_toggle=toggle, grl=grl).backward()
            res[name] = {n.replace('module.', '', 1): g for n, g in _grads(ddp).items()}
            res[name + '_local'] = _local(rank, grl=grl)
        res['det_local'] = _local(rank, which='det')
        torch.save(res, Path(out_dir) / ('rank%d.pt' % rank))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope='module')
def rank_results(tmp_path_factory):
    if not dist.is_gloo_available():
        pytest.skip('gloo backend not available')
    d = tmp_path_factory.mktemp('ddp_dann')
    ctx = mp.get_context('spawn')
    procs = [ctx.Process(target=_worker, args=(r, str(d / 'init'), str(d))) for r in range(WORLD)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    return [torch.load(d / ('rank%d.pt' % r)) for r in range(WORLD)]


def _mean_local(res, key, name):
    return torch.stack([r[key][name] for r in res]).mean(0)


@pytest.mark.parametrize('mode', ['all', 'all_nogrl'])
def test_ddp_reduces_every_parameter_including_the_discriminator(rank_results, mode):
    names = list(rank_results[0][mode])
    assert any(n.startswith('domain_discriminator.') for n in names)
    for n in names:
        a, b = rank_results[0][mode][n], rank_results[1][mode][n]
        assert a is not None and b is not None, n
        assert torch.allclose(a, b, atol=1e-12), ('ranks disagree', mode, n)
        assert torch.allclose(a, _mean_local(rank_results, mode + '_local', n), atol=1e-12), ('not the mean', mode, n)


def _domain_part(rank_results, mode, n):
    """The domain loss's share of the DDP-reduced gradient: summed-loss gradient minus the reduced detection gradient
    (= the mean of the per-rank detection gradients; None where the detection loss does not reach)."""
    det = [r['det_local'][n] for r in rank_results]
    det = torch.zeros_like(rank_results[0][mode][n]) if det[0] is None else torch.stack(det).mean(0)
    return rank_results[0][mode][n] - det


def test_domain_loss_gradient_into_the_backbone_is_reversed_and_non_zero_under_ddp(rank_results):
    for n in ('backbone.0.weight', 'backbone.1.weight'):
        g, g_plain = _domain_part(rank_results, 'all', n), _domain_part(rank_results, 'all_nogrl', n)
        assert g.abs().max() > 1e-6, ('zero gradient through the GRL', n, g.abs().max())
        assert torch.allclose(g, -g_plain, rtol=1e-6, atol=1e-12), ('GRL did not negate', n)


def test_discriminator_gradient_is_not_reversed_and_the_detection_loss_does_not_reach_it(rank_results):
    names = [n for n in rank_results[0]['all'] if n.startswith('domain_discriminator.')]
    assert names
    for n in names:
        for r in rank_results:
            assert r['det_local'][n] is None, ('detection loss reached the discriminator', n)
        g, g_plain = _domain_part(rank_results, 'all', n), _domain_part(rank_results, 'all_nogrl', n)
        assert g.abs().max() > 0, n
        assert torch.allclose(g, g_plain, rtol=1e-6, atol=1e-12), n


def test_detection_head_gets_no_domain_gradient(rank_results):
    """The tap is the backbone output; parameters downstream of it see the detection loss only."""
    for n in ('head.weight', 'head.bias'):
        assert torch.allclose(_domain_part(rank_results, 'all', n), torch.zeros(()).double(), atol=1e-12), n


# ---------------------------------------------------------------- the row (PL S->B + DANN)

def _flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, pre + k + '.'))
        else:
            out[pre + k] = v
    return out


def test_pl_s2b_dann_row_differs_from_the_30243_row_only_in_dann_weight(monkeypatch):
    """T05 §5 "User 10-24 ~02:40" item 2: 30243's config unchanged plus the fork's DANN block exactly as in
    centerpoint-st3d-dann-lyft2nuscenes.yaml (dann_weight 0.1 in the student's LOSS_WEIGHTS; teacher untouched)."""
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(ROOT / 'tools')

    def load(p):
        c = EasyDict()
        cfg_from_yaml_file(p, c)
        return c

    base = load('cfgs/da-ieee-access/centerpoint-st3d-nuscenes-singapore2boston.yaml')
    row = load('cfgs/da-ieee-access/centerpoint-st3d-dann-nuscenes-singapore2boston.yaml')
    ref = load('cfgs/da-ieee-access/centerpoint-st3d-dann-lyft2nuscenes.yaml')
    a, b = _flat(base), _flat(row)
    diffs = sorted(k for k in set(a) | set(b) if not k.endswith('_BASE_CONFIG_') and str(a.get(k)) != str(b.get(k)))
    assert diffs == ['MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.dann_weight'], diffs
    w = 'MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.dann_weight'
    assert b[w] == _flat(ref)[w] == 0.1
    assert 'dann_weight' not in row.SELF_TRAIN.MODEL_TEACHER.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS
    assert not row.SELF_TRAIN.get('USE_TORCHJD', False), 'plain summed backward, as the published DANN row'
    assert not row.SELF_TRAIN.get('DSNORM', False) and not row.SELF_TRAIN.get('TEACHER_ADABN', False)
    assert list(row.SELF_TRAIN.SCORE_THRESH) == [0.2, 0.33, 0.46]
    assert list(row.SELF_TRAIN.NEG_THRESH) == [0.149, 0.15, 0.151]
