"""Regression tests: `--seed` never reached the DDP training sampler.

Under DDP, build_dataloader built the training sampler as
`torch.utils.data.distributed.DistributedSampler(subset)`, with no `seed`. torch seeds that
sampler's permutation with `seed + epoch` and defaults to seed 0, so from `--seed` (2026-09-29,
`907e02a`) until 2026-11-08 every 2-GPU run saw the same frame order and the same rank split at
every seed. A 2-GPU second seed varied only the initial weights and the augmentation draws; a
1-GPU second seed (DataLoader shuffle from the global RNG) also varied the order. Found while
answering where the training seed acts (experiments_md T05 item 33d, 2026-11-07), fixed opt-in
(`train.py --ddp_sampler_seed`, absent = historical behaviour; T05 item 40).

No process group is needed: the samplers get `num_replicas` / `rank` explicitly.
"""
import argparse
import ast
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Subset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pcdet.datasets import build_sampler  # noqa: E402

N = 37  # odd, so the 2-rank split pads one index, as in a real run
TRAIN_PY = REPO / 'tools' / 'train.py'


def _subset():
    return Subset(list(range(N)), indices=list(range(N)))


def _order(sampler, epoch):
    sampler.set_epoch(epoch)
    return list(iter(sampler))


def _pre_fix_training_sampler(subset, num_replicas, rank, seed):
    """The training branch as it stood before the fix: `seed` is accepted by the caller but never
    reaches the sampler, exactly as `--seed` never did."""
    del seed
    return torch.utils.data.distributed.DistributedSampler(subset, num_replicas=num_replicas, rank=rank)


def test_default_is_the_historical_seed0_sampler():
    """sampler_seed=None must keep the sampler every past DDP run used: torch's seed 0 + epoch."""
    subset = _subset()
    for epoch in (0, 1, 7):
        for rank in (0, 1):
            got = _order(build_sampler(subset, True, True, num_replicas=2, rank=rank), epoch)
            ref = torch.utils.data.distributed.DistributedSampler(subset, num_replicas=2, rank=rank)
            assert got == _order(ref, epoch)
            assert build_sampler(subset, True, True, num_replicas=2, rank=rank).seed == 0


def test_opt_in_seed_changes_the_frame_order():
    """With a sampler seed, seeds 666 and 667 see different orders in the same epoch."""
    subset = _subset()
    for epoch in (0, 1, 7):
        a = _order(build_sampler(subset, True, True, sampler_seed=666, num_replicas=2, rank=0), epoch)
        b = _order(build_sampler(subset, True, True, sampler_seed=667, num_replicas=2, rank=0), epoch)
        assert a != b, 'seeds 666 and 667 gave the same DDP frame order at epoch %d' % epoch
        again = _order(build_sampler(subset, True, True, sampler_seed=666, num_replicas=2, rank=0), epoch)
        assert a == again, 'one seed must give one order (reproducible)'


def test_pre_fix_code_ignores_the_seed():
    """The defect itself: before the fix the order is the same at every seed. This is the case the
    opt-in path must make fail; it documents why every 2-GPU two-seed read shares its frame order."""
    subset = _subset()
    a = _order(_pre_fix_training_sampler(subset, 2, 0, seed=666), 3)
    b = _order(_pre_fix_training_sampler(subset, 2, 0, seed=667), 3)
    assert a == b


@pytest.mark.parametrize('sampler_seed', [None, 666, 667])
def test_two_ranks_still_partition_every_epoch(sampler_seed):
    """The seeded order must still hand each epoch's frames to the two ranks disjointly and completely
    (one index repeats as padding when N is odd, as torch does)."""
    subset = _subset()
    for epoch in (0, 1, 5):
        r0 = _order(build_sampler(subset, True, True, sampler_seed=sampler_seed, num_replicas=2, rank=0), epoch)
        r1 = _order(build_sampler(subset, True, True, sampler_seed=sampler_seed, num_replicas=2, rank=1), epoch)
        assert len(r0) == len(r1) == (N + 1) // 2
        assert set(r0) | set(r1) == set(range(N))
        assert len(r0) + len(r1) - len(set(r0) | set(r1)) == 1  # exactly the one padded index


def test_one_gpu_and_eval_samplers_unchanged():
    subset = _subset()
    assert build_sampler(subset, False, True, sampler_seed=667) is None
    assert build_sampler(subset, False, False) is None
    # The eval sampler ignores sampler_seed: sequential, one padded index, as before.
    ev = build_sampler(subset, True, False, sampler_seed=667, num_replicas=2, rank=1)
    assert list(iter(ev)) == (list(range(N)) + [0])[1::2]


# --- train.py wiring, read with ast (train.py imports wandb / the full model stack at module level) ---

def _train_tree():
    return ast.parse(TRAIN_PY.read_text())


def _train_sampler_seed_fn():
    tree = _train_tree()
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_sampler_seed')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(TRAIN_PY), 'exec'), ns)
    return ns['train_sampler_seed']


def test_entry_passes_seed_only_when_opted_in():
    f = _train_sampler_seed_fn()
    assert f(argparse.Namespace(seed=667, ddp_sampler_seed=False)) is None
    assert f(argparse.Namespace(seed=667, ddp_sampler_seed=True)) == 667
    assert f(argparse.Namespace(seed=667)) is None  # an args object without the flag: historical


def test_flag_is_declared_off_by_default():
    calls = [n for n in ast.walk(_train_tree()) if isinstance(n, ast.Call)
             and getattr(n.func, 'attr', None) == 'add_argument' and n.args
             and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--ddp_sampler_seed']
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert kw['action'].value == 'store_true' and kw['default'].value is False


def test_every_training_loader_in_train_py_gets_the_sampler_seed():
    """Source and target TRAINING loaders both take sampler_seed=train_sampler_seed(args); a training
    loader without it would silently keep seed 0 under the flag."""
    training_calls = []
    for n in ast.walk(_train_tree()):
        if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'build_dataloader':
            kw = {k.arg: k.value for k in n.keywords}
            if isinstance(kw.get('training'), ast.Constant) and kw['training'].value is True:
                training_calls.append(kw)
    assert len(training_calls) == 2, 'expected the source and the target training loaders'
    for kw in training_calls:
        v = kw.get('sampler_seed')
        assert v is not None and isinstance(v, ast.Call) and v.func.id == 'train_sampler_seed'
