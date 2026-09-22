"""Regression tests for checkpoint claiming in `tools/test.py::repeat_eval_ckpt` under DDP.

`repeat_eval_ckpt` runs on every rank with no rank guard, and decides what to evaluate by
reading a shared record file and immediately appending to it. That read-then-append is a
race, and its loser deadlocks the job: whichever rank claims a checkpoint first makes the
others see it as already evaluated, so they return from `repeat_eval_ckpt` while the winner
enters `eval_one_epoch`, wraps the model in `DistributedDataParallel` and blocks forever in
an NCCL collective nobody else will join.

Observed on job 25714 (2026-09-22): KITTI, 2 GPUs, one epoch. Training finished, the epoch
closed, the checkpoint saved, then eval hung with a record file holding exactly one line.
Every DDP smoke test before it had passed because they all stop inside the training loop.

Fixed by `claim_next_ckpt`: rank 0 alone reads and claims, then broadcasts its decision, so
every rank evaluates the same checkpoint or leaves the loop together. Single-GPU runs take
the original code path with no collective at all.

See experiments_md/20260922_06_throughput_levers_and_ddp_enablement.md §2c.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))


def _load_tools_test():
    """Load tools/test.py under its own module name.

    It cannot be imported as `test` - that is a stdlib package name, and pytest would also
    try to collect it as a test module.
    """
    spec = importlib.util.spec_from_file_location('st3d_tools_test', REPO / 'tools' / 'test.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tools_test = _load_tools_test()


@pytest.fixture
def ckpt_env(tmp_path):
    """A checkpoint dir with epochs 1..2 and an empty record file, as a real run leaves them."""
    ckpt_dir = tmp_path / 'ckpt'
    ckpt_dir.mkdir()
    for epoch in (1, 2):
        (ckpt_dir / ('checkpoint_epoch_%d.pth' % epoch)).touch()
    record = tmp_path / 'eval_list_val.txt'
    record.touch()
    args = argparse.Namespace(start_epoch=0)
    return ckpt_dir, record, args


def _claimed(record):
    return [line.strip() for line in record.read_text().splitlines() if line.strip()]


def test_single_gpu_claims_and_records_exactly_as_before(ckpt_env):
    """`dist_test=False` must be the untouched original path: claim, append, no collective."""
    ckpt_dir, record, args = ckpt_env

    epoch_id, ckpt = tools_test.claim_next_ckpt(ckpt_dir, record, args, dist_test=False)
    assert epoch_id == '1'
    assert ckpt.endswith('checkpoint_epoch_1.pth')
    assert _claimed(record) == ['1']

    epoch_id, _ = tools_test.claim_next_ckpt(ckpt_dir, record, args, dist_test=False)
    assert epoch_id == '2'
    assert _claimed(record) == ['1', '2']

    # Exhausted: -1 is returned and nothing further is recorded.
    epoch_id, ckpt = tools_test.claim_next_ckpt(ckpt_dir, record, args, dist_test=False)
    assert (epoch_id, ckpt) == (-1, None)
    assert _claimed(record) == ['1', '2']


def _unguarded_two_rank_claim(ckpt_dir, record, args):
    """`get_no_evaluated_ckpt` called the way the pre-fix loop called it - once per rank."""
    rank0_epoch, _ = tools_test.get_no_evaluated_ckpt(ckpt_dir, record, args)
    with open(record, 'a') as f:
        print('%s' % rank0_epoch, file=f)
    rank1_epoch, _ = tools_test.get_no_evaluated_ckpt(ckpt_dir, record, args)
    return rank0_epoch, rank1_epoch


def test_unguarded_claim_deadlocks_with_one_checkpoint(tmp_path):
    """The bug as job 25714 hit it: one checkpoint, so the losing rank has nothing to do.

    `--num_epochs_to_eval 1` is the normal case, and it leaves exactly one unevaluated
    checkpoint. Rank 0 claims it; rank 1 sees it as already evaluated, gets -1 and leaves
    `repeat_eval_ckpt` - while rank 0 walks into the eval collective alone.
    """
    ckpt_dir = tmp_path / 'ckpt'
    ckpt_dir.mkdir()
    (ckpt_dir / 'checkpoint_epoch_1.pth').touch()
    record = tmp_path / 'eval_list_val.txt'
    record.touch()

    rank0_epoch, rank1_epoch = _unguarded_two_rank_claim(
        ckpt_dir, record, argparse.Namespace(start_epoch=0))

    assert rank0_epoch == '1'
    assert rank1_epoch == -1


def test_unguarded_claim_splits_ranks_across_checkpoints(ckpt_env):
    """With more than one checkpoint the same race misbehaves differently, not harmlessly.

    Nobody gets -1 here, so nothing hangs - instead the two ranks evaluate *different*
    checkpoints while sharing one set of collectives, and each checkpoint is scored by half
    the data. That is a wrong number rather than a hang, which is worse to catch. Both
    failure modes come from the same unguarded read-then-append.
    """
    ckpt_dir, record, args = ckpt_env

    rank0_epoch, rank1_epoch = _unguarded_two_rank_claim(ckpt_dir, record, args)

    assert rank0_epoch == '1'
    assert rank1_epoch == '2'


def test_distributed_ranks_all_receive_rank0s_claim(ckpt_env, monkeypatch):
    """Both ranks must come out with the same checkpoint, claimed exactly once."""
    ckpt_dir, record, args = ckpt_env
    wire = {}

    def fake_broadcast(object_list, src=0, device=None, **kwargs):
        # src fills the wire; every other rank reads it, as NCCL's broadcast would.
        if object_list[0] is not None or object_list[1] is not None:
            wire['payload'] = list(object_list)
        else:
            object_list[:] = list(wire['payload'])

    monkeypatch.setattr(tools_test.dist, 'broadcast_object_list', fake_broadcast)

    monkeypatch.setattr(tools_test.common_utils, 'get_dist_info', lambda: (0, 2))
    rank0 = tools_test.claim_next_ckpt(ckpt_dir, record, args, dist_test=True)

    monkeypatch.setattr(tools_test.common_utils, 'get_dist_info', lambda: (1, 2))
    rank1 = tools_test.claim_next_ckpt(ckpt_dir, record, args, dist_test=True)

    assert rank0 == rank1 == ('1', str(ckpt_dir / 'checkpoint_epoch_1.pth'))
    # The non-zero rank must not have written its own claim - one claim per checkpoint.
    assert _claimed(record) == ['1']


def test_distributed_ranks_leave_the_loop_together(ckpt_env, monkeypatch):
    """When rank 0 finds nothing left, every rank must get -1 too, not a stale claim."""
    ckpt_dir, record, args = ckpt_env
    record.write_text('1\n2\n')
    wire = {}

    def fake_broadcast(object_list, src=0, device=None, **kwargs):
        if object_list[0] is not None or object_list[1] is not None:
            wire['payload'] = list(object_list)
        else:
            object_list[:] = list(wire['payload'])

    monkeypatch.setattr(tools_test.dist, 'broadcast_object_list', fake_broadcast)

    monkeypatch.setattr(tools_test.common_utils, 'get_dist_info', lambda: (0, 2))
    rank0 = tools_test.claim_next_ckpt(ckpt_dir, record, args, dist_test=True)
    monkeypatch.setattr(tools_test.common_utils, 'get_dist_info', lambda: (1, 2))
    rank1 = tools_test.claim_next_ckpt(ckpt_dir, record, args, dist_test=True)

    assert rank0 == rank1 == (-1, None)


def test_world_size_one_takes_the_non_collective_path(ckpt_env, monkeypatch):
    """A `--launcher pytorch` job on a single GPU must not attempt a broadcast."""
    ckpt_dir, record, args = ckpt_env

    def exploding_broadcast(*a, **kw):
        raise AssertionError('broadcast attempted at world_size=1')

    monkeypatch.setattr(tools_test.dist, 'broadcast_object_list', exploding_broadcast)
    monkeypatch.setattr(tools_test.common_utils, 'get_dist_info', lambda: (0, 1))

    epoch_id, _ = tools_test.claim_next_ckpt(ckpt_dir, record, args, dist_test=True)
    assert epoch_id == '1'
    assert _claimed(record) == ['1']
