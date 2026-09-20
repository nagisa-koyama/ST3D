"""Regression tests for DataLoader worker state under `persistent_workers`.

Background: build_dataloader sets `persistent_workers=True` (77b1baa, 2026-08-17) to stop
workers being re-forked after CUDA init, which was causing untraceable segfaults at epoch
boundaries. The side effect is that a worker holds a snapshot of the dataset taken when its
loader was FIRST iterated: no later mutation in the main process reaches it.

That silently broke self-training. `train_model_st` iterated the target loader before the epoch
loop (forking workers with training=True), then called `dataset.eval()` and reused the same
loader for pseudo-label generation. The workers stayed in train mode, so `__getitem__` called
`fill_pseudo_labels()` during the generation pass and raised
`ValueError: Cannot find pseudo label for frame: ...` - asking for the labels it was creating.
Job 25502 (2026-09-21) died this way; job 20603 (2026-07-28, pre-77b1baa) did not.

See experiments_md/20260921_02_persistent_workers_stale_dataset_state.md.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets import build_inference_dataloader, restart_persistent_workers  # noqa: E402


class _ModeProbeDataset(torch.utils.data.Dataset):
    """Stands in for DatasetTemplate: a `training` flag that `__getitem__` reports back.

    Mirrors DatasetTemplate.eval()/train() and the real failure's shape - in production the
    flag decides whether `__getitem__` calls `fill_pseudo_labels()`.
    """

    def __init__(self, n=8):
        self.training = True
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor([1 if self.training else 0])

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


def _build_like_build_dataloader(dataset, workers=2, batch_size=4):
    """Same DataLoader construction as pcdet.datasets.build_dataloader, minus pcdet deps."""
    return DataLoader(
        Subset(dataset, list(range(len(dataset)))),
        batch_size=batch_size, num_workers=workers, shuffle=False,
        drop_last=False, sampler=None, timeout=0,
        persistent_workers=(workers > 0),
    )


def _modes_seen(loader):
    """Iterate a loader and return the distinct `training` values its workers reported."""
    return set(torch.cat([b for b in iter(loader)]).flatten().tolist())


def _shutdown(*loaders):
    for loader in loaders:
        restart_persistent_workers(loader)


def test_persistent_workers_do_not_see_post_fork_mutation():
    """Documents the trap the other tests exist to work around. If this ever starts failing,
    PyTorch changed its semantics and the extra loader below may no longer be needed."""
    ds = _ModeProbeDataset()
    loader = _build_like_build_dataloader(ds)
    try:
        assert _modes_seen(loader) == {1}   # forked in train mode
        ds.eval()                           # main-process mutation, as train_model_st does
        assert _modes_seen(loader) == {1}, \
            'workers observed the mutation - persistent_workers semantics changed'
    finally:
        _shutdown(loader)


def test_inference_loader_workers_fork_in_eval_mode():
    """The fix: generation gets its OWN loader, first iterated while the dataset is in eval
    mode, so its workers are frozen in eval mode for the run's lifetime."""
    ds = _ModeProbeDataset()
    train_loader = _build_like_build_dataloader(ds)
    gen_loader = build_inference_dataloader(train_loader)
    try:
        ds.eval()
        assert _modes_seen(gen_loader) == {0}, \
            'generation workers must see training=False, or fill_pseudo_labels() is called'
        ds.train()

        # The training loader is only forked AFTER generation, and must be in train mode.
        assert _modes_seen(train_loader) == {1}

        # A later generation pass reuses the eval-mode pool even though the dataset object is
        # currently back in train mode - that is the point of a dedicated loader.
        assert _modes_seen(gen_loader) == {0}
    finally:
        _shutdown(train_loader, gen_loader)


def test_inference_loader_shares_the_dataset_object():
    """It must not rebuild the dataset - that would re-read the infos from disk."""
    ds = _ModeProbeDataset()
    train_loader = _build_like_build_dataloader(ds)
    gen_loader = build_inference_dataloader(train_loader)
    try:
        assert gen_loader.dataset is train_loader.dataset
        assert gen_loader.dataset.dataset is ds
    finally:
        _shutdown(train_loader, gen_loader)


def test_restart_persistent_workers_makes_a_mutation_visible():
    """Used at PROG_AUG.UPDATE_AUG epochs: re_prepare() alone cannot reach live workers."""
    ds = _ModeProbeDataset()
    loader = _build_like_build_dataloader(ds)
    try:
        assert _modes_seen(loader) == {1}
        ds.eval()
        assert _modes_seen(loader) == {1}        # still stale
        restart_persistent_workers(loader)
        assert _modes_seen(loader) == {0}        # re-forked, mutation now visible
    finally:
        _shutdown(loader)


def test_restart_persistent_workers_is_safe_when_never_iterated():
    ds = _ModeProbeDataset()
    loader = _build_like_build_dataloader(ds)
    restart_persistent_workers(loader)   # no iterator yet - must not raise
    try:
        assert _modes_seen(loader) == {1}
    finally:
        _shutdown(loader)


@pytest.mark.parametrize('workers', [0, 2])
def test_inference_loader_matches_source_worker_count(workers):
    ds = _ModeProbeDataset()
    train_loader = _build_like_build_dataloader(ds, workers=workers)
    gen_loader = build_inference_dataloader(train_loader)
    try:
        assert gen_loader.num_workers == workers
        # persistent_workers is invalid with num_workers=0.
        assert gen_loader.persistent_workers == (workers > 0)
        ds.eval()
        assert _modes_seen(gen_loader) == {0}
    finally:
        _shutdown(train_loader, gen_loader)
