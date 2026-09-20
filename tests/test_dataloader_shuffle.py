"""Regression tests for training-set shuffling in build_dataloader.

`032aa5c` (2025-01-30) hoisted the shuffle expression out of the DataLoader call and inverted it
from `(sampler is None)` to `(sampler is not None)`. Effects, both silent:

  * single GPU (`dist=False`, every job on this cluster): `sampler is None`, so `shuffle` became
    False and training saw a FIXED sample order every epoch. For scene-sequential datasets
    (nuScenes: 99.8% of adjacent samples share a scene; PandaSet: strictly sequence-ordered) that
    makes each batch roughly one scene.
  * DDP training: `shuffle` became True *while* a sampler was passed, which DataLoader rejects
    outright with "sampler option is mutually exclusive with shuffle".

Fixed 2026-09-21. See experiments_md/20260921_03_dataloader_shuffle_disabled_in_training.md.
"""
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pcdet.datasets as pcdet_datasets  # noqa: E402
from pcdet.datasets import build_dataloader, build_inference_dataloader  # noqa: E402

N = 32


class _FakeDataset(torch.utils.data.Dataset):
    """Minimal stand-in so build_dataloader can be exercised without a real dataset on disk."""

    def __init__(self, dataset_cfg=None, class_names=None, root_path=None, training=True,
                 logger=None, model_ontology=None):
        self.training = training
        self.dataset_cfg = dataset_cfg

    def __len__(self):
        return N

    def __getitem__(self, idx):
        return idx

    @staticmethod
    def collate_batch(batch):
        return torch.tensor(batch)


@pytest.fixture
def fake_dataset_registered():
    pcdet_datasets.__all__['FakeDataset'] = _FakeDataset
    try:
        yield EasyDict({'DATASET': 'FakeDataset'})
    finally:
        pcdet_datasets.__all__.pop('FakeDataset', None)


def _order(loader):
    return torch.cat([b for b in iter(loader)]).tolist()


def test_training_loader_shuffles(fake_dataset_registered):
    _, loader, sampler = build_dataloader(
        fake_dataset_registered, ['Car'], batch_size=4, dist=False, workers=0, training=True)
    assert sampler is None
    first, second = _order(loader), _order(loader)
    assert sorted(first) == list(range(N)), 'every sample must still be visited exactly once'
    assert first != list(range(N)) or second != list(range(N)), \
        'training order is sequential in both epochs - shuffle is off'
    assert first != second, 'training order must differ between epochs'


def test_eval_loader_does_not_shuffle(fake_dataset_registered):
    _, loader, sampler = build_dataloader(
        fake_dataset_registered, ['Car'], batch_size=4, dist=False, workers=0, training=False)
    assert sampler is None
    assert _order(loader) == list(range(N))
    assert _order(loader) == list(range(N)), 'eval order must be stable across passes'


def test_force_no_shuffle_disables_shuffling(fake_dataset_registered):
    """tools/demo.py relies on this to keep frames in order."""
    _, loader, _ = build_dataloader(
        fake_dataset_registered, ['Car'], batch_size=4, dist=False, workers=0, training=True,
        force_no_shuffle=True)
    assert _order(loader) == list(range(N))


def test_ddp_training_loader_constructs_and_leaves_shuffling_to_the_sampler(
        fake_dataset_registered, monkeypatch):
    """Passing a sampler AND shuffle=True raises in DataLoader, so the DDP training branch could
    not build a loader at all while the condition was inverted."""
    class _FakeDistributedSampler(torch.utils.data.Sampler):
        def __init__(self, dataset, *a, **kw):
            self.n = len(dataset)

        def __iter__(self):
            return iter(range(self.n))

        def __len__(self):
            return self.n

    monkeypatch.setattr(torch.utils.data.distributed, 'DistributedSampler',
                        _FakeDistributedSampler)

    _, loader, sampler = build_dataloader(   # must not raise ValueError
        fake_dataset_registered, ['Car'], batch_size=4, dist=True, workers=0, training=True)
    assert sampler is not None
    assert loader.sampler is sampler, 'the sampler, not DataLoader, must drive ordering under DDP'


def test_shuffling_is_not_frozen_by_persistent_workers(fake_dataset_registered):
    """persistent_workers reuses one worker pool. Confirm it still draws a NEW permutation each
    time the loader is iterated - otherwise the fix above would be cosmetic."""
    _, loader, _ = build_dataloader(
        fake_dataset_registered, ['Car'], batch_size=4, dist=False, workers=2, training=True)
    try:
        assert loader.persistent_workers
        orders = [_order(loader) for _ in range(3)]
        assert all(sorted(o) == list(range(N)) for o in orders)
        assert len({tuple(o) for o in orders}) > 1, \
            'persistent workers reused a frozen permutation across epochs'
    finally:
        pcdet_datasets.restart_persistent_workers(loader)


def test_pseudo_label_generation_loader_never_shuffles(fake_dataset_registered):
    """Item 5: the generation loader is derived from a now-SHUFFLING training loader, so confirm
    it does not inherit that. Order does not affect correctness - pseudo-labels are keyed by
    frame_id - but a deterministic sweep keeps the tqdm counter, the memory-ensemble merge and
    the saved ps_label pkl reproducible."""
    _, train_loader, _ = build_dataloader(
        fake_dataset_registered, ['Car'], batch_size=4, dist=False, workers=0, training=True)
    gen_loader = build_inference_dataloader(train_loader)

    assert _order(train_loader) != list(range(N)), 'precondition: the training loader shuffles'
    assert gen_loader.sampler is not None and isinstance(
        gen_loader.sampler, torch.utils.data.SequentialSampler), \
        'generation must sweep sequentially'
    assert _order(gen_loader) == list(range(N))
    assert _order(gen_loader) == list(range(N)), 'generation order must be stable across refreshes'
