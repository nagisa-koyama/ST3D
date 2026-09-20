import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DistributedSampler as _DistributedSampler
from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .waymo.waymo_dataset import WaymoDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .lyft.lyft_dataset import LyftDataset
from .pandaset.pandaset_dataset import PandasetDataset


__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'WaymoDataset': WaymoDataset,
    'NuScenesDataset': NuScenesDataset,
    'LyftDataset': LyftDataset,
    'PandasetDataset': PandasetDataset
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0, model_ontology=None, force_no_shuffle=None, use_subset=False):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
        model_ontology=model_ontology,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    # `sampler is None`, NOT `is not None`. A sampler and shuffle=True are mutually exclusive in
    # DataLoader, so the DataLoader's own shuffling is only available when no sampler is passed;
    # under DDP the DistributedSampler does the shuffling instead (its default is shuffle=True,
    # and the eval branch above disables it explicitly).
    #
    # This read `is not None` from 032aa5c (2025-01-30) until 2026-09-21, which meant single-GPU
    # training fed samples in a FIXED order every epoch, and DDP training could not construct a
    # DataLoader at all. See
    # experiments_md/20260921_03_dataloader_shuffle_disabled_in_training.md.
    shuffle = (sampler is None) and training
    if force_no_shuffle is not None:
        shuffle = shuffle and not force_no_shuffle
    length = len(dataset) if not use_subset else min(16, len(dataset))
    if logger is not None:
        logger.info(f'Total number of samples: {length}, using subset: {use_subset}')
    subset = Subset(dataset, indices=list(range(length)))
    dataloader = DataLoader(
        subset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=shuffle, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0,
        # persistent_workers keeps worker processes alive across epochs instead of tearing
        # them down and re-forking them at every epoch boundary. Re-forking workers after the
        # main process has already initialized a CUDA context is a known source of instability
        # (segfaults with no Python traceback); this was observed deterministically at the
        # epoch1->epoch2 boundary for centerpoint-sourceonly.yaml regardless of --mem (jobs
        # 21396/21416). persistent_workers=True is invalid when num_workers=0, hence the guard.
        persistent_workers=(workers > 0)
    )

    return dataset, dataloader, sampler


def build_inference_dataloader(dataloader, sampler=None):
    """Build a SECOND DataLoader over an already-built dataset, for an inference-mode pass.

    Why this exists: `persistent_workers` above means a DataLoader's worker processes hold a
    forked snapshot of the dataset, taken when the loader is first iterated, and no later
    mutation in the main process ever reaches them - not `dataset.eval()`, not
    `data_augmentor.re_prepare()`. PyTorch's DataLoader.__iter__ returns the *same* iterator
    for a persistent-worker loader and `_reset` does not re-send the dataset.

    So a training loader and an inference pass cannot share one loader: whichever mode the
    workers forked with is the mode they keep for the run's lifetime. Self-training's
    pseudo-label generation needs the dataset in eval mode while training needs it in train
    mode, and from 77b1baa (2026-08-17) until 2026-09-21 it used the training loader for both -
    so generation ran with training=True workers, which called fill_pseudo_labels() and died
    asking for the very labels it was about to create. See
    experiments_md/20260921_02_persistent_workers_stale_dataset_state.md.

    This returns a loader with its OWN worker pool over the SAME dataset object, so nothing is
    re-read from disk. Iterate it for the first time while the dataset is in the mode you want
    its workers frozen in - after that the mode is fixed, which is exactly what makes it safe.
    """
    return DataLoader(
        dataloader.dataset,  # the Subset wrapper build_dataloader already created
        batch_size=dataloader.batch_size,
        pin_memory=True,
        num_workers=dataloader.num_workers,
        shuffle=False,
        collate_fn=dataloader.collate_fn,
        drop_last=False,
        sampler=sampler,
        timeout=0,
        persistent_workers=(dataloader.num_workers > 0),
    )


def restart_persistent_workers(loader):
    """Drop a DataLoader's persistent worker pool so the next `iter()` re-forks it.

    Worker processes hold a snapshot of the dataset taken when the loader was first iterated;
    a later mutation in the main process is invisible to them (see
    pcdet/datasets/__init__.py::build_inference_dataloader). Call this after mutating the
    dataset - e.g. `data_augmentor.re_prepare()` - if the change has to reach the workers.
    """
    it = getattr(loader, '_iterator', None)
    if it is not None:
        loader._iterator = None
        try:
            it._shutdown_workers()
        except Exception:
            pass  # already torn down; the next iter() re-forks regardless
