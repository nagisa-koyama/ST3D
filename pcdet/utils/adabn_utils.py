"""BatchNorm statistics re-estimation on unlabelled target data (AdaBN), for any model in this repo.

Why it exists (experiments_md/20261004_01): on Lyft -> nuScenes and nuScenes -> KITTI, re-estimating a
source-only CenterPoint's BatchNorm statistics on the target's TRAIN point clouds - no labels, no
weight update - recovers what UADA3D's adversarial training gains (its loop forwards target batches in
train mode, so its statistics end up ~half target) and what the S1 density correction gains. The same
operation applied to a frozen self-training TEACHER (`SELF_TRAIN.TEACHER_ADABN`) changes every
downstream use of its target predictions: the pseudo-labels, the count-balance cuts and foreground
statistics derived from them, and any size estimate read from its confident boxes.

UDA-legal by construction: the frames come from the target TRAIN split, read in the eval-mode view
(no augmentation), and only `running_mean` / `running_var` change.

Determinism under DDP: every rank reads the same strided frames in the same order and BatchNorm is not
synchronised, so every rank arrives at identical statistics without communication.

Per-domain BatchNorm (`SELF_TRAIN.DSNORM`, pcdet/models/model_utils/dsnorm.py): a DSNorm layer keeps a
SOURCE and a TARGET set of statistics. A teacher converted to DSNorm and loaded from a source-only
checkpoint starts with both sets equal to the source's, so before this module it labelled the target
with source statistics. Here only the TARGET set is re-estimated; the source set is left exactly as
trained. Plain BatchNorm layers (DSNORM off) have one set, which is re-estimated.
"""
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def bn_layers(model):
    """Every normalisation layer with running statistics: plain BatchNorm and per-domain DSNorm."""
    from pcdet.models.model_utils.dsnorm import DSNorm
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, DSNorm))]


def _is_ds(m):
    return hasattr(m, 'running_mean_target')


def _stats(m):
    """The statistics re-estimation acts on: the TARGET set of a DSNorm layer, else the only set."""
    if _is_ds(m):
        return m.running_mean_target, m.running_var_target
    return m.running_mean, m.running_var


def train_split_stats_loader(dataset_cfg, class_names, model_ontology, frames, batch_size, workers, logger=None):
    """Eval-mode loader over `frames` strided frames of the block's TRAIN split (no augmentation)."""
    from pcdet.datasets import __all__ as DATASETS  # local: pcdet.datasets imports this package
    block = copy.deepcopy(dataset_cfg)
    block.DATA_SPLIT['test'] = block.DATA_SPLIT['train']
    if 'INFO_PATH' in block:  # Waymo has none: its infos follow DATA_SPLIT (per-sequence pickles)
        block.INFO_PATH['test'] = block.INFO_PATH['train']
    ds = DATASETS[block.DATASET](dataset_cfg=block, class_names=class_names, root_path=None,
                                 training=False, logger=logger, model_ontology=model_ontology)
    stride = max(1, len(ds) // frames)
    idx = list(range(0, len(ds), stride))[:frames]
    if logger is not None:
        logger.info('[adabn] statistics from %d of %d %s frames of %s (stride %d), eval-mode view, no labels'
                    % (len(idx), len(ds), block.DATA_SPLIT['test'], block.DATASET, stride))
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=workers,
                      collate_fn=ds.collate_batch, pin_memory=True)


@torch.no_grad()
def reestimate_bn(model, loader, logger=None, mix=1.0, to_gpu=True):
    """Reset every BN layer, re-estimate as a cumulative average over `loader`, optionally mix.

    The rest of the model stays in eval mode (no target assignment, no loss), so this works on any
    detector and on a teacher whose head has no GT to assign. Returns the saved statistics.
    `mix` < 1 keeps `mix * new + (1 - mix) * saved`.
    """
    if to_gpu:
        from pcdet.models import load_data_to_gpu
    layers = bn_layers(model)
    saved = {n: tuple(t.clone() for t in _stats(m)) for n, m in layers}
    momenta = {n: m.momentum for n, m in layers}
    tracked = {n: m.num_batches_tracked.clone() for n, m in layers}
    domains = {n: m.domain_label for n, m in layers if _is_ds(m)}
    was_training = model.training
    model.eval()
    for _, m in layers:
        mean, var = _stats(m)
        mean.zero_()
        var.fill_(1)
        m.num_batches_tracked.zero_()  # momentum None averages over num_batches_tracked
        m.momentum = None
        if _is_ds(m):
            m.set_domain_label(1)  # forward reads and updates the TARGET set only
        m.train()
    for i, batch in enumerate(loader):
        if to_gpu:
            load_data_to_gpu(batch)
        model(batch)
        if logger is not None and i % 50 == 0:
            logger.info('[adabn] statistics pass %d/%d' % (i + 1, len(loader)))
    for n, m in layers:
        m.momentum = momenta[n]
        m.num_batches_tracked.copy_(tracked[n])
        if _is_ds(m):
            m.set_domain_label(domains[n])
        if mix != 1.0:
            mu0, var0 = saved[n]
            mean, var = _stats(m)
            mean.mul_(mix).add_((1 - mix) * mu0)
            var.mul_(mix).add_((1 - mix) * var0)
    model.train(was_training)
    return saved


@torch.no_grad()
def bn_gap(model, saved):
    """Per layer: mean |mu_new - mu_old| / sigma_old, mean and max |log sigma_new / sigma_old|."""
    out = {}
    for n, m in bn_layers(model):
        mu0, var0 = saved[n]
        mean, var = _stats(m)
        sd0 = (var0 + m.eps).sqrt()
        log_sd = 0.5 * torch.log((var + m.eps) / (var0 + m.eps))
        out[n] = dict(
            mean_shift=float(((mean - mu0).abs() / sd0).mean()),
            log_sd_ratio=float(log_sd.abs().mean()),
            max_abs_log_sd_ratio=float(log_sd.abs().max()),
            frac_sd_collapse=float((log_sd < -np.log(10.0)).float().mean()),
            channels=int(mu0.numel()))
    return out


def summarise_gap(gap):
    groups = {}
    for n, g in gap.items():
        if n.startswith('discriminator'):
            continue
        groups.setdefault(n.split('.')[0], []).append(g)
    return {k: dict(layers=len(v), mean_shift=float(np.mean([g['mean_shift'] for g in v])),
                    log_sd_ratio=float(np.mean([g['log_sd_ratio'] for g in v])),
                    max_abs_log_sd_ratio=float(np.max([g['max_abs_log_sd_ratio'] for g in v])),
                    frac_sd_collapse=float(np.mean([g['frac_sd_collapse'] for g in v]))) for k, v in groups.items()}


def adapt_teacher(model_teacher, cfg, class_names, model_ontology, logger, workers=4):
    """Apply `SELF_TRAIN.TEACHER_ADABN` (dict: FRAMES, MIX, BATCH_SIZE) to a frozen teacher, in place.

    Reads the TARGET's train split (`cfg.DATA_CONFIG_TAR`). Call it after the teacher's weights are
    loaded and before anything iterates a loader over the teacher or wraps it in DDP.
    """
    spec = cfg.SELF_TRAIN.TEACHER_ADABN
    frames, mix, bs = int(spec.get('FRAMES', 1000)), float(spec.get('MIX', 1.0)), int(spec.get('BATCH_SIZE', 4))
    loader = train_split_stats_loader(cfg.DATA_CONFIG_TAR, class_names, model_ontology, frames, bs, workers, logger)
    saved = reestimate_bn(model_teacher, loader, logger, mix=mix)
    summary = summarise_gap(bn_gap(model_teacher, saved))
    for k, v in summary.items():
        logger.info('[adabn] teacher BN gap %-12s layers=%3d mean_shift=%.3f |log sd ratio|=%.3f max=%.2f'
                    % (k, v['layers'], v['mean_shift'], v['log_sd_ratio'], v['max_abs_log_sd_ratio']))
    ds = any(_is_ds(m) for _, m in bn_layers(model_teacher))
    logger.info('[adabn] teacher %s re-estimated on the target train split (FRAMES=%d, MIX=%g); weights '
                'untouched' % ('per-domain BN: TARGET statistics' if ds else 'BatchNorm statistics', frames, mix))
    return summary
