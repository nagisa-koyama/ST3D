"""Train only the normalization layers (OPTIMIZATION.TRAIN_ONLY_BN; experiments_md 20261003_05 §5 item 58, ANALYSIS).

Every BatchNorm keeps its affine parameters trainable (and its running statistics update in train mode as usual); every other
parameter is frozen (requires_grad False). Modules of `keep_types` (item 58's per-range modulation, adabn_range.RangeAffine)
stay trainable too.

The optimizer is rebuilt over the returned container. The family's adam_onecycle wrapper (fastai_optim.OptimWrapper) already
skips parameters with requires_grad False in its decoupled weight decay (fastai_optim.py, "When some parameters are fixed"), so
freezing alone would also keep them fixed; the container keeps the optimizer state to the trained tensors only.
tests/test_bn_only.py pins that frozen weights stay bitwise equal either way.
"""
import torch


def freeze_all_but_bn(model, keep_types=()):
    """Freeze every parameter except BatchNorm affine parameters and those of `keep_types` modules.
    Returns (nn.ModuleList of the trainable modules, number of trainable tensors, number of frozen tensors)."""
    keep = [m for m in model.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) or (keep_types and isinstance(m, keep_types))]
    train_ids = {id(p) for m in keep for p in m.parameters(recurse=False)}
    n_frozen = 0
    for p in model.parameters():
        if id(p) in train_ids:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
            n_frozen += 1
    return torch.nn.ModuleList(keep), len(train_ids), n_frozen
