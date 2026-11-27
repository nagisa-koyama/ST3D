"""Sparsity-invariant (count-normalized) sparse convolution for the 3D backbone (`MODEL.COUNT_NORM_CONV`; absent = unchanged).

Why (experiments_md 20261003_05 §5 item 64, design fixed 2026-11-27 13:02 JST, before any number): a sparse convolution sums
over the occupied input sites of each output site's window, so its output scales with the number of occupied neighbours, which
a change of scan lattice changes. Normalized convolution (Uhrig et al., "Sparsity Invariant CNNs", 3DV 2017; the operator is
theirs) multiplies each output site by K / n: K the kernel volume, n the number of occupied input sites in its window. n comes
from a weightless one-channel twin of the convolution on the same indices (the submanifold twin shares the indice key, so it
reuses the pairs; a strided twin computes its own, and the output indices are checked to be the same).

ANALYSIS (item 64); UDA-legal (no data statistic, no label). Tests: tests/test_count_norm_conv.py (GPU: spconv has no CPU path
in this image).
"""
import math

import spconv.pytorch as spconv
import torch


class CountNormConv(spconv.SparseModule):
    """A SparseModule, so spconv's SparseSequential hands it the sparse tensor."""

    def __init__(self, conv, scale=True):
        super().__init__()
        assert isinstance(conv, (spconv.SubMConv3d, spconv.SparseConv3d)), type(conv)
        self.conv = conv
        self.subm = isinstance(conv, spconv.SubMConv3d)
        self.K = float(math.prod(conv.kernel_size))
        cls = type(conv)
        kw = dict(kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, bias=False,
                  indice_key=conv.indice_key if self.subm else None)
        counter = cls(1, 1, **kw)
        with torch.no_grad():
            counter.weight.fill_(1.0)
        counter.weight.requires_grad_(False)
        self._counter = [counter.to(conv.weight.device)]     # not a submodule: no parameter, no state key, no optimizer entry
        self.scale = scale

    def count(self, x):
        counter = self._counter[0]
        if counter.weight.device != x.features.device:
            self._counter[0] = counter = counter.to(x.features.device)
        ones = x.replace_feature(torch.ones(x.features.shape[0], 1, device=x.features.device, dtype=x.features.dtype))
        return counter(ones)

    def forward(self, x):
        y = self.conv(x)
        if not self.scale:
            return y
        with torch.no_grad():
            c = self.count(x)
            n = align(c.indices, c.features, y.indices, y.spatial_shape)
            s = self.K / n.clamp_min(1.0)
        return y.replace_feature(y.features * s)


def site_keys(indices, spatial_shape):
    """One int64 per (batch, z, y, x) site."""
    i = indices.long()
    z, y, x = [int(v) for v in spatial_shape]
    return ((i[:, 0] * z + i[:, 1]) * y + i[:, 2]) * x + i[:, 3]


def align(src_indices, src_values, dst_indices, spatial_shape):
    """`src_values` (rows of `src_indices`) reordered to the rows of `dst_indices`. A strided spconv convolution cannot reuse
    another's index pairs (spconv: "only support reuse subm indices"), and its output order is not reproducible, so the
    counting twin's sites are matched by key; a different SET of sites raises."""
    if src_indices.shape == dst_indices.shape and torch.equal(src_indices, dst_indices):
        return src_values
    if src_indices.shape[0] != dst_indices.shape[0]:
        raise RuntimeError('CountNormConv: the counting twin has %d output sites, the convolution %d'
                           % (src_indices.shape[0], dst_indices.shape[0]))
    ks, kd = site_keys(src_indices, spatial_shape), site_keys(dst_indices, spatial_shape)
    order = torch.argsort(ks)
    pos = torch.searchsorted(ks[order], kd).clamp_max(ks.shape[0] - 1)
    if not torch.equal(ks[order][pos], kd):
        raise RuntimeError('CountNormConv: the counting twin produced other output sites than the convolution')
    return src_values[order][pos]


def install_count_norm(model, scale=True):
    """Wrap every SubMConv3d / SparseConv3d of `model.backbone_3d` in CountNormConv (in place, at build time)."""
    root = model.backbone_3d
    targets = [(n, m) for n, m in root.named_modules() if isinstance(m, (spconv.SubMConv3d, spconv.SparseConv3d))]
    for name, conv in targets:
        parent = root
        *path, leaf = name.split('.')
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, CountNormConv(conv, scale=scale))
    return len(targets)
