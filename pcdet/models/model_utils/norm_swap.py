"""LayerNorm / GroupNorm in place of every BatchNorm (`MODEL.NORM_SWAP`; absent = unchanged).

Why (experiments_md 20261003_05 §5 item 68, design fixed 2026-11-28 16:42 JST, before any number): items 57-60r located the
forward pair's far residual in the normalization layers. LN and GN keep no running statistics, so there is no train / test
mismatch and nothing for AdaBN to fix; item 68 asks whether either transfers better than BN on nuScenes -> KITTI.

    MODEL:
        NORM_SWAP: {TYPE: layer}            # or {TYPE: group, MAX_GROUPS: 32}

Every BatchNorm1d / BatchNorm2d is replaced at build time (pcdet/models/__init__.build_network), keeping its eps:
- `layer`: per-element LayerNorm over channels. A sparse layer's (N, C) voxel features and a BEV map's (B, C, H, W) cells are
  each normalized by their own C values; per-channel gamma / beta.
- `group`: GroupNorm with G = min(MAX_GROUPS, C) groups (C must divide). Statistics per frame: in the sparse 3D backbone over
  the frame's active voxels (the frame of each row is the batch index of the last sparse convolution's output, recorded by a
  forward hook, as range_bn's positions), on BEV maps over every cell (torch.nn.GroupNorm). Per-channel gamma / beta.
gamma = 1, beta = 0 at initialization (a fresh model). Single GPU or DDP alike: nothing is synchronised, nothing is running.
UDA-legal: an architecture fixed a priori. Tests: tests/test_norm_swap.py.
"""
import torch
import torch.nn.functional as F


class _Rows:
    """The indices of the last sparse convolution's output (shared by every sparse norm layer of a model)."""

    def __init__(self):
        self.last = None

    def hook(self, module, inputs, output):
        self.last = output.indices

    def batch_index(self, n):
        if self.last is None or self.last.shape[0] != n:
            raise RuntimeError('sparse norm input of %d rows does not follow a recorded sparse convolution output (%s)'
                               % (n, None if self.last is None else self.last.shape[0]))
        return self.last[:, 0].long()


class ElementLayerNorm(torch.nn.Module):
    """LayerNorm over the channel axis of (N, C) or (B, C, H, W) input, per element."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features, self.eps = int(num_features), float(eps)
        self.weight = torch.nn.Parameter(torch.ones(self.num_features))
        self.bias = torch.nn.Parameter(torch.zeros(self.num_features))

    def extra_repr(self):
        return '%d, eps=%g' % (self.num_features, self.eps)

    def forward(self, x):
        if x.dim() == 2:
            return F.layer_norm(x, (self.num_features,), self.weight, self.bias, self.eps)
        if x.dim() == 4:
            y = F.layer_norm(x.permute(0, 2, 3, 1), (self.num_features,), self.weight, self.bias, self.eps)
            return y.permute(0, 3, 1, 2)
        raise ValueError('ElementLayerNorm input of shape %s' % (tuple(x.shape),))


class SparseGroupNorm(torch.nn.Module):
    """GroupNorm over (N, C) sparse-voxel features, statistics per frame (the batch index of each row)."""

    def __init__(self, num_groups, num_features, rows, eps=1e-5):
        super().__init__()
        assert num_features % num_groups == 0, (num_features, num_groups)
        self.num_groups, self.num_features, self.eps = int(num_groups), int(num_features), float(eps)
        self.weight = torch.nn.Parameter(torch.ones(self.num_features))
        self.bias = torch.nn.Parameter(torch.zeros(self.num_features))
        self._rows = [rows]                                   # shared, not a submodule

    def extra_repr(self):
        return '%d, %d, eps=%g' % (self.num_groups, self.num_features, self.eps)

    def forward(self, x):
        assert x.dim() == 2 and x.shape[1] == self.num_features, tuple(x.shape)
        n, C, G = x.shape[0], self.num_features, self.num_groups
        b = self._rows[0].batch_index(n)
        B = int(b.max()) + 1 if n else 0
        xg = x.view(n, G, C // G)
        cnt = torch.zeros(B, device=x.device, dtype=x.dtype).index_add(0, b, torch.ones_like(b, dtype=x.dtype))
        cnt = (cnt * (C // G)).clamp_min(1.0)[:, None]                                         # (B, 1)
        mean = torch.zeros(B, G, device=x.device, dtype=x.dtype).index_add(0, b, xg.sum(-1)) / cnt
        xc = xg - mean[b][:, :, None]
        var = torch.zeros(B, G, device=x.device, dtype=x.dtype).index_add(0, b, (xc * xc).sum(-1)) / cnt
        y = (xc / torch.sqrt(var[b][:, :, None] + self.eps)).reshape(n, C)
        return y * self.weight + self.bias


def norm_swap_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (ElementLayerNorm, SparseGroupNorm,
                                                                        torch.nn.GroupNorm))]


def install_norm_swap(model, cfg):
    """Replace every BatchNorm1d / BatchNorm2d of `model` in place (at build time); returns the number replaced."""
    kind = cfg['TYPE']
    assert kind in ('layer', 'group'), kind
    max_groups = int(cfg.get('MAX_GROUPS', 32))
    rows = _Rows()
    if kind == 'group':
        import spconv.pytorch as spconv
        model._norm_swap_hooks = [m.register_forward_hook(rows.hook) for m in model.modules()
                                  if isinstance(m, spconv.conv.SparseConvolution)]
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    for name, bn in targets:
        C, eps = bn.num_features, bn.eps
        if kind == 'layer':
            new = ElementLayerNorm(C, eps)
        elif isinstance(bn, torch.nn.BatchNorm1d):
            new = SparseGroupNorm(min(max_groups, C), C, rows, eps)
        else:
            new = torch.nn.GroupNorm(min(max_groups, C), C, eps=eps, affine=True)
        new.to(bn.weight.device)
        parent = model
        *path, leaf = name.split('.')
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, new)
    return len(targets)
