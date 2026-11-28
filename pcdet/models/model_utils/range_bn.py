"""Range-stratified BatchNorm for TRAINING (RS-BN, `MODEL.RANGE_BN`; absent = unchanged).

Why (experiments_md 20261003_05 §5 item 60, design fixed 2026-11-27 12:09 JST, before any number): item 57 re-estimated BN
statistics per range on a model trained with one set of statistics per channel, and it widened the far residual. RS-BN trains
the source model with per-range normalization from the start, so re-estimating per-range statistics on the target
(`reestimate_range_bn`) is the AdaBN that matches its training.

Per BatchNorm1d / BatchNorm2d layer, an element at x-y range rho (a voxel's centre from the last sparse convolution's indices,
a BEV cell's centre; pcdet/utils/adabn_range._Pos) gets item 57's piecewise-linear weights w_k(rho) over the knots, and

    y = gamma * (x - sum_k w_k mu_k) / sqrt(sum_k w_k var_k + eps) + beta

- training: mu_k, var_k are the w-weighted batch mean and biased variance of knot k (gradients flow through them, as in BN);
  running statistics per knot with the layer's momentum (unbiased variance); a knot whose weight sum in the batch is under
  `thin` elements uses the batch's pooled moments and is not updated (counted in `thin_steps`);
- evaluation: the running statistics, interpolated the same way;
- gamma / beta stay the wrapped layer's, one per channel; the wrapped layer's own running buffers are removed, so code that
  expects a plain BatchNorm's statistics fails loudly instead of re-estimating unused buffers;
- one knot: exactly BatchNorm (tests/test_range_bn.py);
- `set_range_scale` (item 60r): the knots looked up at scale * range, e.g. the target's row step mapped to the source's;
  1 (the default) is the unscaled lookup;
- `pooled` (item 60g, set only by `reestimate_range_bn(..., pooled=True)`): in training mode every element is normalized with
  the batch's pooled moments and every knot receives the same running update, so the layer ends holding one set of
  statistics per channel (global AdaBN on an RS-BN model). Off by default; nothing else sets it.

UDA-legal: the knots are fixed a priori; `reestimate_range_bn` reads unlabelled clouds only. Single GPU (no SyncBN).
"""
import torch

from pcdet.utils.adabn_range import _Pos, knot_weights


def _flat(x):
    if x.dim() == 2:
        return x, lambda y: y
    if x.dim() == 4:
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(-1, c), lambda y: y.reshape(b, h, w, c).permute(0, 3, 1, 2)
    raise ValueError('RangeBN input of shape %s' % (tuple(x.shape),))


class RangeBN(torch.nn.Module):
    def __init__(self, bn, pos, knots, thin=32.0):
        super().__init__()
        assert isinstance(bn, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)) and bn.affine and bn.track_running_stats
        self.bn = bn                                          # holds gamma / beta (the optimizer finds them as before)
        self.knots = tuple(float(k) for k in knots)
        self.thin = float(thin)
        self.eps = float(bn.eps)
        self.momentum = bn.momentum
        self.pooled = False                                   # item 60g only; not a buffer, so state keys are unchanged
        K, C = len(self.knots), bn.num_features
        dev = bn.weight.device
        self.register_buffer('running_mean', bn.running_mean.detach().clone()[None].repeat(K, 1))
        self.register_buffer('running_var', bn.running_var.detach().clone()[None].repeat(K, 1))
        self.register_buffer('num_batches_tracked', torch.zeros((), dtype=torch.long, device=dev))
        self.register_buffer('knot_batches', torch.zeros(K, dtype=torch.long, device=dev))
        self.register_buffer('thin_steps', torch.zeros(K, dtype=torch.long, device=dev))
        for b in ('running_mean', 'running_var', 'num_batches_tracked'):
            delattr(bn, b)
            bn.register_buffer(b, None)
        bn.track_running_stats = False
        self._pos = [pos]                                     # shared, not a submodule

    def extra_repr(self):
        return 'knots=%s, thin=%g' % (list(self.knots), self.thin)

    def weights(self, x):
        pos = self._pos[0]
        if x.dim() == 4:                                      # BEV cells: fixed per map size, cached on the shared positions
            cache = pos.__dict__.setdefault('wcache', {})
            key = (x.shape[0], x.shape[2], x.shape[3], str(x.device), x.dtype, self.knots, getattr(pos, 'scale', 1.0))
            if key not in cache:
                rho = pos.bev(x.shape[2], x.shape[3], x.shape[0], x.device)
                cache[key] = knot_weights(rho, self.knots).to(device=x.device, dtype=x.dtype)
            return cache[key]
        return knot_weights(pos.voxels(x.shape[0]), self.knots).to(device=x.device, dtype=x.dtype)

    def _forward_pooled(self, xf):
        """Training mode with `pooled`: plain BatchNorm's batch moments for every element; every knot updated with them."""
        n = xf.shape[0]
        mu = xf.mean(0)
        var = xf.var(0, unbiased=False)
        with torch.no_grad():
            ok = 1.0 if n >= self.thin else 0.0               # n is a Python int: no host sync
            K = self.running_mean.shape[0]
            self.num_batches_tracked += 1
            self.knot_batches += int(ok)
            self.thin_steps += 1 - int(ok)
            okv = torch.full((K,), ok, dtype=self.running_mean.dtype, device=xf.device)
            f = okv / self.knot_batches.clamp_min(1).to(okv.dtype) if self.momentum is None else okv * self.momentum
            unb = var * (n / max(n - 1, 1))
            self.running_mean.add_(f[:, None] * (mu.to(okv.dtype)[None] - self.running_mean))
            self.running_var.add_(f[:, None] * (unb.to(okv.dtype)[None] - self.running_var))
        return mu, var

    def forward(self, x):
        xf, back = _flat(x)
        if self.training and self.pooled:
            mu, var = self._forward_pooled(xf)
            y = (xf - mu) / torch.sqrt(var + self.eps) * self.bn.weight + self.bn.bias
            return back(y)
        w = self.weights(x)                                   # (n, K)
        if self.training:
            m0 = xf.mean(0).detach()                          # numerics only: mu_k = w^T x / s does not depend on it
            xc = xf - m0
            s = w.sum(0)                                      # (K,)
            sc = s.clamp_min(1e-12)[:, None]
            mk = (w.t() @ xc) / sc
            vk = ((w.t() @ (xc * xc)) / sc - mk * mk).clamp_min(0.0)
            pm = xc.mean(0, keepdim=True)
            pv = ((xc * xc).mean(0, keepdim=True) - pm * pm).clamp_min(0.0)
            thin = (s < self.thin)[:, None]
            mk = torch.where(thin, pm.expand_as(mk), mk)
            vk = torch.where(thin, pv.expand_as(vk), vk)
            mu = w @ mk + m0
            var = w @ vk
            with torch.no_grad():                             # no host sync: every knot updated by a factor, 0 if thin
                ok = (~thin[:, 0]).to(self.running_mean.dtype)
                self.num_batches_tracked += 1
                self.knot_batches += ok.long()
                self.thin_steps += (1 - ok).long()
                f = ok / self.knot_batches.clamp_min(1).to(ok.dtype) if self.momentum is None else ok * self.momentum
                unb = vk * (s / (s - 1.0).clamp_min(1.0))[:, None]
                self.running_mean.add_(f[:, None] * ((mk + m0).to(ok.dtype) - self.running_mean))
                self.running_var.add_(f[:, None] * (unb.to(ok.dtype) - self.running_var))
        else:
            mu = w @ self.running_mean.to(xf.dtype)
            var = w @ self.running_var.to(xf.dtype)
        y = (xf - mu) / torch.sqrt(var + self.eps) * self.bn.weight + self.bn.bias
        return back(y)


def range_bn_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, RangeBN)]


@torch.no_grad()
def knot_spread(m):
    """Largest absolute difference between any knot's running statistics and knot 0's (mean, variance)."""
    return (float((m.running_mean - m.running_mean[:1]).abs().max()),
            float((m.running_var - m.running_var[:1]).abs().max()))


def set_range_scale(model, scale):
    """Item 60r (experiments_md 20261003_05 §5 item 60r, design fixed 2026-11-28 15:32 JST): every RangeBN of `model` looks
    its knots up at `scale` * range from now on (voxels and BEV cells; the shared positions' caches are keyed by the scale).
    1.0 restores the unscaled lookup. Returns the scale set."""
    layers = range_bn_layers(model)
    if not layers:
        raise ValueError('no RangeBN layer in the model')
    scale = float(scale)
    assert scale > 0, scale
    for pos in {id(m._pos[0]): m._pos[0] for _, m in layers}.values():
        pos.scale = scale
    return scale


def install_range_bn(model, knots, pc_range, thin=32.0):
    """Wrap every BatchNorm1d / BatchNorm2d of `model` in RangeBN (in place, at build time) and register the sparse-convolution
    hooks that give voxel positions. Returns the number of wrapped layers."""
    import spconv.pytorch as spconv
    pos = _Pos(pc_range)
    model._range_bn_hooks = [m.register_forward_hook(pos.hook) for m in model.modules()
                             if isinstance(m, spconv.conv.SparseConvolution)]
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    for name, bn in targets:
        parent = model
        *path, leaf = name.split('.')
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, RangeBN(bn, pos, knots, thin))
    return len(targets)


@torch.no_grad()
def reestimate_range_bn(model, loader, logger=None, to_gpu=True, pooled=False):
    """AdaBN for an RS-BN model: every knot's statistics re-estimated as a cumulative average of its batch statistics over
    `loader` (the rest of the model in eval mode, as adabn_utils.reestimate_bn). A knot never above the threshold keeps its
    previous (source) statistics. Returns {layer: number of such knots}.

    `pooled=True` (item 60g): global AdaBN instead. During the pass each layer normalizes with the batch's pooled moments
    and every knot receives their cumulative average, so all knots end equal (`knot_spread` 0)."""
    if to_gpu:
        from pcdet.models import load_data_to_gpu
    layers = range_bn_layers(model)
    if not layers:
        raise ValueError('no RangeBN layer in the model')
    saved = {n: (m.running_mean.clone(), m.running_var.clone(), m.momentum, m.num_batches_tracked.clone(),
                 m.knot_batches.clone(), m.thin_steps.clone()) for n, m in layers}
    was_training = model.training
    model.eval()
    for _, m in layers:
        m.running_mean.zero_()
        m.running_var.fill_(1.0)
        m.knot_batches.zero_()
        m.momentum = None
        m.pooled = bool(pooled)
        m.train()
    for i, batch in enumerate(loader):
        if to_gpu:
            load_data_to_gpu(batch)
        model(batch)
        if logger is not None and i % 50 == 0:
            logger.info('[range-bn adabn] statistics pass %d/%d' % (i + 1, len(loader)))
    never = {}
    for n, m in layers:
        mean0, var0, mom, nbt, kb, ts = saved[n]
        m.pooled = False
        unseen = m.knot_batches == 0
        m.running_mean[unseen] = mean0[unseen]
        m.running_var[unseen] = var0[unseen]
        never[n] = int(unseen.sum())
        m.momentum = mom
        m.num_batches_tracked.copy_(nbt)
        m.knot_batches.copy_(kb)
        m.thin_steps.copy_(ts)
    model.train(was_training)
    if logger is not None:
        logger.info('[range-bn adabn] %d layers re-estimated%s; knots never above the threshold (kept source): %d'
                    % (len(layers), ' (pooled: one set of statistics for every knot)' if pooled else '',
                       sum(never.values())))
    return never
