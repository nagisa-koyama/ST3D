"""Range-stratified AdaBN (RS-AdaBN): BatchNorm statistics re-estimated per range from the sensor, with each range keeping the
offset and spread it had on the source, in the normalized units the network was trained in.

Why (experiments_md 20261003_05 §5 item 57, design fixed 2026-11-27 00:10 JST): global AdaBN keeps one mean and variance per
channel, pooled over every element of a layer. On item 48's forward pair it removes 98-99 % of the lattice term within 10 m and
leaves +9 to +12 AP at 20-40 m. If the lattice shift acts differently by range, one pooled correction fits the near elements that
dominate the pool. RS-AdaBN tests that.

The transform, per BN layer, for an element at range rho (x-y distance of a voxel's or BEV cell's centre from the sensor):

    y = gamma * [ (x - mu_t(rho)) / sqrt(var_t(rho) + eps) * r(rho) + o(rho) ] + beta

- knots k (item 57: 5 / 15 / 25 / 35 / 55 m); each element carries piecewise-linear weights over them (constant beyond the end
  knots), and mu_t, var_t, r, o are interpolated with those weights, so no seam appears between knots;
- source pass (`collect`): the checkpoint as saved (eval mode) on source TRAIN clouds; per knot the weighted moments of each BN
  input give o_k = (mu_s_k - mu_s) / sigma_s and r_k = sigma_s_k / sigma_s against the same pass's pooled moments;
- target pass (`estimate`): `adabn_utils.reestimate_bn`'s procedure per knot - each batch normalized with its own (biased,
  weighted) statistics on the way through, running statistics the cumulative average of the batch statistics (unbiased);
- with ONE knot, o = 0 and r = 1 and the result is exactly global AdaBN (tests/test_adabn_range.py).

Positions: a BatchNorm2d input (B, C, H, W) is a BEV map laid out as spconv's dense() lays it out over POINT_CLOUD_RANGE (rows y,
columns x), as `adabn_masked` reads it. A BatchNorm1d input is the feature matrix of the spconv tensor the BN follows; a forward
hook on every sparse convolution records that output's indices (batch, z, y, x) and spatial shape.

UDA-legal: statistics from unlabelled TRAIN clouds of the source and the target, knots fixed a priori, no label read; only the
normalization changes, no weight.
"""
import numpy as np
import torch

from pcdet.utils import adabn_utils

KNOTS_ITEM57 = (5.0, 15.0, 25.0, 35.0, 55.0)   # centres of item 48's rings 0-10 / 10-20 / 20-30 / 30-40 / 40-70.4 m
# Which BN layers take the knots, per `rs_adabn_eval.py --knots` arm (a regex on the module name; None = all layers):
# item57_3d = item 57b; item57_2d = item 70 (experiments_md T05 §5 item 70). The 3d and 2d arms partition CenterPoint's BN layers.
ARM_LAYERS = {'global': None, 'item57': None, 'item57_3d': r'^backbone_3d\.', 'item57_2d': r'^(backbone_2d|dense_head)\.'}


def knot_weights(rho, knots):
    """(n, K) piecewise-linear weights of ranges `rho` (n,) over increasing `knots`; rows sum to 1; one knot -> all ones."""
    rho = torch.as_tensor(rho)
    k = torch.as_tensor(knots, dtype=rho.dtype if rho.is_floating_point() else torch.float32, device=rho.device)
    rho = rho.to(k.dtype)
    K = k.numel()
    w = torch.zeros(rho.numel(), K, dtype=k.dtype, device=rho.device)
    if K == 1:
        return w.fill_(1.0)
    r = rho.clamp(k[0], k[-1])
    j = torch.searchsorted(k, r, right=True).clamp(1, K - 1)                 # segment [k[j-1], k[j]]
    lo, hi = k[j - 1], k[j]
    t = (r - lo) / (hi - lo)
    rows = torch.arange(rho.numel(), device=rho.device)
    w[rows, j - 1] = 1.0 - t
    w[rows, j] += t
    return w


def weighted_moments(x, w):
    """x (n, C), w (n, K) -> (s (K,), mean (K, C), biased var (K, C)), in float64, centred on the pooled mean first."""
    x = x.double()
    w = w.double()
    m0 = x.mean(0, keepdim=True)
    xc = x - m0
    s = w.sum(0)
    safe = s.clamp_min(1e-12)[:, None]
    mean_c = (w.t() @ xc) / safe
    var = ((w.t() @ (xc * xc)) / safe - mean_c * mean_c).clamp_min(0.0)
    return s, mean_c + m0, var


class _Pos:
    """Element ranges per BN input: BEV cells from the map size, voxels from the last sparse convolution's output."""

    def __init__(self, pc_range):
        self.x0, self.y0, _, self.x1, self.y1, _ = [float(v) for v in pc_range]
        self.last = None
        self.cache = {}
        self.scale = 1.0          # item 60r: knots looked up at scale * range (range_bn.set_range_scale); 1 = unchanged

    def hook(self, module, inputs, output):
        self.last = (output.indices, list(output.spatial_shape))

    def bev(self, h, w, batch, device):
        key = (h, w, str(device), self.scale)
        if key not in self.cache:
            xs = self.x0 + (torch.arange(w, dtype=torch.float64, device=device) + 0.5) * (self.x1 - self.x0) / w
            ys = self.y0 + (torch.arange(h, dtype=torch.float64, device=device) + 0.5) * (self.y1 - self.y0) / h
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            self.cache[key] = self._scaled(torch.hypot(xx, yy).reshape(-1))
        return self.cache[key].repeat(batch)

    def _scaled(self, rho):
        return rho if self.scale == 1.0 else rho * self.scale

    def voxels(self, n):
        if self.last is None or self.last[0].shape[0] != n:
            raise RuntimeError('BatchNorm1d input of %d rows does not follow a recorded sparse convolution output (%s)'
                               % (n, None if self.last is None else self.last[0].shape[0]))
        idx, shape = self.last
        _, ny, nx = shape[-3:]
        x = self.x0 + (idx[:, 3].double() + 0.5) * (self.x1 - self.x0) / nx
        y = self.y0 + (idx[:, 2].double() + 0.5) * (self.y1 - self.y0) / ny
        return self._scaled(torch.hypot(x, y))


def _flat(x):
    """(n, C) view of a BN input and a function putting an (n, C) result back in its shape."""
    if x.dim() == 2:
        return x, lambda y: y
    if x.dim() == 4:
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(-1, c), lambda y: y.reshape(b, h, w, c).permute(0, 3, 1, 2)
    raise ValueError('BN input of shape %s' % (tuple(x.shape),))


class RangeStratifiedBN:
    """Holds the hooks and per-layer statistics; `collect` (source), `estimate` (target), `apply`, `restore`."""

    def __init__(self, model, pc_range):
        import spconv.pytorch as spconv
        self.model = model
        self.pos = _Pos(pc_range)
        self.layers = adabn_utils.bn_layers(model)
        if any(adabn_utils._is_ds(m) for _, m in self.layers):
            raise NotImplementedError('per-domain BN is not supported by RS-AdaBN')
        bad = [n for n, m in self.layers if not isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
        if bad:
            raise NotImplementedError('only BatchNorm1d (sparse) and BatchNorm2d (BEV) layers: %s' % bad[:3])
        self.handles = [m.register_forward_hook(self.pos.hook) for m in model.modules()
                        if isinstance(m, spconv.conv.SparseConvolution)]
        self.saved = {n: (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone(), m.momentum)
                      for n, m in self.layers}
        self.src = None
        self.tgt = None

    def ranges(self, x):
        if x.dim() == 4:
            return self.pos.bev(x.shape[2], x.shape[3], x.shape[0], x.device)
        return self.pos.voxels(x.shape[0])

    def _run(self, loader, logger, what, to_gpu=True):
        if to_gpu:
            from pcdet.models import load_data_to_gpu
        for i, batch in enumerate(loader):
            if to_gpu:
                load_data_to_gpu(batch)
            self.model(batch)
            if logger is not None and i % 50 == 0:
                logger.info('[rs-adabn] %s pass %d/%d' % (what, i + 1, len(loader)))

    def _install(self, make):
        for n, m in self.layers:
            m.forward = make(n, m)

    def _uninstall(self):
        for _, m in self.layers:
            if 'forward' in m.__dict__:
                del m.forward

    @torch.no_grad()
    def collect(self, loader, knots, logger=None, to_gpu=True):
        """Source pass, the checkpoint as saved (eval mode): per layer, per knot (s, M, Q) weighted sums and pooled sums."""
        self.model.eval()
        acc = {}

        def make(name, bn):
            orig = type(bn).forward

            def forward(x):
                xf, _ = _flat(x)
                w = knot_weights(self.ranges(x), knots).to(x.device)
                _accumulate(acc, name, xf, w)
                return orig(bn, x)
            return forward
        self._install(make)
        try:
            self._run(loader, logger, 'source', to_gpu)
        finally:
            self._uninstall()
        self.src = {n: _finish(a, bn_eps(m)) for (n, m) in self.layers for a in [acc[n]]}
        self.src_knots = tuple(knots)
        return self.src

    @torch.no_grad()
    def estimate(self, loader, knots, logger=None, diag_knots=None, to_gpu=True, layers=None):
        """Target pass (train-mode-like, per knot). With one knot: global AdaBN. `layers` (a regex on the module name,
        re.search; item 57b) gives `knots` to the matching layers only and one knot (global AdaBN) to every other layer.
        `diag_knots` also records the per-knot moments of every BN input during the pass (stage 0 of item 57; needs
        `collect` over the same knots for the comparison)."""
        import re
        knots = tuple(knots)
        if len(knots) > 1 and (self.src is None or self.src_knots != knots):
            raise ValueError('collect() the source over the same knots first')
        lk = {n: knots if layers is None or re.search(layers, n) else (0.0,) for n, _ in self.layers}
        if layers is not None and logger is not None:
            logger.info('[rs-adabn] %d of %d BN layers take %d knots (%s); the rest one knot'
                        % (sum(len(v) > 1 for v in lk.values()), len(lk), len(knots), layers))
        K = len(knots)
        self.model.eval()
        st, acc = {}, {}
        for n, m in self.layers:
            C = m.num_features
            dev = m.running_mean.device
            K = len(lk[n])
            if K == 1:
                o, r = torch.zeros(1, C, dtype=torch.float64, device=dev), torch.ones(1, C, dtype=torch.float64, device=dev)
            else:
                o, r = self.src[n]['o'].to(dev), self.src[n]['r'].to(dev)
            st[n] = dict(mean=torch.zeros(K, C, dtype=torch.float64, device=dev), var=torch.ones(K, C, dtype=torch.float64, device=dev),
                         cnt=torch.zeros(K, dtype=torch.float64, device=dev), o=o, r=r, knots=lk[n])
        K = len(knots)

        def make(name, bn):
            def forward(x):
                xf, back = _flat(x)
                rho = self.ranges(x)
                e = st[name]
                w = knot_weights(rho, e['knots']).to(x.device)
                s, mean, var = weighted_moments(xf, w)
                present = s >= 2.0
                if not bool(present.all()):                       # a knot this batch barely touches: pooled moments
                    _, pm, pv = weighted_moments(xf, torch.ones(xf.shape[0], 1, device=x.device))
                    mean = torch.where(present[:, None], mean, pm)
                    var = torch.where(present[:, None], var, pv)
                e['cnt'] += present.double()
                c = e['cnt'].clamp_min(1.0)[:, None]
                unb = var * (s / (s - 1.0).clamp_min(1e-12))[:, None]
                e['mean'] = torch.where(present[:, None], e['mean'] + (mean - e['mean']) / c, e['mean'])
                e['var'] = torch.where(present[:, None], e['var'] + (unb - e['var']) / c, e['var'])
                if diag_knots is not None:
                    _accumulate(acc, name, xf, knot_weights(rho, diag_knots).to(x.device))
                return back(_transform(xf, w, mean, var, e['o'], e['r'], bn).to(x.dtype))
            return forward
        self._install(make)
        try:
            self._run(loader, logger, 'target (%d knots)' % K, to_gpu)
        finally:
            self._uninstall()
        missing = {n: int((e['cnt'] == 0).sum()) for n, e in st.items() if bool((e['cnt'] == 0).any())}
        if missing and logger is not None:
            logger.info('[rs-adabn] knots never seen on the target (pooled fallback at apply): %s' % missing)
        self.tgt = (knots, st)
        for n, m in self.layers:                                  # pooled running statistics, as AdaBN would leave them
            e = st[n]
            if len(e['knots']) == 1:
                m.running_mean.copy_(e['mean'][0].to(m.running_mean.dtype))
                m.running_var.copy_(e['var'][0].to(m.running_var.dtype))
        diag = None
        if diag_knots is not None:
            diag = {n: _finish(acc[n], bn_eps(m)) for n, m in self.layers}
        return diag

    def apply(self):
        """Evaluation with the estimated per-knot statistics (instance forward overrides)."""
        knots, st = self.tgt

        def make(name, bn):
            e = st[name]
            seen = e['cnt'] > 0
            if not bool(seen.all()):
                pooled_m, pooled_v = e['mean'][seen].mean(0), e['var'][seen].mean(0)
                e = dict(e, mean=torch.where(seen[:, None], e['mean'], pooled_m), var=torch.where(seen[:, None], e['var'], pooled_v))

            def forward(x):
                xf, back = _flat(x)
                w = knot_weights(self.ranges(x), e['knots']).to(x.device)
                return back(_transform(xf, w, e['mean'], e['var'], e['o'], e['r'], bn).to(x.dtype))
            return forward
        self.model.eval()
        self._install(make)

    def restore(self):
        """Remove the forward overrides and put the saved statistics back (keep the hooks)."""
        self._uninstall()
        for n, m in self.layers:
            mean, var, tracked, momentum = self.saved[n]
            m.running_mean.copy_(mean)
            m.running_var.copy_(var)
            m.num_batches_tracked.copy_(tracked)
            m.momentum = momentum

    def close(self):
        self.restore()
        for h in self.handles:
            h.remove()


def bn_eps(m):
    return float(m.eps)


def _accumulate(acc, name, xf, w):
    x = xf.double()
    w = w.double()
    a = acc.setdefault(name, dict(s=0.0, M=0.0, Q=0.0, n=0, m=0.0, q=0.0))
    a['s'] = a['s'] + w.sum(0)
    a['M'] = a['M'] + w.t() @ x
    a['Q'] = a['Q'] + w.t() @ (x * x)
    a['n'] += x.shape[0]
    a['m'] = a['m'] + x.sum(0)
    a['q'] = a['q'] + (x * x).sum(0)


def _finish(a, eps):
    """Per knot: weight share, mean, var; pooled mean / var; o = (mu_k - mu) / sigma, r = sigma_k / sigma (eps inside)."""
    s = a['s']
    mu_k = a['M'] / s.clamp_min(1e-12)[:, None]
    var_k = (a['Q'] / s.clamp_min(1e-12)[:, None] - mu_k * mu_k).clamp_min(0.0)
    mu = a['m'] / a['n']
    var = (a['q'] / a['n'] - mu * mu).clamp_min(0.0)
    sd = (var + eps).sqrt()
    o = (mu_k - mu) / sd
    r = (var_k + eps).sqrt() / sd
    thin = s < 100.0                                                   # too few elements to say anything: neutral
    o = torch.where(thin[:, None], torch.zeros_like(o), o)
    r = torch.where(thin[:, None], torch.ones_like(r), r)
    return dict(share=(s / s.sum()), s=s, mean_k=mu_k, var_k=var_k, mean=mu, var=var, o=o, r=r, n=a['n'])


def _transform(xf, w, mean, var, o, r, bn):
    """gamma * [(x - mu(rho)) / sqrt(var(rho) + eps) * r(rho) + o(rho)] + beta with knot statistics interpolated by `w`."""
    wd = w.to(mean.dtype)
    mu = wd @ mean
    sd = (wd @ var + bn.eps).sqrt()
    y = (xf.to(mean.dtype) - mu) / sd * (wd @ r) + wd @ o
    if bn.affine:
        y = y * bn.weight.to(y.dtype) + bn.bias.to(y.dtype)
    return y


def mismatch(src, tgt):
    """Stage 0: per knot, mean over channels of |o_t - o_s| and |log r_t - log r_s|, and the weight shares."""
    return dict(share_src=src['share'].tolist(), share_tgt=tgt['share'].tolist(),
                d_mean=(tgt['o'] - src['o']).abs().mean(1).tolist(),
                d_logsd=(tgt['r'].log() - src['r'].log()).abs().mean(1).tolist())


class RangeAffine(torch.nn.Module):
    """Per-knot, per-channel modulation parameters a, b (K, C), zero at start (item 58 arm R). A leaf module, so the family's
    optimizer (which collects leaf modules) finds its parameters."""

    def __init__(self, knots, channels, device=None):
        super().__init__()
        self.knots = tuple(float(k) for k in knots)
        self.a = torch.nn.Parameter(torch.zeros(len(self.knots), channels, device=device))
        self.b = torch.nn.Parameter(torch.zeros(len(self.knots), channels, device=device))


class RangeAffineBN(torch.nn.Module):
    """y' = BN(x) * (1 + w(rho) a) + w(rho) b, with w the piecewise-linear knot weights of each element's x-y range (item 58
    arm R, ANALYSIS). Starts exactly at BN (a = b = 0)."""

    def __init__(self, bn, pos, knots):
        super().__init__()
        self.bn = bn
        self.mod = RangeAffine(knots, bn.num_features, device=bn.weight.device)
        self._pos = [pos]                                    # not a submodule

    def forward(self, x):
        y = self.bn(x)
        pos = self._pos[0]
        if x.dim() == 4:
            rho = pos.bev(x.shape[2], x.shape[3], x.shape[0], x.device)
        else:
            rho = pos.voxels(x.shape[0])
        w = knot_weights(rho, self.mod.knots).to(device=x.device, dtype=y.dtype)
        a, b = w @ self.mod.a, w @ self.mod.b                 # (n, C)
        if y.dim() == 4:
            B, C, H, W = y.shape
            a = a.reshape(B, H, W, C).permute(0, 3, 1, 2)
            b = b.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return y * (1.0 + a) + b


def install_range_affine(model, knots, pc_range):
    """Wrap every BatchNorm1d / BatchNorm2d of `model` in RangeAffineBN (in place, after the weights are loaded). Registers the
    sparse-convolution hooks that give voxel positions. Returns the number of wrapped layers."""
    import spconv.pytorch as spconv
    pos = _Pos(pc_range)
    model._range_affine_hooks = [m.register_forward_hook(pos.hook) for m in model.modules()
                                 if isinstance(m, spconv.conv.SparseConvolution)]
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    for name, bn in targets:
        parent = model
        *path, leaf = name.split('.')
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, RangeAffineBN(bn, pos, knots))
    return len(targets)
