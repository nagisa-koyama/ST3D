"""AdaBN whose BEV statistics are taken only inside the target sensor's field of view.

Why (experiments_md T05 §5 item 45; 20261004_01 §2.5, 20261005_01 §9): on PandaSet spin -> flash, test-time
AdaBN collapses the source-only model (rule A 70.32 -> 12.20), while re-estimating the 3D backbone alone
lifts it (73.22). The 3D backbone's BatchNorm runs on occupied voxels only; every BatchNorm2d of the 2D
backbone and the head averages over the whole BEV map. PandarGT fills a +-30 deg cone, so most cells of a
target map are empty, and its 2D statistics describe the share of empty cells, not the content. Here each
BatchNorm2d takes its statistics over the cells whose centre lies in the cone; every other BatchNorm is
re-estimated exactly as `adabn_utils.reestimate_bn` does.

UDA-legal: the cone is the target sensor's published field of view, the statistics come from unlabelled
target TRAIN clouds, and no label is read.
"""
import math

import torch

from pcdet.utils import adabn_utils


def bev_cone_mask(h, w, pc_range, fov_degree, heading_degree=0.0, device=None):
    """(h, w) bool tensor over a BEV map laid out as spconv's dense() lays it out: rows follow y, columns x,
    both spanning `pc_range`. True where the cell centre passes DatasetTemplate.extract_fov_gt's test: a
    cone of full angle `fov_degree` about `heading_degree` (0 = +x, increasing clockwise), apex at the
    origin, forward half only."""
    x0, y0, _, x1, y1, _ = [float(v) for v in pc_range]
    xs = x0 + (torch.arange(w, dtype=torch.float64) + 0.5) * (x1 - x0) / w
    ys = y0 + (torch.arange(h, dtype=torch.float64) + 0.5) * (y1 - y0) / h
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    a = -math.radians(heading_degree)  # extract_fov_gt rotates by -heading with rotate_points_along_z
    xr = xx * math.cos(a) - yy * math.sin(a)
    yr = xx * math.sin(a) + yy * math.cos(a)
    half = math.radians(fov_degree) / 2.0
    mask = (torch.atan2(yr, xr).abs() < half) & (xr > 0)
    return mask.to(device) if device is not None else mask


class _MaskCache:
    def __init__(self, pc_range, fov_degree, heading_degree):
        self.args = (pc_range, fov_degree, heading_degree)
        self.cache = {}

    def __call__(self, h, w, device):
        key = (h, w, str(device))
        if key not in self.cache:
            self.cache[key] = bev_cone_mask(h, w, *self.args, device=device)
        return self.cache[key]


@torch.no_grad()
def reestimate_bn_bev_masked(model, loader, mask_fn, logger=None, to_gpu=True):
    """Re-estimate every BatchNorm layer on `loader` (MIX 1), each BatchNorm2d over the cells `mask_fn(h, w,
    device)` selects only. The accumulation is torch's momentum=None one: running_mean is the mean of the batch
    means and running_var the mean of the unbiased batch variances, and each batch is normalised with its own
    (biased) statistics on the way through, as in train mode. With an all-True mask the result equals
    `adabn_utils.reestimate_bn` (tests/test_adabn_masked.py). Returns (saved statistics, diagnostics): per
    BatchNorm2d, the share of the first batch's |activation| that lies inside the mask and the mask's share
    of cells - the check that the mask covers the target's content.
    """
    if to_gpu:
        from pcdet.models import load_data_to_gpu
    layers = adabn_utils.bn_layers(model)
    if any(adabn_utils._is_ds(m) for _, m in layers):
        raise NotImplementedError('per-domain BN is not supported by the masked re-estimation')
    saved = {n: tuple(t.clone() for t in adabn_utils._stats(m)) for n, m in layers}
    momenta = {n: m.momentum for n, m in layers}
    tracked = {n: m.num_batches_tracked.clone() for n, m in layers}
    was_training = model.training
    model.eval()
    count, diag = {}, {}

    def masked_forward(name, bn):
        def forward(x):
            mask = mask_fn(x.shape[-2], x.shape[-1], x.device)
            if not bool(mask.any()):
                raise ValueError(f'{name}: the BEV mask selects no cell of a {tuple(x.shape)} map')
            c = x.shape[1]
            xm = x.transpose(0, 1)[:, :, mask].reshape(c, -1)  # (C, N * cells in mask)
            k = xm.shape[1]
            mean = xm.mean(1)
            var = xm.var(1, unbiased=False)
            count[name] += 1
            bn.running_mean += (mean - bn.running_mean) / count[name]
            bn.running_var += (var * k / max(k - 1, 1) - bn.running_var) / count[name]
            if name not in diag:
                a = x.abs().sum((0, 1))
                diag[name] = dict(content_in_mask=float(a[mask].sum() / a.sum().clamp_min(1e-12)),
                                  mask_share=float(mask.float().mean()), map=[int(x.shape[-2]), int(x.shape[-1])])
            y = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + bn.eps)
            if bn.affine:
                y = y * bn.weight[None, :, None, None] + bn.bias[None, :, None, None]
            return y
        return forward

    for n, m in layers:
        m.running_mean.zero_()
        m.running_var.fill_(1)
        m.num_batches_tracked.zero_()
        m.momentum = None
        if isinstance(m, torch.nn.BatchNorm2d):
            count[n] = 0
            m.forward = masked_forward(n, m)
        else:
            m.train()
    try:
        for i, batch in enumerate(loader):
            if to_gpu:
                load_data_to_gpu(batch)
            model(batch)
            if logger is not None and i % 50 == 0:
                logger.info('[adabn] statistics pass %d/%d (BEV mask)' % (i + 1, len(loader)))
    finally:
        for n, m in layers:
            if 'forward' in m.__dict__:
                del m.forward
            m.momentum = momenta[n]
            m.num_batches_tracked.copy_(tracked[n])
        model.train(was_training)
    if logger is not None:
        for n, d in diag.items():
            logger.info('[adabn] BEV mask %-40s map %s mask share %.3f |activation| in mask %.4f'
                        % (n, d['map'], d['mask_share'], d['content_in_mask']))
    return saved, diag


def cone_mask_fn(pc_range, fov_degree, heading_degree=0.0):
    return _MaskCache(pc_range, fov_degree, heading_degree)
