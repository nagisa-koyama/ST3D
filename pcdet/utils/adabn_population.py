"""AdaBN whose statistics are taken over a SUBPOPULATION of each BatchNorm layer's elements (experiments_md 20261003_05 §5
item 62, design fixed 2026-11-27 12:53 JST, before any number). ANALYSIS.

Why: per channel, BN in eval mode is one affine map, so item 58's labelled change of gamma / beta (G) equals some change of the
statistics under P_K's gamma / beta (59a, corrected). Global AdaBN (the target's moments over all elements) does not give it.
Item 62 asks whether the moments of a subset of the target's elements do: far elements, elements under the teacher's boxes,
or (labelled reference) elements under the GT boxes.

Every BatchNorm1d (sparse voxels) and BatchNorm2d (BEV cells) is re-estimated over the loader as
`adabn_utils.reestimate_bn` does (momentum None: the mean of the batch means and of the unbiased batch variances; each batch
normalized with its own biased moments on the way through), except that each layer's batch moments are taken over the
elements `member(kind, b, x, y, ctx)` selects; every element is normalized with them. Element positions: a voxel's centre from
the last sparse convolution's indices, a BEV cell's centre (as pcdet/utils/adabn_range._Pos). A layer and batch with fewer
than `min_members` members is normalized with its pooled moments and not counted; a layer never counted keeps its previous
statistics. With every element a member the result equals `adabn_utils.reestimate_bn` (tests/test_adabn_population.py).
"""
import math

import torch

from pcdet.utils import adabn_utils


class Elements:
    """x-y centres (and frame index) of a BN input's elements, from the map size or the last sparse convolution's output."""

    def __init__(self, pc_range):
        self.x0, self.y0, _, self.x1, self.y1, _ = [float(v) for v in pc_range]
        self.last = None

    def hook(self, module, inputs, output):
        self.last = (output.indices, list(output.spatial_shape))

    def bev(self, b, h, w, device):
        xs = self.x0 + (torch.arange(w, dtype=torch.float32, device=device) + 0.5) * (self.x1 - self.x0) / w
        ys = self.y0 + (torch.arange(h, dtype=torch.float32, device=device) + 0.5) * (self.y1 - self.y0) / h
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        bi = torch.arange(b, device=device)[:, None, None].expand(b, h, w)
        return bi.reshape(-1), xx.reshape(-1).repeat(b), yy.reshape(-1).repeat(b)

    def voxels(self, n):
        if self.last is None or self.last[0].shape[0] != n:
            raise RuntimeError('BatchNorm1d input of %d rows does not follow a recorded sparse convolution output' % n)
        idx, shape = self.last
        _, ny, nx = shape[-3:]
        x = self.x0 + (idx[:, 3].float() + 0.5) * (self.x1 - self.x0) / nx
        y = self.y0 + (idx[:, 2].float() + 0.5) * (self.y1 - self.y0) / ny
        return idx[:, 0].long(), x, y


def in_footprints(x, y, boxes):
    """(n,) bool: (x, y) inside the x-y footprint of any box (M, >= 7: cx, cy, cz, dx, dy, dz, heading)."""
    if boxes is None or boxes.shape[0] == 0:
        return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    b = boxes.to(device=x.device, dtype=torch.float32)
    dx = x[:, None] - b[None, :, 0]
    dy = y[:, None] - b[None, :, 1]
    c, s = torch.cos(b[:, 6]), torch.sin(b[:, 6])
    lx = dx * c + dy * s                                  # rotate by -heading into the box frame
    ly = -dx * s + dy * c
    return ((lx.abs() <= b[:, 3] / 2) & (ly.abs() <= b[:, 4] / 2)).any(1)


def member_all(kind, bi, x, y, ctx):
    return torch.ones(x.shape[0], dtype=torch.bool, device=x.device)


def member_far(r0):
    def f(kind, bi, x, y, ctx):
        return torch.hypot(x, y) > r0
    return f


def member_boxes(kind, bi, x, y, ctx):
    """Inside a footprint of `ctx['boxes'][frame]`, per frame of the batch."""
    out = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for f, boxes in enumerate(ctx['boxes']):
        sel = bi == f
        if bool(sel.any()):
            out[sel] = in_footprints(x[sel], y[sel], boxes)
    return out


def gt_boxes_of(batch):
    """Per frame, the real GT boxes of an eval-mode batch (padding rows have zero size)."""
    g = batch['gt_boxes']
    g = torch.as_tensor(g) if not torch.is_tensor(g) else g
    return [b[(b[:, 3:6] > 0).all(1)] for b in g]


def teacher_boxes(teacher, batch, cuts):
    """Per frame, the teacher's boxes at or above its class's cut (`cuts`: {1-based label: score})."""
    with torch.no_grad():
        pred, _ = teacher(dict(batch))                    # a shallow copy: the teacher's keys never reach the model's pass
    out = []
    for p in pred:
        th = torch.tensor([cuts.get(int(c), float('inf')) for c in p['pred_labels']], device=p['pred_scores'].device)
        out.append(p['pred_boxes'][p['pred_scores'] >= th])
    return out


@torch.no_grad()
def reestimate_bn_population(model, loader, member, pc_range, logger=None, to_gpu=True, min_members=32, boxes_fn=None):
    """Returns (saved statistics, diagnostics per layer: batches counted / skipped, mean member share; frames' box counts)."""
    import spconv.pytorch as spconv
    if to_gpu:
        from pcdet.models import load_data_to_gpu
    layers = adabn_utils.bn_layers(model)
    if any(adabn_utils._is_ds(m) for _, m in layers):
        raise NotImplementedError('per-domain BN is not supported')
    el = Elements(pc_range)
    hooks = [m.register_forward_hook(el.hook) for m in model.modules() if isinstance(m, spconv.conv.SparseConvolution)]
    saved = {n: (m.running_mean.clone(), m.running_var.clone()) for n, m in layers}
    momenta = {n: m.momentum for n, m in layers}
    tracked = {n: m.num_batches_tracked.clone() for n, m in layers}
    diag = {n: dict(counted=0, skipped=0, share=0.0) for n, _ in layers}
    ctx = {'boxes': None}
    box_counts = []
    was_training = model.training
    model.eval()

    def patched(name, bn):
        def forward(x):
            if x.dim() == 4:
                B, C, H, W = x.shape
                bi, xs, ys = el.bev(B, H, W, x.device)
                xf = x.permute(0, 2, 3, 1).reshape(-1, C)
            else:
                bi, xs, ys = el.voxels(x.shape[0])
                xf = x
            mask = member('bev' if x.dim() == 4 else 'voxel', bi, xs, ys, ctx)
            k = int(mask.sum())
            d = diag[name]
            d['share'] += k / max(xf.shape[0], 1)
            if k >= min_members:
                xm = xf[mask]
                mean, var = xm.mean(0), xm.var(0, unbiased=False)
                d['counted'] += 1
                bn.running_mean += (mean - bn.running_mean) / d['counted']
                bn.running_var += (var * k / max(k - 1, 1) - bn.running_var) / d['counted']
            else:
                mean, var = xf.mean(0), xf.var(0, unbiased=False)
                d['skipped'] += 1
            y = (xf - mean) / torch.sqrt(var + bn.eps)
            if bn.affine:
                y = y * bn.weight + bn.bias
            if x.dim() == 4:
                y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return y
        return forward

    for n, m in layers:
        m.momentum = None
        m.forward = patched(n, m)
    try:
        for i, batch in enumerate(loader):
            if to_gpu:
                load_data_to_gpu(batch)
            if boxes_fn is not None:
                ctx['boxes'] = boxes_fn(batch)
                box_counts.extend(int(b.shape[0]) for b in ctx['boxes'])
            model(batch)
            if logger is not None and i % 50 == 0:
                logger.info('[pop-adabn] statistics pass %d/%d' % (i + 1, len(loader)))
    finally:
        for h in hooks:
            h.remove()
        for n, m in layers:
            if 'forward' in m.__dict__:
                del m.forward
            m.momentum = momenta[n]
            m.num_batches_tracked.copy_(tracked[n])
            if diag[n]['counted'] == 0:                    # never counted: keep the previous statistics
                m.running_mean.copy_(saved[n][0])
                m.running_var.copy_(saved[n][1])
        model.train(was_training)
    total = max(len(loader), 1)
    for n in diag:
        diag[n]['share'] /= total
    never = [n for n in diag if diag[n]['counted'] == 0]
    if logger is not None:
        logger.info('[pop-adabn] %d layers; never counted (kept previous): %d; mean member share %.4f; boxes per frame %s'
                    % (len(layers), len(never), sum(d['share'] for d in diag.values()) / max(len(diag), 1),
                       ('%.2f' % (sum(box_counts) / len(box_counts))) if box_counts else '-'))
    return saved, dict(layers=diag, never=never, boxes_per_frame=(sum(box_counts) / len(box_counts)) if box_counts else None)
