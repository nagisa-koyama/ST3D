"""Training and evaluation loop for pcseg (experiments_md episodes/20261116_01 §5 and §8, step S1).

Pointcept's SpUNet-v1m1 nuScenes recipe (semseg-spunet-v1m1-0-base): AdamW, OneCycleLR stepped per iteration
(pct_start 0.04, cosine, div 10, final div 100, torch's default momentum cycling), cross-entropy with IGNORE, AMP.

Evaluation scores ORIGINAL points: each point takes the prediction of its voxel through the grid-sampling inverse map
(Pointcept's in-training evaluator does a 1-NN query to the sampled points instead; the voxel's representative is
that point's nearest sample in all but boundary cases). A voxel-level matrix and, optionally, a cone-restricted
point-level matrix are kept beside it. Under DDP each rank scores a disjoint shard (no padding, so no frame is
counted twice) and the matrices are summed.
"""
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from . import IGNORE
from .losses_metrics import ce_loss, ConfusionMatrix

MODEL_KEYS = ('grid_coord', 'feat', 'offset', 'segment')


# --- distributed helpers --------------------------------------------------------------------------------------------

def dist_on():
    return dist.is_available() and dist.is_initialized()


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist_on() else (0, 1)


def all_reduce_sum(x):
    if dist_on():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


class ShardSampler(torch.utils.data.Sampler):
    """Evaluation sampler: rank r takes indices r, r + W, r + 2W, ... Unlike DistributedSampler it never pads, so
    every frame is scored exactly once across ranks."""

    def __init__(self, n, rank, world):
        self.idx = list(range(rank, n, world))

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


def convert_sync_bn(model, opt_cfg, world):
    """OPTIMIZATION.SYNC_BN: at world > 1, every BatchNorm becomes a SyncBatchNorm, so the batch statistics are taken
    over the GLOBAL batch, as on one GPU (20261116_01 §8.3). Call it before DistributedDataParallel.

    Returns (model, number of layers converted). At one rank it returns the model unchanged: there torch's
    SyncBatchNorm would fall back to F.batch_norm anyway, and the one-GPU rows (G0 33189, the oracles 33191 / 33192)
    keep the code path they ran. The state_dict keys are those of BatchNorm, so checkpoints load either way."""
    if world <= 1 or not opt_cfg.get('SYNC_BN', False):
        return model, 0
    n = sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model), n


# --- optimisation ---------------------------------------------------------------------------------------------------

def build_optimizer(model, opt_cfg):
    return torch.optim.AdamW(model.parameters(), lr=float(opt_cfg['LR']), weight_decay=float(opt_cfg['WEIGHT_DECAY']))


def build_scheduler(optimizer, opt_cfg, total_steps):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(opt_cfg['LR']), total_steps=int(total_steps),
        pct_start=float(opt_cfg.get('PCT_START', 0.04)), anneal_strategy='cos',
        div_factor=float(opt_cfg.get('DIV_FACTOR', 10.0)), final_div_factor=float(opt_cfg.get('FINAL_DIV_FACTOR', 100.0)))


def to_device(batch, device, keys=MODEL_KEYS):
    return {k: (batch[k].to(device, non_blocking=True) if k in keys else batch[k]) for k in batch}


def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch, device, amp=True, clip_grad=None,
                    log_every=50, log_fn=None, logger=None, max_iters=None):
    """One pass over `loader`. Returns {loss, iters, data_s, step_s, frames}. Raises on a non-finite loss."""
    model.train()
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    t_data = t_step = 0.0
    loss_sum, n_it, n_frames = 0.0, 0, 0
    end = time.time()
    for it, batch in enumerate(loader):
        if max_iters is not None and it >= max_iters:
            break
        batch = to_device(batch, device)
        t0 = time.time()
        t_data += t0 - end
        with torch.autocast('cuda', enabled=amp):
            logits = model(batch)
        loss = ce_loss(logits.float(), batch['segment'])
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite loss %s at epoch %d iteration %d (%s)'
                                     % (float(loss), epoch, it, batch['frame_key']))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        lv = float(loss)                       # synchronises, so t_step is the real step time
        end = time.time()
        t_step += end - t0
        loss_sum += lv
        n_it += 1
        n_frames += len(batch['frame_key'])
        if log_every and it % log_every == 0:
            lr = optimizer.param_groups[0]['lr']
            if logger is not None:
                logger.info('epoch %d iter %d/%d loss %.4f lr %.6f data %.3fs step %.3fs'
                            % (epoch, it, len(loader), lv, lr, t_data / n_it, t_step / n_it))
            if log_fn is not None:
                log_fn({'train/loss': lv, 'train/lr': lr, 'train/epoch': epoch})
    stats = torch.tensor([loss_sum, n_it, t_data, t_step, n_frames], dtype=torch.float64, device=device)
    all_reduce_sum(stats)
    _, world = rank_world()
    return {'loss': float(stats[0] / max(stats[1], 1)), 'iters': int(stats[1] / world),
            'data_s': float(stats[2] / world), 'step_s': float(stats[3] / world), 'frames': int(stats[4])}


# --- evaluation -----------------------------------------------------------------------------------------------------

def cone_mask(xy, fov_deg, heading_deg=0.0):
    """Points inside a forward cone, in ST3D's convention (DatasetTemplate.extract_fov_gt: heading counted clockwise
    from +x, |azimuth| < fov / 2 after rotating by -heading, and x > 0 there). xy must be about the SENSOR's xy
    origin (SHIFT_COOR's xy part removed)."""
    h = -np.deg2rad(heading_deg)
    c, s = np.cos(h), np.sin(h)
    x = xy[:, 0] * c - xy[:, 1] * s
    y = xy[:, 0] * s + xy[:, 1] * c
    return (np.abs(np.arctan2(y, x)) < np.deg2rad(fov_deg) / 2) & (x > 0)


def metrics_from_matrix(mat, class_names, prefix):
    cm = ConfusionMatrix(len(class_names))
    cm.mat = mat
    iou = cm.iou()
    gt = mat.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        recall = np.where(gt > 0, np.diag(mat) / gt, np.nan)
    out = {prefix + '/miou': cm.miou(require_gt=True),
           prefix + '/macc': float(np.nanmean(recall)) if (gt > 0).any() else float('nan'),
           prefix + '/allacc': float(np.diag(mat).sum() / max(mat.sum(), 1)),
           prefix + '/points': int(mat.sum())}
    for c, name in enumerate(class_names):
        out['%s/iou/%s' % (prefix, name)] = float(iou[c])
    out[prefix + '/classes_without_gt'] = [n for n, g in zip(class_names, gt) if g == 0]
    return out


@torch.no_grad()
def evaluate(model, loader, class_names, device, shift=(0.0, 0.0, 0.0), cone=None):
    """Score `loader` (an eval PandasetSeg loader: samples carry inverse / origin_segment / origin_coord).

    Always in fp32, never under autocast, as Pointcept's evaluator and tester do (`with torch.no_grad()` only). Under
    autocast, spconv 2.3.8 in eval mode crashes the SpUNet (smoke 33127, probe 33180): the 5^3 stem (kernel volume
    125 > 32) uses the Native algorithm, whose autograd Function casts its inputs to fp16, so the stem emits fp16
    features; the next conv uses implicit GEMM, which in eval mode calls ops.implicit_gemm directly with the fp32
    weights uncast, and the tuner finds no fp16-feature / fp32-weight kernel ("can't find suitable algorithm for 0").
    In training both kinds go through autograd Functions that cast, which is why training ran.

    cone: None or (fov_deg, heading_deg); adds a point-level matrix restricted to that cone.
    Returns (metrics dict, {'point', 'voxel'[, 'cone']} summed confusion matrices as int64 numpy)."""
    model.eval()
    n = len(class_names)
    mats = {'point': ConfusionMatrix(n), 'voxel': ConfusionMatrix(n)}
    if cone is not None:
        mats['cone'] = ConfusionMatrix(n)
    shift = np.asarray(shift, dtype=np.float32)
    n_frames = 0
    for batch in loader:
        b = to_device(batch, device)
        logits = model(b)
        pred_vox = logits.float().argmax(1)
        pred_pt = pred_vox[b['inverse'].to(device)].cpu().numpy()
        mats['voxel'].update(pred_vox.cpu().numpy(), batch['segment'].numpy())
        seg = batch['origin_segment'].numpy()
        mats['point'].update(pred_pt, seg)
        if cone is not None:
            m = cone_mask(batch['origin_coord'].numpy()[:, :2] - shift[:2], *cone)
            mats['cone'].update(pred_pt[m], seg[m])
        n_frames += len(batch['frame_key'])
    out_mats = {}
    for k, cm in mats.items():
        t = torch.as_tensor(cm.mat, dtype=torch.int64, device=device)
        out_mats[k] = all_reduce_sum(t).cpu().numpy()
    nf = torch.tensor([n_frames], dtype=torch.int64, device=device)
    metrics = {'val/frames': int(all_reduce_sum(nf))}
    for k, mat in out_mats.items():
        metrics.update(metrics_from_matrix(mat, class_names, 'val_' + k))
    return metrics, out_mats


# --- target dump (bug_prevention_checklist rule 4) ------------------------------------------------------------------

def target_dump(datasets, class_names, n_frames=4):
    """For a few frames of each dataset: per-class POINT counts (after the class map) and VOXEL counts (the labels the
    loss sees), the ignored share, and, for eval samples, how many points get their own label back through the
    inverse map. Raises if a voxel carries a label its frame's points do not, or a label outside [IGNORE, C)."""
    n = len(class_names)
    out = {}
    for name, ds in datasets.items():
        idxs = np.linspace(0, len(ds) - 1, num=min(n_frames, len(ds))).round().astype(int)
        rows = []
        for i in idxs:
            _, _, seg, key = ds.load(int(i))
            s = ds[int(i)]
            vox = np.asarray(s['segment'])
            for arr in (seg, vox):
                if arr.min() < IGNORE or arr.max() >= n:
                    raise ValueError('%s %s: label outside [%d, %d)' % (name, key, IGNORE, n))
            if not set(np.unique(vox)) <= set(np.unique(seg)):
                raise ValueError('%s %s: voxel labels %s not among point labels' % (name, key, np.unique(vox)))
            row = {'frame': key, 'points': int(len(seg)), 'voxels': int(len(vox)),
                   'ignored_points': int((seg == IGNORE).sum()), 'ignored_voxels': int((vox == IGNORE).sum()),
                   'point_counts': {class_names[c]: int((seg == c).sum()) for c in range(n) if (seg == c).any()},
                   'voxel_counts': {class_names[c]: int((vox == c).sum()) for c in range(n) if (vox == c).any()}}
            if 'inverse' in s:
                row['inverse_agreement'] = float((vox[s['inverse']] == seg).mean())
            rows.append(row)
        out[name] = rows
    return out


def format_dump(dump):
    lines = []
    for name, rows in dump.items():
        for r in rows:
            lines.append('%s %s: %d points -> %d voxels; ignored %d points / %d voxels%s'
                         % (name, r['frame'], r['points'], r['voxels'], r['ignored_points'], r['ignored_voxels'],
                            '; inverse agreement %.4f' % r['inverse_agreement'] if 'inverse_agreement' in r else ''))
            for c in r['point_counts']:
                lines.append('    %-16s points %8d  voxels %8d' % (c, r['point_counts'][c], r['voxel_counts'].get(c, 0)))
    return '\n'.join(lines)


# --- checkpoints ----------------------------------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, extra=None):
    m = model.module if hasattr(model, 'module') else model
    state = {'epoch': epoch, 'model_state': {k: v.cpu() for k, v in m.state_dict().items()},
             'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
             'scaler_state': scaler.state_dict()}
    if extra:
        state.update(extra)
    tmp = str(path) + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location='cpu'):
    """Restores every state given; returns the epoch the checkpoint finished (training resumes at epoch + 1)."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    m = model.module if hasattr(model, 'module') else model
    m.load_state_dict(state['model_state'])
    for obj, key in ((optimizer, 'optimizer_state'), (scheduler, 'scheduler_state'), (scaler, 'scaler_state')):
        if obj is not None:
            obj.load_state_dict(state[key])
    return int(state['epoch'])


def prune_checkpoints(ckpt_dir, keep):
    ck = sorted(ckpt_dir.glob('checkpoint_epoch_*.pth'), key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    for p in ck[:-keep] if keep > 0 else []:
        p.unlink()


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
