"""pcseg SYNC_BN / GLOBAL_LOSS_MEAN equivalence check (experiments_md episodes/20261116_01 §8.3; user 2026-11-19: remove
the dependence on the GPU count). --backend gloo: both ranks on ONE GPU (job 33824); --backend nccl: rank r on cuda:r,
the two-GPU path (the DDP smoke, §8.4).

Question: does a two-rank run with OPTIMIZATION.SYNC_BN compute what a one-rank run computes on the same global batch?

Four train frames (fixed indices, augmented once with a fixed seed in this process) go through the real SpUNet at one
initialisation, in four arms:
  ref     one process, all four frames, plain BatchNorm: what a one-GPU row does
  ref2    the same again: the GPU's own run-to-run floor (spconv is not bitwise reproducible, 20261116_01 §5)
  sync    two processes on the SAME GPU over gloo (torch 2.5's SyncBatchNorm has a gloo path), frames {0,1} and {2,3},
          engine.convert_sync_bn + DistributedDataParallel exactly as train_seg.py
  nosync  the same without SYNC_BN: what a two-GPU row computed before ST3D's SYNC_BN commit
Each arm runs in fp32 and under autocast (AMP, as training). Every BN momentum is set to 1, so after a forward the
running buffers ARE that forward's batch mean and unbiased variance. Recorded per arm: those statistics for every BN
layer, the logits of all four frames, and the parameter gradients of two losses:
  'rank'   each rank's mean CE (engine.train_loss without GLOBAL_LOSS_MEAN), averaged over ranks by DDP;
  'global' sum CE / global point count, times the world size (DDP averages): the one-rank mean, exactly. Up to job
           33824 (ST3D 8d044d3) the script computed it with its own formula; since GLOBAL_LOSS_MEAN (411962e) it calls
           engine.train_loss(global_mean=True), which all-reduces the count, i.e. the production path.
Comparisons use rel(a, ref) = max|a - ref| / max|ref| per tensor, then the maximum over tensors.

Readings (pre-declared in 20261116_01 §8.3 before the job):
  R1 fp32 BN statistics, sync vs ref, both ranks:          <= max(1e-4, 10 x ref2 floor)
  R2 fp32 logits, sync vs ref, all four frames:             <= max(1e-3, 10 x ref2 floor)
  R3 fp32 'global' gradients, sync vs ref:                  <= max(1e-3, 10 x ref2 floor)
  R4 power: nosync BN statistics vs ref >= 100 x R1's value, or the check cannot tell the arms apart and is void
  R5 AMP, sync vs ref on BN statistics and logits:          <= max(1e-2, 10 x the AMP ref2 floor); everything finite
  Not a reading: the 'rank' gradients' difference from ref, i.e. the loss-weighting dependence left after SYNC_BN.
PASS = R1-R3 and R5 hold and R4 is met.

Run from ST3D/tools in the container on a 1-GPU (gloo) or 2-GPU (nccl) allocation:
  python analysis/pcseg_syncbn_check.py --cfg cfgs/pcseg/spunet-nuscenes.yaml --backend gloo \
      --out /home/koyama/data/pcseg/syncbn_check/<job>
(inside the container the data tree is /storage; --out is written as given).
"""
import argparse
import json
import os
import shutil
import socket
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/
import _init_path  # noqa: F401,E402
import train_seg  # noqa: E402
from pcseg import engine  # noqa: E402
from pcseg.pandaset_seg import collate  # noqa: E402

BN = torch.nn.modules.batchnorm._BatchNorm


def bn_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, BN)]


def run_arm(model, batch, amp):
    """Two forwards (one per loss) on `batch`; returns numpy copies of BN statistics, logits, gradients."""
    inner = model.module if hasattr(model, 'module') else model
    for _, m in bn_layers(inner):
        m.momentum = 1.0
    model.train()
    out = {}
    for kind in ('rank', 'global'):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=amp):
            logits = model(batch)
        logits = logits.float()
        seg = batch['segment']
        loss = engine.train_loss(logits, seg, global_mean=(kind == 'global'))   # = ce_loss in a lone process
        loss.backward()
        out['grad_' + kind] = {n: p.grad.detach().float().cpu().numpy().copy() for n, p in inner.named_parameters()}
        out['loss_' + kind] = float(loss)
    out['logits'] = logits.detach().cpu().numpy()
    out['bn'] = {n: (m.running_mean.detach().cpu().numpy().copy(), m.running_var.detach().cpu().numpy().copy())
                 for n, m in bn_layers(inner)}
    out['n_bn'] = len(out['bn'])
    out['sync_layers'] = sum(isinstance(m, torch.nn.SyncBatchNorm) for m in inner.modules())
    out['finite'] = bool(np.isfinite(out['logits']).all()
                         and all(np.isfinite(g).all() for g in out['grad_global'].values()))
    return out


def to_cuda(batch, dev=0):
    return engine.to_device(batch, torch.device('cuda', dev))


def child(rank, world, port, backend, cfg_file, nc, state_path, samples, sync, amp, out_path):
    os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = '127.0.0.1', str(port)
    dev = rank if backend == 'nccl' else 0
    torch.cuda.set_device(dev)
    dist.init_process_group(backend, rank=rank, world_size=world)
    model = train_seg.build_model(train_seg.load_cfg(cfg_file), nc)
    model.load_state_dict(torch.load(state_path, weights_only=True))
    model = model.to(torch.device('cuda', dev))
    model, n_sync = engine.convert_sync_bn(model, {'SYNC_BN': sync}, world)        # as train_seg.py
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev])
    per = len(samples) // world
    batch = to_cuda(collate(samples[rank * per:(rank + 1) * per]), dev)
    out = run_arm(model, batch, amp)
    out['gpu'] = torch.cuda.get_device_name(dev)
    out['n_sync_reported'] = n_sync
    out['points_valid'] = int((batch['segment'] >= 0).sum())
    if rank != 0:
        out.pop('grad_rank')
        out.pop('grad_global')        # identical on every rank after DDP's all-reduce
    torch.save(out, out_path % rank)
    dist.barrier()
    dist.destroy_process_group()


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


def rel(a, b):
    return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-12))


def rel_dict(a, b):
    """max over keys of rel(a[k], b[k]); returns (value, worst key)."""
    worst = max(((rel(a[k], b[k]), k) for k in b), key=lambda t: t[0])
    return worst


def rel_bn(a, b):
    vals = [(max(rel(a[k][0], b[k][0]), rel(a[k][1], b[k][1])), k) for k in b]
    return max(vals, key=lambda t: t[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='cfgs/pcseg/spunet-nuscenes.yaml')
    p.add_argument('--frames', type=int, nargs=4, default=[0, 7000, 14000, 21000])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--backend', choices=['gloo', 'nccl'], default='gloo')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='syncbn_')
    try:
        cfg = train_seg.load_cfg(args.cfg)
        from pcseg.class_map import build_class_map
        nc = build_class_map(cfg.DATA.CLASS_MAP).num_classes
        ds = train_seg.build_datasets(cfg)['train']
        np.random.seed(args.seed)
        samples = [ds[i] for i in args.frames]
        keys = [s['frame_key'] for s in samples]
        n_valid = [int((s['segment'] >= 0).sum()) for s in samples]
        print('frames', keys, 'valid voxels', n_valid, flush=True)
        torch.manual_seed(args.seed)
        model0 = train_seg.build_model(cfg, nc)
        state_path = os.path.join(tmp, 'init.pth')
        torch.save(model0.state_dict(), state_path)

        if args.backend == 'nccl' and torch.cuda.device_count() < 2:
            raise SystemExit('--backend nccl needs two visible GPUs')
        res = {'cfg': args.cfg, 'backend': args.backend, 'frames': keys, 'valid_voxels': n_valid,
               'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'readings': {}}
        ok = True
        for amp in (False, True):
            tag = 'amp' if amp else 'fp32'
            arms = {}
            for name in ('ref', 'ref2'):
                m = train_seg.build_model(cfg, nc)
                m.load_state_dict(model0.state_dict())
                m = m.cuda()
                arms[name] = run_arm(m, to_cuda(collate(samples)), amp)
                del m
                torch.cuda.empty_cache()
            for name, sync in (('sync', True), ('nosync', False)):
                outp = os.path.join(tmp, '%s_%s_rank%%d.pth' % (tag, name))
                mp.spawn(child, args=(2, free_port(), args.backend, args.cfg, nc, state_path, samples, sync, amp,
                                      outp), nprocs=2, join=True)
                arms[name] = [torch.load(outp % r, weights_only=False) for r in range(2)]
            ref, ref2, sync, nosync = arms['ref'], arms['ref2'], arms['sync'], arms['nosync']
            n0 = len(sync[0]['logits'])                     # rank 0's voxels = the first rows of ref's batch
            r = {
                'n_bn': ref['n_bn'], 'sync_layers': [s['sync_layers'] for s in sync],
                'nosync_sync_layers': [s['sync_layers'] for s in nosync],
                'rank_valid_voxels': [s['points_valid'] for s in sync],
                'rank_gpus': [s['gpu'] for s in sync],
                'floor_bn': rel_bn(ref2['bn'], ref['bn']), 'floor_logits': rel(ref2['logits'], ref['logits']),
                'floor_grad': rel_dict(ref2['grad_global'], ref['grad_global']),
                'sync_bn': max(rel_bn(s['bn'], ref['bn']) for s in sync),
                'sync_logits': max(rel(sync[0]['logits'], ref['logits'][:n0]),
                                   rel(sync[1]['logits'], ref['logits'][n0:])),
                'sync_grad_global': rel_dict(sync[0]['grad_global'], ref['grad_global']),
                'sync_grad_rank': rel_dict(sync[0]['grad_rank'], ref['grad_global']),
                'nosync_bn': max(rel_bn(s['bn'], ref['bn']) for s in nosync),
                'nosync_logits': max(rel(nosync[0]['logits'], ref['logits'][:n0]),
                                     rel(nosync[1]['logits'], ref['logits'][n0:])),
                'nosync_grad_global': rel_dict(nosync[0]['grad_global'], ref['grad_global']),
                'finite': all([ref['finite'], ref2['finite']] + [s['finite'] for s in sync + nosync]),
                'loss_ref': ref['loss_global'], 'loss_sync_global': sync[0]['loss_global'],
            }
            assert all(n == ref['n_bn'] for n in r['sync_layers']), 'not every BN was converted'
            assert all(n == 0 for n in r['nosync_sync_layers'])
            if not amp:
                r['R1'] = r['sync_bn'][0] <= max(1e-4, 10 * r['floor_bn'][0])
                r['R2'] = r['sync_logits'] <= max(1e-3, 10 * r['floor_logits'])
                r['R3'] = r['sync_grad_global'][0] <= max(1e-3, 10 * r['floor_grad'][0])
                r['R4'] = r['nosync_bn'][0] >= 100 * r['sync_bn'][0]
                ok &= r['R1'] and r['R2'] and r['R3'] and r['R4'] and r['finite']
            else:
                r['R5'] = (r['sync_bn'][0] <= max(1e-2, 10 * r['floor_bn'][0])
                           and r['sync_logits'] <= max(1e-2, 10 * r['floor_logits']) and r['finite'])
                ok &= r['R5']
            res['readings'][tag] = r
            print(tag, json.dumps(r, indent=1), flush=True)
            del arms, ref, ref2, sync, nosync
        res['PASS'] = bool(ok)
        res['wall_s'] = time.time() - t0
        with open(os.path.join(args.out, 'result.json'), 'w') as f:
            json.dump(res, f, indent=1)
        print('SYNC_BN check:', 'PASS' if ok else 'FAIL', '(%.0f s)' % res['wall_s'], flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
