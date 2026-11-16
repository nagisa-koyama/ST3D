"""Score a pcseg checkpoint on its config's val split: the plain evaluation (engine.evaluate, as in training) and
Pointcept's test protocol (pcseg/tta.py, the ten-view TTA that the G0 pass rule reads; experiments_md
episodes/20261116_01 §5). One GPU, no W&B; a long split is cut into shards run as separate jobs, then merged.

    python test_seg.py --cfg_file cfgs/pcseg/spunet-nuscenes.yaml --ckpt <run>/files/ckpt/checkpoint_epoch_50.pth \
        --out <dir> [--shard r --num_shards W] [--no_tta] [--no_plain]
    python test_seg.py --cfg_file <same> --merge <dir>       # sums the shard matrices, writes merged.json

Runs from the live checkout (no frozen snapshot): commit before submitting, and the JSON records HEAD.
"""
import _init_path  # noqa: F401  (sys.path and the allocator guard, before torch touches CUDA)
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--cfg_file', required=True)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--out', default=None, help='output directory (shard files go here)')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--no_tta', action='store_true')
    p.add_argument('--no_plain', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--merge', default=None, help='a directory of shard files to merge (no GPU)')
    return p.parse_args(argv)


def head_commit():
    try:
        return subprocess.check_output(['git', '-C', os.path.dirname(os.path.abspath(__file__)), 'rev-parse', 'HEAD'],
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def merge(cfg, names, shard_dir):
    import numpy as np
    from pcseg import engine
    files = sorted(Path(shard_dir).glob('shard_*_of_*.json'))
    if not files:
        raise FileNotFoundError('no shard files in %s' % shard_dir)
    shards = [json.load(open(f)) for f in files]
    w = {s['num_shards'] for s in shards}
    if len(w) != 1 or sorted(s['shard'] for s in shards) != list(range(w.pop())):
        raise ValueError('incomplete or mixed shard set: %s' % [f.name for f in files])
    if len({s['ckpt'] for s in shards}) != 1:
        raise ValueError('shards of different checkpoints')
    out = {'ckpt': shards[0]['ckpt'], 'epoch': shards[0]['epoch'], 'shards': [f.name for f in files],
           'commits': sorted({s['commit'] or '' for s in shards})}
    for mode in ('plain', 'tta'):
        if all(mode in s for s in shards):
            mats = {}
            for k in shards[0][mode]['matrices']:
                mats[k] = sum(np.asarray(s[mode]['matrices'][k], dtype=np.int64) for s in shards)
            m = {}
            for k, mat in mats.items():
                m.update(engine.metrics_from_matrix(mat, names, 'val_' + k))
            m['frames'] = sum(s[mode]['frames'] for s in shards)
            if mode == 'tta':
                m['forwards'] = sum(s[mode]['forwards'] for s in shards)
            out[mode] = dict(m, matrices={k: v.tolist() for k, v in mats.items()})
    engine.write_json(Path(shard_dir) / 'merged.json', out)
    for mode in ('plain', 'tta'):
        if mode in out:
            print('%s: point mIoU %.4f over %d frames' % (mode, out[mode]['val_point/miou'], out[mode]['frames']))
    return out


def main(argv=None):
    args = parse_args(argv)
    import train_seg
    from pcseg.class_map import build_class_map
    cfg = train_seg.load_cfg(args.cfg_file)
    names = build_class_map(cfg.DATA.CLASS_MAP).names
    if args.merge:
        merge(cfg, names, args.merge)
        return

    import torch
    from pcseg import engine, tta
    from pcseg.pandaset_seg import collate
    if not (args.ckpt and args.out):
        raise ValueError('--ckpt and --out are required unless --merge')
    if not 0 <= args.shard < args.num_shards:
        raise ValueError('shard %d of %d' % (args.shard, args.num_shards))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / ('shard_%03d_of_%03d.json' % (args.shard, args.num_shards))
    if out_file.exists():
        raise FileExistsError(out_file)

    device = torch.device('cuda')
    val = train_seg.build_datasets(cfg)['val']
    model = train_seg.build_model(cfg, len(names)).to(device)
    epoch = engine.load_checkpoint(args.ckpt, model)
    model.eval()
    idx = list(range(args.shard, len(val), args.num_shards))
    cone = cfg.get('EVAL', {}).get('CONE', None)
    cone = None if cone is None else (float(cone.FOV_DEG), float(cone.get('HEADING_DEG', 0.0)))
    shift = list(cfg.DATA.SHIFT_COOR)
    res = {'cfg_file': args.cfg_file, 'ckpt': os.path.abspath(args.ckpt), 'epoch': epoch, 'shard': args.shard,
           'num_shards': args.num_shards, 'frames_in_shard': len(idx), 'commit': head_commit(),
           'slurm_job': os.environ.get('SLURM_JOB_ID'), 'gpu': torch.cuda.get_device_name(0)}
    if not args.no_plain:
        t0 = time.time()
        loader = torch.utils.data.DataLoader(
            val, batch_size=int(cfg.OPTIMIZATION.get('EVAL_BATCH_PER_GPU', 4)), sampler=idx, num_workers=args.workers,
            collate_fn=collate)
        metrics, mats = engine.evaluate(model, loader, names, device, shift=shift, cone=cone)
        res['plain'] = {'frames': metrics['val/frames'], 'seconds': time.time() - t0,
                        'matrices': {k: v.tolist() for k, v in mats.items()}}
        print('plain: point mIoU %.4f (this shard, %d frames)' % (metrics['val_point/miou'], metrics['val/frames']),
              flush=True)
    if not args.no_tta:
        t0 = time.time()
        mats, frames, fwd = tta.evaluate_tta(model, val, names, device, indices=idx, cone=cone, shift=shift)
        res['tta'] = {'frames': frames, 'forwards': fwd, 'seconds': time.time() - t0,
                      'matrices': {k: v.tolist() for k, v in mats.items()}}
        m = engine.metrics_from_matrix(mats['point'], names, 'val_point')
        print('tta: point mIoU %.4f (this shard, %d frames, %d forwards, %.0f s)'
              % (m['val_point/miou'], frames, fwd, res['tta']['seconds']), flush=True)
    res['peak_mem_gib'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
    engine.write_json(out_file, res)
    print('wrote', out_file)


if __name__ == '__main__':
    main()
