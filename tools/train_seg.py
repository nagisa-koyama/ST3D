"""pcseg training entry point (experiments_md episodes/20261116_01 §8, step S1).

Launched through the detection launcher unchanged, which freezes the code into the job's scratch and handles DDP:

    ENTRY=train_seg.py BATCH=<total batch> scripts/submit.sh ... scripts/run_sourceonly_2gpu.sh cfgs/pcseg/<x>.yaml [tag] [run_name]

It takes the launcher's arguments (--cfg_file, --batch_size TOTAL across ranks, --fix_random_seed, --run_name,
--extra_tag, --launcher, --tcp_port) and never imports a pcdet dataset, model or the detection evaluation. Its end
marker is 'pcseg training finished', NOT the detection 'End training' that arms the launcher's detection-eval recovery.

The DDP training sampler is always seeded with --seed (pcseg has no history to keep; --ddp_sampler_seed is accepted
and changes nothing). Each rank seeds its RNGs with seed + rank, so ranks draw different augmentations; DDP broadcasts
rank 0's initial weights.
"""
import _init_path  # noqa: F401  (sys.path and the allocator guard, before torch touches CUDA)
import argparse
import datetime
import os
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='pcseg training')
    p.add_argument('--cfg_file', required=True)
    p.add_argument('--batch_size', type=int, default=None, help='TOTAL across ranks; default OPTIMIZATION.BATCH_SIZE')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--workers', type=int, default=None, help='per rank; default OPTIMIZATION.NUM_WORKERS')
    p.add_argument('--extra_tag', default='default')
    p.add_argument('--run_name', default=None)
    p.add_argument('--ckpt', default=None, help='resume: model, optimizer, scheduler, scaler and epoch')
    p.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    p.add_argument('--tcp_port', type=int, default=18888)
    p.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    p.add_argument('--fix_random_seed', action='store_true')
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--ddp_sampler_seed', action='store_true', help='accepted for launcher parity; always on here')
    p.add_argument('--no_wandb', action='store_true', help='local checks only: write to --out_dir instead')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    return p.parse_args(argv)


def load_cfg(cfg_file, set_cfgs=None):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file, cfg_from_list
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    if set_cfgs:
        cfg_from_list(set_cfgs, cfg)
    if not cfg.get('PCSEG', False):
        raise ValueError('%s is not a pcseg config (PCSEG: True missing)' % cfg_file)
    for key in ('DATA', 'MODEL', 'OPTIMIZATION'):
        if key not in cfg:
            raise ValueError('%s: %s block missing (unresolved _BASE_CONFIG_?)' % (cfg_file, key))
    cfg.TAG = Path(cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_file.split('/')[1:-1])
    return cfg


def build_datasets(cfg, logger=None):
    from pcseg.pandaset_seg import PandasetSeg
    d = cfg.DATA
    if d.NAME != 'PandasetSeg':
        raise ValueError('unknown DATA.NAME %s' % d.NAME)
    common = {k: d[k] for k in ('ROOT', 'LIDAR_DEVICE', 'SHIFT_COOR', 'CLASS_MAP', 'GRID_SIZE')}
    sets = {'train': PandasetSeg(dict(common, SEQUENCES=d.SPLITS.train, AUG=d.get('AUG', None)), True, logger),
            'val': PandasetSeg(dict(common, SEQUENCES=d.SPLITS.val), False, logger)}
    for split, n in (d.get('EXPECT_SEQUENCES', None) or {}).items():
        if len(sets[split].sequences) != int(n):
            raise ValueError('%s: %d sequences with semseg, config expects %d (skipped %s)'
                             % (split, len(sets[split].sequences), int(n), sets[split].skipped))
    return sets


def build_model(cfg, num_classes):
    from pcseg.spunet import SpUNet
    m = cfg.MODEL
    if m.NAME != 'SpUNet':
        raise ValueError('unknown MODEL.NAME %s' % m.NAME)
    return SpUNet(in_channels=int(m.IN_CHANNELS), num_classes=num_classes, channels=tuple(m.CHANNELS),
                  layers=tuple(m.LAYERS))


def main(argv=None):
    args = parse_args(argv)
    cfg = load_cfg(args.cfg_file, args.set_cfgs)

    import numpy as np
    import torch
    import torch.distributed as dist
    from pcdet.utils import common_utils
    from pcseg import engine
    from pcseg.class_map import build_class_map
    from pcseg.pandaset_seg import collate

    if args.launcher == 'none':
        world, rank = 1, 0
    else:
        world, rank = common_utils.init_dist_pytorch(args.tcp_port, args.local_rank, backend='nccl')
    device = torch.device('cuda', torch.cuda.current_device())
    opt = cfg.OPTIMIZATION
    total_bs = int(opt.BATCH_SIZE) if args.batch_size is None else args.batch_size
    if total_bs % world:
        raise ValueError('total batch %d is not divisible by %d ranks' % (total_bs, world))
    bs = total_bs // world
    epochs = int(opt.NUM_EPOCHS) if args.epochs is None else args.epochs
    workers = int(opt.get('NUM_WORKERS', 8)) if args.workers is None else args.workers
    if args.fix_random_seed:
        common_utils.set_random_seed(args.seed + rank)

    # --- output dir: the W&B run dir, shared by all ranks (as train.py) ---
    run = None
    if args.no_wandb:
        output_dir = Path(args.out_dir or 'pcseg_local_%s' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    else:
        import wandb
        # Checkpoints stay on disk (NFS) and are never uploaded: ~470 MB each with the optimiser state.
        os.environ.setdefault('WANDB_IGNORE_GLOBS', '*.pth,*.pth.tmp')
        if rank == 0:
            run = wandb.init(config=dict(cfg), project='st3d', name=args.run_name, dir='/storage',
                             notes=common_utils.wandb_notes_with_job_id(),
                             tags=common_utils.wandb_tags(cfg, 'train_seg'))
            wandb.config.update(common_utils.wandb_launch_config(world, bs))
            wandb.config.update({'args': vars(args)})
            print('W&B run directory:', wandb.run.dir)
        obj = [wandb.run.dir if rank == 0 else None]
        if world > 1:
            dist.broadcast_object_list(obj, src=0)
        output_dir = Path(obj[0])
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(output_dir / ('log_train_seg_%s.txt'
                                                      % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')), rank=rank)
    log_fn = (lambda d: run.log(d)) if run is not None else None
    logger.info('pcseg train: %s, tag %s, %d rank(s), global batch size: %d (%d per GPU x %d GPU%s)'
                % (args.cfg_file, args.extra_tag, world, total_bs, bs, world, '' if world == 1 else 's'))
    for k, v in vars(args).items():
        logger.info('%-18s %s' % (k, v))
    from pcdet.config import log_config_to_file
    log_config_to_file(cfg, logger=logger)
    if rank == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    # --- data ---
    class_map = build_class_map(cfg.DATA.CLASS_MAP)
    names = class_map.names
    sets = build_datasets(cfg, logger)
    if rank == 0:
        dump = engine.target_dump(sets, names, n_frames=int(cfg.get('TARGET_DUMP_FRAMES', 4)))
        engine.write_json(output_dir / 'target_dump.json', dump)
        logger.info('target dump (bug_prevention_checklist rule 4):\n' + engine.format_dump(dump))
    if world > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(sets['train'], num_replicas=world, rank=rank,
                                                                        shuffle=True, seed=args.seed, drop_last=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(sets['train'])
    train_loader = torch.utils.data.DataLoader(
        sets['train'], batch_size=bs, sampler=train_sampler, num_workers=workers, collate_fn=collate,
        pin_memory=True, drop_last=True, persistent_workers=False)
    val_loader = torch.utils.data.DataLoader(
        sets['val'], batch_size=int(opt.get('EVAL_BATCH_PER_GPU', 4)),
        sampler=engine.ShardSampler(len(sets['val']), rank, world), num_workers=workers, collate_fn=collate,
        pin_memory=True)
    max_iters = opt.get('MAX_ITERS_PER_EPOCH', None)
    iters_per_epoch = len(train_loader) if max_iters is None else min(len(train_loader), int(max_iters))
    logger.info('train: %d frames, %d iterations per epoch, %d epochs, %d optimiser steps; val: %d frames'
                % (len(sets['train']), iters_per_epoch, epochs, iters_per_epoch * epochs, len(sets['val'])))

    # --- model and optimisation ---
    model = build_model(cfg, class_map.num_classes).to(device)
    if world > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) if opt.get('SYNC_BN', False) else model
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    logger.info('model %s: %d parameters' % (cfg.MODEL.NAME, sum(p.numel() for p in model.parameters())))
    optimizer = engine.build_optimizer(model, opt)
    scheduler = engine.build_scheduler(optimizer, opt, iters_per_epoch * epochs)
    amp = bool(opt.get('AMP', True))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start_epoch = 0
    if args.ckpt:
        start_epoch = engine.load_checkpoint(args.ckpt, model, optimizer, scheduler, scaler) + 1
        logger.info('resumed from %s: continuing at epoch %d' % (args.ckpt, start_epoch))

    cone = cfg.get('EVAL', {}).get('CONE', None)
    cone = None if cone is None else (float(cone.FOV_DEG), float(cone.get('HEADING_DEG', 0.0)))
    eval_every = int(cfg.get('EVAL', {}).get('EVERY', 5))
    save_every = int(opt.get('SAVE_EVERY', 5))
    keep = int(opt.get('MAX_CKPT', 3))
    shift = list(cfg.DATA.SHIFT_COOR)

    # --- loop ---
    t_start = time.time()
    timing = []
    last_metrics = None
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr = engine.train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, device, amp=amp,
                                    clip_grad=opt.get('CLIP_GRAD', None), log_every=int(opt.get('LOG_EVERY', 50)),
                                    log_fn=log_fn, logger=logger, max_iters=max_iters)
        t_train = time.time() - t0
        row = {'epoch': epoch, 'train_s': t_train, 'data_s': tr['data_s'], 'step_s': tr['step_s'],
               'train/loss_epoch': tr['loss'], 'iters': tr['iters']}
        logger.info('epoch %d: loss %.4f, %d iterations, %.0f s (data wait %.0f s, steps %.0f s)'
                    % (epoch, tr['loss'], tr['iters'], t_train, tr['data_s'], tr['step_s']))
        last = epoch == epochs - 1
        if last or (eval_every > 0 and (epoch + 1) % eval_every == 0):
            t1 = time.time()
            metrics, mats = engine.evaluate(model, val_loader, names, device, amp=amp, shift=shift, cone=cone)
            row['eval_s'] = time.time() - t1
            last_metrics = dict(metrics, epoch=epoch)
            logger.info('epoch %d eval (%.0f s, %d frames): point mIoU %.4f, voxel mIoU %.4f%s'
                        % (epoch, row['eval_s'], metrics['val/frames'], metrics['val_point/miou'],
                           metrics['val_voxel/miou'],
                           ', cone mIoU %.4f' % metrics['val_cone/miou'] if cone is not None else ''))
            for n_ in names:
                logger.info('    %-20s IoU %.4f' % (n_, metrics['val_point/iou/' + n_]))
            if rank == 0:
                engine.write_json(output_dir / ('eval_epoch_%d.json' % epoch),
                                  dict(metrics, epoch=epoch, matrices={k: v.tolist() for k, v in mats.items()}))
                if log_fn is not None:
                    log_fn({k: v for k, v in metrics.items() if not isinstance(v, list)})
        if log_fn is not None:
            log_fn({'train/loss_epoch': tr['loss'], 'time/train_s': t_train, 'time/data_s': tr['data_s'],
                    'time/step_s': tr['step_s'], 'train/epoch': epoch})
        timing.append(row)
        if rank == 0 and (last or (save_every > 0 and (epoch + 1) % save_every == 0)):
            engine.save_checkpoint(ckpt_dir / ('checkpoint_epoch_%d.pth' % (epoch + 1)), model, optimizer, scheduler,
                                   scaler, epoch, extra={'cfg_file': args.cfg_file, 'class_names': names})
            engine.prune_checkpoints(ckpt_dir, keep)
    wall = time.time() - t_start
    if world > 1:
        dist.barrier()
    if rank == 0:
        summary = {'wall_s': wall, 'gpus': world, 'gpu_hours': wall * world / 3600.0, 'epochs': timing,
                   'final': last_metrics}
        engine.write_json(output_dir / 'train_summary.json', summary)
        if run is not None:
            run.summary.update({'time/wall_s': wall, 'time/gpu_hours': summary['gpu_hours']})
            if last_metrics is not None:
                run.summary.update({'final/' + k: v for k, v in last_metrics.items() if not isinstance(v, list)})
    logger.info('pcseg training finished: %.0f s wall, %.2f GPU hours' % (wall, wall * world / 3600.0))
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
