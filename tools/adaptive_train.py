import _init_path
import os
from pathlib import Path
import argparse
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from test import repeat_eval_ckpt

import wandb

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import assert_target_labels_are_not_used, build_dataloader, link_point_calibration
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler, build_grl_scheduler
from train_utils.train_utils_adaptive import train_model_adaptive


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    # default=None, not 4, so the yaml's OPTIMIZATION.NUM_WORKERS is actually used - same
    # pattern as --batch_size above, and for the same reason: a non-None argparse default
    # silently wins over the config and nobody notices (train.py's --batch_size carried a
    # default=16 for months that way, fixed in 5456291).
    parser.add_argument('--workers', type=int, default=None, help='dataloader workers per GPU; '
                        'default comes from OPTIMIZATION.NUM_WORKERS, else 4')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=100, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    # default=None so OPTIMIZATION.NUM_EPOCHS_TO_EVAL is reachable; 100 if neither is set,
    # which preserves the historical behaviour for every config that does not declare it.
    parser.add_argument('--num_epochs_to_eval', type=int, default=None,
                        help='how many trailing checkpoints to evaluate; default comes from '
                             'OPTIMIZATION.NUM_EPOCHS_TO_EVAL, else 100')
    parser.add_argument('--run_name', type=str, default=None, help='run name for wandb')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        # torchrun sets LOCAL_RANK as an env var (not a CLI arg) in PyTorch >= 1.10.
        args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus
    assert args.batch_size % 2 == 0, (
        f"batch_size must be divisible by two (split evenly between source and target domains), "
        f"got per-GPU batch_size={args.batch_size}"
    )

    # Per-source, because the right value depends on whether that source's loader actually
    # stalls the GPU - measured in experiments_md/20260922_06. Absent key keeps the old 4.
    if args.workers is None:
        args.workers = cfg.get('OPTIMIZATION', {}).get('NUM_WORKERS', 4)

    # Evaluation is NOT free: one evaluation costs about as much as one KITTI training epoch
    # (299.5 s vs 300.2 s, job 25730), and over half of that is single-threaded AP
    # computation on rank 0. At the historical default of 100 that is ~12 h per long run.
    # See experiments_md/20260922_06 section 4d.
    if args.num_epochs_to_eval is None:
        args.num_epochs_to_eval = cfg.get('OPTIMIZATION', {}).get('NUM_EPOCHS_TO_EVAL', 100)

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    # Sanity-check: mismatched ranges produce different BEV spatial dims and break the
    # per-class conditional discriminator (source/target cls_preds_spatial shapes must match).
    assert list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE) == list(cfg.DATA_CONFIG_TAR.POINT_CLOUD_RANGE), (
        "POINT_CLOUD_RANGE mismatch between DATA_CONFIG and DATA_CONFIG_TAR — "
        "both must be identical for domain adaptation (BEV feature map dims must match)."
    )

    if cfg.LOCAL_RANK == 0:
        wandb.init(config=vars(cfg), project="st3d", name=args.run_name, dir="/storage")
        print("W&B run directory:", wandb.run.dir)

    # Share wandb output dir across all ranks so every process uses the same paths.
    if dist_train:
        object_list = [wandb.run.dir if cfg.LOCAL_RANK == 0 else None]
        dist.broadcast_object_list(object_list, src=0)
        output_dir = Path(object_list[0])
    else:
        output_dir = Path(wandb.run.dir)

    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    # -----------------------create dataloaders---------------------------
    source_set, source_loader, source_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size // 2, dist=dist_train, workers=args.workers,
        logger=logger, training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, total_epochs=args.epochs
    )
    # A DA target must never contribute its REAL labels. Checked here, before a
    # loader exists, so a misconfigured run dies in seconds rather than minutes.
    assert_target_labels_are_not_used(cfg, True, logger)
    target_set, target_loader, target_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size // 2, dist=dist_train, workers=args.workers,
        logger=logger, training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, total_epochs=args.epochs
    )
    logger.info('source dataset: %s', source_set.__class__.__name__)
    logger.info('target dataset: %s', target_set.__class__.__name__)

    # Measure the density-correction histograms from the datasets actually being trained on,
    # rather than loading hist_dist_*.npy files whose frame counts, preprocessing and
    # POINT_CLOUD_RANGE may no longer match. Must happen before either loader is iterated:
    # DataLoader workers fork a copy of the dataset and never see later mutations.
    # Foreground-aware calibration is measured later, inside the self-training loop, because it
    # needs the target's pseudo-labels to exist. Running the whole-cloud version here as well would
    # both double the measurement cost and make that later measurement circular - it would see a
    # source already being corrected.
    if cfg.DATA_CONFIG.get('HIST_DIST_ON_THE_FLY', False) \
            and not cfg.DATA_CONFIG.get('HIST_DIST_FOREGROUND_FROM_PSEUDO_LABELS', False):
        link_point_calibration(
            source_set, target_set,
            num_frames=cfg.DATA_CONFIG.get("HIST_DIST_FRAMES", 1000),
            num_bins=cfg.DATA_CONFIG.get('HIST_DIST_BINS', 50),
            max_dist=cfg.DATA_CONFIG.get('HIST_DIST_MAX_DIST', 75.0),
            logger=logger
        )

    # -----------------------create networks---------------------------
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=source_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    logger.info('****** model: %s ******', model.__class__.__name__)
    logger.info(model)

    # -----------------------create optimizer---------------------------
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # -----------------------load pretrained model if specified---------------------------
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
        logger.info('pretrained_model %s is loaded to model %s', args.pretrained_model, model.__class__.__name__)

    # -----------------------load checkpoint if specified---------------------------
    start_epoch = it = 0
    last_epoch = -1
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1

    # -----------------------configure networks---------------------------
    model.train()
    if dist_train:
        # broadcast_buffers=False is INTENTIONAL: prevents BN running-stat sync across the two
        # domains, which have different feature distributions (UADA3D port).
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], broadcast_buffers=False
        )

    # -----------------------create scheduler---------------------------
    total_iters_each_epoch = len(source_loader)
    if args.merge_all_iters_to_one_epoch:
        total_iters_each_epoch = total_iters_each_epoch // max(args.epochs, 1)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )
    grl_scheduler = build_grl_scheduler(total_iters_each_epoch=total_iters_each_epoch, total_epochs=args.epochs)

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model_adaptive(
        model, optimizer, source_loader=source_loader, target_loader=target_loader,
        model_func=model_fn_decorator(), lr_scheduler=lr_scheduler, grl_scheduler=grl_scheduler,
        optim_cfg=cfg.OPTIMIZATION, start_epoch=start_epoch, total_epochs=args.epochs, start_iter=it,
        rank=cfg.LOCAL_RANK, tb_log=None, ckpt_save_dir=ckpt_dir,
        source_sampler=source_sampler, target_sampler=target_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler, ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
    )

    if hasattr(source_set, 'use_shared_memory') and source_set.use_shared_memory:
        source_set.clean_shared_memory()
    if hasattr(target_set, 'use_shared_memory') and target_set.use_shared_memory:
        target_set.clean_shared_memory()

    if dist_train:
        dist.barrier()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # -----------------------start evaluation---------------------------
    # Reuses TARGET_DATA_CONFIG (named DATA_CONFIG_TAR to match ST3D's existing convention) +
    # ST3D's existing repeat_eval_ckpt/test.py — same evaluation() code path as every other
    # ST3D experiment, which is the whole point of this migration (like-for-like A/B eval).
    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES,
        batch_size=min(args.batch_size, 10), dist=dist_train, workers=args.workers,
        logger=logger, training=False
    )

    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)

    repeat_eval_ckpt(
        model.module if dist_train else model,
        [test_loader], args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
