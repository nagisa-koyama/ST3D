import _init_path
import os
from pathlib import Path
import argparse
import copy
import datetime
import glob
import subprocess

import torch
import torch.nn as nn
import torch.distributed as dist
from test import repeat_eval_ckpt, get_all_configs, get_eval_configs

import wandb

from torchinfo import summary

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader, link_point_calibration
from pcdet.models.model_utils.dsnorm import DSNorm
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from train_utils.train_st_utils import train_model_st


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
    parser.add_argument('--pretrained_model_teacher', type=str, default=None, help='pretrained_model for teacher model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    # Default falls back to LOCAL_RANK env var, not a hardcoded 0. Modern torch.distributed.launch
    # /torchrun (torch>=2.x) no longer passes --local_rank to the child process: by default it
    # passes --local-rank (hyphen), which this argparse does not recognize (argparse does not
    # treat --local_rank/--local-rank as aliases) and the whole launch fails with "unrecognized
    # arguments"; with --use-env it passes NEITHER and only sets the LOCAL_RANK env var, which the
    # old default of 0 ignored - every process then defaulted to local_rank=0 and collided in
    # init_dist_pytorch's `dist.init_process_group(rank=local_rank, ...)` (rank 0 claimed twice,
    # world_size mismatch). This default lets --use-env work without every process racing to be
    # rank 0; explicit --local_rank N on the command line still overrides it as before. Found
    # 2026-09-22 validating multi-GPU DDP for the da-ieee-access throughput profiling - no DDP job
    # in this repo appears to have run successfully against this torch version before.
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)),
                        help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=100, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--eval_fov_only', action='store_true', default=False, help='')
    parser.add_argument('--eval_src', action='store_true', default=False, help='')
    # default=None so OPTIMIZATION.NUM_EPOCHS_TO_EVAL is reachable; 100 if neither is set,
    # which preserves the historical behaviour for every config that does not declare it.
    parser.add_argument('--num_epochs_to_eval', type=int, default=None,
                        help='how many trailing checkpoints to evaluate; default comes from '
                             'OPTIMIZATION.NUM_EPOCHS_TO_EVAL, else 100')
    parser.add_argument('--run_name', type=str, default=None, help='run name for wandb')
    parser.add_argument('--use_subset', action='store_true', help='use subset of data for quick test')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='disable training-set shuffling. Only for A/B-ing the 2026-09-21 '
                             'shuffle fix against runs made while it was broken - see '
                             'experiments_md/20260921_03_dataloader_shuffle_disabled_in_training.md. '
                             'Not for normal training.')

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
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

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

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

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
    ps_label_dir = output_dir / 'ps_label'
    ps_label_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    # The GLOBAL batch is derived (per-GPU x GPU count) and appears in no config file and in no
    # argparse value, so it is the one number a reader cannot check anywhere else - and getting it
    # wrong is silent. Launching the da-ieee-access family on 2 GPUs without `--batch_size 6` gives
    # a global batch of 12 and half the optimizer steps, with no error. Log it, and the step count
    # it implies, so a wrong recipe is visible in the first screen of every job log.
    logger.info('global batch size: %d (%d per GPU x %d GPU%s)'
                % (total_gpus * args.batch_size, args.batch_size, total_gpus,
                   '' if total_gpus == 1 else 's'))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    # -----------------------create dataloaders---------------------------
    if cfg.get('DATA_CONFIG', None):
        data_configs = {'DATA_CONFIG': cfg.DATA_CONFIG}
    elif cfg.get('DATA_CONFIGS', None):
        data_configs = cfg.DATA_CONFIGS
    else:
        assert False, "Eigher DATA_CONFIG or DATA_CONFIGS should be defined"
    source_datasets = list()
    source_class_names = cfg.CLASS_NAMES
    if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('MODEL_TEACHER', None):
        teacher_class_names = cfg.SELF_TRAIN.MODEL_TEACHER.get('CLASS_NAMES', None)
        if teacher_class_names is not None:
            source_class_names = copy.deepcopy(teacher_class_names)

    # Set None to model_ontology of source dataset if teacher model is head_per_dataset.
    source_model_ontology = cfg.get('ONTOLOGY', None)
    if cfg.get('SELF_TRAIN', None):
        if cfg.SELF_TRAIN.get('MODEL_TEACHER', None):
            if cfg.SELF_TRAIN.MODEL_TEACHER.get('ONTOLOGY', None):
                source_model_ontology = None
    for data_config in data_configs.values():
        source_set, source_loader, source_sampler = build_dataloader(
            dataset_cfg=data_config,
            class_names=source_class_names,
            batch_size=args.batch_size,
            dist=dist_train, workers=args.workers,
            logger=logger,
            training=True,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            total_epochs=args.epochs,
            model_ontology=source_model_ontology,
            use_subset=args.use_subset,
            force_no_shuffle=args.no_shuffle
        )
        dataset = dict(dataset_class=source_set, loader=source_loader, sampler=source_sampler)
        source_datasets.append(dataset)

    if cfg.get('SELF_TRAIN', None):
        target_set, target_loader, target_sampler = build_dataloader(
            cfg.DATA_CONFIG_TAR, cfg.CLASS_NAMES, args.batch_size,
            dist_train, workers=args.workers, logger=logger, training=True,
            model_ontology=cfg.get('ONTOLOGY', None),
            use_subset=args.use_subset,
            force_no_shuffle=args.no_shuffle
        )
    else:
        target_set = target_loader = target_sampler = None

    # log datasets
    for index, source in enumerate(source_datasets):
        logger.info('source dataset %d: %s', index, source['dataset_class'].__class__.__name__)
    if target_set is not None:
        logger.info('target dataset: %s', target_set.__class__.__name__)

    # Density correction, measured from the datasets actually being trained on. adaptive_train.py
    # has the same hook; without one here a source-only config that lists sample_points_hist_based
    # would run it with no histograms installed, and the processor early-returns - a silent no-op
    # that makes the corrected run identical to the uncorrected one with nothing to show for it.
    # Read per SOURCE config, not from cfg.DATA_CONFIG: a multi-source run declares DATA_CONFIGS
    # instead and has no DATA_CONFIG at all, so reaching for it would raise AttributeError before
    # training starts. Each source also gets its own histogram against the shared target, which is
    # the point - two Lyft platforms are different sensors and want different rates.
    wants_foreground = any(dc.get('HIST_DIST_FOREGROUND_FROM_PSEUDO_LABELS', False)
                          for dc in data_configs.values())
    if any(dc.get('HIST_DIST_ON_THE_FLY', False) for dc in data_configs.values()):
        # The foreground channel comes from the TARGET's pseudo-labels, so it can only be measured
        # once a pseudo-label pass has run. That happens in train_st_utils.train_model_st, which
        # THIS file dispatches to - so the requirement is a SELF_TRAIN config, not a different
        # entry point. (The message here used to say adaptive_train.py; that file has no
        # SELF_TRAIN handling at all, and following it cost job 25930.)
        assert not wants_foreground or cfg.get('SELF_TRAIN', None), \
            ('HIST_DIST_FOREGROUND_FROM_PSEUDO_LABELS needs the target foreground channel, which '
             'comes from pseudo-labels, so it requires a SELF_TRAIN block. This config has none, '
             'so there would be no pseudo-labels to measure.')
        if wants_foreground:
            # Do NOT install the global correction here. link_foreground_calibration installs the
            # whole-cloud pair itself, and it measures the SOURCE while doing so - if the source
            # were already being corrected, that measurement would read points the correction had
            # thinned and compound the rate on every pass, which the function's own docstring
            # warns about. Leave the source uncorrected until train_model_st installs both.
            logger.info('foreground-aware correction requested: deferring calibration to the '
                        'first pseudo-label pass (train_model_st)')
        # A measurement-only view of the target: only its point clouds are read, and eval mode
        # avoids requiring labels the UDA setup does not have.
        calib_target = None if wants_foreground else target_set
        if calib_target is None and not wants_foreground:
            calib_target, _, _ = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1,
                dist=False, workers=0, logger=logger, training=False,
                model_ontology=cfg.get('ONTOLOGY', None))
        for dc, source in zip(data_configs.values(), source_datasets):
            if wants_foreground or not dc.get('HIST_DIST_ON_THE_FLY', False):
                continue
            link_point_calibration(
                source['dataset_class'], calib_target,
                num_frames=dc.get('HIST_DIST_FRAMES', 1000),
                num_bins=dc.get('HIST_DIST_BINS', 50),
                max_dist=dc.get('HIST_DIST_MAX_DIST', 75.0),
                logger=logger)

    # A correction listed in the pipeline but never given histograms is a no-op that looks like a
    # run. Say so rather than letting the result be quietly identical to the uncorrected arm.
    for dc, source in zip(data_configs.values(), source_datasets):
        proc = source['dataset_class'].data_processor
        if any(p.get('NAME') == 'sample_points_hist_based'
               for p in dc.get('DATA_PROCESSOR', [])) and proc.hist_dist_src is None:
            logger.warning('sample_points_hist_based is configured but no histograms were '
                           'installed - the correction will NOT run. Set '
                           'DATA_CONFIG.HIST_DIST_ON_THE_FLY: True.')

    # -----------------------config validity---------------------------
    miss_spelled_configs = ['BACKWORD_TOGETHER']
    for miss_spelled_config in miss_spelled_configs:
        assert cfg.get(miss_spelled_config, None) is None, "{} is miss-spelled.".format(miss_spelled_config)
        if cfg.get('SELF_TRAIN', None):
            assert cfg.SELF_TRAIN.get(miss_spelled_config, None) is None, "SELF_TRAIN.{} is miss-spelled.".format(miss_spelled_config)
            assert cfg.SELF_TRAIN.SRC.get(miss_spelled_config, None) is None, "SELF_TRAIN.SRC.{} is miss-spelled.".format(miss_spelled_config)
            assert cfg.SELF_TRAIN.TAR.get(miss_spelled_config, None) is None, "SELF_TRAIN.TAR.{} is miss-spelled.".format(miss_spelled_config)

    # -----------------------create networks---------------------------
    for source_dataset in source_datasets:
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES),
                              dataset=target_set if cfg.get('SELF_TRAIN', None) else source_dataset['dataset_class'])
        if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('MODEL_TEACHER', None):
            model_teacher = build_network(model_cfg=cfg.SELF_TRAIN.MODEL_TEACHER, num_class=len(source_class_names),
                            dataset=source_dataset['dataset_class'])
        else:
            model_teacher = None
        break

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if model_teacher is not None:
            model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_teacher)
    elif cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
        model = DSNorm.convert_dsnorm(model)
        if model_teacher is not None:
            model_teacher = DSNorm.convert_dsnorm(model_teacher)
    # uplaod to GPU
    model.cuda()
    if model_teacher is not None:
        model_teacher.cuda()

    # if cfg.LOCAL_RANK == 0:
    #     wandb.watch(model, log_freq=100)

    # log networks
    logger.info('****** model: %s ******', model.__class__.__name__)
    logger.info(model)
    if model_teacher is not None:
        logger.info('****** model_teacher: %s ******', model_teacher.__class__.__name__)
        logger.info(model_teacher)

    # -----------------------create optimizer---------------------------
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # -----------------------load pretrained model if specified---------------------------
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
        logger.info('pretrained_model %s is loaded to model %s', args.pretrained_model, model.__class__.__name__)
    if args.pretrained_model_teacher is not None and model_teacher is not None:
        model_teacher.load_params_from_file(filename=args.pretrained_model_teacher, to_cpu=dist_train, logger=logger)
        logger.info('pretrained_model_teacher %s is loaded to model_teacher %s', args.pretrained_model_teacher,
                    model_teacher.__class__.__name__)

    # -----------------------load checkpoint if specified---------------------------
    start_epoch = it = 0
    last_epoch = -1
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    # TODO: Delete this block once we confirm that we don't need to load from checkpoint in the local folder.
    # else:
    #     ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
    #     if len(ckpt_list) > 0:
    #         ckpt_list.sort(key=os.path.getmtime)
    #         it, start_epoch = model.load_params_with_optimizer(
    #             ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
    #         )
    #         last_epoch = start_epoch + 1

    # -----------------------configure networks---------------------------
    # configurte distributed run
    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if model_teacher is not None:
        model_teacher.eval() # model_teacher should not be updated
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
        if model_teacher is not None:
            model_teacher = nn.parallel.DistributedDataParallel(
                model_teacher, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()],
                find_unused_parameters=True
            )

    # -----------------------create scheduler---------------------------
    def total_iters_each_epoch_per_dataloader(dataloader, merge_all_iter_to_one_epoch, epochs):
        return len(dataloader) if not merge_all_iter_to_one_epoch else len(dataloader) // epochs

    if cfg.get('SELF_TRAIN', None):
        total_iters_each_epoch = total_iters_each_epoch_per_dataloader(target_loader,
                                                                       args.merge_all_iters_to_one_epoch,
                                                                       args.epochs)
    else:
        iters_each_epoch_list = list()
        for source_dataset in source_datasets:
            iters_each_epoch = total_iters_each_epoch_per_dataloader(source_dataset['loader'],
                                                                     args.merge_all_iters_to_one_epoch,
                                                                     args.epochs)
            iters_each_epoch_list.append(iters_each_epoch)

        # No upsampling.
        total_iters_each_epoch = sum(iters_each_epoch_list)
        # Upsample smaller dataset.
        # total_iters_each_epoch = max(iters_each_epoch_list)*len(iters_each_epoch_list)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # select proper trainer
    train_func = train_model_st if cfg.get('SELF_TRAIN', None) else train_model

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    source_loaders = [dataset['loader'] for dataset in source_datasets]
    source_samplers = [dataset['sampler'] for dataset in source_datasets]
    train_func(
        model,
        model_teacher,
        optimizer,
        source_loaders,
        target_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=None,
        ckpt_save_dir=ckpt_dir,
        ps_label_dir=ps_label_dir,
        source_samplers=source_samplers,
        target_sampler=target_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        ema_model=None
    )

    # Exclude pth files from wandb upload.
    # TODO: Make this work.
    # pth_list = glob.glob(str(cfg.ROOT_DIR / '**/*.pth'), recursive=True)
    # pth_list_str = ','.join(pth_list)
    # os.environ['WANDB_IGNORE_GLOBS'] = pth_list_str

    # Ensure all ranks finished training before starting evaluation.
    if dist_train:
        dist.barrier()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    if args.eval_fov_only:
        cfg.DATA_CONFIG_TAR.FOV_POINTS_ONLY = True

    test_data_configs = get_eval_configs(cfg)
    test_datasets = list()
   # load ontology for eval. This is necessary to evaluate multihead model with remapping.
    model_ontology_eval = cfg.get('EVAL_ONTOLOGY', None)
    if model_ontology_eval is None:
        model_ontology_eval = cfg.get('ONTOLOGY', None)
    for test_data_config in test_data_configs.values():
        test_set, test_loader, test_sampler = build_dataloader(
            dataset_cfg=test_data_config,
            class_names=cfg.CLASS_NAMES,
            batch_size=min(args.batch_size, 10),
            dist=dist_train, workers=args.workers,
            logger=logger, training=False,
            model_ontology=model_ontology_eval,
        )
        test_dataset = dict(dataset_class=test_set, loader=test_loader, sampler=test_sampler)
        test_datasets.append(test_dataset)

    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    # Only evaluate the last args.num_epochs_to_eval epochs
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)

    test_loaders = [dataset['loader'] for dataset in test_datasets]

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loaders, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
