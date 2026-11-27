"""Evaluate a count-normalized checkpoint, optionally after global AdaBN (experiments_md 20261003_05 §5 item 64, design fixed
2026-11-27 13:02 JST, before any number). ANALYSIS.

    python analysis/item64_si_eval.py --cfg_file <cfg with MODEL.COUNT_NORM_CONV> --ckpt <ckpt> --adapt none|ada --out_dir DIR
        [--stats_frames 1000] [--stats_batch 4] [--smoke] [--run_name ...]

`ada`: global AdaBN (adabn_utils.reestimate_bn) on `--stats_frames` strided frames of the eval dataset's TRAIN split (no
labels). result.pkl and log_si_*.txt in DIR. The checkpoint must fill every state key of the model exactly (checked).
"""
import argparse
import datetime
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402  (also installs the allocator guard)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402

from eval_utils import eval_utils  # noqa: E402
from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network  # noqa: E402
from pcdet.utils import adabn_utils, common_utils  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cfg_file', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--adapt', choices=['none', 'ada'], required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--stats_frames', type=int, default=1000)
    p.add_argument('--stats_batch', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--run_name', default=None)
    args = p.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    assert cfg.MODEL.get('COUNT_NORM_CONV', None), 'item64_si_eval.py needs a config with MODEL.COUNT_NORM_CONV'
    np.random.seed(1024)
    eval_block = cfg.DATA_CONFIG_TAR if cfg.get('DATA_CONFIG_TAR', None) else cfg.DATA_CONFIG
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(out_dir / f'log_si_{datetime.datetime.now():%Y%m%d-%H%M%S}.txt', rank=0)
    wandb.init(config=vars(cfg), project='st3d', name=args.run_name, notes=common_utils.wandb_notes_with_job_id(),
               tags=common_utils.wandb_tags(cfg, 'item64_si_eval', cfg_file=args.cfg_file))
    wandb.config.update({'item64': dict(adapt=args.adapt, ckpt=args.ckpt, stats_frames=args.stats_frames)})
    for k, v in vars(args).items():
        logger.info(f'{k:16} {v}')
    log_config_to_file(cfg, logger=logger)
    ontology = cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None)
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=eval_block, class_names=cfg.CLASS_NAMES, batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False, model_ontology=ontology, use_subset=args.smoke)
    if args.smoke:
        test_set.kitti_infos = test_set.kitti_infos[:len(test_loader.dataset)]
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    ck = torch.load(args.ckpt, map_location='cpu')
    bad = sorted(set(model.state_dict()) ^ set(ck['model_state']))
    assert not bad, 'checkpoint does not match the count-normalized model: %s' % bad[:6]
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    logger.info('[si] %d count-normalized sparse convolutions' % model.n_count_norm)
    if args.adapt == 'ada':
        loader = adabn_utils.train_split_stats_loader(eval_block, cfg.CLASS_NAMES, ontology, args.stats_frames,
                                                      args.stats_batch, args.workers, logger)
        adabn_utils.reestimate_bn(model, loader, logger)
    eval_utils.eval_one_epoch(cfg, model, test_loader, args.adapt, logger, dist_test=False,
                              result_dir=out_dir, save_to_file=False, args=args)
    logger.info(f'[si] done: {out_dir}')


if __name__ == '__main__':
    main()
