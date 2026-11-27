"""Evaluate a range-stratified-BN checkpoint, optionally after per-knot AdaBN (experiments_md 20261003_05 §5 item 60, design
fixed 2026-11-27 12:09 JST, before any number).

    python analysis/item60_rsbn_eval.py --cfg_file <cfg with MODEL.RANGE_BN> --ckpt <ckpt> --adapt none|ada --out_dir DIR
        [--stats_frames 1000] [--stats_batch 4] [--run_name ...]

- `none`: the checkpoint's own per-knot statistics;
- `ada`: every knot's statistics re-estimated on `--stats_frames` strided frames of the eval dataset's TRAIN split (eval-mode
  view, no labels; pcdet/models/model_utils/range_bn.reestimate_range_bn); a knot never above the threshold keeps its source
  statistics;
- then evaluation on the eval block (DATA_CONFIG_TAR, else DATA_CONFIG): result.pkl and log_rsbn_*.txt in DIR, plus rsbn.json
  (per layer: training-time thin-knot fallbacks from the checkpoint; knots never seen in re-estimation).

The checkpoint must fill every state key of the RS-BN model exactly (checked). UDA-legal: unlabelled TRAIN clouds only.
"""
import argparse
import datetime
import json
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
from pcdet.models.model_utils import range_bn  # noqa: E402
from pcdet.utils import adabn_utils, common_utils  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cfg_file', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--adapt', choices=['none', 'ada'], required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--stats_frames', type=int, default=1000)
    p.add_argument('--stats_batch', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max_frames', type=int, default=0, help='smoke only: evaluate the first N frames')
    p.add_argument('--run_name', default=None)
    return p.parse_args()


def key_mismatch(model_keys, ckpt_keys):
    return sorted(set(model_keys) ^ set(ckpt_keys))


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    assert cfg.MODEL.get('RANGE_BN', None), 'item60_rsbn_eval.py needs a config with MODEL.RANGE_BN'
    np.random.seed(1024)
    eval_block = cfg.DATA_CONFIG_TAR if cfg.get('DATA_CONFIG_TAR', None) else cfg.DATA_CONFIG
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(out_dir / f'log_rsbn_{datetime.datetime.now():%Y%m%d-%H%M%S}.txt', rank=0)
    wandb.init(config=vars(cfg), project='st3d', name=args.run_name, notes=common_utils.wandb_notes_with_job_id(),
               tags=common_utils.wandb_tags(cfg, 'item60_rsbn_eval', cfg_file=args.cfg_file))
    wandb.config.update({'item60': dict(adapt=args.adapt, ckpt=args.ckpt, stats_frames=args.stats_frames,
                                        knots=list(cfg.MODEL.RANGE_BN.KNOTS))})
    for k, v in vars(args).items():
        logger.info(f'{k:16} {v}')
    log_config_to_file(cfg, logger=logger)

    ontology = cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None)
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=eval_block, class_names=cfg.CLASS_NAMES, batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False, model_ontology=ontology)
    if args.max_frames:
        test_set.kitti_infos = test_set.kitti_infos[:args.max_frames]
        logger.info('[rsbn] SMOKE: evaluating the first %d frames only' % args.max_frames)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    ck = torch.load(args.ckpt, map_location='cpu')
    bad = key_mismatch(model.state_dict().keys(), ck['model_state'].keys())
    assert not bad, 'checkpoint does not match the RS-BN model: %s' % bad[:6]
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    layers = range_bn.range_bn_layers(model)
    logger.info('[rsbn] %d RangeBN layers, knots %s' % (len(layers), list(cfg.MODEL.RANGE_BN.KNOTS)))
    info = {n: dict(train_thin_steps=m.thin_steps.tolist(), train_knot_batches=m.knot_batches.tolist()) for n, m in layers}
    if args.adapt == 'ada':
        loader = adabn_utils.train_split_stats_loader(eval_block, cfg.CLASS_NAMES, ontology, args.stats_frames,
                                                      args.stats_batch, args.workers, logger)
        never = range_bn.reestimate_range_bn(model, loader, logger)
        for n in info:
            info[n]['ada_knots_never_seen'] = never[n]
        info['_stats_frames'] = len(loader.dataset)
    json.dump(info, open(out_dir / 'rsbn.json', 'w'), indent=1)
    eval_utils.eval_one_epoch(cfg, model, test_loader, args.adapt, logger, dist_test=False,
                              result_dir=out_dir, save_to_file=False, args=args)
    logger.info(f'[rsbn] done: {out_dir}')


if __name__ == '__main__':
    main()
