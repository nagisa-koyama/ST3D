"""Range-stratified AdaBN (RS-AdaBN) at test time, then evaluation (experiments_md 20261003_05 §5 item 57, design fixed
2026-11-27 00:10 JST, experiments_md commit b430082, before any number). Only the normalization changes; no weight is touched.

    python analysis/rs_adabn_eval.py --cfg_file <eval cfg> --src_cfg <cfg whose DATA_CONFIG_TAR train split is the source>
        --ckpt <ckpt> --knots item57|global [--stats_frames 1000] [--extra_tag ...] [--eval_tag ...] [--run_name ...]

- source pass: the checkpoint as saved, on `--stats_frames` strided frames of the source TRAIN clouds (the eval-mode view of
  `--src_cfg`'s DATA_CONFIG_TAR with its train split; for item 57 that is S' train, P_K's training cut, no augmentation);
- target pass: the same number of strided frames of the eval dataset's TRAIN split, batch `--stats_batch`, per knot
  (pcdet/utils/adabn_range.py); `global` = one knot = global AdaBN;
- evaluation on the eval dataset as `adabn_eval.py` does it; result.pkl and a log_rsadabn_*.txt in
  output/<group>/<tag>/<extra_tag>/eval/epoch_<N>/<eval_tag>/, plus stage0.json (item 57 stage 0: per BN layer and knot, the
  element shares and the normalized-moment mismatch between the source pass and this target pass).

UDA-legal: unlabelled TRAIN clouds of source and target, knots fixed a priori. ANALYSIS rows in item 57 (target labels score).
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
import wandb  # noqa: E402
from easydict import EasyDict  # noqa: E402

from eval_utils import eval_utils  # noqa: E402
from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network  # noqa: E402
from pcdet.utils import adabn_range, adabn_utils, common_utils  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cfg_file', required=True)
    p.add_argument('--src_cfg', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--knots', choices=['item57', 'item57_3d', 'global'], required=True,
                   help='item57_3d (item 57b): the 5 knots in backbone_3d only, one knot (global AdaBN) elsewhere')
    p.add_argument('--stats_frames', type=int, default=1000)
    p.add_argument('--stats_batch', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--extra_tag', default='rsadabn')
    p.add_argument('--eval_tag', default=None)
    p.add_argument('--run_name', default=None)
    return p.parse_args()


def group_summary(per_layer):
    out = {}
    for n, d in per_layer.items():
        g = out.setdefault(n.split('.')[0], dict(layers=0, d_mean=[], d_logsd=[], share_src=[], share_tgt=[]))
        g['layers'] += 1
        for k in ('d_mean', 'd_logsd', 'share_src', 'share_tgt'):
            g[k].append(d[k])
    return {g: dict(layers=v['layers'], **{k: np.mean(np.array(v[k]), 0).tolist()
                                            for k in ('d_mean', 'd_logsd', 'share_src', 'share_tgt')}) for g, v in out.items()}


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    np.random.seed(1024)
    src_cfg = cfg_from_yaml_file(args.src_cfg, EasyDict())
    eval_block = cfg.DATA_CONFIG_TAR if cfg.get('DATA_CONFIG_TAR', None) else cfg.DATA_CONFIG
    src_block = src_cfg.DATA_CONFIG_TAR
    pc_range = list(eval_block.POINT_CLOUD_RANGE)
    assert list(src_block.POINT_CLOUD_RANGE) == pc_range, 'source and target must share the model input range'

    epoch_id = ''.join(c for c in Path(args.ckpt).stem if c.isdigit()) or 'no_number'
    tag = args.eval_tag or 'rsadabn_%s_n%d' % (args.knots, args.stats_frames)
    out_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag / 'eval' / f'epoch_{epoch_id}' / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(out_dir / f'log_rsadabn_{datetime.datetime.now():%Y%m%d-%H%M%S}.txt', rank=0)
    wandb.init(config=vars(cfg), project='st3d', name=args.run_name, notes=common_utils.wandb_notes_with_job_id(),
               tags=common_utils.wandb_tags(cfg, 'rs_adabn_eval', cfg_file=args.cfg_file))
    knots = adabn_range.KNOTS_ITEM57 if args.knots in ('item57', 'item57_3d') else (0.0,)
    layers = r'^backbone_3d\.' if args.knots == 'item57_3d' else None
    wandb.config.update({'rs_adabn': dict(knots=list(knots), layers=layers, src_cfg=args.src_cfg, stats_frames=args.stats_frames,
                                          ckpt=args.ckpt)})
    for k, v in vars(args).items():
        logger.info(f'{k:16} {v}')
    log_config_to_file(cfg, logger=logger)

    ontology = cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None)
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=eval_block, class_names=cfg.CLASS_NAMES, batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False, model_ontology=ontology)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()

    rs = adabn_range.RangeStratifiedBN(model, pc_range)
    logger.info('[rs-adabn] source: TRAIN split of %s (%s); target: TRAIN split of the eval dataset; knots %s; %d BN layers'
                % (args.src_cfg, src_block.get('EVAL_FACTOR_SWAP', None), list(knots), len(rs.layers)))
    src_loader = adabn_utils.train_split_stats_loader(src_block, cfg.CLASS_NAMES, ontology, args.stats_frames,
                                                      args.stats_batch, args.workers, logger)
    src = rs.collect(src_loader, adabn_range.KNOTS_ITEM57, logger)
    tgt_loader = adabn_utils.train_split_stats_loader(eval_block, cfg.CLASS_NAMES, ontology, args.stats_frames,
                                                      args.stats_batch, args.workers, logger)
    diag = rs.estimate(tgt_loader, knots, logger, diag_knots=adabn_range.KNOTS_ITEM57, layers=layers)
    per_layer = {n: adabn_range.mismatch(src[n], diag[n]) for n, _ in rs.layers}
    groups = group_summary(per_layer)
    for g, v in groups.items():
        logger.info('[rs-adabn] stage 0 %-12s layers=%3d share src %s tgt %s | |o_t - o_s| %s | |log r_t/r_s| %s' % (
            g, v['layers'], np.round(v['share_src'], 3).tolist(), np.round(v['share_tgt'], 3).tolist(),
            np.round(v['d_mean'], 3).tolist(), np.round(v['d_logsd'], 3).tolist()))
    json.dump(dict(knots=list(adabn_range.KNOTS_ITEM57), arm_knots=list(knots), layers=layers, per_layer=per_layer, groups=groups,
                   src_frames=len(src_loader.dataset), tgt_frames=len(tgt_loader.dataset)),
              open(out_dir / 'stage0.json', 'w'), indent=1)
    rs.apply()
    eval_utils.eval_one_epoch(cfg, model, test_loader, epoch_id, logger, dist_test=False,
                              result_dir=out_dir, save_to_file=False, args=args)
    logger.info(f'[rs-adabn] done: {out_dir}')


if __name__ == '__main__':
    main()
