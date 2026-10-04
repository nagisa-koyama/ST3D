"""Feature-statistics correction at TEST time (AdaBN): re-estimate BatchNorm statistics, then evaluate.

Why (experiments_md/20261004_01): UADA3D's loop forwards a source batch and a target batch through the
network in train mode every iteration, so its BatchNorm running statistics end up roughly half target.
Its +25 BEV on nuScenes -> KITTI (27174) may therefore come from normalisation statistics as much as
from the adversarial loss. This script separates the two on checkpoints that already exist, with no
training:

  --stats none     evaluate as saved (must reproduce the logged AP)
  --stats target   re-estimate every BN layer's statistics on the TARGET's TRAIN split (no labels read,
                   no augmentation, the eval-mode view the network will be scored on) = AdaBN
  --stats source   re-estimate them on the SOURCE's train split instead. On a source-only checkpoint
                   this should reproduce `none` (a check of the procedure); on a UADA3D checkpoint it
                   strips the target share out of its statistics, leaving only what the adversarial
                   loss changed in the weights
  --mix A          after re-estimation, use A * new + (1 - A) * saved statistics (A=0.5 on a
                   source-only checkpoint imitates UADA3D's half-target statistics)

UDA-legal: target statistics come from target TRAIN point clouds only, never val, never labels.
Only BatchNorm buffers change; no weight is touched. Statistics are a cumulative average over the
batches (momentum None). The gap between the saved and the re-estimated statistics is written per
layer to bn_gap.json, a direct measure of the feature-statistics shift each model leaves (e.g. how much
input-space accumulation already removed).

    python analysis/adabn_eval.py --cfg_file cfgs/... --ckpt ... --stats target [--stats_frames 1000]
        [--mix 1.0] [--extra_tag ...] [--eval_tag ...] [--run_name ...] [--set KEY VAL ...]
"""
import argparse
import copy
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
from torch.utils.data import DataLoader, Subset  # noqa: E402

from eval_utils import eval_utils  # noqa: E402
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file  # noqa: E402
from pcdet.datasets import __all__ as DATASETS, build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cfg_file', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--stats', choices=['none', 'target', 'source'], default='target')
    p.add_argument('--stats_frames', type=int, default=1000, help='frames of the train split, strided')
    p.add_argument('--stats_batch', type=int, default=4)
    p.add_argument('--mix', type=float, default=1.0)
    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--extra_tag', default='adabn')
    p.add_argument('--eval_tag', default=None)
    p.add_argument('--run_name', default=None)
    p.add_argument('--save_stats', action='store_true', help='also write saved/new BN statistics to bn_stats.npz')
    p.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    return p.parse_args()


def eval_dataset_config():
    if cfg.get('DATA_CONFIG_TAR', None):
        return cfg.DATA_CONFIG_TAR
    return cfg.DATA_CONFIG


def stats_dataset_config(which):
    """The dataset block whose TRAIN split is read, in eval mode (no augmentation)."""
    if which == 'target':
        block = eval_dataset_config()
    else:
        if cfg.get('DATA_CONFIG', None) is not None:
            block = cfg.DATA_CONFIG
        else:  # DATA_CONFIGS (per-platform sources): use the first; the caller is told which
            name, block = next(iter(cfg.DATA_CONFIGS.items()))
            print(f'[adabn] source statistics from DATA_CONFIGS.{name} only')
    block = copy.deepcopy(block)
    block.DATA_SPLIT['test'] = block.DATA_SPLIT['train']
    block.INFO_PATH['test'] = block.INFO_PATH['train']
    return block


def bn_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def reestimate(model, loader, logger):
    layers = bn_layers(model)
    saved = {n: (m.running_mean.clone(), m.running_var.clone()) for n, m in layers}
    momenta = {n: m.momentum for n, m in layers}
    model.eval()
    for _, m in layers:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over every batch seen
        m.train()
    for i, batch in enumerate(loader):
        load_data_to_gpu(batch)
        model(batch)
        if i % 50 == 0:
            logger.info(f'[adabn] statistics pass {i + 1}/{len(loader)}')
    for n, m in layers:
        m.momentum = momenta[n]
    model.eval()
    return saved


@torch.no_grad()
def bn_gap(model, saved):
    """Per layer: mean over channels of |mu_new - mu_old| / sigma_old and |log sigma_new / sigma_old|."""
    out = {}
    for n, m in bn_layers(model):
        mu0, var0 = saved[n]
        sd0 = (var0 + m.eps).sqrt()
        log_sd = 0.5 * torch.log((m.running_var + m.eps) / (var0 + m.eps))
        out[n] = dict(
            mean_shift=float(((m.running_mean - mu0).abs() / sd0).mean()),
            log_sd_ratio=float(log_sd.abs().mean()),
            max_abs_log_sd_ratio=float(log_sd.abs().max()),
            # channels whose new sd is under a tenth of the saved one: re-normalising them by the new
            # statistics multiplies their deviations by more than 10x
            frac_sd_collapse=float((log_sd < -np.log(10.0)).float().mean()),
            channels=int(mu0.numel()))
    return out


def summarise(gap):
    groups = {}
    for n, g in gap.items():
        if n.startswith('discriminator'):
            continue
        key = n.split('.')[0]
        groups.setdefault(key, []).append(g)
    return {k: dict(layers=len(v), mean_shift=float(np.mean([g['mean_shift'] for g in v])),
                    log_sd_ratio=float(np.mean([g['log_sd_ratio'] for g in v])),
                    max_abs_log_sd_ratio=float(np.max([g['max_abs_log_sd_ratio'] for g in v])),
                    frac_sd_collapse=float(np.mean([g['frac_sd_collapse'] for g in v]))) for k, v in groups.items()}


@torch.no_grad()
def apply_mix(model, saved, alpha):
    for n, m in bn_layers(model):
        mu0, var0 = saved[n]
        m.running_mean.mul_(alpha).add_((1 - alpha) * mu0)
        m.running_var.mul_(alpha).add_((1 - alpha) * var0)


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    np.random.seed(1024)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    epoch_id = ''.join(c for c in Path(args.ckpt).stem if c.isdigit()) or 'no_number'
    tag = args.eval_tag or f'stats_{args.stats}_mix{args.mix:g}_n{args.stats_frames}'
    out_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag / 'eval' / f'epoch_{epoch_id}' / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(out_dir / f'log_adabn_{datetime.datetime.now():%Y%m%d-%H%M%S}.txt', rank=0)
    wandb.init(config=vars(cfg), project='st3d', name=args.run_name, notes=common_utils.wandb_notes_with_job_id(),
               tags=common_utils.wandb_tags(cfg, 'adabn_eval', cfg_file=args.cfg_file))
    wandb.config.update({'adabn': dict(stats=args.stats, stats_frames=args.stats_frames, mix=args.mix, ckpt=args.ckpt)})
    for k, v in vars(args).items():
        logger.info(f'{k:16} {v}')
    log_config_to_file(cfg, logger=logger)

    ontology = cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None)
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=eval_dataset_config(), class_names=cfg.CLASS_NAMES, batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False, model_ontology=ontology)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()

    summary = {}
    if args.stats != 'none':
        block = stats_dataset_config(args.stats)
        ds = DATASETS[block.DATASET](dataset_cfg=block, class_names=cfg.CLASS_NAMES, root_path=None,
                                     training=False, logger=logger, model_ontology=ontology)
        stride = max(1, len(ds) // args.stats_frames)
        idx = list(range(0, len(ds), stride))[:args.stats_frames]
        logger.info(f'[adabn] {args.stats} statistics: {len(idx)} of {len(ds)} {block.DATA_SPLIT["test"]} frames '
                    f'of {block.DATASET} (stride {stride}), eval-mode view, no labels used')
        loader = DataLoader(Subset(ds, idx), batch_size=args.stats_batch, shuffle=False, num_workers=args.workers,
                            collate_fn=ds.collate_batch, pin_memory=True)
        saved = reestimate(model, loader, logger)
        gap = bn_gap(model, saved)
        summary = summarise(gap)
        for k, v in summary.items():
            logger.info(f'[adabn] BN gap {k:14s} layers={v["layers"]:3d} mean_shift={v["mean_shift"]:.3f} '
                        f'|log sd ratio|={v["log_sd_ratio"]:.3f} max={v["max_abs_log_sd_ratio"]:.2f} '
                        f'sd<0.1x={v["frac_sd_collapse"]:.4f}')
        json.dump(dict(per_layer=gap, summary=summary, frames=len(idx)), open(out_dir / 'bn_gap.json', 'w'), indent=1)
        if args.save_stats:
            arrays = {}
            for n, m in bn_layers(model):
                arrays[n + '|saved_mean'], arrays[n + '|saved_var'] = (t.cpu().numpy() for t in saved[n])
                arrays[n + '|new_mean'], arrays[n + '|new_var'] = m.running_mean.cpu().numpy(), m.running_var.cpu().numpy()
            np.savez(out_dir / 'bn_stats.npz', **arrays)
        wandb.config.update({'adabn_gap': summary})
        if args.mix != 1.0:
            apply_mix(model, saved, args.mix)
            logger.info(f'[adabn] statistics mixed: {args.mix} x new + {1 - args.mix} x saved')

    eval_utils.eval_one_epoch(cfg, model, test_loader, epoch_id, logger, dist_test=False,
                              result_dir=out_dir, save_to_file=False, args=args)
    logger.info(f'[adabn] done: {out_dir}')


if __name__ == '__main__':
    main()
