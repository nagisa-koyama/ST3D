"""Population AdaBN, then evaluation (experiments_md 20261003_05 §5 item 62, design fixed 2026-11-27 12:53 JST, before any
number). ANALYSIS: P_K's BN statistics re-estimated on 1,000 strided KITTI train clouds with each layer's moments taken over
a subpopulation of its elements (pcdet/utils/adabn_population.py), then KITTI val.

    python analysis/item62_pop_adabn_eval.py --cfg_file <P_K's eval cfg> --ckpt <P_K ckpt> --arm all|far20|tbox|gtbox
        --seed 666|667 --out_dir DIR [--stats_frames 1000] [--stats_batch 4] [--smoke] [--run_name ...]

- `all`: every element (= global AdaBN; the procedure check against P_RS1);
- `far20`: elements beyond 20 m (x-y range), label-free;
- `tbox`: elements under the footprint of a teacher box: the same checkpoint under global AdaBN (P_RS1) on the same batch,
  kept at 59b's rule-A cuts of `--seed` (/home/koyama/data/item59/cuts.json), label-free;
- `gtbox`: under the KITTI train GT boxes' footprints (labelled reference).
result.pkl, log_popada_*.txt and pop.json (per layer: batches counted / skipped, member share; boxes per frame) in DIR.
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

from eval_utils import eval_utils  # noqa: E402
from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network  # noqa: E402
from pcdet.utils import adabn_population as ap, adabn_utils, common_utils  # noqa: E402

CUTS = '/home/koyama/data/item59/cuts.json'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cfg_file', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--arm', choices=['all', 'far20', 'tbox', 'gtbox'], required=True)
    p.add_argument('--seed', type=int, required=True, help='whose rule-A cuts the teacher uses (tbox)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--stats_frames', type=int, default=1000)
    p.add_argument('--stats_batch', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--smoke', action='store_true', help='smoke only: evaluate 16 frames')
    p.add_argument('--run_name', default=None)
    return p.parse_args()


def rule_a_cuts(seed, class_names, path=CUTS):
    """{1-based label: rule A's cut} for `seed` from 59b's P0 (every class of the model must have one)."""
    r = json.load(open(path))[str(seed)]['rule_a']
    return {i + 1: float(r[c]['neg']) for i, c in enumerate(class_names)}


def build(cfg_, test_set, ckpt, logger):
    m = build_network(model_cfg=cfg_.MODEL, num_class=len(cfg_.CLASS_NAMES), dataset=test_set)
    m.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    return m.cuda()


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    np.random.seed(1024)
    eval_block = cfg.DATA_CONFIG_TAR if cfg.get('DATA_CONFIG_TAR', None) else cfg.DATA_CONFIG
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(out_dir / f'log_popada_{datetime.datetime.now():%Y%m%d-%H%M%S}.txt', rank=0)
    wandb.init(config=vars(cfg), project='st3d', name=args.run_name, notes=common_utils.wandb_notes_with_job_id(),
               tags=common_utils.wandb_tags(cfg, 'item62_pop_adabn_eval', cfg_file=args.cfg_file))
    wandb.config.update({'item62': dict(arm=args.arm, ckpt=args.ckpt, seed=args.seed, stats_frames=args.stats_frames)})
    for k, v in vars(args).items():
        logger.info(f'{k:16} {v}')
    log_config_to_file(cfg, logger=logger)

    ontology = cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None)
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=eval_block, class_names=cfg.CLASS_NAMES, batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False, model_ontology=ontology, use_subset=args.smoke)
    if args.smoke:
        test_set.kitti_infos = test_set.kitti_infos[:len(test_loader.dataset)]
    stats = adabn_utils.train_split_stats_loader(eval_block, cfg.CLASS_NAMES, ontology, args.stats_frames,
                                                 args.stats_batch, args.workers, logger)
    model = build(cfg, test_set, args.ckpt, logger)
    pc_range = list(eval_block.POINT_CLOUD_RANGE)
    boxes_fn = None
    if args.arm == 'tbox':
        teacher = build(cfg, test_set, args.ckpt, logger)
        adabn_utils.reestimate_bn(teacher, stats, logger)            # P_RS1: global AdaBN on the same frames
        teacher.eval()
        cuts = rule_a_cuts(args.seed, cfg.CLASS_NAMES)
        logger.info('[pop-adabn] teacher: the checkpoint under global AdaBN, rule-A cuts %s' % cuts)
        boxes_fn = lambda batch: ap.teacher_boxes(teacher, batch, cuts)                     # noqa: E731
    elif args.arm == 'gtbox':
        def boxes_fn(batch):
            assert 'gt_boxes' in batch, 'gtbox: the statistics batches carry no gt_boxes'
            return ap.gt_boxes_of(batch)
    member = {'all': ap.member_all, 'far20': ap.member_far(20.0), 'tbox': ap.member_boxes,
              'gtbox': ap.member_boxes}[args.arm]
    _, diag = ap.reestimate_bn_population(model, stats, member, pc_range, logger, boxes_fn=boxes_fn)
    json.dump(dict(arm=args.arm, stats_frames=len(stats.dataset), **diag), open(out_dir / 'pop.json', 'w'), indent=1)
    eval_utils.eval_one_epoch(cfg, model, test_loader, args.arm, logger, dist_test=False,
                              result_dir=out_dir, save_to_file=False, args=args)
    logger.info(f'[pop-adabn] done: {out_dir}')


if __name__ == '__main__':
    main()
