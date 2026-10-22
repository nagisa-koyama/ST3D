"""generate_pseudo_labels.py for a BN-ADAPTED teacher whose statistics come from ANOTHER config's target.

Rule B for an AdaBN-adapted teacher (experiments_md/20261022_07 §2.1): the teacher that labels the target carries
target (nuScenes TRAIN) BatchNorm statistics, and rule B scores THAT network on the labelled SOURCE val. The rule-B
flat config's own target slot is the source val (Waymo val through RING_PATTERN), so generate_pseudo_labels.py's
`--teacher_adabn` would re-estimate on Waymo. This script adapts the teacher with `--stats_cfg`'s
`SELF_TRAIN.TEACHER_ADABN` on `--stats_cfg`'s `DATA_CONFIG_TAR` (exactly `adabn_utils.adapt_teacher`, the code
train.py runs), then generates on `--cfg_file`'s target as generate_pseudo_labels.py does. Output format identical
(ps_label_e0.pkl + generate.log). No W&B. Run from tools/ on a GPU node:

    python analysis/pseudo_label_threshold/gen_adapted_teacher_pass.py \
        --cfg_file analysis/pseudo_label_threshold/cfgs/teacher_27807_on_waymo_val_int5_ringpattern.yaml \
        --stats_cfg cfgs/da-ieee-access/centerpoint-ringpattern-st3d-dsbn-adabnteacher-waymo2nuscenes.yaml \
        --teacher_ckpt /storage/wandb/run-20261008_212958-87h61vfd/files/ckpt/checkpoint_epoch_7.pth \
        --out_dir /storage/pseudo_labels/87h61vfd_ep7_adabn_waymo_val_int5_ringpattern_thr0.0001
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _init_path  # noqa: F401
import torch  # noqa: F401
import wandb
from easydict import EasyDict

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader, build_inference_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils, self_training_utils
from pcdet.utils.adabn_utils import adapt_teacher


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg_file', required=True, help='flat rule-B config: its DATA_CONFIG_TAR is what gets labelled')
    ap.add_argument('--stats_cfg', required=True,
                    help='training config whose SELF_TRAIN.TEACHER_ADABN and DATA_CONFIG_TAR give the BN statistics')
    ap.add_argument('--teacher_ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--thresh', type=float, default=0.0001)
    ap.add_argument('--batch_size', type=int, default=6)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(out / 'generate.log')
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.LOCAL_RANK = 0
    nc = len(cfg.CLASS_NAMES)
    cfg.SELF_TRAIN.SCORE_THRESH = [args.thresh] * nc
    cfg.SELF_TRAIN.NEG_THRESH = [args.thresh] * nc
    cfg.SELF_TRAIN.INIT_PS = None
    logger.info('pseudo-label thresholds overridden: SCORE_THRESH = NEG_THRESH = %g' % args.thresh)
    wandb.init(mode='disabled')

    stats_cfg = cfg_from_yaml_file(args.stats_cfg, EasyDict())
    assert stats_cfg.SELF_TRAIN.get('TEACHER_ADABN', None), '--stats_cfg has no SELF_TRAIN.TEACHER_ADABN'
    logger.info('BN statistics: %s of %s %s (TEACHER_ADABN %s); generated set: %s %s of %s'
                % (stats_cfg.DATA_CONFIG_TAR.DATASET, stats_cfg.DATA_CONFIG_TAR.DATA_SPLIT.get('train'), args.stats_cfg,
                   dict(stats_cfg.SELF_TRAIN.TEACHER_ADABN), cfg.DATA_CONFIG_TAR.DATASET,
                   dict(cfg.DATA_CONFIG_TAR.DATA_SPLIT), args.cfg_file))

    target_set, target_loader, _ = build_dataloader(
        cfg.DATA_CONFIG_TAR, cfg.CLASS_NAMES, args.batch_size, dist=False, workers=args.workers,
        logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY', None))
    teacher_cfg = cfg.SELF_TRAIN.MODEL_TEACHER
    teacher_classes = teacher_cfg.get('CLASS_NAMES', None) or cfg.CLASS_NAMES
    model = build_network(model_cfg=teacher_cfg, num_class=len(teacher_classes), dataset=target_set)
    model.load_params_from_file(filename=args.teacher_ckpt, to_cpu=False, logger=logger)
    model.cuda().eval()
    # The same BN-adapted teacher train.py builds for --stats_cfg: statistics from ITS target's train split.
    adapt_teacher(model, stats_cfg, stats_cfg.CLASS_NAMES, stats_cfg.get('ONTOLOGY', None), logger, workers=args.workers)
    model.eval()

    target_set.eval()
    gen_loader = build_inference_dataloader(target_loader)
    self_training_utils.save_pseudo_label_epoch(model, gen_loader, rank=0, leave_pbar=True,
                                                ps_label_dir=str(out), cur_epoch=0)
    n = sum(len(v['gt_boxes']) for v in self_training_utils.PSEUDO_LABELS.values())
    logger.info('wrote %s: %d frames, %d boxes (%.1f / frame)'
                % (out / 'ps_label_e0.pkl', len(self_training_utils.PSEUDO_LABELS), n,
                   n / max(len(self_training_utils.PSEUDO_LABELS), 1)))


if __name__ == '__main__':
    main()
