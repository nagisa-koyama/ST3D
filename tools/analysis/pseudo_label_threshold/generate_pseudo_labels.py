"""Generate pseudo-labels for a self-training config's TARGET with a frozen teacher, standalone.

Exists so the stored score distribution is not truncated: a training run stores nothing below
`SELF_TRAIN.NEG_THRESH` (0.1 in the da-ieee-access family), which left every score-mixture fit in
experiments_md/20260926_06 weakly identified. Run at --thresh 0.0001 (the teacher's own
post-processing floor) the file holds every box the teacher emits, up to NMS_POST_MAXSIZE per
frame, in exactly the format `save_pseudo_label_epoch` writes - so every analysis script that
reads ps_label_e0.pkl works unchanged.

No W&B run is created (wandb is initialised disabled: `gather_and_dump_pseudo_label_result`
calls wandb.log). Run from tools/ on a GPU node:

    python analysis/pseudo_label_threshold/generate_pseudo_labels.py \
        --cfg_file cfgs/da-ieee-access/centerpoint-foreground-lyft2nuscenes.yaml \
        --teacher_ckpt /storage/wandb/run-20260923_093504-iwg6l5v1/files/ckpt/checkpoint_epoch_30.pth \
        --out_dir /storage/pseudo_labels/iwg6l5v1_ep30_nuscenes_train_thr0.0001 --thresh 0.0001
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _init_path  # noqa: F401
import torch  # noqa: F401
import wandb

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader, build_inference_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils, self_training_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--teacher_ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--thresh', type=float, default=0.0001)
    ap.add_argument('--batch_size', type=int, default=6)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max_obj', type=int, default=None,
                    help='override the teacher head\'s MAX_OBJ_PER_SAMPLE (heatmap top-K) and both '
                         'NMS_POST_MAXSIZE caps. The default 500 floors the stored scores at ~0.04 '
                         'by RANK: the 500th heatmap peak, before SCORE_THRESH is applied.')
    ap.add_argument('--teacher_adabn', type=float, default=None, metavar='MIX',
                    help='re-estimate the teacher\'s BatchNorm statistics on the target train split '
                         'before generating (as SELF_TRAIN.TEACHER_ADABN does in train.py), with this MIX '
                         '(1.0 = target statistics). Overrides whatever the config says.')
    ap.add_argument('--adabn_frames', type=int, default=1000)
    ap.add_argument('--adabn_layers', default=None, metavar='REGEX',
                    help='with --teacher_adabn: re-estimate only BN layers matching this regex (as '
                         "TEACHER_ADABN.LAYERS), e.g. '^backbone_3d\\.conv_input\\.1$' for the first BN")
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

    target_set, target_loader, _ = build_dataloader(
        cfg.DATA_CONFIG_TAR, cfg.CLASS_NAMES, args.batch_size, dist=False, workers=args.workers,
        logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY', None))
    teacher_cfg = cfg.SELF_TRAIN.MODEL_TEACHER
    if args.max_obj is not None:
        teacher_cfg.DENSE_HEAD.POST_PROCESSING.MAX_OBJ_PER_SAMPLE = args.max_obj
        teacher_cfg.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_PRE_MAXSIZE = max(4096, args.max_obj)
        teacher_cfg.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE = args.max_obj
        teacher_cfg.POST_PROCESSING.NMS_CONFIG.NMS_PRE_MAXSIZE = max(4096, args.max_obj)
        teacher_cfg.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE = args.max_obj
        logger.info('teacher MAX_OBJ_PER_SAMPLE / NMS_POST_MAXSIZE overridden to %d' % args.max_obj)
    teacher_classes = teacher_cfg.get('CLASS_NAMES', None) or cfg.CLASS_NAMES
    model = build_network(model_cfg=teacher_cfg, num_class=len(teacher_classes), dataset=target_set)
    model.load_params_from_file(filename=args.teacher_ckpt, to_cpu=False, logger=logger)
    model.cuda().eval()
    if args.teacher_adabn is not None:
        cfg.SELF_TRAIN.TEACHER_ADABN = {'FRAMES': args.adabn_frames, 'MIX': args.teacher_adabn}
        if args.adabn_layers:
            cfg.SELF_TRAIN.TEACHER_ADABN['LAYERS'] = args.adabn_layers
    if cfg.SELF_TRAIN.get('TEACHER_ADABN', None):
        # The same BN-adapted teacher train.py would use (experiments_md/20261004_01), so cuts derived
        # from these labels belong to the teacher that will actually generate them.
        from pcdet.utils.adabn_utils import adapt_teacher
        adapt_teacher(model, cfg, cfg.CLASS_NAMES, cfg.get('ONTOLOGY', None), logger, workers=args.workers)
        model.eval()

    # Same order as train_st_utils: dataset into eval mode BEFORE the inference loader's workers
    # fork on first iteration, so they stay in eval mode (20260921_02).
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
