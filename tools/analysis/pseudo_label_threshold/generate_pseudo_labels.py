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
    ap.add_argument('--calib_from_cfg', default=None, metavar='TRAINING_CFG',
                    help='install the global density correction of TRAINING_CFG on the generated set: '
                         'measured exactly as train.py does for that config (its source DATA_CONFIG in '
                         'training mode against its DATA_CONFIG_TAR train split, HIST_DIST_* keys), then '
                         'applied to the clouds generated here. For a thinned-source teacher run over '
                         'its labelled SOURCE val (rule B), so the teacher sees its training input.')
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
    if args.calib_from_cfg:
        # Before any loader is iterated: workers fork a copy of the dataset (20260921_02).
        from easydict import EasyDict
        from pcdet.datasets import link_point_calibration
        from pcdet.datasets.point_calibration import calibration_target_config
        tcfg = cfg_from_yaml_file(args.calib_from_cfg, EasyDict())
        dc = tcfg.DATA_CONFIG
        assert dc.get('HIST_DIST_ON_THE_FLY', False), '--calib_from_cfg: that config has no on-the-fly correction'
        assert any(p.get('NAME') == 'sample_points_hist_based' for p in cfg.DATA_CONFIG_TAR.DATA_PROCESSOR), \
            'the generated set has no sample_points_hist_based step to install the correction on'
        meas_set, _, _ = build_dataloader(dc, tcfg.CLASS_NAMES, 1, dist=False, workers=0, logger=logger,
                                          training=True, model_ontology=tcfg.get('ONTOLOGY', None))
        calib_cfg, calib_split = calibration_target_config(tcfg.DATA_CONFIG_TAR)
        calib_tgt, _, _ = build_dataloader(calib_cfg, tcfg.CLASS_NAMES, 1, dist=False, workers=0, logger=logger,
                                           training=False, model_ontology=tcfg.get('ONTOLOGY', None))
        logger.info('calibration as in %s: source %s train, target measured on its %s split'
                    % (args.calib_from_cfg, dc.DATASET, calib_split))
        max_dist = dc.get('HIST_DIST_MAX_DIST', 75.0)
        src, tgt = link_point_calibration(
            meas_set, calib_tgt, num_frames=dc.get('HIST_DIST_FRAMES', 1000), num_bins=dc.get('HIST_DIST_BINS', 50),
            max_dist=max_dist, logger=logger, fov_degree=dc.get('HIST_DIST_FOV_DEGREE', None),
            fov_heading=dc.get('HIST_DIST_FOV_HEADING', 0.0))
        target_set.data_processor.set_hist_dist(src, tgt, max_dist=max_dist)
        rate = target_set.data_processor.per_bin_sample_rate()
        logger.info('installed on the generated set: sample rate %.2f..%.2f, below 1 in %d of %d bins'
                    % (rate.min(), rate.max(), int((rate < 1).sum()), len(rate)))
        del meas_set, calib_tgt
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
