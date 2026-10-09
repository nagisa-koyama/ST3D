"""Bug-prevention checklist item 4 (experiments_md memory/repo/bug_prevention_checklist.md): dump assigned targets of real batches and compare them with the input labels.

    python analysis/target_dump.py <cfg> <DATA_CONFIGS key | DATA_CONFIG | TAR> <n_frames> [ps_label_e0.pkl]

For each frame, after prepare_data + collate + CenterHead.assign_targets (the row's real head geometry):
- input labels: positives per class, pseudo ignore band (-1..-3) per class, IGNORE_CLASS_LABEL;
- assigned: heatmap peaks per channel, regression targets, ignored footprint pixels;
- checks: peaks per channel == positive boxes of that class whose centre pixel is distinct and inside the map;
  no peak at an ignored box's centre pixel unless a positive shares that pixel; every ignored box's centre pixel
  has weight 0 unless a positive shares it; no weight-0 pixel holds a peak.
"""
import sys, pickle
import numpy as np, torch
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models.dense_heads.center_head import CenterHead
from pcdet.utils import common_utils

cfg_file, key, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
dcfg = cfg.DATA_CONFIG_TAR if key == 'TAR' else (cfg.DATA_CONFIG if key == 'DATA_CONFIG' else cfg.DATA_CONFIGS[key])
ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                            logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
# TARGET_DUMP_CALIB_FRAMES=<n> (env, opt-in): install the global density calibration first, as train.py does for a
# HIST_DIST_ON_THE_FLY source, so a sample_points_hist_based step is live (otherwise it is a no-op here).
import os
if os.environ.get('TARGET_DUMP_CALIB_FRAMES') and dcfg.get('HIST_DIST_ON_THE_FLY', False):
    from pcdet.datasets.point_calibration import calibration_target_config, link_point_calibration
    tcfg, _ = calibration_target_config(cfg.DATA_CONFIG_TAR)
    tds, _, _ = build_dataloader(dataset_cfg=tcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                 logger=common_utils.create_logger(), training=False, model_ontology=cfg.get('ONTOLOGY'))
    link_point_calibration(ds, tds, num_frames=int(os.environ['TARGET_DUMP_CALIB_FRAMES']),
                           num_bins=dcfg.get('HIST_DIST_BINS', 50), max_dist=dcfg.get('HIST_DIST_MAX_DIST', 75.0),
                           logger=common_utils.create_logger(),
                           skip_point_budget=dcfg.get('HIST_DIST_BEFORE_POINT_BUDGET', False))
if len(sys.argv) > 4:
    ps = pickle.load(open(sys.argv[4], 'rb'))
    ds.set_pseudo_labels(ps)
    frames = [i for i, info in enumerate(ds.kitti_infos) if info['point_cloud']['lidar_idx'] in ps] if hasattr(ds, 'kitti_infos') else range(len(ds))
else:
    frames = range(len(ds))
frames = list(frames)
head = CenterHead.__new__(CenterHead); torch.nn.Module.__init__(head)
head.model_cfg = cfg.MODEL.DENSE_HEAD
head.class_names = list(cfg.CLASS_NAMES); head.class_names_each_head = [list(cfg.CLASS_NAMES)]
head.point_cloud_range = np.array(dcfg.POINT_CLOUD_RANGE, dtype=np.float32)
head.voxel_size = [p for p in dcfg.DATA_PROCESSOR if p.NAME == 'transform_points_to_voxels'][0].VOXEL_SIZE
tac = cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG
stride = tac.FEATURE_MAP_STRIDE
side = int(round((head.point_cloud_range[3] - head.point_cloud_range[0]) / head.voxel_size[0] / stride))
C = len(cfg.CLASS_NAMES)
tot = dict(pos=np.zeros(C), band=np.zeros(C), other=0, peaks=np.zeros(C), expected=np.zeros(C), reg=0, ign_px=0, px=0)
fails = []
step = max(1, len(frames) // n)
for fi in frames[::step][:n]:
    d = ds[fi]
    if isinstance(d, tuple):   # BEAM_DISTILL yields (student, teacher); the student's targets are what is trained
        d = d[0]
    gt = d['gt_boxes']
    lab = gt[:, 7].astype(int)
    for c in range(C):
        tot['pos'][c] += (lab == c + 1).sum(); tot['band'][c] += (lab == -(c + 1)).sum()
    tot['other'] += (lab == common_utils.IGNORE_CLASS_LABEL).sum()
    t = head.assign_targets(torch.from_numpy(gt[None]).float().clone(), feature_map_size=[side, side])
    hm, mask = t['heatmaps'][0][0], t['heatmap_masks'][0]
    mask = torch.ones(side, side) if mask is None else mask[0]
    cell = head.voxel_size[0] * stride
    def pix(b):
        x = min(max((b[0] - head.point_cloud_range[0]) / cell, 0), side - 0.5)
        y = min(max((b[1] - head.point_cloud_range[1]) / cell, 0), side - 0.5)
        return int(y), int(x)
    pos_pix = {}
    for b, l in zip(gt, lab):
        if l > 0 and b[3] > 0 and b[4] > 0:
            pos_pix.setdefault(l - 1, set()).add(pix(b))
    for c in range(C):
        peaks = int(hm[c].eq(1).sum()); exp = len(pos_pix.get(c, ()))
        tot['peaks'][c] += peaks; tot['expected'][c] += exp
        if peaks != exp:
            fails.append(f'frame {fi}: channel {cfg.CLASS_NAMES[c]} peaks {peaks} != positive pixels {exp}')
    all_pos = set().union(*pos_pix.values()) if pos_pix else set()
    for b, l in zip(gt, lab):
        if l < 0 and b[3] > 0 and b[4] > 0:
            r, c_ = pix(b)
            if (r, c_) not in all_pos:
                if hm[:, r, c_].max() >= 1:
                    fails.append(f'frame {fi}: ignored box (label {l}) got a peak')
                if mask[r, c_] != 0:
                    fails.append(f'frame {fi}: ignored box (label {l}) centre has weight {float(mask[r, c_])}')
    if bool(((mask == 0) & hm.eq(1).any(0)).any()):
        fails.append(f'frame {fi}: a positive peak was ignored')
    tot['reg'] += int(t['masks'][0].sum()); tot['ign_px'] += int((mask == 0).sum()); tot['px'] += side * side
print(f'{cfg_file} [{key}], {min(n, len(frames))} frames')
for c in range(C):
    print(f'  {cfg.CLASS_NAMES[c]:10s} input positives {int(tot["pos"][c]):5d}  ignore band {int(tot["band"][c]):5d}  '
          f'heatmap peaks {int(tot["peaks"][c]):5d} (distinct positive pixels {int(tot["expected"][c])})')
print(f'  IGNORE_CLASS_LABEL boxes {tot["other"]}; regression targets {tot["reg"]} (input positives {int(tot["pos"].sum())}); '
      f'ignored footprint {tot["ign_px"] / tot["px"]:.4f} of pixels')
print('  checks:', 'ALL PASS' if not fails else f'{len(fails)} FAIL'); [print('   ', f) for f in fails[:10]]
