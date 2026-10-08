"""Pre-launch check of a GLOBAL-thinning source row with a processor-stage recount (`drop_empty_gt_boxes`): install the
calibration as train.py does, then measure what the processor does to points and to Car training boxes.

Built for C1, the attribution control of 27807 (experiments_md 20261005_01 §15.6): TOP-only Waymo + per-bin thinning
+ recount. Reports, over strided source TRAIN frames (training mode, as trained):
- points per frame within 75 m before / after the processor, against the target TRAIN cloud's (eval-mode loader);
- the radial profile after thinning / target, per ring;
- Car boxes entering the processor: holding 0 points already (kept by the stored count), emptied by the thinning,
  and left with 0 points after the recount (must be 0).

    python analysis/thinning_recount_check.py <cfg> <calibration frames> <source frames> [<target frames>]
"""
import sys
import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import calibration_target_config, link_point_calibration
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

cfg_file, n_calib, n_src = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
n_tgt = int(sys.argv[4]) if len(sys.argv) > 4 else n_src
cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
logger = common_utils.create_logger()
dc = cfg.DATA_CONFIG
src, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                             logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
tcfg, split = calibration_target_config(cfg.DATA_CONFIG_TAR)
tgt, _, _ = build_dataloader(dataset_cfg=tcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                             logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
link_point_calibration(src, tgt, num_frames=n_calib, num_bins=dc.get('HIST_DIST_BINS', 50),
                       max_dist=dc.get('HIST_DIST_MAX_DIST', 75.0), logger=logger)

RINGS = [0, 7.5, 15, 30, 45, 60, 75]


def ring_counts(p):
    r = np.hypot(p[:, 0], p[:, 1])
    return np.histogram(r[r < 75], bins=RINGS)[0]


proc = src.data_processor
orig = proc.forward
cap = {}


def spy(data_dict):
    cap['pts_in'] = data_dict['points'][:, :3].copy()
    cap['box_in'] = None if data_dict.get('gt_boxes') is None else data_dict['gt_boxes'].copy()
    return orig(data_dict=data_dict)


proc.forward = spy
pin, pout, rin, rout = [], [], np.zeros(len(RINGS) - 1), np.zeros(len(RINGS) - 1)
car = dict(entering=0, zero_before=0, emptied_by_thinning=0, zero_after_recount=0, dropped=0, kept=0)
for i in range(0, len(src), max(1, len(src) // n_src))[:n_src]:
    d = src[i]
    p_in, p_out = cap['pts_in'], d['points'][:, :3]
    pin.append((np.hypot(p_in[:, 0], p_in[:, 1]) < 75).sum()); pout.append((np.hypot(p_out[:, 0], p_out[:, 1]) < 75).sum())
    rin += ring_counts(p_in); rout += ring_counts(p_out)
    b_in = cap['box_in']
    if b_in is None or not len(b_in):
        continue
    b_in = b_in[b_in[:, 7] == 1]                      # Car, as it entered the processor
    if len(b_in):
        c_before = roiaware_pool3d_utils.points_in_boxes_cpu(p_in, b_in[:, :7]).sum(1)
        c_after = roiaware_pool3d_utils.points_in_boxes_cpu(p_out, b_in[:, :7]).sum(1) if len(p_out) else np.zeros(len(b_in))
        car['entering'] += len(b_in); car['zero_before'] += int((c_before == 0).sum())
        car['emptied_by_thinning'] += int(((c_before > 0) & (c_after == 0)).sum())
    b_out = d['gt_boxes']; b_out = b_out[b_out[:, 7] == 1]
    car['kept'] += len(b_out)
    if len(b_out):
        car['zero_after_recount'] += int((roiaware_pool3d_utils.points_in_boxes_cpu(p_out, b_out[:, :7]).sum(1) == 0).sum())
car['dropped'] = car['entering'] - car['kept']

tp, tr = [], np.zeros(len(RINGS) - 1)
for i in range(0, len(tgt), max(1, len(tgt) // n_tgt))[:n_tgt]:
    p = tgt[i]['points'][:, :3]
    tp.append((np.hypot(p[:, 0], p[:, 1]) < 75).sum()); tr += ring_counts(p)

ns, nt = len(pin), len(tp)
print(f'\n{cfg_file}: calibration {n_calib} frames, {ns} source frames, {nt} target {split} frames')
print(f'points per frame within 75 m: source before processor {np.mean(pin):.0f}, after {np.mean(pout):.0f}; '
      f'target {np.mean(tp):.0f} (after / target {np.mean(pout) / np.mean(tp):.2f})')
print('| ring (m) | ' + ' | '.join(f'{RINGS[k]}-{RINGS[k + 1]}' for k in range(len(RINGS) - 1)) + ' |')
print('|---' * len(RINGS) + '|')
print('| after / target | ' + ' | '.join(f'{(rout[k] / ns) / max(tr[k] / nt, 1e-9):.2f}' for k in range(len(RINGS) - 1)) + ' |')
print('| before / target | ' + ' | '.join(f'{(rin[k] / ns) / max(tr[k] / nt, 1e-9):.2f}' for k in range(len(RINGS) - 1)) + ' |')
e = max(car['entering'], 1)
print(f"Car boxes entering the processor: {car['entering']}; already 0 points (stored count kept them): "
      f"{car['zero_before']} ({car['zero_before'] / e:.1%}); emptied by the thinning: {car['emptied_by_thinning']} "
      f"({car['emptied_by_thinning'] / e:.1%}); dropped by the processor (range + recount): {car['dropped']} "
      f"({car['dropped'] / e:.1%}); kept with 0 points after the recount: {car['zero_after_recount']}")
