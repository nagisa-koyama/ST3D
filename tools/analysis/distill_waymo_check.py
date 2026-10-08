"""Pre-launch checks for LiDAR Distillation on Waymo -> nuScenes (centerpoint-distill-waymo2nuscenes). CPU.

    python analysis/distill_waymo_check.py <cfg> <frames>

1. the ring labels (stored row order) on real frames: 64 TOP rows, the student's rows are every other declared beam;
2. real __getitem__ in training mode yields a (student, teacher) pair: points per stream, student / teacher ratio;
3. Car boxes with 0 points in the STUDENT cloud (both streams share gt_boxes) and in the teacher cloud;
4. loader seconds per sample (single process).
"""
import sys, time, pickle
import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils.beam_downsample_utils import generate_mask
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

cfg_file, n = sys.argv[1], int(sys.argv[2])
cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
bd = cfg.DATA_CONFIG.BEAM_DISTILL
ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                            workers=0, logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
calib = pickle.load(open(bd.TOP_CALIB, 'rb'))
step = max(1, len(ds.infos) // n); idx = list(range(0, len(ds.infos), step))[:n]
ok_rows = ok_even = 0; frac_top_kept = []; elev_gap = []
for i in idx:
    info = ds.infos[i]
    pts, lab = ds.get_lidar_with_ring_labels(info)
    top = lab >= 0
    ok_rows += len(np.unique(lab[top])) == 64
    m = generate_mask(np.zeros(len(lab)), lab, bd.NUM_BEAMS, bd.BEAM_RATIO, bd.BIN_RATIO)
    ok_even += set(np.unique(lab[m])) == set(range(0, 64, bd.BEAM_RATIO))
    frac_top_kept.append(m.sum() / max(top.sum(), 1))
    inc = np.sort(calib[info['point_cloud']['lidar_sequence']]['inclinations'])      # ascending = label order
    kept_inc = np.degrees(inc[sorted(set(np.unique(lab[m])))])
    band = (kept_inc[:-1] >= -5.4) & (kept_inc[1:] <= -0.5)
    elev_gap += list(np.diff(kept_inc)[band])
print(f'1. {len(idx)} frames: 64 TOP rows labelled {ok_rows}/{len(idx)}; student rows == every {bd.BEAM_RATIO}nd declared beam '
      f'{ok_even}/{len(idx)}; TOP points kept {np.median(frac_top_kept):.3f}; student line spacing in the car band '
      f'{np.min(elev_gap):.2f}-{np.max(elev_gap):.2f} deg (nuScenes 1.33)')
tot = emp_s = emp_t = 0; ns = []; nt = []; t0 = time.time(); fr = idx[:min(len(idx), 100)]
for i in fr:
    s, t = ds[i]
    ns.append(len(s['points'])); nt.append(len(t['points']))
    b = s['gt_boxes']; b = b[b[:, 7] == 1] if len(b) else b
    if not len(b): continue
    for d, acc in ((s, 's'), (t, 't')):
        c = roiaware_pool3d_utils.points_in_boxes_cpu(d['points'][:, :3].astype(np.float32), b[:, :7].astype(np.float32)).sum(1)
        if acc == 's': emp_s += int((c == 0).sum())
        else: emp_t += int((c == 0).sum())
    tot += len(b)
dt = (time.time() - t0) / len(fr)
print(f'2. points per sample after the processor: student {np.mean(ns):.0f}, teacher {np.mean(nt):.0f} (ratio {np.mean(ns)/np.mean(nt):.3f})')
print(f'3. Car training boxes with 0 points: student {emp_s}/{tot} = {emp_s/max(tot,1):.3f}, teacher {emp_t}/{tot} = {emp_t/max(tot,1):.3f}')
print(f'4. loader seconds per (student, teacher) sample: {dt:.3f}')
