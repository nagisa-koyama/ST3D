"""Rates for FACTOR-SWAP S2 R10, the foreground-return row (experiments_md 20261003_05 §5 item 24, design 2026-11-02).

ANALYSIS (KITTI and nuScenes labels are read; the row they feed is a labelled-analysis oracle). Both TRAIN splits go
through their own loaders in eval mode (points range-masked, in the loader frame), the boxes come from the infos:
  - foreground, per class (Car, Pedestrian): points per box inside the box enlarged by BOX_MARGIN, per 5 m bin of the
    box centre's horizontal radius, on UNOCCLUDED boxes only (KITTI `occluded` == 0; nuScenes visibility token '4',
    80-100%), with the nuScenes boxes first shrunk to the KITTI size convention (dims / the R6 ratio from stats.json)
    so R6's term is not counted again. RATE_FG = nuScenes / KITTI points per box, capped at 1; -1 where either side
    has fewer than MIN_BOXES boxes (the loader then uses the background rate);
  - background: points per frame per 1.5 m bin outside every Car / Pedestrian box (all of them, enlarged and, for
    nuScenes, shrunk the same way), counted in the forward sector |azimuth| <= 40 deg on both sides (KITTI labels
    only its camera FOV, so its cars outside it would otherwise count as background). RATE_BG = nuScenes / KITTI,
    capped at 1, bins under 1% of the mean on either side left at 1.
Read-only, CPU; run as a Slurm CPU job:

    python analysis/factorswap_s2_fgbg_stats.py <stats.json from factorswap_s2_stats.py> <out.json>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
os.chdir(TOOLS)
import _init_path  # noqa: E402,F401
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.datasets.osens2_utils import enlarge_boxes  # noqa: E402
from pcdet.datasets.point_calibration import calibration_target_config  # noqa: E402
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

import torch  # noqa: E402

NUS_CFG = 'cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti.yaml'
KITTI_CFG = 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml'
SAMPLE_ANNOTATION = '/home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval/v1.0-trainval/sample_annotation.json'
CLASSES = ('Car', 'Pedestrian')
BOX_MARGIN = 0.25
EDGES_FG = np.arange(0, 80, 5.0)          # 15 bins to 75 m
EDGES_BG = np.linspace(0, 75.0, 51)       # 50 bins of 1.5 m, as R3
SECTOR_DEG = 40.0
MIN_BOXES = 30
NUS_FRAMES = 2000
logger = common_utils.create_logger()


def build(cfg_path):
    cfg = cfg_from_yaml_file(cfg_path, EasyDict())
    dcfg, _ = calibration_target_config(cfg.DATA_CONFIG, 'train')
    ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    return dcfg, ds


def in_boxes(points, boxes):
    """(n_boxes, n_points) membership."""
    if len(boxes) == 0 or len(points) == 0:
        return np.zeros((len(boxes), len(points)), bool)
    return roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(np.ascontiguousarray(points[:, :3], np.float32)),
        torch.from_numpy(np.ascontiguousarray(boxes[:, :7], np.float32))).numpy().astype(bool)


class Acc:
    def __init__(self):
        self.fg_pts = {c: np.zeros(len(EDGES_FG) - 1) for c in CLASSES}
        self.fg_n = {c: np.zeros(len(EDGES_FG) - 1) for c in CLASSES}
        self.bg = np.zeros(len(EDGES_BG) - 1)
        self.frames = 0

    def add(self, points, boxes, names, unocc, forward_axis):
        boxes = enlarge_boxes(boxes, BOX_MARGIN) if len(boxes) else np.zeros((0, 7), np.float32)
        m = in_boxes(points, boxes)
        for k, (nm, ok) in enumerate(zip(names, unocc)):
            if ok and nm in CLASSES:
                b = np.clip(np.searchsorted(EDGES_FG, np.hypot(boxes[k, 0], boxes[k, 1]), 'right') - 1, 0,
                            len(EDGES_FG) - 2)
                self.fg_pts[nm][b] += m[k].sum()
                self.fg_n[nm][b] += 1
        outside = ~m.any(axis=0) if len(boxes) else np.ones(len(points), bool)
        fwd, lat = (points[:, 0], points[:, 1]) if forward_axis == 'x' else (points[:, 1], -points[:, 0])
        sector = np.abs(np.degrees(np.arctan2(lat, fwd))) <= SECTOR_DEG
        p = points[outside & sector]
        self.bg += np.histogram(np.clip(np.hypot(p[:, 0], p[:, 1]), 0, EDGES_BG[-1] - 1e-4), bins=EDGES_BG)[0]
        self.frames += 1


def main(stats_path, out_path):
    ratio = {c: json.load(open(stats_path))['size'][c]['ratio'] for c in CLASSES}
    vis = {a['token']: a['visibility_token'] for a in json.load(open(SAMPLE_ANNOTATION))}

    kd, kds = build(KITTI_CFG)
    shift_k = np.asarray(kd.SHIFT_COOR, np.float32)
    acc_k = Acc()
    for i in range(len(kds)):
        info = kds.kitti_infos[i]
        a = info['annos']
        boxes = a['gt_boxes_lidar'].astype(np.float32).copy()
        boxes[:, :3] += shift_k
        names = a['name'][:len(boxes)]
        unocc = a['occluded'][:len(boxes)] == 0
        keep = np.isin(names, CLASSES)
        acc_k.add(kds[i]['points'], boxes[keep], names[keep], unocc[keep], 'x')

    nd, nds = build(NUS_CFG)
    shift_n = np.asarray(nd.SHIFT_COOR, np.float32)
    to_model = nds.map_ontology_dataset_to_model or {}
    acc_n = Acc()
    step = max(1, len(nds) // NUS_FRAMES)
    for i in range(0, len(nds), step):
        info = nds.infos[i]
        names = np.array([to_model.get(n, n) for n in info['gt_names']])
        keep = np.isin(names, CLASSES)
        boxes = info['gt_boxes'][keep, :7].astype(np.float32).copy()
        names = names[keep]
        for c in CLASSES:
            boxes[names == c, 3:6] /= np.float32(ratio[c])
        boxes[:, :3] += shift_n
        unocc = np.array([vis.get(t, '0') == '4' for t in info['gt_boxes_token'][keep]], bool)
        acc_n.add(nds[i]['points'], boxes, names, unocc, 'y')

    res = {'kitti_cfg': KITTI_CFG, 'nuscenes_cfg': NUS_CFG, 'box_margin': BOX_MARGIN, 'sector_deg': SECTOR_DEG,
           'size_ratio_used': ratio, 'kitti_frames': acc_k.frames, 'nuscenes_frames': acc_n.frames,
           'edges_fg': EDGES_FG.tolist(), 'edges_bg': EDGES_BG.tolist(), 'min_boxes': MIN_BOXES, 'fg': {}}
    for c in CLASSES:
        kp = acc_k.fg_pts[c] / np.maximum(acc_k.fg_n[c], 1)
        npp = acc_n.fg_pts[c] / np.maximum(acc_n.fg_n[c], 1)
        ok = (acc_k.fg_n[c] >= MIN_BOXES) & (acc_n.fg_n[c] >= MIN_BOXES) & (kp > 0)
        rate = np.where(ok, np.minimum(npp / np.maximum(kp, 1e-9), 1.0), -1.0)
        res['fg'][c] = {'kitti_pts_per_box': kp.tolist(), 'kitti_boxes': acc_k.fg_n[c].tolist(),
                        'nuscenes_pts_per_box': npp.tolist(), 'nuscenes_boxes': acc_n.fg_n[c].tolist(),
                        'rate': rate.tolist()}
    hk, hn = acc_k.bg / acc_k.frames, acc_n.bg / acc_n.frames
    trusted = (hk > 0.01 * hk.mean()) & (hn > 0.01 * hn.mean())
    rate_bg = np.ones_like(hk)
    np.divide(hn, hk, out=rate_bg, where=trusted)
    res['bg'] = {'kitti_pts_per_frame': hk.tolist(), 'nuscenes_pts_per_frame': hn.tolist(),
                 'rate': np.minimum(rate_bg, 1.0).tolist()}
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=1)
    for c in CLASSES:
        print(c, 'rate_fg', np.round(res['fg'][c]['rate'], 3).tolist())
        print(c, 'boxes kitti', res['fg'][c]['kitti_boxes'], 'nuscenes', res['fg'][c]['nuscenes_boxes'])
    print('rate_bg', np.round(res['bg']['rate'], 3).tolist())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
