"""Gate for the label-free foreground-aware correction (20260929_05 item 1).

Can the target's foreground density - points per object per range ring, the quantity the
two-channel correction needs - be estimated WITHOUT labels? Ground removal (z > 0.3 m with the
family's SHIFT_COOR putting the ground at z ~ 0), Euclidean clustering (DBSCAN), a Car-like extent
filter, then points per kept cluster per ring. Compared per ring with points per GT Car box holding
>= 1 point (the calibration's own statistic) on the SOURCE - a legal place to measure the estimator's
bias, since source labels exist - and, for validation only, on the target.

    python cluster_foreground_estimate.py [--frames 100]
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

TOOLS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent)); sys.path.insert(0, str(TOOLS / 'analysis'))
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402
from pseudo_label_foreground_audit import disable_augmentation  # noqa: E402

RINGS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 75)]
GROUND_Z, EPS, MIN_SAMPLES = 0.3, 0.6, 8
CAR_L, CAR_W, CAR_H = (2.5, 6.5), (1.0, 3.0), (0.8, 2.5)


def clusters(points):
    p = points[(points[:, 2] > GROUND_Z) & (np.linalg.norm(points[:, :2], axis=1) < 75)]
    if len(p) < MIN_SAMPLES:
        return []
    lab = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(p[:, :3]).labels_
    out = []
    for k in np.unique(lab):
        if k < 0: continue
        q = p[lab == k]
        if len(q) < 10: continue
        xy = q[:, :2] - q[:, :2].mean(0)
        w, v = np.linalg.eigh(np.cov(xy.T)); ax = xy @ v
        L, W = np.ptp(ax[:, 1]), np.ptp(ax[:, 0]); H = np.ptp(q[:, 2])
        if CAR_L[0] <= L <= CAR_L[1] and CAR_W[0] <= W <= CAR_W[1] and CAR_H[0] <= H <= CAR_H[1]:
            out.append((np.linalg.norm(q[:, :2].mean(0)), len(q)))
    return out


def per_ring(ds, frames, class_idx=1):
    est = {r: [] for r in RINGS}; gt = {r: [] for r in RINGS}; ncl = 0; nbox = 0
    for i in np.linspace(0, len(ds) - 1, frames).astype(int):
        d = ds[int(i)]; pts = d['points'][:, :3]
        for r, n in clusters(pts):
            for lo, hi in RINGS:
                if lo <= r < hi: est[(lo, hi)].append(n)
        ncl += len(est)
        boxes = np.asarray(d.get('gt_boxes', np.zeros((0, 8)))).reshape(-1, 8)
        boxes = boxes[boxes[:, 7] == class_idx]
        if len(boxes):
            _, per_box, _ = DataProcessor.box_occupancy(pts, boxes[:, :7])
            rb = np.linalg.norm(boxes[:, :2], axis=1)
            for r, n in zip(rb, per_box):
                if n >= 1:
                    for lo, hi in RINGS:
                        if lo <= r < hi: gt[(lo, hi)].append(n)
    return est, gt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--frames', type=int, default=100)
    a = ap.parse_args()
    cfg = cfg_from_yaml_file('cfgs/da-ieee-access/centerpoint-foreground-v2-lyft2nuscenes.yaml', EasyDict())
    log = common_utils.create_logger()
    src_cfg = copy.deepcopy(cfg.DATA_CONFIGS.LYFT_40BEAM); src_cfg.HIST_DIST_ON_THE_FLY = False
    src, _, _ = build_dataloader(src_cfg, cfg.CLASS_NAMES, 1, False, workers=0, logger=log, training=True, model_ontology=cfg.ONTOLOGY)
    tar_cfg = copy.deepcopy(cfg.DATA_CONFIG_TAR); tar_cfg.USE_PSEUDO_LABEL = False   # real GT, VALIDATION ONLY
    tgt, _, _ = build_dataloader(tar_cfg, cfg.CLASS_NAMES, 1, False, workers=0, logger=log, training=True, model_ontology=cfg.ONTOLOGY)
    disable_augmentation(src); disable_augmentation(tgt)
    res = {}
    for name, ds in (('lyft40 (source, legal)', src), ('nuscenes (target, VALIDATION)', tgt)):
        est, gt = per_ring(ds, a.frames); res[name] = (est, gt)
        print('\n== %s: %d frames' % (name, a.frames))
        print('%-8s %10s %10s %10s %10s %8s' % ('ring', 'clusters/f', 'pts/clust', 'gtCar/f', 'pts/gtbox', 'ratio'))
        for r in RINGS:
            e, g = est[r], gt[r]
            print('%-8s %10.2f %10.1f %10.2f %10.1f %8.2f' % ('%d-%d' % r, len(e) / a.frames, np.mean(e) if e else np.nan,
                  len(g) / a.frames, np.mean(g) if g else np.nan, (np.mean(e) / np.mean(g)) if e and g else np.nan))
    es, gs = res['lyft40 (source, legal)']; et, gt_ = res['nuscenes (target, VALIDATION)']
    print('\n== foreground rate F_t/F_s per ring: from clusters (label-free) vs from GT (validation)')
    for r in RINGS:
        if es[r] and et[r] and gs[r] and gt_[r]:
            print('%-8s clusters %.3f   GT %.3f   -> the label-free estimate is %.2fx the GT one' % (
                '%d-%d' % r, np.mean(et[r]) / np.mean(es[r]), np.mean(gt_[r]) / np.mean(gs[r]),
                (np.mean(et[r]) / np.mean(es[r])) / (np.mean(gt_[r]) / np.mean(gs[r]))))


if __name__ == '__main__':
    main()
