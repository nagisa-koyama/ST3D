"""Intensity statistics for a self-training config, from PSEUDO-LABELS, checked against real target GT.

Measures what an intensity-calibration stage would be fitted on - per range ring and per channel
(background, Car, Pedestrian, Cyclist), source from its own labels, target from the teacher's
pseudo-labels - and scores it against the target's real boxes on DIFFERENT frames (diagnosis only;
the real labels never reach a fitted quantity). For each class and ring it prints the held-out W1
(normalised units) to the real target class distribution of:

    raw     the source class distribution, unit-rescaled only
    global  the source class points through ONE map per ring (target point cloud only, no labels)
    ps      the class map fitted on pseudo-labels  -> what the method would actually produce
    gt      the class map fitted on real labels on the other half of frames = the sampling floor

A quantile map transfers the target table exactly, so `ps` is the distance between the pseudo-label
class distribution and the real one; `ps` near `gt` means pseudo-label contamination does not matter.

    python analysis/intensity_statistics.py --cfg_file cfgs/da-ieee-access/centerpoint-accum-foreground-v2-nuscenes2kitti.yaml \\
        --ps_label /storage/wandb/run-<id>/files/ps_label/ps_label_e0.pkl --target_fov 40 --save out.npz
"""
import argparse
import copy
import logging
import pickle

import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _init_path  # noqa: F401,E402
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets import intensity_calibration as ic

SCALE_STEP = {'NuScenesDataset': (255.0, 1.0), 'KittiDataset': (1.0, 0.01), 'PandasetDataset': (1.0, 1.0 / 255),
              'LyftDataset': (255.0, 1.0)}


def build(dcfg, cfg, log):
    ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=log, training=True, model_ontology=cfg.get('ONTOLOGY', None))
    return ds


def stats(ds, frames, indices=None, fov=None, label=''):
    scale, step = SCALE_STEP[type(ds).__name__]
    idx = ds.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list.index('intensity')
    return ic.compute_intensity_statistics(ds, num_frames=frames, scale=scale, step=step, intensity_index=idx,
                                           indices=indices, fov_degree=fov, label=label)['hist']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--ps_label', required=True)
    ap.add_argument('--frames', type=int, default=300, help='per source platform, and per target half')
    ap.add_argument('--target_fov', type=float, default=None, help='full-angle cone for the TARGET (KITTI: 80)')
    ap.add_argument('--save', default=None)
    ap.add_argument('--reuse', default=None, help='npz from an earlier run: reuse its source and real-label tables')
    ap.add_argument('--ps_min_score', default='', help='e.g. "Pedestrian:0.5,Cyclist:0.5": pseudo-labels of '
                    'those classes scoring below are treated as IGNORED for the statistics only. The statistics '
                    'need pure boxes, not many, so a stricter cut than training\'s count balance is label-free and fair.')
    a = ap.parse_args()
    log = logging.getLogger('intensity'); log.addHandler(logging.StreamHandler()); log.setLevel(logging.WARNING)
    cfg = cfg_from_yaml_file(a.cfg_file, EasyDict())

    ps_labels = pickle.load(open(a.ps_label, 'rb'))
    for item in filter(None, a.ps_min_score.split(',')):
        name, thr = item.split(':'); cls = cfg.CLASS_NAMES.index(name) + 1
        for v in ps_labels.values():
            b = v['gt_boxes']
            low = (np.abs(b[:, 7]) == cls) & (b[:, 8] < float(thr))
            b[low, 7] = -cls

    tgt_ps = build(cfg.DATA_CONFIG_TAR, cfg, log)
    tgt_ps.set_pseudo_labels(ps_labels)
    half_a, half_b = list(range(0, len(tgt_ps), 2)), list(range(1, len(tgt_ps), 2))
    ps_a = stats(tgt_ps, a.frames, half_a, a.target_fov, 'target ps A')
    if a.reuse:
        old = np.load(a.reuse); src, gt_a, gt_b = old['src'], old['gt_a'], old['gt_b']
    else:
        sources = cfg.DATA_CONFIGS if cfg.get('DATA_CONFIGS', None) else {'SOURCE': cfg.DATA_CONFIG}
        src = sum(stats(build(d, cfg, log), a.frames, label=k) for k, d in sources.items())
        gt_cfg = copy.deepcopy(cfg.DATA_CONFIG_TAR); gt_cfg.USE_PSEUDO_LABEL = False
        tgt_gt = build(gt_cfg, cfg, log)
        gt_a = stats(tgt_gt, a.frames, half_a, a.target_fov, 'target gt A')
        gt_b = stats(tgt_gt, a.frames, half_b, a.target_fov, 'target gt B')
    if a.save:
        np.savez(a.save, src=src, ps_a=ps_a, gt_a=gt_a, gt_b=gt_b, edges=np.asarray(ic.DEFAULT_RING_EDGES),
                 class_names=np.asarray(cfg.CLASS_NAMES))

    names = ['background'] + list(cfg.CLASS_NAMES)
    edges = ic.DEFAULT_RING_EDGES
    Q = ic.quantiles
    print('channel     ring    | points src / ps / gt       | W1 to real target (held out): raw    global  ps     gt(floor)')
    for c in [1, 2, 3, 0]:
        for r in range(len(edges) - 1):
            n = (src[r, c].sum(), ps_a[r, c].sum(), gt_a[r, c].sum(), gt_b[r, c].sum())
            head = '%-11s %3g-%-3g | %8d / %7d / %7d' % (names[c], edges[r], edges[r + 1], n[0], n[1], n[2])
            if min(n) < 500:
                print(head + ' |  -'); continue
            truth = Q(gt_b[r, c])
            src_c = Q(src[r, c])
            # global map: source ring pool -> target ring pool (the cloud alone, here from the GT view)
            glob = np.interp(np.interp(src_c, Q(src[r].sum(0)), ic.QUANTILE_LEVELS), ic.QUANTILE_LEVELS, Q(gt_a[r].sum(0)))
            print(head + ' |     %.3f  %.3f   %.3f  %.3f' % (ic.w1(src_c, truth), ic.w1(glob, truth),
                                                             ic.w1(Q(ps_a[r, c]), truth), ic.w1(Q(gt_a[r, c]), truth)))


if __name__ == '__main__':
    main()
