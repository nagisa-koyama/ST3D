"""A4 (experiments_md 20261017_09 §4): apply the derived cuts to the teacher's floor-0.0001 Waymo TRAIN pseudo-labels exactly
as `self_training_utils.save_pseudo_label_batch` does at generation, and check the result on the row's own target loader.

    python analysis/a4_cut_check.py <cfg> <ps_label_e0.pkl at floor 0.0001> <out_cut.pkl> --score S1 S2 S3 --neg N1 N2 N3 [--frames 200]

1. Cut (all frames): a box whose score is below its class's NEG_THRESH is dropped; one below its SCORE_THRESH keeps a
   NEGATIVE label (the ignore band). Prints per-frame positives and ignore-band boxes per class: the expected values of
   the run's first pass (`generate_ps_e0 ... pos_ps_box=(...), ign_ps_box=(...)`, a running mean over rank 0's frames).
   Writes the cut dict (same format as ps_label_e0.pkl) for analysis/target_dump.py.
2. Points per box (a frame-alignment sanity check, bug_prevention_checklist rule 4): the row's TARGET dataset (training
   mode, as the row trains, rule-G SHIFT_COOR included) serves --frames frames with the cut labels; points inside each
   positive and ignore-band box after prepare_data are counted (labels read from gt_boxes[:, 7], never gt_names).
   Pseudo-labels stored in a different frame from the cloud would show as mostly empty positive boxes.
Memory voting does not act on the first pass (nothing is remembered yet), so it is not replicated. No target label is read.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _init_path  # noqa: F401,E402  (must precede pcdet)
from easydict import EasyDict  # noqa: E402

from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


def cut_frame(info, score, neg):
    """Replica of save_pseudo_label_batch's NEG removal and SCORE ignore marking on one stored frame (labels >= 1 here)."""
    g = info['gt_boxes'].copy()
    if len(g) == 0:
        return dict(info, gt_boxes=g)
    lab = g[:, 7].astype(np.int64)
    assert (lab >= 1).all(), 'input must be a floor file (every label positive)'
    keep = g[:, 8] >= neg[lab - 1]
    g, lab = g[keep], lab[keep]
    ign = g[:, 8] < score[lab - 1]
    g[ign, 7] = -lab[ign]
    out = {'gt_boxes': g, 'memory_counter': np.zeros(len(g))}
    for k in ('cls_scores', 'iou_scores', 'teacher_classes'):
        if k in info and info[k] is not None:
            v = np.asarray(info[k])[keep]
            if k == 'teacher_classes':
                v = v.copy(); v[ign] = 'Ignored'
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfg'); ap.add_argument('ps'); ap.add_argument('out')
    ap.add_argument('--score', type=float, nargs=3, required=True)
    ap.add_argument('--neg', type=float, nargs=3, required=True)
    ap.add_argument('--frames', type=int, default=200)
    a = ap.parse_args()
    score, neg = np.array(a.score), np.array(a.neg)
    assert (score >= neg).all(), 'rule A step 6: a SCORE_THRESH below its NEG_THRESH must already be raised to it'
    ps = pickle.load(open(a.ps, 'rb'))
    cut = {k: cut_frame(v, score, neg) for k, v in ps.items()}
    pickle.dump(cut, open(a.out, 'wb'))
    n = len(cut)
    pos = np.zeros(3); ign = np.zeros(3)
    for v in cut.values():
        lab = v['gt_boxes'][:, 7].astype(np.int64)
        for c in range(3):
            pos[c] += (lab == c + 1).sum(); ign[c] += (lab == -(c + 1)).sum()
    names = ['Car', 'Pedestrian', 'Cyclist']
    print(f'cuts SCORE {score.tolist()} NEG {neg.tolist()}; {n} frames; wrote {a.out}')
    print('per frame, all frames: positives ' + ' / '.join(f'{names[c]} {pos[c] / n:.3f}' for c in range(3))
          + '; ignore band ' + ' / '.join(f'{names[c]} {ign[c] / n:.3f}' for c in range(3)))

    cfg = EasyDict(); cfg_from_yaml_file(a.cfg, cfg)
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=common_utils.create_logger(), training=True,
                                model_ontology=cfg.get('ONTOLOGY'))
    ds.set_pseudo_labels(cut)
    print(f'target loader: {cfg.DATA_CONFIG_TAR.DATASET}, {len(ds)} frames, SHIFT_COOR {cfg.DATA_CONFIG_TAR.SHIFT_COOR}')
    pts_pos = {c: [] for c in range(3)}; pts_ign = {c: [] for c in range(3)}
    step = max(1, len(ds) // a.frames)
    for i in list(range(0, len(ds), step))[:a.frames]:
        d = ds[i]
        g = d['gt_boxes']
        if len(g) == 0:
            continue
        inside = roiaware_pool3d_utils.points_in_boxes_cpu(d['points'][:, :3], g[:, :7]).sum(1)
        lab = g[:, 7].astype(np.int64)
        for c in range(3):
            pts_pos[c] += inside[lab == c + 1].tolist(); pts_ign[c] += inside[lab == -(c + 1)].tolist()
    for c in range(3):
        for tag, lst in (('positive', pts_pos[c]), ('ignore', pts_ign[c])):
            x = np.array(lst)
            if len(x):
                print(f'  {names[c]:10s} {tag:8s} boxes {len(x):6d}  points per box median {np.median(x):7.1f}  '
                      f'p10 {np.percentile(x, 10):6.1f}  empty {np.mean(x == 0):.3f}')
            else:
                print(f'  {names[c]:10s} {tag:8s} boxes      0')


if __name__ == '__main__':
    main()
