"""CPU check before the A3 rule-B pass (experiments_md 20261017_06 §3): does the eval-mode generation set of the flat
config serve teacher 27807 its TRAINING input?

generate_pseudo_labels.py builds the target set with training=True and calls .eval() before generating; RING_PATTERN
acts in training only unless APPLY_AT_EVAL is set. On N strided Waymo val frames of the flat config this prints:
  1. identity: with the same np.random state, the eval-mode cloud (APPLY_AT_EVAL) equals the training-mode cloud
     point for point (same function, same phases) -> the pass sees exactly what training saw, up to the random phase;
  2. points per frame in the model range (|x|, |y| <= 75.2 m, z in [-2, 4]) for the eval view, the training view and
     the raw five-lidar cloud (APPLY_AT_EVAL off); the eval view after prepare_data as the generator's loader yields it;
  3. eval mode leaves the GT boxes untouched (no recount, no pseudo-label slot), so the generator's loader works.
Exit 1 on any identity failure.

    python analysis/a3_ringpattern_eval_view_check.py [--cfg <flat cfg>] [--frames 60]
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

FLAT = 'analysis/pseudo_label_threshold/cfgs/teacher_27807_on_waymo_val_int5_ringpattern.yaml'


def in_range(p):
    return int(((np.abs(p[:, 0]) <= 75.2) & (np.abs(p[:, 1]) <= 75.2) & (p[:, 2] >= -2) & (p[:, 2] <= 4)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default=FLAT)
    ap.add_argument('--frames', type=int, default=60)
    args = ap.parse_args()
    cfg = cfg_from_yaml_file(args.cfg, EasyDict())
    logger = common_utils.create_logger()
    tcfg = cfg.DATA_CONFIG_TAR
    assert tcfg.RING_PATTERN.get('APPLY_AT_EVAL', False), 'the flat config must set RING_PATTERN.APPLY_AT_EVAL'
    ds, _, _ = build_dataloader(tcfg, cfg.CLASS_NAMES, 1, dist=False, workers=0, logger=logger, training=True,
                                model_ontology=cfg.get('ONTOLOGY', None))
    raw_cfg = copy.deepcopy(tcfg); raw_cfg.RING_PATTERN.APPLY_AT_EVAL = False
    ds_raw, _, _ = build_dataloader(raw_cfg, cfg.CLASS_NAMES, 1, dist=False, workers=0, logger=logger, training=True,
                                    model_ontology=cfg.get('ONTOLOGY', None))
    ds_raw.eval()
    print(f'flat config {args.cfg}: {len(ds)} Waymo val frames (split {tcfg.DATA_SPLIT}, interval {tcfg.SAMPLED_INTERVAL})')
    idx = np.linspace(0, len(ds) - 1, args.frames).astype(int)
    fails, ev, tr, rw, prep, nbox = [], [], [], [], [], []
    for k, i in enumerate(idx):
        info = ds.infos[i]
        ds.eval(); np.random.seed(1000 + k); pe = ds.get_lidar_with_sweeps(info)
        ds.train(); np.random.seed(1000 + k); pt = ds.get_lidar_with_sweeps(info)
        if pe.shape != pt.shape or not np.array_equal(pe, pt):
            fails.append(f'frame {i}: eval view {pe.shape} != training view {pt.shape}')
        pr = ds_raw.get_lidar_with_sweeps(info)
        ev.append(in_range(pe)); tr.append(in_range(pt)); rw.append(in_range(pr))
        ds.eval(); d = ds[i]
        prep.append(len(d['points']))
        nbox.append(len(d['gt_boxes']) if d.get('gt_boxes', None) is not None else -1)
    ds.eval()
    print(f'1. eval view == training view (same np.random state): {len(idx) - len(fails)} / {len(idx)} frames')
    print(f'2. points per frame in the model range, mean (median): eval view {np.mean(ev):.0f} ({np.median(ev):.0f}), '
          f'training view {np.mean(tr):.0f} ({np.median(tr):.0f}), raw five-lidar cloud {np.mean(rw):.0f} '
          f'({np.median(rw):.0f}); ratio eval / raw {np.mean(ev) / np.mean(rw):.3f}')
    print(f'   eval view after prepare_data (the generator loader): {np.mean(prep):.0f} ({np.median(prep):.0f}) points per frame')
    print(f'3. eval-mode GT boxes per frame (served, unused by the generator): mean {np.mean(nbox):.1f}, '
          f'frames without a gt_boxes array: {int((np.array(nbox) < 0).sum())}')
    print('checks:', 'ALL PASS' if not fails else f'{len(fails)} FAIL')
    for f in fails[:10]:
        print('  ', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
