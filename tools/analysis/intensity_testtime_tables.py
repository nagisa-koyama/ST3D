"""Tables for the test-time intensity calibration stage (`map_intensity_to_reference`).

For a cross-dataset config (DATA_CONFIG = the model's SOURCE sensor, DATA_CONFIG_TAR = the target), measure
the per-range-ring intensity distribution of each dataset's TRAIN split from point clouds alone - every point
pooled, no class split, so no label is used - and write two tables that map TARGET intensity onto the SOURCE's:

    <prefix>_rings.npz   one map per ring (0/10/20/30/40/50/75 m); a ring with < MIN_POINTS on either side
                         falls back to the all-range map
    <prefix>_global.npz  one map for all ranges (the "does range matter?" control)

Units are the in-pipeline ones (after each loader's own scaling: nuScenes INTENSITY_SCALE, Waymo tanh,
PandaSet /255, KITTI as stored), and integer-valued intensities are dequantised over one step before counting.
experiments_md/20261003_03 (intensity matrix), plan in the conversation of 2026-10-04.

    python analysis/intensity_testtime_tables.py --cfg <cross cfg with intensity> --out_prefix /path/name
"""
import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.datasets import intensity_calibration as ic  # noqa: E402

STEP = {'NuScenesDataset': 1 / 255.0, 'KittiDataset': 0.01, 'PandasetDataset': 1 / 255.0, 'WaymoDataset': 0.0,
        'LyftDataset': 1 / 255.0}
MIN_POINTS = 2000


def pooled_hist(dcfg, cfg, frames, log):
    ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=log, training=True, model_ontology=cfg.get('ONTOLOGY', None))
    idx = dcfg.POINT_FEATURE_ENCODING.src_feature_list.index('intensity')
    st = ic.compute_intensity_statistics(ds, num_frames=frames, scale=1.0, step=STEP[dcfg.DATASET], intensity_index=idx)
    return st['hist'].sum(axis=1), st['edges'], STEP[dcfg.DATASET]      # (rings, levels): every channel pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--out_prefix', required=True)
    ap.add_argument('--frames', type=int, default=300)
    a = ap.parse_args()
    log = logging.getLogger('tables'); log.addHandler(logging.StreamHandler()); log.setLevel(logging.WARNING)
    cfg = cfg_from_yaml_file(a.cfg, EasyDict())
    src, edges, _ = pooled_hist(cfg.DATA_CONFIG, cfg, a.frames, log)
    tgt, _, tgt_step = pooled_hist(cfg.DATA_CONFIG_TAR, cfg, a.frames, log)
    Q = ic.quantiles
    s_all, t_all = src.sum(axis=0), tgt.sum(axis=0)
    from_q, to_q = [], []
    print('%s -> %s (map TARGET onto SOURCE), %d frames each' % (cfg.DATA_CONFIG_TAR.DATASET, cfg.DATA_CONFIG.DATASET, a.frames))
    for r in range(len(edges) - 1):
        own = src[r].sum() >= MIN_POINTS and tgt[r].sum() >= MIN_POINTS
        from_q.append(Q(tgt[r] if own else t_all)); to_q.append(Q(src[r] if own else s_all))
        print('  ring %4.0f-%-4.0f m  target n %9d p50 %.3f | source n %9d p50 %.3f%s' % (
            edges[r], edges[r + 1], tgt[r].sum(), ic.quantiles(tgt[r], np.array([0.5]))[0] if tgt[r].sum() else np.nan,
            src[r].sum(), ic.quantiles(src[r], np.array([0.5]))[0] if src[r].sum() else np.nan, '' if own else '  (fallback: all ranges)'))
    os.makedirs(os.path.dirname(a.out_prefix), exist_ok=True)
    np.savez(a.out_prefix + '_rings.npz', edges=edges, from_q=np.array(from_q), to_q=np.array(to_q), from_step=tgt_step,
             hist_src=src, hist_tgt=tgt)
    np.savez(a.out_prefix + '_global.npz', edges=np.array([0.0, edges[-1]]), from_q=Q(t_all)[None], to_q=Q(s_all)[None],
             from_step=tgt_step)
    print('  all ranges: target p50 %.3f, source p50 %.3f; wrote %s_{rings,global}.npz' % (
        Q(t_all, np.array([0.5]))[0], Q(s_all, np.array([0.5]))[0], a.out_prefix))


if __name__ == '__main__':
    main()
