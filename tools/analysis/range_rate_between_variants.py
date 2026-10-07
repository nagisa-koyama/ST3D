"""Per-range-bin keep rate that thins eval variant B's points to variant A's radial count (test A', 20261007_03 §11).

Builds both eval datasets (DATA_CONFIG_TAR of two configs, eval mode, the loader's own pipeline up to the point cloud),
histograms planar radius over N strided frames, writes npz(edges, rate = min(1, hist_A / hist_B)) for
`sample_points_by_range_rate`. Source-domain clouds only; no labels read.

    python analysis/range_rate_between_variants.py <cfg_A> <cfg_B> <frames> <out.npz>
"""
import sys
import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

EDGES = np.arange(0.0, 75.2 + 1e-6, 1.5)
cfg_a, cfg_b, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
logger = common_utils.create_logger()


def hist(cfg_file):
    cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
    dc = cfg.DATA_CONFIG_TAR
    dc.DATA_PROCESSOR = [p for p in dc.DATA_PROCESSOR if p.NAME != 'sample_points_by_range_rate']
    ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    h = np.zeros(len(EDGES) - 1); used = 0
    for i in range(0, len(ds), max(1, len(ds) // n))[:n]:
        p = ds[i]['points']; r = np.hypot(p[:, 0], p[:, 1]); h += np.histogram(r, bins=EDGES)[0]; used += 1
    return h / used


ha, hb = hist(cfg_a), hist(cfg_b)
rate = np.where(hb > 0, np.minimum(1.0, ha / np.maximum(hb, 1e-9)), 1.0)
np.savez(out, edges=EDGES, rate=rate, hist_a=ha, hist_b=hb)
print('points/frame A %.0f, B %.0f, B after thinning %.0f' % (ha.sum(), hb.sum(), (hb * rate).sum()))
print('rate by 7.5 m: ' + ' '.join('%.2f' % rate[k:k + 5].mean() for k in range(0, len(rate), 5)))
