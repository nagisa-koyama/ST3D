"""The nuScenes-side values of the FACTOR-SWAP S2 rows (experiments_md 20261003_05 §5 item 24), from the TRAIN splits.

ANALYSIS (KITTI labels are read; the rows they feed are labelled-analysis oracles, not methods). Writes one JSON with:
  - RANGE_KEEP (R3): per-frame radial point histograms of KITTI TRAIN and nuScenes TRAIN measured through each
    dataset's own pipeline (point_calibration.compute_range_histogram: range-masked, loader frame, 1,000 strided
    frames, 50 bins to 75 m), and the keep rate per bin = nuScenes / KITTI, capped at 1, with bins under 1% of the
    mean on either side left at 1 (the per_bin_sample_rate guard);
  - GT_DIM_SCALE (R6): median isotropic box scale (dx dy dz)^(1/3) per class, KITTI TRAIN vs nuScenes TRAIN (boxes the
    loaders train on: >= 1 point, inside POINT_CLOUD_RANGE in x / y), and the ratio nuScenes / KITTI;
  - GT_IGNORE_BY_RANGE (R7): the Car box range histogram on both sides in 10 m bins to 70 m plus one beyond, and the
    keep probability per bin = (p_nus / p_kitti) / max over trusted bins (KITTI >= 50 boxes), capped at 1;
  - PITCH (R5): the share of Boston (n008) frames in nuScenes TRAIN, for the per-frame Boston / Singapore mixture.
The S2 sources are the configs of the S2 control (nuScenes, single sweep) and the KITTI oracle the rows are built on.
Read-only, CPU; run as a Slurm CPU job:

    python analysis/factorswap_s2_stats.py <out.json>
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
from pcdet.datasets.point_calibration import calibration_target_config, compute_range_histogram  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

NUS_CFG = 'cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti.yaml'
KITTI_CFG = 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml'
BOX_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 1000]
MIN_KITTI_BOXES = 50


def build(cfg_path):
    cfg = cfg_from_yaml_file(cfg_path, EasyDict())
    dcfg, split = calibration_target_config(cfg.DATA_CONFIG, 'train')
    assert split == 'train'
    ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    return cfg, dcfg, ds


def in_range_xy(boxes, pcr):
    return (np.abs(boxes[:, 0]) <= pcr[3]) & (np.abs(boxes[:, 1]) <= pcr[4]) if len(boxes) else np.zeros(0, bool)


def kitti_boxes(ds, pcr):
    """(name, dx dy dz, xy) of the KITTI TRAIN boxes the oracle trains on (non-DontCare, >= 1 point, in range)."""
    out = []
    for info in ds.kitti_infos:
        a = info['annos']
        b = a['gt_boxes_lidar']
        n = a['name'][:len(b)]
        pts = a['num_points_in_gt'][:len(b)]
        ok = (pts >= 1) & in_range_xy(b, pcr)
        out += [(nm, bb) for nm, bb in zip(n[ok], b[ok])]
    return out


def nus_boxes(ds, dcfg, pcr):
    to_model = ds.map_ontology_dataset_to_model or {}
    min_pts = dcfg.get('FILTER_MIN_POINTS_IN_GT', 0) or 0
    out = []
    for info in ds.infos:
        b, n, pts = info['gt_boxes'], info['gt_names'], info['num_lidar_pts']
        ok = (pts >= max(min_pts, 1)) & in_range_xy(b, pcr)
        out += [(to_model.get(nm, nm), bb) for nm, bb in zip(n[ok], b[ok])]
    return out


def iso_median(boxes, cls):
    v = np.array([bb[3:6] for nm, bb in boxes if nm == cls], dtype=np.float64)
    v = v[np.all(v > 0, axis=1)]
    return float(np.median(np.cbrt(v.prod(axis=1)))), [float(x) for x in np.median(v, axis=0)], int(len(v))


def range_hist(boxes, cls):
    r = np.array([np.hypot(bb[0], bb[1]) for nm, bb in boxes if nm == cls])
    return np.histogram(r, bins=BOX_EDGES)[0]


if __name__ == '__main__':
    out_path = sys.argv[1]
    logger = common_utils.create_logger()
    ncfg, ndcfg, nds = build(NUS_CFG)
    kcfg, kdcfg, kds = build(KITTI_CFG)
    pcr = np.asarray(kcfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float64)
    assert np.allclose(pcr, ncfg.DATA_CONFIG.POINT_CLOUD_RANGE), 'the two configs crop differently'
    res = {'nuscenes_cfg': NUS_CFG, 'kitti_cfg': KITTI_CFG, 'split': 'train',
           'nuscenes_train_frames': len(nds), 'kitti_train_frames': len(kds)}

    # R5: Boston share of nuScenes TRAIN frames (the lidar file name starts with the vehicle, n008 = Boston).
    veh = [Path(i['lidar_path']).name[:4] for i in nds.infos]
    res['pitch'] = {'n008_boston_frames': veh.count('n008'), 'n015_singapore_frames': veh.count('n015'),
                    'boston_share': veh.count('n008') / max(len(veh), 1)}

    # R6 / R7: boxes.
    kb, nb = kitti_boxes(kds, pcr), nus_boxes(nds, ndcfg, pcr)
    res['size'] = {}
    for c in ('Car', 'Pedestrian', 'Cyclist'):
        km, kdim, kn = iso_median(kb, c)
        nm, ndim, nn = iso_median(nb, c)
        res['size'][c] = {'kitti_iso_median': km, 'kitti_dim_median': kdim, 'kitti_n': kn,
                          'nuscenes_iso_median': nm, 'nuscenes_dim_median': ndim, 'nuscenes_n': nn,
                          'ratio': nm / km}
    kh, nh = range_hist(kb, 'Car'), range_hist(nb, 'Car')
    pk, pn = kh / kh.sum(), nh / nh.sum()
    trusted = kh >= MIN_KITTI_BOXES
    ratio = np.where(kh > 0, pn / np.maximum(pk, 1e-12), 1.0)
    keep = np.minimum(1.0, ratio / ratio[trusted].max())
    res['label_range'] = {'edges': BOX_EDGES, 'kitti_car_hist': kh.tolist(), 'nuscenes_car_hist': nh.tolist(),
                          'kitti_share_within_30m': float(pk[:3].sum()), 'nuscenes_share_within_30m': float(pn[:3].sum()),
                          'keep': keep.tolist(), 'kept_share_of_kitti_cars': float((pk * keep).sum()),
                          'min_kitti_boxes_trusted': MIN_KITTI_BOXES}

    # R3: radial point histograms through each pipeline.
    hk = compute_range_histogram(kds, logger=logger)
    hn = compute_range_histogram(nds, logger=logger)
    trusted = (hk > 0.01 * hk.mean()) & (hn > 0.01 * hn.mean())
    rate = np.ones_like(hk)
    np.divide(hn, hk, out=rate, where=trusted)
    rate = np.minimum(rate, 1.0)
    res['range_keep'] = {'edges': np.linspace(0, 75.0, 51).tolist(), 'kitti_pts_per_frame': hk.tolist(),
                         'nuscenes_pts_per_frame': hn.tolist(), 'rate': rate.tolist(),
                         'kitti_total': float(hk.sum()), 'nuscenes_total': float(hn.sum()),
                         'expected_kept_share': float((hk * rate).sum() / hk.sum())}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'range_keep'}, indent=1))
    print('range_keep: kitti %.0f pts/frame, nuscenes %.0f, expected kept %.3f'
          % (hk.sum(), hn.sum(), res['range_keep']['expected_kept_share']))
