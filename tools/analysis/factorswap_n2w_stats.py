"""The nuScenes-side values of the FACTOR-SWAP N->W rows (experiments_md 20261003_05 §5 item 24, the nuScenes -> Waymo
port of factorswap_s2_stats.py), from the TRAIN splits. ANALYSIS (Waymo labels are read; the rows they feed are
labelled-analysis oracles, not methods). Writes one JSON with:
  - size (R6): median isotropic box scale (dx dy dz)^(1/3) per MODEL class (each dataset's names through its ontology
    map to the model's CLASS_NAMES; Waymo 'Vehicle' -> Car includes trucks and buses), on the boxes each loader trains
    on (Waymo: stored num_points_in_gt >= 1, the oracle's zero-point filter; nuScenes: num_lidar_pts >= 1, as
    factorswap_s2_stats; both inside POINT_CLOUD_RANGE in x / y), and the ratio nuScenes / Waymo;
  - label_range (R7): the Car box range histogram on both sides in 10 m bins to 70 m plus one beyond, and the keep
    probability per bin = (p_nus / p_waymo) / max over trusted bins (Waymo >= 50 boxes), capped at 1;
  - pitch (R5): the Boston (n008) share of nuScenes TRAIN frames; the slopes are the measured forward ground slopes of
    20261004_02 (constants below), not re-measured here;
  - pivot (R5): the Waymo TOP lidar position in the vehicle frame, from each segment's TOP extrinsic
    (/home/koyama/data/waymo_top_calib/top_calib.pkl, written by tools/analysis/waymo_top_calib_cache.py), with its
    spread over segments; and a check that the vehicle frame's x is FORWARD (ego displacement between consecutive
    TRAIN frames, from the info poses, expressed in the earlier frame's vehicle frame).
Every frame of both TRAIN splits is read (infos only: boxes and poses, no point cloud). Read-only, CPU; run as a Slurm
CPU job:

    python analysis/factorswap_n2w_stats.py <out.json>
"""
import json
import os
import pickle
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
from pcdet.utils import common_utils  # noqa: E402

NUS_CFG = 'cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2waymo.yaml'
WAYMO_CFG = 'cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml'
TOP_CALIB = '/home/koyama/data/waymo_top_calib/top_calib.pkl'
BOX_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 1000]
MIN_WAYMO_BOXES = 50
# Forward ground slopes in each loader's frame (memory dataset_sensor_and_platform_facts.md; experiments_md 20261004_02,
# tools/analysis/ground_tilt.py): nuScenes n008 Boston +2.29, n015 Singapore +1.35, Waymo +0.11 degrees.
SLOPE_DEG = {'nuscenes_boston': 2.29, 'nuscenes_singapore': 1.35, 'waymo': 0.11}


def build(cfg_path):
    """The TRAINING dataset of a config (train split, the training SAMPLED_INTERVAL): its infos are what it trains on."""
    cfg = cfg_from_yaml_file(cfg_path, EasyDict())
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
    return cfg, ds


def in_range_xy(boxes, pcr):
    return (np.abs(boxes[:, 0]) <= pcr[3]) & (np.abs(boxes[:, 1]) <= pcr[4]) if len(boxes) else np.zeros(0, bool)


def waymo_boxes(ds, pcr):
    """(model name, box) of the Waymo TRAIN boxes the oracle trains on ('unknown' dropped, stored count >= 1, in range)."""
    to_model = ds.map_ontology_dataset_to_model or {}
    min_pts = ds.dataset_cfg.get('MIN_POINTS_OF_GT', 1)
    out = []
    for info in ds.infos:
        a = info['annos']
        keep = a['name'] != 'unknown'
        b, n, pts = a['gt_boxes_lidar'][keep], a['name'][keep], a['num_points_in_gt'][keep]
        ok = (pts >= min_pts) & in_range_xy(b, pcr)
        out += [(to_model.get(nm, nm), bb) for nm, bb in zip(n[ok], b[ok])]
    return out


def nus_boxes(ds, pcr):
    to_model = ds.map_ontology_dataset_to_model or {}
    min_pts = ds.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', 0) or 0
    out = []
    for info in ds.infos:
        b, n, pts = info['gt_boxes'], info['gt_names'], info['num_lidar_pts']
        ok = (pts >= max(min_pts, 1)) & in_range_xy(b, pcr)
        out += [(to_model.get(nm, nm), bb) for nm, bb in zip(n[ok], b[ok])]
    return out


def iso_median(boxes, cls):
    v = np.array([bb[3:6] for nm, bb in boxes if nm == cls], dtype=np.float64).reshape(-1, 3)
    v = v[np.all(v > 0, axis=1)]
    return float(np.median(np.cbrt(v.prod(axis=1)))), [float(x) for x in np.median(v, axis=0)], int(len(v))


def range_hist(boxes, cls):
    r = np.array([np.hypot(bb[0], bb[1]) for nm, bb in boxes if nm == cls])
    return np.histogram(r, bins=BOX_EDGES)[0]


def top_pivot(segments):
    with open(TOP_CALIB, 'rb') as f:
        calib = pickle.load(f)
    missing = [s for s in segments if s not in calib]
    t = np.array([np.asarray(calib[s]['extrinsic'])[:3, 3] for s in segments if s in calib], dtype=np.float64)
    rot = np.array([np.asarray(calib[s]['extrinsic'])[:3, :3] for s in segments if s in calib], dtype=np.float64)
    # mounting pitch of the TOP frame: elevation of its x axis in the vehicle frame (not used by the rows; recorded)
    pitch = np.degrees(np.arcsin(rot[:, 2, 0]))
    return {'pivot_median': [float(x) for x in np.median(t, axis=0)],
            'pivot_min': [float(x) for x in t.min(axis=0)], 'pivot_max': [float(x) for x in t.max(axis=0)],
            'n_segments': int(len(t)), 'n_unique_translations': int(len(np.unique(np.round(t, 6), axis=0))),
            'segments_missing_from_calib': len(missing),
            'top_x_axis_elevation_deg_median': float(np.median(pitch))}


def x_is_forward(infos):
    """Ego displacement between consecutive frames of a sequence, in the earlier frame's vehicle frame."""
    d = []
    for a, b in zip(infos[:-1], infos[1:]):
        if a['point_cloud']['lidar_sequence'] != b['point_cloud']['lidar_sequence']:
            continue
        T = np.linalg.inv(np.asarray(a['pose'])) @ np.asarray(b['pose'])
        d.append(T[:3, 3])
    d = np.asarray(d)
    moving = np.hypot(d[:, 0], d[:, 1]) > 0.5
    dm = d[moving]
    return {'pairs': int(len(d)), 'moving_pairs': int(moving.sum()),
            'share_moving_with_dx_positive': float(np.mean(dm[:, 0] > 0)),
            'share_moving_lateral_below_0p2_of_forward': float(np.mean(np.abs(dm[:, 1]) < 0.2 * np.abs(dm[:, 0]))),
            'median_abs_dz_m': float(np.median(np.abs(dm[:, 2])))}


if __name__ == '__main__':
    out_path = sys.argv[1]
    logger = common_utils.create_logger()
    wcfg, wds = build(WAYMO_CFG)
    ncfg, nds = build(NUS_CFG)
    pcr = np.asarray(wcfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float64)
    assert np.allclose(pcr, ncfg.DATA_CONFIG.POINT_CLOUD_RANGE), 'the two configs crop differently'
    assert list(wcfg.CLASS_NAMES) == list(ncfg.CLASS_NAMES)
    res = {'nuscenes_cfg': NUS_CFG, 'waymo_cfg': WAYMO_CFG, 'split': 'train',
           'nuscenes_train_frames': len(nds.infos), 'waymo_train_frames': len(wds.infos),
           'waymo_sampled_interval_train': int(wds.dataset_cfg.SAMPLED_INTERVAL['train'])}

    # R5: Boston share of nuScenes TRAIN frames (the lidar file name starts with the vehicle, n008 = Boston), the slopes.
    veh = [Path(i['lidar_path']).name[:4] for i in nds.infos]
    share = veh.count('n008') / max(len(veh), 1)
    res['pitch'] = {'n008_boston_frames': veh.count('n008'), 'n015_singapore_frames': veh.count('n015'),
                    'boston_share': share, 'slopes_deg': SLOPE_DEG,
                    'pitch_deg': [SLOPE_DEG['nuscenes_boston'] - SLOPE_DEG['waymo'],
                                  SLOPE_DEG['nuscenes_singapore'] - SLOPE_DEG['waymo']],
                    'pitch_prob': [share, 1.0 - share]}
    segments = sorted({i['point_cloud']['lidar_sequence'] for i in wds.infos})
    res['pivot'] = top_pivot(segments)
    res['x_forward'] = x_is_forward(wds.infos)

    # R6 / R7: boxes.
    wb, nb = waymo_boxes(wds, pcr), nus_boxes(nds, pcr)
    res['size'] = {}
    for c in wcfg.CLASS_NAMES:
        wm, wdim, wn = iso_median(wb, c)
        nm, ndim, nn = iso_median(nb, c)
        res['size'][c] = {'waymo_iso_median': wm, 'waymo_dim_median': wdim, 'waymo_n': wn,
                          'nuscenes_iso_median': nm, 'nuscenes_dim_median': ndim, 'nuscenes_n': nn,
                          'ratio': nm / wm}
    wh, nh = range_hist(wb, 'Car'), range_hist(nb, 'Car')
    pw, pn = wh / wh.sum(), nh / nh.sum()
    trusted = wh >= MIN_WAYMO_BOXES
    ratio = np.where(wh > 0, pn / np.maximum(pw, 1e-12), 1.0)
    keep = np.minimum(1.0, ratio / ratio[trusted].max())
    res['label_range'] = {'edges': BOX_EDGES, 'waymo_car_hist': wh.tolist(), 'nuscenes_car_hist': nh.tolist(),
                          'waymo_share_within_30m': float(pw[:3].sum()), 'nuscenes_share_within_30m': float(pn[:3].sum()),
                          'keep': keep.tolist(), 'kept_share_of_waymo_cars': float((pw * keep).sum()),
                          'min_waymo_boxes_trusted': MIN_WAYMO_BOXES, 'trusted': trusted.tolist()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
