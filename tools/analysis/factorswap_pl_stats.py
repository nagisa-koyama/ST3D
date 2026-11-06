"""Swap values of the FACTOR-SWAP rows on PandaSet Pandar64 (spin), Lyft 64-beam and Lyft 40-beam (experiments_md
20261003_05 §5 item 34), from the TRAIN splits only. ANALYSIS (target TRAIN labels are read; the rows they feed are
labelled-analysis oracles, not methods). The nuScenes-side values are item 24's (/home/koyama/data/factorswap_s2/
stats.json, Slurm CPU job 31406): Car / Pedestrian median isotropic scale, the trained-box Car range histogram, the Boston
share of nuScenes TRAIN frames.

Every TRAIN frame of the target goes through the target oracle's own TRAINING loader up to (not into) prepare_data, so
the cloud and boxes are exactly what prepare_data's zero-point filter sees (SHIFT_COOR applied, no augmentation yet;
Lyft-40 restricted to its KEEP_FRAMES list by the loader). Per frame:
  - trained boxes = the boxes prepare_data keeps: points inside >= MIN_POINTS_OF_GT by prepare_data's own count, MODEL
    class (through the loader's ontology map), centre inside POINT_CLOUD_RANGE in x / y (items 24 / 26's definition);
  - the ground: 20261004_02's estimator (tools/analysis/ground_tilt.py, copied verbatim in osens2_utils): second-lowest
    return per 1 m BEV cell at 4-40 m horizontal range, RANSAC plane z = a x + b y + c, least-squares refit on the
    inliers; slope along the VEHICLE's forward axis in degrees (PandaSet loader +x; Lyft's sensor x points backward, so
    -x, approximately: its yaw is 176-179 deg), positive = ground rises ahead; c in the RAW frame (loader z - SHIFT_COOR).
Outputs (JSON):
  - size (R6): median isotropic scale (dx dy dz)^(1/3) per model class and the ratio nuScenes / target;
  - label_range (R7): the Car histogram in 10 m bins against nuScenes', KEEP = (p_nus / p_tgt) / max over bins with
    >= 50 target boxes, capped at 1 (item 24's rule), and the kept share;
  - ground: per-frame forward / left slope and raw-frame ground height, median and IQR; for PandaSet the R5 angles
    (nuScenes Boston +2.29 / Singapore +1.35 minus PandaSet's forward slope, mixed at the Boston share) and the ground
    height in the loader frame (the pitch pivot [0, 0, 1.88] assumes +0.02); for Lyft rule G's target SHIFT_COOR
    (g_nus + 1.75 - g_tgt, g_nus = -1.815, 20261007_01), the label-free height alignment;
  - x_forward (PandaSet): ego displacement between consecutive TRAIN frames from the poses, in the loader frame.
The density row on Lyft-64 (D) gets its two histograms here as files (--target lyft64_hist): the Lyft-64 training
source (the oracle's training-mode loader, augmentation on, as train.py measures a source) and nuScenes TRAIN through
the eval-mode view of the C1-analog 30313's target block (point_calibration.calibration_target_config, as train.py
measures a target), 1000 strided frames, 50 bins to 75 m: point_calibration.compute_range_histogram, as 30313.
Read-only, CPU; run as a Slurm CPU job from ST3D/tools inside the container:

    python analysis/factorswap_pl_stats.py --target pandaset|lyft64|lyft40|lyft64_hist [--workers 8] [--max-frames N]
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
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
from pcdet.datasets.loader_factor_swap import oracle_counts  # noqa: E402
from pcdet.datasets.osens2_utils import fit_plane, lowest_per_cell  # noqa: E402
from pcdet.datasets.waymo.waymo_factor_swap import model_names  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

TARGETS = {
    # target: (oracle config, training block, forward sign along loader x, output json)
    'pandaset': ('cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml', 'DATA_CONFIG', 1.0,
                 '/home/koyama/data/factorswap_pandaset/stats.json'),
    'lyft64': ('cfgs/da-ieee-access-analysis/factorswap-lyft64/centerpoint-factorswap-lyft64-oracle.yaml', 'DATA_CONFIG',
               -1.0, '/home/koyama/data/factorswap_lyft/stats_lyft64.json'),
    'lyft40': ('cfgs/da-ieee-access-analysis/factorswap-lyft40/centerpoint-factorswap-lyft40-oracle.yaml', 'LYFT_40BEAM',
               -1.0, '/home/koyama/data/factorswap_lyft/stats_lyft40.json'),
}
NUS_STATS = '/home/koyama/data/factorswap_s2/stats.json'
C1_LYFT40 = 'cfgs/da-ieee-access/centerpoint-global-recount-lyft40-lyft2nuscenes.yaml'   # its target block = nuScenes
HIST_DIR = '/home/koyama/data/factorswap_lyft'
BOX_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 1000]
MIN_TGT_BOXES = 50
# Forward ground slopes of nuScenes in its loader frame (20261004_02, ground_tilt.py on val frames): n008 Boston +2.29,
# n015 Singapore +1.35 degrees; nuScenes ground under the vehicle (raw frame, TRAIN, the same estimator, 20261007_01):
# -1.817 (n008) / -1.813 (n015), -1.815 pooled; nuScenes SHIFT_COOR 1.75.
NUS_SLOPE = {'boston': 2.29, 'singapore': 1.35}
NUS_GROUND, NUS_SHIFT = -1.815, 1.75

logger = common_utils.create_logger()
DS = None          # the dataset, inherited by forked workers


def load_cfg(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return cfg_from_yaml_file(path, EasyDict())


def block_of(cfg, name):
    return cfg.DATA_CONFIGS[name] if name not in cfg else cfg[name]


def build(cfg, block, training=True):
    ds, _, _ = build_dataloader(dataset_cfg=block, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=logger, training=training, model_ontology=cfg.get('ONTOLOGY'))
    return ds


def _init_worker():
    import torch
    torch.set_num_threads(1)


def frame_stats(i):
    """(per trained box: class, dims, radius), ground fit, frame count stats - for frame i of DS (pre-prepare_data)."""
    d = DS[i]
    pts, boxes = d['points'], d['gt_boxes']
    names = model_names(d['gt_names'], DS.map_ontology_dataset_to_model)
    n = oracle_counts(pts, boxes) if len(boxes) else np.zeros(0, np.int64)
    pcr = DS.point_cloud_range
    inr = (np.abs(boxes[:, 0]) <= pcr[3]) & (np.abs(boxes[:, 1]) <= pcr[4]) if len(boxes) else np.zeros(0, bool)
    trained = (n >= DS.dataset_cfg.get('MIN_POINTS_OF_GT', 1)) & inr & np.isin(names, DS.class_names)
    rows = [(str(nm), [float(x) for x in b[3:6]], float(np.hypot(b[0], b[1]))) for nm, b in zip(names[trained],
                                                                                              boxes[trained])]
    shift_z = float((DS.dataset_cfg.get('SHIFT_COOR', None) or [0, 0, 0])[2])
    raw = np.asarray(pts[:, :3], dtype=np.float64) - np.array([0.0, 0.0, shift_z])
    fit = fit_plane(lowest_per_cell(raw))
    ground = None if fit is None else [float(v) for v in fit[0]]
    car = names == 'Car'
    return {'rows': rows, 'ground': ground, 'n_boxes': int(len(boxes)), 'n_car': int(car.sum()),
            'n_car_empty': int((car & (n < 1)).sum()), 'n_car_out_of_range': int((car & ~inr).sum()),
            'n_points': int(len(pts))}


def iso_median(rows, cls):
    v = np.array([r[1] for r in rows if r[0] == cls], dtype=np.float64).reshape(-1, 3)
    v = v[np.all(v > 0, axis=1)]
    if len(v) == 0:
        return None, None, 0
    return float(np.median(np.cbrt(v.prod(axis=1)))), [float(x) for x in np.median(v, axis=0)], int(len(v))


def q(v):
    v = np.asarray(v, dtype=np.float64)
    return {'median': float(np.median(v)), 'p25': float(np.percentile(v, 25)), 'p75': float(np.percentile(v, 75)),
            'n': int(len(v))}


def pandaset_x_forward(ds):
    """Ego displacement between consecutive TRAIN frames (same sequence), in the earlier frame's LOADER frame."""
    import pandaset as ps
    by_seq = {}
    for info in ds.pandaset_infos:
        by_seq.setdefault(info['sequence'], []).append(info)
    d = []
    for seq, infos in by_seq.items():
        infos = sorted(infos, key=lambda x: x['frame_idx'])
        poses = [ds._get_pose(i) for i in infos]
        for a, b, ia, ib in zip(poses[:-1], poses[1:], infos[:-1], infos[1:]):
            if ib['frame_idx'] != ia['frame_idx'] + 1:
                continue
            p = np.array([[b['position']['x'], b['position']['y'], b['position']['z']]])
            e = ps.geometry.lidar_points_to_ego(p, a)[0]
            d.append([e[1], -e[0], e[2]])            # PandaSet ego (x right, y forward) -> loader (x forward, y left)
    d = np.asarray(d)
    moving = np.hypot(d[:, 0], d[:, 1]) > 0.5
    dm = d[moving]
    return {'pairs': int(len(d)), 'moving_pairs': int(moving.sum()),
            'share_moving_with_dx_positive': float(np.mean(dm[:, 0] > 0)),
            'share_moving_lateral_below_0p2_of_forward': float(np.mean(np.abs(dm[:, 1]) < 0.2 * np.abs(dm[:, 0]))),
            'median_abs_dz_m': float(np.median(np.abs(dm[:, 2])))}


def code_hashes():
    """sha256 of the files this measurement depends on, so the result can be tied to a commit afterwards."""
    import hashlib
    files = ['analysis/factorswap_pl_stats.py', '../pcdet/datasets/loader_factor_swap.py', '../pcdet/datasets/osens2_utils.py',
             '../pcdet/datasets/pandaset/pandaset_dataset.py', '../pcdet/datasets/lyft/lyft_dataset.py',
             '../pcdet/datasets/dataset.py', '../pcdet/datasets/point_calibration.py']
    return {f: hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16] for f in files}


def run_target(target, workers, max_frames):
    global DS
    cfg_path, block_name, fsign, out_path = TARGETS[target]
    cfg = load_cfg(cfg_path)
    block = block_of(cfg, block_name)
    assert block.get('TRAIN_FACTOR_SWAP', None) is None, 'stats are measured on the oracle'
    DS = build(cfg, block)
    DS.prepare_data = lambda data_dict: data_dict          # stop where prepare_data would start
    n = len(DS)
    idx = list(range(n)) if not max_frames else list(np.linspace(0, n - 1, max_frames).astype(int))
    print('%s: %s, %d TRAIN frames, reading %d' % (target, cfg_path, n, len(idx)), flush=True)
    ctx = mp.get_context('fork')
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        res = []
        for k, r in enumerate(pool.imap(frame_stats, idx, chunksize=8)):
            res.append(r)
            if (k + 1) % 500 == 0:
                print('  %d / %d frames' % (k + 1, len(idx)), flush=True)
    rows = [row for r in res for row in r['rows']]
    nus = json.load(open(NUS_STATS))
    out = {'target': target, 'config': cfg_path, 'block': block_name, 'split': 'train', 'train_frames': n,
           'frames_read': len(idx), 'shift_coor': list(block.get('SHIFT_COOR', None) or [0, 0, 0]),
           'point_cloud_range': [float(v) for v in DS.point_cloud_range],
           'trained_box_definition': 'prepare_data count >= MIN_POINTS_OF_GT on the loaded cloud, model class, centre '
                                     'inside POINT_CLOUD_RANGE in x / y',
           'nuscenes_stats': NUS_STATS, 'code_sha256_16': code_hashes(),
           'points_per_frame': q([r['n_points'] for r in res]),
           'car_boxes_all': int(sum(r['n_car'] for r in res)),
           'car_boxes_no_point': int(sum(r['n_car_empty'] for r in res)),
           'car_boxes_out_of_range_xy': int(sum(r['n_car_out_of_range'] for r in res))}
    # R6
    out['size'] = {}
    for c in cfg.CLASS_NAMES:
        tm, tdim, tn = iso_median(rows, c)
        ns = nus['size'][c]
        out['size'][c] = {'target_iso_median': tm, 'target_dim_median': tdim, 'target_n': tn,
                          'nuscenes_iso_median': ns['nuscenes_iso_median'], 'nuscenes_n': ns['nuscenes_n'],
                          'ratio': None if tm is None else ns['nuscenes_iso_median'] / tm}
    # R7
    r_car = np.array([r[2] for r in rows if r[0] == 'Car'])
    th = np.histogram(r_car, bins=BOX_EDGES)[0]
    nh = np.asarray(nus['label_range']['nuscenes_car_hist'], dtype=np.float64)
    assert nus['label_range']['edges'] == BOX_EDGES
    pt, pn = th / th.sum(), nh / nh.sum()
    trusted = th >= MIN_TGT_BOXES
    ratio = np.where(th > 0, pn / np.maximum(pt, 1e-12), 1.0)
    keep = np.minimum(1.0, ratio / ratio[trusted].max())
    out['label_range'] = {'edges': BOX_EDGES, 'target_car_hist': th.tolist(), 'nuscenes_car_hist': nh.astype(int).tolist(),
                          'target_share_within_30m': float(pt[:3].sum()), 'nuscenes_share_within_30m': float(pn[:3].sum()),
                          'keep': keep.tolist(), 'kept_share_of_target_cars': float((pt * keep).sum()),
                          'min_target_boxes_trusted': MIN_TGT_BOXES, 'trusted': trusted.tolist()}
    # ground
    g = np.array([r['ground'] for r in res if r['ground'] is not None], dtype=np.float64)
    fwd = np.degrees(np.arctan(fsign * g[:, 0]))
    left = np.degrees(np.arctan(fsign * g[:, 1]))          # x backward -> y right; sign flips with fsign as well
    shift_z = float(out['shift_coor'][2])
    out['ground'] = {'frames_fitted': int(len(g)), 'frames_failed': int(len(res) - len(g)),
                     'forward_axis_sign_along_x': fsign,
                     'forward_slope_deg': q(fwd), 'left_slope_deg': q(left),
                     'height_raw_frame_m': q(g[:, 2]), 'height_loader_frame_m': q(g[:, 2] + shift_z)}
    if target == 'pandaset':
        s = round(float(np.median(fwd)), 2)
        share = float(nus['pitch']['boston_share'])
        out['pitch'] = {'nuscenes_slopes_deg': NUS_SLOPE, 'pandaset_forward_slope_deg': s,
                        'pitch_deg': [round(NUS_SLOPE['boston'] - s, 2), round(NUS_SLOPE['singapore'] - s, 2)],
                        'pitch_prob': [share, 1.0 - share], 'boston_share_source': NUS_STATS,
                        'pivot_loader_frame': [0.0, 0.0, 1.88],
                        'pivot_note': 'published Pandar64 height 1.86 m over a ground at +0.02 after SHIFT_COOR (item 32e)'}
        out['x_forward'] = pandaset_x_forward(DS)
    else:
        gt = float(np.median(g[:, 2]))
        out['height_rule_g'] = {'g_nuscenes_raw': NUS_GROUND, 'nuscenes_shift': NUS_SHIFT, 'g_target_raw': gt,
                                'target_shift_rule_g': NUS_GROUND + NUS_SHIFT - gt,
                                'target_shift_config': shift_z}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def run_hist():
    """The D row's histograms (Lyft-64 source, nuScenes TRAIN target), as train.py's on-the-fly path measures them."""
    from pcdet.datasets.point_calibration import calibration_target_config, compute_range_histogram
    cfg = load_cfg(TARGETS['lyft64'][0])
    np.random.seed(0)
    src = build(cfg, cfg.DATA_CONFIG, training=True)
    c1 = load_cfg(C1_LYFT40)
    sub = c1.DATA_CONFIGS.LYFT_40BEAM
    frames, bins, max_dist = int(sub.HIST_DIST_FRAMES), int(sub.HIST_DIST_BINS), float(sub.HIST_DIST_MAX_DIST)
    tcfg, split = calibration_target_config(c1.DATA_CONFIG_TAR)
    assert split == 'train'
    tgt = build(c1, tcfg, training=False)
    h_src = compute_range_histogram(src, frames, bins, max_dist, logger=logger)
    h_tgt = compute_range_histogram(tgt, frames, bins, max_dist, logger=logger)
    Path(HIST_DIR).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(HIST_DIR, 'hist_dist_lyft64_train.npy'), h_src)
    np.save(os.path.join(HIST_DIR, 'hist_dist_nuscenes_train.npy'), h_tgt)
    # the rate the processor will apply (per_bin_sample_rate with MIN_HIST_BIN_FRACTION 0.01)
    from pcdet.datasets.processor.data_processor import DataProcessor
    proc = DataProcessor([], point_cloud_range=src.point_cloud_range, training=True, num_point_features=3)
    proc.set_hist_dist(h_src, h_tgt, max_dist=max_dist)
    rate = proc.per_bin_sample_rate(EasyDict({'MIN_HIST_BIN_FRACTION': 0.01}))
    out = {'code_sha256_16': code_hashes(), 'source': TARGETS['lyft64'][0] + ' DATA_CONFIG (training mode)', 'target': C1_LYFT40 + ' DATA_CONFIG_TAR, '
           'eval-mode view of its TRAIN split', 'frames': frames, 'bins': bins, 'max_dist': max_dist,
           'src_points_per_frame': float(h_src.sum()), 'tgt_points_per_frame': float(h_tgt.sum()),
           'rate': rate.tolist(), 'expected_kept_share': float((h_src * np.minimum(rate, 1.0)).sum() / h_src.sum()),
           'bins_below_1': int((rate < 1).sum()),
           'files': [os.path.join(HIST_DIR, 'hist_dist_lyft64_train.npy'),
                     os.path.join(HIST_DIR, 'hist_dist_nuscenes_train.npy')]}
    with open(os.path.join(HIST_DIR, 'hist_lyft64_to_nuscenes.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=sorted(TARGETS) + ['lyft64_hist'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-frames', type=int, default=0, help='0 = every TRAIN frame (a strided subset for smoke tests)')
    a = ap.parse_args()
    if a.target == 'lyft64_hist':
        run_hist()
    else:
        run_target(a.target, a.workers, a.max_frames)
