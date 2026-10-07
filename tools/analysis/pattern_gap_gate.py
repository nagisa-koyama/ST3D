"""The gate of experiments_md/20261003_04 section 5: does the per-bin density rule leave a
scan-PATTERN gap to the target that a learned sampler could close?

For a config that carries the on-the-fly correction (sample_points_hist_based), measures per-frame
descriptors of local point geometry for three clouds and reports, per 10 m ring and descriptor, the
Jensen-Shannon distance to the pooled target against the target's OWN spread (the median distance of a
single target frame to the pooled target - a pooled-halves floor is near zero and says nothing):

  (a) the source as the loader yields it (accumulated if the config accumulates), rule OFF
  (b) the same frames after the per-bin rule, exactly as training applies it
  (c) the target, eval view, TRAIN split (never val - the calibration-split lesson, 20260930_02)

Descriptors (histograms, per ring of planar radius):
  nn      nearest-neighbour distance (24 log bins, 1 cm - 2 m)        - local density / ring smear
  voxel   points per occupied voxel at the detector's 0.1/0.1/0.15 m   - what the encoder sees
  elev    elevation angle of the point about the sensor (0.25 deg)   - ring structure
  radial  planar radius (1.5 m bins)                                 - what the rule matches

The source is sampled with augmentation OFF (world rotation/flip would smear every descriptor; the
S4 calibration bug, 20261001_03 section 5c). The rule's histograms are linked exactly as train.py does
(link_point_calibration, target train split), with --hist_frames frames per side.

    python pattern_gap_gate.py <cfg> [--frames 300] [--hist_frames 300] [--out table.md]

Opt-in options (2026-10-07, experiments_md 20261007_03): --no_rule for a config without the correction ((b) = (a));
--target_sensor_z / --source_sensor_z for frames whose origin is not the sensor (Waymo: TOP lidar at +2.184 m,
SHIFT_COOR 0); --inside_boxes to restrict every descriptor to points inside Car GT boxes (ANALYSIS: reads labels).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader, link_point_calibration  # noqa: E402
from pcdet.datasets.point_calibration import _augmentation_off, calibration_target_config  # noqa: E402

RINGS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
NN_EDGES = np.logspace(-2, np.log10(2.0), 25)
VOX_EDGES = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 9.5, 19.5, 1e9])
ELEV_EDGES = np.arange(-30, 10.01, 0.25)
RAD_EDGES = np.arange(0, 50.01, 1.5)
VOXEL = np.array([0.1, 0.1, 0.15])


def descriptors(xyz, shift_z):
    """Per-ring histograms (unnormalised counts) for one cloud; xyz in the loader's (shifted) frame."""
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    z = xyz[:, 2] - shift_z
    elev = np.degrees(np.arctan2(z, r))
    out = {}
    if len(xyz) > 1:
        nn = cKDTree(xyz).query(xyz, k=2)[0][:, 1]
    else:
        nn = np.zeros(len(xyz))
    vox_idx = np.floor(xyz / VOXEL).astype(np.int64)
    for lo, hi in RINGS:
        m = (r >= lo) & (r < hi)
        d = {}
        d['nn'] = np.histogram(np.clip(nn[m], NN_EDGES[0], NN_EDGES[-1] - 1e-6), bins=NN_EDGES)[0]
        if m.any():
            _, counts = np.unique(vox_idx[m], axis=0, return_counts=True)
        else:
            counts = np.zeros(0)
        d['voxel'] = np.histogram(counts, bins=VOX_EDGES)[0]
        d['elev'] = np.histogram(np.clip(elev[m], -30, 9.99), bins=ELEV_EDGES)[0]
        out[(lo, hi)] = d
    out['radial'] = np.histogram(np.clip(r, 0, 49.99), bins=RAD_EDGES)[0]
    return out


def accumulate(acc, d):
    if acc is None:
        return {k: ({kk: vv.astype(np.float64) for kk, vv in v.items()} if isinstance(v, dict) else v.astype(np.float64))
                for k, v in d.items()}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                acc[k][kk] += vv
        else:
            acc[k] += v
    return acc


def js(p, q):
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12); m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def inside_car_boxes(d):
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    b = d.get('gt_boxes')
    if b is None or not len(b):
        return np.zeros(len(d['points']), dtype=bool)
    b = b[b[:, 7] == 1]
    if not len(b):
        return np.zeros(len(d['points']), dtype=bool)
    return roiaware_pool3d_utils.points_in_boxes_cpu(d['points'][:, :3], b[:, :7]).any(0)


INSIDE_BOXES = False


def sample(dataset, n_frames, shift_z, label, keep_frames=False):
    """Pooled descriptors over n_frames strided frames; with keep_frames also every frame's own."""
    n = len(dataset); step = max(1, n // n_frames)
    acc = None; frames = []
    used = 0
    for idx in range(0, n, step):
        if used >= n_frames:
            break
        dd = dataset[idx]
        pts = dd['points'][:, :3]
        if INSIDE_BOXES:
            pts = pts[inside_car_boxes(dd)]
        if len(pts) == 0:
            continue
        d = descriptors(pts, shift_z)
        acc = accumulate(acc, d)
        if keep_frames:
            frames.append(d)
        used += 1
        if used % 50 == 0:
            print(f'  {label}: {used} frames', flush=True)
    return acc, frames


def frame_floor(frames, pooled, ring, key):
    """Median JS distance of a single target frame to the pooled target: the target's own spread."""
    if ring == 'radial':
        vals = [js(f['radial'], pooled['radial']) for f in frames]
    else:
        vals = [js(f[ring][key], pooled[ring][key]) for f in frames]
    return float(np.median(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cfg'); ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--hist_frames', type=int, default=300); ap.add_argument('--out', default=None)
    ap.add_argument('--no_rule', action='store_true'); ap.add_argument('--inside_boxes', action='store_true')
    ap.add_argument('--target_sensor_z', type=float, default=None); ap.add_argument('--source_sensor_z', type=float, default=None)
    args = ap.parse_args()
    global INSIDE_BOXES
    INSIDE_BOXES = args.inside_boxes
    os.chdir(TOOLS)
    import logging
    logger = logging.getLogger('gate'); logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)
    cfg = EasyDict(); cfg_from_yaml_file(args.cfg, cfg)
    data_configs = cfg.get('DATA_CONFIGS') or {'DATA_CONFIG': cfg.DATA_CONFIG}
    calib_cfg, split = calibration_target_config(cfg.DATA_CONFIG_TAR)
    target, _, _ = build_dataloader(dataset_cfg=calib_cfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                    workers=0, logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    tz = float(calib_cfg.get('SHIFT_COOR', [0, 0, 0])[2]) if args.target_sensor_z is None else args.target_sensor_z
    print(f'target: {type(target).__name__}, {split} split, {len(target)} frames', flush=True)
    tgt, tgt_frames = sample(target, args.frames, tz, 'target', keep_frames=True)

    src_a = src_b = None
    per_source = max(1, args.frames // len(data_configs))
    for name, dc in data_configs.items():
        ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                    workers=0, logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
        sz = float(dc.get('SHIFT_COOR', [0, 0, 0])[2]) if args.source_sensor_z is None else args.source_sensor_z
        if args.no_rule:
            with _augmentation_off(ds):
                a, _ = sample(ds, per_source, sz, f'{name} (no rule)')
            src_a = accumulate(src_a, a); src_b = accumulate(src_b, {k: ({kk: vv.copy() for kk, vv in v.items()} if isinstance(v, dict) else v.copy()) for k, v in a.items()})
            continue
        with _augmentation_off(ds):
            link_point_calibration(ds, target, num_frames=args.hist_frames, num_bins=dc.get('HIST_DIST_BINS', 50),
                                   max_dist=dc.get('HIST_DIST_MAX_DIST', 75.0), logger=logger,
                                   fov_degree=dc.get('HIST_DIST_FOV_DEGREE', None),
                                   fov_heading=dc.get('HIST_DIST_FOV_HEADING', 0.0))
            proc = ds.data_processor
            saved = proc.hist_dist_src
            proc.hist_dist_src = None  # the rule early-returns: cloud (a)
            a, _ = sample(ds, per_source, sz, f'{name} rule off')
            proc.hist_dist_src = saved  # cloud (b)
            b, _ = sample(ds, per_source, sz, f'{name} rule on')
        src_a = accumulate(src_a, a)
        src_b = accumulate(src_b, b)

    lines = [f'# pattern gap gate: {args.cfg}', '',
             f'frames: target {args.frames} ({split} split), source {per_source} per platform; rule linked on {args.hist_frames} frames',
             '', 'Jensen-Shannon distance (bits) to the pooled target; floor = median distance of ONE target frame to the pooled target (the target\'s own spread). Ratio = (b) / floor: 1x means the rule-thinned source is as close as a typical target frame.', '',
             '| ring | descriptor | rule OFF (a) | rule ON (b) | floor | (b)/floor |', '|---|---|---|---|---|---|']
    for ring in RINGS:
        for key in ('nn', 'voxel', 'elev'):
            fa = js(src_a[ring][key], tgt[ring][key]); fb = js(src_b[ring][key], tgt[ring][key])
            fl = frame_floor(tgt_frames, tgt, ring, key)
            lines.append(f'| {ring[0]}-{ring[1]} m | {key} | {fa:.4f} | {fb:.4f} | {fl:.4f} | {fb / max(fl, 1e-6):.1f}x |')
    fa = js(src_a['radial'], tgt['radial']); fb = js(src_b['radial'], tgt['radial'])
    fl = frame_floor(tgt_frames, tgt, 'radial', 'radial')
    lines.append(f'| 0-50 m | radial | {fa:.4f} | {fb:.4f} | {fl:.4f} | {fb / max(fl, 1e-6):.1f}x |')
    lines.append('')
    lines.append('points per frame: rule off %.0f, rule on %.0f, target %.0f' % (
        src_a['radial'].sum() / per_source / len(data_configs), src_b['radial'].sum() / per_source / len(data_configs),
        tgt['radial'].sum() / args.frames))
    text = '\n'.join(lines)
    print('\n' + text, flush=True)
    if args.out:
        Path(args.out).write_text(text + '\n')


if __name__ == '__main__':
    main()
