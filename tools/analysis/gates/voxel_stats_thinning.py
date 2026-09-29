"""Gate for the covariance-preserving-thinning hypothesis (20260929_05 item 2).

Does POINT thinning (what sample_points_hist_based does: drop points i.i.d. per range bin) move
the per-voxel statistics GBlobs reads AWAY from the target's, while VOXEL thinning (drop whole
voxels at the same rate) keeps them? Measured on Lyft 40-beam frames at the Lyft -> nuScenes rates
against real nuScenes voxels, per range ring: mean points per occupied voxel, the share of voxels
with < 3 points (degenerate covariance), and the mean log of the largest and smallest covariance
eigenvalues over voxels with >= 3 points.

    python voxel_stats_thinning.py [--frames 60] [--hist_frames 200]
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader, compute_range_histogram  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

RINGS = [(0, 20), (20, 40), (40, 60), (60, 75)]


def voxelize(ds, points):
    d = ds.data_processor.forward({'points': points.astype(np.float32), 'use_lead_xyz': True})
    return d['voxels'][:, :, :3], d['voxel_num_points'], d['voxel_coords']


def voxel_stats(ds, points):
    vox, n, coords = voxelize(ds, points)
    centres = vox.sum(1) / np.maximum(n, 1)[:, None]
    r = np.linalg.norm(centres[:, :2], axis=1)
    rows = {}
    for lo, hi in RINGS:
        m = (r >= lo) & (r < hi)
        if m.sum() < 50:
            rows[(lo, hi)] = None; continue
        nn = n[m]; deg = (nn < 3).mean()
        ok = np.where(m & (n >= 3))[0]
        l1, l3 = [], []
        for i in ok[:4000]:
            p = vox[i, :n[i]]; c = np.cov((p - p.mean(0)).T) if n[i] > 1 else np.zeros((3, 3))
            ev = np.sort(np.linalg.eigvalsh(c))[::-1]
            l1.append(np.log(max(ev[0], 1e-9))); l3.append(np.log(max(ev[2], 1e-9)))
        rows[(lo, hi)] = dict(voxels=int(m.sum()), pts_per_voxel=float(nn.mean()), degenerate=float(deg),
                              log_l1=float(np.mean(l1)) if l1 else np.nan, log_l3=float(np.mean(l3)) if l3 else np.nan)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--frames', type=int, default=60); ap.add_argument('--hist_frames', type=int, default=200)
    a = ap.parse_args()
    cfg = cfg_from_yaml_file('cfgs/da-ieee-access/centerpoint-global-lyft2nuscenes.yaml', EasyDict())
    log = common_utils.create_logger()
    src_cfg = copy.deepcopy(cfg.DATA_CONFIGS.LYFT_40BEAM); src_cfg.HIST_DIST_ON_THE_FLY = False
    src, _, _ = build_dataloader(src_cfg, cfg.CLASS_NAMES, 1, False, workers=0, logger=log, training=False, model_ontology=cfg.ONTOLOGY)
    tgt, _, _ = build_dataloader(cfg.DATA_CONFIG_TAR, cfg.CLASS_NAMES, 1, False, workers=0, logger=log, training=False, model_ontology=cfg.ONTOLOGY)
    bins, max_dist = 50, 75.0
    hs = compute_range_histogram(src, a.hist_frames, bins, max_dist); ht = compute_range_histogram(tgt, a.hist_frames, bins, max_dist)
    rate = np.minimum(np.where(hs > 0, ht / np.maximum(hs, 1e-9), 1.0), 1.0)
    edges = np.linspace(0, max_dist, bins + 1)
    print('rate per bin (min/median/max): %.2f / %.2f / %.2f' % (rate.min(), np.median(rate), rate.max()))

    def binidx(xy):
        return np.clip(np.digitize(np.linalg.norm(xy, axis=1), edges) - 1, 0, bins - 1)

    acc = {k: {r: [] for r in RINGS} for k in ('lyft_raw', 'lyft_point_thin', 'lyft_voxel_thin', 'nuscenes')}
    rng = np.random.default_rng(0)
    idx = np.linspace(0, len(src) - 1, a.frames).astype(int)
    for i in idx:
        pts = src[int(i)]['points'][:, :3]
        acc['lyft_raw'] and None
        for k, rows in (('lyft_raw', voxel_stats(src, pts)),):
            for r, v in rows.items():
                if v: acc[k][r].append(v)
        keep = rng.random(len(pts)) < rate[binidx(pts[:, :2])]
        for r, v in voxel_stats(src, pts[keep]).items():
            if v: acc['lyft_point_thin'][r].append(v)
        vox, n, coords = voxelize(src, pts)
        centres = vox.sum(1) / np.maximum(n, 1)[:, None]
        keep_v = rng.random(len(vox)) < rate[binidx(centres[:, :2])]
        # rebuild the point cloud from the surviving voxels, untouched inside
        surv = np.concatenate([vox[j, :n[j]] for j in np.where(keep_v)[0]]) if keep_v.any() else np.zeros((0, 3))
        for r, v in voxel_stats(src, surv).items():
            if v: acc['lyft_voxel_thin'][r].append(v)
    for i in np.linspace(0, len(tgt) - 1, a.frames).astype(int):
        for r, v in voxel_stats(tgt, tgt[int(i)]['points'][:, :3]).items():
            if v: acc['nuscenes'][r].append(v)
    print('%-16s %-7s %8s %8s %8s %8s %8s' % ('variant', 'ring', 'voxels', 'pts/vox', 'deg<3', 'log_l1', 'log_l3'))
    for r in RINGS:
        for k in ('nuscenes', 'lyft_raw', 'lyft_point_thin', 'lyft_voxel_thin'):
            rows = acc[k][r]
            if not rows: continue
            print('%-16s %-7s %8.0f %8.2f %8.3f %8.2f %8.2f' % (k, '%d-%d' % r, np.mean([x['voxels'] for x in rows]), np.mean([x['pts_per_voxel'] for x in rows]),
                  np.mean([x['degenerate'] for x in rows]), np.nanmean([x['log_l1'] for x in rows]), np.nanmean([x['log_l3'] for x in rows])))
        print()


if __name__ == '__main__':
    main()
