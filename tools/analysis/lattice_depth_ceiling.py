"""Coverage ceiling of the virtual Waymo lattice against sweep depth, with its cost (experiments_md 20261008_01 §5).

Source mode, one nuScenes platform of a displacement-spread config, the SAME strided frames at every depth (the
dataset reads MAX_SWEEPS per __getitem__, so one object serves all depths); training-mode loader, augmentation off,
whatever DATA_PROCESSOR the config has (use the -disp- config: no sampler):

    python analysis/lattice_depth_ceiling.py source <cfg> <DATA_CONFIGS key> <depths, e.g. 15,30,60> <base depth> <frames>

Waymo mode, the reference: occupied detector voxels per ring from strided TRAIN clouds (unlabelled; the rule's target)
and, for analysis only, per-box arrangement of Waymo val Car boxes with 10-199 points:

    python analysis/lattice_depth_ceiling.py waymo <cfg> <train frames> <val frames>

Per ring (planar range 0-10 / ... / 50-75 m): occupied virtual-lattice pixels per frame (z-buffer from the anchor, the
published Waymo TOP inclinations, 2,650 columns, sensor 2.184 m above ground) against Waymo's native valid pixels, and
occupied detector voxels per frame. Per box: Car GT with 10-199 points at the base depth, paired across depths by box
identity, rings 20-40 / 40-75 m: median points, occupied voxels (0.1 x 0.1 x 0.15 m), z-layers (0.15 m). Cost: points
and occupied voxels per frame, frames above the 250k training voxel cap, loader seconds per frame (master node: read
as a ratio to the base depth; timed warm, after one untimed call).
"""
import sys
import time
import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import _augmentation_off, calibration_target_config
from pcdet.datasets.processor.point_sampler import lattice_pixels, DET_VOXEL, DET_PCR
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

RINGS = [0, 10, 20, 30, 40, 50, 75]
BOX_RINGS = [(20, 40), (40, 75)]
BOX_VOX = np.array([0.1, 0.1, 0.15])
CAP = 250000
STATS = '/home/koyama/data/samplers/waymo_top_native_lattice_stats.npz'
N_COLS, SENSOR_Z = 2650, 2.184


def ring_of(r):
    g = np.digitize(r, RINGS) - 1
    return np.where((g >= 0) & (g < len(RINGS) - 1), g, -1)


def voxels(x):
    """Occupied detector voxels per ring (by voxel-centre planar range) and in total."""
    ins = np.all((x >= DET_PCR[:3]) & (x < DET_PCR[3:]), 1)
    ijk = np.floor((x[ins] - DET_PCR[:3]) / DET_VOXEL).astype(np.int64)
    uniq = np.unique((ijk[:, 0] * 2000 + ijk[:, 1]) * 100 + ijk[:, 2])
    i0, i1 = uniq // 200000, (uniq // 100) % 2000
    cen = DET_PCR[:2] + (np.stack([i0, i1], 1) + 0.5) * DET_VOXEL[:2]
    g = ring_of(np.hypot(cen[:, 0], cen[:, 1]))
    return np.bincount(g[g >= 0], minlength=6)[:6], len(uniq)


def pixels(x, inc):
    """Occupied virtual pixels per ring, each pixel placed by its nearest point's planar range."""
    pix, _, _ = lattice_pixels(x, inc, N_COLS, SENSOR_Z)
    ok = pix >= 0
    up, inv = np.unique(pix[ok], return_inverse=True)
    near = np.full(len(up), np.inf)
    np.minimum.at(near, inv, np.hypot(x[ok, 0], x[ok, 1]))
    g = ring_of(near)
    return np.bincount(g[g >= 0], minlength=6)[:6]


def car_boxes(d):
    b = d.get('gt_boxes')
    if b is None or not len(b):
        return np.zeros((0, 8)), []
    b = b[b[:, 7] == 1]
    if not len(b):
        return b, []
    pts = d['points'][:, :3]
    m = roiaware_pool3d_utils.points_in_boxes_cpu(pts, b[:, :7])
    stats = []
    for j in range(len(b)):
        p = pts[m[j] > 0]
        v = np.floor(p / BOX_VOX).astype(np.int64)
        stats.append((len(p), len(np.unique(v, axis=0)) if len(p) else 0, len(np.unique(v[:, 2])) if len(p) else 0))
    return b, stats


def fmt(v, scale=1.0, f='{:.1f}'):
    return ' / '.join(f.format(a * scale) for a in v)


def source_mode(cfg, key, depths, base, n):
    st = np.load(STATS); inc = st['inclinations']; W_pix = st['ring_valid']
    dc = cfg.DATA_CONFIGS[key]
    ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
    assert base in depths
    acc = {d: dict(pix=np.zeros(6), vox=np.zeros(6), pts=[], nvox=[], sec=[]) for d in depths}
    boxes = {d: [] for d in depths}   # per frame: (box rows, stats)
    step = max(1, len(ds) // n); frames = list(range(0, len(ds), step))[:n]
    with _augmentation_off(ds):
        for fi, i in enumerate(frames):
            for d in depths:
                ds.dataset_cfg.MAX_SWEEPS = d
                ds[i]   # warm the page cache so every depth is timed warm (cold reads scale with the d files read)
                t = time.perf_counter(); s = ds[i]; sec = time.perf_counter() - t
                x = s['points'][:, :3].astype(np.float64)
                a = acc[d]; a['pix'] += pixels(x, inc); vr, nv = voxels(x); a['vox'] += vr
                a['pts'].append(len(x)); a['nvox'].append(nv); a['sec'].append(sec)
                boxes[d].append(car_boxes(s))
            if (fi + 1) % 10 == 0:
                print(f'  {key}: {fi + 1} / {len(frames)} frames', flush=True)
    m = len(frames)
    print(f'\n## {key}: {m} strided frames, SWEEP_SELECTION {dc.get("SWEEP_SELECTION")}, depths {depths} (base {base})')
    print('\nPer ring 0-10 / 10-20 / 20-30 / 30-40 / 40-50 / 50-75 m, per frame:')
    print('| depth | occupied virtual pixels (k) | share of Waymo native valid pixels | occupied detector voxels (k) |')
    print('|---|---|---|---|')
    print(f'| Waymo native | {fmt(W_pix, 1e-3)} | 1 | (waymo mode) |')
    for d in depths:
        a = acc[d]
        print(f'| {d} | {fmt(a["pix"] / m, 1e-3)} | {fmt(a["pix"] / m / W_pix, 1, "{:.2f}")} | {fmt(a["vox"] / m, 1e-3)} |')
    print('\nCost per frame:')
    print('| depth | points (k, median) | occupied voxels (k, median / p90 / max) | frames > 250k voxels | loader s (median) | x base |')
    print('|---|---|---|---|---|---|')
    tb = np.median(acc[base]['sec'])
    for d in depths:
        a = acc[d]; nv = np.array(a['nvox'])
        print(f'| {d} | {np.median(a["pts"]) / 1e3:.0f} | {np.median(nv) / 1e3:.0f} / {np.percentile(nv, 90) / 1e3:.0f} / '
              f'{nv.max() / 1e3:.0f} | {(nv > CAP).mean():.2f} | {np.median(a["sec"]):.2f} | {np.median(a["sec"]) / tb:.1f} |')
    # paired per box: Car boxes with 10-199 points at the base depth
    rows, unmatched = [], 0
    for f in range(m):
        b0, s0 = boxes[base][f]
        for j in range(len(b0)):
            if not 10 <= s0[j][0] < 200:
                continue
            r = np.hypot(*b0[j, :2]); rec = [r]
            for d in depths:
                bd, sd = boxes[d][f]
                k = np.where(np.all(np.isclose(bd[:, :7], b0[j, :7]), 1))[0] if len(bd) else []
                rec += list(sd[k[0]]) if len(k) else [0, 0, 0]
                unmatched += int(not len(k))
            rows.append(rec)
    rows = np.array(rows)
    print(f'\nPaired sparse cars (10-199 points at depth {base}), medians; {unmatched} box-depth pairs unmatched (counted as 0):')
    print('| ring | boxes | ' + ' | '.join(f'depth {d}: points / voxels / z-layers' for d in depths) + ' |')
    print('|---|---|' + '---|' * len(depths))
    for lo, hi in BOX_RINGS:
        s = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if s.sum() < 5:
            continue
        cells = []
        for k in range(len(depths)):
            q = np.median(rows[s, 1 + 3 * k: 4 + 3 * k], axis=0)
            cells.append(f'{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}')
        print(f'| {lo}-{hi} | {s.sum()} | ' + ' | '.join(cells) + ' |')


def waymo_mode(cfg, n_train, n_val):
    logger = common_utils.create_logger()
    st = np.load(STATS)
    calib_cfg, split = calibration_target_config(cfg.DATA_CONFIG_TAR)
    tr, _, _ = build_dataloader(dataset_cfg=calib_cfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    step = max(1, len(tr) // n_train); vox, nvox, pts = np.zeros(6), [], []; per = []
    for i in list(range(0, len(tr), step))[:n_train]:
        x = tr[i]['points'][:, :3].astype(np.float64); vr, nv = voxels(x)
        vox += vr; per.append(vr); nvox.append(nv); pts.append(len(x))
    m = len(per); per = np.array(per)
    print(f'\n## Waymo {split} split, {m} strided frames (unlabelled): the depth rule\'s target')
    print(f'occupied detector voxels per ring per frame (k), mean: {fmt(vox / m, 1e-3)}')
    print(f'  median: {fmt(np.median(per, 0), 1e-3)}; p25: {fmt(np.percentile(per, 25, 0), 1e-3)}; p75: {fmt(np.percentile(per, 75, 0), 1e-3)}')
    print(f'points per frame (k, median) {np.median(pts) / 1e3:.0f}; occupied voxels (k, median) {np.median(nvox) / 1e3:.0f}; '
          f'native valid pixels per ring (k) {fmt(st["ring_valid"], 1e-3)}')
    va, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    step = max(1, len(va) // n_val); rows = []
    for i in list(range(0, len(va), step))[:n_val]:
        b, s = car_boxes(va[i])
        for j in range(len(b)):
            if 10 <= s[j][0] < 200:
                rows.append((np.hypot(*b[j, :2]),) + tuple(s[j]))
    rows = np.array(rows)
    print(f'\nWaymo val Car boxes with 10-199 points (ANALYSIS: reads labels), {n_val} strided frames, medians:')
    print('| ring | boxes | points / voxels / z-layers |')
    print('|---|---|---|')
    for lo, hi in BOX_RINGS:
        s = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        q = np.median(rows[s, 1:], axis=0)
        print(f'| {lo}-{hi} | {s.sum()} | {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} |')


if __name__ == '__main__':
    mode, cfg_file = sys.argv[1], sys.argv[2]
    cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
    if mode == 'source':
        source_mode(cfg, sys.argv[3], [int(v) for v in sys.argv[4].split(',')], int(sys.argv[5]), int(sys.argv[6]))
    else:
        waymo_mode(cfg, int(sys.argv[3]), int(sys.argv[4]))
