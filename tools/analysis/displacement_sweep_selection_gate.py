"""Gate for displacement-based sweep selection (user's idea, 2026-10-07 23:30: pick accumulation frames by EGO
DISPLACEMENT so the same spots on a distant object are not stacked).

Mechanism (experiments_md 20261007_03 §9): consecutive sweeps move a ring's hit on a distant body by dr * tan(theta)
(~1.7 cm per 0.5 m sweep at 40 m), so 15 consecutive sweeps stack returns into clumps 4-5 cm apart. Spreading the
SAME number of sweeps over a displacement span D lets the rings land on new stripes. This measures, for the same anchors
and the same sweep count, the per-box arrangement (occupied voxels, points per voxel, z-layers, NN distance) under
  (a) consecutive sweeps (the loader as it is), and
  (b) sweeps chosen nearest to N-1 displacement targets spread uniformly over [0, min(D, available)].
Compensation, remove_ego_points and the processor are the loader's own. Source GT boxes only (UDA-legal as a method
input; they are the training labels). Reference target values: 20261007_03 §9.2.

    python analysis/displacement_sweep_selection_gate.py <cfg> <DATA_CONFIGS key> <frames> <D metres>
"""
import sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import _augmentation_off
from pcdet.datasets.nuscenes.nuscenes_dataset import sweep_range_mask
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

VOX = np.array([0.1, 0.1, 0.15])
PTS_BINS = [(10, 50), (50, 200), (200, 10**9)]
RINGS = [(0, 20), (20, 40), (40, 75)]
cfg_file, key, n_frames, D = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
dc = cfg.DATA_CONFIGS[key]
logger = common_utils.create_logger()
ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                            logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
MAX_SWEEPS = dc.MAX_SWEEPS
SELECT = {'mode': 'consecutive'}
spans = []


def select_sweeps(info, max_sweeps):
    sw = info['sweeps']
    if SELECT['mode'] == 'consecutive':
        return list(range(min(max_sweeps - 1, len(sw))))
    disp = np.array([np.linalg.norm(s['transform_matrix'][:3, 3]) if s['transform_matrix'] is not None else 0.0 for s in sw])
    reach = min(D, disp.max()) if len(disp) else 0.0
    spans.append(reach)
    targets = np.linspace(0, reach, max_sweeps)[1:]  # N-1 targets, the anchor is displacement 0
    chosen = []
    for t in targets:
        order = np.argsort(np.abs(disp - t))
        k = next((int(o) for o in order if int(o) not in chosen), None)
        if k is not None:
            chosen.append(k)
    return sorted(chosen)


def get_lidar_with_sweeps(index, max_sweeps=1):  # the loader's method with the sweep index list injected
    self = ds
    info = self.infos[index]
    points = np.fromfile(str(self.root_path / info['lidar_path']), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
    points = self.remove_ego_points(points, center_radius=1.5)
    pl = [points]; tl = [np.zeros((points.shape[0], 1))]
    schedule = self.dataset_cfg.get('ACCUMULATION_DEPTH_BY_RANGE', None)
    for j, k in enumerate(select_sweeps(info, max_sweeps)):
        ps, ts = self.get_sweep(info['sweeps'][k])
        if self._sweep_compensator is not None:
            ps = self._sweep_compensator.compensate_sweep(info, info['sweeps'][k], ps)
        if schedule is not None:
            keep = sweep_range_mask(ps, j + 1, schedule); ps, ts = ps[keep], ts[keep]
        pl.append(ps); tl.append(ts)
    points = np.concatenate(pl, 0); times = np.concatenate(tl, 0).astype(points.dtype)
    return np.concatenate((points, times), 1)


ds.get_lidar_with_sweeps = get_lidar_with_sweeps


def per_box(n):
    rows = []
    step = max(1, len(ds) // n)
    for i in range(0, len(ds), step)[:n]:
        d = ds[i]
        b = d.get('gt_boxes')
        if b is None or not len(b):
            continue
        b = b[b[:, 7] == 1]
        if not len(b):
            continue
        pts = d['points'][:, :3]
        m = roiaware_pool3d_utils.points_in_boxes_cpu(pts, b[:, :7])
        for j in range(len(b)):
            p = pts[m[j] > 0]
            if len(p) < 10:
                continue
            v = np.floor(p / VOX).astype(np.int64)
            occ = np.unique(v, axis=0)
            nn = cKDTree(p).query(p, k=2)[0][:, 1]
            rows.append((np.hypot(*b[j, :2]), len(p), len(occ), len(p) / len(occ), len(np.unique(v[:, 2])),
                         len(np.unique(v[:, :2], axis=0)), np.median(nn)))
    return np.array(rows)


def report(name, r):
    print(f'\n{name}: {len(r)} Car boxes with >= 10 points, median points per box by ring: ' +
          ', '.join(f'{lo}-{hi} m {np.median(r[(r[:,0]>=lo)&(r[:,0]<hi),1]):.0f}' for lo, hi in RINGS))
    print('| ring | points in box | boxes | occupied voxels | pts / occupied voxel | z-layers (0.15 m) | BEV cells (0.1 m) | median NN (m) |')
    print('|---|---|---|---|---|---|---|---|')
    for lo, hi in RINGS:
        for a, z in PTS_BINS:
            s = (r[:, 0] >= lo) & (r[:, 0] < hi) & (r[:, 1] >= a) & (r[:, 1] < z)
            if s.sum() < 5:
                continue
            q = np.median(r[s], axis=0)
            print(f'| {lo}-{hi} | {a}-{z - 1 if z < 10**9 else ""} | {s.sum()} | {q[2]:.0f} | {q[3]:.2f} | {q[4]:.0f} | {q[5]:.0f} | {q[6]:.3f} |')


with _augmentation_off(ds):
    SELECT['mode'] = 'consecutive'
    report(f'(a) {key}: {MAX_SWEEPS} consecutive sweeps', per_box(n_frames))
    SELECT['mode'] = 'displacement'
    rb = per_box(n_frames)
    sp = np.array(spans)
    report(f'(b) {key}: {MAX_SWEEPS} sweeps spread over displacement <= {D:.0f} m (reached: p25/p50/p75 = '
           f'{np.percentile(sp, 25):.1f} / {np.percentile(sp, 50):.1f} / {np.percentile(sp, 75):.1f} m)', rb)
