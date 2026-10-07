"""Side views of the points on individual Car surfaces: consecutive accumulation vs displacement-spread accumulation
(same nuScenes anchors, same sweep count) vs Waymo. Picks Car boxes at 20-50 m with 30-200 points, draws each box's
points in the box frame (x along the body, z up) from the side and from above, three columns. ANALYSIS: reads Car GT
boxes on both sides. Output: PNG.

    python analysis/visualize_box_patterns.py <cfg_waymo_target> <out.png> [n_boxes_per_column]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import _augmentation_off
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

cfg_file, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4
cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
logger = common_utils.create_logger()
rng = np.random.RandomState(0)


def box_frame(p, b):
    c, s = np.cos(-b[6]), np.sin(-b[6])
    q = p[:, :3] - b[:3]
    x = c * q[:, 0] - s * q[:, 1]; y = s * q[:, 0] + c * q[:, 1]
    return np.stack([x, y, q[:, 2]], 1)


def moved(ds, i, min_m):
    sw = ds.infos[i].get('sweeps', [])
    if not sw:
        return True
    return max(np.linalg.norm(s['transform_matrix'][:3, 3]) if s.get('transform_matrix') is not None else 0.0 for s in sw) >= min_m


def pick(ds, n, lo=20, hi=50, pmin=30, pmax=200, indices=None, min_move=0.0):
    """(range, points in box frame, box dims, index) for n boxes, ONE per frame; min_move skips anchors whose ego
    travelled less than that over the stored sweeps (a stationary ego makes the displacement rule a no-op)."""
    out = []
    idxs = indices if indices is not None else range(0, len(ds), max(1, len(ds) // 400))
    for i in idxs:
        if min_move > 0 and not moved(ds, i, min_move):
            continue
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
            r = np.hypot(*b[j, :2]); c = int(m[j].sum())
            if lo <= r < hi and pmin <= c <= pmax:
                out.append((r, box_frame(pts[m[j] > 0], b[j]), b[j, 3:6], i, j))
                break  # one box per frame
        if len(out) >= n:
            break
    return out[:n]


cols = []
src_key = 'NUSCENES_N008'
dc = cfg.DATA_CONFIGS[src_key]
dsa, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                             logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
with _augmentation_off(dsa):
    a = pick(dsa, N, min_move=20.0)
frames = [x[3] for x in a]
dc2 = EasyDict(dict(dc)); dc2['SWEEP_SELECTION'] = {'MODE': 'displacement', 'SPAN_M': 27.0}
dsb, _, _ = build_dataloader(dataset_cfg=dc2, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                             logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
with _augmentation_off(dsb):
    b = []
    for (r, _, dims, i, j) in a:  # the SAME boxes under the spread selection
        d = dsb[i]; bb = d['gt_boxes']; bb = bb[bb[:, 7] == 1]
        pts = d['points'][:, :3]; m = roiaware_pool3d_utils.points_in_boxes_cpu(pts, bb[:, :7])
        b.append((np.hypot(*bb[j, :2]), box_frame(pts[m[j] > 0], bb[j]), bb[j, 3:6], i, j))
dst, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                             workers=0, logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
w = pick(dst, N)

cols = [('nuScenes Boston, 15 CONSECUTIVE sweeps (as trained); ego moved >= 20 m over the stored 10 s', a),
        ('same cars, 15 sweeps spread over <= 27 m of ego travel', b),
        ('Waymo val (single frame, 64 beams)', w)]
fig, axes = plt.subplots(2 * N, 3, figsize=(13, 3.1 * N))
for ci, (title, boxes) in enumerate(cols):
    for bi in range(N):
        if bi >= len(boxes):
            continue
        r, p, dims, _, _ = boxes[bi]
        ax = axes[2 * bi, ci]; ax.scatter(p[:, 0], p[:, 2], s=4, c='k'); ax.set_aspect('equal')
        ax.set_xlim(-dims[0] / 2 - 0.2, dims[0] / 2 + 0.2); ax.set_ylim(-dims[2] / 2 - 0.2, dims[2] / 2 + 0.2)
        ax.set_title(f'{title}\nside view, {r:.0f} m, {len(p)} pts' if bi == 0 else f'side view, {r:.0f} m, {len(p)} pts', fontsize=8)
        ax.set_xlabel('along body (m)', fontsize=7); ax.set_ylabel('up (m)', fontsize=7); ax.tick_params(labelsize=6)
        ax = axes[2 * bi + 1, ci]; ax.scatter(p[:, 0], p[:, 1], s=4, c='k'); ax.set_aspect('equal')
        ax.set_xlim(-dims[0] / 2 - 0.2, dims[0] / 2 + 0.2); ax.set_ylim(-dims[1] / 2 - 0.2, dims[1] / 2 + 0.2)
        ax.set_title('top view', fontsize=8); ax.set_xlabel('along body (m)', fontsize=7); ax.set_ylabel('across (m)', fontsize=7); ax.tick_params(labelsize=6)
plt.tight_layout(); plt.savefig(out, dpi=110)
print('saved', out, 'boxes', [len(x[1]) for _, x in cols])
