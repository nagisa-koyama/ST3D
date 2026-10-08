"""H1 / H8: are accumulated source cars MORE COMPLETE than the target's single-frame cars, and are sparse positives rare?

Per Car box, AS THE MODEL SEES IT (training mode, augmentation off, after the data processor):
- points inside; the share VISIBLE from the anchor sensor (z-buffer on a 0.25 x 0.25 deg virtual lattice about the
  frame's own sensor position: a point is visible if its range is within TOL m of the nearest point in its pixel);
- the number of box faces carrying points (4 sides + roof; a face counts if >= 3 points lie within FACE_M of it).
The same measures on the TARGET's single-frame val boxes (eval mode; visible share ~1 by construction, a sanity check).
Waymo labels are read to diagnose only (ANALYSIS). Reading for H1: if accumulated source cars show more faces and a
large invisible share where Waymo's show 1-2 faces, accumulation de-occludes the source. H8: histogram of points per
training box against the target's.

    python analysis/source_box_visibility.py <cfg> <frames per source> <target frames> <out.npz>
"""
import sys
import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import _augmentation_off
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

DEG = 0.25; TOL = 0.5; FACE_M = 0.2; WAYMO_SENSOR_Z = 2.184
RINGS = [0, 20, 43, 60, 75.2]; PTS = [(10, 50), (50, 200), (200, 10 ** 9)]


def visible(x, sensor_z):
    d = x[:, :3] - np.array([0.0, 0.0, sensor_z])
    r = np.hypot(d[:, 0], d[:, 1]); rng = np.hypot(r, d[:, 2])
    el = np.degrees(np.arctan2(d[:, 2], np.maximum(r, 1e-6))); az = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    pix = (np.floor((el + 90) / DEG).astype(np.int64) * 2000 + np.floor((az + 180) / DEG).astype(np.int64))
    o = np.lexsort((rng, pix)); p, g = pix[o], rng[o]
    first = np.r_[True, p[1:] != p[:-1]]
    near = np.empty(len(o)); idx = np.cumsum(first) - 1
    near_per_pix = g[first]
    near[:] = near_per_pix[idx]
    vis = np.zeros(len(x), bool); vis[o] = g <= near + TOL
    return vis


def faces(pts, box):
    """Which of (front, back, left, right, roof) carry >= 3 points within FACE_M of the face plane."""
    c, s = np.cos(-box[6]), np.sin(-box[6])
    d = pts[:, :3] - box[:3]
    lx = d[:, 0] * c - d[:, 1] * s; ly = d[:, 0] * s + d[:, 1] * c; lz = d[:, 2]
    l, w, h = box[3:6]
    near = [np.abs(lx - l / 2) < FACE_M, np.abs(lx + l / 2) < FACE_M, np.abs(ly - w / 2) < FACE_M,
            np.abs(ly + w / 2) < FACE_M, np.abs(lz - h / 2) < FACE_M]
    return np.array([m.sum() >= 3 for m in near])


def collect(ds, n, sensor_z, car_label=1):
    out = []
    step = max(1, len(ds) // n)
    for i in list(range(0, len(ds), step))[:n]:
        d = ds[i]; b = d['gt_boxes']
        if b is None or not len(b):
            continue
        b = b[b[:, 7] == car_label]
        if not len(b):
            continue
        x = d['points']
        vis = visible(x, sensor_z)
        inb = roiaware_pool3d_utils.points_in_boxes_cpu(x[:, :3], b[:, :7]) > 0
        for j in range(len(b)):
            m = inb[j]; n_in = int(m.sum())
            if n_in < 1:
                continue
            f = faces(x[m], b[j])
            out.append((np.hypot(b[j, 0], b[j, 1]), n_in, vis[m].mean(), f.sum(), f[4]))
    return np.array(out)


def table(name, a):
    print(f'\n{name}: {len(a)} Car boxes with >= 1 point')
    print('| ring | pts bin | boxes | median pts | visible share (median / p25) | faces with points (mean) | roof share | share of boxes 10-49 pts in ring |')
    print('|---|---|---|---|---|---|---|---|')
    for k in range(len(RINGS) - 1):
        ring = (a[:, 0] >= RINGS[k]) & (a[:, 0] < RINGS[k + 1])
        sparse_share = ((a[ring, 1] >= 10) & (a[ring, 1] < 50)).sum() / max((a[ring, 1] >= 10).sum(), 1)
        for lo, hi in PTS:
            s = ring & (a[:, 1] >= lo) & (a[:, 1] < hi)
            if s.sum() < 5:
                continue
            print(f'| {RINGS[k]:.0f}-{RINGS[k+1]:.0f} | {lo}-{hi-1 if hi < 10**9 else ""} | {s.sum()} | {np.median(a[s, 1]):.0f} | '
                  f'{np.median(a[s, 2]):.2f} / {np.percentile(a[s, 2], 25):.2f} | {a[s, 3].mean():.2f} | {a[s, 4].mean():.2f} | {sparse_share:.2f} |')
    print('points-per-box histogram (boxes with >= 10 pts): ' + ', '.join(
        f'{lo}-{hi-1 if hi < 10**9 else ""}: {((a[:, 1] >= lo) & (a[:, 1] < hi)).mean():.2f}' for lo, hi in [(10, 50), (50, 200), (200, 10 ** 9)]))


if __name__ == '__main__':
    cfg_file, n_src, n_tgt, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger()
    res = {}
    for key, dc in cfg.DATA_CONFIGS.items():
        ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                    logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
        sz = (dc.get('SHIFT_COOR', None) or [0, 0, 0])[2]
        with _augmentation_off(ds):
            res[key] = collect(ds, n_src, sz)
        table(f'SOURCE {key} (training mode, aug off, MAX_SWEEPS {dc.get("MAX_SWEEPS")}, sensor z {sz})', res[key])
        del ds
    dst, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                 workers=0, logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    tz = (cfg.DATA_CONFIG_TAR.get('SHIFT_COOR', None) or [0, 0, 0])[2] or WAYMO_SENSOR_Z
    res['TARGET'] = collect(dst, n_tgt, tz)
    table(f'TARGET val (eval mode, single frame, sensor z {tz})', res['TARGET'])
    np.savez(out, **res)
