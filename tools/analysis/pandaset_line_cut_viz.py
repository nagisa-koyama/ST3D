"""Visual check of the PandaSet 'lines /2' cut and its count-matched random control (experiments_md 20261011_07 §22.4;
CenterPoint oracle 26389 D = -0.6, IA-SSD P2 D = -1.2 / adjusted -0.3: no line cue on the Pandar64). CPU; run as a
Slurm job. It draws figures and records counts; it changes no method and decides nothing.

The cut is the one the scored twins ran, called through the loader itself:
`cfgs/da-ieee-access-analysis/sensitivity-pandaset/centerpoint-pandaset-spin2spin-ringrows.yaml` (EVAL_RING_THIN MODE
rows, STRIDE 2: keep channels 0, 2, 4, ... of the recovered Pandar64 channels, 0 = top laser) and its twin `-ringrandom`
(the same number of points drawn uniformly, seeded by the frame). PandaSet VAL cuboids are read only to pick and frame
two cars (one near, one far); no label enters any mask.

    python analysis/pandaset_line_cut_viz.py --out DIR [--n_scan 60]       (from ST3D/tools, inside the container,
                                                                            with --bind /home/koyama/code/ST3D:/root/ST3D)
Figures: fig1_spec.png (channel elevations and spacing, Pandar64 against Waymo TOP), fig2_frame.png (one frame,
elevation against azimuth), fig3_cars.png (two cars seen from the sensor), plus summary.json.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, '.')
import _init_path  # noqa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from easydict import EasyDict

TWIN = 'cfgs/da-ieee-access-analysis/sensitivity-pandaset/centerpoint-pandaset-spin2spin-ring%s.yaml'
WAYMO_CALIB = '/home/koyama/data/waymo_top_calib/top_calib.pkl'
NEAR, FAR = (15.0, 30.0), (40.0, 60.0)   # m from the sensor, horizontal
ARMS = ('full', 'rows', 'random')
TITLES = {'full': 'Pandar64 as recorded', 'rows': 'lines /2 (every other channel)',
          'random': 'random, same point count'}


def build(mode):
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    cfg = EasyDict()
    cfg_from_yaml_file(TWIN % mode, cfg)
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=common_utils.create_logger(), training=False,
                                model_ontology=cfg.get('ONTOLOGY'))
    return ds


def load_frame(ds, i):
    """Device-0 points in stored order (PandaSet ego axes), channel labels, normative points, Car boxes."""
    import pandas as pd
    import pandaset as ps
    from pcdet.datasets.pandaset import pandaset_rings as R
    info = ds.pandaset_infos[i]
    pose = ds._get_pose(info)
    raw = pd.read_pickle(info['lidar_path'])
    raw = raw[raw.d == 0]
    ego = ps.geometry.lidar_points_to_ego(raw[['x', 'y', 'z']].to_numpy(), pose)
    labels = R.load_cached_labels(ds.eval_ring_thin_cfg['LABEL_CACHE'], info['sequence'], info['frame_idx'], len(ego))
    norm = np.stack([ego[:, 1], -ego[:, 0], ego[:, 2]], axis=1)          # the loader's axis swap
    boxes, names, _ = ds._get_annotations(info, pose)
    return info, ego, labels, norm, np.asarray(boxes, dtype=np.float64)[np.asarray(names) == 'Car']


def in_box(p, box, margin=0.0):
    c, d, yaw = box[:3], box[3:6] / 2.0 + margin, box[6]
    q = p - c
    ca, sa = np.cos(yaw), np.sin(yaw)
    u, v = q[:, 0] * ca + q[:, 1] * sa, -q[:, 0] * sa + q[:, 1] * ca
    return (np.abs(u) <= d[0]) & (np.abs(v) <= d[1]) & (np.abs(q[:, 2]) <= d[2])


def sensor_xy():
    from pcdet.datasets.pandaset import pandaset_rings as R
    return np.array([R.ORIGIN[1], -R.ORIGIN[0]])                         # the sensor in normative x, y


def pick_cars(ds, n_scan):
    """Per band, the Car box with the most device-0 points over strided val frames."""
    best = {'near': None, 'far': None}
    s = sensor_xy()
    for i in range(0, len(ds), max(1, len(ds) // n_scan))[:n_scan]:
        info, ego, labels, norm, cars = load_frame(ds, i)
        for b in cars:
            r = float(np.hypot(*(b[:2] - s)))
            n = int(in_box(norm, b).sum())
            for band, (lo, hi) in (('near', NEAR), ('far', FAR)):
                if lo <= r < hi and (best[band] is None or n > best[band][2]):
                    best[band] = (i, b, n, r)
    return best


def masks(ds_rows, ds_rand, i, ego):
    info = ds_rows.pandaset_infos[i]
    keep = {'full': np.ones(len(ego), dtype=bool), 'rows': ds_rows._eval_ring_thin_mask(info, ego),
            'random': ds_rand._eval_ring_thin_mask(info, ego)}
    assert keep['rows'].sum() == keep['random'].sum(), 'count-matched control must keep the same number of points'
    return keep


def channel_colors(ch):
    return plt.cm.tab20(np.asarray(ch) % 20)


def waymo_inclinations():
    with open(WAYMO_CALIB, 'rb') as f:
        d = pickle.load(f)
    incs = [np.sort(np.degrees(np.asarray(v['inclinations'], dtype=np.float64)))[::-1] for v in d.values()]
    return np.median(np.stack([x for x in incs if len(x) == 64]), axis=0), len(incs)


def fig_spec(out):
    from pcdet.datasets.pandaset import pandaset_rings as R
    el = R.EL
    kept = np.arange(64) % 2 == 0
    wy, n_seg = waymo_inclinations()
    fig, ax = plt.subplots(1, 3, figsize=(15, 6.2), gridspec_kw={'width_ratios': [1, 1, 1.6]})
    for a, (lo, hi), ttl in ((ax[0], (-26, 16), 'whole vertical FOV'), (ax[1], (-7, 3), 'zoom: -7 to +3 deg')):
        for x, e, k, src in [(0, el, np.ones(64, bool), 'P'), (1, el, kept, 'P'), (2.6, wy, np.ones(64, bool), 'W'),
                             (3.6, wy, np.arange(64) % 2 == 0, 'W')]:
            for v, kk in zip(e, k):
                a.plot([x - 0.35, x + 0.35], [v, v], color=('C0' if src == 'P' else 'C3') if kk else '0.82',
                       lw=1.2 if kk else 0.8)
        a.set_xticks([0, 1, 2.6, 3.6])
        a.set_xticklabels(['Pandar64\n64 ch', 'Pandar64\nlines /2', 'Waymo TOP\n64 ch', 'Waymo TOP\nlines /2'],
                          fontsize=8)
        a.set_ylim(lo, hi)
        a.set_ylabel('elevation (deg)')
        a.set_title(ttl)
        a.grid(axis='y', alpha=0.3)
    a = ax[2]
    for e, k, c, lab in [(el, np.ones(64, bool), 'C0', 'Pandar64, 64 ch'), (el, kept, 'C0', 'Pandar64, lines /2'),
                         (wy, np.ones(64, bool), 'C3', 'Waymo TOP, 64 ch'),
                         (wy, np.arange(64) % 2 == 0, 'C3', 'Waymo TOP, lines /2')]:
        v = e[k]
        a.plot((v[:-1] + v[1:]) / 2, -np.diff(v), 'o-' if 'lines' not in lab else 's--', color=c, ms=3, lw=1,
               alpha=1.0 if 'lines' not in lab else 0.7, label=lab)
    for y, lab in ((1 / 3, 'KITTI HDL-64E upper block 1/3 deg'), (0.5, 'KITTI HDL-64E lower block 1/2 deg'),
                   (1.33, 'nuScenes HDL-32E 1.33 deg')):
        a.axhline(y, color='0.4', ls=':', lw=1)
        a.text(15.5, y * 1.04, lab, ha='right', va='bottom', fontsize=7, color='0.3')
    a.set_yscale('log')
    a.set_xlim(-26, 16)
    a.set_xlabel('elevation (deg, midpoint of two neighbouring lines)')
    a.set_ylabel('spacing between neighbouring lines (deg)')
    a.set_title('vertical line spacing before and after the cut')
    a.legend(fontsize=8, loc='upper left')
    a.grid(alpha=0.3, which='both')
    fig.suptitle('Pandar64 channel elevations (template from TRAIN clouds, within ~0.15 deg of the published profile) '
                 'and Waymo TOP beam inclinations (median of %d segments)' % n_seg, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig1_spec.png'), dpi=150)
    plt.close(fig)
    fine = (el > -6.3) & (el < 2.0)
    sp = lambda v: -np.diff(v)                                                          # noqa: E731

    def band(v, lo=-5.5, hi=-0.5):   # where a car's side sits at 20-50 m from a ~1.9-2.2 m high sensor
        mid, s = (v[:-1] + v[1:]) / 2, sp(v)
        return float(np.median(s[(mid >= lo) & (mid <= hi)]))
    return {'car_band_deg': [-5.5, -0.5],
            'car_band_spacing_median': {'pandar64': band(el), 'pandar64_rows': band(el[kept]), 'waymo': band(wy),
                                        'waymo_rows': band(wy[::2])},
            'pandar64_fine_band_spacing_median': float(np.median(sp(el[fine]))),
            'pandar64_fine_band_spacing_after_rows_median': float(np.median(sp(el[fine & kept]))),
            'waymo_spacing_median': float(np.median(sp(wy))),
            'waymo_spacing_after_rows_median': float(np.median(sp(wy[::2]))),
            'waymo_spacing_min': float(sp(wy).min()), 'waymo_segments': n_seg}


def fig_frame(out, frame, car):
    from pcdet.datasets.pandaset import pandaset_rings as R
    i, ego, labels, keep, tag = frame
    _, el, az = R.sensor_angles(ego)
    a0 = float(np.median(az[car])) if car.any() else 90.0
    da = (az - a0 + 180.0) % 360.0 - 180.0
    win = (np.abs(da) < 35) & (el > -8) & (el < 3)
    fig, ax = plt.subplots(3, 1, figsize=(14, 12), sharex=True, sharey=True)
    for a, arm in zip(ax, ARMS):
        m = win & keep[arm]
        a.scatter(da[m], el[m], s=0.6, c=channel_colors(labels[m]), linewidths=0)
        mc = m & car
        a.scatter(da[mc], el[mc], s=6, facecolors='none', edgecolors='k', linewidths=0.4)
        a.set_title('%s: %d points in the window, %d channels' % (TITLES[arm], m.sum(), len(np.unique(labels[m]))),
                    fontsize=10)
        a.set_ylabel('elevation about the sensor (deg)')
    ax[-1].set_xlabel('azimuth relative to the near car (deg); black rings = points in that car\'s box')
    fig.suptitle('One PandaSet val frame (%s), colour = Pandar64 channel' % tag, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig2_frame.png'), dpi=150)
    plt.close(fig)


def fig_cars(out, cars):
    s = sensor_xy()
    fig, ax = plt.subplots(len(cars), 3, figsize=(15, 4.6 * len(cars)), squeeze=False)
    rec = []
    for r, (band, i, b, norm, labels, keep, tag) in enumerate(cars):
        m_box = in_box(norm, b, margin=0.25)
        los = b[:2] - s
        rng = float(np.hypot(*los))
        u = (norm[:, :2] - b[:2]) @ (np.array([-los[1], los[0]]) / rng)                # across the line of sight
        row = {'band': band, 'frame': tag, 'range_m': round(rng, 1), 'box_lwh': [round(float(x), 2) for x in b[3:6]]}
        for c, arm in enumerate(ARMS):
            m = m_box & keep[arm]
            a = ax[r, c]
            a.scatter(u[m], norm[m, 2], s=10, c=channel_colors(labels[m]), linewidths=0)
            for ch in np.unique(labels[m]):
                mm = m & (labels == ch)
                a.text(u[mm].max() + 0.05, np.median(norm[mm, 2]), str(ch), fontsize=6, va='center', color='0.3')
            n_ch = len(np.unique(labels[m]))
            a.set_title('%s\n%d points on %d lines' % (TITLES[arm], m.sum(), n_ch), fontsize=9)
            a.set_xlabel('across the line of sight (m)')
            if c == 0:
                a.set_ylabel('%s car, %.1f m: height (m)' % (band, rng))
            a.set_aspect('equal')
            row[arm] = {'points': int(m.sum()), 'lines': n_ch}
        lim_x = [min(a.get_xlim()[0] for a in ax[r]), max(a.get_xlim()[1] for a in ax[r]) + 0.3]
        lim_y = [min(a.get_ylim()[0] for a in ax[r]), max(a.get_ylim()[1] for a in ax[r])]
        for a in ax[r]:
            a.set_xlim(lim_x)
            a.set_ylim(lim_y)
        rec.append(row)
    fig.suptitle('Cars seen from the sensor (points within 0.25 m of the PandaSet val box); colour and number = '
                 'Pandar64 channel (0 = top laser)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig3_cars.png'), dpi=150)
    plt.close(fig)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_scan', type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = {'spec': fig_spec(args.out)}
    ds_rows, ds_rand = build('rows'), build('random')
    best = pick_cars(ds_rows, args.n_scan)
    cars = []
    frames = {}
    for band in ('near', 'far'):
        i, b, n, r = best[band]
        info, ego, labels, norm, _ = load_frame(ds_rows, i)
        keep = masks(ds_rows, ds_rand, i, ego)
        tag = '%s/%02d' % (info['sequence'], int(info['frame_idx']))
        frames[band] = (i, ego, labels, keep, tag, norm, b)
        cars.append((band, i, b, norm, labels, keep, tag))
        summary.setdefault('frames', {})[band] = {
            'val_index': int(i), 'frame': tag, 'points': int(len(ego)), 'kept_rows': int(keep['rows'].sum()),
            'kept_random': int(keep['random'].sum()), 'channels_full': int(len(np.unique(labels))),
            'channels_rows': int(len(np.unique(labels[keep['rows']])))}
    i, ego, labels, keep, tag, norm, b = frames['near']
    fig_frame(args.out, (i, ego, labels, keep, tag), in_box(norm, b))
    summary['cars'] = fig_cars(args.out, cars)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
