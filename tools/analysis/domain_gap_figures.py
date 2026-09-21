"""Render the domain-gap figure set with capture platforms as the unit, not datasets.

Lyft and nuScenes each contain two capture platforms that differ enough to matter (see
experiments_md/20260922_02), so every figure here splits them:

    KITTI | nuScenes n008 Boston | nuScenes n015 Singapore | Lyft 40-beam
    Lyft 64-beam | PandaSet Pandar64 | PandaSet PandarGT | Waymo

Run from ST3D/tools inside the container:

    singularity exec --bind /home/koyama/data/:/storage <image>.sif \
        python analysis/domain_gap_figures.py --out ../../experiments_md [--frames N]

Reuses the platform registry in domain_gap_analysis.py, so the loaders apply exactly what each
dataset class applies. Waymo is excluded from box-level panels (its processed points disagree with
its own annotations in this checkout) and PandaSet/Waymo from the beam panel (ground-origin frames).
"""
import argparse
import collections
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch, Rectangle

from domain_gap_analysis import (Frame, build_platforms, mask_range, points_in_boxes,
                                 radial_hist, MAX_DIST)

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9'
BLUE, ORANGE, RED, DARK = '#2a78d6', '#eb6834', '#d03b3b', '#17457c'
CMAP = LinearSegmentedColormap.from_list('b', ['#dce9f9', '#9cc3ee', '#4a8fdd', '#2a78d6', '#17457c'])
CMAP.set_bad(SURFACE)

ORDER = ['KITTI', 'nuScenes n008 Boston', 'nuScenes n015 Singapore', 'Lyft 40-beam',
         'Lyft 64-beam', 'PandaSet Pandar64', 'PandaSet PandarGT', 'Waymo']
# Waymo's PROCESSED POINTS disagree geometrically with its own annotations in this checkout
# (median counted/annotated = 0.06), so it cannot appear in any panel that counts points inside
# boxes. Its annotations themselves are fine, so it belongs in the label-only panels.
NO_POINT_IN_BOX = {'Waymo'}
NO_BOXES = set()
SENSOR_FRAME = {'KITTI', 'nuScenes n008 Boston', 'nuScenes n015 Singapore',
                'Lyft 40-beam', 'Lyft 64-beam'}
R10 = np.arange(0, 90, 10)
R50 = np.linspace(0, MAX_DIST, 51)
XY = np.arange(-150, 160, 10)
AX = np.arange(-80, 80.5, 1.0)
ZB = np.arange(-4, 6.02, 0.05)
EL = np.arange(-35, 15.02, 0.1)
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
CLASS_MAP = {
    'KITTI': {'Car': {'Car'}, 'Pedestrian': {'Pedestrian'}, 'Cyclist': {'Cyclist'}},
    'nuScenes': {'Car': {'car'}, 'Pedestrian': {'pedestrian'}, 'Cyclist': {'bicycle'}},
    'Lyft': {'Car': {'car'}, 'Pedestrian': {'pedestrian'}, 'Cyclist': {'bicycle'}},
    'PandaSet': {'Car': {'Car'}, 'Pedestrian': {'Pedestrian', 'Pedestrian with Object'},
                 'Cyclist': {'Bicycle'}},
    'Waymo': {'Car': {'Vehicle'}, 'Pedestrian': {'Pedestrian'}, 'Cyclist': {'Cyclist'}},
}


def class_names(platform_name, cls):
    for family, mapping in CLASS_MAP.items():
        if platform_name.startswith(family):
            return mapping[cls]
    raise KeyError(platform_name)


def style(ax, grid_axis='both'):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, lw=.7)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def grid(nrow, ncol, figsize):
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    return fig, np.atleast_2d(axes)


def collect(platform, frames):
    """One pass over sampled frames, gathering everything the point-level panels need."""
    out = dict(r10=np.zeros(len(R10) - 1), r50=np.zeros(len(R50) - 1),
               bev=np.zeros((len(XY) - 1, len(XY) - 1)), x=np.zeros(len(AX) - 1),
               y=np.zeros(len(AX) - 1), z=np.zeros(len(ZB) - 1), el=np.zeros(len(EL) - 1),
               inten=collections.defaultdict(list), box=[], total=0.0, n=0)
    car = class_names(platform.name, 'Car')
    for info in platform.sample(frames):
        fr = platform.frame(info)
        p = mask_range(fr.points)
        out['r10'] += radial_hist(p, R10)
        out['r50'] += radial_hist(p, R50)
        out['total'] += len(p)
        out['n'] += 1
        pxy = fr.points[(np.abs(fr.points[:, 0]) < 150) & (np.abs(fr.points[:, 1]) < 150)]
        out['bev'] += np.histogram2d(pxy[:, 0], pxy[:, 1], bins=[XY, XY])[0]
        inb = fr.points[(np.abs(fr.points[:, 0]) < 75.2) & (np.abs(fr.points[:, 1]) < 75.2)]
        out['x'] += np.histogram(inb[:, 0], bins=AX)[0]
        out['y'] += np.histogram(inb[:, 1], bins=AX)[0]
        out['z'] += np.histogram(inb[:, 2], bins=ZB)[0]
        r = np.linalg.norm(p[:, :2], axis=1)
        ok = r > 3
        out['el'] += np.histogram(np.degrees(np.arctan2(p[ok, 2], r[ok])), bins=EL)[0]
        idx = np.digitize(r, R10) - 1
        for k in range(len(R10) - 1):
            m = idx == k
            if m.sum():
                out['inten'][k].append(p[m, 3])
        if platform.name not in NO_POINT_IN_BOX:
            m = np.isin(fr.names, list(car))
            if m.sum():
                out['box'].append(np.stack([np.linalg.norm(fr.boxes[m][:, :2], axis=1),
                                            points_in_boxes(p, fr.boxes[m])], 1))
    for key in ('r10', 'r50', 'bev', 'x', 'y', 'z', 'el'):
        out[key] = out[key] / max(out['n'], 1)
    out['total'] /= max(out['n'], 1)
    out['box'] = np.concatenate(out['box']) if out['box'] else np.zeros((0, 2))
    return out


def collect_labels(platform, frames):
    """Box dimensions and per-frame counts. Infos carry the boxes except for PandaSet."""
    dims = {c: [] for c in CLASSES}
    counts = {c: [] for c in CLASSES}
    pos = {c: [] for c in CLASSES}
    source = platform.sample(frames * 5) if platform.name.startswith('PandaSet') else platform.infos
    for info in source:
        if platform.name.startswith('PandaSet'):
            fr = platform.frame(info)
            boxes, names = fr.boxes, fr.names
        elif 'gt_boxes' in info:
            boxes, names = np.asarray(info['gt_boxes']), np.asarray(info['gt_names'])
        elif 'annos' in info:
            boxes = np.asarray(info['annos']['gt_boxes_lidar'])
            names = np.asarray(info['annos']['name'])[:len(boxes)]
        else:
            continue
        for c in CLASSES:
            m = np.isin(names, list(class_names(platform.name, c)))
            counts[c].append(int(m.sum()))
            if m.sum():
                dims[c].append(np.asarray(boxes)[m][:, [3, 4, 5, 2]])
                pos[c].append(np.asarray(boxes)[m][:, :2])
    n_frames = max(len(counts[CLASSES[0]]), 1)
    return ({c: (np.concatenate(v) if v else np.zeros((0, 4))) for c, v in dims.items()},
            {c: np.array(v) for c, v in counts.items()},
            {c: (np.concatenate(v) if v else np.zeros((0, 2))) for c, v in pos.items()},
            n_frames)


def kitti_box_reference(platform):
    """KITTI points-per-box over the FULL split, from its own num_points_in_gt.

    Sampling a handful of frames leaves most range bins under the >=10-box floor (KITTI has
    only 3.87 cars per frame), so the reference curve has to come from every frame.
    """
    rows = []
    for info in platform.infos:
        a = info['annos']
        g = np.asarray(a['gt_boxes_lidar'])
        nm = np.asarray(a['name'])[:len(g)]
        m = nm == 'Car'
        if m.sum():
            rows.append(np.stack([np.linalg.norm(g[m][:, :2], axis=1),
                                  np.asarray(a['num_points_in_gt'])[:len(g)][m]], 1))
    return np.concatenate(rows)


def med_by_bin(a, edges=R10, floor=10):
    if len(a) == 0:
        return np.full(len(edges) - 1, np.nan)
    idx = np.digitize(a[:, 0], edges) - 1
    return np.array([np.median(a[idx == i, 1]) if (idx == i).sum() >= floor else np.nan
                     for i in range(len(edges) - 1)])


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def fig_range(D, out):
    ref = D['KITTI']['r10']
    centres = (R10[:-1] + R10[1:]) / 2
    fig, axes = grid(2, 4, (15.5, 8))
    for ax, name in zip(axes.ravel(), ORDER):
        h = D[name]['r10']
        style(ax)
        cols = [RED if (name != 'KITTI' and h[i] < ref[i]) else BLUE for i in range(len(h))]
        ax.bar(centres, h, width=7.4, color=cols, linewidth=0)
        if name != 'KITTI':
            ax.step(np.r_[R10[:-1], R10[-1]], np.r_[ref, ref[-1]], where='post',
                    color=INK2, lw=2, zorder=5)
        ax.set_yscale('log')
        ax.set_ylim(20, 3e5)
        ax.set_xticks([0, 40, 80])
        ax.set_title('%s\n%s pts/frame%s' % (name, format(int(D[name]['total']), ','),
                     '' if name == 'KITTI' else '  (%.2fx)' % (h.sum() / ref.sum())),
                     fontsize=10, color=INK, pad=6)
    for ax in axes[1]:
        ax.set_xlabel('radial distance from sensor (m)', color=INK2, fontsize=8.5)
    axes[1, 0].set_ylabel('points per frame', color=INK2, fontsize=9)
    axes[0, 0].set_ylabel('points per frame', color=INK2, fontsize=9)
    axes[0, 0].legend(handles=[Patch(facecolor=BLUE, label='at or above KITTI'),
                               Patch(facecolor=RED, label='below KITTI')],
                      loc='lower left', frameon=False, fontsize=8)
    fig.suptitle('Point density by radial range, per capture platform — 10 m bins, single frame',
                 fontsize=13, color=INK, y=.975)
    fig.tight_layout(rect=[0, 0, 1, .945])
    fig.savefig(out / 'platform_range_density.png', dpi=150, facecolor=SURFACE)


def fig_bev(D, out):
    vmax = max(D[n]['bev'].max() for n in ORDER)
    fig, axes = grid(2, 4, (15.5, 8.4))
    for ax, name in zip(axes.ravel(), ORDER):
        h = D[name]['bev']
        ax.set_facecolor(SURFACE)
        im = ax.imshow(np.ma.masked_where(h <= 0, h)[::-1, ::-1], extent=[150, -150, -150, 150],
                       norm=LogNorm(vmin=1, vmax=vmax), cmap=CMAP, interpolation='nearest',
                       aspect='equal')
        ax.add_patch(Rectangle((-75.2, -75.2), 150.4, 150.4, fill=False, ec=INK2, lw=1, ls='--'))
        for rad in (50, 100):
            ax.add_patch(plt.Circle((0, 0), rad, fill=False, ec=INK2, lw=.55, alpha=.4))
        ax.plot(0, 0, marker='+', ms=6, mew=1.3, color=INK)
        ax.set_xlim(150, -150)
        ax.set_ylim(-150, 150)
        ax.set_xticks([-100, 0, 100])
        ax.set_yticks([-100, 0, 100])
        ax.tick_params(colors=INK2, labelsize=7.5)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title('%s\n%d of %d cells' % (name, int((h > 0).sum()), h.size),
                     fontsize=10, color=INK, pad=5)
    axes[1, 0].set_xlabel('y — left (m)', color=INK2, fontsize=9)
    axes[1, 0].set_ylabel('x — forward (m)', color=INK2, fontsize=9)
    cb = fig.colorbar(im, ax=axes, location='right', fraction=.018, pad=.012, shrink=.6)
    cb.set_label('points per frame in a 10x10 m cell', color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.suptitle("Bird's-eye point density, per capture platform — 10 x 10 m cells, +/-150 m",
                 fontsize=13, color=INK, y=.975)
    fig.savefig(out / 'platform_bev_density.png', dpi=150, facecolor=SURFACE, bbox_inches='tight')


def fig_points_per_box(D, out):
    centres = (R10[:-1] + R10[1:]) / 2
    kitti = med_by_bin(D['KITTI'].get('box_ref', D['KITTI']['box']))
    names = [n for n in ORDER if n not in NO_POINT_IN_BOX and n != 'KITTI']
    fig, axes = grid(2, 4, (15.5, 7.6))
    for ax, name in zip(axes.ravel(), names):
        style(ax)
        m = med_by_bin(D[name]['box'])
        ok = ~np.isnan(kitti)
        ax.plot(centres[ok], kitti[ok], color=INK2, lw=2, marker='s', ms=4, zorder=5)
        ok = ~np.isnan(m) & (m > 0)        # median 0 is unplottable on a log axis
        if ok.sum():
            ax.plot(centres[ok], m[ok], color=BLUE, lw=2, marker='o', ms=5)
        else:
            ax.text(.5, .45, 'no bin reaches 10 boxes with points:\nthe sensor is a forward cone\n'
                             'while the cuboids span 360 deg',
                    transform=ax.transAxes, ha='center', fontsize=9, color=RED)
        ax.set_yscale('log')
        ax.set_ylim(1, 5000)
        ax.set_title(name, fontsize=10, color=INK, pad=6)
        ax.set_xlabel('range (m)', color=INK2, fontsize=9)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    axes[0, 0].set_ylabel('points in a Car box (median)', color=INK, fontsize=9.5)
    axes[1, 0].set_ylabel('points in a Car box (median)', color=INK, fontsize=9.5)
    axes[0, 0].plot([], [], color=BLUE, lw=2, marker='o', ms=5, label='this platform')
    axes[0, 0].plot([], [], color=INK2, lw=2, marker='s', ms=4, label='KITTI (target)')
    axes[0, 0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc='lower left')
    fig.suptitle('Points landing inside a Car box, per capture platform', fontsize=13,
                 color=INK, y=.975)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(out / 'platform_points_per_box.png', dpi=150, facecolor=SURFACE)


def fig_dimensions(L, out):
    names = [n for n in ORDER if n not in NO_BOXES]
    fig, axes = grid(3, 3, (14, 9.4))
    for i, cls in enumerate(CLASSES):
        for j, (dim_name, col) in enumerate([('length', 0), ('width', 1), ('height', 2)]):
            ax = axes[i, j]
            style(ax, 'x')
            data = [L[n][0][cls][:, col] if len(L[n][0][cls]) else np.array([np.nan])
                    for n in names]
            bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=.6,
                            medianprops=dict(color=SURFACE, lw=1.5))
            for k, box in enumerate(bp['boxes']):
                box.set_facecolor(DARK if names[k] == 'KITTI' else BLUE)
                box.set_edgecolor('none')
            for el in ('whiskers', 'caps'):
                for it in bp[el]:
                    it.set_color(INK2)
                    it.set_lw(1)
            kd = L['KITTI'][0][cls]
            if len(kd):
                ax.axvline(np.median(kd[:, col]), color=INK2, lw=1.1, ls='--', zorder=0)
            ax.set_yticklabels(names if j == 0 else [''] * len(names), fontsize=8, color=INK)
            ax.invert_yaxis()
            if i == 0:
                ax.set_title(dim_name, fontsize=11.5, color=INK, pad=8)
            if j == 2:
                ax.text(1.03, .5, cls, transform=ax.transAxes, rotation=270, va='center',
                        fontsize=11.5, color=INK)
            if i == 2:
                ax.set_xlabel('metres', color=INK2, fontsize=9)
    fig.suptitle('Ground-truth box dimensions, per capture platform', fontsize=13, color=INK,
                 y=.975)
    fig.text(.5, .012, 'Dark bar = KITTI (the target); dashed line marks its median. Cyclist = '
                       'bicycle only (motorcycle maps to Misc and is excluded).',
             ha='center', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=[0, .035, 1, .94])
    fig.savefig(out / 'platform_box_dimensions.png', dpi=150, facecolor=SURFACE)


def fig_intensity(D, out):
    centres = (R10[:-1] + R10[1:]) / 2
    fig, axes = grid(2, 4, (15.5, 7.6))
    for ax, name in zip(axes.ravel(), ORDER):
        style(ax)
        vals = D[name]['inten']
        med = np.array([np.median(np.concatenate(vals[k])) if k in vals else np.nan
                        for k in range(len(centres))])
        q1 = np.array([np.percentile(np.concatenate(vals[k]), 25) if k in vals else np.nan
                       for k in range(len(centres))])
        q3 = np.array([np.percentile(np.concatenate(vals[k]), 75) if k in vals else np.nan
                       for k in range(len(centres))])
        degenerate = np.nanstd(med) == 0
        colour = RED if degenerate else BLUE
        ok = ~np.isnan(med)
        ax.fill_between(centres[ok], q1[ok], q3[ok], color=colour, alpha=.2, linewidth=0)
        ax.plot(centres[ok], med[ok], color=colour, lw=2, marker='o', ms=4)
        if degenerate:
            ax.text(.5, .55, 'single value\nzero information', transform=ax.transAxes,
                    ha='center', fontsize=10, color=RED, fontweight='bold')
        ax.set_title(name, fontsize=10, color=INK, pad=6)
        ax.set_xlabel('range (m)', color=INK2, fontsize=9)
    axes[0, 0].set_ylabel('intensity (native units)', color=INK2, fontsize=9)
    axes[1, 0].set_ylabel('intensity (native units)', color=INK2, fontsize=9)
    fig.suptitle('Intensity vs range, per capture platform — median and IQR, native units',
                 fontsize=13, color=INK, y=.975)
    fig.text(.5, .012, 'Y-axes are NOT comparable between panels: each is in its own loader\'s '
                       'units after that loader\'s transform. Only the shapes compare.',
             ha='center', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=[0, .035, 1, .94])
    fig.savefig(out / 'platform_intensity.png', dpi=150, facecolor=SURFACE)


def beam_spacing_break(hist, centres, expected_beams):
    """Locate the elevation where ring spacing changes most, and the medians either side.

    Multi-beam lidars concentrate beams near the horizon, where distant objects are, and spread
    them out below, where the beams hit nearby ground. The HDL-64E does this with two discrete
    32-laser blocks, so the change is a hard step; other sensors ramp.
    """
    from scipy.signal import find_peaks
    bin_deg = float(centres[1] - centres[0])
    best = None
    for prom in (0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25):
        for sep_deg in (0.12, 0.16, 0.2, 0.3, 0.4, 0.5):        # minimum ring separation
            dist = max(1, int(round(sep_deg / bin_deg)))
            pk, _ = find_peaks(hist, prominence=hist.max() * prom, distance=dist)
            if best is None or abs(len(pk) - expected_beams) < abs(best[0] - expected_beams):
                best = (len(pk), centres[pk])
    ang = np.sort(best[1])
    if len(ang) < 12:
        return None
    gaps = np.diff(ang)
    mid = (ang[:-1] + ang[1:]) / 2
    # A ratio-maximising scan drifts upward, because spacing tightens continuously towards the
    # horizon. Find the largest STEP in the gap profile instead - for a two-block sensor like the
    # HDL-64E that lands on the boundary between the laser blocks.
    k = 5
    best = None
    for i in range(k, len(gaps) - k):
        lo_med, hi_med = np.median(gaps[i - k:i]), np.median(gaps[i:i + k])
        if hi_med > 0 and (best is None or lo_med / hi_med > best[0]):
            best = (lo_med / hi_med, mid[i])
    if best is None:
        return None
    cut = best[1]
    lo, hi = ang[:-1] < cut, ang[:-1] >= cut
    if lo.sum() < 3 or hi.sum() < 3:
        return None
    return dict(cut=cut, below=np.median(gaps[lo]), above=np.median(gaps[hi]),
                ratio=np.median(gaps[lo]) / np.median(gaps[hi]), n=len(ang),
                n_below=int((ang < cut).sum()), n_above=int((ang >= cut).sum()))


def fig_beams(D, out):
    names = [n for n in ORDER if n in SENSOR_FRAME]
    centres = (EL[:-1] + EL[1:]) / 2
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 1.75 * len(names)), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(np.atleast_1d(axes), names):
        style(ax)
        h = D[name]['el']
        ax.fill_between(centres, 0, h, color=BLUE, lw=0)
        ax.set_xlim(-12, 2)
        ax.set_ylim(0, None)
        ax.set_ylabel(name.replace(' ', '\n', 1), color=INK, fontsize=8.5)
        span = centres[h > h.max() * 0.005]
        expected = 64 if ('KITTI' in name or '64' in name) else (40 if '40' in name else 32)
        brk = beam_spacing_break(h, centres, expected)
        note = 'full span %.1f to %.1f deg' % (span.min(), span.max())
        if brk is not None and brk['ratio'] > 1.2:
            ax.axvline(brk['cut'], color=RED, lw=1.2, ls='--', zorder=6)
            bbox = dict(facecolor=SURFACE, edgecolor='none', alpha=.85, pad=1.5)
            ax.text(brk['cut'] - .15, ax.get_ylim()[1] * .62, '%.2f deg apart' % brk['below'],
                    ha='right', fontsize=7.5, color=RED, bbox=bbox)
            ax.text(brk['cut'] + .15, ax.get_ylim()[1] * .62, '%.2f deg apart' % brk['above'],
                    ha='left', fontsize=7.5, color=RED, bbox=bbox)
            note += '   |   spacing breaks at %.1f deg (%.1fx coarser below)' % (brk['cut'],
                                                                                 brk['ratio'])
        ax.text(.995, .84, note, transform=ax.transAxes, ha='right', fontsize=8, color=INK2)
    np.atleast_1d(axes)[-1].set_xlabel('elevation angle from the sensor (degrees)',
                                       color=INK2, fontsize=10)
    fig.suptitle('Beam structure, per capture platform — elevation angle, 0.1 deg bins',
                 fontsize=13, color=INK, y=.985)
    fig.text(.5, .01, 'Only platforms whose point frame IS the sensor frame. PandaSet and Waymo use '
                      'ground-origin vehicle frames and need the sensor extrinsic.\nRed line marks where '
                      'ring spacing changes most - beams are concentrated near the horizon and spread out '
                      'below, where they hit nearby ground.',
             ha='center', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=[0, .045, 1, .955])
    fig.savefig(out / 'platform_beams.png', dpi=150, facecolor=SURFACE)


def fig_xyz(D, out):
    ac = (AX[:-1] + AX[1:]) / 2
    zc = (ZB[:-1] + ZB[1:]) / 2
    fig, axes = grid(3, 8, (19, 7.6))
    for j, name in enumerate(ORDER):
        for i, (key, centres, label) in enumerate([('x', ac, 'x — forward (m)'),
                                                   ('y', ac, 'y — left (m)'),
                                                   ('z', zc, 'z — up (m)')]):
            ax = axes[i, j]
            style(ax)
            ax.fill_between(centres, 0, D[name][key], color=BLUE, lw=0, alpha=.85)
            if name != 'KITTI':
                ax.plot(centres, D['KITTI'][key], color=INK2, lw=1.2, zorder=4)
            ax.set_yscale('log')
            ax.set_ylim(1, 3e5)
            if key == 'z':
                ax.set_xlim(-4, 4)
                ax.axvline(-2, color=INK2, ls='--', lw=.9)
                ax.axvline(4, color=INK2, ls='--', lw=.9)
            else:
                ax.set_xlim(-80, 80)
            if j > 0:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(name.replace(' ', '\n', 1), fontsize=8.5, color=INK, pad=6)
            if j == 0:
                ax.set_ylabel(label, color=INK, fontsize=9)
    fig.suptitle('Point coordinate distributions, per capture platform — blue = platform, '
                 'grey = KITTI', fontsize=13, color=INK, y=.975)
    fig.text(.5, .012, 'Masked to |x|,|y| < 75.2 m; log y. Dashed lines in the z row are the '
                       'POINT_CLOUD_RANGE limits; the z mode is the ground plane.',
             ha='center', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=[0, .035, 1, .94])
    fig.savefig(out / 'platform_point_xyz.png', dpi=150, facecolor=SURFACE)


def fig_scene(L, out):
    names = [n for n in ORDER if n not in NO_BOXES]
    fig, axes = grid(3, 2, (13.5, 9.4))
    for i, cls in enumerate(CLASSES):
        for j, (kind, title) in enumerate([('z', 'box centre height  z  (m)'),
                                           ('n', 'objects of this class per frame')]):
            ax = axes[i, j]
            style(ax, 'x')
            if kind == 'z':
                data = [L[n][0][cls][:, 3] if len(L[n][0][cls]) else np.array([np.nan])
                        for n in names]
            else:
                data = [L[n][1][cls] for n in names]
            bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=.6,
                            medianprops=dict(color=SURFACE, lw=1.5))
            for k, box in enumerate(bp['boxes']):
                box.set_facecolor(DARK if names[k] == 'KITTI' else BLUE)
                box.set_edgecolor('none')
            for el in ('whiskers', 'caps'):
                for it in bp[el]:
                    it.set_color(INK2)
                    it.set_lw(1)
            ax.set_yticklabels(names if j == 0 else [''] * len(names), fontsize=8, color=INK)
            ax.invert_yaxis()
            if kind == 'n':
                ax.set_xscale('symlog', linthresh=1)
                ax.text(1.02, .5, cls, transform=ax.transAxes, rotation=270, va='center',
                        fontsize=11.5, color=INK)
            if i == 0:
                ax.set_title(title, fontsize=11.5, color=INK, pad=8)
    fig.suptitle('Scene-level gaps, per capture platform', fontsize=13, color=INK, y=.975)
    fig.text(.5, .012, 'Dark bar = KITTI. Object counts are symlog; outliers hidden. Cyclist = '
                       'bicycle only (motorcycle maps to Misc and is excluded).',
             ha='center', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=[0, .035, 1, .94])
    fig.savefig(out / 'platform_scene.png', dpi=150, facecolor=SURFACE)


def fig_label_range(L, out):
    edges = np.arange(0, 160, 10)
    centres = (edges[:-1] + edges[1:]) / 2
    names = [n for n in ORDER if n not in NO_BOXES]
    fig, axes = grid(2, 4, (15.5, 8))
    for ax, name in zip(axes.ravel(), names):
        style(ax, 'y')
        dims, counts, pos, nf = L[name]
        for k, (cls, colour, off) in enumerate([('Car', BLUE, -3.1), ('Pedestrian', ORANGE, 0),
                                                ('Cyclist', DARK, 3.1)]):
            p = pos[cls]
            if not len(p):
                continue
            h = np.histogram(np.linalg.norm(p, axis=1), bins=edges)[0].astype(float)
            ax.bar(centres + off, 100 * h / h.sum(), width=3.0, color=colour, linewidth=0,
                   label=cls if name == names[0] else None)
        ax.axvline(75.2, color=INK2, lw=1.1, ls='--')
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 62)
        ax.set_title('%s\n%.1f car + %.1f ped per frame'
                     % (name, len(pos['Car']) / nf, len(pos['Pedestrian']) / nf),
                     fontsize=9.5, color=INK, pad=6)
        ax.set_xlabel('range (m)', color=INK2, fontsize=8.5)
    axes[0, 0].set_ylabel("% of that platform's boxes", color=INK2, fontsize=9)
    axes[1, 0].set_ylabel("% of that platform's boxes", color=INK2, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    fig.suptitle('Ground-truth box centres by range, per capture platform — 10 m bins',
                 fontsize=13, color=INK, y=.975)
    fig.text(.5, .012, "Each panel normalised to its own platform, so shapes compare despite large "
                       "differences in boxes per frame. Dashed line is the POINT_CLOUD_RANGE edge.\n"
                       "Cyclist = bicycle only, following pcdet/utils/ontology_mapping.py; motorcycle "
                       "maps to Misc and is deliberately excluded (nuScenes has 8,846 of them against "
                       "8,185 bicycles).",
             ha='center', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=[0, .04, 1, .94])
    fig.savefig(out / 'platform_label_range.png', dpi=150, facecolor=SURFACE)


def fig_label_bev(L, out):
    names = [n for n in ORDER if n not in NO_BOXES]
    per = {}
    for name in names:
        dims, counts, pos, nf = L[name]
        pts = np.concatenate([pos['Car'], pos['Pedestrian']]) if len(pos['Car']) else pos['Pedestrian']
        per[name] = np.histogram2d(pts[:, 0], pts[:, 1], bins=[XY, XY])[0] / nf
    vmax = max(h.max() for h in per.values())
    fig, axes = grid(2, 4, (15.5, 8.4))
    for ax, name in zip(axes.ravel(), names):
        h = per[name]
        ax.set_facecolor(SURFACE)
        im = ax.imshow(np.ma.masked_where(h <= 0, h)[::-1, ::-1], extent=[150, -150, -150, 150],
                       norm=LogNorm(vmin=1e-3, vmax=vmax), cmap=CMAP, interpolation='nearest',
                       aspect='equal')
        ax.add_patch(Rectangle((-75.2, -75.2), 150.4, 150.4, fill=False, ec=INK2, lw=1, ls='--'))
        for rad in (50, 100):
            ax.add_patch(plt.Circle((0, 0), rad, fill=False, ec=INK2, lw=.55, alpha=.4))
        ax.plot(0, 0, marker='+', ms=6, mew=1.3, color=INK)
        ax.set_xlim(150, -150)
        ax.set_ylim(-150, 150)
        ax.set_xticks([-100, 0, 100])
        ax.set_yticks([-100, 0, 100])
        ax.tick_params(colors=INK2, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title('%s\n%d of %d cells' % (name, int((h > 0).sum()), h.size),
                     fontsize=9.5, color=INK, pad=5)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    axes[1, 0].set_xlabel('y — left (m)', color=INK2, fontsize=9)
    axes[1, 0].set_ylabel('x — forward (m)', color=INK2, fontsize=9)
    cb = fig.colorbar(im, ax=axes, location='right', fraction=.018, pad=.012, shrink=.6)
    cb.set_label('Car + Pedestrian boxes per frame in a 10x10 m cell', color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.suptitle('Ground-truth box centres in the X-Y plane, per capture platform — '
                 '10 x 10 m cells', fontsize=13, color=INK, y=.975)
    fig.savefig(out / 'platform_label_bev.png', dpi=150, facecolor=SURFACE, bbox_inches='tight')


FIGURES = dict(label_range=fig_label_range, label_bev=fig_label_bev,range=fig_range, bev=fig_bev, points_per_box=fig_points_per_box,
               dimensions=fig_dimensions, intensity=fig_intensity, beams=fig_beams,
               xyz=fig_xyz, scene=fig_scene)
NEEDS_LABELS = {'dimensions', 'scene', 'label_range', 'label_bev'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='../../experiments_md')
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--only', nargs='*', default=None, choices=sorted(FIGURES))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wanted = args.only or sorted(FIGURES)
    P = build_platforms()
    D, L = {}, {}
    if set(wanted) - NEEDS_LABELS:
        for name in ORDER:
            print('  points: %s' % name, flush=True)
            D[name] = collect(P[name], args.frames)
        D['KITTI']['box_ref'] = kitti_box_reference(P['KITTI'])
    if set(wanted) & NEEDS_LABELS:
        for name in ORDER:
            if name in NO_BOXES:
                continue
            print('  labels: %s' % name, flush=True)
            L[name] = collect_labels(P[name], args.frames)
    for key in wanted:
        FIGURES[key](L if key in NEEDS_LABELS else D, out)
        print('wrote platform_%s' % key, flush=True)


if __name__ == '__main__':
    main()
