"""Figures of experiments_md 20261003_05 §5 item 38 from item38_kitti_rings.py's JSON (CPU, matplotlib).

    python item38_kitti_rings_figures.py RINGS_AP.json DENSITY.json OUT_DIR

Colours are the dataviz reference palette's categorical slots in their fixed order, one slot per entity across all
figures (the oracle and the full KITTI cloud share slot 1: the oracle trains on it). Light surface only.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
COLOR = {'O': SLOT[0], 'full KITTI': SLOT[0], 'R1': SLOT[1], 'R1 (32 lines)': SLOT[1], 'R2': SLOT[2],
         'R3': SLOT[3], 'R3 (density)': SLOT[3], 'P_K': SLOT[4], 'P_K (lattice)': SLOT[4], 'L_K': SLOT[5],
         'L_K (P_K count)': SLOT[5], 'control': SLOT[6], 'teacher': SLOT[7]}
NAME = {'O': 'O, oracle', 'R1': 'R1, 32 lines', 'R2': 'R2, random half', 'R3': 'R3, density', 'P_K': 'P_K, lattice',
        'L_K': 'L_K, P_K count', 'R4': 'R4, azimuth', 'R5': 'R5, pitch', 'R6': 'R6, label size',
        'R7': 'R7, label range', 'R8': 'R8, foreground 1/2', 'R9': 'R9, background 1/2', 'R11': 'R11, reflection',
        'R11r': 'R11r, reflection random', 'control': 'source-only control (nuScenes, epoch 20)',
        'teacher': 'accumulation teacher (nuScenes, epoch 20)'}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False, 'savefig.dpi': 130})


def by_row(rows, group):
    out = defaultdict(list)
    for r in rows:
        if r['group'] == group and r['reproduced']:
            out[r['row']].append(r)
    return out


def curve(runs, key='bev'):
    return np.array([[ring[key] if ring[key] is not None else np.nan for ring in r['rings']] for r in runs])


def draw(ax, x, runs, color, label, key='bev', z=3):
    """Seeds as thin lines with markers, the mean bold."""
    c = curve(runs, key)
    for s, row in zip(c, runs):
        ax.plot(x, s, color=color, lw=0.9, alpha=0.45, marker='o', ms=3.5, zorder=z)
    ax.plot(x, np.nanmean(c, 0), color=color, lw=2.2, marker='o', ms=6, mec=SURFACE, mew=1.2, zorder=z + 1,
            label=label)
    return np.nanmean(c, 0)


def seeds(runs):
    return '/'.join(str(r['seed']) for r in runs)


def finish(ax, labels, ylabel):
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_xlabel('ring of box-centre range (m)'); ax.set_ylabel(ylabel)
    ax.tick_params(length=0)


def fig_a(ap, out):
    labels = ['%g-%g' % (lo, hi) for lo, hi in ap['rings']]
    x = np.arange(len(labels))
    swap = by_row(ap['rows'], 'swap')
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for row in ['O', 'R1', 'R2', 'R3', 'P_K', 'L_K']:
        if row in swap:
            draw(ax, x, swap[row], COLOR[row], f'{NAME[row]} (seeds {seeds(swap[row])})')
    ax.set_ylim(0, 100); ax.set_xlim(-0.3, len(labels) - 0.7)
    finish(ax, labels, 'Car BEV AP_R40, moderate, IoU 0.7')
    ax.set_title('Per-range AP of the KITTI oracle and its line / density / pattern swaps', loc='left', fontsize=10.5)
    fig.text(0.01, 0.005, 'KITTI val, epoch 152, official evaluator per ring (out-of-ring GT ignored, out-of-ring '
             'detections dropped). Thin: one seed; bold: mean.', fontsize=7.5, color=MUTED)
    ax.legend(loc='lower left', fontsize=7.5, ncol=2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / 'fa1_ap_by_range_lines_density_pattern.png'); plt.close(fig)

    rest = [r for r in ['R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R11', 'R11r'] if r in swap]
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.8), sharex=True, sharey=True)
    for ax, row in zip(axes.flat, rest):
        draw(ax, x, swap['O'], AXIS, 'O', z=2)
        draw(ax, x, swap[row], SLOT[0] if row != 'O' else AXIS, row)
        ax.set_title(f'{NAME[row]} (seeds {seeds(swap[row])})', loc='left', fontsize=8.5)
        ax.set_ylim(0, 100); ax.set_xticks(x); ax.set_xticklabels(['0', '10', '20', '30', '40+'])
        ax.tick_params(length=0)
    for ax in axes.flat[len(rest):]:
        ax.set_visible(False)
    fig.supxlabel('ring of box-centre range (m; 40+ = 40-70.4)', fontsize=8.5, color=INK2)
    fig.supylabel('Car BEV AP_R40, moderate, IoU 0.7', fontsize=8.5, color=INK2)
    fig.suptitle('Per-range AP of the other KITTI factor swaps (blue) against the oracle O (gray)', x=0.01, ha='left',
                 fontsize=10.5)
    fig.text(0.01, 0.925, 'KITTI val, epoch 152, official evaluator per ring. Thin: one seed; bold: mean. Rows whose '
             'pooled score is not reproduced within 0.02 AP are left out.', fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(out / 'fa2_ap_by_range_other_swaps.png'); plt.close(fig)


def fig_b(den, out):
    labels = ['%g-%g' % (lo, hi) for lo, hi in den['rings']]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for name, rows in den['table'].items():
        y = [r['median'] for r in rows]
        ax.plot(x, y, color=COLOR[name], lw=2.2, marker='o', ms=6, mec=SURFACE, mew=1.2, label=name)
        ax.annotate(name.split(' ')[0], (x[-1], y[-1]), xytext=(6, 0), textcoords='offset points', va='center',
                    color=INK2, fontsize=8.5)
    ax.set_yscale('log'); ax.set_xlim(-0.3, len(labels) - 0.5)
    finish(ax, labels, 'median points per Car box (log)')
    n = ', '.join(str(r['n_boxes']) for r in next(iter(den['table'].values())))
    ax.set_title('Points per Car box by range: full KITTI and each training cut', loc='left', fontsize=10.5)
    fig.text(0.01, 0.005, f'KITTI val, moderate-valid Car GT (boxes per ring: {n}).\nEach cut is the training '
             'row\'s own frame-seeded function applied to the val cloud.', fontsize=7.5, color=MUTED)
    ax.legend(loc='lower left', fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out / 'fb_points_per_box_by_range.png'); plt.close(fig)


def fig_c(ap, out):
    labels = ['%g-%g' % (lo, hi) for lo, hi in ap['rings']]
    x = np.arange(len(labels))
    swap, tr = by_row(ap['rows'], 'swap'), by_row(ap['rows'], 'transfer')
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    draw(ax, x, swap['O'], AXIS, f'KITTI oracle O (seeds {seeds(swap["O"])}, epoch 152)', z=2)
    for row in ['control', 'teacher']:
        if row in tr:
            m = draw(ax, x, tr[row], COLOR[row], f'{NAME[row]}, seeds {seeds(tr[row])}')
    ax.set_ylim(0, 100); ax.set_xlim(-0.3, len(labels) - 0.5)
    finish(ax, labels, 'Car BEV AP_R40, moderate, IoU 0.7')
    ax.set_title('Per-range AP on KITTI of the nuScenes-trained control and teacher', loc='left', fontsize=10.5)
    fig.text(0.01, 0.005, 'KITTI val (camera FOV), official evaluator per ring. Thin: one seed; bold: mean.',
             fontsize=7.5, color=MUTED)
    ax.legend(loc='lower left', fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / 'fc_ap_by_range_control_teacher.png'); plt.close(fig)


def spread(runs, key='bev'):
    c = curve(runs, key)
    return np.nanmax(c, 0) - np.nanmin(c, 0)


def fig_d(ap, out):
    """Seed spread per ring, each model on full KITTI val and on the val cut its own way (or another way).

    Colour follows the CLOUD the model is scored on, as in fig_b: full KITTI, R1's 32 lines, R3's density.
    """
    labels = ['0', '10', '20', '30', '40+']
    x = np.arange(len(labels))
    rows = defaultdict(list)
    for r in ap['rows']:
        if r['reproduced']:
            rows[(r['group'], r['row'])].append(r)
    full, line, dens = ('full KITTI val', SLOT[0]), ("R1's 32-line val", SLOT[1]), ("R3's density val", SLOT[3])
    panels = [
        ('R1, 32 lines', [(('swap', 'R1'), full), (('own-cut 33d', 'R1 on 32-line val'), line)]),
        ('R3, density', [(('swap', 'R3'), full), (('own-cut 33e', 'R3 on own-density val'), dens)]),
        ('O, oracle (full KITTI)', [(('swap', 'O'), full), (('own-cut 33d', 'O on 32-line val'), line),
                                    (('own-cut 33e', 'O on own-density val'), dens)]),
        ('nuScenes control, epoch 20', [(('transfer', 'control'), full)]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.6), sharey=True)
    seen = {}
    for ax, (title, series) in zip(axes, panels):
        n = []
        for key, (name, color) in series:
            if key not in rows:
                continue
            runs = rows[key]
            n.append(len(runs))
            h, = ax.plot(x, spread(runs), color=color, lw=2.2, marker='o', ms=6, mec=SURFACE, mew=1.2, label=name)
            seen.setdefault(name, h)
        ax.axhline(3, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.set_title(f"{title} ({'/'.join(str(k) for k in sorted(set(n)))} seeds)", loc='left', fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.tick_params(length=0)
    axes[0].set_ylabel('seed spread, max - min (BEV AP)')
    axes[0].annotate('3 AP', (x[-1], 3), xytext=(0, 3), textcoords='offset points', ha='right', va='bottom',
                     color=MUTED, fontsize=7.5)
    fig.supxlabel('ring of box-centre range (m; 40+ = 40-70.4)', fontsize=8.5, color=INK2, y=0.07)
    fig.legend(list(seen.values()), list(seen.keys()), loc='upper right', ncol=3, fontsize=7.5,
               title='scored on', title_fontsize=7.5)
    fig.suptitle('Seed spread per range ring: on full KITTI val and on the val cut each way', x=0.01, ha='left',
                 fontsize=10.5)
    fig.text(0.01, 0.005, 'Car BEV AP_R40, moderate, IoU 0.7, official evaluator per ring. Spread over every '
             'seed that exists (O: 2; R1, R3, control: 3). The 0-10 m ring holds 501 GT.', fontsize=7.5,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.1, 1, 0.88)); fig.savefig(out / 'fd_seed_spread_by_range.png'); plt.close(fig)


def main():
    ap, den, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    fig_a(ap, out); fig_b(den, out); fig_c(ap, out); fig_d(ap, out)


if __name__ == '__main__':
    main()
