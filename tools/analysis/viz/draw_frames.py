"""Draw one run's figure from its dump (experiments_md/20261003_02 §2). CPU, from tools/:

    python analysis/viz/draw_frames.py --rows 25835,25781 [--dump /storage/viz/dump] [--out <experiments_md>/viz]

Clouds are drawn as BEV point DENSITY (points per 0.4 m cell) on one fixed log scale shared by every
panel of every figure, so equal shading means equal density - across source and target, and across
runs. Full detection grid, +-75.2 m. Ground truth green, predictions (score >= 0.3) red, pseudo-label
rule blue (positives solid, ignore band dashed). Car thick, Pedestrian thin; Cyclist is not drawn.
Every BEV is rotated so the vehicle faces UP (forward yaw per frame, from viz_frames.json).
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

GRID, CELL, VMAX = 75.2, 0.4, 200
EDGES = np.arange(-GRID, GRID + CELL / 2, CELL)
DRAW_SCORE = 0.3
NORM = LogNorm(vmin=1, vmax=VMAX)
# one point per cell is already a clearly visible mid-grey; empty cells stay white
CMAP = LinearSegmentedColormap.from_list('density', ['#b8b8b8', '#000000'])
COL = dict(gt='#1a9850', pred='#d73027', ps='#2c7bb6')
RING_EDGES = np.arange(0, 80, 5.0)
BOX_RINGS = np.arange(0, 80, 10.0)
SRC_COL = ['#e66101', '#5e3c99', '#1b9e77']


def corners(b):
    x, y, dx, dy, h = b[0], b[1], b[3], b[4], b[6]
    c, s = np.cos(h), np.sin(h)
    pts = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]]) / 2
    return np.stack([x + pts[:, 0] * c - pts[:, 1] * s, y + pts[:, 0] * s + pts[:, 1] * c], 1)


def boxes(ax, b, names, color, ls='-', mask=None):
    for i in range(len(b)):
        if names[i] not in ('Car', 'Pedestrian'):
            continue
        k = corners(b[i])
        ax.plot(k[:, 0], k[:, 1], color=color, lw=1.1 if names[i] == 'Car' else 0.6,
                ls=ls if mask is None or mask[i] else '--')


def rot(xy, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([xy[:, 0] * c - xy[:, 1] * s, xy[:, 0] * s + xy[:, 1] * c], 1)


def turn(b, theta):
    """Boxes rotated about the sensor by theta (radians)."""
    if b is None or not len(b):
        return b
    b = np.array(b, dtype=np.float32, copy=True)
    b[:, :2] = rot(b[:, :2], theta)
    b[:, 6] += theta
    return b


def bev(ax, pts, title, gt=None, pred=None, ps=None, wedge=None, yaw=0.0):
    # forward up: rotate the frame so the vehicle's heading (yaw, lidar frame) points to +y
    th = np.radians(90.0 - yaw)
    xy = rot(pts[:, :2], th)
    gt = None if gt is None else (turn(gt[0], th), gt[1])
    pred = None if pred is None else (turn(pred[0], th),) + tuple(pred[1:])
    ps = None if ps is None else (turn(ps[0], th),) + tuple(ps[1:])
    h = np.histogram2d(xy[:, 0], xy[:, 1], bins=[EDGES, EDGES])[0]
    h[h == 0] = np.nan
    ax.imshow(h.T, origin='lower', extent=[-GRID, GRID, -GRID, GRID], cmap=CMAP, norm=NORM,
              interpolation='nearest')
    for r in (25, 50, 75):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, lw=0.3, color='#999999', ls=':'))
    if wedge is not None:
        for a in (90 - wedge / 2, 90 + wedge / 2):
            ax.plot([0, GRID * np.cos(np.radians(a))], [0, GRID * np.sin(np.radians(a))], lw=0.4, color='#4393c3')
    if gt is not None:
        boxes(ax, gt[0], gt[1], COL['gt'])
    if pred is not None:
        b, s, n = pred
        k = s >= DRAW_SCORE
        boxes(ax, b[k], n[k], COL['pred'])
    if ps is not None:
        b, s, n, pos = ps
        boxes(ax, b, n, COL['ps'], mask=pos)
    ax.set_xlim(-GRID, GRID); ax.set_ylim(-GRID, GRID); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def image(ax, path, title):
    ax.axis('off')
    if path:
        try:
            ax.imshow(plt.imread(path))
        except Exception as e:  # noqa: BLE001
            ax.text(0.5, 0.5, 'image unreadable: %s' % e, ha='center', fontsize=7, transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'no front image on disk', ha='center', va='center', fontsize=8, transform=ax.transAxes)
    ax.set_title(title, fontsize=8)


def stat_text(ax, lines):
    ax.axis('off')
    ax.text(0.02, 0.98, '\n'.join(lines), va='top', ha='left', fontsize=7, family='monospace', transform=ax.transAxes)


def car_pts(c):
    """median points per Car box per 10 m ring, and box counts."""
    cb = c['car_boxes']
    med, n = [], []
    for lo, hi in zip(BOX_RINGS[:-1], BOX_RINGS[1:]):
        v = cb[(cb[:, 0] >= lo) & (cb[:, 0] < hi), 1] if len(cb) else np.zeros(0)
        med.append(np.median(v) if len(v) >= 3 else np.nan); n.append(len(v))
    return np.array(med), np.array(n)


YAW = {}


def load_yaw():
    import json
    for v in json.load(open(Path(__file__).resolve().parent / 'viz_frames.json')).values():
        for f in v['frames']:
            YAW[f['fid']] = f.get('yaw', 0.0)


def short(fid):
    return Path(str(fid)).name[-42:]


def draw(d, out):
    row = d['row']
    S, T = d['sources'], d['target']
    nf = 3
    fig = plt.figure(figsize=(6.2 * nf, 18.5))
    gs = GridSpec(6, 2 * nf, figure=fig, height_ratios=[1.45, 3.1, 1.45, 3.1, 0.15, 2.3], hspace=0.13, wspace=0.04,
                  top=0.925, bottom=0.03, left=0.04, right=0.99)
    src_frames = [(s, f) for s in S for f in s['frames']][:nf]
    for j, (s, f) in enumerate(src_frames):
        image(fig.add_subplot(gs[0, 2 * j:2 * j + 2]), f['image'], 'SOURCE %s (%s) %s' % (s['kind'], s['name'], short(f['fid'])))
        y = YAW[f['fid']]
        bev(fig.add_subplot(gs[1, 2 * j]), f['raw'], 'raw sweep: %s pts' % format(len(f['raw']), ','), gt=f['raw_gt'], yaw=y)
        bev(fig.add_subplot(gs[1, 2 * j + 1]), f['points'], 'model input: %s pts' % format(len(f['points']), ','),
            gt=f['gt'], pred=f['pred'], yaw=y)
    wedge = 90 if T['kind'] == 'kitti' else None
    for j, f in enumerate(T['frames'][:nf]):
        image(fig.add_subplot(gs[2, 2 * j:2 * j + 2]), f['image'], 'TARGET %s %s' % (T['kind'], short(f['fid'])))
        bev(fig.add_subplot(gs[3, 2 * j]), f['points'], 'model input: %s pts' % format(len(f['points']), ','),
            gt=f['gt'], pred=f['pred'], ps=f.get('pseudo'), wedge=wedge, yaw=YAW[f['fid']])
        lines = ['target frame %d' % (j + 1), '']
        gb, gn = f['gt']
        lines += ['labels   Car %d  Ped %d' % ((gn == 'Car').sum(), (gn == 'Pedestrian').sum())]
        b, s_, n = f['pred']
        k = s_ >= DRAW_SCORE
        lines += ['pred>=%.1f Car %d  Ped %d' % (DRAW_SCORE, (n[k] == 'Car').sum(), (n[k] == 'Pedestrian').sum())]
        if f.get('pseudo') is not None:
            pb, ps_, pn, pos = f['pseudo']
            lines += ['pseudo   Car %d (+%d ign)  Ped %d (+%d ign)' % (
                ((pn == 'Car') & pos).sum(), ((pn == 'Car') & ~pos).sum(),
                ((pn == 'Pedestrian') & pos).sum(), ((pn == 'Pedestrian') & ~pos).sum())]
        if f.get('check'):
            c = f['check']
            lines += ['', 'vs scored result.pkl: %d / %d boxes' % (c['n_ours'], c['n_ref']),
                      '  max centre offset %s' % ('%.4f m' % c['max_offset'] if c['max_offset'] is not None else '-')]
        stat_text(fig.add_subplot(gs[3, 2 * j + 1]), lines)

    # closeness strip
    ax1 = fig.add_subplot(gs[5, 0:2]); ax2 = fig.add_subplot(gs[5, 2:4]); ax3 = fig.add_subplot(gs[5, 4:6])
    mid = (RING_EDGES[:-1] + RING_EDGES[1:]) / 2
    tr = T['close']['rings'].mean(0)
    ax1.plot(mid, tr, color='k', lw=2, label='target model input')
    for i, s in enumerate(S):
        c = SRC_COL[i % 3]
        ax1.plot(mid, s['close_raw']['rings'].mean(0), color=c, ls='--', lw=1, label='%s raw' % s['name'])
        ax1.plot(mid, s['close']['rings'].mean(0), color=c, lw=1.6, label='%s model input' % s['name'])
        with np.errstate(divide='ignore', invalid='ignore'):
            ax2.plot(mid, s['close_raw']['rings'].mean(0) / tr, color=c, ls='--', lw=1)
            ax2.plot(mid, s['close']['rings'].mean(0) / tr, color=c, lw=1.6)
    ax1.set_yscale('log'); ax1.set_xlabel('range ring (m)', fontsize=8); ax1.set_ylabel('points / frame / 5 m ring', fontsize=8)
    ax1.legend(fontsize=6.5); ax1.set_title('(a) density by range, %d frames per cloud' % T['close']['n_frames'], fontsize=8)
    ax2.axhline(1, color='k', lw=1); ax2.set_yscale('log'); ax2.set_ylim(0.05, 50)
    ax2.set_xlabel('range ring (m)', fontsize=8); ax2.set_ylabel('source / target', fontsize=8)
    ax2.set_title('(b) how close: source over target per ring (1 = matched)', fontsize=8)
    bm = (BOX_RINGS[:-1] + BOX_RINGS[1:]) / 2
    tm, tn = car_pts(T['close'])
    ax3.plot(bm, tm, color='k', lw=2, marker='o', ms=3, label='target')
    for x, v, n in zip(bm, tm, tn):
        if n:
            ax3.annotate(str(n), (x, v if np.isfinite(v) else 1), fontsize=5.5, color='k', xytext=(2, 2), textcoords='offset points')
    for i, s in enumerate(S):
        c = SRC_COL[i % 3]
        m, _ = car_pts(s['close_raw']); ax3.plot(bm, m, color=c, ls='--', lw=1, marker='.', ms=3)
        m, n = car_pts(s['close']); ax3.plot(bm, m, color=c, lw=1.6, marker='o', ms=3, label='%s input' % s['name'])
    ax3.set_yscale('log'); ax3.set_xlabel('Car box centre ring (m)', fontsize=8); ax3.set_ylabel('median points per Car box', fontsize=8)
    ax3.set_title('(c) object density by range (target box counts printed; < 3 boxes not drawn)', fontsize=8)
    ax3.legend(fontsize=6.5)
    for a in (ax1, ax2, ax3):
        a.tick_params(labelsize=7); a.grid(alpha=0.3, lw=0.4)

    # header
    desc = []
    for s in S:
        cal = s.get('calib')
        if cal and 'error' in cal:
            c = 'correction: %s' % cal['error']
        elif cal:
            c = 'density correction recomputed: rate %.2f..%.2f, below 1 in %d bins (src %.0f / tgt %.0f pts/frame)' % (
                cal['rate_min'], cal['rate_max'], cal['bins_below_1'], cal['src_pts'], cal['tgt_pts'])
        else:
            c = 'no density correction'
        desc.append('%s: %d sweep(s)%s; %s' % (s['name'], s['max_sweeps'], ' + motion comp.' if s['motion_comp'] else '', c))
    if d.get('pseudo_rule'):
        r = d['pseudo_rule']
        desc.append('pseudo-label rule on these val frames: teacher %s, SCORE_THRESH %s, NEG_THRESH %s' % (
            Path(r['teacher']).parts[-4] if 'wandb' in r['teacher'] else r['teacher'], r['score'], r['neg']))
    fig.suptitle('job %s  ·  %s  ·  %s\n%s\nconfig: %s\ndensity: points per %.1f m cell, log 1..%d, same scale in every panel'
                 ' · vehicle faces up · green labels, red predictions >= %.1f, blue pseudo-label rule (dashed = ignore band)' % (
                     row['job'], row['row'], row.get('ap', ''), '\n'.join(desc), d['cfg_source'], CELL, VMAX, DRAW_SCORE),
                 fontsize=10, y=0.985, x=0.01, ha='left')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, pil_kwargs=dict(quality=82))
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True)
    ap.add_argument('--dump', default='/home/koyama/data/viz/dump')
    ap.add_argument('--out', default='/home/koyama/code/experiments_md/viz')
    a = ap.parse_args()
    load_yaw()
    for job in a.rows.split(','):
        d = pickle.load(open(Path(a.dump) / ('%s.pkl' % job), 'rb'))
        o = Path(a.out) / ('%s.jpg' % job)
        draw(d, o)
        print('wrote', o, '%.0f KB' % (o.stat().st_size / 1024))
