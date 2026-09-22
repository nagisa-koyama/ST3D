"""What the foreground-aware density correction changes, at the recommended accumulation depth.

The plain correction gives every point in a radial bin the same keep-probability, which scales the
points on objects and the points on everything else by the same factor and so cannot change a bin's
foreground SHARE. Matching the global radial profile therefore leaves source objects at
sigma_src/sigma_tgt of the target's object density. The foreground-aware variant corrects the
inside-box and outside-box channels separately.

These figures put the two side by side at the UDA-legal depth from
experiments_md/20260922_07, against KITTI.

HONESTY ABOUT THE TARGET BOXES: in a real run the target's foreground channel comes from the
teacher's PSEUDO-LABELS. Here it is built from KITTI's real annotations, which is the recall = 1
case - an upper bound on what the mechanism delivers, not what a run gets. Figure 4 exists for
exactly that reason: it degrades the target boxes to a given recall and shows what the bias costs.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from domain_gap_figures import (SURFACE, INK, INK2, GRID, BLUE, ORANGE, RED, DARK, style, grid)
from domain_gap_analysis import (build_platforms, mask_range, points_in_boxes, quat_to_rot,
                                 MAX_DIST, DATA)
import collections
import json
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# domain_gap_analysis.points_in_boxes returns per-box COUNTS; this returns a per-POINT mask.
from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402

point_mask = DataProcessor.points_in_any_box

BINS = 50
EDGES = np.linspace(0, MAX_DIST, BINS + 1)
CENTRES = (EDGES[:-1] + EDGES[1:]) / 2
R10 = np.arange(0, 80, 10)
TARGET = 'KITTI'
# source -> recommended depth, global criterion, from 20260922_07
PAIRS = [('nuScenes n015 Singapore', 9), ('Lyft 40-beam', 4), ('PandaSet Pandar64', 3)]
RECALLS = [1.0, 0.8, 0.6, 0.4, 0.2]


def attach_pandaset_sweeps(P):
    """PandaSet has no sweep mechanism in the loader; chain sequence frames by ego pose.

    Its points are stored in WORLD coordinates with a per-frame pose, so the anchor's pose alone
    maps every frame of the sequence into the anchor frame - no per-frame composition needed.
    """
    infos = pickle.load(open(DATA / 'pandaset/pandaset_infos_train.pkl', 'rb'))
    by_seq = collections.defaultdict(dict)
    for i in infos:
        by_seq[i['sequence']][i['frame_idx']] = i
    fix = lambda q: str(q).replace('/root/ST3D/data/pandaset', str(DATA / 'pandaset'))
    cache = {}

    def pose(seq, idx):
        if seq not in cache:
            cache[seq] = json.load(open(DATA / f'pandaset/dataset/{seq}/lidar/poses.json'))
        q = cache[seq][idx]
        return (quat_to_rot(*[q['heading'][k] for k in 'wxyz']),
                np.array([q['position'][k] for k in 'xyz']))

    def factory(device):
        def sweeps(info, n):
            import pandas as pd
            seq, a = info['sequence'], info['frame_idx']
            R, t = pose(seq, a)
            out = []
            for k in range(a, max(-1, a - n), -1):
                if k not in by_seq[seq]:
                    break
                df = pd.read_pickle(fix(by_seq[seq][k]['lidar_path']))
                if device != -1:
                    df = df[df.d == device]
                ego = (R.T @ (df[['x', 'y', 'z']].to_numpy() - t).T).T
                out.append(np.column_stack([ego[:, 1], -ego[:, 0], ego[:, 2],
                                            df['i'].to_numpy() / 255.0]))
            return np.concatenate(out)
        return sweeps

    for device, label in [(0, 'PandaSet Pandar64'), (1, 'PandaSet PandarGT')]:
        P[label]._sweeps = factory(device)
        P[label].infos = [i for i in infos if i['frame_idx'] % 4 == 0]


def frame_parts(plat, info, n):
    """Accumulated points, their radial bin, a foreground mask, and the Car boxes."""
    pts = mask_range(plat.accumulate(info, n) if n > 1 else plat.frame(info).points)
    fr = plat.frame(info)
    m = fr.names == plat.car_class
    boxes = fr.boxes[m] if m.sum() else np.zeros((0, 7))
    if len(boxes):
        r = np.linalg.norm(boxes[:, :2], axis=1)
        boxes = boxes[(r >= 5.0) & (r < 70.0)]
    fg = point_mask(pts, boxes)
    b = np.floor(np.clip(np.linalg.norm(pts[:, :2], axis=1), 0, MAX_DIST - 1e-4)
                 / MAX_DIST * BINS).astype(int)
    return pts, b, fg, boxes


def measure(plat, n, frames, recall=1.0, rng=None):
    """Foreground PER BOX and background per frame, plus the per-frame cache.

    The asymmetry matters: foreground points per frame conflates sampling density with how many
    objects a dataset labels (Lyft 22.0 Car/frame against KITTI's 4.31, and KITTI annotates 0%
    behind the vehicle), which is labelling policy rather than anything the correction can act on.
    """
    fg = np.zeros(BINS)
    bg = np.zeros(BINS)
    nbox = np.zeros(BINS)
    cache, used = [], 0
    for info in plat.sample(frames):
        pts, b, f, boxes = frame_parts(plat, info, n)
        seen = boxes
        if recall < 1.0 and len(boxes):
            # simulate a teacher that finds only `recall` of the objects
            seen = boxes[rng.random(len(boxes)) < recall]
            f = point_mask(pts, seen)
        fg += np.bincount(b[f], minlength=BINS)
        bg += np.bincount(b[~f], minlength=BINS)
        if len(seen):
            bb = np.floor(np.clip(np.linalg.norm(seen[:, :2], axis=1), 0, MAX_DIST - 1e-4)
                          / MAX_DIST * BINS).astype(int)
            nbox += np.bincount(bb, minlength=BINS)
        cache.append((pts, b, f, boxes))
        used += 1
    return np.divide(fg, nbox, out=np.zeros(BINS), where=nbox > 0), bg / used, cache


def raw_total(cache):
    """Per-frame whole-cloud profile. Not fg + bg any more, since fg is normalised per box."""
    h = np.zeros(BINS)
    for _, b, _, _ in cache:
        h += np.bincount(b, minlength=BINS)
    return h / len(cache)


def rate(src, tgt, frac=0.01):
    """Matches DataProcessor.per_bin_sample_rate: both sides guarded, ratio clipped at 1."""
    src, tgt = np.asarray(src, float), np.asarray(tgt, float)
    trusted = (src > frac * src.mean()) & (tgt > frac * tgt.mean())
    r = np.ones_like(src)
    np.divide(tgt, src, out=r, where=trusted)
    r[~np.isfinite(r)] = 1.0
    return np.minimum(r, 1.0)


def apply_and_measure(cache, r_fg, r_bg, rng):
    """Radial profile and per-bin median points per Car box after corrections."""
    prof = np.zeros(BINS)
    per_box = {i: [] for i in range(len(R10))}
    for pts, b, fg, boxes in cache:
        p = np.where(fg, r_fg[b], r_bg[b])
        keep = rng.random(len(pts)) < p
        prof += np.bincount(b[keep], minlength=BINS)
        if len(boxes):
            counts = points_in_boxes(pts[keep], boxes)   # per-box counts
            br = np.linalg.norm(boxes[:, :2], axis=1)
            for c, d in zip(counts, br):
                i = int(d // 10)
                if i < len(R10):
                    per_box[i].append(c)
    med = np.array([np.median(per_box[i]) if per_box[i] else np.nan for i in range(len(R10))])
    return prof / len(cache), med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--out', type=Path, default=Path('.'))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    P = build_platforms()
    attach_pandaset_sweeps(P)

    tgt_fg, tgt_bg, tgt_cache = measure(P[TARGET], 1, args.frames)
    tgt_prof, tgt_med = apply_and_measure(tgt_cache, np.ones(BINS), np.ones(BINS), rng)
    print('%-26s %9.0f pts/frame' % (TARGET, tgt_prof.sum()), flush=True)
    # The degraded target channels do not depend on the source, so measure them once rather than
    # once per source - it is the same KITTI pass repeated.
    tgt_by_recall = {rec: measure(P[TARGET], 1, args.frames, recall=rec,
                                  rng=np.random.default_rng(1))[:2] for rec in RECALLS}

    R = {}
    for name, n in PAIRS:
        plat = P[name]
        fg_s, bg_s, cache = measure(plat, n, args.frames)
        r_all = rate(raw_total(cache), tgt_prof)
        r_fg, r_bg = rate(fg_s, tgt_fg), rate(bg_s, tgt_bg)
        raw_prof, raw_med = apply_and_measure(cache, np.ones(BINS), np.ones(BINS), rng)
        g_prof, g_med = apply_and_measure(cache, r_all, r_all, rng)
        f_prof, f_med = apply_and_measure(cache, r_fg, r_bg, rng)
        sens = []
        for rec in RECALLS:
            t_fg, t_bg = tgt_by_recall[rec]
            _, m = apply_and_measure(cache, rate(fg_s, t_fg), rate(bg_s, t_bg), rng)
            sens.append(np.nanmedian(m[1:7] / tgt_med[1:7]))
        R[name] = dict(n=n, r_all=r_all, r_fg=r_fg, r_bg=r_bg, raw=(raw_prof, raw_med),
                       glob=(g_prof, g_med), fga=(f_prof, f_med), sens=sens)
        print('%-26s N=%-3d objects: raw %.2fx  global %.2fx  fg-aware %.2fx (of KITTI)'
              % (name, n, np.nanmedian(raw_med[1:7] / tgt_med[1:7]),
                 np.nanmedian(g_med[1:7] / tgt_med[1:7]),
                 np.nanmedian(f_med[1:7] / tgt_med[1:7])), flush=True)

    npan = len(PAIRS)
    # --- 1. global radial profile ------------------------------------------------------------
    fig, axes = grid(1, npan, (5.2 * npan, 4.4))
    for ax, (name, _) in zip(axes.ravel(), PAIRS):
        d = R[name]
        style(ax)
        ax.plot(CENTRES, d['raw'][0], color=BLUE, lw=2, label='accumulated, uncorrected')
        ax.plot(CENTRES, d['glob'][0], color=ORANGE, lw=2, label='+ global correction')
        ax.plot(CENTRES, d['fga'][0], color=DARK, lw=2, ls='--', label='+ foreground-aware')
        ax.plot(CENTRES, tgt_prof, color=INK2, lw=2, ls=':', label='KITTI target')
        ax.set_yscale('log')
        ax.set_title('%s  N=%d' % (name, d['n']), fontsize=10.5, color=INK, pad=6)
        ax.set_xlabel('radial distance (m)', color=INK2, fontsize=9)
    axes[0, 0].set_ylabel('points per frame per bin', color=INK2, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle('Both corrections match the global profile — that was never the problem',
                 fontsize=13, color=INK, y=.98)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(args.out / 'fg_correction_profile.png', dpi=150, facecolor=SURFACE)

    # --- 2. points per Car box ---------------------------------------------------------------
    fig, axes = grid(1, npan, (5.2 * npan, 4.4))
    x = R10 + 5
    for ax, (name, _) in zip(axes.ravel(), PAIRS):
        d = R[name]
        style(ax)
        ax.plot(x, d['raw'][1], color=BLUE, lw=2, marker='o', ms=4, label='accumulated, uncorrected')
        ax.plot(x, d['glob'][1], color=ORANGE, lw=2, marker='o', ms=4, label='+ global correction')
        ax.plot(x, d['fga'][1], color=DARK, lw=2, ls='--', marker='s', ms=4, label='+ foreground-aware')
        ax.plot(x, tgt_med, color=INK2, lw=2, ls=':', label='KITTI target')
        ax.set_yscale('log')
        ax.set_title('%s  N=%d' % (name, d['n']), fontsize=10.5, color=INK, pad=6)
        ax.set_xlabel('distance to box (m)', color=INK2, fontsize=9)
    axes[0, 0].set_ylabel('median points per Car box', color=INK2, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle('Where they differ: the global correction starves the objects',
                 fontsize=13, color=INK, y=.98)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(args.out / 'fg_correction_points_per_box.png', dpi=150, facecolor=SURFACE)

    # --- 3. the rates themselves -------------------------------------------------------------
    fig, axes = grid(1, npan, (5.2 * npan, 4.0))
    for ax, (name, _) in zip(axes.ravel(), PAIRS):
        d = R[name]
        style(ax)
        ax.plot(CENTRES, d['r_all'], color=ORANGE, lw=2, label='single rate (global)')
        ax.plot(CENTRES, d['r_fg'], color=DARK, lw=2, label='inside boxes')
        ax.plot(CENTRES, d['r_bg'], color=RED, lw=2, ls='--', label='outside boxes')
        ax.set_ylim(-0.03, 1.08)
        ax.set_title('%s  N=%d' % (name, d['n']), fontsize=10.5, color=INK, pad=6)
        ax.set_xlabel('radial distance (m)', color=INK2, fontsize=9)
    axes[0, 0].set_ylabel('keep probability', color=INK2, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle('The foreground channel is kept where the single rate would have thinned it',
                 fontsize=13, color=INK, y=.98)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(args.out / 'fg_correction_rates.png', dpi=150, facecolor=SURFACE)

    # --- 4. what imperfect pseudo-labels cost ------------------------------------------------
    fig, axes = grid(1, 1, (6.6, 4.4))
    ax = axes[0, 0]
    style(ax)
    for name, col in zip([p[0] for p in PAIRS], [BLUE, ORANGE, DARK]):
        ax.plot(RECALLS, R[name]['sens'], color=col, lw=2, marker='o', ms=5, label=name)
    ax.axhline(1.0, color=INK2, lw=1.5, ls=':')
    ax.text(0.22, 1.03, 'parity with KITTI', color=INK2, fontsize=8.5)
    ax.set_xlabel('teacher recall used to build the target foreground channel', color=INK2, fontsize=9)
    ax.set_ylabel('object density, relative to KITTI', color=INK2, fontsize=9)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle('The bias: missed target objects under-feed the foreground channel',
                 fontsize=12.5, color=INK, y=.97)
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(args.out / 'fg_correction_recall_sensitivity.png', dpi=150, facecolor=SURFACE)
    print('wrote 4 figures to', args.out)


if __name__ == '__main__':
    main()
