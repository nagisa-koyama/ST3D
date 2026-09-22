"""What the foreground-aware density correction changes, at the recommended accumulation depth.

The plain correction gives every point in a radial bin the same keep-probability, which scales the
points on objects and the points on everything else by the same factor and so cannot change a bin's
foreground SHARE. Matching the global radial profile therefore leaves source objects at
sigma_src/sigma_tgt of the target's object density. The foreground-aware variant corrects the
inside-box and outside-box channels separately.

These figures put the two side by side against KITTI, at the OBJECT-criterion depth from
experiments_md/20260922_07 - the depth at which each source's points-per-box reaches the target's.
Using the global-criterion depth instead would beg the question: that criterion stops as soon as
TOTAL density matches, which by construction leaves objects short, so an object-density figure drawn
at it would find objects short at a depth chosen to leave them short. nuScenes is shown at both
depths (N=9 global, N=15 object) so the gap between the two is visible rather than asserted.

HONESTY ABOUT THE TARGET BOXES: in a real run the target's foreground channel comes from the
teacher's PSEUDO-LABELS. Here it is built from KITTI's real annotations, which is the recall = 1
case - an upper bound on what the mechanism delivers, not what a run gets. Figure 4 exists for
exactly that reason: it degrades the target boxes to a given recall and shows what the bias costs.
"""
import argparse
import zlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from domain_gap_figures import (SURFACE, INK, INK2, GRID, BLUE, ORANGE, RED, DARK, style, grid)  # noqa: E501
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
# (source, N) at the OBJECT-criterion depth from 20260922_07, because these are object-density
# figures. Sizing them on the global criterion would be circular: that criterion stops as soon as
# TOTAL density matches, which by construction leaves objects short - so the objects would be found
# short at a depth chosen to leave them short. nuScenes appears at both depths so the gap between
# the two criteria is visible rather than asserted.
# nuScenes appears three times. 9 is the global-criterion depth, 15 the pooled-object one, and
# 150 is what the PER-RANGE table needs for the last ring (60-70 m) to reach parity - the depth at
# which every ring is at or above KITTI rather than only the pooled average.
PAIRS = [('nuScenes n015 Singapore', 9), ('nuScenes n015 Singapore', 15),
         ('nuScenes n015 Singapore', 150),
         ('Lyft 40-beam', 2), ('PandaSet Pandar64', 2)]
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
            # (points, frames actually delivered) - the same contract the nuScenes chaining uses,
            # so a caller can tell a short sequence from a mechanism that has run out.
            return np.concatenate(out), len(out)
        return sweeps

    for device, label in [(0, 'PandaSet Pandar64'), (1, 'PandaSet PandarGT')]:
        P[label]._sweeps = factory(device)
        P[label].infos = [i for i in infos if i['frame_idx'] % 4 == 0]


def attach_nuscenes_chaining(P):
    """Let nuScenes accumulate past N=10 by chaining keyframes.

    A nuScenes info carries exactly 9 stored sweeps, so the registry's accumulate() saturates at
    keyframe + 9 and silently returns the SAME cloud for N=10, 15, 20 or 30. Chaining keyframes
    lifts that; see accumulation_deep_nuscenes.py for the transform and its 1e-13 check.
    """
    from accumulation_deep_nuscenes import frames_back, INFOS
    tok2idx = {inf['token']: i for i, inf in enumerate(INFOS)}

    def sweeps(info, n):
        parts = list(frames_back(tok2idx[info['token']], n))
        out = np.concatenate(parts)
        return np.column_stack([out, np.zeros(len(out))]), len(parts)

    for label in ['nuScenes n008 Boston', 'nuScenes n015 Singapore']:
        P[label]._sweeps = sweeps


def attach_nuscenes_compensated(P):
    """Per-object motion-compensated accumulation for nuScenes.

    The default path - and the production one, get_lidar_with_sweeps - concatenates ego-transformed
    sweeps and does nothing about objects that MOVE. Over 30 frames a moving object gains x1.01
    against a static one's x2.74 (20260922_05 Ablation 3), so uncompensated depth accrues almost
    entirely to parked cars. That is not a detail: it is what makes the per-box distribution go
    bimodal, and the skew is what separates the mean the correction targets from the median.

    This moves each object's points onto that object's box in the anchor frame using its track id,
    interpolating boxes for the sweeps between annotated keyframes. Measured static/moving ratio
    1.04 (20260922_05 Ablation 7).
    """
    from nuscenes_motion_compensation import accumulate_exact
    from nuscenes_chain import INFOS
    tok2idx = {inf['token']: i for i, inf in enumerate(INFOS)}

    def sweeps(info, n):
        pts, used = accumulate_exact(tok2idx[info['token']], n, compensate_motion=True)
        return np.column_stack([pts, np.zeros(len(pts))]), used

    for label in ['nuScenes n008 Boston', 'nuScenes n015 Singapore']:
        P[label]._sweeps = sweeps


def frame_parts(plat, info, n):
    """Accumulated points, their radial bin, a foreground mask, and the Car boxes."""
    if n > 1:
        got = plat.accumulate(info, n)
        if isinstance(got, tuple):
            # A chaining mechanism reports what it actually delivered. Falling short means this
            # anchor sits too near the start of its scene, which is a property of the anchor, not
            # of the mechanism - the caller drops it.
            raw, used = got
            if used < n:
                return None
        else:
            # Stored sweeps cap at len(sweeps) + 1 and return the SAME cloud for anything beyond,
            # which would silently relabel an N=10 accumulation as N=15. That is a mechanism
            # limit, not a bad anchor, so it is fatal.
            raw = got
            cap = len(info.get('sweeps', [])) + 1
            assert n <= cap, ('stored sweeps cap at N=%d, so N=%d would silently reuse the N=%d '
                              'cloud. Attach keyframe chaining for this depth.' % (cap, n, cap))
    else:
        raw = plat.frame(info).points
    pts = mask_range(raw)
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


def measure(plat, n, frames, recall=1.0, rng=None, infos=None):
    """Foreground PER BOX and background per frame, plus the per-frame cache.

    The asymmetry matters: foreground points per frame conflates sampling density with how many
    objects a dataset labels (Lyft 22.0 Car/frame against KITTI's 4.31, and KITTI annotates 0%
    behind the vehicle), which is labelling policy rather than anything the correction can act on.
    """
    fg = np.zeros(BINS)
    bg = np.zeros(BINS)
    nbox = np.zeros(BINS)
    cache, used, kept, skipped = [], 0, [], 0
    for info in (plat.sample(frames) if infos is None else infos):
        parts = frame_parts(plat, info, n)
        if parts is None:                      # anchor too close to the start of its scene
            skipped += 1
            continue
        pts, b, f, boxes = parts
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
        kept.append(info)
        used += 1
    if skipped:
        print('    (N=%d: dropped %d of %d anchors that could not chain back that far)'
              % (n, skipped, skipped + used), flush=True)
    return np.divide(fg, nbox, out=np.zeros(BINS), where=nbox > 0), bg / used, cache, kept


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


def apply_and_measure(cache, r_fg, r_bg, seed):
    """Each variant draws from its OWN generator, seeded by name.

    A single shared generator makes every result depend on the order the variants happen to be
    computed in - adding one measurement upstream shifted nuScenes' corrected object density from
    0.28x to 0.21x, which is sampling noise masquerading as a finding.
    """
    # crc32, not hash(): Python salts string hashing per process, so hash() would make this
    # irreproducible across runs - the opposite of the point.
    rng = np.random.default_rng(zlib.crc32(str(seed).encode()))
    prof = np.zeros(BINS)
    per_box = {i: [] for i in range(len(R10))}
    pooled = []
    # Per-BIN foreground density, points per box in that bin. This is the quantity rate_fg
    # actually controls - it is applied independently per bin - and unlike any pooled figure it
    # carries no mixture term, so comparing it against the target's is like for like. Pooling
    # instead averages the target's densities over the SOURCE's box-range distribution, and
    # nuScenes puts 74.5% of its boxes inside 30 m against KITTI's 52.4%.
    fg_bin = np.zeros(BINS)
    nbox_bin = np.zeros(BINS)
    for pts, b, fg, boxes in cache:
        p = np.where(fg, r_fg[b], r_bg[b])
        keep = rng.random(len(pts)) < p
        prof += np.bincount(b[keep], minlength=BINS)
        fg_bin += np.bincount(b[keep & fg], minlength=BINS)
        if len(boxes):
            bb = np.floor(np.clip(np.linalg.norm(boxes[:, :2], axis=1), 0, MAX_DIST - 1e-4)
                          / MAX_DIST * BINS).astype(int)
            nbox_bin += np.bincount(bb, minlength=BINS)
        if len(boxes):
            counts = points_in_boxes(pts[keep], boxes)   # per-box counts
            br = np.linalg.norm(boxes[:, :2], axis=1)
            for c, d in zip(counts, br):
                i = int(d // 10)
                if i < len(R10):
                    per_box[i].append(c)
                if 5.0 <= d < 70.0:
                    pooled.append(c)
    # MEAN is the primary statistic because it is what the correction actually targets: a
    # per-point keep-probability p leaves p x sum(points), so rate_fg = sum_t/sum_s matches the
    # mean exactly and can match no other statistic exactly. The median is carried alongside
    # because the gap between them IS the finding - uncompensated deep accumulation gives parked
    # cars every frame and moving cars one, so the per-box distribution goes bimodal and mean and
    # median separate by 35x at N=150 against KITTI's 2.6x.
    avg = np.array([np.mean(per_box[i]) if per_box[i] else np.nan for i in range(len(R10))])
    med = np.array([np.median(per_box[i]) if per_box[i] else np.nan for i in range(len(R10))])
    pool_avg = float(np.mean(pooled)) if pooled else float('nan')
    pool_med = float(np.median(pooled)) if pooled else float('nan')
    dens = np.divide(fg_bin, nbox_bin, out=np.zeros(BINS), where=nbox_bin > 0)
    return prof / len(cache), avg, pool_avg, med, pool_med, dens, nbox_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--compensate', action='store_true',
                    help='per-object motion compensation for nuScenes (production does NOT do '
                         'this; without it, depth accrues almost entirely to parked cars)')
    ap.add_argument('--out', type=Path, default=Path('.'))
    args = ap.parse_args()
    P = build_platforms()
    attach_pandaset_sweeps(P)
    if args.compensate:
        attach_nuscenes_compensated(P)
        print('nuScenes: PER-OBJECT MOTION COMPENSATION ON', flush=True)
    elif any(n > 10 and name.startswith('nuScenes') for name, n in PAIRS):
        attach_nuscenes_chaining(P)
    if not args.compensate:
        print('nuScenes: motion compensation OFF - this is the production MAX_SWEEPS behaviour, '
              'and depth accrues almost entirely to static objects', flush=True)

    tgt_fg, tgt_bg, tgt_cache, _ = measure(P[TARGET], 1, args.frames)
    tgt_prof, tgt_avg, tgt_pool, tgt_med, tgt_pmed, tgt_d, nb_t = apply_and_measure(
        tgt_cache, np.ones(BINS), np.ones(BINS), 'target')
    print('%-26s %9.0f pts/frame  %.0f pts/Car box (5-70 m, pooled)'
          % (TARGET, tgt_prof.sum(), tgt_pool), flush=True)
    # The degraded target channels do not depend on the source, so measure them once rather than
    # once per source - it is the same KITTI pass repeated.
    tgt_by_recall = {rec: measure(P[TARGET], 1, args.frames, recall=rec,
                                  rng=np.random.default_rng(1))[:2] for rec in RECALLS}

    R = {}
    for name, n in PAIRS:
        key = (name, n)
        plat = P[name]
        fg_s, bg_s, cache, kept = measure(plat, n, args.frames)
        r_all = rate(raw_total(cache), tgt_prof)
        r_fg, r_bg = rate(fg_s, tgt_fg), rate(bg_s, tgt_bg)
        # N=1 is the baseline everything else is measured against: what a single frame gives
        # before any accumulation or correction at all.
        # the same anchors, so the N=1 baseline is a like-for-like comparison
        _, _, one_cache, _ = measure(plat, 1, args.frames, infos=kept)
        one_prof, one_avg, one_pool, one_med, one_pmed, one_d, _ = apply_and_measure(
            one_cache, np.ones(BINS), np.ones(BINS), 'one%s%d' % (name, n))
        raw_prof, raw_avg, raw_pool, raw_med, raw_pmed, raw_d, nb_s = apply_and_measure(
            cache, np.ones(BINS), np.ones(BINS), 'raw%s%d' % (name, n))
        g_prof, g_avg, g_pool, g_med, g_pmed, g_d, _ = apply_and_measure(
            cache, r_all, r_all, 'glob%s%d' % (name, n))
        f_prof, f_avg, f_pool, f_med, f_pmed, f_d, _ = apply_and_measure(
            cache, r_fg, r_bg, 'fga%s%d' % (name, n))
        sens = []
        for rec in RECALLS:
            t_fg, t_bg = tgt_by_recall[rec]
            _, _, mp, _, _, sd, _ = apply_and_measure(cache, rate(fg_s, t_fg), rate(bg_s, t_bg),
                                                       'sens%s%d%.1f' % (name, n, rec))
            ok = (nb_s >= 3) & (nb_t >= 3) & (tgt_d > 0)
            sens.append(float(np.median(sd[ok] / tgt_d[ok])) if ok.any() else float('nan'))
        print('%-26s N=%-3d recall sweep %s -> %s'
              % (name, n, RECALLS, ['%.2f' % v for v in sens]), flush=True)
        R[key] = dict(n=n, r_all=r_all, r_fg=r_fg, r_bg=r_bg, one=(one_prof, one_avg),
                       raw=(raw_prof, raw_avg), glob=(g_prof, g_avg), fga=(f_prof, f_avg),
                       sens=sens, d=dict(one=one_d, raw=raw_d, glob=g_d, fga=f_d),
                       tgt_d=tgt_d, ok=ok)
        ok = (nb_s >= 3) & (nb_t >= 3) & (tgt_d > 0)
        rel = lambda d: float(np.median(d[ok] / tgt_d[ok])) if ok.any() else float('nan')
        print('%-26s N=%-3d PER-BIN achieved/target (%d live bins): N=1 %.2f  accum %.2f  '
              'global %.2f  fg-aware %.2f'
              % (name, n, int(ok.sum()), rel(one_d), rel(raw_d), rel(g_d), rel(f_d)), flush=True)
        print('%-26s N=%-3d MEAN pts/box (KITTI %.0f): N=1 %.0f (%.2fx)  accum %.0f (%.2fx)  '
              'global %.0f (%.2fx)  fg-aware %.0f (%.2fx)'
              % (name, n, tgt_pool, one_pool, one_pool / tgt_pool, raw_pool, raw_pool / tgt_pool,
                 g_pool, g_pool / tgt_pool, f_pool, f_pool / tgt_pool), flush=True)
        print('%-26s       median      (KITTI %.0f): N=1 %.0f (%.2fx)  accum %.0f (%.2fx)  '
              'global %.0f (%.2fx)  fg-aware %.0f (%.2fx)   skew accum %.1fx, KITTI %.1fx'
              % ('', tgt_pmed, one_pmed, one_pmed / tgt_pmed, raw_pmed, raw_pmed / tgt_pmed,
                 g_pmed, g_pmed / tgt_pmed, f_pmed, f_pmed / tgt_pmed,
                 raw_pool / max(raw_pmed, 1e-9), tgt_pool / max(tgt_pmed, 1e-9)), flush=True)

    npan = len(PAIRS)
    # --- 1. global radial profile ------------------------------------------------------------
    fig, axes = grid(1, npan, (5.2 * npan, 4.4))
    for ax, key in zip(axes.ravel(), PAIRS):
        name, d = key[0], R[key]
        style(ax)
        ax.plot(CENTRES, d['one'][0], color=RED, lw=1.6, ls='-', marker='^', ms=3.5,
                markevery=6, label='single frame (N=1)')
        ax.plot(CENTRES, d['raw'][0], color=BLUE, lw=2, label='accumulated, uncorrected')
        ax.plot(CENTRES, d['glob'][0], color=ORANGE, lw=2, label='+ global correction')
        ax.plot(CENTRES, d['fga'][0], color=DARK, lw=2, ls='--', label='+ foreground-aware')
        ax.plot(CENTRES, tgt_prof, color=INK2, lw=2, ls=':', label='KITTI target')
        ax.set_yscale('log')
        ax.set_title('%s  N=%d' % (name, d['n']), fontsize=10.5, color=INK, pad=6)
        ax.set_xlabel('radial distance (m)', color=INK2, fontsize=9)
    axes[0, 0].set_ylabel('points per frame per bin', color=INK2, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle('From a single frame to the corrected source — both corrections match the target',
                 fontsize=13, color=INK, y=.98)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(args.out / 'fg_correction_profile.png', dpi=150, facecolor=SURFACE)

    # --- 2. per-bin foreground density, absolute, against the KITTI target ------------------
    # Points per Car box WITHIN each range bin - the quantity rate_fg controls, since the rate is
    # applied independently per bin. Plotted absolute with KITTI as a reference curve, the same way
    # the profile figure plots points per frame: the foreground-aware curve lying ON the KITTI
    # curve is the result. A POOLED points-per-box number cannot show this - it averages the
    # target's per-bin densities over the SOURCE's box-range distribution, and nuScenes puts 74.5%
    # of its boxes inside 30 m against KITTI's 52.4%.
    fig, axes = grid(1, npan, (5.2 * npan, 4.4))
    for ax, key in zip(axes.ravel(), PAIRS):
        name, d = key[0], R[key]
        style(ax)
        ok, tg = d['ok'], d['tgt_d']
        xr = CENTRES[ok]
        ax.plot(xr, d['d']['one'][ok], color=RED, lw=1.6, marker='^', ms=4,
                label='single frame (N=1)')
        ax.plot(xr, d['d']['raw'][ok], color=BLUE, lw=2, marker='o', ms=4,
                label='accumulated, uncorrected')
        ax.plot(xr, d['d']['glob'][ok], color=ORANGE, lw=2, marker='o', ms=4,
                label='+ global correction')
        ax.plot(xr, d['d']['fga'][ok], color=DARK, lw=2, ls='--', marker='s', ms=4,
                label='+ foreground-aware')
        ax.plot(xr, tg[ok], color=INK2, lw=2, ls=':', label='KITTI target')
        ax.set_yscale('log')
        ax.set_title('%s  N=%d' % (name, d['n']), fontsize=10.5, color=INK, pad=6)
        ax.set_xlabel('distance to box (m)', color=INK2, fontsize=9)
    axes[0, 0].set_ylabel('points per Car box, within the bin', color=INK2, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle('Foreground density per range bin - the foreground-aware curve sits on the target',
                 fontsize=12.5, color=INK, y=.98)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(args.out / 'fg_correction_per_bin_density.png', dpi=150, facecolor=SURFACE)

    # --- 3. the rates themselves -------------------------------------------------------------
    fig, axes = grid(1, npan, (5.2 * npan, 4.0))
    for ax, key in zip(axes.ravel(), PAIRS):
        name, d = key[0], R[key]
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
    for key, col in zip(PAIRS, [BLUE, RED, DARK, ORANGE, INK2]):
        ax.plot(RECALLS, R[key]['sens'], color=col, lw=2, marker='o', ms=5,
                label='%s  N=%d' % (key[0], key[1]))
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
