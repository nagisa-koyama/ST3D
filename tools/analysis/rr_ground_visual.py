"""Figures of the ground-height exploration (experiments_md 20261018_02 §11.5; 20261018_03 §13). CPU, ANALYSIS, source labels
only.

g1: error = estimate - GT bottom against BEV range, per platform, for E5a, E5b-R20, E0-40, E1-raw-p10 (v4 / v5 rule),
    E2-raw-p10 (v6 rule) and the D1 v4 reference (defined cars only), from the saved per-car rows
    (/home/koyama/data/relocate_render_20261018/ground_explore/ground_*.npz; E5's fallback resolved by
    rr_ground_explore.resolve): median and p10-p90 per 5 m bin, +-0.5 m and +-1 m guides.
g2: six source cars at 50-75.2 m, per platform the cars nearest the p25 / p50 / p75 of E1-raw-p10's error (ties: the smaller
    |error - quantile|, then the earlier row). Their frames are re-captured with rr_ground_explore's own capture (the
    data_dict entering RR), the car re-identified by its range and GT bottom, and every shown estimate recomputed with
    rr_ground_explore.estimate_car and checked against the saved row. Side view: horizontal distance along the car's radial
    line x z, scene points within 10 m (BEV) of the car centre.

    python analysis/rr_ground_visual.py g1 --out <dir>
    python analysis/rr_ground_visual.py g2 --out <dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS.parent, TOOLS, TOOLS / 'analysis'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _init_path  # noqa: F401,E402

GE = Path('/home/koyama/data/relocate_render_20261018/ground_explore')
RR4_CFG = 'cfgs/da-ieee-access/centerpoint-gblobs-accum-legaldepth-rr4-nuscenes2waymo.yaml'
SHOW = ('E5a', 'E5b-R20', 'E0-40', 'E1-raw-p10', 'E2-raw-p10', 'D1v4ref')
LABEL = {'E5a': 'E5a: plane + median residual of other cars within 30 m (selected)',
         'E5b-R20': 'E5b-R20: inverse-distance mean of other cars\' bottoms within 20 m',
         'E0-40': 'E0-40: the 4-40 m plane (v2-v5)', 'E1-raw-p10': 'E1: 3 m ring, raw p10 (v4 / v5 loader)',
         'E2-raw-p10': 'E2: 10 m radius, raw p10 (v6 loader)', 'D1v4ref': 'D1 v4 reference (defined cars only)'}
EDGES = np.arange(0.0, 80.0, 5.0)
PLAT_NAME = {0: 'Boston', 1: 'Singapore'}
KEYS = ('NUSCENES_N008', 'NUSCENES_N015')
SURF, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#b4b2ab'
COL = {'E5a': '#2a78d6', 'E5b-R20': '#1baf7a', 'E0-40': '#52514e', 'E1-raw-p10': '#eb6834', 'E2-raw-p10': '#e87ba4',
       'D1v4ref': '#6250d6'}


# ------------------------------------------------------------------------------------------------ pure (tested)
def bin_stats(r, err, edges=EDGES, min_n=5):
    """Per bin [edges[k], edges[k+1]): (centre, n, median, p10, p90); nan where fewer than min_n."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        e = err[(r >= lo) & (r < hi) & np.isfinite(err)]
        out.append([0.5 * (lo + hi), len(e)] + ([float(np.median(e)), float(np.percentile(e, 10)), float(np.percentile(e, 90))]
                                                if len(e) >= min_n else [np.nan] * 3))
    return np.array(out)


def pick_near_quantiles(values, qs=(25, 50, 75)):
    """Indices (into values) nearest each quantile, distinct; ties to the smaller |v - q|, then the earlier index."""
    v = np.asarray(values, np.float64); qv = np.percentile(v, qs); taken = []
    for q in qv:
        d = np.abs(v - q); d[taken] = np.inf
        taken.append(int(np.argmin(d)))
    return np.array(taken), qv


def load_rows():
    import rr_ground_explore as GX
    res = [np.load(GE / f'ground_{k}.npz') for k in KEYS]
    V = [str(x) for x in res[0]['variants']]
    rows = np.concatenate([r['rows'] for r in res])
    est, fb, best14 = GX.resolve(rows, V)
    return rows, V, est, fb, best14


# ---------------------------------------------------------------------------------------------------------- g1
def g1(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, V, est, fb, best14 = load_rows()
    plat, r, gtb = rows[:, 0], rows[:, 2], rows[:, 3]
    stats = {}
    fig, axs = plt.subplots(len(SHOW) + 1, 2, figsize=(20, 3.0 * (len(SHOW) + 1) + 1.2), facecolor=SURF, sharex=True)
    for ci, p in enumerate((0, 1)):
        m0 = plat == p
        for ri, k in enumerate(SHOW):
            err = est[k] - gtb
            m = m0 & (~fb[k] if k == 'D1v4ref' else np.ones(len(rows), bool))
            s = bin_stats(r[m], err[m])
            stats[f'{PLAT_NAME[p]}|{k}'] = s.round(4).tolist()
            for ax in (axs[ri + 1, ci], axs[0, ci]):
                ok = np.isfinite(s[:, 2])
                if ax is axs[ri + 1, ci]:
                    ax.fill_between(s[ok, 0], s[ok, 3], s[ok, 4], color=COL[k], alpha=0.22, lw=0, label='p10-p90')
                    ax.plot(s[ok, 0], s[ok, 2], color=COL[k], lw=2.2, marker='o', ms=4, label='median')
                    for x_, n_ in zip(s[:, 0], s[:, 1]):
                        if n_:
                            ax.text(x_, 1.0, f'{int(n_)}', transform=ax.get_xaxis_transform(), ha='center', va='top',
                                    fontsize=7.5, color=MUTED)
                    ax.set_title(f'{PLAT_NAME[p]}: {LABEL[k]}' + (f'  (fallback {best14} where no other car)' if k.startswith('E5') else ''),
                                 fontsize=11, color=INK, loc='left')
                else:
                    ax.plot(s[ok, 0], s[ok, 2], color=COL[k], lw=2.0, marker='o', ms=3, label=k)
        axs[0, ci].set_title(f'{PLAT_NAME[p]}: medians of all six', fontsize=11.5, color=INK, loc='left', weight='bold')
        axs[0, ci].legend(fontsize=9.5, frameon=False, ncol=3, labelcolor=INK2, loc='lower left')
    for ax in axs.ravel():
        ax.set_facecolor(SURF); ax.set_ylim(-2.0, 2.5); ax.set_xlim(0, 75.2)
        for yv, ls in ((0, '-'), (0.5, '--'), (-0.5, '--'), (1.0, ':'), (-1.0, ':')):
            ax.axhline(yv, color=MUTED if yv else INK2, lw=0.9 if yv else 0.8, ls=ls, zorder=0)
        ax.tick_params(colors=INK2, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
    for ci in range(2):
        axs[-1, ci].set_xlabel('BEV range of the car centre (m), 5 m bins', fontsize=10, color=INK2)
    for ri in range(len(SHOW) + 1):
        axs[ri, 0].set_ylabel('estimate - GT bottom (m)', fontsize=9.5, color=INK2)
    fig.suptitle('Ground-height estimators against real nuScenes TRAIN cars, leave-one-out (20261018_02 §11.5): error by range. '
                 'Dashed +-0.5 m, dotted +-1 m; grey numbers = cars per bin', fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'G1_error_vs_range.png', dpi=100, facecolor=SURF); plt.close(fig)
    json.dump(dict(stats=stats, cols=['centre', 'n', 'median', 'p10', 'p90'], e5_fallback=best14),
              open(out / 'G1_stats.json', 'w'), indent=1, default=float)
    print(f'[g1] {out / "G1_error_vs_range.png"}; E5 fallback {best14}', flush=True)


# ---------------------------------------------------------------------------------------------------------- g2
class _Captured(Exception):
    pass


def g2(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import rr_ground_explore as GX
    from rr_gate import build
    from pcdet.datasets.augmentor import relocate_render as rr
    rows, V, est, fb, best14 = load_rows()
    plat, fr, r, gtb = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2], rows[:, 3]
    e1 = est['E1-raw-p10'] - gtb
    picks = []
    for p in (0, 1):
        idx = np.flatnonzero((plat == p) & (r >= 50.0) & (r < GX.R_MAX))
        k, qv = pick_near_quantiles(e1[idx])
        picks += [(p, int(idx[j]), q, float(qq)) for j, q, qq in zip(k, (25, 50, 75), qv)]
    print(f'[g2] picks (platform, row, quantile, E1 error at that quantile): {picks}', flush=True)
    data = []
    for p, key in enumerate(KEYS):
        mine = [x for x in picks if x[0] == p]
        cfg, dcfg, ds = build(RR4_CFG, key)
        aug = ds.data_augmentor
        CAP = {}

        def capture(data_dict=None):
            CAP['before'] = {kk: np.copy(data_dict[kk]) for kk in ('points', 'gt_boxes', 'gt_names')}
            raise _Captured
        aug.data_augmentor_queue = [capture] + list(aug.data_augmentor_queue[1:])
        for _, ri, q, qq in mine:
            CAP.clear()
            try:
                ds[int(fr[ri])]
            except _Captured:
                pass
            B = CAP['before']
            P = np.asarray(B['points'], np.float64)[:, :3]; boxes = np.asarray(B['gt_boxes'], np.float64)
            names = np.asarray(B['gt_names']); car = names == 'Car'
            rb = np.hypot(boxes[:, 0], boxes[:, 1])
            j = int(np.flatnonzero(car & (np.abs(rb - r[ri]) < 1e-6) & (np.abs(boxes[:, 2] - boxes[:, 5] / 2 - gtb[ri]) < 1e-6))[0])
            b = boxes[j, :7]
            keep = GX.leave_one_out(P, b, rr._BevGrid(P))
            X = P[keep]
            pl40 = rr.ground_plane(X, GX.PLANE_A)
            oth = boxes[car & (np.arange(len(boxes)) != j), :7]
            e = GX.estimate_car(X, b, oth, pl40, rr.ground_plane(X, GX.PLANE_B))
            chk = {k: (e[k][0], float(rows[ri, 5 + V.index(k)])) for k in ('E0-40', 'E1-raw-p10', 'E2-raw-p10', 'E5a', 'D1v4ref')}
            same = all((np.isnan(x) and np.isnan(y)) or abs(x - y) < 1e-9 for x, y in chk.values())
            data.append(dict(plat=p, row=ri, q=q, qv=qq, frame=int(fr[ri]), b=b, P=P, keep=keep, pl40=pl40, oth=oth, est=e,
                             e5a_resolved=float(est['E5a'][ri]), e5a_fb=bool(fb['E5a'][ri] or not np.isfinite(rows[ri, 5 + V.index('E5a')])),
                             same=same, chk=chk))
            print(f'[g2] {key} info {fr[ri]} car r {r[ri]:.1f}: recomputed == saved {same}', flush=True)
        del ds
    # figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 11.5), facecolor=SURF)
    for d in data:
        ax = axs[d['plat'], [25, 50, 75].index(d['q'])]
        b = d['b']; P = d['P']; c = b[:2]; er = c / np.linalg.norm(c); el = np.array([-er[1], er[0]])
        rel = P[:, :2] - c; u = rel @ er
        near = np.hypot(rel[:, 0], rel[:, 1]) <= 10.0
        inbox = ~d['keep']
        ax.scatter(u[near & ~inbox], P[near & ~inbox, 2], s=2.5, c='#9a988f', lw=0, rasterized=True, label='scene points (leave-one-out)')
        ax.scatter(u[near & inbox], P[near & inbox, 2], s=3, c='#c9c7c0', lw=0, rasterized=True, label='the car\'s own points (removed)')
        cu = (rr.box_corners(b)[:, :2] - c) @ er
        zb, zt = b[2] - b[5] / 2, b[2] + b[5] / 2
        u0, u1 = float(cu.min()), float(cu.max())
        ax.add_patch(plt.Rectangle((u0, zb), u1 - u0, zt - zb, fill=False, ec=INK, lw=1.6, label='GT box'))
        uu = np.linspace(-10, 10, 50); xy = c + uu[:, None] * er
        ax.plot(uu, rr.plane_z(d['pl40'], xy[:, 0], xy[:, 1]), color=COL['E0-40'], lw=1.6, label='4-40 m plane (E0-40)')
        e = d['est']
        for k, lab, dy in (('E1-raw-p10', 'E1 ring p10', 0), ('E2-raw-p10', 'E2 radius p10', 0)):
            ax.hlines(e[k][0], u0 - 1.2, u1 + 1.2, color=COL[k], lw=2.0, label=f'{lab}{" (plane fallback)" if e[k][1] else ""}')
        ax.hlines(d['e5a_resolved'], u0 - 1.6, u1 + 1.6, color=COL['E5a'], lw=2.6,
                  label='E5a' + (f' (fallback {best14})' if d['e5a_fb'] else ''))
        o = d['oth']
        if len(o):
            dd = np.hypot(o[:, 0] - c[0], o[:, 1] - c[1]); s = dd <= 30.0
            zc = float(rr.plane_z(d['pl40'], c[0], c[1]))
            res_ = (o[s, 2] - o[s, 5] / 2) - rr.plane_z(d['pl40'], o[s, 0], o[s, 1])
            ax.scatter(np.full(int(s.sum()), u1 + 2.3), zc + res_, marker='_', s=160, c=COL['E5a'], lw=1.6, zorder=5,
                       label=f'other cars within 30 m: bottom - plane, moved to this car ({int(s.sum())})')
            for ob in o[(np.abs((o[:, :2] - c) @ er) <= 10) & (np.abs((o[:, :2] - c) @ el) <= 10)]:
                ocu = (rr.box_corners(ob)[:, :2] - c) @ er
                ax.add_patch(plt.Rectangle((ocu.min(), ob[2] - ob[5] / 2), np.ptp(ocu), ob[5], fill=False, ec='#8a8880', lw=1.0,
                                           ls='--'))
        gt = zb
        ax.set_title(f"{PLAT_NAME[d['plat']]} · E1 error near p{d['q']} ({d['qv']:+.2f} m) · train info {d['frame']} · r {np.hypot(*c):.1f} m\n"
                     f"GT bottom {gt:+.2f} · E5a {d['e5a_resolved'] - gt:+.2f} · plane {e['E0-40'][0] - gt:+.2f} · "
                     f"E1 {e['E1-raw-p10'][0] - gt:+.2f} · E2 {e['E2-raw-p10'][0] - gt:+.2f} m (errors)"
                     + ('' if d['same'] else ' · NOT REPRODUCED'), fontsize=10.5, color=INK, loc='left')
        ax.set_xlim(-10.5, 10.5)
        zz = [gt, zt, d['e5a_resolved'], e['E1-raw-p10'][0], e['E2-raw-p10'][0]]
        ax.set_ylim(min(zz) - 1.5, max(zz) + 1.5)
        ax.set_facecolor(SURF); ax.tick_params(colors=INK2, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
        ax.set_xlabel('distance along the car\'s radial line from its centre (m; + = away from the sensor)', fontsize=9.5, color=INK2)
        ax.set_ylabel('z, ground-origin loader frame (m)', fontsize=9.5, color=INK2)
        ax.legend(fontsize=8.2, frameon=False, labelcolor=INK2, loc='upper left')
    fig.suptitle('Ground placement for six source cars at 50-75 m (per platform, the cars nearest the p25 / p50 / p75 of E1\'s error): '
                 'side view along the radial line, scene points within 10 m (BEV)', fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'G2_side_views.png', dpi=100, facecolor=SURF); plt.close(fig)
    json.dump([dict(platform=PLAT_NAME[d['plat']], row=d['row'], quantile=d['q'], e1_quantile_value=d['qv'], frame=d['frame'],
                    range_m=float(np.hypot(*d['b'][:2])), gt_bottom=float(d['b'][2] - d['b'][5] / 2), reproduced=d['same'],
                    errors={k: (v[0] - float(d['b'][2] - d['b'][5] / 2)) for k, v in d['chk'].items()},
                    e5a_resolved_error=d['e5a_resolved'] - float(d['b'][2] - d['b'][5] / 2), e5a_fallback=d['e5a_fb'])
               for d in data], open(out / 'G2_picks.json', 'w'), indent=1, default=float)
    print(f'[g2] {out / "G2_side_views.png"}; reproduced {sum(d["same"] for d in data)} of {len(data)}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['g1', 'g2'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time()
    {'g1': g1, 'g2': g2}[a.mode](a)
    print(f'[{a.mode}] done in {time.time() - t0:.0f} s', flush=True)
