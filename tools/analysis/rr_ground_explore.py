"""Ground-height estimator exploration for RR copies (ANALYSIS; experiments_md 20261018_02 §11, protocol fixed in T05 §5,
experiments_md c25bd9a): leave-one-out on real nuScenes TRAIN cars. Source labels only, so it is UDA-legal.

run: per platform, 400 TRAIN keyframes (np.random.default_rng(0) over the infos not in the gate's 200), the data_dict that
    enters RR (27471's source loader; RR never runs). For every GT `Car` within 75.2 m: the cloud minus the points within
    0.5 m (BEV) of its rectangle, the planes refitted on that cloud, and every variant's estimate of the height at the
    car's BEV centre (E0 planes; E1 ring, E2 radius and E3 annular sectors x raw / cell percentiles; E5 other cars' GT
    bottoms; the D1 v4 reference). Writes <out>/ground_<key>.npz.
report: error = estimate - GT bottom, per platform x range bin and pooled over 40-75.2 m; the declared selection rule.

    python analysis/rr_ground_explore.py run --cfg <rr4 cfg> --key NUSCENES_N008 --out <dir> [--max_frames k]
    python analysis/rr_ground_explore.py report --dir <dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _init_path  # noqa: F401,E402
from pcdet.datasets.augmentor import relocate_render as rr  # noqa: E402

D_LO = 0.5                       # the grown footprint (BEV distance to the rectangle), excluded everywhere
E1_DHI = 3.0
E2_RADIUS = 10.0
E3_SHAPES = ((2.0, 5.0), (2.0, 10.0), (5.0, 5.0), (5.0, 10.0))      # (dr, w): range half-width, arc half-length at r
PCTS = (5, 10, 20, 30)
MIN_POINTS, MIN_CELLS, CELL, CELL_MIN_PTS = 10, 3, 1.0, 3
E5A_R, E5B_R, IDW_DMIN = 30.0, (10.0, 20.0), 0.1
REF_RADIUS, REF_PCT, REF_MIN_CELLS = 5.0, 20.0, 3                    # D1 v4's reference (§8.2)
PLANE_A, PLANE_B = 40.0, 75.2
R_MAX = 75.2
BINS = (0.0, 20.0, 40.0, 50.0, 60.0, 75.2)
GATE_N, N_SAMPLE = 200, 400
PLAT_NAME = {0: 'Boston', 1: 'Singapore'}
PLAT_ID = {'NUSCENES_N008': 0, 'NUSCENES_N015': 1}


def regions():
    return ['E1', 'E2'] + [f'E3-dr{dr:g}-w{w:g}' for dr, w in E3_SHAPES]


def e14_names():
    return [f'{g}-{k}-p{p}' for g in regions() for k in ('raw', 'cell') for p in PCTS]


def variant_names():
    return ['E0-40', 'E0-75'] + e14_names() + ['E5a', 'E5b-R10', 'E5b-R20', 'D1v4ref']


def gate_frames(n_infos, gate_n=GATE_N):
    step = n_infos // gate_n
    return np.array([k * step for k in range(gate_n)], np.int64)


def sample_frames(n_infos, n=N_SAMPLE, seed=0, gate_n=GATE_N):
    """400 info indices, disjoint from the gate's 200 (§11.1)."""
    pool = np.setdiff1d(np.arange(n_infos), gate_frames(n_infos, gate_n))
    sel = np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))
    assert not np.isin(sel, gate_frames(n_infos, gate_n)).any()
    return sel


def leave_one_out(xyz, box, grid=None):
    """Keep-mask of the cloud without the points within D_LO (BEV) of the box's rectangle, at any height."""
    b = np.asarray(box, np.float64)
    keep = np.ones(len(xyz), bool)
    if not len(xyz):
        return keep
    c = grid.query(rr.bev_corners(b[None], D_LO + 0.05)[0]) if grid is not None else np.arange(len(xyz))
    if len(c):
        keep[c[rr.footprint_distance(xyz[c, :2], b) <= D_LO]] = False
    return keep


def cell_values(pts):
    """Second-lowest z of the 1 m cells of `pts` holding >= 3 points (the loader's GroundCells)."""
    if not len(pts):
        return np.zeros(0)
    v = rr.GroundCells(pts, CELL, CELL_MIN_PTS).vals
    return v[np.isfinite(v)]


def region_masks(xyz, box):
    """{region: boolean mask over xyz} for E1, E2 and the four E3 sectors (grown footprint excluded)."""
    b = np.asarray(box, np.float64)
    xy = np.asarray(xyz[:, :2], np.float64)
    d = rr.footprint_distance(xy, b)
    out = {'E1': (d > D_LO) & (d <= E1_DHI),
           'E2': (d > D_LO) & (np.hypot(xy[:, 0] - b[0], xy[:, 1] - b[1]) <= E2_RADIUS)}
    R = np.hypot(xy[:, 0], xy[:, 1]); r = float(np.hypot(b[0], b[1]))
    dphi = np.abs((np.arctan2(xy[:, 1], xy[:, 0]) - np.arctan2(b[1], b[0]) + np.pi) % (2 * np.pi) - np.pi)
    for dr, w in E3_SHAPES:
        half = min(w / max(r, 1e-6), np.pi)
        out[f'E3-dr{dr:g}-w{w:g}'] = (d > D_LO) & (R >= r - dr) & (R <= r + dr) & (dphi <= half)
    return out


def estimate_car(xyz, box, others, pl40, pl75):
    """{variant: (estimate, fallback)} for one car on its leave-one-out cloud `xyz` (k, 3). others: (m, >= 7) the frame's
    other Car boxes; pl40 / pl75: planes of `xyz` (pl40 not None). E5's estimate is nan where no other car is in range
    (the fallback is resolved by `report`)."""
    b = np.asarray(box, np.float64); x, y = b[0], b[1]
    z40 = float(rr.plane_z(pl40, x, y))
    out = {'E0-40': (z40, False),
           'E0-75': (float(rr.plane_z(pl75, x, y)), False) if pl75 is not None else (z40, True)}
    grid = rr._BevGrid(xyz) if len(xyz) else None
    masks = region_masks(xyz, b)
    for g, m in masks.items():
        z = np.asarray(xyz[m, 2], np.float64)
        cv = cell_values(xyz[m])
        for p in PCTS:
            if g in ('E1', 'E2'):           # the loader's own function (v4 / v5 ring; v6 radius)
                out[f'{g}-raw-p{p}'] = rr.nearby_bottom(xyz, grid, b, pl40, D_LO, E1_DHI, MIN_POINTS, float(p),
                                                        radius=E2_RADIUS if g == 'E2' else None)
            else:
                out[f'{g}-raw-p{p}'] = (float(np.percentile(z, p)), False) if len(z) >= MIN_POINTS else (z40, True)
            out[f'{g}-cell-p{p}'] = (float(np.percentile(cv, p)), False) if len(cv) >= MIN_CELLS else (z40, True)
    o = np.asarray(others, np.float64).reshape(-1, 7)
    if len(o):
        dd = np.hypot(o[:, 0] - x, o[:, 1] - y); bot = o[:, 2] - o[:, 5] / 2
        s = dd <= E5A_R
        out['E5a'] = (z40 + float(np.median(bot[s] - rr.plane_z(pl40, o[s, 0], o[s, 1]))), False) if s.any() \
            else (np.nan, True)
        for R in E5B_R:
            s = dd <= R
            wgt = 1.0 / np.maximum(dd[s], IDW_DMIN)
            out[f'E5b-R{R:g}'] = (float(np.sum(wgt * bot[s]) / np.sum(wgt)), False) if s.any() else (np.nan, True)
    else:
        out['E5a'] = (np.nan, True)
        for R in E5B_R:
            out[f'E5b-R{R:g}'] = (np.nan, True)
    sq = (np.abs(xyz[:, 0] - x) <= REF_RADIUS + 2.0) & (np.abs(xyz[:, 1] - y) <= REF_RADIUS + 2.0)
    v = rr.GroundCells(xyz[sq], CELL, CELL_MIN_PTS).local(x, y, REF_RADIUS) if sq.any() else np.zeros(0)
    out['D1v4ref'] = (float(np.percentile(v, REF_PCT)), False) if len(v) >= REF_MIN_CELLS else (z40, True)
    return out


def frame_rows(P, boxes, names):
    """Rows [r, gt bottom, n removed, est..., fb...] for every evaluated car of one frame; and the cars dropped for a None
    plane."""
    V = variant_names()
    P = np.asarray(P, np.float64)[:, :3]; boxes = np.asarray(boxes, np.float64).reshape(-1, np.shape(boxes)[-1])
    car = np.asarray(names) == 'Car'
    r = np.hypot(boxes[:, 0], boxes[:, 1]) if len(boxes) else np.zeros(0)
    ev = np.flatnonzero(car & (r > 0) & (r < R_MAX))
    rows, dropped = [], 0
    if not len(ev):
        return rows, dropped
    gridF = rr._BevGrid(P)
    for j in ev:
        b = boxes[j, :7]
        keep = leave_one_out(P, b, gridF)
        X = P[keep]
        pl40 = rr.ground_plane(X, PLANE_A)
        if pl40 is None:
            dropped += 1
            continue
        pl75 = rr.ground_plane(X, PLANE_B)
        oth = boxes[car & (np.arange(len(boxes)) != j), :7]
        est = estimate_car(X, b, oth, pl40, pl75)
        rows.append([r[j], b[2] - b[5] / 2, float((~keep).sum())] + [est[k][0] for k in V] + [float(est[k][1]) for k in V])
    return rows, dropped


class _Captured(Exception):
    pass


def run(a):
    from rr_gate import build
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plat = PLAT_ID[a.key]
    cfg, dcfg, ds = build(a.cfg, a.key)
    aug = ds.data_augmentor
    q0 = aug.data_augmentor_queue[0]
    assert getattr(q0, 'func', None) is not None and q0.func.__name__ == 'relocate_render', 'RR is not first'
    CAP = {}

    def capture(data_dict=None):
        CAP['before'] = {k: np.copy(data_dict[k]) for k in ('points', 'gt_boxes', 'gt_names')}
        raise _Captured
    aug.data_augmentor_queue = [capture] + list(aug.data_augmentor_queue[1:])
    frames = sample_frames(len(ds), a.n, a.seed)[:a.max_frames]
    print(f'[ground] {a.key}: {len(ds)} infos; {len(frames)} frames (seed {a.seed}, disjoint from the gate\'s {GATE_N}); '
          f'first {frames[:5].tolist()}', flush=True)
    rows, fr, n_drop, t0 = [], [], 0, time.time()
    names_seen = set()
    for n_done, i in enumerate(frames):
        CAP.clear()
        try:
            ds[int(i)]
        except _Captured:
            pass
        B = CAP['before']
        names_seen |= set(np.unique(B['gt_names']).tolist())
        rw, dr = frame_rows(B['points'], B['gt_boxes'], B['gt_names'])
        n_drop += dr
        rows += [[plat, int(i)] + x for x in rw]
        fr.append([plat, int(i), len(B['points']), len(B['gt_boxes']), int((np.asarray(B['gt_names']) == 'Car').sum()), len(rw), dr])
        if (n_done + 1) % 25 == 0 or n_done == len(frames) - 1:
            print(f'[ground] {n_done + 1}/{len(frames)} frames, {len(rows)} cars, {n_drop} dropped (no plane), '
                  f'{time.time() - t0:.0f} s', flush=True)
    print(f'[ground] names seen: {sorted(names_seen)}', flush=True)
    np.savez(out / f'ground_{a.key}.npz', rows=np.array(rows, np.float64), frames=np.array(fr, np.int64),
             variants=np.array(variant_names()), dropped=n_drop)
    print(f'[ground] wrote {out / f"ground_{a.key}.npz"}', flush=True)


# ------------------------------------------------------------------------------------------------------------ report
def metrics(err, fb):
    a = np.abs(err)
    return dict(n=len(err), med=float(np.median(a)) if len(a) else np.nan,
                sgn=float(np.median(err)) if len(err) else np.nan,
                gt05=float(np.mean(a > 0.5)) if len(a) else np.nan, gt1=float(np.mean(a > 1.0)) if len(a) else np.nan,
                fb=float(np.mean(fb)) if len(fb) else np.nan)


def select(table):
    """table: {name: metrics}. The smallest share > 0.5 m; within 0.01 of it, the smallest median |err| (§11.3)."""
    best = min(m['gt05'] for m in table.values())
    tied = [k for k, m in table.items() if m['gt05'] <= best + 0.01 + 1e-12]
    return min(tied, key=lambda k: (table[k]['med'], table[k]['gt05'], k))


def resolve(rows, V):
    """(estimates, fallbacks) per variant with E5's fallback resolved to the best E1-E4 variant; and that variant."""
    nv = len(V)
    est = {k: rows[:, 5 + j].copy() for j, k in enumerate(V)}
    fb = {k: rows[:, 5 + nv + j] > 0 for j, k in enumerate(V)}
    gtb = rows[:, 3]; far = rows[:, 2] >= 40.0
    t14 = {k: metrics(est[k][far] - gtb[far], fb[k][far]) for k in e14_names()}
    best14 = select(t14)
    for k in ('E5a', 'E5b-R10', 'E5b-R20'):
        miss = fb[k] | ~np.isfinite(est[k])
        est[k] = np.where(miss, est[best14], est[k])
    return est, fb, best14


def report(a):
    D = Path(a.dir)
    res = [np.load(p) for p in sorted(D.glob('ground_NUSCENES_*.npz'))]
    V = [str(x) for x in res[0]['variants']]
    rows = np.concatenate([r['rows'] for r in res]); fr = np.concatenate([r['frames'] for r in res])
    print(f'ground-height exploration (20261018_02 §11): {len(rows)} cars from {len(fr)} frames; cars dropped for a None '
          f'4-40 m plane {sum(int(r["dropped"]) for r in res)}')
    for p in (0, 1):
        f = fr[fr[:, 0] == p]; m = rows[:, 0] == p
        print(f'  {PLAT_NAME[p]}: {len(f)} frames, points per frame {f[:, 2].mean():.0f}, Car boxes {int(f[:, 4].sum())}, '
              f'evaluated {int(m.sum())}; per bin ' + ', '.join(
                  f'{lo:g}-{hi:g}: {int((m & (rows[:, 2] >= lo) & (rows[:, 2] < hi)).sum())}' for lo, hi in zip(BINS[:-1], BINS[1:]))
              + f'; points removed per car median {np.median(rows[m, 4]):.0f}')
    est, fb, best14 = resolve(rows, V)
    gtb = rows[:, 3]; far = rows[:, 2] >= 40.0
    names = [k for k in V if k != 'D1v4ref']
    pooled = {k: metrics(est[k][far] - gtb[far], fb[k][far]) for k in names}
    sel = select(pooled)
    print(f'\nbest E1-E4 variant (E5 fallback): {best14}')
    print(f'SELECTED (pooled 40-75.2 m, both platforms; smallest share > 0.5 m, ties within 1 point to the smaller median '
          f'|err|): {sel}')
    print('\n== all variants, pooled 40-75.2 m, both platforms, sorted by share > 0.5 m ==')
    print('| rank | variant | n | median abs err | signed median | > 0.5 m | > 1 m | fallback |')
    print('|---|---|---|---|---|---|---|---|')
    order = sorted(names, key=lambda k: (pooled[k]['gt05'], pooled[k]['med'], k))
    for i, k in enumerate(order):
        m = pooled[k]
        print(f"| {i + 1} | {k}{' **selected**' if k == sel else ''} | {m['n']} | {m['med']:.3f} | {m['sgn']:+.3f} | "
              f"{m['gt05']:.1%} | {m['gt1']:.1%} | {m['fb']:.1%} |")
    ref = rows[:, 5 + V.index('D1v4ref')]; reffb = rows[:, 5 + len(V) + V.index('D1v4ref')] > 0
    mr = metrics(ref[far] - gtb[far], reffb[far]); dm = far & ~reffb
    md = metrics(ref[dm] - gtb[dm], reffb[dm])
    print('\n== D1 v4 reference as an estimator (descriptive, not in the selection), pooled 40-75.2 m ==')
    print(f"with the 4-40 m plane where undefined: n {mr['n']}, median abs {mr['med']:.3f}, signed {mr['sgn']:+.3f}, "
          f"> 0.5 m {mr['gt05']:.1%}, > 1 m {mr['gt1']:.1%}, undefined {mr['fb']:.1%}")
    print(f"defined cars only: n {md['n']}, median abs {md['med']:.3f}, signed {md['sgn']:+.3f}, > 0.5 m {md['gt05']:.1%}, "
          f"> 1 m {md['gt1']:.1%}")
    print('\n== per platform x range bin (n / median abs / signed / > 0.5 m / > 1 m / fallback) ==')
    for k in names + ['D1v4ref', 'D1v4ref-defined']:
        print(f'-- {k}')
        for p in (0, 1):
            cells = []
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                m = (rows[:, 0] == p) & (rows[:, 2] >= lo) & (rows[:, 2] < hi)
                if k == 'D1v4ref':
                    e, f = ref[m] - gtb[m], reffb[m]
                elif k == 'D1v4ref-defined':
                    mm = m & ~reffb; e, f = ref[mm] - gtb[mm], reffb[mm]
                else:
                    e, f = est[k][m] - gtb[m], fb[k][m]
                q = metrics(e, f)
                cells.append(f"{lo:g}-{hi:g}: {q['n']} / {q['med']:.3f} / {q['sgn']:+.3f} / {q['gt05']:.1%} / "
                             f"{q['gt1']:.1%} / {q['fb']:.1%}")
            print(f'  {PLAT_NAME[p]}: ' + '; '.join(cells))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--cfg', required=True); r.add_argument('--key', required=True); r.add_argument('--out', required=True)
    r.add_argument('--n', type=int, default=N_SAMPLE); r.add_argument('--seed', type=int, default=0)
    r.add_argument('--max_frames', type=int, default=None)
    q = sp.add_parser('report'); q.add_argument('--dir', required=True)
    a = ap.parse_args()
    {'run': run, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()
