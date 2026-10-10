"""Lyft LIDAR_TOP ring recovery under motion compensation (experiments_md 20261010_07). CPU; run as a Slurm job.

  explore  - per platform: sweep time lags (rotation period), scan direction, and the azimuth where one laser's sweep
             ends and the next begins, read on STANDSTILL frames (no compensation distortion there); then the
             reference-time fraction of the compensation, fitted on MOVING frames by the sharpness of the de-skewed
             elevation histogram. Label-free: Lyft points and poses only.

Usage (from ST3D/tools, inside the container):
  python3 analysis/lyft_deskew_rings.py explore --out /home/koyama/data/lyft_rings/explore.json
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from pcdet.datasets.lyft import lyft_rings as lr  # noqa: E402

ROOT = '/home/koyama/data/level5-3d-object-detection/trainval/'
H64 = ('host-a101', 'host-a102')
N_LASERS = {'40': 40, '64': 64}


def platform(info):
    return '64' if info['lidar_path'].split('/')[-1].split('_')[0] in H64 else '40'


def load(info):
    return np.fromfile(ROOT + info['lidar_path'], np.float32).reshape(-1, 5)[:, :3].astype(np.float64)


def speed(info):
    m = lr.ego_motion(info)
    return np.nan if m is None else float(np.linalg.norm(m[0][:2]))


def switch_azimuths(xyz, min_range=12.0, half=30, sep=200, drop_deg=0.1):
    """Azimuths (deg) of the strongest persistent elevation drops along the stored order, far points only."""
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    far = np.nonzero(r > min_range)[0]
    if len(far) < 4 * half:
        return np.zeros(0)
    e = lr.elevation_deg(xyz[far])
    med = np.median(np.lib.stride_tricks.sliding_window_view(e, half), axis=1)
    drop = np.zeros(len(e))
    drop[half:len(e) - half + 1] = med[:len(e) - 2 * half + 1] - med[half:]
    picked = []
    for j in np.argsort(-drop):
        if drop[j] < drop_deg:
            break
        if all(abs(j - q) > sep for q in picked):
            picked.append(j)
    return lr.azimuth_deg(xyz[far[sorted(picked)]])


def circ_median(a_deg):
    """Circular median-ish: the median of angles unwrapped about their circular mean."""
    if len(a_deg) == 0:
        return np.nan
    m = np.degrees(np.arctan2(np.mean(np.sin(np.radians(a_deg))), np.mean(np.cos(np.radians(a_deg)))))
    return float(m + np.median((np.asarray(a_deg) - m + 180.0) % 360.0 - 180.0))


def rows_by_crossing(az, sign, az_start):
    """Number of stored-order segments when a new one starts each time the phase about az_start wraps."""
    ph = ((az - az_start) * sign) % 360.0
    return int(np.sum(np.diff(ph) < -180.0)) + 1


def lowest_laser_swing(xyz, n_lasers):
    """G1's metric (20261010_07 §2): the last 1/n of the stored order, points 3-15 m away; p95 - p5 of the medians of
    10-deg azimuth bins of their elevation."""
    tail = np.arange(len(xyz)) >= len(xyz) - len(xyz) // n_lasers
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    m = tail & (r > 3) & (r < 15)
    if m.sum() < 200:
        return np.nan
    el, az = lr.elevation_deg(xyz[m]), lr.azimuth_deg(xyz[m])
    b = np.floor((az + 180) / 10).astype(int)
    med = np.array([np.median(el[b == k]) for k in np.unique(b) if np.sum(b == k) > 5])
    return float(np.percentile(med, 95) - np.percentile(med, 5))


def sharpness(xyz, lo=3.0, hi=30.0, bin_deg=0.02):
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    el = lr.elevation_deg(xyz[(r > lo) & (r < hi)])
    c, _ = np.histogram(el, bins=np.arange(-30, 20, bin_deg))
    return float(np.sum(c.astype(np.float64) ** 2) / max(np.sum(c), 1) ** 2)


def explore(args):
    t0 = time.time()
    infos = pickle.load(open(ROOT + 'lyft_infos_train.pkl', 'rb'))
    rng = np.random.default_rng(args.seed)
    out = {}
    for plat in ('40', '64'):
        sel = [i for i in infos if platform(i) == plat]
        res = {}
        lags = np.array([s['time_lag'] for i in sel[::20] for s in (i.get('sweeps') or [])[:1] if s.get('time_lag')])
        res['sweep0_time_lag_s'] = {'p5': float(np.percentile(lags, 5)), 'median': float(np.median(lags)),
                                    'p95': float(np.percentile(lags, 95)), 'n': int(len(lags))}
        sp = np.array([speed(i) for i in sel])
        res['speed_share'] = {'<0.3': float(np.mean(sp < 0.3)), '>8': float(np.mean(sp > 8)),
                              'no_sweep': float(np.mean(~np.isfinite(sp)))}
        # --- standstill frames: frame structure without distortion
        still = [sel[k] for k in np.nonzero(sp < 0.3)[0]]
        still = [still[k] for k in rng.choice(len(still), min(args.n_still, len(still)), replace=False)]
        dirs, sw_az, first_az = [], [], []
        for inf in still:
            p = load(inf)
            az = lr.azimuth_deg(p)
            dirs.append(lr.scan_direction(az))
            a = switch_azimuths(p)
            sw_az.append(circ_median(a))
            first_az.append(float(az[0]))
        sw_az = np.array(sw_az)
        az_start = circ_median(sw_az[np.isfinite(sw_az)])
        spread = (sw_az - az_start + 180.0) % 360.0 - 180.0
        rows = [rows_by_crossing(lr.azimuth_deg(load(inf)), d, az_start) for inf, d in zip(still, dirs)]
        res['standstill'] = {
            'frames': len(still), 'scan_direction': {str(k): int(v) for k, v in zip(*np.unique(dirs, return_counts=True))},
            'switch_az_platform_deg': az_start,
            'switch_az_frame_minus_platform_deg': {'p5': float(np.nanpercentile(spread, 5)),
                                                   'median': float(np.nanmedian(spread)),
                                                   'p95': float(np.nanpercentile(spread, 95))},
            'first_point_az_deg_p5_p50_p95': [float(x) for x in np.percentile(first_az, [5, 50, 95])],
            'rows_by_crossing': {str(k): int(v) for k, v in zip(*np.unique(rows, return_counts=True))},
            'swing_deg_median': float(np.nanmedian([lowest_laser_swing(load(i), N_LASERS[plat]) for i in still]))}
        print(plat, 'standstill', json.dumps(res['standstill']), flush=True)
        # --- moving frames: fit the reference-time fraction
        mov = [sel[k] for k in np.nonzero(sp > 8)[0]]
        mov = [mov[k] for k in rng.choice(len(mov), min(args.n_move, len(mov)), replace=False)]
        periods = sorted({round(float(res['sweep0_time_lag_s']['median']), 3), 0.1, 0.2})
        fracs = np.round(np.arange(-0.5, 1.51, 0.05), 3)
        grid = {}
        swing_before = []
        frames = []
        for inf in mov:
            p = load(inf)
            az = lr.azimuth_deg(p)
            frames.append((inf, p, az, lr.scan_direction(az), lr.ego_motion(inf)))
            swing_before.append(lowest_laser_swing(p, N_LASERS[plat]))
        for T in periods:
            for f in fracs:
                sc = []
                for inf, p, az, d, (v, w) in frames:
                    dt = lr.firing_offset(az, d, az_start, T, f)
                    sc.append(sharpness(lr.deskew(p, v, w, dt)))
                grid['%.3f_%.2f' % (T, f)] = float(np.mean(sc))
        best = max(grid, key=grid.get)
        T_best, f_best = (float(x) for x in best.split('_'))
        swing_after = []
        for inf, p, az, d, (v, w) in frames:
            q = lr.deskew(p, v, w, lr.firing_offset(az, d, az_start, T_best, f_best))
            swing_after.append(lowest_laser_swing(q, N_LASERS[plat]))
        res['moving_fit'] = {'frames': len(frames), 'periods_tried': periods, 'best_period_s': T_best,
                             'best_ref_fraction': f_best, 'sharpness_best': grid[best],
                             'sharpness_no_deskew': float(np.mean([sharpness(fr[1]) for fr in frames])),
                             'swing_deg_median_before': float(np.nanmedian(swing_before)),
                             'swing_deg_median_after': float(np.nanmedian(swing_after)),
                             'grid': grid}
        print(plat, 'moving', json.dumps({k: v for k, v in res['moving_fit'].items() if k != 'grid'}), flush=True)
        out[plat] = res
    out['seconds'] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)
    print('wrote', args.out, '%.0f s' % out['seconds'])


def purity(q, ring, lev, min_range=20.0):
    """G2: share of points >= min_range m away whose de-skewed elevation is nearest their own ring's level."""
    r = np.hypot(q[:, 0], q[:, 1])
    m = r >= min_range
    if m.sum() == 0 or len(lev) == 0:
        return np.nan
    el = lr.elevation_deg(q[m])
    return float(np.mean(np.argmin(np.abs(el[:, None] - lev[None, :]), axis=1) == ring[m]))


def frame_rings(p, info, prm, iters, start_rule, yaw_only=False, unwrap=False):
    """De-skew + rings for one stored scan under the parameters `prm` (from explore) and a variant. `unwrap` runs
    lr.deskew_and_rings (per-ring phase unwrap, added after the synthetic unit test found the sweep-start case)."""
    az = lr.azimuth_deg(p)
    sign = lr.scan_direction(az)
    a0 = prm['az_start'] if start_rule == 'platform' else float(az[0]) - sign * 0.5
    if unwrap:
        return lr.deskew_and_rings(p, lr.ego_motion(info, yaw_only=yaw_only), sign, a0, prm['period'], prm['ref'],
                                   iters=iters)
    q = lr.deskew_frame(p, lr.ego_motion(info, yaw_only=yaw_only), sign, a0, prm['period'], prm['ref'], iters=iters)
    ring, lev = lr.rings_from_deskewed(q, sign, a0)
    return q, ring, lev


def params_from_explore(path):
    ex = json.load(open(path))
    return {plat: {'az_start': ex[plat]['standstill']['switch_az_platform_deg'],
                   'period': ex[plat]['moving_fit']['best_period_s'],
                   'ref': ex[plat]['moving_fit']['best_ref_fraction']} for plat in ('40', '64')}


def method(args):
    """Variants on frames drawn with the EXPLORE seed (the gate draws fresh ones): de-skew iterations and the rule for
    the sweep start (platform constant from explore vs the frame's first point)."""
    infos = pickle.load(open(ROOT + 'lyft_infos_train.pkl', 'rb'))
    prm = params_from_explore(args.explore)
    rng = np.random.default_rng(args.seed)
    out = {'params': prm}
    for plat in ('40', '64'):
        sel = [i for i in infos if platform(i) == plat]
        sp = np.array([speed(i) for i in sel])
        pick = list(rng.choice(np.nonzero(sp < 0.3)[0], args.n_still, replace=False)) + \
            list(rng.choice(np.nonzero(sp > 8)[0], args.n_move, replace=False))
        frames = [(sel[k], load(sel[k]), sp[k]) for k in pick]
        res = {}
        for iters, rot in ((2, 'yaw'), (2, 'full')):
            for rule in ('platform', 'first_point'):
                rows, pur, sw, ts = [], [], [], []
                for inf, p, v in frames:
                    t0 = time.time()
                    q, ring, lev = frame_rings(p, inf, prm[plat], iters, rule, yaw_only=(rot == 'yaw'))
                    ts.append(time.time() - t0)
                    rows.append(len(lev))
                    pur.append(purity(q, ring, lev))
                    if v > 8:
                        sw.append(lowest_laser_swing(q, N_LASERS[plat]))
                rows = np.array(rows)
                key = 'iters%d_%s_%s' % (iters, rot, rule)
                res[key] = {'rows_exact_share': float(np.mean(rows == N_LASERS[plat])),
                            'rows_min_max': [int(rows.min()), int(rows.max())],
                            'purity_r20_mean': float(np.nanmean(pur)), 'purity_r20_p5': float(np.nanpercentile(pur, 5)),
                            'swing_moving_median': float(np.nanmedian(sw)), 'sec_per_frame': float(np.mean(ts))}
                print(plat, key, json.dumps(res[key]), flush=True)
        out[plat] = res
    json.dump(out, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


def startscan(args):
    """Diagnostic (64-beam): which sweep-start azimuth fits each frame. For a grid of starts, de-skew + rings + far
    purity; report the best start per frame beside the platform constant, with host, speed and first-point azimuth."""
    infos = pickle.load(open(ROOT + 'lyft_infos_train.pkl', 'rb'))
    prm = params_from_explore(args.explore)
    rng = np.random.default_rng(args.seed)
    sel = [i for i in infos if platform(i) == args.plat]
    sp = np.array([speed(i) for i in sel])
    pick = list(rng.choice(np.nonzero(sp > 8)[0], args.n_move, replace=False))
    grid = np.arange(-180.0, 180.0, args.step)
    rows = []
    for k in pick:
        inf = sel[k]
        p = load(inf)
        az = lr.azimuth_deg(p)
        sign = lr.scan_direction(az)
        best = None
        for a0 in grid:
            q = lr.deskew_frame(p, lr.ego_motion(inf), sign, a0, prm[args.plat]['period'], prm[args.plat]['ref'], iters=2)
            ring, lev = lr.rings_from_deskewed(q, sign, a0)
            pu = purity(q, ring, lev)
            if best is None or pu > best[0]:
                best = (pu, float(a0), len(lev))
        q = lr.deskew_frame(p, lr.ego_motion(inf), sign, prm[args.plat]['az_start'], prm[args.plat]['period'],
                            prm[args.plat]['ref'], iters=2)
        ring, lev = lr.rings_from_deskewed(q, sign, prm[args.plat]['az_start'])
        row = {'file': inf['lidar_path'].split('/')[-1], 'speed': float(sp[k]), 'sign': sign, 'first_az': float(az[0]),
               'platform_purity': purity(q, ring, lev), 'platform_rings': len(lev),
               'best_az': best[1], 'best_purity': best[0], 'best_rings': best[2]}
        rows.append(row)
        print('SCAN', json.dumps(row), flush=True)
    json.dump(rows, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


def gate(args):
    """The pre-declared CPU gate G1-G4 (20261010_07 §4) on FRESH frames (seed differs from explore / method), with the
    method frozen by --iters / --start_rule. Also records what RING_PATTERN keeps (rings, share, lattice spacing)."""
    infos = pickle.load(open(ROOT + 'lyft_infos_train.pkl', 'rb'))
    prm = params_from_explore(args.explore)
    rng = np.random.default_rng(args.seed)
    out = {'params': prm, 'iters': args.iters, 'start_rule': args.start_rule, 'seed': args.seed, 'unwrap': args.unwrap}
    for plat in ('40', '64'):
        sel = [i for i in infos if platform(i) == plat]
        pick = rng.choice(len(sel), args.n_gate, replace=False)
        n_ok = n_far = 0
        rows_exact, sw, ts, kept_rings, kept_share, kept_gap, no_motion = [], [], [], [], [], [], 0
        for k in pick:
            inf = sel[k]
            p = load(inf)
            v = speed(inf)
            no_motion += int(not np.isfinite(v))
            t0 = time.time()
            rule = {'40': 'first_point', '64': 'platform'}[plat] if args.start_rule == 'frozen' else args.start_rule
            q, ring, lev = frame_rings(p, inf, prm[plat], args.iters, rule, unwrap=args.unwrap)
            keep, kr = lr.ring_pattern_mask(p, ring, lev, 1.33, 0.332, rng.random(), rng.random())
            ts.append(time.time() - t0)
            r = np.hypot(q[:, 0], q[:, 1]); m = r >= 20.0
            if m.any():
                el = lr.elevation_deg(q[m])
                n_ok += int(np.sum(np.argmin(np.abs(el[:, None] - lev[None, :]), axis=1) == ring[m]))
                n_far += int(m.sum())
            rows_exact.append(len(lev) == N_LASERS[plat])
            if np.isfinite(v) and v > 8:
                sw.append(lowest_laser_swing(q, N_LASERS[plat]))
            kept_rings.append(len(kr)); kept_share.append(float(keep.mean()))
            if len(kr) > 1:
                kept_gap.append(float(np.median(np.abs(np.diff(np.sort(lev[kr]))))))
        g = {'frames': int(args.n_gate), 'frames_without_ego_motion': no_motion,
             'G1_swing_moving_median_deg': float(np.nanmedian(sw)), 'G1_frames': len(sw),
             'G2_purity_r20': n_ok / max(n_far, 1), 'G3_exact_ring_count_share': float(np.mean(rows_exact)),
             'G4_sec_per_frame_mean': float(np.mean(ts)), 'G4_sec_per_frame_p95': float(np.percentile(ts, 95)),
             'pattern_kept_rings': {str(a): int(b) for a, b in zip(*np.unique(kept_rings, return_counts=True))},
             'pattern_kept_point_share_mean': float(np.mean(kept_share)),
             'pattern_kept_ring_gap_deg_median': float(np.median(kept_gap)) if kept_gap else None}
        g['pass'] = {'G1': g['G1_swing_moving_median_deg'] <= 0.2, 'G2': g['G2_purity_r20'] >= 0.98,
                     'G3': g['G3_exact_ring_count_share'] >= 0.95, 'G4': g['G4_sec_per_frame_mean'] <= 0.1}
        out[plat] = g
        print(plat, 'GATE', json.dumps(g), flush=True)
    json.dump(out, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['explore', 'method', 'gate', 'startscan'])
    ap.add_argument('--iters', type=int, default=2)
    ap.add_argument('--start_rule', choices=['platform', 'first_point', 'frozen'], default='frozen')
    ap.add_argument('--n_gate', type=int, default=200)
    ap.add_argument('--plat', default='64')
    ap.add_argument('--unwrap', action='store_true')
    ap.add_argument('--step', type=float, default=5.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--explore', default='/home/koyama/data/lyft_rings/explore_20261010.json')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--n_still', type=int, default=40)
    ap.add_argument('--n_move', type=int, default=40)
    args = ap.parse_args()
    {'explore': explore, 'method': method, 'gate': gate, 'startscan': startscan}[args.mode](args)
