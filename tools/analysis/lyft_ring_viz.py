"""Visual check of Lyft LIDAR_TOP ring recovery (experiments_md 20261022_05). CPU; run as a Slurm job.

The user asked (2026-10-22) to SEE the ring recovery that the 2026-10-10 CPU gate failed (20261010_07 §4b) and to judge
it qualitatively. This tool only draws figures and records per-frame facts; it changes no method and decides nothing.

Code on record, imported UNCHANGED: pcdet/datasets/lyft/lyft_rings.py and tools/analysis/lyft_deskew_rings.py (ST3D
6ac42f8), with gate run 2's settings: the explore parameters (rotation 0.1 s, reference at the end of the rotation), two
de-skew passes, full angular velocity, sweep start at the frame's first point (40-beam) or at -1.65 deg (64-beam), and
the per-ring phase unwrap (`lyft_rings.deskew_and_rings`).

  frames - replay the gate's frame draw (seed 202: `rng.choice` per platform, then two `rng.random()` per frame for the
           RING_PATTERN phases, in the order of lyft_deskew_rings.gate), recompute run 1 (no unwrap) and run 2 (unwrap)
           on every frame, check the replay against the gate JSONs, and record per frame: sample token, file, host,
           speed, ring counts, far purity, what RING_PATTERN keeps. Then a two-level DIAGNOSTIC of every frame (for the
           figures only; neither level is part of any method):
           level 1 (no template): a recovered ring with fewer than MIN_BLOCK points is a FRAGMENT (one laser sweep holds
             ~1,700-1,800 points); each fragment is joined to the full ring next to it in stored order whose end is
             nearer in azimuth, and RING_PATTERN is rerun on the joined rings with the gate's phases;
           level 2 (template reading): each point gets the nearest level of a per-host laser-elevation template (median
             ring elevations of the run-2 exact-count frames) where that label is unambiguous; stored-order cuts where
             that label changes are compared with the level-1 ring cuts (missing / extra / displaced).
  plots  - the figures, from the frames JSON.
  all    - both.
  table  - the per-frame table of every wrong-count frame (the columns of 20261022_05 §4.4) for both platforms, the
           fragment facts, and, line by line of RING_PATTERN's lattice, why the recovered and the joined rings keep
           different rings (`pattern_lines`). Reads the frames JSON, recomputes each wrong-count frame's rings; writes
           wrong_count_table_20261023.json and wrong_count_table_<platform>.md. Changes no figure.

--align_platforms (opt-in, user 2026-10-23: 「60ビームの可視化方法を40ビームと合わせてくれないか」, 20261022_05 §6): figure 4 draws
EVERY wrong-count frame on both platforms (default: all on 40-beam, six on 64-beam), and figure 1 draws the moving
frame at the 10th percentile of far purity on both platforms (default: 64-beam only). Without it the output is unchanged.

All angles are about the SENSOR: the raw LIDAR_TOP .bin is read directly (the loader and its SHIFT_COOR are not used);
after de-skew, about the sensor at each point's firing time. Lyft's LIDAR_TOP frame is yawed 176-179 deg in the car
frame (calibrated_sensor), so the car's forward direction sits at a sensor azimuth near +-180 deg; nuScenes' LIDAR_TOP
is yawed -90 deg (forward at +90). Figures in the sensor azimuth mark the car's forward direction; the BEV and the
Lyft / nuScenes comparison are rotated about the sensor's z axis into the car's heading (yaw only, elevations unchanged),
using each info's `ref_from_car`.

Usage (from ST3D/tools, inside the container):
  python3 analysis/lyft_ring_viz.py all --out_dir /home/koyama/data/lyft_rings/viz_20261022
  python3 analysis/lyft_ring_viz.py plots --align_platforms --out_dir /home/koyama/data/lyft_rings/viz_20261022
  python3 analysis/lyft_ring_viz.py table --out_dir /home/koyama/data/lyft_rings/viz_20261022
"""
import argparse
import io
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
from pcdet.datasets.lyft import lyft_rings as lr  # noqa: E402
import lyft_deskew_rings as ldr  # noqa: E402

N = ldr.N_LASERS                       # {'40': 40, '64': 64}
SPACING, AZ_RES = 1.33, 0.332          # RING_PATTERN as in the gate (20261010_07 §3 item 5)
START_RULE = {'40': 'first_point', '64': 'platform'}
GATE_SEED, N_GATE = 202, 200
RINGS_DIR = '/home/koyama/data/lyft_rings/'
NUSC_ROOT = '/home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval/'
OFFSET_DEG_M = 1.15                    # a 2 cm laser offset seen from r metres moves elevation by ~1.15 / r deg
MIN_BLOCK = 30                         # points: below this a recovered ring is a fragment; also the template block size

# 8-hue categorical order (dataviz skill palette, light mode); ring IDs cycle through it
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID, MUTED = '#1f1f1e', '#5b5b57', '#e6e5e0', '#c9c8c2'
AZ_LABEL = 'azimuth in the LIDAR_TOP frame, atan2(y, x) (deg)'
ZOOM_AZ = (-30.0, 30.0)


def host(info):
    return info['lidar_path'].split('/')[-1].split('_')[0]


def recover(p, info, prm, plat, unwrap=True):
    """The gate's recovery for one stored scan: (de-skewed xyz, ring per point, ring elevations)."""
    return ldr.frame_rings(p, info, prm[plat], 2, START_RULE[plat], unwrap=unwrap)


def sweep_start(p, prm, plat):
    """(scan direction, sweep-start azimuth) exactly as lyft_deskew_rings.frame_rings sets them."""
    az = lr.azimuth_deg(p)
    sign = lr.scan_direction(az)
    a0 = prm[plat]['az_start'] if START_RULE[plat] == 'platform' else float(az[0]) - sign * 0.5
    return sign, a0


def fwd_az(info):
    """Azimuth (deg) of the car's forward axis in the sensor frame, from `ref_from_car` (car -> sensor)."""
    R = np.asarray(info['ref_from_car'], dtype=np.float64)[:3, :3]
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def to_car_yaw(xyz, az_f):
    """Rotate about the sensor's z so that the car's forward axis is +x (yaw only; elevations unchanged)."""
    a = -np.radians(az_f)
    c, s = np.cos(a), np.sin(a)
    out = np.array(xyz, dtype=np.float64, copy=True)
    out[:, 0] = c * xyz[:, 0] - s * xyz[:, 1]
    out[:, 1] = s * xyz[:, 0] + c * xyz[:, 1]
    return out


def wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def fnum(x):
    return None if x is None or not np.isfinite(x) else float(x)


# ----------------------------------------------------------------------------------------------- diagnosis
def ring_bounds(ring):
    """Rings are contiguous stored-order blocks with consecutive IDs (lyft_rings.rings_from_deskewed)."""
    cuts = np.r_[0, np.nonzero(np.diff(ring))[0] + 1, len(ring)]
    assert len(cuts) == int(ring.max()) + 2 and np.all(ring[cuts[:-1]] == np.arange(int(ring.max()) + 1))
    return cuts[:-1], cuts[1:] - 1, cuts[1:] - cuts[:-1]


def merge_fragments(ring, az, el):
    """Level 1: rings with < MIN_BLOCK points join the neighbouring full ring whose end is nearer in azimuth.
    Returns (joined ring per point, recovered -> joined ID map, fragment records)."""
    first, last, size = ring_bounds(ring)
    full = np.nonzero(size >= MIN_BLOCK)[0]
    target = np.arange(len(size))
    frags = []
    for i in np.nonzero(size < MIN_BLOCK)[0]:
        prev, nxt = full[full < i], full[full > i]
        dp = float(abs(wrap(az[first[i]] - az[last[prev[-1]]]))) if len(prev) else np.inf
        dn = float(abs(wrap(az[last[i]] - az[first[nxt[0]]]))) if len(nxt) else np.inf
        to = int(prev[-1]) if dp <= dn else int(nxt[0])
        target[i] = to
        idx = np.arange(first[i], last[i] + 1)
        frags.append({'ring': int(i), 'points': int(size[i]), 'stored_index': int(first[i]),
                      'azimuth_deg': float(az[first[i]]), 'elevation_deg': float(np.median(el[idx])),
                      'joins': 'preceding' if to < i else 'following', 'joins_ring': to,
                      'az_gap_preceding_deg': fnum(dp), 'az_gap_following_deg': fnum(dn)})
    cm = np.searchsorted(full, target)
    for f in frags:
        f['joins_joined_id'] = int(cm[f['joins_ring']])
    return cm[ring], cm, frags


def laser_labels(q, T):
    """Nearest template level per point, and whether that label is unambiguous: the point lies within a quarter of the
    local level gap, at a range where a 2 cm laser offset stays under that quarter gap."""
    el, r = lr.elevation_deg(q), np.hypot(q[:, 0], q[:, 1])
    lab = np.argmin(np.abs(el[:, None] - T[None, :]), axis=1)
    d = np.abs(np.diff(T))
    gap = np.minimum(np.r_[np.inf, d], np.r_[d, np.inf])
    quarter = gap[lab] / 4.0
    conf = (np.abs(el - T[lab]) < quarter) & (r >= np.maximum(3.0, OFFSET_DEG_M / quarter))
    return lab, conf


def corrected_lasers(ring, lev, lab, conf, T):
    """Level 2: laser per point from the template labels, cut only inside rings: in each ring every laser with
    >= MIN_BLOCK unambiguous points and >= 5% of them is a candidate; consecutive candidates are separated at the
    stored-order change point, accepted only if both sides are >= 80% their own laser."""
    corr = np.empty(len(ring), dtype=np.int64)
    rings = []
    for i in range(int(ring.max()) + 1):
        idx = np.nonzero(ring == i)[0]
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        c, L = conf[lo:hi], lab[lo:hi]
        nc = int(c.sum())
        cnt = np.bincount(L[c], minlength=len(T))
        dom = [int(k) for k in np.nonzero((cnt >= MIN_BLOCK) & (cnt >= 0.05 * max(nc, 1)))[0]]
        main = int(np.argmax(cnt)) if nc else int(np.argmin(np.abs(T - lev[i])))
        if main not in dom:
            dom.append(main)
        rel = np.arange(hi - lo)
        pos = {k: float(np.median(rel[c & (L == k)])) if np.any(c & (L == k)) else 0.0 for k in dom}
        order = sorted(dom, key=lambda k: pos[k])
        segs, s0, cur = [], 0, order[0]
        for b in order[1:]:
            sub = np.nonzero(c & ((L == cur) | (L == b)))[0]
            sub = sub[sub >= s0]
            if len(sub) < 2:
                continue
            isb = (L[sub] == b).astype(np.int64)
            ca = np.r_[0, np.cumsum(1 - isb)]
            cb = np.r_[np.cumsum(isb[::-1])[::-1], 0]
            j = int(np.argmax(ca + cb))
            pa, pb = ca[j] / max(j, 1), cb[j] / max(len(sub) - j, 1)
            if ca[j] >= MIN_BLOCK and cb[j] >= MIN_BLOCK and pa >= 0.8 and pb >= 0.8:
                segs.append((cur, s0, int(sub[j])))
                s0, cur = int(sub[j]), b
        segs.append((cur, s0, hi - lo))
        for k, a_, b_ in segs:
            corr[lo + a_:lo + b_] = k
        rings.append({'ring': i, 'n': hi - lo, 'n_conf': nc, 'level': float(lev[i]), 'main': main,
                      'segments': [[int(k), lo + int(a_), lo + int(b_)] for k, a_, b_ in segs if b_ > a_]})
    return corr, rings


def template_events(ring, corr, az, plat):
    """Compare the ring cuts with the template-label cuts: matched, missing, extra or displaced."""
    n = len(ring)
    rec_cuts = np.nonzero(np.diff(ring) != 0)[0] + 1
    cor_cuts = np.nonzero(np.diff(corr) != 0)[0] + 1
    tol, sweep = max(5, int(round(n / N[plat] / 36.0))), n / N[plat]
    used, matched, missing = set(), [], []
    for c in cor_cuts:
        if len(rec_cuts):
            j = int(np.argmin(np.abs(rec_cuts - c)))
            if abs(int(rec_cuts[j]) - int(c)) <= tol and j not in used:
                used.add(j)
                matched.append((int(c), int(rec_cuts[j])))
                continue
        missing.append(int(c))
    extra = [int(rec_cuts[j]) for j in range(len(rec_cuts)) if j not in used]
    events = []
    for c in missing:
        near = [e for e in extra if abs(e - c) < sweep and not any(min(e, c) < m[1] < max(e, c) for m in matched)]
        if near:
            e = min(near, key=lambda e: abs(e - c))
            extra.remove(e)
            events.append({'type': 'displaced', 'stored_index': c, 'recovered_cut': e, 'shift_points': e - c,
                           'lasers': [int(corr[c - 1]), int(corr[c])], 'rings': [int(ring[e - 1]), int(ring[e])],
                           'azimuth_deg': float(az[c]), 'recovered_cut_azimuth_deg': float(az[e])})
        else:
            events.append({'type': 'missing', 'stored_index': c, 'lasers': [int(corr[c - 1]), int(corr[c])],
                           'rings': [int(ring[c])], 'azimuth_deg': float(az[c])})
    for e in extra:
        events.append({'type': 'extra', 'stored_index': e, 'lasers': [int(corr[e])],
                       'rings': [int(ring[e - 1]), int(ring[e])], 'azimuth_deg': float(az[e])})
    events.sort(key=lambda d: d['stored_index'])
    return events, rec_cuts, cor_cuts, matched


def diagnose(p, q, ring, lev, T, plat, pv, ph):
    """Per-frame diagnostic (see module doc). Returns (summary dict, arrays dict)."""
    az, el = lr.azimuth_deg(q), lr.elevation_deg(q)
    ring_j, cm, frags = merge_fragments(ring, az, el)
    nj = int(ring_j.max()) + 1
    lev_j = np.array([np.median(el[ring_j == k]) for k in range(nj)])
    keep_r, kr_r = lr.ring_pattern_mask(p, ring, lev, SPACING, AZ_RES, pv, ph)
    keep_j, kr_j = lr.ring_pattern_mask(p, ring_j, lev_j, SPACING, AZ_RES, pv, ph)
    fset = {f['ring'] for f in frags}
    kept_r = [{'ring': int(i), 'joined_id': int(cm[i]), 'fragment': int(i) in fset,
               'points_kept': int(np.sum(keep_r & (ring == i)))} for i in kr_r]
    kept_j = [{'joined_id': int(k), 'points_kept': int(np.sum(keep_j & (ring_j == k)))} for k in kr_j]
    same = (not any(d['fragment'] for d in kept_r)) and \
        sorted(d['joined_id'] for d in kept_r) == [d['joined_id'] for d in kept_j]
    # level 2 on the joined rings
    lab, conf = laser_labels(q, T)
    corr, rings_info = corrected_lasers(ring_j, lev_j, lab, conf, T)
    events, j_cuts, cor_cuts, matched = template_events(ring_j, corr, az, plat)
    present = np.unique(corr)
    ring_c = np.searchsorted(present, corr)
    lev_c = np.array([np.median(el[ring_c == k]) for k in range(len(present))])
    keep_c, kr_c = lr.ring_pattern_mask(p, ring_c, lev_c, SPACING, AZ_RES, pv, ph)
    kept_lj = sorted({s[0] for k in kr_j for s in rings_info[int(k)]['segments']})
    kept_lc = sorted(int(x) for x in present[kr_c])
    seq = [int(corr[0])] + [int(corr[c]) for c in cor_cuts]
    tmpl = {'segments': int(len(cor_cuts) + 1), 'sequence_is_0_to_N': bool(seq == list(range(N[plat]))),
            'absent_lasers': [int(k) for k in range(len(T)) if k not in set(present.tolist())],
            'unambiguous_share': float(conf.mean()), 'events': events,
            'n_missing': sum(e['type'] == 'missing' for e in events),
            'n_extra': sum(e['type'] == 'extra' for e in events),
            'n_displaced': sum(e['type'] == 'displaced' for e in events),
            'kept_lasers_joined': kept_lj, 'kept_lasers_corrected': kept_lc,
            'kept_same': bool(kept_lj == kept_lc and all(len(rings_info[int(k)]['segments']) == 1 for k in kr_j)),
            'rings': rings_info if (nj != N[plat] or events) else None}
    summ = {'recovered_rings': int(len(lev)), 'n_fragments': len(frags), 'fragments': frags,
            'rings_after_join': nj, 'kept_recovered': kept_r, 'kept_after_join': kept_j, 'kept_same': bool(same),
            'kept_points_recovered': int(keep_r.sum()), 'kept_points_after_join': int(keep_j.sum()),
            'kept_points_differing': int(np.sum(keep_r != keep_j)), 'template': tmpl}
    arrays = {'ring_j': ring_j, 'cm': cm, 'corr': corr, 'keep_r': keep_r, 'keep_j': keep_j,
              'rec_cuts': np.nonzero(np.diff(ring) != 0)[0] + 1, 'cor_cuts': cor_cuts}
    return summ, arrays


# ----------------------------------------------------------------------------------------------- frames stage
def stage_frames(args):
    t0 = time.time()
    infos = pickle.load(open(ldr.ROOT + 'lyft_infos_train.pkl', 'rb'))
    prm = ldr.params_from_explore(args.explore)
    g = {1: json.load(open(RINGS_DIR + 'gate_20261010.json')), 2: json.load(open(RINGS_DIR + 'gate2_unwrap_20261010.json'))}
    rng = np.random.default_rng(GATE_SEED)
    out = {'params': prm, 'seed': GATE_SEED, 'spacing_deg': SPACING, 'az_res_deg': AZ_RES, 'min_block': MIN_BLOCK,
           'platforms': {}}
    for plat in ('40', '64'):
        sel = [i for i in infos if ldr.platform(i) == plat]
        pick = rng.choice(len(sel), N_GATE, replace=False)
        recs, agg, sw = [], {1: [0, 0], 2: [0, 0]}, []
        for k in pick:
            inf = sel[k]
            p = ldr.load(inf)
            v = ldr.speed(inf)
            res = {run: recover(p, inf, prm, plat, unwrap) for run, unwrap in ((1, False), (2, True))}
            pv, ph = float(rng.random()), float(rng.random())     # same two draws per frame as the gate
            rec = {'sel_index': int(k), 'token': inf['token'], 'file': inf['lidar_path'].split('/')[-1],
                   'host': host(inf), 'speed_mps': fnum(v), 'n_points': int(len(p)), 'phase_v': pv, 'phase_h': ph,
                   'forward_azimuth_deg': fwd_az(inf), 'sweep_start_azimuth_deg': sweep_start(p, prm, plat)[1]}
            for run in (1, 2):
                q, ring, lev = res[run]
                keep, kr = lr.ring_pattern_mask(p, ring, lev, SPACING, AZ_RES, pv, ph)
                m = np.hypot(q[:, 0], q[:, 1]) >= 20.0
                n_ok = 0
                if m.any():
                    el = lr.elevation_deg(q[m])
                    n_ok = int(np.sum(np.argmin(np.abs(el[:, None] - lev[None, :]), axis=1) == ring[m]))
                agg[run][0] += n_ok
                agg[run][1] += int(m.sum())
                rec['run%d' % run] = {'rings': int(len(lev)), 'purity_r20': (n_ok / int(m.sum())) if m.any() else None,
                                      'kept_rings': int(len(kr)), 'kept_share': float(keep.mean())}
            q, ring, lev = res[2]
            rec['run2']['levels'] = [round(float(x), 4) for x in lev]
            if np.isfinite(v) and v > 8:
                rec['run2']['swing_deg'] = fnum(ldr.lowest_laser_swing(q, N[plat]))
                sw.append(rec['run2']['swing_deg'])
            recs.append(rec)
        chk = {}
        for run in (1, 2):
            gj = g[run][plat]
            g3 = float(np.mean([r['run%d' % run]['rings'] == N[plat] for r in recs]))
            g2 = agg[run][0] / max(agg[run][1], 1)
            kept = {str(a): b for a, b in sorted(Counter(r['run%d' % run]['kept_rings'] for r in recs).items())}
            chk['run%d' % run] = {'G3_replay': g3, 'G3_gate_json': gj['G3_exact_ring_count_share'],
                                  'G2_replay': g2, 'G2_gate_json': gj['G2_purity_r20'],
                                  'kept_rings_replay': kept, 'kept_rings_gate_json': gj['pattern_kept_rings'],
                                  'match': bool(abs(g3 - gj['G3_exact_ring_count_share']) < 1e-9
                                                and abs(g2 - gj['G2_purity_r20']) < 1e-9
                                                and kept == gj['pattern_kept_rings'])}
        chk['run2']['G1_replay'] = float(np.nanmedian([s for s in sw if s is not None]))
        chk['run2']['G1_gate_json'] = g[2][plat]['G1_swing_moving_median_deg']
        print(plat, 'REPLAY', json.dumps(chk), flush=True)
        exact = [r for r in recs if r['run2']['rings'] == N[plat]]
        T_plat = np.median(np.array([r['run2']['levels'] for r in exact]), axis=0)
        T_host, n_host = {}, {}
        for h in sorted({r['host'] for r in recs}):
            L = [r['run2']['levels'] for r in exact if r['host'] == h]
            n_host[h] = len(L)
            T_host[h] = np.median(np.array(L), axis=0) if len(L) >= 5 else T_plat
        own = [h for h in T_host if n_host[h] >= 5]
        spread = float(np.max(np.ptp(np.array([T_host[h] for h in own]), axis=0))) if len(own) > 1 else 0.0
        for r in recs:
            inf = sel[r['sel_index']]
            p = ldr.load(inf)
            q, ring, lev = recover(p, inf, prm, plat, True)
            r['diag'], _ = diagnose(p, q, ring, lev, T_host[r['host']], plat, r['phase_v'], r['phase_h'])
        wrong = [r for r in recs if r['run2']['rings'] != N[plat]]
        tally = {
            'frames': len(recs), 'wrong_count_run2': len(wrong),
            'wrong_count_exact_after_join': sum(r['diag']['rings_after_join'] == N[plat] for r in wrong),
            'after_join_count_hist': {str(a): b for a, b in sorted(Counter(r['diag']['rings_after_join']
                                                                          for r in recs).items())},
            'frames_with_fragments': sum(r['diag']['n_fragments'] > 0 for r in recs),
            'exact_count_frames_with_fragments': sum(r['diag']['n_fragments'] > 0 for r in exact),
            'fragment_points_hist': {str(a): b for a, b in sorted(Counter(f['points'] for r in recs
                                                                         for f in r['diag']['fragments']).items())},
            'kept_same_after_join': {'wrong': sum(r['diag']['kept_same'] for r in wrong),
                                     'exact': sum(r['diag']['kept_same'] for r in exact)},
            'template_events_frames': {'wrong': sum(bool(r['diag']['template']['events']) for r in wrong),
                                       'exact': sum(bool(r['diag']['template']['events']) for r in exact)}}
        out['platforms'][plat] = {
            'replay_check': chk, 'tally': tally, 'records': recs, 'template_platform': T_plat.round(4).tolist(),
            'template_host': {h: T_host[h].round(4).tolist() for h in T_host}, 'template_frames_per_host': n_host,
            'template_max_spread_between_hosts_deg': spread}
        print(plat, 'TALLY', json.dumps(tally), flush=True)
        for r in wrong:
            d = r['diag']
            print(plat, 'WRONG', r['file'], 'v=%s' % r['speed_mps'], 'rings', r['run2']['rings'], 'run1',
                  r['run1']['rings'], 'frags', [(f['points'], round(f['azimuth_deg'], 1), f['joins'], f['joins_joined_id'])
                                                for f in d['fragments']][:8],
                  'joined', d['rings_after_join'], 'kept_same', d['kept_same'], 'tmpl_events',
                  len(d['template']['events']), flush=True)
    out['seconds'] = time.time() - t0
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(args.out_dir, 'frames_20261022.json'), 'w'), indent=1)
    print('wrote frames JSON, %.0f s' % out['seconds'], flush=True)


# ----------------------------------------------------------------------------------------------- plotting helpers
def mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.edgecolor': '#bdbcb6', 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'figure.facecolor': 'white',
                         'axes.facecolor': 'white'})
    return plt


def pal_rgba():
    from matplotlib.colors import to_rgba_array
    return to_rgba_array(PAL)


def scatter_rings(ax, x, y, ring, s, zorder=2):
    ax.scatter(x, y, c=pal_rgba()[np.asarray(ring) % len(PAL)], s=s, marker='.', linewidths=0, rasterized=True,
               zorder=zorder)


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def mark_forward(ax, f_az, lo=-180.0, hi=180.0):
    """Mark the car's forward direction (and the rear) on a sensor-azimuth axis."""
    for base, txt in ((f_az, 'car forward'), (f_az + 180.0, 'car rear')):
        for x in (base - 360.0, base, base + 360.0):
            if lo <= x <= hi:
                ax.axvline(x, color=INK2, lw=0.8, ls='-.', zorder=1)
                ax.text(x, 1.005, txt, transform=ax.get_xaxis_transform(), fontsize=7.5, color=INK2, ha='center',
                        va='bottom')


def save(fig, path, max_bytes=950_000):
    """PNG under max_bytes: render, quantise to an adaptive palette (the plots use few colours), lower dpi if needed."""
    import matplotlib.pyplot as plt
    from PIL import Image
    data, dpi_used = None, None
    for dpi in (130, 115, 100, 90, 80, 70):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
        try:
            im = Image.open(io.BytesIO(buf.getvalue())).convert('RGB').quantize(colors=128, method=0, dither=0)
            o = io.BytesIO()
            im.save(o, format='PNG', optimize=True)
            data = o.getvalue()
        except Exception:                                   # noqa: BLE001 - fall back to the plain PNG
            data = buf.getvalue()
        dpi_used = dpi
        if len(data) <= max_bytes:
            break
    with open(path, 'wb') as f:
        f.write(data)
    plt.close(fig)
    print('saved', path, len(data), 'bytes, dpi', dpi_used, flush=True)
    return len(data)


def fov(T, pad=1.2):
    return float(np.min(T) - pad), float(np.max(T) + pad)


def zoom_band(T, n=16):
    d = np.abs(np.diff(T))
    w = np.convolve(d, np.ones(n - 1), 'valid')
    k0 = int(np.argmin(w))
    sub = T[k0:k0 + n]
    gap = float(np.median(d[k0:k0 + n - 1]))
    return float(np.min(sub) - gap / 2), float(np.max(sub) + gap / 2)


def header(plat, label, rec):
    v = rec['speed_mps']
    return ('Lyft %s-beam LIDAR_TOP, %s: %s, ego speed %s\nrecovered rings %d (gate run 2; run 1: %d); '
            'sample token %s' % (plat, label, rec['file'], 'n/a' if v is None else '%.1f m/s' % v,
                                 rec['run2']['rings'], rec['run1']['rings'], rec['token']))


# ----------------------------------------------------------------------------------------------- figures
def fig_range(plat, label, rec, p, q, ring, T, path):
    """Figure 1: range image before / after de-skew, coloured by recovered ring."""
    from matplotlib.patches import Rectangle
    plt = mpl()
    lo, hi = fov(T, 1.6)
    zlo, zhi = zoom_band(T)
    fig = plt.figure(figsize=(16, 9.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.3, 1.2], hspace=0.40, wspace=0.12, left=0.05, right=0.99,
                          top=0.80, bottom=0.065)
    rows = ((p, 'BEFORE de-skew: stored (motion-compensated) points, angles about the sensor origin'),
            (q, 'AFTER de-skew: each point about the sensor at its own firing time'))
    for row, (xyz, what) in enumerate(rows):
        az, el = lr.azimuth_deg(xyz), lr.elevation_deg(xyz)
        ax = fig.add_subplot(gs[row, 0])
        scatter_rings(ax, az, el, ring, 0.7)
        ax.set_xlim(-180, 180)
        ax.set_ylim(lo, hi)
        mark_forward(ax, rec['forward_azimuth_deg'])
        ax.axvline(rec['sweep_start_azimuth_deg'], color=INK, lw=0.7, ls=':', zorder=1)
        ax.add_patch(Rectangle((ZOOM_AZ[0], zlo), ZOOM_AZ[1] - ZOOM_AZ[0], zhi - zlo, fill=False, ec=INK, lw=0.9,
                               ls='--'))
        ax.set_title(what, loc='left', pad=14)
        style(ax, AZ_LABEL, 'elevation (deg)')
        m = (az >= ZOOM_AZ[0]) & (az <= ZOOM_AZ[1]) & (el >= zlo) & (el <= zhi)
        ax2 = fig.add_subplot(gs[row, 1])
        scatter_rings(ax2, az[m], el[m], ring[m], 5)
        ax2.axvline(rec['sweep_start_azimuth_deg'], color=INK, lw=0.7, ls=':', zorder=1)
        ax2.set_xlim(*ZOOM_AZ)
        ax2.set_ylim(zlo, zhi)
        ax2.set_title('zoom (dashed box): the 16 most closely spaced rings', loc='left', fontsize=9, pad=14)
        style(ax2, 'azimuth (deg)', '')
    fig.suptitle(header(plat, label, rec), x=0.05, y=0.985, ha='left', fontsize=10.5)
    fig.text(0.05, 0.905, 'Colour = recovered ring ID (stored-order segment after de-skew), cycling through 8 hues '
             '(neighbouring IDs never share a hue); both rows use the same IDs. Dotted vertical line = azimuth where each '
             'laser\'s sweep starts (the compensation seam).\nDash-dot = the car\'s forward / rear direction (Lyft\'s '
             'LIDAR_TOP is yawed ~178 deg in the car frame). Raw .bin, angles about the sensor, no SHIFT_COOR.',
             fontsize=8.5, color=INK2, va='top')
    return save(fig, path)


def elev_hist(xyz_list, bins):
    c = np.zeros(len(bins) - 1)
    for xyz in xyz_list:
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        c += np.histogram(lr.elevation_deg(xyz[r >= 3.0]), bins)[0]
    return c / max(c.sum(), 1)


def fig_hist(plat, rec, p, q, moving, still, T, path, mov_host):
    """Figure 2: elevation histograms before / after de-skew; one moving frame and pooled."""
    plt = mpl()
    lo, hi = fov(T, 1.6)
    zlo, zhi = zoom_band(T)
    bins = np.arange(lo, hi + 0.02, 0.02)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    rows = [(elev_hist([p], bins), '#8a8984', 'one moving frame (%s, %.1f m/s): BEFORE de-skew' % (rec['file'], rec['speed_mps'])),
            (elev_hist([q], bins), PAL[0], 'same frame: AFTER de-skew'),
            (elev_hist([a for a, _ in moving], bins), '#8a8984',
             'pooled: %d moving frames (> 8 m/s, host %s, gate sample): BEFORE de-skew' % (len(moving), mov_host)),
            (elev_hist([b for _, b in moving], bins), PAL[0], 'same %d frames: AFTER de-skew' % len(moving)),
            (elev_hist([b for _, b in still], bins), PAL[2],
             'pooled: %d standstill frames (< 0.5 m/s, all hosts, gate sample), after de-skew (~ no change at rest)' % len(still))]
    fig = plt.figure(figsize=(16, 13.5))
    gs = fig.add_gridspec(len(rows), 2, width_ratios=[2.6, 1.4], hspace=0.55, wspace=0.12, left=0.05, right=0.99,
                          top=0.91, bottom=0.05)
    for i, (h, col, title) in enumerate(rows):
        for j, (xl, xh) in enumerate(((lo, hi), (zlo, zhi))):
            ax = fig.add_subplot(gs[i, j])
            m = (ctr >= xl) & (ctr <= xh)
            ax.fill_between(ctr[m], h[m], step='mid', color=col, lw=0, alpha=0.9)
            ax.vlines(T[(T >= xl) & (T <= xh)], 0.9, 1.0, transform=ax.get_xaxis_transform(), color=INK, lw=0.6)
            ax.set_xlim(xl, xh)
            ax.set_ylim(0, max(h[m].max() * 1.12, 1e-6))
            ax.set_title(title if j == 0 else 'zoom: the 16 most closely spaced rings', loc='left',
                         fontsize=9.5 if j == 0 else 9)
            style(ax, 'elevation about the sensor (deg)' if i == len(rows) - 1 else '',
                  'share of points / 0.02 deg' if j == 0 else '')
    fig.suptitle('Lyft %s-beam LIDAR_TOP: elevation histograms before and after de-skew (points >= 3 m from the sensor, '
                 '0.02 deg bins)\nblack ticks at the top = ring elevations of the template (median recovered ring '
                 'elevations of the gate frames with an exact ring count, host %s)' % (plat, rec['host']),
                 x=0.05, ha='left', fontsize=10.5)
    return save(fig, path)


def fig_bev(plat, items, path):
    """Figure 3: BEV of the near ground (0-30 m, z < -1 m about the sensor), coloured by ring; car heading up the x."""
    plt = mpl()
    fig, axs = plt.subplots(len(items), 2, figsize=(13, 6.6 * len(items)))
    axs = np.atleast_2d(axs)
    for i, (label, rec, p, q, ring, motion) in enumerate(items):
        f_az, a0 = rec['forward_azimuth_deg'], rec['sweep_start_azimuth_deg']
        for j, (xyz, what) in enumerate(((p, 'stored (motion-compensated) coordinates'),
                                         (q, 'de-skewed: sensor at each point\'s firing time'))):
            ax = axs[i, j]
            c = to_car_yaw(xyz, f_az)
            m = (np.hypot(c[:, 0], c[:, 1]) <= 30.0) & (c[:, 2] < -1.0)
            scatter_rings(ax, c[m, 0], c[m, 1], ring[m], 1.6)
            sa = np.radians(a0 - f_az)
            ax.plot([0, 30 * np.cos(sa)], [0, 30 * np.sin(sa)], ls='--', color=INK, lw=0.8)
            if motion is not None and np.linalg.norm(motion[0][:2]) > 0.5:
                vv = to_car_yaw(np.asarray(motion[0], dtype=np.float64)[None, :], f_az)[0]
                ax.annotate('', xy=(vv[0], vv[1]), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=INK, lw=1.4))
                ax.text(vv[0], vv[1] + 1.2, 'ego velocity x 1 s', fontsize=8, color=INK, ha='center')
            ax.plot(0, 0, marker='+', color=INK, ms=8)
            ax.set_aspect('equal')
            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)
            v = rec['speed_mps']
            ax.set_title('%s, %s m/s, %s\n%s' % (label, 'n/a' if v is None else '%.1f' % v, rec['file'], what),
                         loc='left', fontsize=9.5)
            style(ax, 'forward (m)', 'left (m)')
    fig.suptitle('Lyft %s-beam: near ground in BEV, points within 30 m with z < -1.0 m about the sensor, colour = '
                 'recovered ring ID (8 hues, cycling).\nRotated about the sensor\'s z axis into the car\'s heading '
                 '(ref_from_car); raw .bin, no SHIFT_COOR.\nDashed line: azimuth where each laser\'s sweep starts '
                 '(the compensation seam).' % plat, x=0.04, ha='left', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, path)


def focus_of(d, ring):
    """(stored indices to centre on, label, ring IDs of interest) for figure 4."""
    fr = d['fragments']
    if fr:
        kept_frag = {k['ring'] for k in d['kept_recovered'] if k['fragment']}
        f = next((x for x in fr if x['ring'] in kept_frag), fr[0])
        s0, s1 = f['stored_index'], f['stored_index'] + f['points'] - 1
        fset = {x['ring'] for x in fr}
        nr = int(ring.max()) + 1
        lo_r, hi_r = f['ring'] - 1, f['ring'] + 1                 # widen to the nearest full rings either side
        while lo_r > 0 and lo_r in fset:
            lo_r -= 1
        while hi_r < nr - 1 and hi_r in fset:
            hi_r += 1
        rr = list(range(max(lo_r, 0), min(hi_r, nr - 1) + 1))
        return [s0, s1], 'fragment ring %d (%d point%s, %s)' % (f['ring'], f['points'], '' if f['points'] == 1 else 's',
                                                               'kept by RING_PATTERN' if f['ring'] in kept_frag
                                                               else 'not kept'), rr
    ev = d['template']['events']
    if ev:
        e = ev[0]
        idx = [e['stored_index']] + ([e['recovered_cut']] if e['type'] == 'displaced' else [])
        rr = sorted({int(ring[i]) for i in idx} | {int(ring[i - 1]) for i in idx})
        return idx, 'template reading: %s cut, lasers %s, at %.1f deg' % (e['type'], e['lasers'], e['azimuth_deg']), rr
    return None, 'no fragment and no template event', []


def fig_wrong(plat, label, rec, p, q, ring, lev, T, d, arr, path):
    """Figure 4: one wrong-count frame: fragments / boundary events and what RING_PATTERN keeps."""
    plt = mpl()
    az, el = lr.azimuth_deg(q), lr.elevation_deg(q)
    first, last, size = ring_bounds(ring)
    a0, sign = rec['sweep_start_azimuth_deg'], lr.scan_direction(lr.azimuth_deg(p))
    idx, flabel, rr = focus_of(d, ring)
    rr = [r for r in rr if 0 <= r < len(size)]
    fset = {f['ring'] for f in d['fragments']}
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1.0], height_ratios=[1.0, 1.0, 1.0], hspace=0.45, wspace=0.14,
                          left=0.05, right=0.985, top=0.9, bottom=0.05)
    # (a) points per recovered ring
    ax = fig.add_subplot(gs[0, 0])
    cols = [PAL[7] if i in fset else PAL[0] for i in range(len(size))]
    ax.bar(np.arange(len(size)), size, color=cols, width=0.75)
    kept = {k['ring']: k['points_kept'] for k in d['kept_recovered']}
    for i, n_k in kept.items():
        ax.plot(i, size[i] * 1.6, marker='s', ms=6, mfc='none', mec=INK, mew=0.9, ls='none')
    ax.set_yscale('log')
    ax.set_ylim(0.7, size.max() * 4)
    ax.set_xlim(-1, len(size))
    ax.set_title('(a) points in each recovered ring (log); red = fragment (< %d points); square = kept by RING_PATTERN'
                 % MIN_BLOCK, loc='left', fontsize=9)
    style(ax, 'recovered ring ID (stored order: top laser first)', 'points')
    # text
    axt = fig.add_subplot(gs[0, 1])
    axt.axis('off')
    lines = ['recovered rings: %d; fragments (< %d points): %d; rings after joining' % (d['recovered_rings'], MIN_BLOCK,
                                                                                    d['n_fragments']),
             'each fragment to its neighbour: %d' % d['rings_after_join']]
    for f in d['fragments'][:8]:
        lines.append('  ring %d: %d pt, az %.1f, el %.2f deg -> joins %s ring %d' % (
            f['ring'], f['points'], f['azimuth_deg'], f['elevation_deg'], f['joins'], f['joins_ring']))
    if len(d['fragments']) > 8:
        lines.append('  ... %d more in the JSON' % (len(d['fragments']) - 8))
    kr = ['%d%s:%d' % (k['ring'], '*' if k['fragment'] else '', k['points_kept']) for k in d['kept_recovered']]
    kj = ['%d:%d' % (k['joined_id'], k['points_kept']) for k in d['kept_after_join']]
    lines += ['RING_PATTERN, gate phases %.3f / %.3f (ring:kept points; * = fragment)' % (rec['phase_v'], rec['phase_h'])]
    for name, items in (('recovered', kr), ('joined', kj)):
        for c0 in range(0, len(items), 8):
            lines.append(('  %-9s ' % (name + ':') if c0 == 0 else ' ' * 12) + ' '.join(items[c0:c0 + 8]))
    lines.append('  same kept rings: %s; kept points %d vs %d; %d points differ' % (
        'yes' if d['kept_same'] else 'NO', d['kept_points_recovered'], d['kept_points_after_join'],
        d['kept_points_differing']))
    t = d['template']
    if t['events'] or d['rings_after_join'] != N[plat]:
        lines.append('template reading of the joined rings: %d missing, %d extra, %d displaced cuts' % (
            t['n_missing'], t['n_extra'], t['n_displaced']))
        for e in t['events'][:4]:
            lines.append('  %s: lasers %s at az %.1f deg' % (e['type'], e['lasers'], e['azimuth_deg']))
        if len(t['events']) > 4:
            lines.append('  ... %d more in the JSON' % (len(t['events']) - 4))
    axt.text(0, 1.02, '\n'.join(lines), va='top', ha='left', fontsize=8.2, family='monospace', color=INK,
             transform=axt.transAxes)
    if idx is not None:
        s_lo, s_hi = min(idx), max(idx)
        f_az = float(az[s_lo])
        full_nb = [r for r in rr if r not in fset]
        levs = [lev[r] for r in full_nb] if full_nb else [float(np.median(el[s_lo:s_hi + 1]))]
        gap = max(float(np.median(np.abs(np.diff(T)))), 0.3)
        f_el = float(np.median(el[s_lo:s_hi + 1]))
        if abs(f_el - np.clip(f_el, min(levs), max(levs))) < 4.0:   # include the focus if it is near the band
            levs.append(f_el)
        blo, bhi = min(levs) - 2.0 * gap, max(levs) + 2.0 * gap
        inv = np.isin(ring, rr)
        isf = np.isin(ring, list(fset))
        n_out = int(np.sum(inv & isf & ((el < blo) | (el > bhi))))
        # (b1) band, 360 deg
        axb = fig.add_subplot(gs[1, 0])
        mb = (el >= blo) & (el <= bhi)
        axb.scatter(az[mb & ~inv], el[mb & ~inv], c=MUTED, s=1.2, marker='.', linewidths=0, rasterized=True)
        scatter_rings(axb, az[mb & inv & ~isf], el[mb & inv & ~isf], ring[mb & inv & ~isf], 2.5)
        axb.scatter(az[inv & isf], el[inv & isf], marker='X', s=60, c=PAL[7], edgecolors='white', linewidths=0.6,
                    zorder=5)
        axb.axvline(a0, color=INK, lw=0.7, ls=':')
        mark_forward(axb, rec['forward_azimuth_deg'])
        axb.set_xlim(-180, 180)
        axb.set_ylim(blo, bhi)
        axb.set_title('(b) de-skewed range image; colour = rings %d-%d, grey = others, red X = fragment points%s; '
                      'dotted = sweep start' % (rr[0], rr[-1], '' if not n_out else ' (%d off the axis)' % n_out),
                      loc='left', fontsize=8.6, pad=14)
        style(axb, AZ_LABEL, 'elevation (deg)')
        # (b2) zoom around the focus azimuth
        axz = fig.add_subplot(gs[1, 1])
        mz = (np.abs(wrap(az - f_az)) <= 8.0)
        ylo, yhi = axb.get_ylim()
        mzb = mz & (el >= ylo) & (el <= yhi)
        axz.scatter(wrap(az[mzb & ~inv] - f_az), el[mzb & ~inv], c=MUTED, s=6, marker='.', linewidths=0, rasterized=True)
        scatter_rings(axz, wrap(az[mzb & inv & ~isf] - f_az), el[mzb & inv & ~isf], ring[mzb & inv & ~isf], 14)
        axz.scatter(wrap(az[mz & inv & isf] - f_az), el[mz & inv & isf], marker='X', s=90, c=PAL[7],
                    edgecolors='white', linewidths=0.6, zorder=5)
        axz.axvline(wrap(a0 - f_az), color=INK, lw=0.8, ls=':')
        axz.set_xlim(-8, 8)
        axz.set_ylim(ylo, yhi)
        axz.set_title('zoom: +-8 deg about the focus (%.1f deg)' % f_az, loc='left', fontsize=8.6, pad=14)
        style(axz, 'azimuth minus %.1f deg' % f_az, '')
        # (c1) / (c2) along the stored order
        w0, w1 = max(0, s_lo - 160), min(len(ring), s_hi + 161)
        ii = np.arange(w0, w1)
        fi = ii[np.isin(ring[ii], list(fset))]
        for col, (yv, ylab_, (ylo_c, yhi_c), ttl) in enumerate((
                (el, 'de-skewed elevation (deg)', (blo, bhi),
                 '(c) elevation along the stored order; black line = ring cut, dotted = template levels'),
                (sign * wrap(az - a0), 'azimuth past the sweep start (deg)', (-45.0, 45.0),
                 'same points: azimuth past the sweep start (%.1f deg), wrapped' % a0))):
            axc = fig.add_subplot(gs[2, col])
            scatter_rings(axc, ii, yv[ii], ring[ii], 7)
            axc.scatter(fi, yv[fi], marker='X', s=70, c=PAL[7], edgecolors='white', linewidths=0.6, zorder=5)
            for c in arr['rec_cuts']:
                if w0 < c < w1:
                    axc.axvline(c - 0.5, color=INK, lw=0.9)
            if col == 0:
                for k in T[(T >= ylo_c) & (T <= yhi_c)]:
                    axc.axhline(k, color=INK2, lw=0.5, ls=':')
            else:
                axc.axhline(0, color=INK, lw=0.7, ls=':')
            n_clip = int(np.sum((yv[ii] < ylo_c) | (yv[ii] > yhi_c)))
            axc.set_xlim(w0, w1)
            axc.set_ylim(ylo_c, yhi_c)
            axc.set_title(ttl + ('' if not n_clip else '; %d points off the axis' % n_clip), loc='left', fontsize=8.6)
            style(axc, 'stored point index', ylab_)
    fig.suptitle(header(plat, label, rec) + '\nfocus: ' + flabel, x=0.05, y=0.99, ha='left', fontsize=10.5)
    return save(fig, path)


def fig_summary(plat, recs, path):
    """Figure 4 overview: every wrong-count frame's fragments by position, and what RING_PATTERN keeps."""
    plt = mpl()
    from matplotlib.lines import Line2D
    wrong = sorted([r for r in recs if r['run2']['rings'] != N[plat]],
                   key=lambda r: (r['run2']['rings'], r['file']))
    h = max(4.5, 0.25 * len(wrong) + 2.6)
    fig, ax = plt.subplots(figsize=(15, h))
    ylab = []
    for y, r in enumerate(wrong):
        d = r['diag']
        kj = [k['joined_id'] for k in d['kept_after_join']]
        ax.plot(kj, [y] * len(kj), ls='none', marker='o', ms=3.2, color=MUTED, zorder=1)
        kr_full = {k['joined_id'] for k in d['kept_recovered'] if not k['fragment']}
        only_r = sorted(kr_full - set(kj))
        only_j = sorted(set(kj) - kr_full)
        ax.plot(only_r, [y] * len(only_r), ls='none', marker='^', ms=7, mfc='none', mec=INK, mew=1.0, zorder=3)
        ax.plot(only_j, [y] * len(only_j), ls='none', marker='v', ms=7, mfc='none', mec=INK, mew=1.0, zorder=3)
        for f in d['fragments']:
            x = f['joins_joined_id'] + (0.35 if f['joins'] == 'preceding' else -0.35)
            kept_f = any(k['fragment'] and k['ring'] == f['ring'] for k in d['kept_recovered'])
            ax.plot(x, y, ls='none', marker='x', color=PAL[7], ms=6 + 2 * np.log10(f['points']), mew=1.6, zorder=4)
            if kept_f:
                ax.plot(x, y, ls='none', marker='o', ms=11, mfc='none', mec=PAL[7], mew=1.0, zorder=4)
        for e in d['template']['events']:
            st = {'missing': ('v', PAL[1]), 'extra': ('+', PAL[0]), 'displaced': ('D', PAL[3])}[e['type']]
            ax.plot(float(np.mean(e['lasers'])), y, ls='none', marker=st[0], color=st[1], ms=6, mew=1.4, zorder=2)
        v = r['speed_mps']
        stem = r['file'].replace('host-', '').replace('_lidar1_', '_').replace('.bin', '')
        ylab.append('%s %s m/s %3d -> %2d%s' % (stem, '  n/a' if v is None else '%5.1f' % v,
                                                r['run2']['rings'], d['rings_after_join'],
                                                '' if d['kept_same'] else ' *'))
    ax.set_yticks(range(len(wrong)))
    ax.set_yticklabels(ylab, fontsize=7.5, family='monospace')
    ax.set_ylim(len(wrong) - 0.5, -0.5)
    ax.set_xlim(-1, max(N[plat], max(r['diag']['rings_after_join'] for r in wrong)) + 0.5)
    style(ax, 'ring ID after joining the fragments (= laser index, top first, when %d rings remain)' % N[plat], '')
    hd = [Line2D([], [], ls='none', marker='x', color=PAL[7], mew=1.6, label='fragment (< %d points), drawn beside the '
                 'ring it joins; size ~ log points' % MIN_BLOCK),
          Line2D([], [], ls='none', marker='o', ms=10, mfc='none', mec=PAL[7], label='fragment kept by RING_PATTERN'),
          Line2D([], [], ls='none', marker='o', color=MUTED, ms=4, label='ring kept by RING_PATTERN after joining'),
          Line2D([], [], ls='none', marker='^', mfc='none', mec=INK, label='full ring kept only without joining'),
          Line2D([], [], ls='none', marker='v', mfc='none', mec=INK, label='ring kept only after joining'),
          Line2D([], [], ls='none', marker='v', color=PAL[1], label='template reading: missing cut'),
          Line2D([], [], ls='none', marker='+', color=PAL[0], mew=1.4, label='template reading: extra cut'),
          Line2D([], [], ls='none', marker='D', color=PAL[3], ms=5, label='template reading: displaced cut')]
    fig.legend(handles=hd, loc='lower center', ncol=3, fontsize=8, frameon=False)
    ax.set_title('Lyft %s-beam: the %d gate frames (run 2) without exactly %d recovered rings; rows: file, speed, '
                 'recovered rings -> rings after joining fragments.\n"*" = RING_PATTERN (gate phases) keeps a different '
                 'ring set, or a fragment, than after joining' % (plat, len(wrong), N[plat]), loc='left', fontsize=9.5)
    fig.tight_layout(rect=(0, 1.0 / h, 1, 1))
    return save(fig, path)


def nusc_speed(info):
    for sw in info.get('sweeps', []) or []:
        T, lag = sw.get('transform_matrix', None), sw.get('time_lag', 0.0)
        if T is None or not lag or not (0.02 < lag < 0.12):
            continue
        return float(np.linalg.norm(np.asarray(T, dtype=np.float64)[:2, 3]) / lag)
    return np.nan


def fig_pair(plat, label, rec, p, ring, lev, nus, path):
    """Figure 5: Lyft stored, Lyft RING_PATTERN-thinned, nuScenes TRAIN frame; range image and BEV in the car's heading."""
    plt = mpl()
    keep, kr = lr.ring_pattern_mask(p, ring, lev, SPACING, AZ_RES, rec['phase_v'], rec['phase_h'])
    lo_, hi_ = float(lev.min()), float(lev.max())
    j0 = int(np.floor((lo_ - rec['phase_v'] * SPACING) / SPACING)) - 1
    j1 = int(np.ceil((hi_ - rec['phase_v'] * SPACING) / SPACING)) + 1
    tg = (np.arange(j0, j1 + 1) + rec['phase_v']) * SPACING
    tg = tg[(tg >= lo_) & (tg <= hi_)]
    pc = to_car_yaw(p, rec['forward_azimuth_deg'])
    nc = to_car_yaw(nus['points'], nus['forward_azimuth_deg'])
    cols = [(pc, ring, 'Lyft %s-beam as stored: %d points, %d recovered rings' % (plat, len(p), len(lev))),
            (pc[keep], ring[keep], 'Lyft after RING_PATTERN: %d points (%.1f%%), %d rings kept' % (
                keep.sum(), 100 * keep.mean(), len(kr))),
            (nc, nus['ring'], 'nuScenes TRAIN keyframe: %d points (ego returns removed), 32 stored rings' % len(nc))]
    fig = plt.figure(figsize=(18, 12.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.55], hspace=0.22, wspace=0.12, left=0.04, right=0.99,
                          top=0.87, bottom=0.05)
    for j, (xyz, rg, title) in enumerate(cols):
        ax = fig.add_subplot(gs[0, j])
        scatter_rings(ax, lr.azimuth_deg(xyz), lr.elevation_deg(xyz), rg, 0.5 if j == 0 else 1.0)
        if j == 1:
            for t in tg:
                ax.axhline(t, color=INK2, lw=0.4, ls=':')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-32, 17)
        ax.set_title(title, loc='left', fontsize=9.5)
        style(ax, 'azimuth in the car\'s heading (deg; 0 = forward, +90 = left)', 'elevation (deg)' if j == 0 else '')
        axb = fig.add_subplot(gs[1, j])
        m = (np.abs(xyz[:, 0]) <= 50) & (np.abs(xyz[:, 1]) <= 50)
        scatter_rings(axb, xyz[m, 0], xyz[m, 1], rg[m], 0.35 if j == 0 else 0.8)
        axb.plot(0, 0, marker='+', color=INK, ms=9)
        axb.set_aspect('equal')
        axb.set_xlim(-50, 50)
        axb.set_ylim(-50, 50)
        style(axb, 'forward (m)', 'left (m)' if j == 0 else '')
    v = rec['speed_mps']
    fig.suptitle('%s Lyft %s-beam frame vs a nuScenes TRAIN frame at a similar ego speed. Range images (top; dotted '
                 'lines = the 1.33 deg lattice) and BEV +-50 m (bottom); colour = ring ID, 8 hues cycling.\n'
                 'Lyft: %s, host %s, %s m/s, token %s; RING_PATTERN phases %.3f / %.3f (the gate\'s).\n'
                 'nuScenes: %s, %.1f m/s, token %s.\nStored (compensated) coordinates for both, angles about the sensor, '
                 'rotated about z into the car\'s heading (ref_from_car, yaw only); no SHIFT_COOR.' % (
                     label, plat, rec['file'], rec['host'], 'n/a' if v is None else '%.1f' % v, rec['token'],
                     rec['phase_v'], rec['phase_h'], nus['file'], nus['speed'], nus['token']),
                 x=0.04, ha='left', fontsize=10)
    return save(fig, path)


# ----------------------------------------------------------------------------------------------- plots stage
def draws_lowpurity(plat, align=False):
    """Figure 1 draws the moving frame at p10 of far purity: 64-beam always; 40-beam only with --align_platforms."""
    return plat == '64' or bool(align)


def f4_wrong_frames(recs, plat, align=False):
    """The wrong-count frames figure 4 draws, ordered by |rings - N|, then file. Default: all on 40-beam and six on
    64-beam spanning the range of |rings - 64|; with --align_platforms, all on both platforms."""
    wrong = sorted([r for r in recs if r['run2']['rings'] != N[plat]],
                   key=lambda r: (abs(r['run2']['rings'] - N[plat]), r['file']))
    if plat == '64' and len(wrong) > 6 and not align:
        wrong = [wrong[int(round(x * (len(wrong) - 1)))] for x in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    return wrong


def stage_plots(args):
    t0 = time.time()
    align = bool(getattr(args, 'align_platforms', False))
    fr = json.load(open(os.path.join(args.out_dir, 'frames_20261022.json')))
    infos = pickle.load(open(ldr.ROOT + 'lyft_infos_train.pkl', 'rb'))
    prm = ldr.params_from_explore(args.explore)
    nus_infos = pickle.load(open(NUSC_ROOT + 'nuscenes_infos_10sweeps_train.pkl', 'rb'))
    nus_sp = np.array([nusc_speed(i) for i in nus_infos])
    nrng = np.random.default_rng(20261022)
    manifest = {'figures': [], 'frames': {}}
    if align:
        manifest['align_platforms'] = True
    for plat in ('40', '64'):
        P = fr['platforms'][plat]
        recs = P['records']
        sel = [i for i in infos if ldr.platform(i) == plat]

        def get(rec):
            inf = sel[rec['sel_index']]
            assert inf['token'] == rec['token'], 'replayed frame does not match the frames JSON'
            p = ldr.load(inf)
            q, ring, lev = recover(p, inf, prm, plat, True)
            return inf, p, q, ring, lev

        def T_of(rec):
            return np.array(P['template_host'][rec['host']])

        def med_purity(cands):
            cands = sorted([r for r in cands if r['run2']['purity_r20'] is not None],
                           key=lambda r: (r['run2']['purity_r20'], r['file']))
            return cands[len(cands) // 2], cands

        def meta(rec):
            m = {k: rec[k] for k in ('token', 'file', 'host', 'speed_mps')}
            m.update({'rings': rec['run2']['rings'], 'purity_r20': rec['run2']['purity_r20']})
            return m

        still_c = [r for r in recs if r['speed_mps'] is not None and r['speed_mps'] < 0.5]
        mov_c = [r for r in recs if r['speed_mps'] is not None and r['speed_mps'] > 8]
        still, _ = med_purity(still_c)
        mov, mov_sorted = med_purity(mov_c)
        picks = [('standstill', 'standstill', still), ('moving', 'moving', mov)]
        if draws_lowpurity(plat, align):
            picks.append(('moving_lowpurity', 'moving, far purity at p10 of moving frames',
                          mov_sorted[len(mov_sorted) // 10]))
        manifest['frames'][plat] = {}
        bev_items = []
        for tag, label, rec in picks:
            inf, p, q, ring, lev = get(rec)
            path = os.path.join(args.out_dir, 'f1_range_%s_%s.png' % (plat, tag))
            fig_range(plat, label, rec, p, q, ring, T_of(rec), path)
            manifest['figures'].append(path)
            manifest['frames'][plat][tag] = meta(rec)
            if tag in ('standstill', 'moving'):
                bev_items.append((label, rec, p, q, ring, lr.ego_motion(inf)))
        path = os.path.join(args.out_dir, 'f3_bev_ground_%s.png' % plat)
        fig_bev(plat, bev_items, path)
        manifest['figures'].append(path)
        hosts = Counter(r['host'] for r in mov_c)
        mov_host = hosts.most_common(1)[0][0]
        mv = sorted([r for r in mov_c if r['host'] == mov_host], key=lambda r: r['file'])[:40]
        moving = [get(r)[1:3] for r in mv]
        stills = [get(r)[1:3] for r in still_c]
        _, p, q, ring, lev = get(mov)
        path = os.path.join(args.out_dir, 'f2_elev_hist_%s.png' % plat)
        fig_hist(plat, mov, p, q, moving, stills, T_of(mov), path, mov_host)
        manifest['figures'].append(path)
        manifest['frames'][plat]['hist_pooled_moving'] = {'host': mov_host, 'files': [r['file'] for r in mv]}
        manifest['frames'][plat]['hist_pooled_standstill'] = [r['file'] for r in still_c]
        # Figure 4: wrong-count frames, plus one exact-count frame the template reading flags
        wrong = f4_wrong_frames(recs, plat, align)
        draw = [('wrong', 'gate frame with a wrong ring count', r) for r in wrong]
        ex_t = sorted([r for r in recs if r['run2']['rings'] == N[plat] and r['diag']['template']['events']],
                      key=lambda r: r['file'])
        if ex_t:
            draw.append(('exact', 'gate frame with an exact ring count that the template reading flags', ex_t[0]))
        manifest['frames'][plat]['f4_drawn'] = []
        for kind, label, rec in draw:
            inf, p, q, ring, lev = get(rec)
            d, arr = diagnose(p, q, ring, lev, T_of(rec), plat, rec['phase_v'], rec['phase_h'])
            path = os.path.join(args.out_dir, 'f4_%s_%s_%s.png' % (kind, plat, rec['file'].replace('.bin', '')))
            fig_wrong(plat, label, rec, p, q, ring, lev, T_of(rec), d, arr, path)
            manifest['figures'].append(path)
            manifest['frames'][plat]['f4_drawn'].append(dict(meta(rec), kind=kind))
        path = os.path.join(args.out_dir, 'f4_wrong_summary_%s.png' % plat)
        fig_summary(plat, recs, path)
        manifest['figures'].append(path)
        # Figure 5: thinned Lyft vs nuScenes TRAIN, moving and standstill
        for label, rec in (('moving', mov), ('standstill', still)):
            inf, p, q, ring, lev = get(rec)
            v = rec['speed_mps'] or 0.0
            cand = np.nonzero(np.isfinite(nus_sp) & (np.abs(nus_sp - v) < 0.5))[0]
            k = int(nrng.choice(cand))
            ni = nus_infos[k]
            pts = np.fromfile(NUSC_ROOT + ni['lidar_path'], dtype=np.float32).reshape(-1, 5)
            pts = pts[~((np.abs(pts[:, 0]) < 1.5) & (np.abs(pts[:, 1]) < 1.5))]   # the loader's keyframe ego cut
            nus = {'points': pts[:, :3].astype(np.float64), 'ring': pts[:, 4].astype(np.int64),
                   'file': ni['lidar_path'].split('/')[-1], 'token': ni['token'], 'speed': float(nus_sp[k]),
                   'forward_azimuth_deg': fwd_az(ni)}
            path = os.path.join(args.out_dir, 'f5_pair_%s_%s.png' % (plat, label))
            fig_pair(plat, label, rec, p, ring, lev, nus, path)
            manifest['figures'].append(path)
            manifest['frames'][plat]['pair_%s' % label] = {
                'lyft': meta(rec), 'lyft_forward_azimuth_deg': rec['forward_azimuth_deg'],
                'nuscenes': {'token': nus['token'], 'file': nus['file'], 'speed_mps': nus['speed'],
                             'forward_azimuth_deg': nus['forward_azimuth_deg'],
                             'candidates_within_0.5mps': int(len(cand))}}
    manifest['seconds'] = time.time() - t0
    json.dump(manifest, open(os.path.join(args.out_dir, 'manifest_20261022.json'), 'w'), indent=1)
    print('wrote manifest, %.0f s' % manifest['seconds'], flush=True)


# ----------------------------------------------------------------------------------------------- table stage
TABLE_HEAD = ['file', 'sample token', 'speed (m/s)', 'rings, run 2 (run 1)', 'fragments', 'rings after joining',
              'RING_PATTERN rings kept, recovered -> joined', 'what differs', 'kept points differing']
TEMPLATE_HEAD = 'template reading after joining'


def lattice(ring_el, phase_v):
    """RING_PATTERN's vertical lattice exactly as lr.ring_pattern_mask builds it: (line index j, line elevation, ring
    nearest the line) for each line inside [min, max] of `ring_el`. A line sits at (j + phase_v) * SPACING whatever the
    rings, so one j is the same line for the recovered and for the joined rings."""
    ring_el = np.asarray(ring_el, dtype=np.float64)
    lo, hi = ring_el.min(), ring_el.max()
    j0 = int(np.floor((lo - phase_v * SPACING) / SPACING)) - 1
    j1 = int(np.ceil((hi - phase_v * SPACING) / SPACING)) + 1
    j = np.arange(j0, j1 + 1)
    t = (j + phase_v) * SPACING
    m = (t >= lo) & (t <= hi)
    assert m.any(), 'field of view narrower than one lattice spacing'
    return j[m], t[m], np.argmin(np.abs(ring_el[None, :] - t[m][:, None]), axis=1)


def pattern_lines(lev, lev_j, cm, frag_rings, phase_v):
    """Line by line, why RING_PATTERN keeps different rings with the recovered rings (levels `lev`) and with the joined
    rings (levels `lev_j`; `cm` maps recovered ID -> joined ID). Kind of each line:
      widened    - the line exists only for the recovered rings (a recovered ring lies outside the joined range);
      narrowed   - the line exists only for the joined rings;
      fragment   - a fragment is the recovered ring nearest the line;
      other_ring - the nearest recovered full ring is not the joined ring nearest the line;
      same       - the same ring either way.
    Returns (lines, recovered rings kept, joined rings kept); the kept sets are lr.ring_pattern_mask's."""
    fset = {int(x) for x in frag_rings}
    jr, tr, nr = lattice(lev, phase_v)
    jj, tj, nj = lattice(lev_j, phase_v)
    R = {int(a): (float(b), int(c)) for a, b, c in zip(jr, tr, nr)}
    J = {int(a): (float(b), int(c)) for a, b, c in zip(jj, tj, nj)}
    lines = []
    for j in sorted(set(R) | set(J)):
        r = R[j][1] if j in R else None
        k = J[j][1] if j in J else None
        if k is None:
            kind = 'widened'
        elif r is None:
            kind = 'narrowed'
        elif r in fset:
            kind = 'fragment'
        elif int(cm[r]) != k:
            kind = 'other_ring'
        else:
            kind = 'same'
        lines.append({'line': j, 'elevation_deg': (R[j] if j in R else J[j])[0], 'kind': kind, 'recovered_ring': r,
                      'recovered_is_fragment': None if r is None else r in fset,
                      'recovered_joined_id': None if r is None else int(cm[r]), 'joined_ring': k})
    return lines, sorted({int(x) for x in nr}), sorted({int(x) for x in nj})


def _and(items):
    items = [str(x) for x in items]
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def what_differs(d):
    """The 'what differs' cell of the 20261022_05 §4.4 tables, from one frame's diagnostic. Ring numbers are joined IDs
    (= laser index when N rings remain after joining); '(k pt)' is a kept fragment's points that RING_PATTERN KEEPS."""
    if d['kept_same']:
        return 'same rings'
    kj = {k['joined_id'] for k in d['kept_after_join']}
    kr = {k['joined_id'] for k in d['kept_recovered'] if not k['fragment']}
    only_r, only_j = sorted(kr - kj), sorted(kj - kr)
    size = {f['ring']: f['points'] for f in d['fragments']}
    by_id = {}                                              # joined ID -> [(points kept, fragment points)], kept fragments
    for k in sorted(d['kept_recovered'], key=lambda k: (k['joined_id'], k['ring'])):
        if k['fragment']:
            by_id.setdefault(k['joined_id'], []).append((k['points_kept'], size[k['ring']]))

    def rings(ids):
        return ('ring %d' % ids[0]) if len(ids) == 1 else 'rings ' + ', '.join(str(i) for i in ids)

    parts = []
    if only_r:
        s = 'recovered also keeps ' + rings(only_r)
        if len(only_r) == 1 and list(by_id) == only_r and len(by_id[only_r[0]]) == 1:
            s += ' and its fragment (%d pt)' % by_id.pop(only_r[0])[0][0]
        parts.append(s)
    if only_j:
        parts.append(rings(only_j) + ' kept only after joining')
    if len(by_id) == 1 and len(next(iter(by_id.values()))) == 1:
        (i, [(kp, n)]), = by_id.items()
        parts.append('recovered keeps a 1-point fragment of %d' % i if n == 1 else
                     'recovered keeps a fragment of %d (%d pt)' % (i, kp))
    elif by_id:
        groups = []                                          # [ids, kept points or None, text for a multi-fragment ID]
        for i, fs in by_id.items():
            if len(fs) == 1 and groups and groups[-1][1] == fs[0][0]:
                groups[-1][0].append(i)
            elif len(fs) == 1:
                groups.append([[i], fs[0][0], None])
            else:
                groups.append([[i], None, '%d (%d fragments, %s pt)' % (i, len(fs), _and([kp for kp, _ in fs]))])
        txt = []
        for ids, kp, multi in groups:
            if multi is not None:
                txt.append(multi)
            elif len(ids) == 1:
                txt.append('%d (%d pt)' % (ids[0], kp))
            else:
                txt.append((_and(ids) if len(groups) == 1 else ', '.join(str(i) for i in ids)) + ' (%d pt each)' % kp)
        parts.append('recovered keeps fragments of ' + _and(txt))
    return '; '.join(parts)


def template_cell(t):
    """The template-reading cell: cut events of the joined rings, or 'none'."""
    parts = [(t['n_extra'], 'extra'), (t['n_missing'], 'missing'), (t['n_displaced'], 'displaced')]
    parts = [('%d %s' % (n, w), n) for n, w in parts if n]
    if not parts:
        return 'none'
    return _and([p for p, _ in parts]) + (' cut' if sum(n for _, n in parts) == 1 else ' cuts')


def fragments_cell(d):
    """points @ sensor azimuth (deg) -> joined ring, per fragment in stored order."""
    def a1(x):
        s = '%.1f' % x
        return '0.0' if s == '-0.0' else s
    return ', '.join('%d @ %s -> %d' % (f['points'], a1(f['azimuth_deg']), f['joins_joined_id'])
                     for f in d['fragments']) or 'none'


def table_row(rec, plat, with_template):
    d = rec['diag']
    cells = [rec['file'].replace('.bin', ''), '`%s`' % rec['token'],
             'n/a' if rec['speed_mps'] is None else '%.1f' % rec['speed_mps'],
             '%d (%d)' % (rec['run2']['rings'], rec['run1']['rings']), fragments_cell(d), '%d' % d['rings_after_join'],
             '%d -> %d' % (len(d['kept_recovered']), len(d['kept_after_join'])), what_differs(d),
             '{:,}'.format(d['kept_points_differing'])]
    if with_template:
        cells.append(template_cell(d['template']))
    return '| ' + ' | '.join(cells) + ' |'


def stage_table(args):
    """Per-frame tables, fragment facts and RING_PATTERN line kinds of every wrong-count frame (both platforms)."""
    t0 = time.time()
    fr = json.load(open(os.path.join(args.out_dir, 'frames_20261022.json')))
    infos = pickle.load(open(ldr.ROOT + 'lyft_infos_train.pkl', 'rb'))
    prm = ldr.params_from_explore(args.explore)
    out = {'note': 'experiments_md 20261022_05 §4.4 / §6; rows sorted by (recovered rings run 2, file)',
           'platforms': {}}
    for plat in ('40', '64'):
        P = fr['platforms'][plat]
        recs = P['records']
        sel = [i for i in infos if ldr.platform(i) == plat]
        wrong = sorted([r for r in recs if r['run2']['rings'] != N[plat]], key=lambda r: (r['run2']['rings'], r['file']))
        exact = [r for r in recs if r['run2']['rings'] == N[plat]]
        frames, frags = [], []
        for r in wrong:
            inf = sel[r['sel_index']]
            assert inf['token'] == r['token'], 'replayed frame does not match the frames JSON'
            p = ldr.load(inf)
            q, ring, lev = recover(p, inf, prm, plat, True)
            d = r['diag']
            d_now, _ = diagnose(p, q, ring, lev, np.array(P['template_host'][r['host']]), plat, r['phase_v'], r['phase_h'])
            for key in ('fragments', 'kept_recovered', 'kept_after_join', 'kept_same', 'kept_points_differing',
                        'rings_after_join'):
                assert json.loads(json.dumps(d_now[key])) == d[key], 'recomputed %s differs from the JSON: %s' % (key, r['file'])
            az, el = lr.azimuth_deg(q), lr.elevation_deg(q)
            ring_j, cm, _ = merge_fragments(ring, az, el)
            nj = int(ring_j.max()) + 1
            lev_j = np.array([np.median(el[ring_j == k]) for k in range(nj)])
            lines, kr_r, kr_j = pattern_lines(lev, lev_j, cm, [f['ring'] for f in d['fragments']], r['phase_v'])
            assert kr_r == [k['ring'] for k in d['kept_recovered']], r['file']
            assert kr_j == [k['joined_id'] for k in d['kept_after_join']], r['file']
            kept_r = {k['ring']: k['points_kept'] for k in d['kept_recovered']}
            kept_j = {k['joined_id']: k['points_kept'] for k in d['kept_after_join']}
            for ln in lines:
                ln['recovered_points_kept'] = None if ln['recovered_ring'] is None else kept_r[ln['recovered_ring']]
                ln['joined_points_kept'] = None if ln['joined_ring'] is None else kept_j[ln['joined_ring']]
            kinds = Counter(ln['kind'] for ln in lines)
            fr_rows = []
            for f in d['fragments']:
                jid = f['joins_joined_id']
                nb = [abs(lev_j[jid] - lev_j[x]) for x in (jid - 1, jid + 1) if 0 <= x < nj]
                tr = d['template']['rings']
                fr_rows.append({
                    'file': r['file'], 'ring': f['ring'], 'points': f['points'], 'azimuth_deg': f['azimuth_deg'],
                    'from_sweep_start_deg': float(abs(wrap(f['azimuth_deg'] - r['sweep_start_azimuth_deg']))),
                    'joins': f['joins'], 'joined_id': jid,
                    'laser': jid if nj == N[plat] else (int(tr[jid]['main']) if tr else None),
                    'laser_from': 'joined ID (N rings after joining)' if nj == N[plat] else 'template main laser',
                    'elevation_deg': f['elevation_deg'], 'joined_ring_elevation_deg': float(lev_j[jid]),
                    'offset_deg': float(f['elevation_deg'] - lev_j[jid]), 'local_gap_deg': float(min(nb)),
                    'below_lowest_ring': bool(f['elevation_deg'] < lev_j.min()),
                    'above_highest_ring': bool(f['elevation_deg'] > lev_j.max()),
                    'widens_range': bool(f['elevation_deg'] < lev_j.min() or f['elevation_deg'] > lev_j.max()),
                    'points_kept': kept_r.get(f['ring'], 0)})
            frags += fr_rows
            frames.append({'file': r['file'], 'token': r['token'], 'speed_mps': r['speed_mps'],
                           'rings_run2': r['run2']['rings'], 'rings_run1': r['run1']['rings'],
                           'rings_after_join': d['rings_after_join'], 'n_fragments': d['n_fragments'],
                           'kept_recovered': len(d['kept_recovered']), 'kept_after_join': len(d['kept_after_join']),
                           'kept_points_recovered': d['kept_points_recovered'],
                           'kept_points_after_join': d['kept_points_after_join'],
                           'kept_same': d['kept_same'], 'kept_points_differing': d['kept_points_differing'],
                           'what_differs': what_differs(d), 'template': template_cell(d['template']),
                           'template_counts': [d['template']['n_extra'], d['template']['n_missing'],
                                               d['template']['n_displaced']],
                           'line_kinds': dict(kinds), 'lines': [ln for ln in lines if ln['kind'] != 'same'],
                           'widened': kinds['widened'] > 0, 'fragment_takes_line': kinds['fragment'] > 0,
                           'other_kind': kinds['narrowed'] + kinds['other_ring'] > 0,
                           'row': table_row(r, plat, plat != '40')})
        both = [f for f in frames if f['widened'] and f['fragment_takes_line']]
        facts = {
            'wrong_frames': len(wrong), 'exact_frames': len(exact),
            'run1_wrong_frames_all': sum(r['run1']['rings'] != N[plat] for r in recs),
            'rings_run2': dict(sorted(Counter(f['rings_run2'] for f in frames).items())),
            'rings_run1_on_these': dict(sorted(Counter(f['rings_run1'] for f in frames).items())),
            'after_join': dict(sorted(Counter(f['rings_after_join'] for f in frames).items())),
            'frames_with_fragments': sum(f['n_fragments'] > 0 for f in frames),
            'exact_frames_with_fragments': sum(r['diag']['n_fragments'] > 0 for r in exact),
            'fragments': len(frags), 'fragment_points': dict(sorted(Counter(x['points'] for x in frags).items())),
            'within_2deg_of_sweep_start': sum(x['from_sweep_start_deg'] <= 2.0 for x in frags),
            'median_from_sweep_start_deg': float(np.median([x['from_sweep_start_deg'] for x in frags])) if frags else None,
            'joins': dict(Counter(x['joins'] for x in frags)),
            'lasers': dict(sorted(Counter(x['laser'] for x in frags if x['laser'] is not None).items())),
            'laser_from': dict(Counter(x['laser_from'] for x in frags)),
            'within_half_gap_of_joined_ring': sum(abs(x['offset_deg']) <= x['local_gap_deg'] / 2 for x in frags),
            'below_lowest_ring': sum(x['below_lowest_ring'] for x in frags),
            'above_highest_ring': sum(x['above_highest_ring'] for x in frags),
            'template_frames_resolved': sum(f['rings_after_join'] == N[plat] and f['template'] != 'none' for f in frames),
            'template_frames_unresolved': sum(f['rings_after_join'] != N[plat] and f['template'] != 'none' for f in frames),
            'template_cuts_extra_missing_displaced_wrong': [int(sum(f['template_counts'][i] for f in frames)) for i in range(3)],
            'template_frames_exact': sum(bool(r['diag']['template']['events']) for r in exact),
            'template_cuts_extra_missing_displaced_exact': [int(sum(r['diag']['template'][k] for r in exact))
                                                            for k in ('n_extra', 'n_missing', 'n_displaced')],
            'kept_recovered': dict(sorted(Counter(f['kept_recovered'] for f in frames).items())),
            'kept_after_join': dict(sorted(Counter(f['kept_after_join'] for f in frames).items())),
            'kept_same_frames': sum(f['kept_same'] for f in frames),
            'mechanism_frames': {'widened': sum(f['widened'] for f in frames),
                                 'fragment_takes_line': sum(f['fragment_takes_line'] for f in frames),
                                 'both': len(both), 'other_kind': sum(f['other_kind'] for f in frames),
                                 'differ_without_a_line_kind': sum((not f['kept_same']) and not (
                                     f['widened'] or f['fragment_takes_line'] or f['other_kind']) for f in frames)},
            'example_rule_first_with_both': both[0]['file'] if both else None}
        out['platforms'][plat] = {'facts': facts, 'frames': frames, 'fragments': frags}
        head = TABLE_HEAD + ([TEMPLATE_HEAD] if plat != '40' else [])
        md = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + [f['row'] for f in frames]
        with open(os.path.join(args.out_dir, 'wrong_count_table_%s.md' % plat), 'w') as fh:
            fh.write('\n'.join(md) + '\n')
        print(plat, 'FACTS', json.dumps(facts), flush=True)
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(args.out_dir, 'wrong_count_table_20261023.json'), 'w'), indent=1)
    print('wrote table JSON and markdown, %.0f s' % out['seconds'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['frames', 'plots', 'all', 'table'])
    ap.add_argument('--out_dir', default='/home/koyama/data/lyft_rings/viz_20261022')
    ap.add_argument('--explore', default=RINGS_DIR + 'explore_20261010.json')
    ap.add_argument('--align_platforms', action='store_true',
                    help='opt-in (20261022_05 §6): figure 4 draws every wrong-count frame on both platforms and figure 1 '
                         'the far-purity p10 moving frame on both; default output unchanged')
    args = ap.parse_args()
    if args.mode in ('frames', 'all'):
        stage_frames(args)
    if args.mode in ('plots', 'all'):
        stage_plots(args)
    if args.mode == 'table':
        stage_table(args)
