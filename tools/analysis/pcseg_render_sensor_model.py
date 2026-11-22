"""S3b render study, questions 1-2 (experiments_md episodes/20261116_01 §8.9): the two PandaSet sensors' ray models
from TRAIN-split clouds only. No semantic label is read here.

Parts (run in this order; later parts read the earlier outputs in --out):
  gt_origin   PandarGT origin per frame from LINE PARALLAX: consecutive returns of one scan line (stored order) have
              nearly the same direction about the true origin, whatever their ranges. With a wrong origin, a pair whose
              ranges differ gets an angle step that the pairs of equal range around it do not have. Fitted with the
              pose at the frame time ('static') and with the pose at each point's own time ('interp').
  p64         Pandar64 origin, roll, pitch and per-channel elevation per frame, from the label-free channel ids of
              pandaset_rings (cache /home/koyama/data/pandaset_rings/v1): each channel must be ONE cone about the
              sensor. Compared with pandaset_rings.ORIGIN. Then, with the pooled model, per frame: the attitude drift,
              the share of points whose nearest-elevation row equals their channel, the azimuth step and phase.
  gt_lattice  PandarGT's angular lattice about the pooled origin: field of view, azimuth step along a line, line
              spacing and count per sector, scan time and phase, range; a ray TEMPLATE pooled from the template frames
              (each frame registered to it by a small rotation), and its repeatability on other frames.

Writes <out>/sensor_model_<part>.json, <out>/sensor_models.json (both models) and <out>/pandargt_template.npz.
sensor_models.json is frozen once written: the render caches record its sha256 and pcocc pins it (2026-11-22). The script
refuses an --out that already holds one, so a refit always goes into a new directory, with all its parts in one run.

    python3 analysis/pcseg_render_sensor_model.py --parts gt_origin,p64,gt_lattice
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcseg_render_common import (OUT, ROOT, raw_frame, ring_channels, rings_module, rot_from_roll_pitch,  # noqa: E402
                                 split)
from pcseg.engine import cone_mask                                                               # noqa: E402
from pcseg.render import (PoseTrack, RingLattice, TemplateLattice, angles_deg, to_sensor,         # noqa: E402
                          unit_from_angles)

ORIGIN64_RINGS = np.array([0.5, 0.0, 1.86])   # pandaset_rings.ORIGIN (0, 0.5, 1.86) in PandaSet axes -> normative
GT_GUESS = np.array([1.0, 0.0, 1.8])
SECTORS = ((-31.0, -10.0), (-10.0, 10.0), (10.0, 31.0))


def frame_speed(track, f):
    a, b = max(f - 1, 0), min(f + 1, len(track.times) - 1)
    return float(np.linalg.norm(track.pos[b] - track.pos[a]) / (track.times[b] - track.times[a]))


# --- PandarGT origin -------------------------------------------------------------------------------------------------

def line_pairs(az, el, max_ahead=6):
    """Consecutive returns of one scan line: for each point, the first of the next `max_ahead` stored points that steps
    0.03-0.2 deg forward in azimuth and < 0.3 deg in elevation (the other two sectors fire 20 deg away)."""
    n = len(az)
    used = np.zeros(n, dtype=bool)
    I, J = [], []
    for k in range(1, max_ahead + 1):
        i = np.arange(n - k)
        j = i + k
        ok = ~used[i] & (az[j] - az[i] > 0.03) & (az[j] - az[i] < 0.2) & (np.abs(el[j] - el[i]) < 0.3)
        I.append(i[ok])
        J.append(j[ok])
        used[i[ok]] = True
    return np.concatenate(I), np.concatenate(J)


def parallax_setup(Q, O):
    az, el, rg = angles_deg(Q - O)
    I, J = line_pairs(az, el)
    jump = np.abs(np.log(rg[J] / rg[I])) > 0.3
    b = np.floor(az[I] + 31).astype(np.int64) * 40 + np.floor(el[I] + 20).astype(np.int64)
    slope = (el[J] - el[I]) / (az[J] - az[I])
    step = az[J] - az[I]
    S, T = {}, {}
    for k in np.unique(b[~jump]):
        m = ~jump & (b == k)
        if m.sum() >= 10:
            S[k], T[k] = np.median(slope[m]), np.median(step[m])
    m = jump & np.isin(b, list(S.keys()))
    return I[m], J[m], np.array([S[k] for k in b[m]]), np.array([T[k] for k in b[m]]), len(I)


def parallax_resid(O, Q, I, J, s, st):
    az, el, _ = angles_deg(Q - O)
    daz = az[J] - az[I]
    return np.concatenate([(el[J] - el[I]) - s * daz, 0.5 * (daz - st)])


def gt_origin_frame(job):
    seq, f = job
    track = PoseTrack(ROOT, seq)
    xyz, t, _ = raw_frame(seq, f, 1)
    out = {'seq': seq, 'frame': f, 'speed': frame_speed(track, f), 'n': len(xyz)}
    for mode in ('static', 'interp'):
        Q = track.ego_static(xyz, f) if mode == 'static' else track.ego_at(xyz, t)
        O = GT_GUESS.copy()
        for _ in range(2):                                   # re-pair about the first fit, then refit
            I, J, s, st, n_pairs = parallax_setup(Q, O)
            if len(I) < 30:
                break
            fit = least_squares(parallax_resid, O, args=(Q, I, J, s, st), loss='soft_l1', f_scale=0.02)
            O = fit.x
        r0 = parallax_resid(GT_GUESS, Q, I, J, s, st) if len(I) else np.zeros(1)
        out[mode] = {'origin': O.tolist(), 'n_pairs': int(n_pairs), 'n_jump_pairs': int(len(I)),
                     'mad_guess_deg': float(np.median(np.abs(r0))),
                     'mad_fit_deg': float(np.median(np.abs(parallax_resid(O, Q, I, J, s, st)))) if len(I) else None}
    return out


# --- Pandar64 --------------------------------------------------------------------------------------------------------

def p64_resid(p, Q, ch):
    q = to_sensor(Q, p[:3], rot_from_roll_pitch(p[3], p[4]))
    _, el, _ = angles_deg(q)
    return el - p[5:][ch]


def p64_frame(job):
    seq, f = job
    track = PoseTrack(ROOT, seq)
    xyz, t, _ = raw_frame(seq, f, 0)
    try:
        ch = ring_channels(seq, f, len(xyz))
    except (FileNotFoundError, AssertionError) as e:
        return {'seq': seq, 'frame': f, 'skipped': str(e)}
    rs = np.random.default_rng(f + 1000 * int(seq))
    out = {'seq': seq, 'frame': f, 'speed': frame_speed(track, f), 'n': len(xyz)}
    for mode in ('static', 'interp'):
        Q = track.ego_static(xyz, f) if mode == 'static' else track.ego_at(xyz, t)
        rg = np.linalg.norm(Q - ORIGIN64_RINGS, axis=1)
        idx = np.flatnonzero((rg > 3) & (ch >= 0))
        idx = rs.choice(idx, min(30000, len(idx)), replace=False)
        _, el0, _ = angles_deg(Q[idx] - ORIGIN64_RINGS)
        EL0 = np.array([np.median(el0[ch[idx] == c]) if (ch[idx] == c).any() else 0.0 for c in range(64)])
        p0 = np.concatenate([ORIGIN64_RINGS, [0.0, 0.0], EL0])
        fit = least_squares(p64_resid, p0, args=(Q[idx], ch[idx]), loss='soft_l1', f_scale=0.05)
        out[mode] = {'origin': fit.x[:3].tolist(), 'roll': float(fit.x[3]), 'pitch': float(fit.x[4]),
                     'el': fit.x[5:].tolist(), 'mad_rings_origin_deg': float(np.median(np.abs(p64_resid(p0, Q[idx],
                                                                                                    ch[idx])))),
                     'mad_fit_deg': float(np.median(np.abs(fit.fun)))}
    return out


def p64_check_frame(job):
    """With the pooled model: per-frame roll / pitch (elevations and origin fixed), the nearest-elevation row against
    the channel id, and the azimuth step and phase of each channel's returns."""
    seq, f, model = job
    from pcseg.render import RingLattice
    track = PoseTrack(ROOT, seq)
    xyz, t, _ = raw_frame(seq, f, 0)
    try:
        ch = ring_channels(seq, f, len(xyz))
    except (FileNotFoundError, AssertionError) as e:
        return {'seq': seq, 'frame': f, 'skipped': str(e)}
    O, EL = np.array(model['origin']), np.array(model['el'])
    Q = track.ego_at(xyz, t)
    ok = ch >= 0
    rg = np.linalg.norm(Q - O, axis=1)
    sub = np.flatnonzero(ok & (rg > 3))[::3]

    def res(p):
        _, el, _ = angles_deg(to_sensor(Q[sub], O, rot_from_roll_pitch(p[0], p[1])))
        return el - EL[ch[sub]]
    fit = least_squares(res, [model['roll'], model['pitch']], loss='soft_l1', f_scale=0.05)
    out = {'seq': seq, 'frame': f, 'speed': frame_speed(track, f), 'roll': float(fit.x[0]), 'pitch': float(fit.x[1])}
    for name, (roll, pitch) in (('pooled_attitude', (model['roll'], model['pitch'])), ('frame_attitude', fit.x)):
        q = to_sensor(Q, O, rot_from_roll_pitch(roll, pitch))
        lat = RingLattice(EL, 0.2)
        cell, r = lat.assign(q)
        row = np.where(cell >= 0, lat.table_row(np.maximum(cell, 0)), -1)
        res_ = {}
        for lo, hi in ((0, 5), (5, 20), (20, 60), (60, 1000)):
            m = ok & (r >= lo) & (r < hi)
            res_['%d-%dm' % (lo, hi)] = [float((row[m] == ch[m]).mean()) if m.any() else None, int(m.sum())]
        res_['all'] = [float((row[ok] == ch[ok]).mean()), int(ok.sum())]
        out['row_equals_channel_' + name] = res_
    # azimuth structure, per channel in firing order, sensor frame at the frame attitude
    az, el, r = angles_deg(to_sensor(Q, O, rot_from_roll_pitch(*fit.x)))
    DAZ = rings_module().DAZ
    steps, phases = [], []
    for c in range(64):
        m = np.flatnonzero(ch == c)
        if len(m) < 100:
            continue
        d = (np.diff(az[m]) + 180) % 360 - 180
        steps.append(d[np.abs(d) < 1.0])
        phases.append((az[m] - DAZ[c]) % 0.2)
    steps = np.concatenate(steps)
    ph = np.concatenate(phases)
    # circular mean / spread of the phase on a 0.2-deg circle
    ang = ph / 0.2 * 2 * np.pi
    R = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
    out['az_step_deg'] = np.percentile(steps, [5, 25, 50, 75, 95]).tolist()
    out['az_phase_deg'] = float((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) / (2 * np.pi) * 0.2)
    out['az_phase_resultant'] = float(R)          # 1 = every return on one 0.2-deg grid, 0 = uniform
    out['scan_ms'] = float((t.max() - t.min()) * 1e3)
    return out


# --- PandarGT lattice ------------------------------------------------------------------------------------------------

def kabsch(u, v):
    """Rotation R minimising sum |R u_i - v_i|^2."""
    U, _, Vt = np.linalg.svd(u.T @ v)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def register(u, tpl, tols=(0.08, 0.05, 0.04)):
    """Small rotation taking a frame's unit directions u onto the template's rays (nearest-ray ICP)."""
    R = np.eye(3)
    for tol in tols:
        a, e, _ = angles_deg(u @ R.T)
        i, d = tpl.nearest(a, e)
        m = d < tol
        if m.sum() < 100:
            break
        R = kabsch((u @ R.T)[m], unit_from_angles(*tpl.az_el[i[m]].T)) @ R
    return R


def rot_deg(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_euler('zyx', degrees=True).tolist()     # yaw, pitch, roll


def gt_frame_geometry(job):
    seq, f, O, O64 = job
    track = PoseTrack(ROOT, seq)
    xyz, t, _ = raw_frame(seq, f, 1)
    Q = track.ego_at(xyz, t)
    az, el, rg = angles_deg(Q - O)
    I, J = line_pairs(az, el)
    out = {'seq': seq, 'frame': f, 'speed': frame_speed(track, f), 'n': len(xyz),
           'az_pct': np.percentile(az, [0.1, 50, 99.9]).tolist(), 'el_pct': np.percentile(el, [0.1, 50, 99.9]).tolist(),
           'scan_ms': float((t.max() - t.min()) * 1e3), 'first_el': float(el[0]), 'last_el': float(el[-1]),
           't_first_minus_frame_ms': float((t.min() - track.times[f]) * 1e3),
           'n_unique_t': int(len(np.unique(t)))}
    for k, (lo, hi) in enumerate(SECTORS):
        m = (az[I] >= lo + 1) & (az[I] < hi - 1)
        out['az_step_sector%d' % k] = np.percentile(az[J[m]] - az[I[m]], [25, 50, 75]).tolist() if m.any() else None
        mid = {0: -20.0, 1: 0.0, 2: 20.0}[k]
        # crossings of a 0.12-deg strip, one per line; a crossing = a run of stored indices
        s = np.flatnonzero(np.abs(az - mid) < 0.06)
        if len(s) > 2:
            br = np.flatnonzero(np.diff(s) > 30) + 1
            ge = np.array([np.median(el[g]) for g in np.split(s, br)])
            ge = np.sort(ge)[::-1]
            st = -np.diff(ge)
            out['strip%d' % k] = {'az': mid, 'n_crossings': int(len(ge)), 'el_top': float(ge[0]),
                                  'el_bottom': float(ge[-1]), 'el_step_pct': np.percentile(st, [10, 25, 50, 75]).tolist()}
        out['line_slope_sector%d' % k] = float(np.median((el[J[m]] - el[I[m]]) / (az[J[m]] - az[I[m]]))) \
            if m.any() else None
    # range: the cone as evaluated (static ego frame, xy about the ego origin)
    ego = track.ego_static(xyz, f)
    cone = cone_mask(ego[:, :2], 60.0)
    out['range_pct_cone'] = np.percentile(rg[cone], [50, 90, 99, 99.9, 100]).tolist()
    out['share_cone_beyond_200m'] = float((rg[cone] > 200).mean())
    out['share_cone_beyond_150m'] = float((rg[cone] > 150).mean())
    out['share_in_cone'] = float(cone.mean())
    # Pandar64 range in the same cone, for the 200 m spec
    xyz0, t0, _ = raw_frame(seq, f, 0)
    ego0 = track.ego_static(xyz0, f)
    c0 = cone_mask(ego0[:, :2], 60.0)
    r0 = np.linalg.norm(track.ego_at(xyz0, t0) - O64, axis=1)
    out['p64_range_pct_cone'] = np.percentile(r0[c0], [50, 90, 99, 99.9, 100]).tolist()
    out['p64_share_cone_beyond_200m'] = float((r0[c0] > 200).mean())
    # duplicates: two returns closer than 0.01 deg
    d2, _ = cKDTree(np.c_[az * np.cos(np.radians(el)), el]).query(np.c_[az * np.cos(np.radians(el)), el], k=2)
    out['share_nn_below_0.01deg'] = float((d2[:, 1] < 0.01).mean())
    out['nn_spacing_pct'] = np.percentile(d2[:, 1], [10, 50, 90]).tolist()
    return out


def _frame_dirs(seq, f, O):
    track = PoseTrack(ROOT, seq)
    xyz, t, _ = raw_frame(seq, f, 1)
    a, e, _ = angles_deg(track.ego_at(xyz, t) - O)
    return unit_from_angles(a, e)


def build_template(stationary, extension, O, merge_tol=0.035, ext_tol=0.07, min_count=2):
    """Rays of the PandarGT: (1) merged over STATIONARY frames (the pattern repeats there to ~0.005 deg after a rigid
    registration; moving frames carry 0.02-0.05 deg of non-rigid error from the 10 Hz pose interpolation), each frame
    registered to the growing template; a ray must be seen in >= min_count of them. (2) Extended with the directions of
    moving frames that lie > ext_tol from every ray (lines the stationary scenes never returned, e.g. sky)."""
    sums, counts, tpl = None, None, None
    log = []
    for seq, f in stationary:
        u = _frame_dirs(seq, f, O)
        if sums is None:
            sums, counts = u.copy(), np.ones(len(u))
            tpl = TemplateLattice(np.stack(angles_deg(u)[:2], 1), merge_tol)
            log.append([seq, f, len(u), len(u), 0, 0.0, 'stationary'])
            continue
        R = register(u, tpl)
        u = u @ R.T
        a, e, _ = angles_deg(u)
        i, d = tpl.nearest(a, e)
        within = np.flatnonzero(d < merge_tol)
        o = within[np.lexsort((d[within], i[within]))]
        first = np.ones(len(o), dtype=bool)
        first[1:] = i[o[1:]] != i[o[:-1]]
        claim = o[first]
        np.add.at(sums, i[claim], u[claim])
        counts[i[claim]] += 1
        new = _dedupe(np.flatnonzero(d >= merge_tol), a, e, merge_tol)
        sums = np.vstack([sums, u[new]])
        counts = np.concatenate([counts, np.ones(len(new))])
        mean = sums / np.linalg.norm(sums, axis=1, keepdims=True)
        tpl = TemplateLattice(np.stack(angles_deg(mean)[:2], 1), merge_tol)
        log.append([seq, f, len(u), len(new), int(len(within) - len(claim)), _angle(R), 'stationary'])
        print('template %s/%02d: %d points, %d new rays (total %d), registration %.4f deg' % (
            seq, f, len(u), len(new), len(counts), log[-1][5]), flush=True)
    keep = counts >= min_count
    n_dropped = int((~keep).sum())
    mean = sums[keep] / np.linalg.norm(sums[keep], axis=1, keepdims=True)
    az_el = np.stack(angles_deg(mean)[:2], 1)
    count = counts[keep]
    n_stationary = len(az_el)
    for seq, f in extension:
        tpl = TemplateLattice(az_el, ext_tol)
        u = _frame_dirs(seq, f, O)
        R = register(u, tpl)
        a, e, _ = angles_deg(u @ R.T)
        _, d = tpl.nearest(a, e)
        new = _dedupe(np.flatnonzero(d > ext_tol), a, e, ext_tol)
        az_el = np.vstack([az_el, np.c_[a[new], e[new]]])
        count = np.concatenate([count, np.zeros(len(new))])
        log.append([seq, f, len(u), len(new), 0, _angle(R), 'extension'])
        print('extension %s/%02d: %d new rays (total %d), registration %.4f deg' % (seq, f, len(new), len(az_el),
                                                                                  log[-1][5]), flush=True)
    return az_el, count, n_dropped, n_stationary, log


def _angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _dedupe(idx, a, e, tol):
    """One representative per cluster (radius tol) of the directions idx."""
    if len(idx) == 0:
        return idx
    pts = TemplateLattice._plane(a[idx], e[idx])
    tr = cKDTree(pts)
    alive = np.ones(len(idx), dtype=bool)
    for k in range(len(idx)):
        if alive[k]:
            alive[tr.query_ball_point(pts[k], tol)] = False
            alive[k] = True
    return idx[alive]


def gt_repeat_frame(job):
    seq, f, O, az_el, tol = job
    tpl = TemplateLattice(az_el, tol)
    track = PoseTrack(ROOT, seq)
    xyz, t, _ = raw_frame(seq, f, 1)
    a, e, _ = angles_deg(track.ego_at(xyz, t) - O)
    u = unit_from_angles(a, e)
    out = {'seq': seq, 'frame': f, 'speed': frame_speed(track, f), 'n': len(u)}
    for name, R in (('raw', np.eye(3)), ('registered', register(u, tpl))):
        aa, ee, _ = angles_deg(u @ R.T)
        i, d = tpl.nearest(aa, ee)
        inside = d <= tol
        _, cnt = np.unique(i[inside], return_counts=True)
        out[name] = {'nn_deg_pct': np.percentile(d, [25, 50, 75, 90, 99]).tolist(),
                     'share_within_tol': float(inside.mean()),
                     'share_points_sharing_a_ray': float((cnt[cnt > 1]).sum() / max(inside.sum(), 1)),
                     'rotation_yaw_pitch_roll_deg': rot_deg(R)}
    return out


def stationary_frames(seqs, vmax):
    out = []
    for s in seqs:
        tr = PoseTrack(ROOT, s)
        out += [(s, f) for f in range(len(tr.times)) if frame_speed(tr, f) < vmax]
    return out


def summarize(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    if len(v) == 0:
        return None
    return {'n': int(len(v)), 'median': np.median(v, axis=0).tolist(), 'p25': np.percentile(v, 25, axis=0).tolist(),
            'p75': np.percentile(v, 75, axis=0).tolist(), 'std': v.std(axis=0).tolist(),
            'min': v.min(axis=0).tolist(), 'max': v.max(axis=0).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='gt_origin,p64,gt_lattice')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--fit_frames', default='10,50', help='frame indices of every TRAIN sequence used to FIT')
    ap.add_argument('--check_frames', default='30,70', help='other frame indices, used to CHECK the fitted models')
    ap.add_argument('--max_seqs', type=int, default=0)
    ap.add_argument('--stationary_speed', type=float, default=0.3, help='m/s; template frames')
    ap.add_argument('--procs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    args = ap.parse_args()
    models_path = os.path.join(args.out, 'sensor_models.json')
    if os.path.exists(models_path):
        # Frozen once written (2026-11-22): the render caches and pcocc pin its sha256. A refit goes to a new --out.
        raise SystemExit('%s exists and is frozen: write a refit into a new --out directory' % models_path)
    os.makedirs(args.out, exist_ok=True)
    seqs = split('train')
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    fit_frames = [(s, int(f)) for s in seqs for f in args.fit_frames.split(',')]
    check_frames = [(s, int(f)) for s in seqs for f in args.check_frames.split(',')]
    parts = args.parts.split(',')
    models = {}
    t0 = time.time()
    with Pool(args.procs) as pool:
        if 'gt_origin' in parts:
            rows = pool.map(gt_origin_frame, fit_frames)
            summ = {m: {'origin': summarize([r[m]['origin'] for r in rows]),
                        'mad_fit_deg': summarize([r[m]['mad_fit_deg'] for r in rows]),
                        'mad_guess_deg': summarize([r[m]['mad_guess_deg'] for r in rows])} for m in ('static', 'interp')}
            fast = [r for r in rows if r['speed'] > 8]
            slow = [r for r in rows if r['speed'] < 4]
            for m in ('static', 'interp'):
                summ[m]['origin_fast_gt8ms'] = summarize([r[m]['origin'] for r in fast])
                summ[m]['origin_slow_lt4ms'] = summarize([r[m]['origin'] for r in slow])
            json.dump({'summary': summ, 'frames': rows}, open(os.path.join(args.out, 'sensor_model_gt_origin.json'),
                                                              'w'), indent=1)
            models.setdefault('pandargt', {})['origin'] = summ['interp']['origin']['median']
            models['pandargt']['origin_iqr'] = [summ['interp']['origin']['p25'], summ['interp']['origin']['p75']]
            print('gt_origin', json.dumps(summ['interp']['origin']), '%.0fs' % (time.time() - t0), flush=True)
        if 'p64' in parts:
            rows = [r for r in pool.map(p64_frame, fit_frames)]
            ok = [r for r in rows if 'skipped' not in r]
            summ = {m: {'origin': summarize([r[m]['origin'] for r in ok]),
                        'roll': summarize([r[m]['roll'] for r in ok]), 'pitch': summarize([r[m]['pitch'] for r in ok]),
                        'mad_fit_deg': summarize([r[m]['mad_fit_deg'] for r in ok]),
                        'mad_rings_origin_deg': summarize([r[m]['mad_rings_origin_deg'] for r in ok])}
                    for m in ('static', 'interp')}
            el = np.median(np.array([r['interp']['el'] for r in ok]), axis=0)
            model = {'origin': summ['interp']['origin']['median'], 'roll': summ['interp']['roll']['median'],
                     'pitch': summ['interp']['pitch']['median'], 'el': el.tolist(), 'az_res': 0.2,
                     'origin_iqr': [summ['interp']['origin']['p25'], summ['interp']['origin']['p75']]}
            checks = pool.map(p64_check_frame, [(s, f, model) for s, f in check_frames])
            checks_ok = [c for c in checks if 'skipped' not in c]
            csum = {'roll': summarize([c['roll'] for c in checks_ok]), 'pitch': summarize([c['pitch'] for c in checks_ok]),
                    'az_step_deg_median': summarize([c['az_step_deg'][2] for c in checks_ok]),
                    'az_phase_deg': summarize([c['az_phase_deg'] for c in checks_ok]),
                    'az_phase_resultant': summarize([c['az_phase_resultant'] for c in checks_ok]),
                    'scan_ms': summarize([c['scan_ms'] for c in checks_ok])}
            for name in ('pooled_attitude', 'frame_attitude'):
                for k in checks_ok[0]['row_equals_channel_' + name]:
                    csum['row_equals_channel_%s_%s' % (name, k)] = summarize(
                        [c['row_equals_channel_' + name][k][0] for c in checks_ok])
            json.dump({'summary': summ, 'model': model, 'check_summary': csum, 'frames': rows, 'checks': checks},
                      open(os.path.join(args.out, 'sensor_model_p64.json'), 'w'), indent=1)
            models['pandar64'] = model
            print('p64', json.dumps(model['origin']), model['roll'], model['pitch'], '%.0fs' % (time.time() - t0),
                  flush=True)
        if 'gt_lattice' in parts:
            O = np.array(models['pandargt']['origin'])
            O64 = np.array(models['pandar64']['origin'])
            geo = pool.map(gt_frame_geometry, [(s, f, O, O64) for s, f in fit_frames + check_frames])
            keys_scalar = ['scan_ms', 'first_el', 't_first_minus_frame_ms', 'n_unique_t', 'share_cone_beyond_200m',
                           'share_cone_beyond_150m', 'share_in_cone', 'p64_share_cone_beyond_200m',
                           'share_nn_below_0.01deg', 'n', 'speed']
            gsum = {k: summarize([g[k] for g in geo]) for k in keys_scalar}
            for k in ('az_pct', 'el_pct', 'range_pct_cone', 'p64_range_pct_cone', 'nn_spacing_pct',
                      'az_step_sector0', 'az_step_sector1', 'az_step_sector2', 'line_slope_sector0',
                      'line_slope_sector1', 'line_slope_sector2'):
                gsum[k] = summarize([g[k] for g in geo])
            for k in range(3):
                st = [g['strip%d' % k] for g in geo if 'strip%d' % k in g]
                gsum['strip%d' % k] = {kk: summarize([s[kk] for s in st]) for kk in
                                       ('n_crossings', 'el_top', 'el_bottom', 'el_step_pct')}
            stationary = stationary_frames(seqs, args.stationary_speed)
            az_el, count, n_dropped, n_stationary, log = build_template(stationary, fit_frames, O)
            np.savez(os.path.join(args.out, 'pandargt_template.npz'), az_el=az_el, count=count, origin=O)
            tol = 0.07
            rep = pool.map(gt_repeat_frame, [(s, f, O, az_el, tol) for s, f in check_frames])
            rsum = {name: {k: summarize([r[name][k] for r in rep]) for k in
                           ('nn_deg_pct', 'share_within_tol', 'share_points_sharing_a_ray',
                            'rotation_yaw_pitch_roll_deg')} for name in ('raw', 'registered')}
            # line counts per sector from the template: rays within half an azimuth step of the strip
            step = gsum['az_step_sector1']['median'][1]
            tcount = {}
            for c in (-25, -20, -15, -5, 0, 5, 15, 20, 25):
                m = np.abs(az_el[:, 0] - c) < step / 2
                tcount[str(c)] = {'rays': int(m.sum()), 'el_top': float(az_el[m, 1].max()) if m.any() else None,
                                  'el_bottom': float(az_el[m, 1].min()) if m.any() else None}
            tsum = {'n_rays': int(len(az_el)), 'n_rays_stationary': n_stationary, 'n_dropped_seen_once': n_dropped,
                    'tol_deg': tol, 'stationary_frames': stationary, 'extension_frames': len(fit_frames),
                    'count_pct_stationary': np.percentile(count[:n_stationary], [10, 50, 90]).tolist(),
                    'az_range': [float(az_el[:, 0].min()), float(az_el[:, 0].max())],
                    'el_range': [float(az_el[:, 1].min()), float(az_el[:, 1].max())], 'rays_per_strip': tcount,
                    'build_log': log}
            json.dump({'geometry_summary': gsum, 'template': tsum, 'repeat_summary': rsum, 'geometry': geo,
                       'repeat': rep}, open(os.path.join(args.out, 'sensor_model_gt_lattice.json'), 'w'), indent=1)
            models['pandargt']['template'] = 'pandargt_template.npz'
            models['pandargt']['tol_deg'] = tol
            print('gt_lattice: %d rays; registered NN median %s' % (len(az_el), rsum['registered']['nn_deg_pct']),
                  '%.0fs' % (time.time() - t0), flush=True)
    json.dump(models, open(models_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
