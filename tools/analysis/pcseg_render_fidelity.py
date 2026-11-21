"""S3b render study, questions 3-5 (experiments_md episodes/20261116_01 §8.9): identity-render fidelity of each PandaSet
sensor on its own lattice, and cross-render feasibility between them, with pcseg.render's index-returning z-buffer.

Per TRAIN frame f (one per sequence, --frame), for a TARGET sensor T and a lattice L of T in frame f:
  - 'rays': T's own return directions in frame f (one ray per real return; exact for P1, where both sensors record
    every frame). For Pandar64 also 'ring': the generic RingLattice (fitted elevation table, 0.2 deg columns with the
    frame's azimuth phase and the per-channel offsets DAZ, the frame's roll / pitch).
  - real cells: T's real points of frame f on L (nearest point per cell).
  - renders, all time-aware (each input point is placed with the sensor pose at the firing time of the ray it falls
    on; pcseg.render.assign_time_aware), z-buffer winner kept with its index:
      identity: alone (frame f), acc<k> (frames f-k..f+k), loo<k> (the same without f), loo4_static (one pose for the
                whole scan, the frame's);
      cross:    the OTHER sensor's frame f (alone) or frames f-k..f+k (acc<k>), acc4_static.
  - per config, over the real cells in the 60-degree cone (the evaluation cone, static ego frame) and, for Pandar64
    identity, over all cells: the share of real cells the render occupies; the range error there; the share whose
    carried label (the winner's) equals the real point's (labels are read ONLY here, to measure the renderer); the
    share of rendered cells the real sensor left empty (ring lattice only; a 'rays' lattice has no empty ray).
  - cross, alone and acc4: why real cells stay empty, judged against the SOURCE sensor in frame f: outside its field
    of view, beyond its range, between its rays (no source ray within one source cell), occluded (the source's ray
    there returns nearer), source saw farther, or a source return at the same range (lost to a neighbouring cell).

Labels: ScaLR-17 classes (pcseg.class_map 'pandaset_scalr17'), -1 = ignore. Confusion matrices (real x carried, index
label + 1, 0 = ignore) are summed per config.

    python3 analysis/pcseg_render_fidelity.py --frame 30 --out /home/koyama/data/pcseg/render_study
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcseg_render_common import OUT, ROOT, raw_frame, ring_channels, rings_module, rot_from_roll_pitch, split  # noqa
from pcseg.class_map import build_class_map                                                       # noqa: E402
from pcseg.engine import cone_mask                                                                # noqa: E402
from pcseg.render import (PoseTrack, RingLattice, TemplateLattice, angles_deg, assign_time_aware,  # noqa: E402
                          carry_labels, from_sensor, nearest_time, to_sensor, zbuffer)

NC = 17 + 1           # ScaLR-17 + ignore, index = label + 1
RAY_TOL = 0.07        # deg; a point farther than this from every real return of the frame is off the lattice
RANGE_SPEC = {0: 200.0, 1: 300.0}


def p64_attitude(Q, ch, O, EL, roll0, pitch0):
    sub = np.flatnonzero((ch >= 0) & (np.linalg.norm(Q - O, axis=1) > 3))[::3]

    def res(p):
        _, el, _ = angles_deg(to_sensor(Q[sub], O, rot_from_roll_pitch(p[0], p[1])))
        return el - EL[ch[sub]]
    return least_squares(res, [roll0, pitch0], loss='soft_l1', f_scale=0.05).x


class Sensor:
    """One device in one frame: origin, rotation, its real points (world, time, label, static-ego cone), their exact
    sensor-frame coordinates, its lattices and its firing-time model."""

    def __init__(self, track, f, device, frames, models, DAZ):
        self.device = device
        self.frames = frames                                    # {g: (xyz, t, label)} for every loaded frame g
        xyz, t, lab = frames[f]
        self.xyz, self.t, self.lab = xyz, t, lab
        self.cone = cone_mask(track.ego_static(xyz, f)[:, :2], 60.0)
        if device == 1:
            m = models['pandargt']
            self.O, self.R = np.array(m['origin']), np.eye(3)
        else:
            m = models['pandar64']
            self.O, EL = np.array(m['origin']), np.array(m['el'])
            ch = ring_channels(track.seq, f, len(xyz))
            roll, pitch = p64_attitude(track.ego_at(xyz, t), ch, self.O, EL, m['roll'], m['pitch'])
            self.R = rot_from_roll_pitch(roll, pitch)
            self.roll_pitch = (float(roll), float(pitch))
        self.q = to_sensor(track.ego_at(xyz, t), self.O, self.R)
        az, el, self.r = angles_deg(self.q)
        self.az, self.el = az, el
        self.lattices = {'rays': TemplateLattice(np.c_[az, el], RAY_TOL)}
        if device == 0:
            ok = ch >= 0
            ph = (az[ok] - DAZ[ch[ok]]) % 0.2 / 0.2 * 2 * np.pi
            phase = (np.arctan2(np.sin(ph).mean(), np.cos(ph).mean()) % (2 * np.pi)) / (2 * np.pi) * 0.2
            self.phase = float(phase)
            self.lattices['ring'] = RingLattice(EL, 0.2, az_phase_deg=phase - 0.1, row_daz_deg=DAZ)
            self.lattices['ringfp'] = RingLattice(EL, 0.2, az_phase_deg=phase - 0.1, row_daz_deg=DAZ,
                                                  el_tol_deg=RAY_TOL)
            self.el_fov = (EL.min() - 3.0, EL.max() + 1.9)
            self.ring_el = np.sort(EL)
        # static model: the whole scan seen from the sensor at the frame's pose time
        self.q_static = to_sensor(track.ego_static(xyz, f), self.O, self.R)
        sa, se, _ = angles_deg(self.q_static)
        self.static_lattices = {'rays': TemplateLattice(np.c_[sa, se], RAY_TOL)}
        if device == 0:
            self.static_lattices['ring'] = self.lattices['ring']
            self.static_lattices['ringfp'] = self.lattices['ringfp']
        self.time_of = nearest_time(az, el, t)
        self.t0 = float(np.median(t))
        # coverage per 1-deg azimuth bin (PandarGT field of view from its own returns)
        b = np.floor(az).astype(np.int64)
        self.cov = {k: (el[b == k].min(), el[b == k].max()) for k in np.unique(b) if (b == k).sum() > 20}

    def in_fov(self, az, el):
        if self.device == 0:
            return (el >= self.el_fov[0]) & (el <= self.el_fov[1])
        out = np.zeros(len(az), dtype=bool)
        b = np.floor(az).astype(np.int64)
        for k, (lo, hi) in self.cov.items():
            m = b == k
            out[m] = (el[m] >= lo - 0.2) & (el[m] <= hi + 0.2)
        return out


def gather(sensor, frame_ids):
    xyz = np.concatenate([sensor.frames[g][0] for g in frame_ids])
    lab = np.concatenate([sensor.frames[g][2] for g in frame_ids])
    return xyz, lab


def render_on(track, f, tgt, lat_name, xyz_in, static=False):
    L = (tgt.static_lattices if static else tgt.lattices)[lat_name]
    if static:
        q = to_sensor(track.ego_static(xyz_in, f), tgt.O, tgt.R)
        cell, rng = L.assign(q)
        times = np.full(len(q), track.times[f])
    else:
        cell, q, times = assign_time_aware(track, xyz_in, L, tgt.O, tgt.R, tgt.time_of, tgt.t0)
        rng = np.linalg.norm(q, axis=1)
    cells, win = zbuffer(cell, rng)
    return cells, win, rng, q, times


def real_cells(track, f, tgt, lat_name, static=False):
    L = (tgt.static_lattices if static else tgt.lattices)[lat_name]
    q = tgt.q_static if static else tgt.q
    cell, rng = L.assign(q)
    cells, win = zbuffer(cell, rng)
    return cells, win, rng, cell


def metrics(cR, wR, rR, labR, coneR, cX, wX, rX, labX, coneX):
    """Real cells (cR, real winner wR) against rendered cells (cX, winner wX)."""
    out = {}
    pos = np.searchsorted(cX, cR)
    pos = np.minimum(pos, max(len(cX) - 1, 0))
    hit = (len(cX) > 0) & (cX[pos] == cR) if len(cX) else np.zeros(len(cR), dtype=bool)
    lr = labR[wR]
    lx = np.where(hit, labX[wX[pos]] if len(cX) else -1, -2)
    dr = np.where(hit, (rX[wX[pos]] if len(cX) else 0) - rR[wR], np.nan)
    for region, m in (('cone', coneR[wR]), ('full', np.ones(len(cR), dtype=bool))):
        h = hit & m
        n = int(m.sum())
        d = np.abs(dr[h])
        valid = h & (lr >= 0)
        o = {'real_cells': n, 'hit': int(h.sum()), 'reproduced': float(h.sum() / max(n, 1)),
             'range_abs_err_pct': np.percentile(d, [50, 90]).tolist() if len(d) else None,
             'range_within_0.1m': float((d < 0.1).mean()) if len(d) else None,
             'range_within_0.5m': float((d < 0.5).mean()) if len(d) else None,
             'label_agree': float((lx[valid] == lr[valid]).mean()) if valid.any() else None,
             'label_carried_ignore': float((lx[valid] < 0).mean()) if valid.any() else None,
             'n_label_valid': int(valid.sum())}
        if coneX is not None:
            mx = coneX if region == 'cone' else np.ones(len(cX), dtype=bool)
            extra = mx & ~np.isin(cX, cR)
            o['rendered_cells'] = int(mx.sum())
            o['extra_share'] = float(extra.sum() / max(mx.sum(), 1))
        out[region] = o
    conf = np.zeros((NC, NC), dtype=np.int64)
    m = hit & coneR[wR]
    np.add.at(conf, (lr[m] + 1, lx[m] + 1), 1)
    return out, conf, hit


def miss_reasons(track, f, tgt, src, wR_missing):
    """Why the target's real cells (real winner indices wR_missing) got nothing from the source, judged against the
    source's frame f: outside its field of view; beyond its range spec; between its lines (Pandar64: no ring within
    RAY_TOL of the direction's elevation; PandarGT: no return within RAY_TOL); otherwise, by the source's nearest
    return within 0.3 deg: none, nearer (occluded), farther (source saw past), or about the same range (the source hit
    the same surface but its sample fell outside the target ray's tolerance, e.g. the 0.2-deg azimuth step)."""
    xyz = tgt.xyz[wR_missing]
    if len(xyz) == 0:
        return {}
    L = src.lattices['rays']
    cell, q, _ = assign_time_aware(track, xyz, TemplateLattice(L.az_el, 180.0), src.O, src.R, src.time_of, src.t0)
    az, el, rp = angles_deg(q)
    fov = ~src.in_fov(az, el)
    far = ~fov & (rp > RANGE_SPEC[src.device])
    rest = ~fov & ~far
    k = cell                                               # nearest source return
    d_ang = np.hypot((az - src.az[k]) * np.cos(np.radians(el)), el - src.el[k])
    if src.device == 0:
        d_line = np.min(np.abs(src.ring_el[None, :] - el[:, None]), axis=1)
    else:
        d_line = d_ang
    between = rest & (d_line > RAY_TOL)
    covered = rest & ~between
    near = d_ang < 0.3
    rk = src.r[k]
    margin = np.maximum(0.5, 0.02 * rp)
    none = covered & ~near
    occl = covered & near & (rk < rp - margin)
    farther = covered & near & (rk > rp + margin)
    same = covered & near & ~(rk < rp - margin) & ~(rk > rp + margin)
    n = len(xyz)
    return {'n': n, 'fov': float(fov.sum() / n), 'beyond_range': float(far.sum() / n),
            'between_source_lines': float(between.sum() / n), 'no_source_return_within_0.3deg': float(none.sum() / n),
            'occluded': float(occl.sum() / n), 'source_saw_farther': float(farther.sum() / n),
            'source_same_surface': float(same.sum() / n)}


def run_frame(job):
    seq, f, models, K = job
    t_start = time.time()
    track = PoseTrack(ROOT, seq)
    track.seq = seq
    cmap = build_class_map('pandaset_scalr17')
    DAZ = rings_module().DAZ
    gs = [g for g in range(f - K, f + K + 1) if 0 <= g < len(track.times)]
    frames = {d: {} for d in (0, 1)}
    for g in gs:
        for d in (0, 1):
            xyz, t, _, raw = raw_frame(seq, g, d, labels=True)
            frames[d][g] = (xyz, t, cmap(raw).astype(np.int64))
    sens = {d: Sensor(track, f, d, frames[d], models, DAZ) for d in (0, 1)}
    res = {'seq': seq, 'frame': f, 'speed': float(np.linalg.norm(track.pos[f + 1] - track.pos[f - 1]) / (
        track.times[f + 1] - track.times[f - 1])), 'n0': len(sens[0].xyz), 'n1': len(sens[1].xyz),
        'p64_roll_pitch': sens[0].roll_pitch, 'p64_phase': sens[0].phase, 'configs': {}, 'reasons': {}}
    confs = {}
    for T in (0, 1):
        tgt, src = sens[T], sens[1 - T]
        for lat in tgt.lattices:
            for static in (False, True):
                cR, wR, rR, _ = real_cells(track, f, tgt, lat, static)
                configs = []
                if static:
                    configs = [('identity', 'loo4_static', [g for g in gs if g != f and abs(g - f) <= 4]),
                               ('cross', 'acc4_static', [g for g in gs if abs(g - f) <= 4])]
                else:
                    for k in (0, 1, 2, 4):
                        configs.append(('identity', 'alone' if k == 0 else 'acc%d' % k,
                                        [g for g in gs if abs(g - f) <= k]))
                    for k in (1, 4, 8):
                        configs.append(('identity', 'loo%d' % k, [g for g in gs if g != f and abs(g - f) <= k]))
                    for k in (0, 1, 2, 4, 8):
                        configs.append(('cross', 'alone' if k == 0 else 'acc%d' % k,
                                        [g for g in gs if abs(g - f) <= k]))
                for kind, name, ids in configs:
                    S = tgt if kind == 'identity' else src
                    xyz_in, lab_in = gather(S, ids)
                    cX, wX, rX, qX, tX = render_on(track, f, tgt, lat, xyz_in, static)
                    labX = carry_labels(lab_in, wX)
                    coneX = None
                    if lat in ('ring', 'ringfp'):
                        L = tgt.lattices[lat]
                        p = from_sensor(L.rays(cX) * rX[wX][:, None], tgt.O, tgt.R)
                        world = track.world_from_ego_at(p, tX[wX])
                        coneX = cone_mask(track.ego_static(world, f)[:, :2], 60.0)
                    m, conf, hit = metrics(cR, wR, rR, tgt.lab, tgt.cone, cX, np.arange(len(cX)), rX[wX], labX,
                                           coneX)
                    key = '%s/%s/%s/%s' % ({0: 'p64', 1: 'gt'}[T], lat, kind, name)
                    m['n_input'] = int(len(xyz_in))
                    m['depth'] = len(ids)
                    res['configs'][key] = m
                    confs[key] = conf
                    if kind == 'cross' and name in ('alone', 'acc4', 'acc8') and lat == 'rays':
                        miss = wR[~hit & tgt.cone[wR]]
                        res['reasons'][key] = miss_reasons(track, f, tgt, src, miss)
    res['seconds'] = time.time() - t_start
    print('%s/%02d done in %.0fs' % (seq, f, res['seconds']), flush=True)
    return res, confs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', type=int, default=30)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--tag', default='')
    ap.add_argument('--max_seqs', type=int, default=0)
    ap.add_argument('--procs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    args = ap.parse_args()
    models = json.load(open(os.path.join(args.out, 'sensor_models.json')))
    seqs = split('train')
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    jobs = [(s, args.frame, models, args.K) for s in seqs]
    with Pool(args.procs) as pool:
        out = pool.map(run_frame, jobs)
    rows = [r for r, _ in out]
    conf = {}
    for _, c in out:
        for k, v in c.items():
            conf[k] = conf.get(k, 0) + v
    summary = {}
    for key in rows[0]['configs']:
        s = {}
        for region in ('cone', 'full'):
            vals = [r['configs'][key][region] for r in rows]
            hits = sum(v['hit'] for v in vals)
            s[region] = {'reproduced_pooled': hits / max(sum(v['real_cells'] for v in vals), 1),
                         'reproduced_frame_median': float(np.median([v['reproduced'] for v in vals])),
                         'reproduced_frame_p10': float(np.percentile([v['reproduced'] for v in vals], 10)),
                         'range_median_abs_err_frame_median': float(np.median(
                             [v['range_abs_err_pct'][0] for v in vals if v['range_abs_err_pct']])),
                         'range_p90_abs_err_frame_median': float(np.median(
                             [v['range_abs_err_pct'][1] for v in vals if v['range_abs_err_pct']])),
                         'range_within_0.1m_frame_median': float(np.median(
                             [v['range_within_0.1m'] for v in vals if v['range_within_0.1m'] is not None])),
                         'label_agree_pooled': float(sum((v['label_agree'] or 0) * v['n_label_valid'] for v in vals)
                                                     / max(sum(v['n_label_valid'] for v in vals), 1)),
                         'label_agree_frame_median': float(np.median(
                             [v['label_agree'] for v in vals if v['label_agree'] is not None]))}
            if 'extra_share' in vals[0]:
                s[region]['extra_share_frame_median'] = float(np.median([v['extra_share'] for v in vals]))
        c = conf[key]
        diag = np.diag(c)[1:]
        tot = c[1:].sum(1)
        s['label_agree_per_class_cone'] = (diag / np.maximum(tot, 1)).round(4).tolist()
        s['real_points_per_class_cone'] = tot.tolist()
        summary[key] = s
    reasons = {}
    for key in rows[0]['reasons']:
        vals = [r['reasons'][key] for r in rows if r['reasons'].get(key)]
        n = sum(v['n'] for v in vals)
        reasons[key] = {'n_missing_cells': n, **{k: sum(v[k] * v['n'] for v in vals) / max(n, 1)
                                                for k in vals[0] if k != 'n'}}
    res = {'frame': args.frame, 'K': args.K, 'n_frames': len(rows), 'class_names': list(
        build_class_map('pandaset_scalr17').names), 'summary': summary, 'missing_reasons': reasons,
        'speed': [r['speed'] for r in rows], 'frames': rows,
        'confusion_cone': {k: v.tolist() for k, v in conf.items()}}
    path = os.path.join(args.out, 'fidelity_f%02d%s.json' % (args.frame, args.tag))
    json.dump(res, open(path, 'w'), indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
