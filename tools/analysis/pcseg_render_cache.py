"""Build the offline, frame-keyed S3b render cache that pcseg/render_cache.py reads (experiments_md episodes/20261116_01
§8.9, "b1/b2 preparation"). CPU only.

Kinds (source = PandarGT, device 1, in both):
  gt_on_p64ring  PandarGT frames f-K..f+K rendered onto the Pandar64 lattice of frame f (arms b1 / b2):
                 pcseg.render.RingLattice with the fitted elevation table (render_study/sensor_models.json), 0.2 deg
                 columns with the per-channel offsets DAZ (pcdet pandaset_rings, read only), the frame's azimuth phase,
                 the frame's roll / pitch fitted on its ring channels, and the firing time of the nearest Pandar64
                 return of frame f. Everything about the target comes from the Pandar64 CLOUD of frame f (label-free).
  gt_on_gtrays   PandarGT frames f-K..f+K rendered onto PandarGT frame f's own return directions (the identity-render
                 arm of b1, §8.7 item 1): pcseg.render.TemplateLattice, firing time of the nearest own return.

Fixed before any row (declared in 20261116_01 §8.9 before this script ran):
  - depth: K = 1 (frames f-1, f, f+1, clipped at the sequence ends), the label-free rule "smallest depth that fills
    >= 80% of the target-lattice cells" of the render study. A source frame that repeats another's scan (same absolute
    point times; PandarGT does this at the sequence ends) is used once (`dedup_scans`);
  - footprint / tolerance: half the target's row spacing in its densest band. Pandar64: 0.167 deg / 2 (Hesai's
    published vertical resolution between channels 6 and 54; the fitted table's 48 densest gaps are 0.161-0.2 deg).
    PandarGT: 0.15 deg / 2 (its median line spacing, measured on TRAIN clouds by pcseg_render_sensor_model.py; Hesai
    publishes none). The render study's 0.07 deg was chosen, not derived, and is not used here;
  - labels: the z-buffer winner's (pcseg.render.carry_labels), applied at LOAD time by index; no fill, so no -1 index.
  - `mixed` (stored, never read by the loader): another valid class among the cell's candidates within
    max(0.5 m, 2%) of the winner's range. A diagnostic of the winner rule, from the SOURCE labels only.

Per frame the script also records, label-free, the share of the real target sensor's cells (frame f, in the 60-degree
cone) that the render occupies, at depth K and (every 10th frame) at depth 0, and the share of rendered cone cells the
real sensor left empty.

    python3 analysis/pcseg_render_cache.py --kind gt_on_p64ring --split train val --out /home/koyama/data/pcseg/render_cache/v1
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcseg_render_common import OUT, ROOT, raw_frame, ring_channels, rings_module, rot_from_roll_pitch, split  # noqa
from pcseg.class_map import build_class_map                                                       # noqa: E402
from pcseg.engine import cone_mask                                                                # noqa: E402
from pcseg.pandaset_seg import N_FRAMES                                                           # noqa: E402
from pcseg.render import (PoseTrack, RingLattice, TemplateLattice, angles_deg, assign_time_aware,  # noqa: E402
                          from_sensor, nearest_time, to_sensor, zbuffer)

SOURCE_DEVICE = 1
FOOTPRINT_P64_DEG = 0.167 / 2
TOL_GT_DEG = 0.15 / 2
KINDS = {'gt_on_p64ring': {'target': 'Pandar64 RingLattice (frame f, footprint)', 'tol': FOOTPRINT_P64_DEG},
         'gt_on_gtrays': {'target': 'PandarGT own rays (frame f)', 'tol': TOL_GT_DEG}}
OFFSETS = (-1, 0, 1)
CONE_DEG = 60.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def p64_attitude(Q, ch, O, EL, roll0, pitch0):
    """Roll / pitch of the Pandar64 in frame f from its ring channels (pcseg_render_fidelity.py's fit, unchanged)."""
    sub = np.flatnonzero((ch >= 0) & (np.linalg.norm(Q - O, axis=1) > 3))[::3]

    def res(p):
        _, el, _ = angles_deg(to_sensor(Q[sub], O, rot_from_roll_pitch(p[0], p[1])))
        return el - EL[ch[sub]]
    return least_squares(res, [roll0, pitch0], loss='soft_l1', f_scale=0.05).x


def target_model(kind, track, f, xyz, t, models, DAZ):
    """(lattice, origin, rotation, time_of_direction, t0, extra) of the target sensor in frame f, from its CLOUD."""
    if kind == 'gt_on_p64ring':
        m = models['pandar64']
        O, EL = np.array(m['origin']), np.array(m['el'])
        ch = ring_channels(track.seq, f, len(xyz))
        roll, pitch = p64_attitude(track.ego_at(xyz, t), ch, O, EL, m['roll'], m['pitch'])
        R = rot_from_roll_pitch(roll, pitch)
        q = to_sensor(track.ego_at(xyz, t), O, R)
        az, el, _ = angles_deg(q)
        ok = ch >= 0
        ph = (az[ok] - DAZ[ch[ok]]) % 0.2 / 0.2 * 2 * np.pi
        phase = (np.arctan2(np.sin(ph).mean(), np.cos(ph).mean()) % (2 * np.pi)) / (2 * np.pi) * 0.2
        L = RingLattice(EL, 0.2, az_phase_deg=phase - 0.1, row_daz_deg=DAZ, el_tol_deg=FOOTPRINT_P64_DEG)
        extra = {'roll': float(roll), 'pitch': float(pitch), 'phase': float(phase)}
    else:
        O, R = np.array(models['pandargt']['origin']), np.eye(3)
        q = to_sensor(track.ego_at(xyz, t), O, R)
        az, el, _ = angles_deg(q)
        L = TemplateLattice(np.c_[az, el], TOL_GT_DEG)
        extra = {}
    return L, O, R, nearest_time(az, el, t), float(np.median(t)), q, extra


def render_frame(track, f, L, O, R, time_of, t0, xyz_in):
    cell, q, times = assign_time_aware(track, xyz_in, L, O, R, time_of, t0)
    rng = np.linalg.norm(q, axis=1)
    cells, win = zbuffer(cell, rng)
    p = from_sensor(L.rays(cells) * rng[win][:, None], O, R)
    world = track.world_from_ego_at(p, times[win])
    return cell, rng, cells, win, track.ego_static(world, f)


def dedup_scans(f, src, tol_s=1e-3):
    """PandarGT repeats a scan at the sequence ends: frames 0 and 1 (and 78 and 79) can hold the SAME scan, same
    absolute point times, points ~3 mm apart (found by this script's smoke on sequence 035). Keep one copy of each scan:
    the frame closest to f (f itself first, then the earlier). Returns (kept frames ascending, dropped frames)."""
    kept, dropped = [], []
    for g in sorted(src, key=lambda g: (abs(g - f), g)):
        t = src[g][1]
        span = (t.min(), t.max(), len(t))
        if any(abs(span[0] - k[1][0]) <= tol_s and abs(span[1] - k[1][1]) <= tol_s for k in kept):
            dropped.append(int(g))
        else:
            kept.append((g, span))
    return sorted(g for g, _ in kept), dropped


def mixed_flag(cell, rng, lab, cells, win):
    sel = np.flatnonzero(cell >= 0)
    pos = np.searchsorted(cells, cell[sel])
    w = win[pos]
    near = np.abs(rng[sel] - rng[w]) <= np.maximum(0.5, 0.02 * rng[w])
    diff = near & (lab[sel] >= 0) & (lab[w] >= 0) & (lab[sel] != lab[w])
    mixed = np.zeros(len(cells), dtype=bool)
    mixed[pos[diff]] = True
    return mixed


def fill_stats(track, f, L, q_real, xyz_real, cells, xyz_out):
    c_real, _ = L.assign(q_real)
    cR, wR = zbuffer(c_real, np.linalg.norm(q_real, axis=1))
    cone_real = cone_mask(track.ego_static(xyz_real[wR], f)[:, :2], CONE_DEG)
    hit = np.isin(cR, cells)
    cone_out = cone_mask(xyz_out[:, :2], CONE_DEG)
    extra = cone_out & ~np.isin(cells, cR)
    return {'real_cells_cone': int(cone_real.sum()), 'fill_cone': float(hit[cone_real].mean()),
            'rendered_cone': int(cone_out.sum()), 'extra_share_cone': float(extra.sum() / max(cone_out.sum(), 1))}


def run_seq(job):
    seq, frames, kind, K, out = job
    t_start = time.time()
    models = json.load(open(os.path.join(OUT, 'sensor_models.json')))
    DAZ = rings_module().DAZ
    cmap = build_class_map('pandaset_scalr17')
    track = PoseTrack(ROOT, seq)
    track.seq = seq
    tgt_device = 0 if kind == 'gt_on_p64ring' else 1
    buf = {}

    def get(d, g, labels=False):
        key = (d, g, labels)
        if key not in buf:
            r = raw_frame(seq, g, d, labels=labels)
            buf[key] = (r[0], r[1], cmap(r[3]).astype(np.int64)) if labels else (r[0], r[1])
        return buf[key]

    rows = []
    os.makedirs(os.path.join(out, seq), exist_ok=True)
    for f in frames:
        for key in [k for k in buf if k[1] < f - K - 1]:
            del buf[key]
        cand = [g for g in range(f - K, f + K + 1) if 0 <= g < N_FRAMES]
        src_all = {g: get(SOURCE_DEVICE, g, labels=True) for g in cand}
        gs, dup = dedup_scans(f, src_all)
        src = {g: src_all[g] for g in gs}
        xyz_in = np.concatenate([src[g][0] for g in gs])
        lab_in = np.concatenate([src[g][2] for g in gs])
        off_in = np.concatenate([np.full(len(src[g][0]), g - f, np.int8) for g in gs])
        idx_in = np.concatenate([np.arange(len(src[g][0]), dtype=np.int32) for g in gs])
        n_src = np.array([len(src[f + o][0]) if (f + o) in src else -1 for o in OFFSETS], dtype=np.int32)
        xyz_t, t_t = src[f][:2] if tgt_device == SOURCE_DEVICE else get(tgt_device, f)
        L, O, R, time_of, t0, q_real, extra = target_model(kind, track, f, xyz_t, t_t, models, DAZ)
        cell, rng, cells, win, xyz_out = render_frame(track, f, L, O, R, time_of, t0, xyz_in)
        mixed = mixed_flag(cell, rng, lab_in, cells, win)
        np.savez(os.path.join(out, seq, '%02d.npz' % f), src_off=off_in[win], src_idx=idx_in[win],
                 xyz=xyz_out.astype(np.float32), n_src=n_src, mixed=mixed)
        row = {'seq': seq, 'frame': f, 'depth': len(gs), 'duplicate_scans_dropped': dup,
               'n_in': int(len(xyz_in)), 'n_out': int(len(cells)),
               'n_from_f': int((off_in[win] == 0).sum()), 'mixed': float(mixed.mean()),
               'mixed_per_class': np.bincount(lab_in[win][mixed] + 1, minlength=18).tolist(),
               'out_per_class': np.bincount(lab_in[win] + 1, minlength=18).tolist(), **extra,
               **fill_stats(track, f, L, q_real, xyz_t, cells, xyz_out)}
        if K > 0 and f % 10 == 0:
            m0 = off_in == 0
            _, _, c0, _, x0 = render_frame(track, f, L, O, R, time_of, t0, xyz_in[m0])
            row['depth1'] = fill_stats(track, f, L, q_real, xyz_t, c0, x0)
        rows.append(row)
    print('%s: %d frames in %.0fs' % (seq, len(frames), time.time() - t_start), flush=True)
    return rows


def git_head():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(['git', '-C', here, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:  # noqa: BLE001 - the container may not carry git; the script hash is recorded either way
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', required=True, choices=sorted(KINDS))
    ap.add_argument('--split', nargs='+', default=['train', 'val'])
    ap.add_argument('--K', type=int, default=1, help='half-depth: frames f-K..f+K (1 = the declared depth 3)')
    ap.add_argument('--out', required=True, help='cache root; the kind goes to <out>/<kind>[_k<K> if K != 1]')
    ap.add_argument('--frames', type=int, nargs='*', default=None, help='only these frames (checks)')
    ap.add_argument('--max_seqs', type=int, default=0)
    ap.add_argument('--procs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    args = ap.parse_args()
    if args.K not in (0, 1):
        raise ValueError('K must be 0 (the identity check) or 1 (the declared depth 3)')
    name = args.kind if args.K == 1 else '%s_k%d' % (args.kind, args.K)
    out = os.path.join(args.out, name)
    if os.path.exists(os.path.join(out, 'manifest.json')):
        raise FileExistsError('%s exists: a cache is never rebuilt in place (new version dir instead)' % out)
    os.makedirs(out, exist_ok=True)
    seqs = [s for sp in args.split for s in split(sp)]
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    frames = list(range(N_FRAMES)) if args.frames is None else list(args.frames)
    jobs = [(s, frames, args.kind, args.K, out) for s in seqs]
    t0 = time.time()
    with Pool(args.procs) as pool:
        rows = [r for rs in pool.imap_unordered(run_seq, jobs) for r in rs]
    rows.sort(key=lambda r: (r['seq'], r['frame']))
    here = os.path.dirname(os.path.abspath(__file__))
    files = {p: sha256(os.path.join(here, p)) for p in (
        'pcseg_render_cache.py', 'pcseg_render_common.py', '../../pcseg/render.py',
        '../../pcdet/datasets/pandaset/pandaset_rings.py')}
    files['sensor_models.json'] = sha256(os.path.join(OUT, 'sensor_models.json'))

    def med(key, sub=None):
        v = [(r[sub] if sub else r)[key] for r in rows if (sub is None or sub in r)]
        return {'median': float(np.median(v)), 'p10': float(np.percentile(v, 10)), 'n': len(v)} if v else None
    summary = {'frames_with_duplicate_scans': [[r['seq'], r['frame'], r['duplicate_scans_dropped']] for r in rows
                                               if r['duplicate_scans_dropped']],
               'fill_cone': med('fill_cone'), 'extra_share_cone': med('extra_share_cone'), 'mixed': med('mixed'),
               'n_out': med('n_out'), 'depth1_fill_cone': med('fill_cone', 'depth1'),
               'mixed_per_class': np.sum([r['mixed_per_class'] for r in rows], axis=0).tolist(),
               'out_per_class': np.sum([r['out_per_class'] for r in rows], axis=0).tolist()}
    manifest = {'kind': args.kind, 'source_device': SOURCE_DEVICE, 'target': KINDS[args.kind]['target'],
                'depth': 2 * args.K + 1, 'K': args.K, 'tolerance_deg': KINDS[args.kind]['tol'],
                'label_rule': 'z-buffer winner, by index at load; no fill', 'splits': args.split,
                'n_sequences': len(seqs), 'n_frames': len(rows), 'frames': args.frames,
                'git_head': git_head(), 'sha256': files, 'class_names': list(build_class_map('pandaset_scalr17').names),
                'built': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'seconds': time.time() - t0, 'summary': summary}
    with open(os.path.join(out, 'frames.json'), 'w') as f:
        json.dump(rows, f)
    with open(os.path.join(out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('class_names',)}, indent=1))


if __name__ == '__main__':
    main()
