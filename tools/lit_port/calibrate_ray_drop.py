"""One-shot calibration + acceptance check of the label-free ray-drop (experiments_md 20261017_01 §3).

    python3 -I lit_port/calibrate_ray_drop.py calibrate <root> <plan> <calib.pkl> <nus_root> <nus_infos_train.pkl> <out.json>
    python3 -I lit_port/calibrate_ray_drop.py check     <sim_dir> <apply:0|1> <ray_drop.json> <root> <plan> <calib.pkl> <nus_root> <nus_infos_train.pkl> <out.json>

Only unlabelled data: translated Waymo frames (source side) and the .bin scans of nuScenes TRAIN keyframes (no box, no val).
calibrate: translated frames = every 5th stored frame of the sequences with plan index % 20 == 10; real = 1000 TRAIN keyframes
  (infos[::step][:1000]). Refuses to overwrite an existing output (one shot, no sweeping).
check: translated = every 5th stored frame of the sequences with plan index % 20 == 0 (disjoint from the calibration set; the
  40 sequences of the validation); real = 200 TRAIN keyframes at offset step//2 (disjoint from the calibration keyframes).
  apply=1 filters the stored frames of <sim_dir> with the drop; apply=0 reads a directory already cast with the drop.
  PASS iff mean points per frame (3-70 m) within +-5% of real AND every range bin within +-20% of real, except a bin with
  p == 1 (the translated frames are short there; reported, cannot be raised). Exit 1 on FAIL.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ray_drop as rd  # noqa: E402

TOL_TOTAL, TOL_BIN = 0.05, 0.20


def real_counts(nroot, infos, n, offset):
    step = max(1, len(infos) // n)
    sel = infos[offset::step][:n]
    c = np.zeros(rd.NBIN)
    for inf in sel:
        pts = np.fromfile(Path(nroot) / inf['lidar_path'], np.float32).reshape(-1, 5)[:, :3]
        c += rd.bin_counts(np.linalg.norm(pts.astype(np.float64), axis=1))
    return c / len(sel), len(sel)


def trans_counts(sim, plan, calib, mod, p, apply, every=5):
    c = np.zeros(rd.NBIN); n = 0
    seqs = [s['name'] for i, s in enumerate(plan['sequences']) if i % 20 == mod]
    for name in seqs:
        origin = np.asarray(calib[name]['extrinsic'], np.float64).reshape(4, 4)[:3, 3]
        for fp in sorted((Path(sim) / name).glob('*.npz'))[::every]:
            z = np.load(fp)
            pts = z['local_points']; k = int(z['frame_index'])
            if apply:
                pts = pts[rd.keep_mask(pts, origin, p, name, k)]
            c += rd.bin_counts(rd.point_ranges(pts, origin)); n += 1
    return c / n, n, seqs


def main():
    mode = sys.argv[1]
    if mode == 'calibrate':
        root, plan, calib, nroot, ninfos, out = sys.argv[2:8]
        assert not Path(out).exists(), f'{out} exists: the calibration is one-shot, never overwritten'
        plan = json.load(open(plan)); calib = pickle.load(open(calib, 'rb')); infos = pickle.load(open(ninfos, 'rb'))
        R, nr = real_counts(nroot, infos, 1000, 0)
        T, nt, seqs = trans_counts(Path(root) / 'sim_nuscenes', plan, calib, 10, None, False)
        p = np.where(T > 0, np.minimum(1.0, R / np.maximum(T, 1e-9)), 1.0)
        res = dict(edges=rd.EDGES.tolist(), p=p.tolist(), real_per_frame_by_bin=R.tolist(), translated_per_frame_by_bin=T.tolist(),
                   real_frames=nr, translated_frames=nt, calibration_sequences=seqs,
                   real_total=float(R.sum()), translated_total=float(T.sum()),
                   expected_total_after_drop=float((T * p).sum()))
        Path(out).write_text(json.dumps(res, indent=1))
        print('real per frame', R.sum().round(0), 'translated', T.sum().round(0), 'expected after drop', (T * p).sum().round(0))
        print('p by bin', np.round(p, 3).tolist())
        return
    sim, apply, rdj, root, plan, calib, nroot, ninfos, out = sys.argv[2:11]
    cal = json.load(open(rdj)); p = np.array(cal['p'])
    plan = json.load(open(plan)); calib = pickle.load(open(calib, 'rb')); infos = pickle.load(open(ninfos, 'rb'))
    step = max(1, len(infos) // 1000)
    R, nr = real_counts(nroot, infos, 200, step // 2)   # offset step//2: disjoint from the calibration keyframes (multiples of step)
    T, nt, seqs = trans_counts(sim, plan, calib, 0, p, bool(int(apply)))
    rel = T.sum() / R.sum() - 1
    bins = []
    ok = abs(rel) <= TOL_TOTAL
    for b in range(rd.NBIN):
        r = T[b] / R[b] - 1
        exempt = p[b] >= 1.0
        good = abs(r) <= TOL_BIN or exempt
        ok &= good
        bins.append(dict(bin=f'{rd.EDGES[b]:.0f}-{rd.EDGES[b + 1]:.0f}', real=R[b], translated=T[b], rel=r, p=p[b], exempt_p1=bool(exempt), ok=bool(good)))
    res = dict(pass_=bool(ok), total_real=float(R.sum()), total_translated=float(T.sum()), total_rel=float(rel),
               real_frames=nr, translated_frames=nt, sequences=len(seqs), bins=bins, tol_total=TOL_TOTAL, tol_bin=TOL_BIN)
    Path(out).write_text(json.dumps(res, indent=1))
    print(f'total real {R.sum():.0f} translated {T.sum():.0f} rel {rel:+.3f} (tol {TOL_TOTAL})')
    for b in bins:
        print(f"  {b['bin']:>6} m real {b['real']:8.1f} trans {b['translated']:8.1f} rel {b['rel']:+.3f} p {b['p']:.3f}{' (p=1, exempt)' if b['exempt_p1'] else ''} {'ok' if b['ok'] else 'FAIL'}")
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
