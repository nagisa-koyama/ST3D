"""Ring cache of every Lyft 40-beam TRAIN frame for RING_PATTERN on Lyft, and the kept-frame list (experiments_md
20261024_01; design: 20261003_05 §5 'User 10-24 ~02:40' item 3). CPU only; run as a Slurm CPU job.

Method = gate run 2 of 20261010_07 §4b, imported, not re-implemented: `lyft_deskew_rings.frame_rings(p, info,
prm['40'], iters=2, start_rule='first_point', unwrap=True)` with the explore parameters of
`explore_20261010.json` (rotation 0.1 s, compensation reference at the end of the rotation, full angular velocity from
the previous sweep, sweep start at the frame's first point minus 0.5 deg, per-ring phase unwrap) - i.e. the frozen
§4a method plus the unwrap, `lyft_rings.deskew_and_rings`.

Per frame it writes what the loader needs (the rings are stored-order segments, so ring id per point is a
non-decreasing step function of the raw row index and is stored as per-ring point counts):
    n        number of raw rows (the loader's (N, 5) float32 array, truncated to a multiple of 5 as the loader does)
    crc32    zlib.crc32 of that array's x, y, z columns (float32, C order): the loader refuses a mismatch
    counts   points per recovered ring, top laser first (sum == n)
    lev      recovered ring elevations (deg, de-skewed, about the sensor at firing time)
and a per-frame record (ring count, speed, kept rings / kept share under RING_PATTERN with frame-seeded phases).
Kept list = frames with exactly 40 rings (20261010_07 §7, the drop rule; 20261003_05 §5 item 3).

Before the full pass it replays the gate's 200 40-beam frames (seed 202, the gate's draw) and requires every frame's
ring count and kept rings to equal the published frames JSON of 20261022_05 (discrete values equal, floats within
1e-9 relative), so the cache is gate run 2's method, not a near copy.

    python3 analysis/lyft_ring_cache.py --out_dir /home/koyama/data/lyft_rings/v1_40beam --workers 8
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import lyft_deskew_rings as ldr  # noqa: E402  (gate code, unchanged)
from pcdet.datasets.lyft import lyft_rings as lr  # noqa: E402

N_RINGS = 40
SPACING_DEG, AZ_RES_DEG = 1.33, 0.332
GATE_FRAMES_JSON = '/home/koyama/data/lyft_rings/viz_20261022/frames_20261022.json'
_STATE = {}


def raw_rows(path):
    """The loader's raw array: float32, truncated to a multiple of 5, (N, 5) (LyftDataset.get_lidar_with_sweeps)."""
    a = np.fromfile(str(path), dtype=np.float32, count=-1)
    if a.shape[0] % 5 != 0:
        a = a[: a.shape[0] - (a.shape[0] % 5)]
    return a.reshape([-1, 5])


def xyz_crc32(raw):
    return zlib.crc32(np.ascontiguousarray(raw[:, :3], dtype=np.float32).tobytes()) & 0xFFFFFFFF


def frame_phases(stem):
    """Frame-seeded RING_PATTERN phases for the per-frame diagnostic (training draws its own per sample)."""
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    return float(rng.random()), float(rng.random())


def one_frame(k):
    info, prm = _STATE['infos'][k], _STATE['prm']
    stem = Path(info['lidar_path']).stem
    t0 = time.time()
    raw = raw_rows(ldr.ROOT + info['lidar_path'])
    p = raw[:, :3].astype(np.float64)
    q, ring, lev = ldr.frame_rings(p, info, prm, 2, 'first_point', unwrap=True)
    sec = time.time() - t0
    ring = np.asarray(ring, dtype=np.int64)
    assert len(ring) == len(raw), (stem, len(ring), len(raw))
    assert len(ring) == 0 or (ring[0] == 0 and np.all(np.diff(ring) >= 0) and ring[-1] == len(lev) - 1), \
        '%s: rings are not stored-order segments 0..K-1' % stem
    counts = np.bincount(ring, minlength=len(lev)).astype(np.int32)
    assert np.all(counts > 0) and counts.sum() == len(raw), stem
    pv, ph = frame_phases(stem)
    keep, kr = lr.ring_pattern_mask(p, ring, lev, SPACING_DEG, AZ_RES_DEG, pv, ph)
    m = lr.ego_motion(info)
    rec = {'stem': stem, 'token': info['token'], 'host': stem.split('_')[0], 'n': int(len(raw)), 'rings': int(len(lev)),
           'speed_mps': None if m is None else float(np.linalg.norm(m[0][:2])), 'no_motion': m is None,
           'kept_rings': int(len(kr)), 'kept_share': float(keep.mean()) if len(keep) else 0.0,
           'lev_min': float(np.min(lev)) if len(lev) else None, 'lev_max': float(np.max(lev)) if len(lev) else None,
           'min_ring_points': int(counts.min()) if len(counts) else 0, 'sec': sec}
    cache = {'n': int(len(raw)), 'crc32': xyz_crc32(raw), 'counts': counts, 'lev': np.asarray(lev, dtype=np.float64)}
    return stem, rec, cache


def init_worker(infos, prm):
    _STATE['infos'], _STATE['prm'] = infos, prm


def replay_gate(infos, prm):
    """The gate's 40-beam draw (lyft_deskew_rings.gate, seed 202: rng.choice per platform, then two rng.random() per
    frame), checked against the published per-frame records of 20261022_05 (run 2)."""
    pub = {r['file']: r for r in json.load(open(GATE_FRAMES_JSON))['platforms']['40']['records']}
    sel = [i for i in infos if ldr.platform(i) == '40']
    rng = np.random.default_rng(202)
    pick = rng.choice(len(sel), 200, replace=False)
    init_worker(sel, prm)
    bad, n_exact = [], 0
    for k in pick:
        pv, ph = rng.random(), rng.random()
        info = sel[k]
        name = Path(info['lidar_path']).name
        r = pub[name]
        raw = raw_rows(ldr.ROOT + info['lidar_path'])
        p = raw[:, :3].astype(np.float64)
        q, ring, lev = ldr.frame_rings(p, info, prm, 2, 'first_point', unwrap=True)
        keep, kr = lr.ring_pattern_mask(p, ring, lev, SPACING_DEG, AZ_RES_DEG, pv, ph)
        n_exact += int(len(lev) == N_RINGS)
        same = (len(lev) == r['run2']['rings'] and len(kr) == r['run2']['kept_rings'] and len(raw) == r['n_points']
                and abs(pv - r['phase_v']) <= 1e-12 and abs(ph - r['phase_h']) <= 1e-12
                and abs(keep.mean() - r['run2']['kept_share']) <= 1e-9 * max(1.0, abs(r['run2']['kept_share']))
                and np.allclose(np.round(lev, 4), r['run2']['levels'], rtol=0, atol=1.5e-4))
        if not same:
            bad.append(name)
    return {'frames': len(pick), 'exact_40': n_exact, 'mismatches': bad, 'match': not bad}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--explore', default='/home/koyama/data/lyft_rings/explore_20261010.json')
    ap.add_argument('--limit', type=int, default=0, help='first N frames only (smoke)')
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Only what the recovery reads (path, token, sweep poses), in the infos' own order: the gate's draw indexes it,
    # and forked workers then share a few MB instead of the 334 MB pickle's objects.
    infos = [{'lidar_path': i['lidar_path'], 'token': i['token'],
              'sweeps': [{'transform_matrix': s.get('transform_matrix', None), 'time_lag': s.get('time_lag', 0.0)}
                         for s in (i.get('sweeps', []) or [])]}
             for i in pickle.load(open(ldr.ROOT + 'lyft_infos_train.pkl', 'rb'))]
    prm = ldr.params_from_explore(args.explore)['40']
    sha = subprocess.run(['git', '-C', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'),
                          'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    print('ST3D HEAD', sha, 'params', prm, flush=True)

    rep = replay_gate(infos, prm)
    print('GATE REPLAY', json.dumps(rep), flush=True)
    assert rep['match'], 'the gate replay differs from the published run 2: not gate run 2\'s method, stop'

    sel = [i for i in infos if ldr.platform(i) == '40']
    if args.limit:
        sel = sel[:args.limit]
    print('40-beam train frames: %d of %d' % (len(sel), len(infos)), flush=True)
    recs, cache = [], {}
    with mp.get_context('fork').Pool(args.workers, initializer=init_worker, initargs=(sel, prm)) as pool:
        for j, (stem, rec, c) in enumerate(pool.imap(one_frame, range(len(sel)), chunksize=16)):
            recs.append(rec)
            cache[stem] = c
            if (j + 1) % 1000 == 0:
                print('%d / %d frames, %.0f s' % (j + 1, len(sel), time.time() - t0), flush=True)
    assert len(cache) == len(sel), 'duplicate frame stems'
    rings = np.array([r['rings'] for r in recs])
    kept = sorted(r['stem'] for r in recs if r['rings'] == N_RINGS)
    assert not any(s.split('_')[0] in ldr.H64 for s in kept)

    meta = {'method': 'lyft_deskew_rings.frame_rings(iters=2, start_rule=first_point, unwrap=True) = gate run 2 '
                      '(experiments_md 20261010_07 §4b)', 'explore': args.explore, 'params_40': prm,
            'st3d_head': sha, 'n_rings': N_RINGS, 'created': time.strftime('%Y-%m-%d %H:%M %Z')}
    pkl = out / 'rings_v1_40beam.pkl'
    with open(pkl, 'wb') as f:
        pickle.dump({'meta': meta, 'frames': cache}, f, protocol=4)
    kl = out / 'kept_frames_v1_40beam.txt'
    kl.write_text(''.join(s + '\n' for s in kept))
    json.dump(recs, open(out / 'frames_v1_40beam.json', 'w'))

    def hist(a):
        return {str(int(k)): int(v) for k, v in zip(*np.unique(a, return_counts=True))}
    sp = np.array([np.nan if r['speed_mps'] is None else r['speed_mps'] for r in recs])
    bands = [('no_motion', ~np.isfinite(sp)), ('<0.5', sp < 0.5), ('0.5-3', (sp >= 0.5) & (sp < 3)),
             ('3-8', (sp >= 3) & (sp < 8)), ('>=8', sp >= 8)]
    hosts = sorted({r['host'] for r in recs})
    ks = np.array([r['kept_share'] for r in recs])
    kr = np.array([r['kept_rings'] for r in recs])
    exact = rings == N_RINGS
    lev_lo = [r['lev_min'] for r in recs if r['rings'] == N_RINGS]
    lev_hi = [r['lev_max'] for r in recs if r['rings'] == N_RINGS]
    summ = {
        'meta': meta, 'gate_replay': rep,
        'frames_40beam_train': len(recs), 'kept_exact_40': int(exact.sum()), 'dropped': int((~exact).sum()),
        'kept_share': float(exact.mean()), 'dropped_share': float((~exact).mean()),
        'ring_count_hist': hist(rings),
        'by_host': {h: {'frames': int(sum(r['host'] == h for r in recs)),
                        'kept': int(sum(r['host'] == h and r['rings'] == N_RINGS for r in recs))} for h in hosts},
        'by_speed': {b: {'frames': int(m.sum()), 'kept': int((m & exact).sum())} for b, m in bands},
        'pattern_on_kept': {'kept_rings_hist': hist(kr[exact]), 'kept_point_share_mean': float(ks[exact].mean()),
                            'kept_point_share_p5_p50_p95': [float(x) for x in np.percentile(ks[exact], [5, 50, 95])]},
        'ring_elevation_range_on_kept_deg': [float(np.min(lev_lo)), float(np.max(lev_hi))],
        'hdl32e_fov_deg': [-30.67, 10.67],
        'sec_per_frame_mean': float(np.mean([r['sec'] for r in recs])),
        'files': {'rings': str(pkl), 'rings_sha256': sha256(pkl), 'kept_list': str(kl), 'kept_list_sha256': sha256(kl),
                  'kept_list_bytes': kl.stat().st_size},
        'seconds': time.time() - t0}
    json.dump(summ, open(out / 'summary_v1_40beam.json', 'w'), indent=1)
    print('SUMMARY', json.dumps({k: v for k, v in summ.items() if k != 'meta'}), flush=True)


if __name__ == '__main__':
    main()
