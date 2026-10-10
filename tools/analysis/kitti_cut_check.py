"""CPU check for the KITTI arm of the target-oracle degradation table (experiments_md 20261011_02): is the manipulated
evaluation cloud what each twin says it is? Runs as a Slurm CPU job, never on the master.

  A. Rings from the stored order (kitti_rings.rings_from_order) on every KITTI val frame: ring count, descending
     order, the median elevation profile.
  B. Each EVAL_RING_THIN twin on a val subset: share of points kept, recovered rings still present, the kept-ring
     spacing (HDL-32E render), and the count match of each random control against its structured twin, per frame.
  C. Each vertical-FOV cut: share of points removed, with the elevation measured about the sensor (raw frame).

    python3 analysis/kitti_cut_check.py --out /home/koyama/data/kitti_sensitivity/cut_check.json [--n_b 300]
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from pcdet.datasets.kitti import kitti_rings as kr  # noqa: E402

ROOT = '/home/koyama/data/kitti/'
THIN = {'rows:2': ('rows', 2), 'random:2': ('random', 2), 'rows:3': ('rows', 3), 'random:3': ('random', 3),
        'rows:4': ('rows', 4), 'random:4': ('random', 4), 'cols:2': ('cols', 2), 'random_cols:2': ('random_cols', 2),
        'cols:4': ('cols', 4), 'random_cols:4': ('random_cols', 4), 'azbin:0.332': ('azbin', 0.332),
        'random_azbin:0.332': ('random_azbin', 0.332), 'pattern:hdl32e': ('pattern', None)}
PAIRS = [('rows:2', 'random:2'), ('rows:3', 'random:3'), ('rows:4', 'random:4'), ('cols:2', 'random_cols:2'),
         ('cols:4', 'random_cols:4'), ('azbin:0.332', 'random_azbin:0.332')]
ELEV = [('max', 0.0), ('max', -3.0), ('min', -14.0), ('min', -10.0), ('min', -17.6)]


def cfg_of(name):
    mode, par = THIN[name]
    if mode == 'pattern':
        return {'MODE': 'pattern', 'SPACING_DEG': 1.33, 'AZ_RES_DEG': 0.332}
    if mode.endswith('azbin'):
        return {'MODE': mode, 'AZ_RES_DEG': par}
    return {'MODE': mode, 'STRIDE': par}


def main(args):
    t0 = time.time()
    ids = [l.strip() for l in open(ROOT + 'ImageSets/val.txt') if l.strip()]
    load = lambda i: np.fromfile(ROOT + 'training/velodyne/%s.bin' % i, np.float32).reshape(-1, 4)
    out = {}
    # A
    nring, desc, levs, ts = [], [], [], []
    for i in ids:
        p = load(i)
        t = time.time(); ring, lev = kr.rings_from_order(p[:, :3]); ts.append(time.time() - t)
        nring.append(len(lev)); desc.append(int(np.sum(np.diff(lev) >= 0)))    # non-descending neighbour pairs
        if len(lev) == 64:
            levs.append(lev)
    levs = np.array(levs)
    prof = np.median(levs, 0) if len(levs) else np.zeros(0)
    out['A'] = {'frames': len(ids), 'rings': {str(k): int(v) for k, v in zip(*np.unique(nring, return_counts=True))},
                'share_64': float(np.mean(np.array(nring) == 64)),
                'non_descending_pairs': {str(k): int(v) for k, v in zip(*np.unique(desc, return_counts=True))},
                'median_profile_deg': [round(float(x), 3) for x in prof],
                'profile_gaps_deg': [round(float(x), 3) for x in -np.diff(prof)],
                'level_spread_p95_deg': float(np.percentile(np.abs(levs - prof).max(1), 95)) if len(levs) else None,
                'sec_per_frame': float(np.mean(ts))}
    print('A', json.dumps({k: v for k, v in out['A'].items() if 'profile' not in k}), flush=True)
    print('A profile', out['A']['median_profile_deg'], flush=True)
    # B
    rng0 = np.random.default_rng(args.seed)
    sub = [ids[k] for k in sorted(rng0.choice(len(ids), args.n_b, replace=False))]
    stats = {n: {'kept': [], 'rings': [], 'gap': []} for n in THIN}
    pair_ok = {f'{a}|{b}': 0 for a, b in PAIRS}
    for i in sub:
        p = load(i)
        ring, lev = kr.rings_from_order(p[:, :3])
        kept = {}
        for n in THIN:
            rng = np.random.default_rng(zlib.crc32(str(i).encode()))      # the loader's seeding
            keep = kr.eval_ring_thin_mask(p, cfg_of(n), rng)
            kept[n] = int(keep.sum())
            stats[n]['kept'].append(keep.mean())
            stats[n]['rings'].append(len(np.unique(ring[keep])))
            if n == 'pattern:hdl32e':
                kr_ = np.unique(ring[keep])
                if len(kr_) > 1:
                    stats[n]['gap'].append(float(np.median(np.abs(np.diff(np.sort(lev[kr_]))))))
        for a, b in PAIRS:
            pair_ok[f'{a}|{b}'] += int(kept[a] == kept[b])
    out['B'] = {'frames': len(sub), 'seed': args.seed,
                'arms': {n: {'kept_share_mean': float(np.mean(s['kept'])), 'rings_present_median': float(np.median(s['rings'])),
                             'rings_present_min_max': [int(np.min(s['rings'])), int(np.max(s['rings']))],
                             'kept_ring_gap_median_deg': float(np.median(s['gap'])) if s['gap'] else None}
                         for n, s in stats.items()},
                'count_matched_frames': pair_ok}
    for n, s in out['B']['arms'].items():
        print('B', n, json.dumps(s), flush=True)
    print('B count-matched frames (of %d):' % len(sub), json.dumps(pair_ok), flush=True)
    # C
    rem = {f'{s}:{d}': [] for s, d in ELEV}
    for i in sub:
        p = load(i)
        el = kr.elevation_deg(p[:, :3])
        for s, d in ELEV:
            rem[f'{s}:{d}'].append(float(np.mean(el > d)) if s == 'max' else float(np.mean(el < d)))
    out['C'] = {k: float(np.mean(v)) for k, v in rem.items()}
    print('C removed share', json.dumps(out['C']), flush=True)
    out['seconds'] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)
    print('wrote', args.out, '%.0f s' % out['seconds'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_b', type=int, default=300)
    ap.add_argument('--seed', type=int, default=11)
    main(ap.parse_args())
