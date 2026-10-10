"""Pandar64 channel labels for every PandaSet frame of a split, cached for RING_PATTERN / RING_FOV_CUT, with per-frame
purity diagnostics (pcdet/datasets/pandaset/pandaset_rings.py; experiments_md 20261010_05).

Source data only, no labels. Run inside the container (numba), from ST3D/tools:

    python analysis/pandaset_ring_cache.py --template            # re-derive EL / OFF_US / DAZ, diff with the module
    python analysis/pandaset_ring_cache.py --split train --out /home/koyama/data/pandaset_rings/v1 --procs 16

Each frame writes <out>/<sequence>/<frame:02d>.npy, int8 channel per device-0 point in STORED order (0 = top laser).
Diagnostics per frame (<out>/diagnostics_<split>.csv): points, blocks found, channels with >= 50 points, and three
label agreements. G = elevation + azimuth column (no timing), T = firing time + azimuth column (no elevation),
F = all three cues (what is cached). G = T is the independent-cue purity check: the two share only the azimuth column
and the stored order. `time_consistent` is the share of F's labels whose firing time matches the template offset to
within 0.3 us of their block's median reference (one timestamp quantum is 0.24 us).
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from pcdet.datasets.pandaset import pandaset_rings as R  # noqa: E402

DATA = '/home/koyama/data/pandaset/dataset'
CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cfgs', 'da-ieee-access', 'da_pandaset_dataset.yaml')


def load_frame(seq, fi):
    import json as _json
    import pandas as pd
    import pandaset as ps
    poses = _json.load(open(os.path.join(DATA, seq, 'lidar', 'poses.json')))
    df = pd.read_pickle(os.path.join(DATA, seq, 'lidar', '%02d.pkl.gz' % fi))
    d0 = df[df.d == 0]
    ego = ps.geometry.lidar_points_to_ego(d0[['x', 'y', 'z']].to_numpy(), poses[fi])
    return ego, (d0.t.to_numpy() - d0.t.min()) * 1e6


def one(args):
    seq, fi, out = args
    path = R.cache_path(out, seq, fi)
    ego, t = load_frame(seq, fi)
    F, nb = R.channel_labels(ego, t)
    G, _ = R.channel_labels(ego, t, use_time=False)
    T, _ = R.channel_labels(ego, t, use_elevation=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, F.astype(np.int8))
    # block reference: the median of t - OFF[label] over each block as F sees it (a new block where the label does
    # not increase)
    x = t - R.OFF_US[F]
    bid = np.concatenate([[0], np.cumsum(np.diff(F) <= 0)])
    order = np.argsort(bid, kind='stable')
    bounds = np.searchsorted(bid[order], np.arange(bid.max() + 2))
    med = np.empty(bid.max() + 1)
    for b in range(bid.max() + 1):
        med[b] = np.median(x[order[bounds[b]:bounds[b + 1]]])
    tc = float(np.mean(np.abs(x - med[bid]) <= 0.3))
    found = int(np.sum(np.bincount(F, minlength=R.N_CHANNELS) >= 50))
    return dict(sequence=seq, frame=fi, points=len(t), blocks=nb, channels_found=found,
                g_eq_t=float(np.mean(G == T)), f_eq_g=float(np.mean(F == G)), f_eq_t=float(np.mean(F == T)),
                time_consistent=tc)


def template(seqs, frames=(5, 45)):
    rows = []
    for s in seqs:
        for fi in frames:
            ego, t = load_frame(s, fi)
            r, el, az = R.sensor_angles(ego)
            bid = np.concatenate([[0], np.cumsum(np.diff(el) > R.BLOCK_JUMP_DEG)])
            cnt = np.bincount(bid)
            st = np.concatenate([[0], np.cumsum(cnt)])
            for b in np.nonzero(cnt == R.N_CHANNELS)[0]:
                i = np.arange(st[b], st[b] + R.N_CHANNELS)
                if t[i].max() - t[i].min() > R.BLOCK_SPAN_US:
                    continue
                da = (az[i] - np.median(az[i]) + 180) % 360 - 180
                rows.append(np.stack([el[i], da, t[i] - t[i].min(), r[i]], 1))
    A = np.stack(rows)
    off = np.median(A[:, :, 2], 0)
    daz = np.median(A[:, :, 1], 0)
    el = np.array([np.median(A[A[:, c, 3] > (10 if c < 62 else 2), c, 0]) for c in range(R.N_CHANNELS)])
    print('complete 64-point blocks: %d; firing-offset sd max %.4f us' % (len(A), max(np.std(A[:, c, 2]) for c in range(64))))
    print('max |diff| vs module: EL %.4f deg, OFF %.4f us, DAZ %.4f deg' % (
        np.abs(el - R.EL).max(), np.abs(off - R.OFF_US).max(), np.abs(daz - R.DAZ).max()))
    print('EL', np.round(el, 3).tolist())
    print('OFF_US', np.round(off, 2).tolist())
    print('DAZ', np.round(daz, 2).tolist())


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument('--template', action='store_true')
    ap.add_argument('--split', default='train')
    ap.add_argument('--out', default='/home/koyama/data/pandaset_rings/v1')
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--frames', type=int, default=80)
    a = ap.parse_args()
    seqs = yaml.safe_load(open(CFG))['SEQUENCES'][a.split]
    if a.template:
        template(seqs[:16])
        return
    jobs = [(s, fi, a.out) for s in seqs for fi in range(a.frames)
            if os.path.exists(os.path.join(DATA, s, 'lidar', '%02d.pkl.gz' % fi))]
    t0 = time.time()
    with Pool(a.procs) as p:
        res = p.map(one, jobs, chunksize=4)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'diagnostics_%s.csv' % a.split), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    keys = ['points', 'blocks', 'channels_found', 'g_eq_t', 'f_eq_g', 'f_eq_t', 'time_consistent']
    summ = {k: {q: float(np.percentile([r[k] for r in res], q)) for q in (0, 1, 5, 50, 95, 100)} for k in keys}
    summ.update(frames=len(res), split=a.split, seconds=time.time() - t0, origin=R.ORIGIN.tolist(),
                module='pcdet/datasets/pandaset/pandaset_rings.py')
    json.dump(summ, open(os.path.join(a.out, 'summary_%s.json' % a.split), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
