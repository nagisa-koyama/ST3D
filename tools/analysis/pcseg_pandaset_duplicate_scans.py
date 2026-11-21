"""Census of repeated scans in PandaSet (experiments_md episodes/20261116_01 §8.9, finding 2).

The S3b render-cache build found that PandarGT repeats a scan at the sequence ends (frames 0 / 1 and 78 / 79 hold the
same scan: same absolute point times, points ~3 mm apart). The build only compares frames inside its depth-3 window
(f-1..f+1), only for PandarGT and only on the train + val sequences, so its manifest list is not a census.

This script compares EVERY pair of frames of a sequence, for each device separately (0 = Pandar64, 1 = PandarGT), on
every sequence under the root (labelled or not). Rule, the same as the builder's `dedup_scans`: two frames hold the
same scan when the first and the last point time each agree within 1 ms. For each such pair it also records whether
the point counts are equal and, if so, the per-index distance between the two frames' world points (median, max).

Output (one version directory, never rewritten in place; a changed rule gets a new version):
    <out>/duplicate_scans.json

    python3 analysis/pcseg_pandaset_duplicate_scans.py --root /home/koyama/data/pandaset/dataset \
        --out /home/koyama/data/pcseg/pandaset_duplicate_scans/v1
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from pcseg.pandaset_seg import N_FRAMES, has_semseg                                              # noqa: E402

TOL_S = 1e-3
DEVICES = (0, 1)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def same_scan(a, b, tol_s=TOL_S):
    """a, b: (t_min, t_max) of two frames of one device."""
    return abs(a[0] - b[0]) <= tol_s and abs(a[1] - b[1]) <= tol_s


def pairs_of(spans, tol_s=TOL_S):
    """spans: {frame: (t_min, t_max)}. Every unordered pair (g, h), g < h, holding the same scan."""
    fr = sorted(spans)
    return [(g, h) for i, g in enumerate(fr) for h in fr[i + 1:] if same_scan(spans[g], spans[h], tol_s)]


def run_seq(args):
    root, seq = args
    frames = {}
    for f in range(N_FRAMES):
        p = os.path.join(root, seq, 'lidar', '%02d.pkl.gz' % f)
        if not os.path.isfile(p):
            continue
        lidar = pd.read_pickle(p)
        d = lidar['d'].to_numpy()
        t = lidar['t'].to_numpy()
        xyz = lidar[['x', 'y', 'z']].to_numpy()
        frames[f] = {dev: (t[d == dev], xyz[d == dev]) for dev in DEVICES}
    out = {'seq': seq, 'semseg': has_semseg(root, seq), 'n_frames': len(frames), 'devices': {}}
    for dev in DEVICES:
        spans = {f: (float(v[dev][0].min()), float(v[dev][0].max())) for f, v in frames.items() if len(v[dev][0])}
        rows = []
        for g, h in pairs_of(spans):
            ng, nh = len(frames[g][dev][0]), len(frames[h][dev][0])
            row = {'frames': [g, h], 'n_points': [ng, nh], 'same_count': ng == nh,
                   'dt_first_s': spans[h][0] - spans[g][0], 'dt_last_s': spans[h][1] - spans[g][1]}
            if ng == nh:
                dist = np.linalg.norm(frames[g][dev][1] - frames[h][dev][1], axis=1)
                row['xyz_dist_m'] = {'median': float(np.median(dist)), 'max': float(dist.max())}
            rows.append(row)
        out['devices'][str(dev)] = {'frames_with_points': len(spans), 'pairs': rows}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    path = os.path.join(args.out, 'duplicate_scans.json')
    if os.path.exists(path):
        raise SystemExit('%s exists: a new rule or a rerun goes into a new version directory' % path)
    os.makedirs(args.out, exist_ok=True)
    seqs = sorted(s for s in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, s, 'lidar')))
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        per_seq = list(ex.map(run_seq, [(args.root, s) for s in seqs]))
    here = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.run(['git', '-C', here, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    summary = {}
    for dev in DEVICES:
        k = str(dev)
        withp = [r for r in per_seq if r['devices'][k]['pairs']]
        pos = [(r['seq'], p['frames']) for r in withp for p in r['devices'][k]['pairs']]
        summary[k] = {
            'sequences_with_a_repeat': len(withp),
            'sequences_with_a_repeat_semseg': sum(r['semseg'] for r in withp),
            'pairs': len(pos),
            'pairs_not_0_1_or_78_79': [[s, f] for s, f in pos if f not in ([0, 1], [78, 79])],
            'pairs_with_unequal_counts': sum(not p['same_count'] for r in withp for p in r['devices'][k]['pairs'])}
    result = {
        'rule': 'two frames of one device hold the same scan when first and last point time agree within %g s; every '
                'pair of frames of a sequence is compared' % TOL_S,
        'devices': {'0': 'Pandar64', '1': 'PandarGT'},
        'root': args.root, 'n_sequences': len(per_seq), 'n_sequences_semseg': sum(r['semseg'] for r in per_seq),
        'git_head': head, 'script_sha256': _sha256(os.path.abspath(__file__)),
        'built': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'seconds': time.time() - t0,
        'summary': summary, 'sequences': per_seq}
    with open(path + '.part', 'w') as f:
        json.dump(result, f, indent=1)
    os.rename(path + '.part', path)
    print(json.dumps({k: result[k] for k in ('n_sequences', 'n_sequences_semseg', 'git_head', 'seconds', 'summary')},
                     indent=1))


if __name__ == '__main__':
    main()
