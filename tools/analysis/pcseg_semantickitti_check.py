"""CPU 1 of pair P2 (experiments_md episodes/20261128_01 §8): the SemanticKITTI data, every TRAIN and VAL scan.

Checks, per scan of sequences 00-10:
  - label count equals point count;
  - every semantic id (label & 0xFFFF) is declared by the `semantickitti_cl10` class map (which raises otherwise);
  - every remission is a multiple of 0.01 in [0, 1];
  - every coordinate is finite;
and per sequence that the scans are 0..N-1 with N as declared (00: 4,541 ... 10: 1,201; train 19,130, val 4,071).
Also reports, for information only, the per-split counts of the 10 classes and of ignored points (they choose nothing).

    python3 analysis/pcseg_semantickitti_check.py --out /home/koyama/data/pcseg/semantickitti_check/check.json --workers 8
Run as a Slurm CPU job from <worktree>/tools. Exit status 1 on any violation.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
sys.path.insert(0, REPO)

ROOT = '/home/koyama/data/semantic_kitti/dataset/sequences'
EXPECT = {'00': 4541, '01': 1101, '02': 4661, '03': 801, '04': 271, '05': 2761, '06': 1101, '07': 1101, '08': 4071,
          '09': 1591, '10': 1201}


def seq_check(seq):
    from pcseg.class_map import build_class_map
    from pcseg.semantickitti_seg import list_frames, read_points, read_labels
    cm = build_class_map('semantickitti_cl10')
    frames = list_frames(ROOT, seq)
    errs, counts, ignored, pts = [], np.zeros(cm.num_classes, np.int64), 0, 0
    raw_seen = set()
    for f in frames:
        p = read_points(ROOT, seq, f)
        lab = read_labels(ROOT, seq, f)
        k = '%s/%06d' % (seq, f)
        if len(lab) != len(p):
            errs.append('%s: %d points, %d labels' % (k, len(p), len(lab)))
            continue
        if not np.isfinite(p).all():
            errs.append('%s: non-finite values' % k)
        v = p[:, 3].astype(np.float64)
        if not (np.all((v >= 0) & (v <= 1)) and np.all(np.abs(v * 100 - np.round(v * 100)) < 1e-3)):
            errs.append('%s: remission not a multiple of 0.01 in [0, 1]' % k)
        try:
            y = cm(lab)
        except ValueError as e:
            errs.append('%s: %s' % (k, e))
            continue
        raw_seen |= set(np.unique(lab).tolist())
        counts += np.bincount(y[y >= 0], minlength=cm.num_classes)
        ignored += int((y < 0).sum())
        pts += len(p)
    return seq, len(frames), pts, errs, counts.tolist(), ignored, sorted(raw_seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    from pcseg.class_map import CL10_NAMES
    from pcseg.semantickitti_seg import SPLITS
    t0 = time.time()
    seqs = sorted(EXPECT)
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.workers) as pool:
            res = pool.map(seq_check, seqs, chunksize=1)
    else:
        res = [seq_check(s) for s in seqs]
    out = {'root': ROOT, 'sequences': {}, 'errors': [], 'splits': {}}
    for seq, n, pts, errs, counts, ign, raw in res:
        if n != EXPECT[seq]:
            errs = ['%s: %d scans, expected %d' % (seq, n, EXPECT[seq])] + errs
        out['sequences'][seq] = {'scans': n, 'points': pts, 'errors': len(errs), 'raw_ids_seen': raw}
        out['errors'] += errs
    for split, ss in SPLITS.items():
        c = np.sum([r[4] for r in res if r[0] in ss], axis=0)
        out['splits'][split] = {'scans': int(sum(r[1] for r in res if r[0] in ss)),
                                'points': int(sum(r[2] for r in res if r[0] in ss)),
                                'ignored_points': int(sum(r[5] for r in res if r[0] in ss)),
                                'class_points': {n: int(x) for n, x in zip(CL10_NAMES, c)}}
    out['pass'] = not out['errors']
    out['seconds'] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ('pass', 'splits', 'seconds')}, indent=1))
    print('errors: %d%s' % (len(out['errors']), (', first: %s' % out['errors'][0]) if out['errors'] else ''))
    sys.exit(0 if out['pass'] else 1)


if __name__ == '__main__':
    main()
