"""Compare two S3b render caches frame by frame, every array exact (experiments_md episodes/20261116_01 §8.11;
bug_prevention_checklist rule 8). Used to show that a builder change leaves an existing kind unchanged: rebuild a subset
of frames with the new code into a check directory and compare it with the cache built before.

    python3 analysis/pcseg_render_cache_compare.py --new <check>/gt_on_gtrays --old /home/koyama/data/pcseg/render_cache/v1/gt_on_gtrays

Every frame of --new must exist in --old. Exit status 1 on any difference.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

KEYS = ('src_off', 'src_idx', 'xyz', 'n_src', 'mixed')


def compare(new, old):
    out = {'frames': 0, 'differ': []}
    for p in sorted(glob.glob(os.path.join(new, '*', '*.npz'))):
        rel = os.path.relpath(p, new)
        q = os.path.join(old, rel)
        if not os.path.isfile(q):
            out['differ'].append('%s: absent from the old cache' % rel)
            continue
        with np.load(p) as a, np.load(q) as b:
            for k in KEYS:
                if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k]):
                    out['differ'].append('%s: %s differs' % (rel, k))
        out['frames'] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True)
    ap.add_argument('--old', required=True)
    args = ap.parse_args()
    r = compare(args.new, args.old)
    r.update({'new': args.new, 'old': args.old, 'pass': r['frames'] > 0 and not r['differ']})
    with open(os.path.join(args.new, 'compare.json'), 'w') as f:
        json.dump(r, f, indent=1)
    print(json.dumps({k: r[k] for k in ('new', 'old', 'frames', 'pass')}), '| first differences:', r['differ'][:5])
    sys.exit(0 if r['pass'] else 1)


if __name__ == '__main__':
    main()
