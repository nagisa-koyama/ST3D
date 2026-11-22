"""Compare two occupancy builds (occ_build_pandaset.py outputs) array by array, exactly (experiments_md 20261121_02 §6;
bug_prevention_checklist rule 8).

For each sequence and each frame present in BOTH builds: every array (g64_label, g64_unknown, ...) must be equal. Prints
the differing voxel count per array and exits non-zero on any difference, a missing array or no common frame. Used to
show a code change that should be an identity (e.g. the pose swap to pcseg.render.PoseTrack, ST3D 8d979e7 onward)
reproduces an earlier build. Run as a CPU Slurm job.
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='build dir (holds <seq>/<frame>.npz)')
    ap.add_argument('--b', required=True)
    ap.add_argument('--seqs', nargs='+', required=True)
    args = ap.parse_args()
    bad = 0
    for seq in args.seqs:
        fa = {f for f in os.listdir(os.path.join(args.a, seq)) if f.endswith('.npz')}
        fb = {f for f in os.listdir(os.path.join(args.b, seq)) if f.endswith('.npz')}
        common = sorted(fa & fb)
        if not common:
            print('%s: no common frame' % seq)
            bad += 1
            continue
        diffs = 0
        for f in common:
            with np.load(os.path.join(args.a, seq, f)) as za, np.load(os.path.join(args.b, seq, f)) as zb:
                if set(za.files) != set(zb.files):
                    print('%s/%s: arrays differ: %s vs %s' % (seq, f, sorted(za.files), sorted(zb.files)))
                    diffs += 1
                    continue
                for k in sorted(za.files):
                    x, y = za[k], zb[k]
                    if x.shape != y.shape or x.dtype != y.dtype:
                        print('%s/%s %s: shape / dtype %s %s vs %s %s' % (seq, f, k, x.shape, x.dtype, y.shape, y.dtype))
                        diffs += 1
                    elif not np.array_equal(x, y):
                        print('%s/%s %s: %d voxels differ' % (seq, f, k, int((x != y).sum())))
                        diffs += 1
        print('%s: %d common frames (only in a: %d, only in b: %d), %d differing arrays'
              % (seq, len(common), len(fa - fb), len(fb - fa), diffs), flush=True)
        bad += diffs
    print('IDENTICAL' if not bad else 'DIFFERENT (%d)' % bad)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
