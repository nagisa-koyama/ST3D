"""CPU pre-flight of experiments_md 20261003_05 §5 item 32e (before any GPU evaluation): every twin written by
make_groupcheck_configs_32e.py, through the REAL test loader against its base, on a few strided val frames. ANALYSIS
(val labels only place the boxes the arms are defined by, as evaluation does).

Checks per twin, per frame (base = the oracle's own eval config; Lyft: its -asscored twin, same LIDAR_CONFIG):
  - every twin: same frame ids; LIDAR_CONFIG twins keep only their platform (the loader logs the count);
  - height (dz): GT z moves by dz, x / y unchanged; a perfect prediction un-shifts to the base's (generate_prediction_dicts);
  - pitch: GT unchanged; the fitted ground slope along the VEHICLE's forward axis (Lyft: -x, PandaSet: +x) changes by the
    named angle (pitchp1 = the ground rises 1 deg ahead), within 0.3 deg;
  - region: GT unchanged; foreground / background points kept at the stated share of the base's, the other side unchanged;
  - ground: GT unchanged; in-box points unchanged, some outside points removed;
  - density (Lyft): GT unchanged; points kept at the stated ratio.
Prints one line per twin, "OK" or "FAIL <reason>", and "ALL OK" / "FAILURES n" at the end.

    python analysis/groupcheck_32e_cpu_check.py [--frames 3]      # inside the container, from ST3D/tools
"""
import argparse
import glob
import math
import re
import sys

import numpy as np

sys.path.insert(0, '.')
import _init_path  # noqa: F401,E402
from analysis.osens2_dump import cfg_of, dataset_of, unshifted_predictions  # noqa: E402
from pcdet.datasets import osens2_utils as O  # noqa: E402

G = 'cfgs/da-ieee-access-analysis/groupcheck/'
BASES = [  # (twin stem, base config, vehicle-forward sign along loader x)
    ('centerpoint-sourceonly-lyft2lyft-l64', G + 'centerpoint-sourceonly-lyft2lyft-l64-asscored.yaml', -1.0),
    ('centerpoint-sourceonly-lyft2lyft-l40', G + 'centerpoint-sourceonly-lyft2lyft-l40-asscored.yaml', -1.0),
    ('centerpoint-pandaset-spin2spin', 'cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml', 1.0),
    ('centerpoint-pandaset-flash2flash', 'cfgs/da-ieee-access/centerpoint-pandaset-flash2flash.yaml', 1.0),
]


def forward_slope(points, fsign):
    fit = O.fit_plane(O.lowest_per_cell(np.asarray(points[:, :3], dtype=np.float64)))
    if fit is None:
        return None
    return math.degrees(math.atan(fit[0][0] * fsign))


def inbox(points, boxes):
    return O.in_any_box(points[:, :3], O.enlarge_boxes(np.asarray(boxes), 0.25))


def check(arm, ds_b, ds_t, idx, fsign):
    errs, notes = [], []
    for i in idx:
        b, t = ds_b[i], ds_t[i]
        if b['frame_id'] != t['frame_id']:
            return ['frame ids differ at %d' % i], notes
        gb, gt = b['gt_boxes'][:, :7], t['gt_boxes'][:, :7]
        pb, pt = b['points'], t['points']
        m = re.match(r'dz([mp])(\d+)', arm)
        if m:
            dz = (1 if m.group(1) == 'p' else -1) * int(m.group(2)) / 100.0
            n = min(len(gb), len(gt))
            if len(gb) != len(gt):
                notes.append('frame %d: %d vs %d GT after the range mask' % (i, len(gb), len(gt)))
            if not (np.allclose(gt[:n, :2], gb[:n, :2], atol=1e-4) and np.allclose(gt[:n, 2] - gb[:n, 2], dz, atol=1e-4)):
                errs.append('frame %d: GT not moved by dz %.2f' % (i, dz))
            if len(gb) == len(gt):
                ub, ut = unshifted_predictions(ds_b, b), unshifted_predictions(ds_t, t)
                if not np.allclose(ub, ut, atol=1e-3):
                    errs.append('frame %d: perfect prediction does not un-shift (max %.4f)' % (i, np.abs(ub - ut).max()))
            continue
        if len(gb) != len(gt) or not np.allclose(gb, gt, atol=1e-5):
            errs.append('frame %d: GT changed' % i)
            continue
        m = re.match(r'pitch([mp])(\d+)', arm)
        if m:
            d = (1 if m.group(1) == 'p' else -1) * int(m.group(2))
            sb, st = forward_slope(pb, fsign), forward_slope(pt, fsign)
            if sb is None or st is None:
                errs.append('frame %d: ground fit failed' % i)
            elif abs((st - sb) - d) > 0.3:
                errs.append('frame %d: forward slope %.2f -> %.2f, change %.2f, named %+d' % (i, sb, st, st - sb, d))
            else:
                notes.append('frame %d slope %.2f -> %.2f' % (i, sb, st))
            continue
        ib, it = inbox(pb, gb), inbox(pt, gt)
        nib, nob, nit, nob_t = ib.sum(), (~ib).sum(), it.sum(), (~it).sum()
        m = re.match(r'(bg|fg)keep(\d+)', arm)
        if m:
            keep = int(m.group(2)) / 100.0
            kept_side, other = ((nob_t, nob), (nit, nib)) if m.group(1) == 'bg' else ((nit, nib), (nob_t, nob))
            share = kept_side[0] / max(kept_side[1], 1)
            if abs(share - keep) > 0.03 or other[0] != other[1]:
                errs.append('frame %d: kept share %.3f (named %.2f), other side %d vs %d' % (i, share, keep, other[0],
                                                                                            other[1]))
            else:
                notes.append('frame %d share %.3f' % (i, share))
            continue
        if arm == 'groundrm020':
            if nit != nib or nob_t >= nob:
                errs.append('frame %d: in-box %d vs %d, outside %d vs %d' % (i, nit, nib, nob_t, nob))
            else:
                notes.append('frame %d outside kept %.3f' % (i, nob_t / max(nob, 1)))
            continue
        m = re.match(r'ratio(\d+)', arm)
        if m:
            r = int(m.group(1)) / 100.0
            share = len(pt) / max(len(pb), 1)
            if abs(share - r) > 0.02:
                errs.append('frame %d: points kept %.3f (named %.2f)' % (i, share, r))
            else:
                notes.append('frame %d kept %.3f' % (i, share))
            continue
        errs.append('unknown arm %s' % arm)
    return errs, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=3)
    args = ap.parse_args()
    fails = 0
    for stem, base, fsign in BASES:
        ds_b = dataset_of(cfg_of(base))
        n = len(ds_b)
        idx = sorted(set(np.linspace(0, n - 1, args.frames).astype(int).tolist()))
        print('BASE %s: %d val frames, checking %s' % (base, n, idx), flush=True)
        for path in sorted(glob.glob(G + stem + '-*.yaml')):
            arm = path[len(G + stem) + 1:-5]
            if arm == 'asscored':
                continue
            ds_t = dataset_of(cfg_of(path))
            if len(ds_t) != n:
                print('FAIL %s: %d frames against %d' % (path, len(ds_t), n), flush=True)
                fails += 1
                continue
            errs, notes = check(arm, ds_b, ds_t, idx, fsign)
            fails += bool(errs)
            print('%s %s: %s' % ('FAIL' if errs else 'OK  ', path, '; '.join(errs or notes)), flush=True)
    print('ALL OK' if fails == 0 else 'FAILURES %d' % fails)


if __name__ == '__main__':
    main()
