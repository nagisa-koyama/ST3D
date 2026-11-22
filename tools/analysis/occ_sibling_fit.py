"""Do each device's points fit their own sensor's sibling cuboid? (experiments_md 20261121_02 §6.3, the near-mover
residual; on val 109 frame 70 a car's Pandar64 points lay in its PandarGT sibling's box, 0.68 m from its own).

pcocc gives device d the cuboids with sensor_id in {-1, d}. For every sibling pair, a Pandar64-side cuboid A
(sensor_id 0) and a PandarGT-side cuboid B (sensor_id 1) linked from either side (uuid-level links, movers.Cuboids),
in every frame where both exist, the frame's movable-class points (raw 13-33, not stale) of each device are counted
inside A only and inside B only, with the boxes as annotated. Informative pair-frames are those whose two boxes differ
enough to separate them (centre distance >= 0.3 m) and where each device has >= 5 points in A-only or B-only.
  consistent  Pandar64 points favour A and PandarGT points favour B (what pcocc assumes);
  swapped     both favour the other box;
  p64_in_b    Pandar64 points favour B, PandarGT points favour A or are too few;
  gt_in_a     PandarGT points favour A, Pandar64 points favour B or are too few.
Reported per sequence for movers and stationary pairs separately. GT data, analysis only. Run as a CPU Slurm job.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from pcseg.class_map import build_class_map                          # noqa: E402
from pcseg.pandaset_seg import N_FRAMES                              # noqa: E402
from pcocc.movers import RAW_MOVABLE, Cuboids, inside, to_box        # noqa: E402
from pcocc.poses import Poses                                        # noqa: E402
from pcocc.scans import PANDAR64, PANDARGT, SequenceScans, stale_frames   # noqa: E402

ROOT = '/home/koyama/data/pandaset/dataset'
MIN_SEP_M = 0.3
MIN_PTS = 5


def pairs_of(cub):
    sensor_of = {}
    for r in range(len(cub)):
        sensor_of.setdefault(int(cub.uid[r]), int(cub.sensor[r]))
    pairs = set()
    for r in np.flatnonzero(cub.sibling >= 0):
        u, v = int(cub.uid[r]), int(cub.sibling[r])
        su, sv = sensor_of.get(u), sensor_of.get(v)
        if (su, sv) == (PANDAR64, PANDARGT):
            pairs.add((u, v))
        elif (su, sv) == (PANDARGT, PANDAR64):
            pairs.add((v, u))
    return sorted(pairs)


def fit_sequence(seq, cm):
    s = SequenceScans(ROOT, seq, cm, stale=stale_frames())
    cub = Cuboids(ROOT, seq, Poses(ROOT, seq).times)
    movable = np.isin(s.raw, RAW_MOVABLE) & ~s.stale
    order = np.argsort(s.frame, kind='stable')
    bounds = np.searchsorted(s.frame[order], np.arange(N_FRAMES + 1))
    res = {k: collections.Counter() for k in ('mover', 'stationary')}
    pts = {k: collections.Counter() for k in ('mover', 'stationary')}
    examples = []
    for ua, ub in pairs_of(cub):
        for f in range(N_FRAMES):
            ra, rb = cub.row_at.get((f, ua)), cub.row_at.get((f, ub))
            if ra is None or rb is None:
                continue
            kind = 'stationary' if (cub.stationary[ra] and cub.stationary[rb]) else 'mover'
            sep = float(np.linalg.norm(cub.centre[ra, :2] - cub.centre[rb, :2]))
            if sep < MIN_SEP_M:
                res[kind]['too_close'] += 1
                continue
            idx = order[bounds[f]:bounds[f + 1]]
            idx = idx[movable[idx]]
            r_xy = 0.5 * max(np.hypot(*cub.dims[ra][:2]), np.hypot(*cub.dims[rb][:2])) + sep
            c = 0.5 * (cub.centre[ra] + cub.centre[rb])
            idx = idx[np.hypot(s.xyz[idx, 0] - c[0], s.xyz[idx, 1] - c[1]) <= r_xy]
            ia = inside(to_box(s.xyz[idx], cub.centre[ra], cub.yaw[ra]), cub.dims[ra])
            ib = inside(to_box(s.xyz[idx], cub.centre[rb], cub.yaw[rb]), cub.dims[rb])
            n = {}
            for d in (PANDAR64, PANDARGT):
                md = s.dev[idx] == d
                n[d] = (int((ia & ~ib & md).sum()), int((ib & ~ia & md).sum()))
            for d, name in ((PANDAR64, 'p64'), (PANDARGT, 'gt')):
                pts[kind]['%s_in_a_only' % name] += n[d][0]
                pts[kind]['%s_in_b_only' % name] += n[d][1]
            ok64 = sum(n[PANDAR64]) >= MIN_PTS
            okgt = sum(n[PANDARGT]) >= MIN_PTS
            if not ok64 and not okgt:
                res[kind]['too_few_points'] += 1
                continue
            p64_a = ok64 and n[PANDAR64][0] > n[PANDAR64][1]
            p64_b = ok64 and n[PANDAR64][1] > n[PANDAR64][0]
            gt_a = okgt and n[PANDARGT][0] > n[PANDARGT][1]
            gt_b = okgt and n[PANDARGT][1] > n[PANDARGT][0]
            if p64_b and gt_a:
                cls = 'swapped'
            elif p64_b:
                cls = 'p64_in_b'
            elif gt_a:
                cls = 'gt_in_a'
            elif p64_a or gt_b:
                cls = 'consistent'
            else:
                cls = 'tied'
            res[kind][cls] += 1
            if cls != 'consistent' and kind == 'mover' and len(examples) < 20:
                examples.append({'frame': f, 'a': cub.uuid_names[ua][:8], 'b': cub.uuid_names[ub][:8],
                                 'label': str(cub.label[ra]), 'sep_m': round(sep, 2), 'class': cls,
                                 'p64_a_b': n[PANDAR64], 'gt_a_b': n[PANDARGT]})
    return {'seq': seq, 'pairs': len(pairs_of(cub)), 'pair_frames': {k: dict(v) for k, v in res.items()},
            'points': {k: dict(v) for k, v in pts.items()}, 'examples': examples,
            'min_sep_m': MIN_SEP_M, 'min_points': MIN_PTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cm = build_class_map('pandaset_scalr17')
    for seq in args.seqs:
        r = fit_sequence(seq, cm)
        with open(os.path.join(args.out, 'sibling_fit_%s.json' % seq), 'w') as f:
            json.dump(r, f, indent=1)
        print(seq, json.dumps(r['pair_frames']), json.dumps(r['points']), flush=True)


if __name__ == '__main__':
    main()
