"""Car BEV AP_R40 at IoU 0.7 on GT subsets of Waymo val, by the STORED point count of each GT (ANALYSIS, CPU).

The protocol of experiments_md 20261007_03 §3: a detection matched to an out-of-subset GT is IGNORED (neither TP nor
FP); unmatched detections stay FP. Bins by `num_points_in_gt` from the infos (the uncut cloud), so an eval-time cut
cannot move a car between bins (20261009_02 §3b). Reads Waymo val labels to score only.

    python analysis/waymo_subset_ap.py name=<result.pkl> ...
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from waymo_target_residual import ap_r40, greedy_match, load_gt  # noqa: E402

BINS = [('all', 0, 10**9), ('>=200', 200, 10**9), ('50-199', 50, 200), ('10-49', 10, 50), ('1-9', 1, 10)]


def subset_ap(path, gt):
    res = pickle.load(open(path, 'rb'))
    acc = {b[0]: ([], [], 0) for b in BINS}
    for r in res:
        g = gt[r['frame_id']]; gb, gp = g['boxes'], g['npts']
        m = r['name'] == 'Car'; db = r['boxes_lidar'][m].astype(np.float32); ds = r['score'][m]
        dt_m, _ = greedy_match(gb, db, ds, 0.7)
        for name, lo, hi in BINS:
            ins = (gp >= lo) & (gp < hi)
            tp_mask = dt_m >= 0
            in_sub = np.zeros(len(db), dtype=bool); in_sub[tp_mask] = ins[dt_m[tp_mask]]
            tp = ds[tp_mask & in_sub]; fp = ds[~tp_mask]
            t, f, n = acc[name]; acc[name] = (t + [tp], f + [fp], n + int(ins.sum()))
    return {k: ap_r40(np.concatenate(t), np.concatenate(f), n) for k, (t, f, n) in acc.items()}, \
        {k: n for k, (_, _, n) in acc.items()}


if __name__ == '__main__':
    gt = load_gt()
    rows = []
    for arg in sys.argv[1:]:
        name, path = arg.split('=', 1)
        ap, n = subset_ap(path, gt); rows.append((name, ap))
    print('| model | ' + ' | '.join(b[0] for b in BINS) + ' |'); print('|---' * (len(BINS) + 1) + '|')
    for name, ap in rows:
        print('| %s | ' % name + ' | '.join('%.1f' % ap[b[0]] for b in BINS) + ' |')
    print('| GT | ' + ' | '.join(str(n[b[0]]) for b in BINS) + ' |')
