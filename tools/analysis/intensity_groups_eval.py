"""Score channel groupings (vehicle / VRU / others vs Car / rest) on an intensity_statistics.py npz.

    python analysis/intensity_groups_eval.py <npz>
"""
import sys
import numpy as np
sys.path.insert(0, '/home/koyama/code/ST3D')
from pcdet.datasets import intensity_calibration as ic
d = np.load(sys.argv[1]); src, ps, gta, gtb, edges = d['src'], d['ps_a'], d['gt_a'], d['gt_b'], d['edges']
Q, L = ic.quantiles, ic.QUANTILE_LEVELS
def through(q_pts, s_tab, t_tab):            # map a distribution (quantile table) through one quantile map
    return np.interp(np.interp(q_pts, np.maximum.accumulate(s_tab) + np.arange(len(L)) * 1e-12, L), L, t_tab)
G = {'vehicle': [1], 'VRU': [2, 3], 'others': [0]}
REST = [0, 2, 3]                               # the "Car + everything else" design
print('class points mapped by: global (cloud only) | own group from PS | "rest" group from PS (Car+rest design) | floor;  W1 to real target, held out')
for name, c in (('Pedestrian', 2), ('Cyclist', 3), ('VRU pooled', None)):
    for r in range(len(edges) - 1):
        chans = [2, 3] if c is None else [c]
        sc, tb, ta = src[r, chans].sum(0), gtb[r, chans].sum(0), gta[r, chans].sum(0)
        if min(sc.sum(), tb.sum(), ta.sum(), ps[r, [2, 3]].sum()) < 500:
            continue
        truth, qs = Q(tb), Q(sc)
        glob = through(qs, Q(src[r].sum(0)), Q(gta[r].sum(0)))
        vru = through(qs, Q(src[r, [2, 3]].sum(0)), Q(ps[r, [2, 3]].sum(0)))
        rest = through(qs, Q(src[r, REST].sum(0)), Q(ps[r, REST].sum(0)))
        print('%-11s %3g-%-3g | pts src %7d ps-VRU %6d gt %6d | global %.3f  VRU-map %.3f  rest-map %.3f  floor %.3f'
              % (name, edges[r], edges[r + 1], sc.sum(), ps[r, [2, 3]].sum(), ta.sum(),
                 ic.w1(glob, truth), ic.w1(vru, truth), ic.w1(rest, truth), ic.w1(Q(ta), truth)))
print('\nothers (background) under the three-group design vs the two-group one:')
for r in range(len(edges) - 1):
    truth = Q(gtb[r, 0]); qs = Q(src[r, 0])
    own = through(qs, Q(src[r, 0]), Q(ps[r, 0])); rest = through(qs, Q(src[r, REST].sum(0)), Q(ps[r, REST].sum(0)))
    print('  %3g-%-3g  others-map %.4f  rest-map %.4f  floor %.4f' % (edges[r], edges[r + 1], ic.w1(own, truth), ic.w1(rest, truth), ic.w1(Q(gta[r, 0]), truth)))
