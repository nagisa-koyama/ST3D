"""Item 53 D1 (experiments_md 20261003_05 §5): which box component breaks IA-SSD's 3D@0.7 under full AdaBN on nuScenes ->
KITTI. ANALYSIS: reads KITTI val labels. CPU only, from existing eval predictions.

Matching is kitti_target_box_errors.py's (moderate Car GT, score >= 0.3, greedy by score, BEV IoU >= 0.5 within 4 m), but
the GT identity is kept so two arms can be compared on the same GT. Per matched pair (prediction - GT, KITTI camera
frame): lateral x, forward z, BEV centre distance, bottom / centre height (+ = prediction higher), log(l), log(w), log(h),
heading error in (-pi, pi]. Counterfactual 3D IoU >= 0.7 pass rates replace one component with the GT's.

    python3 analysis/item53_iassd_box_errors.py [--json OUT] [--score_min 0.3]

The readings (T05 item 53) are computed and printed: 1. S_c, the share of the none -> target pass-rate loss that one
counterfactual recovers on the paired GT (>= 0.5 at both seeds: "c carries the collapse"); 2. the direction of each
component (shifted / spreads / level; centre scatters; heading turns), target against none on the paired GT.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_target_box_errors import INFOS, MATCH_BEV_IOU, SCORE_MIN, ious, moderate_car_gt  # noqa: E402

ROW = 'output/da-ieee-access-pointrcnn/iassd-sourceonly-nuscenes2kitti-fovtar/b2_s%d/eval/epoch_20/%s/result.pkl'
ARMS = {'none': 'stats_none_mix1_n1000', 'target': 'stats_target_mix1_n1000', 'source': 'stats_source_mix1_n1000',
        'first': 'stats_target_mix1_n1000_layers-first'}
# Reading 2 thresholds (T05 item 53): about two thirds of the vertical budget of a 3D IoU 0.7 match, half for l / w.
SHIFT = {'bottom_up': 0.10, 'centre_up': 0.10, 'log_h': 0.10, 'log_l': 0.05, 'log_w': 0.05}
IQR_RATIO = 1.5
CENTRE_GROWTH, HEADING_GROWTH, FLIP_POINTS = 0.10, 0.05, 0.05
COUNTERFACTUALS = ('height', 'bottom', 'vertical', 'lw', 'lhw', 'centre', 'heading')


def wrap(a):
    """Angle in (-pi, pi]."""
    return np.pi - np.mod(np.pi - a, 2 * np.pi)


def match_keyed(dets, gt_by_id, score_min=SCORE_MIN):
    """{(frame, gt index): (pred, gt)}: kitti_target_box_errors.match with the GT identity kept."""
    out = {}
    for d in dets:
        G = moderate_car_gt(gt_by_id[d['frame_id']])
        keep = (d['name'] == 'Car') & (d['score'] >= score_min)
        P = np.concatenate([d['location'][keep], d['dimensions'][keep], d['rotation_y'][keep, None]], 1)
        P = P[np.argsort(-d['score'][keep], kind='stable')]
        used = np.zeros(len(G), bool)
        for p in P:
            best, bj = 0.0, -1
            for j, g in enumerate(G):
                if not used[j] and np.hypot(p[0] - g[0], p[2] - g[2]) < 4:
                    b = ious(p, g)[0]
                    if b > best:
                        best, bj = b, j
            if bj >= 0 and best >= MATCH_BEV_IOU:
                used[bj] = True
                out[(d['frame_id'], bj)] = (p, G[bj])
    return out


def components(p, g):
    """Prediction - GT for one (x, y_bottom, z, l, h, w, ry) camera-frame pair; y points down."""
    dyaw = float(wrap(p[6] - g[6]))
    return {
        'dx': p[0] - g[0], 'dz': p[2] - g[2], 'centre': float(np.hypot(p[0] - g[0], p[2] - g[2])),
        'bottom_up': -(p[1] - g[1]), 'centre_up': -((p[1] - p[4] / 2) - (g[1] - g[4] / 2)),
        'log_l': float(np.log(p[3] / g[3])), 'log_h': float(np.log(p[4] / g[4])), 'log_w': float(np.log(p[5] / g[5])),
        'dyaw': dyaw, 'abs_dyaw': abs(dyaw), 'flip': float(abs(dyaw) > np.pi / 2),
    }


def counterfactual_ious(p, g):
    """3D IoU as-is and with one component replaced by the GT's; 'bev' is the 3D IoU with a perfect vertical."""
    def sub(*idx):
        q = p.copy()
        for i in idx:
            q[i] = g[i]
        return ious(q, g)[1]
    b, t = ious(p, g)
    return {'as_is': t, 'bev': b, 'height': sub(4), 'bottom': sub(1), 'vertical': sub(1, 4), 'lw': sub(3, 5),
            'lhw': sub(3, 4, 5), 'centre': sub(0, 2), 'heading': sub(6)}


def table(pairs):
    """Per-pair components and counterfactual IoUs as arrays, in key order."""
    keys = sorted(pairs)
    comp = [components(*pairs[k]) for k in keys]
    cf = [counterfactual_ious(*pairs[k]) for k in keys]
    return keys, {n: np.array([c[n] for c in comp]) for n in comp[0]} if comp else {}, \
        {n: np.array([c[n] for c in cf]) for n in cf[0]} if cf else {}


def describe(x):
    p10, p25, p50, p75, p90 = np.percentile(x, [10, 25, 50, 75, 90])
    return {'median': float(p50), 'p10': float(p10), 'p90': float(p90), 'iqr': float(p75 - p25)}


def summary(pairs, n_gt):
    keys, comp, cf = table(pairs)
    return {'matched': len(keys), 'moderate_gt': n_gt, 'matched_share': len(keys) / max(n_gt, 1),
            'components': {n: describe(v) for n, v in comp.items()},
            'pass_0.7': {n: float(np.mean(v >= 0.7)) for n, v in cf.items()}}


def loss_shares(pass_none, pass_target):
    """Reading 1: S_c = [pass(target, c -> GT) - pass(target, as-is)] / [pass(none, as-is) - pass(target, as-is)]."""
    loss = pass_none['as_is'] - pass_target['as_is']
    if loss <= 0:
        return loss, {c: None for c in COUNTERFACTUALS}
    return loss, {c: (pass_target[c] - pass_target['as_is']) / loss for c in COUNTERFACTUALS}


def directions(comp_none, comp_x):
    """Reading 2 for one seed: target (x) against none on the same GT."""
    out = {}
    for n, thr in SHIFT.items():
        a, b = describe(comp_none[n]), describe(comp_x[n])
        shift = b['median'] - a['median']
        ratio = b['iqr'] / a['iqr'] if a['iqr'] > 0 else float('inf')
        out[n] = {'shift': shift, 'iqr_ratio': ratio,
                  'reading': 'shifted' if abs(shift) >= thr else ('spreads' if ratio >= IQR_RATIO else 'level')}
    a, b = describe(comp_none['centre']), describe(comp_x['centre'])
    ratio = b['iqr'] / a['iqr'] if a['iqr'] > 0 else float('inf')
    out['centre'] = {'growth': b['median'] - a['median'], 'iqr_ratio': ratio,
                     'reading': 'scatters' if (b['median'] - a['median'] >= CENTRE_GROWTH or ratio >= IQR_RATIO)
                     else 'level'}
    g = float(np.median(comp_x['abs_dyaw']) - np.median(comp_none['abs_dyaw']))
    f = float(np.mean(comp_x['flip']) - np.mean(comp_none['flip']))
    out['heading'] = {'growth': g, 'flip_points': f,
                      'reading': 'turns' if (g >= HEADING_GROWTH or f >= FLIP_POINTS) else 'level'}
    return out


def paired(pa, pb):
    """Arm b against arm a on the GT both matched."""
    common = {k: None for k in pa if k in pb}
    ka, ca, fa = table({k: pa[k] for k in common})
    kb, cb, fb = table({k: pb[k] for k in common})
    assert ka == kb
    pass_a = {n: float(np.mean(v >= 0.7)) for n, v in fa.items()}
    pass_b = {n: float(np.mean(v >= 0.7)) for n, v in fb.items()}
    loss, shares = loss_shares(pass_a, pass_b)
    return {'n': len(ka), 'pass_a': pass_a, 'pass_b': pass_b, 'loss': loss, 'S': shares,
            'directions': directions(ca, cb)}


def reading1(per_seed):
    """'c carries the collapse' if one c has the largest S_c, >= 0.5, at both seeds; else 'no single component'."""
    tops = []
    for r in per_seed:
        s = {c: v for c, v in r['S'].items() if v is not None}
        tops.append(max(s, key=s.get) if s and max(s.values()) >= 0.5 else None)
    return ('%s carries the collapse' % tops[0]) if tops[0] is not None and len(set(tops)) == 1 \
        else 'no single component'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--json', default=None)
    p.add_argument('--score_min', type=float, default=SCORE_MIN)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    args = p.parse_args()
    infos = pickle.load(open(Path(args.root) / 'tools' / INFOS, 'rb'))
    gt_by_id = {i['point_cloud']['lidar_idx']: i['annos'] for i in infos}
    n_gt = sum(len(moderate_car_gt(a)) for a in gt_by_id.values())
    out = {'score_min': args.score_min, 'moderate_gt': n_gt, 'seeds': {}}
    for seed in (1, 2):
        pairs = {a: match_keyed(pickle.load(open(Path(args.root) / (ROW % (seed, d)), 'rb')), gt_by_id, args.score_min)
                 for a, d in ARMS.items()}
        out['seeds'][seed] = {'arms': {a: summary(v, n_gt) for a, v in pairs.items()},
                              'paired': {a: paired(pairs['none'], pairs[a]) for a in ('target', 'source', 'first')}}
    tgt = [out['seeds'][s]['paired']['target'] for s in (1, 2)]
    dirs = {}
    for n in list(SHIFT) + ['centre', 'heading']:
        rs = [t['directions'][n]['reading'] for t in tgt]
        dirs[n] = rs[0] if rs[0] == rs[1] else 'seeds disagree (%s / %s)' % tuple(rs)
    out['readings'] = {'1': reading1(tgt), '2': dirs}
    for seed in (1, 2):
        print('=== seed %d' % seed)
        for a, sm in out['seeds'][seed]['arms'].items():
            c = sm['components']
            print('  %-6s matched %4d (%.2f of %d)  pass0.7 as-is %.3f  bottom %+.3f  centre_up %+.3f  log_h %+.3f  '
                  'log_l %+.3f  log_w %+.3f  centre %.3f  |dyaw| %.3f'
                  % (a, sm['matched'], sm['matched_share'], n_gt, sm['pass_0.7']['as_is'], c['bottom_up']['median'],
                     c['centre_up']['median'], c['log_h']['median'], c['log_l']['median'], c['log_w']['median'],
                     c['centre']['median'], c['abs_dyaw']['median']))
        t = out['seeds'][seed]['paired']['target']
        print('  paired none/target n=%d  pass %.3f -> %.3f  S: %s' % (
            t['n'], t['pass_a']['as_is'], t['pass_b']['as_is'],
            ', '.join('%s %s' % (c, 'n/a' if v is None else '%.2f' % v) for c, v in t['S'].items())))
        print('  directions: %s' % {n: (d['reading'], round(d.get('shift', d.get('growth')), 3))
                                     for n, d in t['directions'].items()})
    print('READING 1:', out['readings']['1'])
    print('READING 2:', out['readings']['2'])
    if args.json:
        json.dump(out, open(args.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
