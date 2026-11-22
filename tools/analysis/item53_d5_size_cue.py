"""Item 53 D5 (experiments_md 20261003_05 §5): does AdaBN erase the size cue, i.e. pull the predicted Car size toward the
source's typical size? ANALYSIS: reads KITTI val labels and nuScenes train labels (source). CPU only, from the D1b / D4
predictions.

    python3 analysis/item53_d5_size_cue.py [--json OUT] [--boot 2000]

Per arm X, on the GT matched by both none and X (D1's matching): per dimension, the OLS slope b of log(pred) on log(GT),
b_none and b_X on the same set, rho = b_X / b_none, with 95 % CIs from a bootstrap over FRAMES.
Reading 5 (l and w, both seeds): rho <= 0.5 and CI_hi < 0.8 "size cue lost"; rho >= 0.8 and CI_lo > 0.5 "cue kept, uniform
inflation"; CI covering both 0.5 and 0.8 "unresolved"; else "intermediate"; b_none < 0.2 or CI_lo(b_none) <= 0 "no cue in
none to lose".
Reading 6: f = (median pred_X - median pred_none) / (m_src - median pred_none), m_src the nuScenes train Car median
(name 'car', num_lidar_pts >= 1); f >= 0.7 and CI_lo >= 0.5 "moves to the source size"; f <= 0.3 and CI_hi <= 0.5 "does
not move"; else "partly". A uniform inflation also raises f (KITTI cars are smaller), so the pull is stated only if
reading 5 is "size cue lost" AND reading 6 "moves to the source size" for l and w at both seeds.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_target_box_errors import INFOS, SCORE_MIN  # noqa: E402
from item53_iassd_box_errors import match_keyed  # noqa: E402
from item53_d4_pinning import CP, CP_ARMS, IASSD, IASSD_ARMS  # noqa: E402

NUSC_TRAIN = 'data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl'
ANCHOR = {'l': 4.63, 'w': 1.96, 'h': 1.74}       # IA-SSD BOX_CODER_CONFIG mean_size, Car (nuScenes source means)
DIMS = {'l': 3, 'w': 5, 'h': 4}                   # KITTI camera-frame (x, y, z, l, h, w, ry)
LOST, KEPT, MIN_B_NONE = 0.5, 0.8, 0.2
MOVES, STAYS, MOVES_LO, STAYS_HI = 0.7, 0.3, 0.5, 0.5
PULL = "AdaBN pulls the predicted size toward the source's typical size"
UNIFORM = 'uniform inflation that happens to move toward the source'


def nuscenes_car_sizes(infos):
    """Median and mean (l, w, h) of nuScenes Car GT as training sees them: name 'car', num_lidar_pts >= 1."""
    rows = []
    for info in infos:
        m = np.asarray(info['gt_names']) == 'car'
        if 'num_lidar_pts' in info:
            m &= np.asarray(info['num_lidar_pts']) >= 1
        rows.append(np.asarray(info['gt_boxes'])[m][:, 3:6])        # dx, dy, dz = l, w, h
    S = np.concatenate(rows) if rows else np.zeros((0, 3))
    return {'n': int(len(S)), 'median': dict(zip('lwh', map(float, np.median(S, 0)))),
            'mean': dict(zip('lwh', map(float, np.mean(S, 0))))}


def paired_arrays(pa, pb):
    """GT, none prediction, X prediction and frame id, on the GT both arms matched, in key order."""
    keys = sorted(k for k in pa if k in pb)
    return (np.array([pa[k][1] for k in keys]), np.array([pa[k][0] for k in keys]),
            np.array([pb[k][0] for k in keys]), np.array([k[0] for k in keys]))


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0]) if len(x) > 2 and np.std(x) > 0 else float('nan')


def pull_fraction(p_none, p_x, ref):
    m0, m1 = np.median(p_none), np.median(p_x)
    return float((m1 - m0) / (ref - m0)) if ref - m0 > 0 else float('nan')


def point_stats(g, p0, p1, ref):
    """Per-dimension quantities on one set; ref = {d: m_src}."""
    out = {}
    for d, i in DIMS.items():
        lg, l0, l1 = np.log(g[:, i]), np.log(p0[:, i]), np.log(p1[:, i])
        b0, b1 = slope(lg, l0), slope(lg, l1)
        out[d] = {'b_none': b0, 'b_x': b1, 'rho': b1 / b0 if b0 else float('nan'),
                  'f': pull_fraction(p0[:, i], p1[:, i], ref[d])}
    return out


def frame_groups(frames):
    """(order, starts, lengths): boxes grouped by frame, for a frame-level bootstrap."""
    order = np.argsort(frames, kind='stable')
    _, starts, lengths = np.unique(frames[order], return_index=True, return_counts=True)
    return order, starts, lengths


def frame_resample(groups, rng):
    """Indices of one bootstrap resample: frames drawn with replacement, each with ALL its boxes."""
    order, starts, lengths = groups
    pick = rng.randint(0, len(starts), len(starts))
    return np.concatenate([order[starts[j]:starts[j] + lengths[j]] for j in pick])


def bootstrap(g, p0, p1, frames, ref, n_boot, seed=0):
    rng = np.random.RandomState(seed)
    groups = frame_groups(frames)
    acc = {d: {k: [] for k in ('b_none', 'b_x', 'rho', 'f')} for d in DIMS}
    for _ in range(n_boot):
        idx = frame_resample(groups, rng)
        s = point_stats(g[idx], p0[idx], p1[idx], ref)
        for d in DIMS:
            for k in acc[d]:
                acc[d][k].append(s[d][k])
    return {d: {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in acc[d].items()}
            for d in DIMS}


def reading5_one(s, ci):
    if s['b_none'] < MIN_B_NONE or ci['b_none'][0] <= 0:
        return 'no cue in none to lose'
    rho, (lo, hi) = s['rho'], ci['rho']
    if rho <= LOST and hi < KEPT:
        return 'size cue lost'
    if rho >= KEPT and lo > LOST:
        return 'cue kept, uniform inflation'
    if lo <= LOST and hi >= KEPT:
        return 'unresolved'
    return 'intermediate'


def reading6_one(s, ci):
    f, (lo, hi) = s['f'], ci['f']
    if f >= MOVES and lo >= MOVES_LO:
        return 'moves to the source size'
    if f <= STAYS and hi <= STAYS_HI:
        return 'does not move'
    return 'partly'


def agree(rs):
    return rs[0] if len(set(rs)) == 1 else 'mixed (%s)' % ' / '.join(rs)


def combined(r5s, r6s):
    """The fixed statement, from the l and w readings at every seed."""
    if all(r == 'size cue lost' for r in r5s) and all(r == 'moves to the source size' for r in r6s):
        return PULL
    if all(r == 'cue kept, uniform inflation' for r in r5s) and all(r == 'moves to the source size' for r in r6s):
        return UNIFORM
    return 'as measured (reading 5: %s; reading 6: %s)' % (agree(r5s), agree(r6s))


def descriptive(g, p0, p1, nusc):
    out = {}
    q = lambda a: float(np.subtract(*np.percentile(a, [75, 25])))  # noqa: E731
    for d, i in DIMS.items():
        lg, l0, l1 = np.log(g[:, i]), np.log(p0[:, i]), np.log(p1[:, i])
        m = nusc['median'][d]
        lm0 = np.median(l0)
        out[d] = {'gt_median': float(np.median(g[:, i])), 'pred_none_median': float(np.median(p0[:, i])),
                  'pred_x_median': float(np.median(p1[:, i])), 'm_src': m, 'anchor': ANCHOR[d],
                  'f_anchor': pull_fraction(p0[:, i], p1[:, i], ANCHOR[d]),
                  'f_log': float((np.median(l1) - lm0) / (np.log(m) - lm0)) if np.log(m) > lm0 else float('nan'),
                  'iqr_ratio_log_pred': q(l1) / q(l0) if q(l0) > 0 else float('nan'),
                  'r_none': float(np.corrcoef(lg, l0)[0, 1]), 'r_x': float(np.corrcoef(lg, l1)[0, 1])}
    return out


def analyse(pa, pb, nusc, n_boot):
    g, p0, p1, frames = paired_arrays(pa, pb)
    ref = nusc['median']
    s = point_stats(g, p0, p1, ref)
    ci = bootstrap(g, p0, p1, frames, ref, n_boot)
    return {'n': int(len(g)), 'frames': int(len(np.unique(frames))), 'point': s, 'ci': ci,
            'reading5': {d: reading5_one(s[d], ci[d]) for d in DIMS},
            'reading6': {d: reading6_one(s[d], ci[d]) for d in DIMS},
            'descriptive': descriptive(g, p0, p1, nusc)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--json', default=None)
    p.add_argument('--boot', type=int, default=2000)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    args = p.parse_args()
    root = Path(args.root)
    nusc = nuscenes_car_sizes(pickle.load(open(root / NUSC_TRAIN, 'rb')))
    print('nuScenes train Car (car, num_lidar_pts >= 1): n %d  median l/w/h %.3f / %.3f / %.3f  mean %.3f / %.3f / %.3f'
          % ((nusc['n'],) + tuple(nusc['median'][d] for d in 'lwh') + tuple(nusc['mean'][d] for d in 'lwh')))
    infos = pickle.load(open(root / 'tools' / INFOS, 'rb'))
    gt_by_id = {i['point_cloud']['lidar_idx']: i['annos'] for i in infos}
    load = lambda path: match_keyed(pickle.load(open(root / path, 'rb')), gt_by_id, SCORE_MIN)  # noqa: E731
    out = {'nuscenes_train_car': nusc, 'n_boot': args.boot, 'iassd': {}, 'centerpoint': {}}
    for seed in (1, 2):
        pairs = {a: load(IASSD % (tag, seed, d)) for a, (tag, d) in IASSD_ARMS.items()}
        out['iassd'][seed] = {a: analyse(pairs['none'], pairs[a], nusc, args.boot) for a in ('full', 'bb', 'head')}
    cp = {a: load(CP % d) for a, d in CP_ARMS.items()}
    out['centerpoint'] = {a: analyse(cp['none'], cp[a], nusc, args.boot) for a in ('first', 'deep', 'full')}
    rd = {}
    for a in ('full', 'bb', 'head'):
        r5 = [out['iassd'][s][a]['reading5'][d] for s in (1, 2) for d in ('l', 'w')]
        r6 = [out['iassd'][s][a]['reading6'][d] for s in (1, 2) for d in ('l', 'w')]
        rd[a] = {'reading5': agree(r5), 'reading6': agree(r6), 'combined': combined(r5, r6),
                 'h': {'reading5': agree([out['iassd'][s][a]['reading5']['h'] for s in (1, 2)]),
                       'reading6': agree([out['iassd'][s][a]['reading6']['h'] for s in (1, 2)])}}
    out['readings'] = rd

    def show(tag, r):
        for d in 'lwh':
            s, ci, ds = r['point'][d], r['ci'][d], r['descriptive'][d]
            print('  %-14s %s n %5d  b_none %.2f [%.2f, %.2f]  b_x %.2f [%.2f, %.2f]  rho %.2f [%.2f, %.2f]  '
                  'f %.2f [%.2f, %.2f]  | GT %.2f  none %.2f  X %.2f  src %.2f  anchor %.2f  f_anchor %.2f  f_log %.2f '
                  '1-rho %.2f  IQR %.2f  r %.2f->%.2f  | R5 %s  R6 %s'
                  % (tag, d, r['n'], s['b_none'], *ci['b_none'], s['b_x'], *ci['b_x'], s['rho'], *ci['rho'], s['f'],
                     *ci['f'], ds['gt_median'], ds['pred_none_median'], ds['pred_x_median'], ds['m_src'], ds['anchor'],
                     ds['f_anchor'], ds['f_log'], 1 - s['rho'], ds['iqr_ratio_log_pred'], ds['r_none'], ds['r_x'],
                     r['reading5'][d], r['reading6'][d]))
    for seed in (1, 2):
        for a in ('full', 'bb', 'head'):
            show('IA-SSD s%d %s' % (seed, a), out['iassd'][seed][a])
    for a in ('first', 'deep', 'full'):
        show('CP %s' % a, out['centerpoint'][a])
    print('READINGS:', json.dumps(rd, indent=1))
    if args.json:
        json.dump(out, open(args.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
