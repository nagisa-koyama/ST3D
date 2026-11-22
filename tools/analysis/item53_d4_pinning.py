"""Item 53 D1b / D4 (experiments_md 20261003_05 §5): IA-SSD's box errors when only the backbone (bb) or only the head
re-estimates its BN statistics, and a per-box test of "the box inflates while pinned to its visible faces".
ANALYSIS: reads KITTI val labels. CPU only, from existing eval predictions (D2's job 35185 and E1c's none arm).

    python3 analysis/item53_d4_pinning.py [--json OUT]

D1b: D1's matching, components, readings 1 and 2 (item53_iassd_box_errors.py), paired against none, for bb and head;
an arm "reproduces full" on a component if its reading 2 equals full's at both seeds; forward (camera z) shift >= +0.10 m
against none reads "pushed out". The CenterPoint-comparable size table is the format of 20261005_01 §6.

D4, per box on the paired GT, prediction p against GT g:
- front-back: u = unit BEV ray camera -> GT centre; theta_rel = angle(u, GT heading (cos ry, -sin ry) in camera (x, z));
  dr = BEV centre offset along u (+ farther); x = (dl |cos theta_rel| + dw |sin theta_rel|) / 2. OLS dr = a + b x, Pearson r.
  Reading 3: b in [0.5, 1.5] and r >= 0.3 "pinned to the near faces"; b < 0.5 or r < 0.3 "not pinned"; b > 1.5 and
  r >= 0.3 "pushed beyond the pinning"; seeds must agree, else "mixed".
- up-down: top_up = bottom_up + dh; f_down = median(-bottom_up) / median(dh). Reading 4: median dh < 0.05 m "no height
  growth"; f_down >= 0.75 "pinned to the roof"; <= 0.25 "pinned to the ground"; else "grows about its middle".
CenterPoint S2 25743 arms are descriptive, used only if they reproduce 20261005_01 §6's printed box table.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_target_box_errors import INFOS, SCORE_MIN, moderate_car_gt  # noqa: E402
from item53_iassd_box_errors import counterfactual_ious, match_keyed, paired, components  # noqa: E402

IASSD = 'output/da-ieee-access-pointrcnn/iassd-sourceonly-nuscenes2kitti-fovtar/%s_s%d/eval/epoch_20/%s/result.pkl'
IASSD_ARMS = {'none': ('b2', 'stats_none_mix1_n1000'), 'full': ('i53', 'stats_target_mix1_n1000'),
              'bb': ('i53', 'stats_target_mix1_n1000_layers-bb'), 'head': ('i53', 'stats_target_mix1_n1000_layers-head')}
CP = 'output/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti/adabn/eval/epoch_20/%s/result.pkl'
CP_ARMS = {'none': 'stats_none_mix1_n1000', 'first': 'stats_target_mix1_n1000_layers-first',
           'deep': 'stats_target_mix1_n1000_layers-deep', 'full': 'stats_target_mix1_n1000'}
# 20261005_01 §6, box table as printed: matched GT, length / width / height median bias (m), 3D IoU 0.7 pass share.
CP_PRINTED = {'none': (4781, 0.14, 0.06, -0.00, 0.61), 'first': (6772, 0.24, 0.10, 0.01, 0.72),
              'deep': (7059, 0.35, 0.18, 0.05, 0.54), 'full': (7142, 0.37, 0.19, 0.07, 0.52)}
FORWARD_SHIFT = 0.10
SLOPE_BAND, MIN_R = (0.5, 1.5), 0.3
ROOF, GROUND, MIN_DH = 0.75, 0.25, 0.05
COMPARED = ('log_l', 'log_w', 'log_h', 'bottom_up', 'centre_up', 'centre', 'heading', 'forward')


def ray_geometry(p, g):
    """(dr, x, |cos theta_rel|, |sin theta_rel|) for one camera-frame (x, y_bottom, z, l, h, w, ry) pair."""
    u = np.array([g[0], g[2]]) / np.hypot(g[0], g[2])
    head = np.array([np.cos(g[6]), -np.sin(g[6])])
    c = abs(float(head @ u))
    s = float(np.sqrt(max(0.0, 1.0 - c * c)))
    dr = float((p[0] - g[0]) * u[0] + (p[2] - g[2]) * u[1])
    return dr, 0.5 * ((p[3] - g[3]) * c + (p[5] - g[5]) * s), c, s


def vertical(p, g):
    """(bottom_up, top_up, dh) in metres, + = prediction higher / taller; y points down."""
    bottom_up = -(p[1] - g[1])
    dh = p[4] - g[4]
    return bottom_up, bottom_up + dh, dh


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {'b': float('nan'), 'a': float('nan'), 'r': float('nan'), 'n': len(x)}
    b, a = np.polyfit(x, y, 1)
    return {'b': float(b), 'a': float(a), 'r': float(np.corrcoef(x, y)[0, 1]), 'n': len(x)}


def per_box(pairs):
    """D4 quantities per pair, as arrays in key order."""
    rows = [ray_geometry(p, g) + vertical(p, g) for p, g in (pairs[k] for k in sorted(pairs))]
    if not rows:
        return {}
    R = np.array(rows)
    return {'dr': R[:, 0], 'x': R[:, 1], 'cos': R[:, 2], 'sin': R[:, 3], 'bottom_up': R[:, 4], 'top_up': R[:, 5],
            'dh': R[:, 6]}


def pinning(pairs):
    q = per_box(pairs)
    fit = ols(q['x'], q['dr'])
    end, side = q['cos'] >= 0.7, q['sin'] >= 0.7
    med = lambda a: float(np.median(a)) if len(a) else float('nan')  # noqa: E731
    dh = med(q['dh'])
    return {'front_back': dict(fit, median_x=med(q['x']), median_dr=med(q['dr']),
                               end_on={'n': int(end.sum()), 'dr': med(q['dr'][end]), 'x': med(q['x'][end])},
                               side_on={'n': int(side.sum()), 'dr': med(q['dr'][side]), 'x': med(q['x'][side])}),
            'up_down': {'median_dh': dh, 'median_bottom_up': med(q['bottom_up']), 'median_top_up': med(q['top_up']),
                        'f_down': med(-q['bottom_up']) / dh if dh > 0 else float('nan'),
                        'fit': ols(q['dh'], -q['bottom_up'])}}


def reading3_one(fb):
    b, r = fb['b'], fb['r']
    if not (np.isfinite(b) and np.isfinite(r)) or r < MIN_R or b < SLOPE_BAND[0]:
        return 'not pinned'
    return 'pinned to the near faces' if b <= SLOPE_BAND[1] else 'pushed beyond the pinning'


def reading4_one(ud):
    if not np.isfinite(ud['median_dh']) or ud['median_dh'] < MIN_DH:
        return 'no height growth'
    f = ud['f_down']
    return 'pinned to the roof' if f >= ROOF else ('pinned to the ground' if f <= GROUND else 'grows about its middle')


def agree(rs):
    return rs[0] if len(set(rs)) == 1 else 'mixed (%s)' % ' / '.join(rs)


def forward_direction(pa, pb):
    """Median forward (camera z) shift of arm b against arm a on the GT both matched, and its reading."""
    common = sorted(k for k in pa if k in pb)
    za = np.median([components(*pa[k])['dz'] for k in common])
    zb = np.median([components(*pb[k])['dz'] for k in common])
    s = float(zb - za)
    return {'shift': s, 'reading': 'pushed out' if s >= FORWARD_SHIFT else 'level'}


def reading_key(d):
    """Reading 2 plus the sign of a shift, so 'shifted' up and down differ."""
    if d['reading'] == 'shifted':
        return 'shifted %s' % ('+' if d['shift'] > 0 else '-')
    return d['reading']


def reproduces(dirs_arm, dirs_full):
    """Per component: does the arm's reading (with sign) equal full's at every seed."""
    return {c: all(reading_key(a[c]) == reading_key(f[c]) for a, f in zip(dirs_arm, dirs_full)) for c in COMPARED}


def size_row(pairs):
    """20261005_01 §6's box-table row on the arm's own matched GT (medians, metres)."""
    keys = sorted(pairs)
    D = np.array([pairs[k][0] - pairs[k][1] for k in keys])
    t3 = np.array([counterfactual_ious(*pairs[k])['as_is'] for k in keys])
    bottom = -D[:, 1]
    return {'matched': len(keys), 'length': float(np.median(D[:, 3])), 'width': float(np.median(D[:, 5])),
            'height': float(np.median(D[:, 4])), 'pass_3d_0.7': float(np.mean(t3 >= 0.7)),
            'bottom_up': float(np.median(bottom)), 'top_up': float(np.median(bottom + D[:, 4])),
            'forward': float(np.median(D[:, 2]))}


def provenance(rows, printed=CP_PRINTED):
    """True if every arm reproduces the printed table (matched exact, medians within 0.005 m, pass within 0.005)."""
    out = {}
    for a, (n, l, w, h, t) in printed.items():
        r = rows[a]
        out[a] = (r['matched'] == n and abs(r['length'] - l) <= 0.005 + 1e-9 and abs(r['width'] - w) <= 0.005 + 1e-9
                  and abs(r['height'] - h) <= 0.005 + 1e-9 and abs(r['pass_3d_0.7'] - t) <= 0.005 + 1e-9)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--json', default=None)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    args = p.parse_args()
    root = Path(args.root)
    infos = pickle.load(open(root / 'tools' / INFOS, 'rb'))
    gt_by_id = {i['point_cloud']['lidar_idx']: i['annos'] for i in infos}
    load = lambda path: match_keyed(pickle.load(open(root / path, 'rb')), gt_by_id, SCORE_MIN)  # noqa: E731
    out = {'iassd': {}, 'centerpoint': {}}
    for seed in (1, 2):
        pairs = {a: load(IASSD % (tag, seed, d)) for a, (tag, d) in IASSD_ARMS.items()}
        common = {a: {k: v for k, v in pairs[a].items() if k in pairs['none']} for a in pairs}
        s = {'size_table': {a: size_row(v) for a, v in pairs.items()},
             'paired': {a: paired(pairs['none'], pairs[a]) for a in ('full', 'bb', 'head')},
             'pinning': {a: pinning(common[a]) for a in pairs}}
        for a in ('full', 'bb', 'head'):
            s['paired'][a]['directions']['forward'] = forward_direction(pairs['none'], pairs[a])
        out['iassd'][seed] = s
    rd = {}
    for a in ('full', 'bb', 'head'):
        rd[a] = {'reading3': agree([reading3_one(out['iassd'][s]['pinning'][a]['front_back']) for s in (1, 2)]),
                 'reading4': agree([reading4_one(out['iassd'][s]['pinning'][a]['up_down']) for s in (1, 2)])}
    rd['none'] = {'reading3': agree([reading3_one(out['iassd'][s]['pinning']['none']['front_back']) for s in (1, 2)]),
                  'reading4': agree([reading4_one(out['iassd'][s]['pinning']['none']['up_down']) for s in (1, 2)])}
    full_dirs = [out['iassd'][s]['paired']['full']['directions'] for s in (1, 2)]
    for a in ('bb', 'head'):
        rd[a]['reproduces_full'] = reproduces([out['iassd'][s]['paired'][a]['directions'] for s in (1, 2)], full_dirs)
        rd[a]['reading2'] = {c: agree([reading_key(out['iassd'][s]['paired'][a]['directions'][c]) for s in (1, 2)])
                             for c in COMPARED}
    rd['full']['reading2'] = {c: agree([reading_key(d[c]) for d in full_dirs]) for c in COMPARED}
    out['readings'] = rd
    cp = {a: load(CP % d) for a, d in CP_ARMS.items()}
    rows = {a: size_row(v) for a, v in cp.items()}
    ok = provenance(rows)
    out['centerpoint'] = {'provenance': ok, 'size_table': rows}
    if all(ok.values()):
        out['centerpoint']['pinning'] = {a: pinning({k: v for k, v in cp[a].items() if k in cp['none']}) for a in cp}
        out['centerpoint']['readings'] = {a: {'reading3': reading3_one(out['centerpoint']['pinning'][a]['front_back']),
                                              'reading4': reading4_one(out['centerpoint']['pinning'][a]['up_down'])}
                                          for a in cp}
    for seed in (1, 2):
        print('=== IA-SSD seed %d  size table (own matched GT; medians, m)' % seed)
        for a, r in out['iassd'][seed]['size_table'].items():
            print('  %-5s matched %5d  l %+.3f  w %+.3f  h %+.3f  pass3D %.3f  bottom %+.3f  top %+.3f  fwd %+.3f'
                  % (a, r['matched'], r['length'], r['width'], r['height'], r['pass_3d_0.7'], r['bottom_up'],
                     r['top_up'], r['forward']))
        for a in ('none', 'full', 'bb', 'head'):
            pn = out['iassd'][seed]['pinning'][a]
            fb, ud = pn['front_back'], pn['up_down']
            print('  %-5s D4 front-back b %.2f a %+.3f r %.2f n %d  med x %+.3f dr %+.3f | end-on dr %+.3f x %+.3f | '
                  'side-on dr %+.3f x %+.3f || up-down dh %+.3f bottom %+.3f top %+.3f f_down %.2f (fit b %.2f r %.2f)'
                  % (a, fb['b'], fb['a'], fb['r'], fb['n'], fb['median_x'], fb['median_dr'], fb['end_on']['dr'],
                     fb['end_on']['x'], fb['side_on']['dr'], fb['side_on']['x'], ud['median_dh'],
                     ud['median_bottom_up'], ud['median_top_up'], ud['f_down'], ud['fit']['b'], ud['fit']['r']))
        for a in ('full', 'bb', 'head'):
            t = out['iassd'][seed]['paired'][a]
            print('  paired none/%s n=%d pass %.3f -> %.3f  S: %s' % (
                a, t['n'], t['pass_a']['as_is'], t['pass_b']['as_is'],
                ', '.join('%s %s' % (c, 'n/a' if v is None else '%.2f' % v) for c, v in t['S'].items())))
            print('    directions: %s' % {c: (reading_key(d), round(d.get('shift', d.get('growth')), 3))
                                          for c, d in t['directions'].items()})
    print('=== CenterPoint S2 25743 provenance:', ok)
    for a, r in rows.items():
        print('  %-5s matched %5d  l %+.3f  w %+.3f  h %+.3f  pass3D %.3f  bottom %+.3f  top %+.3f  fwd %+.3f'
              % (a, r['matched'], r['length'], r['width'], r['height'], r['pass_3d_0.7'], r['bottom_up'], r['top_up'],
                 r['forward']))
    if 'pinning' in out['centerpoint']:
        for a, pn in out['centerpoint']['pinning'].items():
            fb, ud = pn['front_back'], pn['up_down']
            print('  %-5s D4 b %.2f a %+.3f r %.2f  med x %+.3f dr %+.3f || dh %+.3f bottom %+.3f top %+.3f f_down %.2f'
                  '  -> %s' % (a, fb['b'], fb['a'], fb['r'], fb['median_x'], fb['median_dr'], ud['median_dh'],
                               ud['median_bottom_up'], ud['median_top_up'], ud['f_down'],
                               out['centerpoint']['readings'][a]))
    print('READINGS:', json.dumps(rd, indent=1))
    if args.json:
        json.dump(out, open(args.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
