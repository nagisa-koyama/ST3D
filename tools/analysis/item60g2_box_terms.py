"""Item 60g' (experiments_md 20261003_05 §5 item 60g', design fixed 2026-11-28 12:27 JST, experiments_md commit 1ecdb49,
before any number): which box term breaks item 60g's near 3D. ANALYSIS: reads KITTI val labels. CPU only, from the
existing result files of item 60g's manifest; nothing is retrained or rescored.

    python3 analysis/item60g2_box_terms.py [--manifest M.json] [--ap AP.json] [--json OUT.json] [--txt OUT.txt]

GT: exactly the AP's moderate Car GT (the official clean_data, `ignored_gt == 0`); range = the GT centre's distance in the
LiDAR x-y plane (`gt_boxes_lidar`), as item 38's rings. Matching, components and counterfactual 3D IoUs are item 53's
(analysis/item53_iassd_box_errors.py), with the GT identity kept so `glob` and `ada` are compared on the GT both matched.
Readings (T05 item 60g'): 1. S_c per bin (near 0-20 m, far 20-40 m), the share of the ada -> glob 3D-pass loss one
counterfactual recovers; 2. near, the direction of each term (item 53's rule and thresholds).
The near / far / all moderate GT counts must equal item 60g's ring n_gt sums, or nothing is read.
Tests: tests/test_item60g2_box_terms.py.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item53_iassd_box_errors as m53  # noqa: E402
from kitti_target_box_errors import MATCH_BEV_IOU, SCORE_MIN, ious  # noqa: E402

OUT = Path('/home/koyama/data/item60')
SEEDS = (666, 667)
ARMS = {'glob': 'P_RS_glob', 'ada': 'P_RS_ada', 'P_RS1': 'P_RS1'}
BINS = {'near': (0.0, 20.0), 'far': (20.0, 40.0), 'all': (0.0, float('inf'))}
RING_SUMS = {'near': ((0.0, 10.0), (10.0, 20.0)), 'far': ((20.0, 30.0), (30.0, 40.0))}
SINGLES = ('height', 'bottom', 'lw', 'centre', 'heading')
COMBOS = (('vertical', 'the vertical terms together carry it'), ('lhw', 'the size together carries it'))
LOSS_MIN = 0.05
NOT_CARRIED = 'the paired boxes do not carry the loss'
NO_SINGLE = 'no single term'


def moderate_gt(annos, clean_data):
    """(row indices, camera boxes (k, 7) = (x, y_bottom, z, l, h, w, ry), LiDAR x-y ranges) of the AP's moderate Car GT."""
    n = len(annos['gt_boxes_lidar'])
    ignored = np.asarray(clean_data(annos, {'name': np.zeros(0, dtype=str)}, 0, 1)[1], dtype=int)
    idx = np.flatnonzero(ignored == 0)
    assert np.all(idx < n), 'a moderate Car GT row without a LiDAR box'
    boxes = np.concatenate([annos['location'][idx], annos['dimensions'][idx], annos['rotation_y'][idx, None]], 1)
    rng = np.hypot(annos['gt_boxes_lidar'][idx, 0], annos['gt_boxes_lidar'][idx, 1])
    return idx, boxes.astype(np.float64), rng


def match(dets, gts, score_min=SCORE_MIN):
    """{(frame, GT row): (pred, gt, range)}: item 53's greedy matching (by score, best BEV IoU >= 0.5 within 4 m)."""
    out = {}
    for d in dets:
        idx, G, R = gts[d['frame_id']]
        keep = (d['name'] == 'Car') & (d['score'] >= score_min)
        P = np.concatenate([d['location'][keep], d['dimensions'][keep], d['rotation_y'][keep, None]], 1)
        P = P[np.argsort(-d['score'][keep], kind='stable')].astype(np.float64)
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
                out[(d['frame_id'], int(idx[bj]))] = (p, G[bj], float(R[bj]))
    return out


def in_bin(pairs, name):
    lo, hi = BINS[name]
    return {k: v[:2] for k, v in pairs.items() if lo <= v[2] < hi}


def gt_counts(gts):
    r = np.concatenate([g[2] for g in gts.values()]) if gts else np.zeros(0)
    return {b: int(np.sum((r >= lo) & (r < hi))) for b, (lo, hi) in BINS.items()}


def ring_counts(ap_rows):
    """{bin: n_gt} from item 38's ring scoring; every scored row must agree (n_gt depends on the GT only)."""
    out = None
    for row in ap_rows:
        by = {(g['lo'], g['hi']): g['n_gt'] for g in row['rings']}
        c = {b: sum(by[r] for r in rs) for b, rs in RING_SUMS.items()}
        c['all'] = row['pooled']['n_gt']
        assert out is None or c == out, 'scored rows disagree on n_gt'
        out = c
    return out


def paired(pa, pb):
    """Arm b against arm a on the GT both matched (item 53's paired), or None when no GT is shared."""
    if not any(k in pb for k in pa):
        return None
    return m53.paired(pa, pb)


def reading1(per_seed, where):
    """T05 item 60g' reading 1 for one bin, from each seed's `paired(ada, glob)`."""
    if any(r is None for r in per_seed):
        return 'not read'
    if any(r['loss'] < LOSS_MIN for r in per_seed):
        return NOT_CARRIED
    tops = []
    for r in per_seed:
        s = {c: r['S'][c] for c in SINGLES}
        c = max(s, key=s.get)
        tops.append(c if s[c] >= 0.5 else None)
    if tops[0] is not None and len(set(tops)) == 1:
        return '%s carries the %s 3D loss' % (tops[0], where)
    for c, word in COMBOS:
        if all(r['S'][c] >= 0.5 for r in per_seed):
            return word
    return NO_SINGLE


def reading2(per_seed):
    """T05 item 60g' reading 2: item 53's direction words, kept only where both seeds agree."""
    if any(r is None for r in per_seed):
        return 'not read'
    out = {}
    for n in list(m53.SHIFT) + ['centre', 'heading']:
        rs = [r['directions'][n]['reading'] for r in per_seed]
        out[n] = rs[0] if len(set(rs)) == 1 else 'seeds disagree (%s)' % ' / '.join(rs)
    return out


def summary(pairs, n_gt):
    if not pairs:
        return {'matched': 0, 'moderate_gt': n_gt}
    return m53.summary(pairs, n_gt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', default=str(OUT / 'item60g_manifest.json'))
    p.add_argument('--ap', default=str(OUT / 'item60g_ap.json'))
    p.add_argument('--json', default=str(OUT / 'item60g2.json'))
    p.add_argument('--txt', default=str(OUT / 'item60g2_read.txt'))
    p.add_argument('--score_min', type=float, default=SCORE_MIN)
    args = p.parse_args()
    lines = []

    def say(s=''):
        print(s, flush=True)
        lines.append(s)

    import kitti_eval_cpu
    official = kitti_eval_cpu.load_official_eval()
    infos = pickle.load(open(kitti_eval_cpu.INFOS, 'rb'))
    gts = {i['point_cloud']['lidar_idx']: moderate_gt(i['annos'], official.clean_data) for i in infos}
    mine = gt_counts(gts)
    ap_rows = [r for r in json.load(open(args.ap))['rows'] if r['row'] in ARMS.values()]
    theirs = ring_counts(ap_rows)
    say('== Check: moderate Car GT per bin, here %s, item 60g ring n_gt %s -> %s'
        % (mine, theirs, 'ok' if mine == theirs else 'MISMATCH'))
    if mine != theirs:
        open(args.txt, 'w').write('\n'.join(lines) + '\n')
        sys.exit(1)

    rows = json.load(open(args.manifest))
    path = {(a, r['seed']): r['result'] for a, name in ARMS.items() for r in rows if r['row'] == name}
    assert sorted(path) == sorted((a, s) for a in ARMS for s in SEEDS), sorted(path)
    pairs = {k: match(pickle.load(open(v, 'rb')), gts, args.score_min) for k, v in sorted(path.items())}

    out = {'design': "T05 item 60g'", 'score_min': args.score_min, 'gt': mine, 'arms': {}, 'paired': {}, 'readings': {}}
    say('\n-- descriptive: per arm and bin, matched share, pass 3D>=0.7 as-is / counterfactuals, BEV IoU>=0.7, medians')
    for (a, s), pr in sorted(pairs.items()):
        for b in BINS:
            sm = summary(in_bin(pr, b), mine[b])
            out['arms']['%s_s%d_%s' % (a, s, b)] = sm
            if not sm['matched']:
                say('  %-5s s%d %-4s matched 0' % (a, s, b))
                continue
            c, q = sm['components'], sm['pass_0.7']
            say('  %-5s s%d %-4s matched %4d (%.2f)  pass %.3f | %s | bev>=0.7 %.3f  med: bottom %+.3f centre_up %+.3f '
                'log_h %+.3f log_l %+.3f log_w %+.3f centre %.3f |dyaw| %.3f'
                % (a, s, b, sm['matched'], sm['matched_share'], q['as_is'],
                   ' '.join('%s %.3f' % (n, q[n]) for n in m53.COUNTERFACTUALS), q['bev'],
                   c['bottom_up']['median'], c['centre_up']['median'], c['log_h']['median'], c['log_l']['median'],
                   c['log_w']['median'], c['centre']['median'], c['abs_dyaw']['median']))

    for ref in ('ada', 'P_RS1'):
        say('\n-- paired: glob against %s%s' % (ref, '' if ref == 'ada' else ' (descriptive, no words)'))
        for b in BINS:
            per = []
            for s in SEEDS:
                r = paired(in_bin(pairs[(ref, s)], b), in_bin(pairs[('glob', s)], b))
                per.append(r)
                out['paired']['glob_vs_%s_s%d_%s' % (ref, s, b)] = r
                if r is None:
                    say('  %-4s s%d: no shared GT' % (b, s))
                    continue
                say('  %-4s s%d: n %d  pass %s %.3f -> glob %.3f (loss %+.3f)  S: %s' % (
                    b, s, r['n'], ref, r['pass_a']['as_is'], r['pass_b']['as_is'], r['loss'],
                    ', '.join('%s %s' % (c, '-' if v is None else '%.2f' % v) for c, v in r['S'].items())))
                say('        directions: %s' % ', '.join(
                    '%s %s (%+.3f)' % (n, d['reading'], d.get('shift', d.get('growth')))
                    for n, d in r['directions'].items()))
            if ref == 'ada' and b in ('near', 'far'):
                out['readings']['1_' + b] = reading1(per, b)
                if b == 'near':
                    out['readings']['2_near'] = reading2(per)

    say('\n== Reading 1, near (0-20 m): %s' % out['readings']['1_near'])
    say('== Reading 1, far (20-40 m): %s' % out['readings']['1_far'])
    say('== Reading 2, near, glob against ada: %s' % out['readings']['2_near'])
    json.dump(out, open(args.json, 'w'), indent=1, default=float)
    open(args.txt, 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
