"""Item 49 (experiments_md 20261003_05 §5 item 49, design fixed 2026-11-21 21:10 JST, experiments_md commit 30c8c87, before any
number): the S2 pseudo-labelling gain by range, split into item 46's components. CPU only, one seed.

    python3 analysis/item49_pl_rings.py manifest OUT.json [--seed 667]          # item 46's six S2 evaluations (T, E, PL x none / src)
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --workers 6      # per-ring scoring (item 38's tool)
    python3 analysis/item49_pl_rings.py read AP.json                            # components per ring and the fixed readings

Components per ring b (item 46): O = T_src - T, G = E_src - T_src, S = E - E_src, L = PL - E; O + G + S + L = PL - T.
Reading per component X (T05 item 49): near = mean over 0-10 and 10-20 m, far = mean over 30-40 and 40-70.4 m; "acts far": far >=
+2 and far >= 2 x max(near, 0); "acts near": the mirror; otherwise "level or spread". A row whose pooled value differs from its
logged value by more than 0.02 is not read (item 38's rule); the pooled components must reproduce item 46's within 0.05.

Tests: tests/test_item49_pl_rings.py.
"""
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'analysis'))

OUT = TOOLS.parent / 'output' / 'da-ieee-access'
DIRS = {'T': OUT / 'centerpoint-accum-legaldepth-nuscenes2kitti', 'E': OUT / 'centerpoint-accum-legaldepth-st3d-nuscenes2kitti',
        'PL': OUT / 'centerpoint-accum-legaldepth-st3d-nuscenes2kitti'}
TAG = {666: {'T': 'T', 'E': 'E', 'PL': 'P'}, 667: {'T': 'T', 'E': 'E2', 'PL': 'P2'}}  # item 49b: T (27468) is shared
JOB = {666: {'T': 27468, 'E': 33101, 'PL': 27566}, 667: {'T': 27468, 'E': None, 'PL': 28283}}  # E2: T05 item 49b
ITEM46 = {'O': -0.61, 'G': 3.40, 'S': -0.10, 'L': 0.27}  # pooled, T05 item 46 (nuScenes->KITTI, one seed)
NEAR, FAR = (0, 1), (3, 4)


def result_dir(row, stats, seed=666):
    t = 'i46_%ss2_%s' % (TAG[seed][row], stats)
    return DIRS[row] / t / 'eval' / 'epoch_20' / t


def manifest(seed=666):
    from item42_collect import read_log
    rows = []
    for row in ('T', 'E', 'PL'):
        for stats in ('none', 'src'):
            d = result_dir(row, stats, seed)
            got = read_log((d, 'log_adabn_*.txt', None))
            rows.append(dict(group='pl', stats=stats, row=row, seed=seed, job=JOB[seed][row], result=str(d / 'result.pkl'),
                             logged_bev=got['bev07'] if got else None,
                             status='ok' if (d / 'result.pkl').exists() and got else 'missing'))
    return rows


def components(v):
    """v[(row, stats)] = list of values (pooled or per ring) -> {O, G, S, L, PL-T}."""
    f = lambda a, b: [x - y for x, y in zip(a, b)]  # noqa: E731
    return {'O': f(v[('T', 'src')], v[('T', 'none')]), 'G': f(v[('E', 'src')], v[('T', 'src')]),
            'S': f(v[('E', 'none')], v[('E', 'src')]), 'L': f(v[('PL', 'none')], v[('E', 'none')]),
            'PL-T': f(v[('PL', 'none')], v[('T', 'none')])}


def mean(xs):
    return sum(xs) / len(xs)


def classify(x):
    near, far = mean([x[i] for i in NEAR]), mean([x[i] for i in FAR])
    if far >= 2.0 and far >= 2 * max(near, 0.0):
        return 'acts far'
    if near >= 2.0 and near >= 2 * max(far, 0.0):
        return 'acts near'
    return 'level or spread'


def cmd_read(path):
    d = json.load(open(path))
    rows = d['rows']
    seed = rows[0]['seed']
    assert all(r['seed'] == seed for r in rows), 'one seed per AP file'
    bad = [r for r in rows if not r.get('reproduced')]
    for r in bad:
        print('NOT REPRODUCED (not read): %s %s pooled %.4f logged %s' % (r['row'], r['stats'], r['pooled']['bev'], r['logged_bev']))
    v = {(r['row'], r['stats']): [g['bev'] for g in r['rings']] for r in rows if r.get('reproduced')}
    p = {(r['row'], r['stats']): [r['pooled']['bev']] for r in rows if r.get('reproduced')}
    print('rings', ['%g-%g' % tuple(x) for x in d['rings']], 'n_gt', '/'.join(str(g['n_gt']) for g in rows[0]['rings']))
    for r in rows:
        print('%-2s %-4s pooled %6.2f | %s' % (r['row'], r['stats'], r['pooled']['bev'], ' '.join('%6.2f' % g['bev'] for g in r['rings'])))
    if len(v) < 6:
        print('components not read: %d of 6 rows reproduced' % len(v))
        return None
    c, cp = components(v), components(p)
    out = {}
    for k in ('O', 'G', 'S', 'L', 'PL-T'):
        ok = '' if k == 'PL-T' or seed != 666 else ('  pooled check %s (item 46 %+.2f)' % (
            'pass' if abs(cp[k][0] - ITEM46[k]) <= 0.05 else 'FLAG', ITEM46[k]))
        out[k] = classify(c[k])
        print('%-4s pooled %+6.2f | %s | near %+.2f far %+.2f -> %s%s' % (
            k, cp[k][0], ' '.join('%+6.2f' % x for x in c[k]), mean([c[k][i] for i in NEAR]), mean([c[k][i] for i in FAR]),
            out[k], ok))
    far_gain = mean([c['PL-T'][i] for i in FAR])
    for k in ('G', 'L'):
        print('share of (PL - T) far mean %+.2f carried by %s: %s' % (
            far_gain, k, '%.2f' % (mean([c[k][i] for i in FAR]) / far_gain) if far_gain else '-'))
    print('seed %d, the reading (L): %s' % (seed, out['L']))
    return out


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == 'manifest':
        rows = manifest(int(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[3] == '--seed' else 666)
        json.dump(rows, open(sys.argv[2], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['stats'], r['job'], r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if len(sys.argv) >= 3 and sys.argv[1] == 'read':
        cmd_read(sys.argv[2])
        return
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
