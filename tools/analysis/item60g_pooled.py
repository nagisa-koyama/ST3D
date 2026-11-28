"""Item 60g (experiments_md 20261003_05 §5 item 60g, design fixed 2026-11-28 10:38 JST, experiments_md commit 64cf625, before
any number): item 60's RS-BN checkpoints under one pooled set of BN statistics (global AdaBN on the RS-BN model). ANALYSIS.

    python3 analysis/item60g_pooled.py dir SEED                     # the output dir of the glob evaluation of SEED
    python3 analysis/item60g_pooled.py manifest OUT.json --jobs s666=37227 s667=37228
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --real-kernel --workers 4
    python3 analysis/item60g_pooled.py read AP.json

Rows: P_RS_glob (analysis/item60_rsbn_eval.py --adapt glob) plus every row of item 60's manifest (P_RS_ada, P_RS1, L_ada, ...).
Readings:
- H_g: item 58's delta and R for P_RS_glob against P_RS1 over 20-30 / 30-40 m ("widens" for "cannot" with delta <= -1 at
  both seeds);
- the split: D = far(P_RS_ada) - far(P_RS_glob), far = mean of the 20-30 and 30-40 m rings (WORDS_SPLIT).
Tests: tests/test_item60g_pooled.py.
"""
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item60_rsbn as m60  # noqa: E402

ROW = 'P_RS_glob'
ITEM60_MANIFEST = m60.OUT / 'item60_manifest.json'
SEEDS = m60.SEEDS
WORDS_SPLIT = ("the per-range re-estimation carries item 60's gain; RS-BN training under global statistics does not",
               "RS-BN training carries item 60's gain; the per-range re-estimation adds nothing",
               'global statistics do better on the RS-BN model',
               'mixed')
CANNOT = ('it cannot', 'it widens the far residual')


def eval_dir(seed):
    return m60.OUT / 'eval' / ('glob_s%d' % seed)


def manifest(jobs):
    from item42_collect import read_log
    rows = []
    for seed, job in sorted(jobs.items()):
        d = eval_dir(seed)
        got = read_log((d, 'log_rsbn_*.txt', None))
        rows.append(dict(group='item60g', row=ROW, seed=seed, job=job, result=str(d / 'result.pkl'),
                         logged_bev=got['bev07'] if got else None,
                         status='ok' if (d / 'result.pkl').exists() and got else 'missing'))
    return rows + [dict(r, group='item60g') for r in json.load(open(ITEM60_MANIFEST))]


def _get(rows, name, seed):
    return m60._get(rows, name, seed)


def far_of(row):
    return m60.far([g['bev'] for g in row['rings']])


def h_g(rows):
    import item58_bn_bound as m58
    seeds = [m58.per_seed(rows, s, ROW) for s in SEEDS]
    word = m58.reading(seeds)
    if word == 'it cannot' and all(x['delta'] <= -1.0 for x in seeds):
        word = 'it widens the far residual'
    return seeds, word


def split(rows):
    _, hg = h_g(rows)
    d = {}
    for s in SEEDS:
        a, g = _get(rows, m60.row_name('ada'), s), _get(rows, ROW, s)
        d[s] = None if a is None or g is None else far_of(a) - far_of(g)
    v = list(d.values())
    if hg == 'not read' or any(x is None for x in v):
        return d, 'not read'
    if hg in CANNOT and all(x >= 2.0 for x in v):
        return d, WORDS_SPLIT[0]
    if all(abs(x) < 1.0 for x in v):
        return d, WORDS_SPLIT[1]
    if all(x <= -2.0 for x in v):
        return d, WORDS_SPLIT[2]
    return d, WORDS_SPLIT[3]


def cmd_read(path):
    rows = json.load(open(path))['rows']
    for r in rows:
        print('%-10s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    seeds, word = h_g(rows)
    print('\n== H_g (P_RS_glob against P_RS1): %s' % word)
    for s, x in zip(SEEDS, seeds):
        if x is None:
            print('  seed %d: not read' % s)
            continue
        print('  seed %d: s_x far %.2f / %.2f, delta %+.2f, R %.2f; near %+.2f / %+.2f vs P_RS1' % (
            s, x['s_x'][2], x['s_x'][3], x['delta'], x['R'], x['s_adabn'][0] - x['s_x'][0], x['s_adabn'][1] - x['s_x'][1]))
    d, word = split(rows)
    print('\n== The split, D = far(P_RS_ada) - far(P_RS_glob): %s' % word)
    for s in SEEDS:
        print('  seed %d: D %s' % (s, '%+.2f' % d[s] if d[s] is not None else '-'))
    print('\n-- descriptive: pooled BEV / 3D and rings, glob against ada and P_RS1')
    for s in SEEDS:
        g, a, p = (_get(rows, n, s) for n in (ROW, m60.row_name('ada'), 'P_RS1'))
        if g is None or a is None or p is None:
            print('  seed %d: -' % s)
            continue
        print('  seed %d: glob - ada pooled %+.2f / %+.2f; glob - P_RS1 pooled %+.2f / %+.2f; rings glob - ada %s' % (
            s, g['pooled']['bev'] - a['pooled']['bev'], g['pooled']['3d'] - a['pooled']['3d'],
            g['pooled']['bev'] - p['pooled']['bev'], g['pooled']['3d'] - p['pooled']['3d'],
            ' '.join('%+.2f' % (x['bev'] - y['bev']) for x, y in zip(g['rings'], a['rings']))))


def main():
    a = sys.argv[1:]
    if a[:1] == ['manifest'] and len(a) >= 4 and a[2] == '--jobs':
        rows = manifest(m60.parse_jobs(a[3:]))
        json.dump(rows, open(a[1], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r.get('job'), r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if a[:1] == ['read'] and len(a) == 2:
        return cmd_read(a[1])
    if a[:1] == ['dir'] and len(a) == 2:
        print(eval_dir(int(a[1])))
        return
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
