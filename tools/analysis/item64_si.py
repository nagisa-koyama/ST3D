"""Item 64 (experiments_md 20261003_05 §5 item 64, design fixed 2026-11-27 13:02 JST, before any number): count-normalized
sparse convolution in P_K's source training. ANALYSIS.

    bash scripts/_item64_evals_20261127.sh JOB SEED                # src / none / ada of one row's epoch-152 checkpoint
    python3 analysis/item64_si.py manifest OUT.json --jobs s666=J s667=J
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --real-kernel --workers 4
    python3 analysis/item64_si.py read AP.json

Rows P_SI_<mode> (src: S' val; none: KITTI val; ada: KITTI val after global AdaBN) and item 60's references (P_src, P_none,
P_RS1, P_RS5, L_ada, O_ada). Readings: H1 (P_SI_src - P_src, item 60's words), H3 (P_SI_none - P_none: +10 / +3 at both
seeds), H2 (item 58's delta and R for P_SI_ada against P_RS1, "widens" as item 60). Tests: tests/test_item64_si.py.
"""
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item60_rsbn as i60  # noqa: E402

OUT = Path('/home/koyama/data/item64')
CFG = 'cfgs/da-ieee-access-analysis/item64/centerpoint-item64-pk-si.yaml'
CFG_SRC = 'cfgs/da-ieee-access-analysis/item64/centerpoint-item64-pk-si-on-sprime.yaml'
MODES = ('src', 'none', 'ada')
SEEDS = i60.SEEDS


def row_name(mode):
    return 'P_SI_' + mode


def eval_dir(mode, seed):
    return OUT / 'eval' / ('%s_s%d' % (mode, seed))


def manifest(jobs):
    from item42_collect import read_log
    rows = []
    for seed, job in sorted(jobs.items()):
        for mode in MODES:
            d = eval_dir(mode, seed)
            got = read_log((d, 'log_si_*.txt', None))
            rows.append(dict(group='item64', row=row_name(mode), seed=seed, job=job, result=str(d / 'result.pkl'),
                             logged_bev=got['bev07'] if got else None,
                             status='ok' if (d / 'result.pkl').exists() and got else 'missing'))
    return rows + [dict(r, group='item64') for r in i60.references()]


def diff_words(rows, a, b, hi, lo, words):
    d = {}
    for s in SEEDS:
        x, y = i60._get(rows, a, s), i60._get(rows, b, s)
        d[s] = None if x is None or y is None else x['pooled']['bev'] - y['pooled']['bev']
    v = list(d.values())
    if any(t is None for t in v):
        return d, 'not read'
    if all(t >= hi for t in v):
        return d, words[0]
    if all(t < lo for t in v):
        return d, words[1]
    return d, words[2]


def h1(rows):
    d, _ = diff_words(rows, row_name('src'), 'P_src', 2.0, -1e9, ('', '', ''))
    v = list(d.values())
    if any(t is None for t in v):
        return d, 'not read'
    if all(t >= 2.0 for t in v):
        return d, 'count normalization helps on the source'
    if all(t <= -2.0 for t in v):
        return d, 'it costs on the source'
    if all(abs(t) < 1.0 for t in v):
        return d, 'no effect on the source'
    return d, 'unclear'


def h3(rows):
    return diff_words(rows, row_name('none'), 'P_none', 10.0, 3.0,
                      ('count normalization removes much of the lattice term without adaptation', 'it does not', 'partly'))


def h2(rows):
    import item58_bn_bound as m58
    seeds = [m58.per_seed(rows, s, row_name('ada')) for s in SEEDS]
    word = m58.reading(seeds)
    if word == 'it cannot' and all(x['delta'] <= -1.0 for x in seeds):
        word = 'it widens the far residual'
    return seeds, word


def cmd_read(path):
    rows = json.load(open(path))['rows']
    for r in rows:
        print('%-10s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    for name, (d, w) in (('H1 (the source, S\' val)', h1(rows)), ('H3 (KITTI val, no adaptation)', h3(rows))):
        print('\n== %s: %s  %s' % (name, w, {s: (None if v is None else round(v, 2)) for s, v in d.items()}))
    seeds, w = h2(rows)
    print('\n== H2 (global AdaBN, against P_RS1): %s' % w)
    for s, x in zip(SEEDS, seeds):
        print('  seed %d: %s' % (s, 'not read' if x is None else 'delta %+.2f, R %.2f' % (x['delta'], x['R'])))


def main():
    a = sys.argv[1:]
    if a[:1] == ['manifest'] and len(a) >= 4 and a[2] == '--jobs':
        rows = manifest(i60.parse_jobs(a[3:]))
        json.dump(rows, open(a[1], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r.get('job'), r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if a[:1] == ['read'] and len(a) == 2:
        return cmd_read(a[1])
    if a[:1] == ['dir'] and len(a) == 3:
        print(eval_dir(a[1], int(a[2])))
        return
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
