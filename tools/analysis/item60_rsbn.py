"""Item 60 (experiments_md 20261003_05 §5 item 60, design fixed 2026-11-27 12:09 JST, experiments_md commit fb425f2, before
any number): range-stratified BN in source training on item 48's forward pair. ANALYSIS.

    bash scripts/_item60_evals_20261127.sh JOB SEED              # src / none / ada of one row's epoch-152 checkpoint
    python3 analysis/item60_rsbn.py manifest OUT.json --jobs s666=J s667=J
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --real-kernel --workers 4
    python3 analysis/item60_rsbn.py read AP.json

Rows P_RS_<mode> (mode src: S' val, none: KITTI val, ada: KITTI val after per-knot AdaBN; analysis/item60_rsbn_eval.py) and
the existing references: P_src (P_K on S' val, own statistics; item 48 reverse), P_none (P_K on KITTI val; item 48 forward),
P_RS1 (P_K + global AdaBN), P_RS5 (P_K + item 57's RS-AdaBN), L_ada, O_ada (item 57b's manifest).
Readings: H1, P_RS_src - P_src pooled Car BEV (+2 / -2 / 1 at both seeds); H2, item 58's delta and R for P_RS_ada against
P_RS1 over 20-30 / 30-40 m ("widens" for "cannot" with delta <= -1 at both seeds). Tests: tests/test_item60_rsbn.py.
"""
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))

OUT = Path('/home/koyama/data/item60')
CFG = 'cfgs/da-ieee-access-analysis/item60/centerpoint-item60-pk-rsbn.yaml'
CFG_SRC = 'cfgs/da-ieee-access-analysis/item60/centerpoint-item60-pk-rsbn-on-sprime.yaml'
EPOCH = 152
MODES = {'src': (CFG_SRC, 'none'), 'none': (CFG, 'none'), 'ada': (CFG, 'ada')}     # mode -> (eval cfg, --adapt)
SEEDS = (666, 667)
REF_57 = '/home/koyama/data/item57/item57b_manifest.json'
REF_48F = '/home/koyama/data/item47/item48_manifest_fwd.json'
REF_48R = '/home/koyama/data/item47/item48_manifest_rev.json'
FAR = (2, 3)
NEAR = (0, 1)


def row_name(mode):
    return 'P_RS_' + mode


def eval_dir(mode, seed):
    return OUT / 'eval' / ('%s_s%d' % (mode, seed))


def parse_jobs(args):
    return {int(a.split('=')[0].lstrip('s')): int(a.split('=')[1]) for a in args}


def references():
    rows = []
    for r in json.load(open(REF_57)):
        if r['row'] in ('P_RS1', 'P_RS5', 'L_ada', 'O_ada'):
            rows.append(dict(r, group='item60'))
    for path, stats, name in ((REF_48F, 'none', 'P_none'), (REF_48R, 'none', 'P_src')):
        for r in json.load(open(path)):
            if r['row'] == 'P' and r.get('stats') == stats:
                rows.append(dict(r, group='item60', row=name))
    return rows


def manifest(jobs):
    from item42_collect import read_log
    rows = []
    for seed, job in sorted(jobs.items()):
        for mode in MODES:
            d = eval_dir(mode, seed)
            got = read_log((d, 'log_rsbn_*.txt', None))
            rows.append(dict(group='item60', row=row_name(mode), seed=seed, job=job, result=str(d / 'result.pkl'),
                             logged_bev=got['bev07'] if got else None,
                             status='ok' if (d / 'result.pkl').exists() and got else 'missing'))
    return rows + references()


def _get(rows, name, seed):
    hit = [r for r in rows if r['row'] == name and r['seed'] == seed]
    return hit[0] if len(hit) == 1 and hit[0].get('reproduced') else None


def h1(rows):
    """P_RS_src - P_src, pooled Car BEV, per seed, and the words."""
    d = {}
    for s in SEEDS:
        a, b = _get(rows, row_name('src'), s), _get(rows, 'P_src', s)
        d[s] = None if a is None or b is None else a['pooled']['bev'] - b['pooled']['bev']
    v = list(d.values())
    if any(x is None for x in v):
        word = 'not read'
    elif all(x >= 2.0 for x in v):
        word = 'range-stratified BN helps on the source'
    elif all(x <= -2.0 for x in v):
        word = 'it costs on the source'
    elif all(abs(x) < 1.0 for x in v):
        word = 'no effect on the source'
    else:
        word = 'unclear'
    return d, word


def h2(rows):
    import item58_bn_bound as m58
    seeds = [m58.per_seed(rows, s, row_name('ada')) for s in SEEDS]
    word = m58.reading(seeds)
    if word == 'it cannot' and all(x['delta'] <= -1.0 for x in seeds):
        word = 'it widens the far residual'
    return seeds, word


def far(rings):
    return sum(rings[i] for i in FAR) / len(FAR)


def cmd_read(path):
    rows = json.load(open(path))['rows']
    for r in rows:
        print('%-10s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    d, word = h1(rows)
    print('\n== H1 (the source, S\' val): %s' % word)
    for s in SEEDS:
        print('  seed %d: P_RS_src - P_src pooled BEV %s' % (s, '%+.2f' % d[s] if d[s] is not None else '-'))
    seeds, word = h2(rows)
    print('\n== H2 (the target, P_RS_ada against P_RS1): %s' % word)
    for s, x in zip(SEEDS, seeds):
        if x is None:
            print('  seed %d: not read' % s)
            continue
        print('  seed %d: s_x far %.2f / %.2f, delta %+.2f, R %.2f; near %+.2f / %+.2f vs P_RS1' % (
            s, x['s_x'][2], x['s_x'][3], x['delta'], x['R'], x['s_adabn'][0] - x['s_x'][0], x['s_adabn'][1] - x['s_x'][1]))
    print('\n-- descriptive: far (20-40 m) gain of adaptation')
    for s in SEEDS:
        g = {n: _get(rows, n, s) for n in (row_name('ada'), row_name('none'), 'P_RS5', 'P_RS1', 'P_none')}
        if any(v is None for v in g.values()):
            print('  seed %d: -' % s)
            continue
        f = {n: far([x['bev'] for x in v['rings']]) for n, v in g.items()}
        print('  seed %d: P_RS ada - none %+.2f (%.2f -> %.2f); P_K RS-AdaBN %+.2f, global AdaBN %+.2f (from %.2f)' % (
            s, f[row_name('ada')] - f[row_name('none')], f[row_name('none')], f[row_name('ada')],
            f['P_RS5'] - f['P_none'], f['P_RS1'] - f['P_none'], f['P_none']))


def main():
    a = sys.argv[1:]
    if a[:1] == ['manifest'] and len(a) >= 4 and a[2] == '--jobs':
        rows = manifest(parse_jobs(a[3:]))
        json.dump(rows, open(a[1], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r.get('job'), r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if a[:1] == ['read'] and len(a) == 2:
        return cmd_read(a[1])
    if a[:1] == ['dir'] and len(a) == 3:                    # for the eval script: the output dir of MODE SEED
        print(eval_dir(a[1], int(a[2])))
        return
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
