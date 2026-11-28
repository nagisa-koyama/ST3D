"""Item 60r (experiments_md 20261003_05 §5 item 60r, design fixed 2026-11-28 15:32 JST, experiments_md commit 7a5c860, before
any number): item 60's RS-BN checkpoints with their knots looked up at s * range on the target, s = 0.4 deg (HDL-64E) /
1.33 deg (S' lattice) = 0.301, so each target element gets the statistics of source elements with the same row step. ANALYSIS.

    python3 analysis/item60r_rowstep.py spec                         # arm seed scale adapt out_dir run_name, one per line
    python3 analysis/item60r_rowstep.py manifest OUT.json --jobs s666=37227 s667=37228
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --real-kernel --workers 4
    python3 analysis/item60r_rowstep.py read AP.json

Rows: P_RS_none_s / P_RS_ada_s (s = 0.301), P_RS_none_s025 (s = 0.251, descriptive), the identity P_RS_none_s1 (s = 1, seed
666), plus every row of item 60g's manifest (item 60's P_RS_none / P_RS_ada, P_RS_glob, P_RS1, L_ada, ...).
Readings (T05 item 60r): H_r0 and H_ra, item 58's delta and R against P_RS1 over 20-30 / 30-40 m; D_r = far(P_RS_ada_s) -
far(P_RS_ada). Nothing is read unless the identity row reproduces item 60's logged P_RS_none (seed 666) within 0.05 and
every scored row reproduces its own logged value.
Tests: tests/test_item60r_rowstep.py.
"""
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item60_rsbn as m60  # noqa: E402

S = 0.301
S_UPPER = 0.251
SEEDS = m60.SEEDS
ITEM60G_MANIFEST = m60.OUT / 'item60g_manifest.json'
# arm -> (row, scale, --adapt, seeds)
ARMS = {'none_s': ('P_RS_none_s', S, 'none', SEEDS), 'ada_s': ('P_RS_ada_s', S, 'ada', SEEDS),
        'none_s025': ('P_RS_none_s025', S_UPPER, 'none', SEEDS), 'ident': ('P_RS_none_s1', 1.0, 'none', (666,))}
IDENTITY_TOL = 0.05
FAR, NEAR = (2, 3), (0, 1)
WORDS_R0 = {'a normalization-level correction can close most of the far residual':
            'row-step-keyed source statistics close most of the far residual without any target data',
            'it cannot': 'they do not', 'it widens the far residual': 'they widen the far residual',
            'partly': 'partly', 'not read': 'not read'}
WORDS_D = ('keying the knots by row step adds to the per-range re-estimation', 'it adds nothing', 'it costs', 'mixed')


def eval_dir(arm, seed):
    _, scale, adapt, _ = ARMS[arm]
    return m60.OUT / 'eval' / ('%s_x%s_s%d' % (adapt, ('%.3f' % scale).replace('.', ''), seed))


def spec():
    for arm, (row, scale, adapt, seeds) in ARMS.items():
        for s in seeds:
            yield arm, s, scale, adapt, eval_dir(arm, s), 'i60r_%s_s%d' % (arm, s)


def manifest(jobs):
    from item42_collect import read_log
    rows = []
    for arm, s, scale, adapt, d, _ in spec():
        got = read_log((d, 'log_rsbn_*.txt', None))
        rows.append(dict(group='item60r', row=ARMS[arm][0], seed=s, job=jobs[s], scale=scale, adapt=adapt,
                         result=str(d / 'result.pkl'), logged_bev=got['bev07'] if got else None,
                         status='ok' if (d / 'result.pkl').exists() and got else 'missing'))
    return rows + [dict(r, group='item60r') for r in json.load(open(ITEM60G_MANIFEST))]


def _get(rows, name, seed):
    return m60._get(rows, name, seed)


def mean_of(row, idx, key='bev'):
    return sum(row['rings'][i][key] for i in idx) / len(idx)


def identity(rows):
    """(ok, scored identity pooled BEV, item 60's logged P_RS_none seed 666)."""
    a, b = _get(rows, ARMS['ident'][0], 666), _get(rows, m60.row_name('none'), 666)
    if a is None or b is None or b.get('logged_bev') is None:
        return False, None, None
    return abs(a['pooled']['bev'] - b['logged_bev']) <= IDENTITY_TOL, a['pooled']['bev'], b['logged_bev']


def h(rows, name):
    import item58_bn_bound as m58
    seeds = [m58.per_seed(rows, s, name) for s in SEEDS]
    word = m58.reading(seeds)
    if word == 'it cannot' and all(x['delta'] <= -1.0 for x in seeds):
        word = 'it widens the far residual'
    return seeds, word


def d_r(rows):
    d = {}
    for s in SEEDS:
        a, b = _get(rows, ARMS['ada_s'][0], s), _get(rows, m60.row_name('ada'), s)
        d[s] = None if a is None or b is None else mean_of(a, FAR) - mean_of(b, FAR)
    v = list(d.values())
    if any(x is None for x in v):
        return d, 'not read'
    if all(x >= 2.0 for x in v):
        return d, WORDS_D[0]
    if all(abs(x) < 1.0 for x in v):
        return d, WORDS_D[1]
    if all(x <= -2.0 for x in v):
        return d, WORDS_D[2]
    return d, WORDS_D[3]


def readings(rows):
    ok, got, want = identity(rows)
    bad = [(r['row'], r['seed']) for r in rows if not r.get('reproduced')]
    out = {'identity': dict(ok=ok, scored=got, item60_logged=want), 'not_reproduced': bad}
    if not ok or bad:
        out.update(H_r0='not read', H_ra='not read', D_r='not read')
        return out
    s0, w0 = h(rows, ARMS['none_s'][0])
    sa, wa = h(rows, ARMS['ada_s'][0])
    d, wd = d_r(rows)
    out.update(H_r0=WORDS_R0[w0], H_r0_seeds=s0, H_ra=wa, H_ra_seeds=sa, D_r=wd, D_r_seeds=d)
    return out


def cmd_read(path):
    rows = json.load(open(path))['rows']
    for r in rows:
        print('%-15s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    out = readings(rows)
    i = out['identity']
    print('\n== Identity (s = 1, seed 666): scored %s, item 60 logged %s -> %s' % (i['scored'], i['item60_logged'],
                                                                                'ok' if i['ok'] else 'FAILED'))
    if out['not_reproduced']:
        print('== NOT REPRODUCED: %s' % out['not_reproduced'])
    print('== H_r0 (none_s, s = %.3f, no target data) against P_RS1: %s' % (S, out['H_r0']))
    for s, x in zip(SEEDS, out.get('H_r0_seeds', [None, None])):
        if x:
            print('  seed %d: delta %+.2f, R %.2f' % (s, x['delta'], x['R']))
    print('== H_ra (ada_s) against P_RS1: %s' % out['H_ra'])
    for s, x in zip(SEEDS, out.get('H_ra_seeds', [None, None])):
        if x:
            print('  seed %d: delta %+.2f, R %.2f' % (s, x['delta'], x['R']))
    print('== D_r = far(ada_s) - far(ada): %s  %s' % (out['D_r'], {k: None if v is None else round(v, 2)
                                                                   for k, v in out.get('D_r_seeds', {}).items()}))
    print('\n-- descriptive: pooled BEV (3D) | near BEV (3D) | far BEV (3D) | 40-70.4 BEV (3D)')
    names = [ARMS[a][0] for a in ('none_s', 'ada_s', 'none_s025')] + [m60.row_name('none'), m60.row_name('ada'),
                                                                       'P_RS_glob', 'P_RS1', 'L_ada']
    for s in SEEDS:
        for n in names:
            r = _get(rows, n, s)
            if r is None:
                continue
            print('  s%d %-15s %6.2f (%6.2f) | %6.2f (%6.2f) | %6.2f (%6.2f) | %6.2f (%6.2f)' % (
                s, n, r['pooled']['bev'], r['pooled']['3d'], mean_of(r, NEAR), mean_of(r, NEAR, '3d'),
                mean_of(r, FAR), mean_of(r, FAR, '3d'), r['rings'][4]['bev'], r['rings'][4]['3d']))
    json.dump(out, open(Path(path).with_name('item60r_readings.json'), 'w'), indent=1, default=float)


def main():
    a = sys.argv[1:]
    if a[:1] == ['spec'] and len(a) == 1:
        for arm, s, scale, adapt, d, name in spec():
            print(arm, s, scale, adapt, d, name)
        return
    if a[:1] == ['manifest'] and len(a) >= 4 and a[2] == '--jobs':
        rows = manifest(m60.parse_jobs(a[3:]))
        json.dump(rows, open(a[1], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r.get('job'), r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if a[:1] == ['read'] and len(a) == 2:
        return cmd_read(a[1])
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
