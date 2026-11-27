"""Item 61 (experiments_md 20261003_05 §5 item 61, design fixed 2026-11-27 12:53 JST, experiments_md commit 3f7c939, before any
number): what does 59b's work. ANALYSIS on item 48's forward pair.

    bash scripts/_item61_adabn_20261127.sh TAG JOB EPOCH ...      # global-AdaBN rescoring, as 59a
    python3 analysis/item61_pl_split.py check OUT.json --jobs gsrc666=J gsrc667=J   # G_src: non-BN tensors equal P_K's
    python3 analysis/item61_pl_split.py manifest OUT.json --jobs gsrc666=J gsrc667=J
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --real-kernel --workers 4
    python3 analysis/item61_pl_split.py read AP.json

Rows: P_PL_BN_ADA / P_PL_ALL_ADA (59b's epoch-10 checkpoints under global AdaBN), P_GSRC_ADA (G_src's epoch-20 checkpoint
under global AdaBN); their own-statistics twins P_PL_BN / P_PL_ALL (59b) and P_GSRC (G_src's in-run evaluation); references
P_RS1, L_ada, O_ada (item 57b), P_BN_G (item 58). Readings: item 58's per_seed / reading against P_RS1.
Tests: tests/test_item61_pl_split.py.
"""
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))

REPO = TOOLS.parent
OUT = Path('/home/koyama/data/item61')
EVAL_CFG = 'cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml'
GSRC_CFG = 'cfgs/da-ieee-access-analysis/item61/centerpoint-item61-p-bnonly-gsrc.yaml'
PK = {666: 32283, 667: 32284}
PK_EPOCH = 152
PL = {'plbn': {666: 37209, 667: 37210}, 'plall': {666: 37211, 667: 37212}}
PL_EPOCH = 10
GSRC_EPOCH = 20
ROWS = {'plbn': 'P_PL_BN_ADA', 'plall': 'P_PL_ALL_ADA', 'gsrc': 'P_GSRC_ADA'}
READ = ('plbn', 'gsrc')                                    # the two pre-declared readings; plall is descriptive
REF_57 = '/home/koyama/data/item57/item57b_manifest.json'
REF_58 = '/home/koyama/data/item58/item58_manifest.json'
REF_59B = '/home/koyama/data/item59/item59b_manifest.json'
SEEDS = (666, 667)


def tag(arm, seed):
    return 'i61_%s_s%d' % (arm, seed)


def epoch_of(arm):
    return GSRC_EPOCH if arm == 'gsrc' else PL_EPOCH


def result_dir(arm, seed):
    t = tag(arm, seed)
    return REPO / 'output' / 'da-ieee-access-analysis' / 'factorswap-s2' / Path(EVAL_CFG).stem / t / 'eval' / (
        'epoch_%d' % epoch_of(arm)) / t


def parse_jobs(args):
    out = {}
    for a in args:
        k, v = a.split('=')
        m = re.match(r'gsrc(\d+)$', k)
        out[int(m.group(1))] = int(v)
    return out


def rescore_args(gsrc=None):
    """TAG JOB EPOCH triples for scripts/_item61_adabn_20261127.sh: 59b's four rows, or G_src's two."""
    if gsrc is None:
        return [(tag(a, s), PL[a][s], PL_EPOCH) for a in ('plbn', 'plall') for s in SEEDS]
    return [(tag('gsrc', s), j, GSRC_EPOCH) for s, j in sorted(gsrc.items())]


def manifest(gsrc):
    from item42_collect import read_log
    import item38_kitti_rings as i38
    rows = []
    for arm in ROWS:
        for s in SEEDS:
            if arm == 'gsrc' and s not in gsrc:
                continue
            d = result_dir(arm, s)
            got = read_log((d, 'log_rsadabn_*.txt', None))
            rows.append(dict(group='item61', row=ROWS[arm], seed=s, job=gsrc[s] if arm == 'gsrc' else PL[arm][s],
                             result=str(d / 'result.pkl'), logged_bev=got['bev07'] if got else None,
                             status='ok' if (d / 'result.pkl').exists() and got else 'missing'))
    for s, j in sorted(gsrc.items()):                      # G_src's own statistics (in-run evaluation)
        r = i38.resolve('item61', 'P_GSRC', s, j, 'epoch_%d' % GSRC_EPOCH)
        if r['status'] == 'ok' and not Path(r['result']).exists():
            r['status'] = 'result.pkl missing'
        rows.append(r)
    for path, keep in ((REF_57, ('P_RS1', 'L_ada', 'O_ada')), (REF_58, ('P_BN_G',)), (REF_59B, ('P_PL_BN', 'P_PL_ALL'))):
        for r in json.load(open(path)):
            if r['row'] in keep:
                rows.append(dict(r, group='item61'))
    return rows


def check(gsrc, load):
    """Item 58's bitwise check on G_src: `load(job, epoch)` -> model_state; ok when no non-BN tensor differs from P_K's."""
    import item58_bn_bound as m58
    out = {}
    for s, j in sorted(gsrc.items()):
        c = m58.compare(load(PK[s], PK_EPOCH), load(j, GSRC_EPOCH))
        c['ok'] = not c['non_bn_differ'] and not c['missing']
        out['P_GSRC s%d' % s] = dict(c, job=j)
    return out


def readings(rows):
    import item58_bn_bound as m58
    out = {}
    for arm in ROWS:
        seeds = [m58.per_seed(rows, s, ROWS[arm]) for s in SEEDS]
        out[arm] = (seeds, m58.reading(seeds))
    return out


WORDS = {'plbn': {'a normalization-level correction can close most of the far residual':
                  "59b's correction lies in its gamma / beta, as G's does",
                  'it cannot': "it lies in the statistics 59b's training estimated", 'partly': 'partly'},
         'gsrc': {'a normalization-level correction can close most of the far residual':
                  "continued training on the source closes it: 59b's gain needs no target data",
                  'it cannot': "continued training alone does not; 59b's gain comes from training on the target clouds",
                  'partly': 'partly'}}


def cmd_read(path):
    rows = json.load(open(path))['rows']
    for r in rows:
        print('%-12s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    g = {s: __import__('item58_bn_bound').per_seed(rows, s, 'P_BN_G') for s in SEEDS}
    for arm, (seeds, word) in readings(rows).items():
        label = WORDS[arm].get(word, word) if arm in READ else word + ' (descriptive)'
        print('\n== %s: %s' % (ROWS[arm], label))
        for s, d in zip(SEEDS, seeds):
            if d is None:
                print('  seed %d: not read' % s)
                continue
            share = d['delta'] / g[s]['delta'] if g[s] and g[s]['delta'] else None
            print("  seed %d: s_x far %.2f / %.2f, delta %+.2f, R %.2f, share of G's delta %s; near %+.2f / %+.2f vs P_RS1" % (
                s, d['s_x'][2], d['s_x'][3], d['delta'], d['R'], '%.2f' % share if share is not None else '-',
                d['s_adabn'][0] - d['s_x'][0], d['s_adabn'][1] - d['s_x'][1]))


def main():
    a = sys.argv[1:]
    if a[:1] == ['rescore_args']:
        for t in rescore_args(parse_jobs(a[1:]) if len(a) > 1 else None):
            print(*t)
        return
    if a[:1] == ['manifest'] and len(a) >= 2:
        rows = manifest(parse_jobs(a[3:]) if len(a) > 2 and a[2] == '--jobs' else {})
        json.dump(rows, open(a[1], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r.get('job'), r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if a[:1] == ['check'] and len(a) >= 4 and a[2] == '--jobs':
        import torch
        import item59_bn_split as m59
        out = check(parse_jobs(a[3:]), lambda j, e: torch.load(m59.ckpt_of(j, e), map_location='cpu')['model_state'])
        json.dump(out, open(a[1], 'w'), indent=1)
        for k, c in out.items():
            print('%s (job %d): non-BN tensors differing %d, BN tensors changed %d, key mismatch %d, counters %s -> %s' % (
                k, c['job'], len(c['non_bn_differ']), c['bn_changed'], len(c['missing']), c['counters'],
                'ok' if c['ok'] else 'FAILED'))
        sys.exit(0 if all(c['ok'] for c in out.values()) else 1)
    if a[:1] == ['read'] and len(a) == 2:
        return cmd_read(a[1])
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
