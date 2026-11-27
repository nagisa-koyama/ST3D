"""Item 62 (experiments_md 20261003_05 §5 item 62, design fixed 2026-11-27 12:53 JST, experiments_md commit 3f7c939, before any
number): P_K's BN statistics from a subpopulation of each layer's elements. ANALYSIS.

    bash scripts/_item62_pop_20261127.sh SEED                    # arms all / far20 / tbox / gtbox (item62_pop_adabn_eval.py)
    python3 analysis/item62_pop.py manifest OUT.json
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --real-kernel --workers 4
    python3 analysis/item62_pop.py read AP.json

Rows P_POP_<ARM>; references P_RS1, L_ada, O_ada (item 57b), P_BN_G (item 58). Readings: item 58's per_seed / reading against
P_RS1 ("widens" for "cannot" with delta <= -1 at both seeds); procedure check: P_POP_ALL reproduces P_RS1 within 0.05.
Tests: tests/test_item62_pop.py.
"""
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))

OUT = Path('/home/koyama/data/item62')
ARMS = ('all', 'far20', 'tbox', 'gtbox')
SEEDS = (666, 667)
PK_CKPT = {666: '/home/koyama/data/wandb/run-20261108_020021-0i5ai4hn/files/ckpt/checkpoint_epoch_152.pth',
           667: '/home/koyama/data/wandb/run-20261108_020023-1ya9lozy/files/ckpt/checkpoint_epoch_152.pth'}
REF_57 = '/home/koyama/data/item57/item57b_manifest.json'
REF_58 = '/home/koyama/data/item58/item58_manifest.json'
TOL_ALL = 0.05


def row_name(arm):
    return 'P_POP_' + arm.upper()


def arm_dir(arm, seed):
    return OUT / ('%s_s%d' % (arm, seed))


def manifest():
    from item42_collect import read_log
    rows = []
    for s in SEEDS:
        for arm in ARMS:
            d = arm_dir(arm, s)
            got = read_log((d, 'log_popada_*.txt', None))
            rows.append(dict(group='item62', row=row_name(arm), seed=s, job=None, result=str(d / 'result.pkl'),
                             logged_bev=got['bev07'] if got else None,
                             status='ok' if (d / 'result.pkl').exists() and got else 'missing'))
    for path, keep in ((REF_57, ('P_RS1', 'L_ada', 'O_ada')), (REF_58, ('P_BN_G',))):
        for r in json.load(open(path)):
            if r['row'] in keep:
                rows.append(dict(r, group='item62'))
    return rows


def all_check(rows, seed):
    a = [r for r in rows if r['row'] == row_name('all') and r['seed'] == seed]
    b = [r for r in rows if r['row'] == 'P_RS1' and r['seed'] == seed]
    if len(a) != 1 or len(b) != 1 or 'pooled' not in a[0] or 'pooled' not in b[0]:
        return False
    return abs(a[0]['pooled']['bev'] - b[0]['pooled']['bev']) <= TOL_ALL


def reading(rows, arm):
    import item58_bn_bound as m58
    seeds = [m58.per_seed(rows, s, row_name(arm)) for s in SEEDS]
    word = m58.reading(seeds)
    if word == 'it cannot' and all(x['delta'] <= -1.0 for x in seeds):
        word = 'it widens the far residual'
    return seeds, word


def cmd_read(path):
    rows = json.load(open(path))['rows']
    for r in rows:
        print('%-12s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    print('\nprocedure: P_POP_ALL within %.2f of P_RS1: %s' % (TOL_ALL, {s: all_check(rows, s) for s in SEEDS}))
    for arm in ARMS[1:]:
        seeds, word = reading(rows, arm)
        print('\n== %s: %s' % (row_name(arm), word))
        for s, d in zip(SEEDS, seeds):
            if d is None:
                print('  seed %d: not read' % s)
                continue
            print('  seed %d: s_x far %.2f / %.2f, delta %+.2f, R %.2f; near %+.2f / %+.2f vs P_RS1' % (
                s, d['s_x'][2], d['s_x'][3], d['delta'], d['R'], d['s_adabn'][0] - d['s_x'][0], d['s_adabn'][1] - d['s_x'][1]))


def main():
    a = sys.argv[1:]
    if a[:1] == ['manifest'] and len(a) == 2:
        rows = manifest()
        json.dump(rows, open(a[1], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if a[:1] == ['read'] and len(a) == 2:
        return cmd_read(a[1])
    if a[:1] == ['ckpt'] and len(a) == 2:
        print(PK_CKPT[int(a[1])])
        return
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
