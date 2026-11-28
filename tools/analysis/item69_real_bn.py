"""Item 69 (experiments_md 20261003_05 §5 item 69, design fixed 2026-11-28 17:51 JST, experiments_md commit fd86a08, before any
number): items 58 / 59a on the real pair nuScenes -> KITTI. Labelled ANALYSIS (KITTI TRAIN labels train the normalization).

    python3 analysis/item69_real_bn.py make SEED GJOB         # G_r bitwise check against 41a + the three hybrids (59a's)
    python3 analysis/item69_real_bn.py ckpt ARM SEED          # the checkpoint an AdaBN arm evaluates (rs1 = 41a's own)
    python3 analysis/item69_real_bn.py manifest OUT.json --jobs 666=E 667=E 668=E   # E = each seed's evaluation job
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --workers 4
    python3 analysis/item69_real_bn.py read AP.json

Rows per seed: G_r (own statistics, eval_checkpoint.sh), rs1 / aff_all / aff_3d / aff_2d (adabn_eval.sh target 1.0), and the
S2 oracle O under AdaBN (two seeds, item 60g's manifest) as the reference.
Readings: far gap g(X) = mean over 20-30 / 30-40 m of O - X (O = its two-seed mean per ring), and the pooled gap; delta =
g(rs1) - g(X), R = delta / g(rs1); "most" R >= 0.5 and delta >= 2 at all three seeds, "cannot" R <= 0.2 or delta < 1 at all
three, else "partly". Nothing is read unless rs1 reproduces 41a's B1 within 0.05, every G_r check and hybrid check passed,
and every scored row reproduces its logged value.
Tests: tests/test_item69_real_bn.py.
"""
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))

SEEDS = (666, 667, 668)
BN41A = {666: 32373, 667: 32411, 668: 32434}                 # item 41a's training rows (epoch 20)
EPOCH = 20
B1 = {666: 63.00, 667: 66.73, 668: 67.73}                     # 41a's AdaBN rows, the rs1 identity
TOL_RS1 = 0.05
ARMS = ('rs1', 'aff_all', 'aff_3d', 'aff_2d')
ROWS = {'G_r': 'G_r', 'rs1': 'rs1', 'aff_all': 'aff_all', 'aff_3d': 'aff_3d', 'aff_2d': 'aff_2d'}
READ = {'G_r': ('H_G', 'a normalization-level correction closes most of the real pair\'s %s residual'),
        'aff_all': ('H_gb', 'the correction lies in gamma / beta'),
        'aff_3d': ('H_3d', 'it sits in the 3D backbone')}
OUT = Path('/home/koyama/data/item69')
ITEM60G_MANIFEST = Path('/home/koyama/data/item60/item60g_manifest.json')
FAR = (2, 3)


def hybrid_path(arm, seed):
    return OUT / 'ckpt' / ('%s_s%d' % (arm, seed)) / ('checkpoint_epoch_%d.pth' % EPOCH)


def tag(arm, seed):
    return 'i69_%s_s%d' % (arm, seed)


def cmd_make(seed, gjob):
    import torch
    import item58_bn_bound as m58
    import item59_bn_split as m59
    pre = torch.load(m59.ckpt_of(BN41A[seed], EPOCH), map_location='cpu', weights_only=False)
    post = torch.load(m59.ckpt_of(gjob, EPOCH), map_location='cpu', weights_only=False)
    a, g = pre['model_state'], post['model_state']
    c = m58.compare(a, g)
    ok = not c['non_bn_differ'] and not c['missing']
    report = {'G_r': dict(non_bn_differ=c['non_bn_differ'], bn_changed=c['bn_changed'], missing=c['missing'],
                          counters=c['counters'], ok=ok), 'change': m59.change_stats(a, g)}
    print('G_r s%d: non-BN tensors differing %d, BN tensors changed %d, key mismatch %d, counters %s -> %s'
          % (seed, len(c['non_bn_differ']), c['bn_changed'], len(c['missing']), c['counters'], 'ok' if ok else 'NOT READ'))
    for arm, groups in m59.GROUPS.items():
        h = m59.hybrid(a, g, groups)
        n, prob = m59.check_hybrid(h, a, g, groups)
        report[arm] = dict(layers_from_G=n, problems=prob[:5], ok=not prob)
        dst = hybrid_path(arm, seed)
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state': h, 'epoch': EPOCH, 'version': pre.get('version')}, dst)
        print('%s s%d: %d layers from G_r, check %s -> %s' % (arm, seed, n, 'ok' if not prob else prob[:3], dst))
    json.dump(report, open(OUT / ('make_s%d.json' % seed), 'w'), indent=1, default=float)
    if not ok or any(not report[x]['ok'] for x in m59.GROUPS):
        sys.exit(1)


def ckpt(arm, seed):
    if arm == 'rs1':
        import item59_bn_split as m59
        return m59.ckpt_of(BN41A[seed], EPOCH)
    return str(hybrid_path(arm, seed))


def manifest(jobs):
    import item38_kitti_rings as m38
    rows = []
    for s in SEEDS:
        for arm in ('G_r',) + ARMS:
            rows.append(m38.resolve('item69', ROWS[arm], s, jobs[s], tag(arm, s)))
    for r in rows:
        if r['status'] == 'ok' and not Path(r['result']).exists():
            r['status'] = 'result.pkl missing'
    oracle = [dict(r, group='item69') for r in json.load(open(ITEM60G_MANIFEST)) if r['row'] == 'O_ada']
    assert sorted(r['seed'] for r in oracle) == [666, 667], oracle
    return rows + oracle


def _get(rows, name, seed):
    hit = [r for r in rows if r['row'] == name and r['seed'] == seed]
    return hit[0] if len(hit) == 1 else None


def gaps(rows, name, seed, ref):
    r = _get(rows, name, seed)
    if r is None:
        return None
    far = sum(ref['rings'][i] - r['rings'][i]['bev'] for i in FAR) / len(FAR)
    return dict(far=far, pooled=ref['pooled'] - r['pooled']['bev'])


def word(seeds):
    if any(x is None for x in seeds):
        return 'not read'
    if all(x['R'] is not None and x['R'] >= 0.5 and x['delta'] >= 2.0 for x in seeds):
        return 'most'
    if all((x['R'] is not None and x['R'] <= 0.2) or x['delta'] < 1.0 for x in seeds):
        return 'cannot'
    return 'partly'


def readings(rows, checks):
    o = [r for r in rows if r['row'] == 'O_ada']
    out = {'not_reproduced': [(r['row'], r['seed']) for r in rows if not r.get('reproduced')], 'rs1': {}, 'checks': checks}
    for s in SEEDS:
        r = _get(rows, 'rs1', s)
        out['rs1'][s] = None if r is None else r['pooled']['bev']
    rs1_ok = all(out['rs1'][s] is not None and abs(out['rs1'][s] - B1[s]) <= TOL_RS1 for s in SEEDS)
    checks_ok = all(checks.get(s, {}).get('ok') for s in SEEDS)
    out['gates'] = dict(rs1=rs1_ok, checks=checks_ok, reproduced=not out['not_reproduced'], oracle=len(o) == 2)
    if not all(out['gates'].values()):
        out['readings'] = 'not read'
        return out
    nr = len(o[0]['rings'])
    ref = dict(rings=[sum(r['rings'][i]['bev'] for r in o) / 2 for i in range(nr)],
               pooled=sum(r['pooled']['bev'] for r in o) / 2)
    res = {}
    for arm in ('G_r', 'aff_all', 'aff_3d', 'aff_2d'):
        for kind in ('far', 'pooled'):
            seeds = []
            for s in SEEDS:
                g0, gx = gaps(rows, 'rs1', s, ref), gaps(rows, arm, s, ref)
                if g0 is None or gx is None:
                    seeds.append(None)
                    continue
                d = g0[kind] - gx[kind]
                seeds.append(dict(gap_rs1=g0[kind], gap=gx[kind], delta=d, R=d / g0[kind] if g0[kind] > 0 else None))
            w = word(seeds)
            if arm in READ and w == 'most':
                w = READ[arm][1] % kind if '%s' in READ[arm][1] else READ[arm][1]
            res['%s_%s' % (arm, kind)] = dict(word=w, seeds=seeds)
    out['readings'] = res
    return out


def cmd_read(path):
    rows = json.load(open(path))['rows']
    checks = {}
    for s in SEEDS:
        f = OUT / ('make_s%d.json' % s)
        if f.exists():
            m = json.load(open(f))
            checks[s] = dict(ok=m['G_r']['ok'] and all(m[a]['ok'] for a in ('aff_all', 'aff_3d', 'aff_2d')),
                             bn_changed=m['G_r']['bn_changed'])
    for r in sorted(rows, key=lambda r: (r['row'], r['seed'])):
        print('%-8s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    out = readings(rows, checks)
    print('\n== gates: %s; rs1 %s against 41a B1 %s; checks %s' % (out['gates'], out['rs1'], B1, checks))
    if out['readings'] == 'not read':
        print('== NOT READ')
    else:
        for k, v in out['readings'].items():
            print('== %-14s %s' % (k, v['word']))
            for s, x in zip(SEEDS, v['seeds']):
                if x:
                    print('    s%d: gap rs1 %.2f -> %.2f, delta %+.2f, R %s' % (
                        s, x['gap_rs1'], x['gap'], x['delta'], 'n/a' if x['R'] is None else '%.2f' % x['R']))
    json.dump(out, open(Path(path).with_name('item69_readings.json'), 'w'), indent=1, default=float)


def main():
    a = sys.argv[1:]
    if a[:1] == ['make'] and len(a) == 3:
        return cmd_make(int(a[1]), int(a[2]))
    if a[:1] == ['ckpt'] and len(a) == 3:
        print(ckpt(a[1], int(a[2])))
        return
    if a[:1] == ['manifest'] and len(a) >= 4 and a[2] == '--jobs':
        jobs = {int(x.split('=')[0]): int(x.split('=')[1]) for x in a[3:]}
        rows = manifest(jobs)
        json.dump(rows, open(a[1], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r.get('job'), r['status'], r.get('wandb'), r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if a[:1] == ['read'] and len(a) == 2:
        return cmd_read(a[1])
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
