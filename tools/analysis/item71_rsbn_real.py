"""Item 71 (experiments_md 20261003_05 §5 item 71, design fixed 2026-11-28 21:31 JST, before any number): items 60 / 67 on the
real pair nuScenes -> KITTI. RS-BN in source training, variants `st` (statistics per range) and `stg` (statistics and gamma /
beta per knot), seeds 666-668, item 41a's recipe; scored on KITTI val with the range lookup and the row-step lookup.

    python3 analysis/item71_rsbn_real.py spec                  # variant arm seed adapt scale out_dir run_name cfg, one per line
    python3 analysis/item71_rsbn_real.py manifest OUT.json --jobs st666=J st667=J st668=J stg666=J stg667=J stg668=J
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --real-kernel --workers 4
    python3 analysis/item71_rsbn_real.py read AP.json          # also reads OUT_source.json written by `manifest`
    python3 analysis/item71_rsbn_real.py ckptcheck CKPT        # epoch-1 check: every RangeBN layer's knots spread (stg: gamma)

Arms: none / ada (range lookup), none_s / ada_s (knots at s * range, s = 0.4 / 1.33 = 0.301 from the specs), glob (pooled
AdaBN, descriptive). References: 41a's BN_none / BN_ada rows and the S2 oracle O under AdaBN (item 68's bn_manifest).
Readings (T05 item 71): gaps g = O - X near (0-10 / 10-20 m), far (20-30 / 30-40 m) and pooled; against a reference Y,
delta = g(Y) - g(X), R = delta / g(Y); "most" R >= 0.5 and delta >= 2 at all three seeds; "widens" delta <= -1 at all three;
"cannot" R <= 0.2 or delta < 1 at all three; "partly" the rest if delta > 0 at all three, else "split".
H_A: ada vs BN_ada; H_S: ada_s vs BN_ada and D_r = ada_s - ada; H_0: none and none_s vs BN_none; H_gb: stg - st (ada, ada_s);
H_src: in-run nuScenes val at epoch 20 against 41a's. Tests: tests/test_item71_rsbn_real.py.
"""
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))

SEEDS = (666, 667, 668)
VARIANTS = ('st', 'stg')
S = 0.301                                                     # 0.4 deg (HDL-64E) / 1.33 deg (HDL-32E), the specs
ARMS = {'none': ('none', 1.0), 'ada': ('ada', 1.0), 'none_s': ('none', S), 'ada_s': ('ada', S), 'glob': ('glob', 1.0)}
OUT = Path('/home/koyama/data/item71')
EPOCH = 20
NEAR, FAR = (0, 1), (2, 3)
KINDS = ('near', 'far', 'pooled')
B1 = {666: 63.00, 667: 66.73, 668: 67.73}                     # 41a + AdaBN (the design's numbers)
B0 = {666: 35.48, 667: 34.86, 668: 44.12}                     # 41a without adaptation
BN_SRC = {666: 44.81, 667: 44.84, 668: 45.18}                 # 41a in-run nuScenes val, epoch 20
REF_TOL = 0.05


def cfg_of(v):
    return 'cfgs/da-ieee-access-analysis/item71/centerpoint-item71-nuscenes2kitti-%s.yaml' % v


def row_name(v, arm):
    return '%s_%s' % (v, arm)


def eval_dir(v, arm, s):
    return OUT / 'eval' / ('%s_%s_s%d' % (v, arm, s))


def spec():
    for v in VARIANTS:
        for s in SEEDS:
            for arm, (adapt, scale) in ARMS.items():
                yield v, arm, s, adapt, scale, eval_dir(v, arm, s), 'i71_%s_%s_s%d' % (v, arm, s), cfg_of(v)


def wandb_id(d):
    for log in sorted(Path(d).glob('log_rsbn_*.txt'), reverse=True):
        m = re.search(r'\[rsbn\] wandb run (\w+)', log.read_text(errors='replace'))
        if m:
            return m.group(1)
    return None


def parse_jobs(args):
    jobs = {}
    for a in args:
        k, j = a.split('=')
        m = re.fullmatch(r'(st|stg)(\d{3})', k)
        assert m, a
        jobs[(m.group(1), int(m.group(2)))] = int(j)
    assert sorted(jobs) == sorted((v, s) for v in VARIANTS for s in SEEDS), jobs
    return jobs


def manifest(jobs):
    from item42_collect import read_log
    import item38_kitti_rings as m38
    import item68_norm_swap as m68
    rows = []
    for v, arm, s, adapt, scale, d, _, _ in spec():
        got = read_log((d, 'log_rsbn_*.txt', None))
        rows.append(dict(group='item71', row=row_name(v, arm), seed=s, job=jobs[(v, s)], adapt=adapt, scale=scale,
                         result=str(d / 'result.pkl'), logged_bev=got['bev07'] if got else None, wandb=wandb_id(d),
                         status='ok' if (d / 'result.pkl').exists() and got else 'missing'))
    rows += [dict(r, group='item71') for r in m68.bn_manifest()]
    src = [m38.resolve('item71', '%s_src' % v, s, jobs[(v, s)], None) for v in VARIANTS for s in SEEDS]
    src += [m38.resolve('item71', 'BN_src', s, t, None) for s, (t, _) in sorted(m68.BN.items())]
    return rows, src


def _get(rows, name, seed):
    hit = [r for r in rows if r['row'] == name and r['seed'] == seed]
    return hit[0] if len(hit) == 1 else None


def _mean(xs):
    return sum(xs) / len(xs)


def gaps(row, ref):
    g = [ref['rings'][i] - row['rings'][i]['bev'] for i in range(len(ref['rings']))]
    return dict(near=_mean([g[i] for i in NEAR]), far=_mean([g[i] for i in FAR]), pooled=ref['pooled'] - row['pooled']['bev'])


def word(seeds):
    if any(x is None for x in seeds):
        return 'not read'
    if all(x['R'] is not None and x['R'] >= 0.5 and x['delta'] >= 2.0 for x in seeds):
        return 'most'
    if all(x['delta'] <= -1.0 for x in seeds):
        return 'widens'
    if all((x['R'] is not None and x['R'] <= 0.2) or x['delta'] < 1.0 for x in seeds):
        return 'cannot'
    return 'partly' if all(x['delta'] > 0 for x in seeds) else 'split'


def dword(ds, pos, neg, zero):
    if any(d is None for d in ds):
        return 'not read'
    if all(d >= 2.0 for d in ds):
        return pos
    if all(d <= -2.0 for d in ds):
        return neg
    return zero


def oracle_ref(rows):
    o = [r for r in rows if r['row'] == 'O_ada']
    nr = len(o[0]['rings'])
    return dict(rings=[sum(r['rings'][i]['bev'] for r in o) / len(o) for i in range(nr)],
                pooled=sum(r['pooled']['bev'] for r in o) / len(o))


def gap_reading(rows, ref, x_name, y_name):
    out = {}
    for kind in KINDS:
        seeds = []
        for s in SEEDS:
            ry, rx = _get(rows, y_name, s), _get(rows, x_name, s)
            if ry is None or rx is None:
                seeds.append(None)
                continue
            gy, gx = gaps(ry, ref)[kind], gaps(rx, ref)[kind]
            d = gy - gx
            seeds.append(dict(gap_ref=gy, gap=gx, delta=d, R=d / gy if gy > 0 else None))
        out[kind] = dict(word=word(seeds), seeds=seeds)
    return out


def diff(rows, a, b, kind):
    out = []
    for s in SEEDS:
        ra, rb_ = _get(rows, a, s), _get(rows, b, s)
        if ra is None or rb_ is None:
            out.append(None)
        elif kind == 'pooled':
            out.append(ra['pooled']['bev'] - rb_['pooled']['bev'])
        else:
            out.append(_mean([ra['rings'][i]['bev'] for i in FAR]) - _mean([rb_['rings'][i]['bev'] for i in FAR]))
    return out


def source_reading(src):
    out = {}
    for v in VARIANTS:
        ds = []
        for s in SEEDS:
            r = _get(src, '%s_src' % v, s)
            ds.append(None if r is None or r.get('logged_bev') is None else r['logged_bev'] - BN_SRC[s])
        if any(d is None for d in ds):
            out[v] = dict(word='not read', d=ds)
            continue
        m = _mean(ds)
        w = 'no effect on the source' if abs(m) < 2.0 else ('costs the source' if all(d <= -2.0 for d in ds) else 'mixed')
        out[v] = dict(word=w, d=ds, mean=m)
    return out


def readings(rows, src):
    out = {'not_reproduced': [(r['row'], r['seed']) for r in rows if not r.get('reproduced')]}
    ref_off = []
    for name, want in (('BN_ada', B1), ('BN_none', B0)):
        for s in SEEDS:
            r = _get(rows, name, s)
            if r is None or abs(r['pooled']['bev'] - want[s]) > REF_TOL:
                ref_off.append((name, s, None if r is None else r['pooled']['bev']))
    out['references_off'] = ref_off
    out['gates'] = dict(reproduced=not out['not_reproduced'], references=not ref_off,
                        oracle=len([r for r in rows if r['row'] == 'O_ada']) == 2)
    out['H_src'] = source_reading(src)
    if not all(out['gates'].values()):
        out['readings'] = 'not read'
        return out
    ref = oracle_ref(rows)
    res = {}
    for v in VARIANTS:
        res['H_A_%s' % v] = gap_reading(rows, ref, row_name(v, 'ada'), 'BN_ada')
        res['H_S_%s' % v] = gap_reading(rows, ref, row_name(v, 'ada_s'), 'BN_ada')
        res['H_0_%s_none' % v] = gap_reading(rows, ref, row_name(v, 'none'), 'BN_none')
        res['H_0_%s_none_s' % v] = gap_reading(rows, ref, row_name(v, 'none_s'), 'BN_none')
        for kind in ('pooled', 'far'):
            d = diff(rows, row_name(v, 'ada_s'), row_name(v, 'ada'), kind)
            res['D_r_%s_%s' % (v, kind)] = dict(d=d, word=dword(d, 'the row-step lookup adds', 'it costs', 'adds nothing'))
    for arm in ('ada', 'ada_s'):
        for kind in ('pooled', 'far'):
            d = diff(rows, row_name('stg', arm), row_name('st', arm), kind)
            res['H_gb_%s_%s' % (arm, kind)] = dict(d=d, word=dword(d, 'per-range gamma / beta add', 'they cost',
                                                                     'no difference'))
    out['readings'] = res
    out['ref'] = ref
    return out


def ckptcheck(path):
    """Epoch-1 check of a training row: each RangeBN layer's knots must have moved apart (statistics, and gamma for stg)."""
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=False)['model_state']
    layers = sorted({k[:-len('.running_mean')] for k in sd if k.endswith('.running_mean') and sd[k].dim() == 2})
    out = []
    for n in layers:
        rm = sd[n + '.running_mean']
        g = sd.get(n + '.affine.weight')
        out.append(dict(layer=n, stat_spread=float((rm - rm[:1]).abs().max()),
                        gamma_spread=None if g is None else float((g - g[:1]).abs().max()),
                        has_bn_weight=(n + '.bn.weight') in sd))
    return out


def cmd_read(path):
    rows = json.load(open(path))['rows']
    src_path = Path(path).with_name(Path(path).name.replace('_ap.json', '_manifest_source.json'))
    src = json.load(open(src_path)) if src_path.exists() else []
    for r in sorted(rows, key=lambda r: (r['row'], r['seed'])):
        print('%-10s s%d pooled %6.2f (3D %6.2f) logged %s | %s | W&B %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), r.get('wandb'), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    for r in src:
        print('source %-8s s%d nuScenes val Car BEV %s (job %s, W&B %s)' % (r['row'], r['seed'], r.get('logged_bev'),
                                                                           r.get('job'), r.get('wandb')))
    out = readings(rows, src)
    print('\n== gates: %s; references off: %s' % (out['gates'], out['references_off']))
    for v, x in out['H_src'].items():
        print('== H_src %-4s %s  d %s' % (v, x['word'], [None if d is None else round(d, 2) for d in x['d']]))
    if out['readings'] == 'not read':
        print('== NOT READ')
    else:
        for k, v in out['readings'].items():
            if 'word' in v:
                print('== %-20s %s  %s' % (k, v['word'], [None if d is None else round(d, 2) for d in v['d']]))
                continue
            for kind, x in v.items():
                print('== %-20s %-6s %-8s %s' % (k, kind, x['word'], ' '.join(
                    'n/a' if y is None else '%+.2f (R %s)' % (y['delta'], 'n/a' if y['R'] is None else '%.2f' % y['R'])
                    for y in x['seeds'])))
    json.dump(out, open(Path(path).with_name('item71_readings.json'), 'w'), indent=1, default=float)


def main():
    a = sys.argv[1:]
    if a == ['spec']:
        for x in spec():
            print(' '.join(str(y) for y in x))
        return
    if a[:1] == ['manifest'] and len(a) >= 3 and a[2] == '--jobs':
        rows, src = manifest(parse_jobs(a[3:]))
        json.dump(rows, open(a[1], 'w'), indent=1)
        json.dump(src, open(a[1].replace('.json', '_source.json'), 'w'), indent=1)
        for r in rows + src:
            print(r['row'], r['seed'], r.get('job'), r['status'], r.get('wandb'), r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if a[:1] == ['read'] and len(a) == 2:
        return cmd_read(a[1])
    if a[:1] == ['ckptcheck'] and len(a) == 2:
        out = ckptcheck(a[1])
        for x in out:
            print(x)
        bad = [x['layer'] for x in out if x['stat_spread'] <= 0 or (x['gamma_spread'] is not None and x['gamma_spread'] <= 0)]
        print('ckptcheck: %d RangeBN layers, %d with no spread -> %s' % (len(out), len(bad), 'ok' if not bad and len(out) == 34
                                                                         else 'FAILED'))
        sys.exit(0 if not bad and len(out) == 34 else 1)
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
