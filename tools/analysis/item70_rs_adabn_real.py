"""Item 70 (experiments_md 20261003_05 §5 item 70, design fixed 2026-11-28 20:10 JST, before any number): item 57's
range-stratified AdaBN on the real pair nuScenes -> KITTI, on item 41a's plain-BN control rows. ANALYSIS (target labels
score only); test-time, no weight touched.

    python3 analysis/item70_rs_adabn_real.py runs SEED           # "seed job arm tag" for scripts/_item70_rsadabn_real_20261128.sh
    python3 analysis/item70_rs_adabn_real.py manifest OUT.json   # 12 rows (4 arms x 3 seeds) + the S2 oracle O under AdaBN
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --real-kernel --workers 4
    python3 analysis/item70_rs_adabn_real.py read AP.json

Arms (rs_adabn_eval.py --knots): global = RS1 (one knot = global AdaBN), item57 = RS5 (5 knots, all 34 BN layers),
item57_3d = RS5_3d (the 21 backbone_3d layers), item57_2d = RS5_2d (the 13 backbone_2d + dense_head layers). Source pass on
nuScenes TRAIN (the control's DATA_CONFIG), target pass on KITTI TRAIN, 1,000 strided frames each.
Readings, per arm and seed: g(X, b) = O(b) - X(b) (O = the oracle's two-seed mean per ring); near = mean over 0-10 / 10-20 m,
far = mean over 20-30 / 30-40 m, and pooled; delta = g(RS1) - g(X), R = delta / g(RS1). "most": R >= 0.5 and delta >= 2 at
all three seeds; "widens": delta <= -1 at all three; "cannot": R <= 0.2 or delta < 1 at all three; else "partly".
H_near on RS5_2d's near; H_far on RS5's and RS5_3d's far. Nothing is read unless RS1 lies within 0.3 of 41a's B1 at every
seed and every scored row reproduces its logged value. Tests: tests/test_item70_rs_adabn_real.py.
"""
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))

REPO = TOOLS.parent
CFG = 'cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti.yaml'
SEEDS = (666, 667, 668)
JOBS41A = {666: 32373, 667: 32411, 668: 32434}               # item 41a's control rows (epoch 20)
EPOCH = 20
B1 = {666: 63.00, 667: 66.73, 668: 67.73}                     # 41a's AdaBN rows (adabn_eval.py, 1,000 frames)
TOL_RS1 = 0.3                                                 # item 57's check
ARMS = {'global': 'RS1', 'item57': 'RS5', 'item57_3d': 'RS5_3d', 'item57_2d': 'RS5_2d'}
OUT = Path('/home/koyama/data/item70')
ITEM60G_MANIFEST = Path('/home/koyama/data/item60/item60g_manifest.json')
NEAR, FAR = (0, 1), (2, 3)                                    # item 48's rings 0-10, 10-20 | 20-30, 30-40 m
KINDS = ('near', 'far', 'pooled')
H = {('RS5_2d', 'near'): ("the BEV layers' range-pooled statistics carry part of global AdaBN's near loss",
                          'they do not'),
     ('RS5', 'far'): ("per-range statistics close most of the real pair's far residual at test time", None),
     ('RS5_3d', 'far'): ("per-range statistics close most of the real pair's far residual at test time", None)}


def tag(arm, seed):
    return 'i70_%s_s%d' % (arm, seed)


def result_dir(arm, seed):
    t = tag(arm, seed)
    return REPO / 'output' / 'da-ieee-access' / Path(CFG).stem / t / 'eval' / ('epoch_%d' % EPOCH) / t


def runs(seed):
    return [(seed, JOBS41A[seed], a, tag(a, seed)) for a in ARMS]


def wandb_id(d):
    for log in sorted(Path(d).glob('log_rsadabn_*.txt'), reverse=True):
        m = re.search(r'\[rs-adabn\] wandb run (\w+)', log.read_text(errors='replace'))
        if m:
            return m.group(1)
    return None


def manifest():
    from item42_collect import read_log
    out = []
    for s in SEEDS:
        for arm, name in ARMS.items():
            d = result_dir(arm, s)
            got = read_log((d, 'log_rsadabn_*.txt', None))
            out.append(dict(group='item70', row=name, seed=s, job=JOBS41A[s], result=str(d / 'result.pkl'),
                            logged_bev=got['bev07'] if got else None, wandb=wandb_id(d),
                            status='ok' if (d / 'result.pkl').exists() and got else 'missing'))
    oracle = [dict(r, group='item70') for r in json.load(open(ITEM60G_MANIFEST)) if r['row'] == 'O_ada']
    assert sorted(r['seed'] for r in oracle) == [666, 667], oracle
    return out + oracle


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
    return 'partly'


def hypothesis(name, kind, w):
    if (name, kind) not in H or w == 'not read':
        return None
    yes, no = H[(name, kind)]
    if name == 'RS5_2d':
        return yes if w in ('most', 'partly') else no
    return yes if w == 'most' else 'not most (%s)' % w


def readings(rows):
    o = [r for r in rows if r['row'] == 'O_ada']
    out = {'not_reproduced': [(r['row'], r['seed']) for r in rows if not r.get('reproduced')], 'rs1': {}}
    for s in SEEDS:
        r = _get(rows, 'RS1', s)
        out['rs1'][s] = None if r is None else r['pooled']['bev']
    rs1_ok = all(out['rs1'][s] is not None and abs(out['rs1'][s] - B1[s]) <= TOL_RS1 for s in SEEDS)
    out['gates'] = dict(rs1=rs1_ok, reproduced=not out['not_reproduced'], oracle=len(o) == 2)
    if not all(out['gates'].values()):
        out['readings'] = 'not read'
        return out
    nr = len(o[0]['rings'])
    ref = dict(rings=[sum(r['rings'][i]['bev'] for r in o) / 2 for i in range(nr)],
               pooled=sum(r['pooled']['bev'] for r in o) / 2)
    res = {}
    for name in ('RS5', 'RS5_3d', 'RS5_2d'):
        for kind in KINDS:
            seeds = []
            for s in SEEDS:
                r0, rx = _get(rows, 'RS1', s), _get(rows, name, s)
                if r0 is None or rx is None:
                    seeds.append(None)
                    continue
                g0, gx = gaps(r0, ref)[kind], gaps(rx, ref)[kind]
                d = g0 - gx
                seeds.append(dict(gap_rs1=g0, gap=gx, delta=d, R=d / g0 if g0 > 0 else None))
            w = word(seeds)
            res['%s_%s' % (name, kind)] = dict(word=w, hypothesis=hypothesis(name, kind, w), seeds=seeds)
    out['readings'] = res
    out['ref'] = ref
    return out


def cmd_read(path):
    rows = json.load(open(path))['rows']
    for r in sorted(rows, key=lambda r: (r['row'], r['seed'])):
        print('%-7s s%d pooled %6.2f (3D %6.2f) logged %s | %s | W&B %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), r.get('wandb'), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    out = readings(rows)
    print('\n== gates: %s; RS1 %s against 41a B1 %s' % (out['gates'], out['rs1'], B1))
    if out['readings'] == 'not read':
        print('== NOT READ')
    else:
        for k, v in out['readings'].items():
            print('== %-14s %-8s %s' % (k, v['word'], v['hypothesis'] or ''))
            for s, x in zip(SEEDS, v['seeds']):
                print('    s%d: gap RS1 %.2f -> %.2f, delta %+.2f, R %s' % (
                    s, x['gap_rs1'], x['gap'], x['delta'], 'n/a' if x['R'] is None else '%.2f' % x['R']))
    for s in SEEDS:
        f = result_dir('global', s) / 'stage0.json'
        if not f.exists():
            continue
        j = json.load(open(f))
        print('\nstage 0, seed %d (RS1 pass; source %d frames, target %d frames)' % (s, j['src_frames'], j['tgt_frames']))
        for g, v in j['groups'].items():
            print('  %-12s layers %3d share src %s tgt %s | |o_t-o_s| %s | |log r_t/r_s| %s' % (
                g, v['layers'], [round(x, 3) for x in v['share_src']], [round(x, 3) for x in v['share_tgt']],
                [round(x, 3) for x in v['d_mean']], [round(x, 3) for x in v['d_logsd']]))
    json.dump(out, open(Path(path).with_name('item70_readings.json'), 'w'), indent=1, default=float)


def main():
    a = sys.argv[1:]
    if a[:1] == ['runs'] and len(a) == 2:
        for r in runs(int(a[1])):
            print(' '.join(str(x) for x in r))
        return
    if a[:1] == ['manifest'] and len(a) == 2:
        rows = manifest()
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
