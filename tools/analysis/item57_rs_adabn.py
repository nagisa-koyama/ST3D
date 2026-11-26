"""Item 57 (experiments_md 20261003_05 §5 item 57, design fixed 2026-11-27 00:10 JST, experiments_md commit b430082, before any
number): range-stratified AdaBN on item 48's forward pair. Rows, the manifest for item 38's ring scorer, and the fixed readings.

    python3 analysis/item57_rs_adabn.py runs                     # "seed job arm tag" for the runner
    python3 analysis/item57_rs_adabn.py manifest OUT.json        # 4 new rows (P_K RS-1 / RS-5 x seeds) + item 48's AdaBN L / O / P
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --real-kernel --workers 4
    python3 analysis/item57_rs_adabn.py read AP.json [STAGE0_DIR...]

Readings (T05 item 57), per seed, s_k(b) = L(b) - P_RSk(b), L = L_K under global AdaBN, item 48's rings:
1. Over 20-30 and 30-40 m: delta = mean(s_1 - s_5), R = delta / mean(s_1). R >= 0.5 and delta >= 2 AP at both seeds: "RS-AdaBN
   removes most of the far residual"; R <= 0.2 or delta < 1 AP at both seeds: "does not remove it"; otherwise "partly".
2. Mean over 0-10 and 10-20 m of P_RS5 - P_RS1 >= -2 AP at both seeds: "near kept"; otherwise "costs near".
Procedure: RS-1 pooled within 0.3 AP of item 48's AdaBN P_K row at a seed, else that seed is not read; every scored row reproduces
its logged pooled value within 0.02. Tests: tests/test_item57_rs_adabn.py.
"""
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))

REPO = TOOLS.parent
EVAL_CFG = 'cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml'
SRC_CFG = 'cfgs/da-ieee-access-analysis/item47/centerpoint-item47-kitti-p-on-nuscenes-lattice.yaml'
FWD = '/home/koyama/data/item47/item48_manifest_fwd.json'
CKPTS = [(666, 32283), (667, 32284)]
EPOCH = 152
ARMS = {'global': 'P_RS1', 'item57': 'P_RS5', 'item57_3d': 'P_RS5_3d'}   # item57_3d: item 57b
REF = {'P': 'P_ada', 'L': 'L_ada', 'O': 'O_ada'}
FAR, NEAR = (2, 3), (0, 1)          # item 48's rings: 20-30, 30-40 | 0-10, 10-20 m
TOL_RS1 = 0.3


def tag(arm, seed, job):
    return 'i57_%s_s%d_%d' % (arm, seed, job)


def result_dir(arm, seed, job):
    t = tag(arm, seed, job)
    return REPO / 'output' / 'da-ieee-access-analysis' / 'factorswap-s2' / Path(EVAL_CFG).stem / t / 'eval' / ('epoch_%d' % EPOCH) / t


def runs(arms=None):
    """Item 57's runs by default (global, item57); item 57b's runner asks for 'item57_3d'."""
    arms = arms or ('global', 'item57')
    return [(s, j, a, tag(a, s, j)) for s, j in CKPTS for a in ARMS if a in arms]


def manifest():
    from item42_collect import read_log
    out = []
    for seed, job in CKPTS:
        for arm, name in ARMS.items():
            d = result_dir(arm, seed, job)
            got = read_log((d, 'log_rsadabn_*.txt', None))
            out.append(dict(group='item57', row=name, seed=seed, job=job, result=str(d / 'result.pkl'),
                            logged_bev=got['bev07'] if got else None,
                            status='ok' if (d / 'result.pkl').exists() and got else 'missing'))
    for r in json.load(open(FWD)):
        if r['stats'] == 'target' and r['row'] in REF:
            out.append(dict(r, group='item57', row=REF[r['row']]))
    return out


def rings_of(rows, name, seed):
    hit = [r for r in rows if r['row'] == name and r['seed'] == seed]
    if len(hit) != 1 or not hit[0].get('reproduced'):
        return None
    return hit[0]


def mean(xs):
    return sum(xs) / len(xs)


def per_seed(rows, seed, arm='P_RS5'):
    """None if the seed is not read; else the quantities of readings 1 and 2 for `arm` (P_RS5, or P_RS5_3d for item 57b)."""
    p1, p5, lref, pada = (rings_of(rows, n, seed) for n in ('P_RS1', arm, 'L_ada', 'P_ada'))
    if any(x is None for x in (p1, p5, lref, pada)):
        return None
    if abs(p1['pooled']['bev'] - pada['logged_bev']) > TOL_RS1:
        return dict(check='RS-1 %.2f vs AdaBN %.2f: not read' % (p1['pooled']['bev'], pada['logged_bev']))
    L = [g['bev'] for g in lref['rings']]
    P1 = [g['bev'] for g in p1['rings']]
    P5 = [g['bev'] for g in p5['rings']]
    s1 = [a - b for a, b in zip(L, P1)]
    s5 = [a - b for a, b in zip(L, P5)]
    delta = mean([s1[i] - s5[i] for i in FAR])
    base = mean([s1[i] for i in FAR])
    return dict(check='ok', s1=s1, s5=s5, delta=delta, R=delta / base if base else None,
                near=mean([P5[i] - P1[i] for i in NEAR]),
                pooled=(p1['pooled']['bev'], p5['pooled']['bev']), pooled3d=(p1['pooled']['3d'], p5['pooled']['3d']),
                far40=(P1[4], P5[4]))


def reading1(seeds):
    if any(s is None or s.get('check') != 'ok' for s in seeds):
        return 'not read'
    if all(s['R'] is not None and s['R'] >= 0.5 and s['delta'] >= 2.0 for s in seeds):
        return 'RS-AdaBN removes most of the far residual'
    if all((s['R'] is not None and s['R'] <= 0.2) or s['delta'] < 1.0 for s in seeds):
        return 'does not remove it'
    return 'partly'


def reading2(seeds):
    if any(s is None or s.get('check') != 'ok' for s in seeds):
        return 'not read'
    return 'near kept' if all(s['near'] >= -2.0 for s in seeds) else 'costs near'


def cmd_read(path, stage0_dirs):
    rows = json.load(open(path))['rows']
    for r in rows:
        flag = '' if r.get('reproduced') else '  NOT REPRODUCED'
        print('%-6s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), flag))
    for arm in ('P_RS5', 'P_RS5_3d'):
        if not any(r['row'] == arm for r in rows):
            continue
        seeds = [per_seed(rows, s, arm) for s, _ in CKPTS]
        print('\n== %s against P_RS1' % arm)
        for (s, _), d in zip(CKPTS, seeds):
            print('seed %d: %s' % (s, d))
        print('reading 1 (far residual): %s' % reading1(seeds))
        print('reading 2 (near guard): %s' % reading2(seeds))
    for d in stage0_dirs:
        j = json.load(open(Path(d) / 'stage0.json'))
        print('\nstage 0 %s (arm knots %s)' % (d, j['arm_knots']))
        for g, v in j['groups'].items():
            print('  %-12s layers %3d share src %s tgt %s | |o_t-o_s| %s | |log r_t/r_s| %s' % (
                g, v['layers'], [round(x, 3) for x in v['share_src']], [round(x, 3) for x in v['share_tgt']],
                [round(x, 3) for x in v['d_mean']], [round(x, 3) for x in v['d_logsd']]))


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == 'runs':
        for r in runs(sys.argv[2:] or None):
            print(' '.join(str(x) for x in r))
        return
    if len(sys.argv) == 3 and sys.argv[1] == 'manifest':
        rows = manifest()
        json.dump(rows, open(sys.argv[2], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r['job'], r['status'], r['logged_bev'])
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if len(sys.argv) >= 3 and sys.argv[1] == 'read':
        cmd_read(sys.argv[2], sys.argv[3:])
        return
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
