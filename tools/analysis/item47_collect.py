"""Collect item 47 (experiments_md 20261003_05 §5 item 47, design fixed 2026-11-21 19:30 JST) and apply its fixed readings and
procedure checks. Reads only logs and the rule-A JSON (seconds; fine on the master node).

    python3 analysis/item47_collect.py --jobs 34799 34800 34801 34802 [--json out.json]

Per row (analysis/make_item47_configs.py ROWS): Car AP_R40 BEV / 3D at IoU 0.7 and BEV at IoU 0.5 from the row's own adabn_eval
log (KITTI moderate; Waymo / nuScenes all objects); Pandar64 rows from analysis/item47_rule_a.py's JSON (rule A); the statistics
line; the W&B id from the eval jobs' logs.

Readings (verbatim from T05 item 47): structure_rev = P - L per seed on S' val, with stats none and target; two seeds, mean >= 2
with the same sign -> read; per-seed spread > 3 -> sign only. Where structure_rev (none) is read "+": target <= 1/2 of it or n.d.
-> "mostly statistics"; target >= 2/3 of it and read "+" -> "mostly weights"; otherwise "partly". Group verdict over K / W / P:
consistent when all three agree. Part B (nuScenes val) is descriptive.
"""
import argparse
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / 'analysis'))
import make_item47_configs as m47  # noqa: E402
from item42_collect import read_log, verdict  # noqa: E402

RULE_A = ['/home/koyama/data/item47/rule_a_none.json', '/home/koyama/data/item47/rule_a_target.json']
W_OWN_LATTICE = {666: 41.59, 667: 41.57}  # 31713 (a100): P on Waymo val thinned to its own training pattern
B_IN_RUN = {('BW', 'P'): (36.12, 36.30), ('BW', 'L'): (30.55, 30.78), ('BW', 'C1'): (31.14, 30.69),
            ('BP', 'P'): (33.93, 34.28), ('BP', 'L'): (27.89, 27.75), ('BP', 'C1'): (28.57, 28.24)}


def wandb_ids(jobs):
    """Row tag -> W&B run id from wandb's closing 'View run <name> at: .../runs/<id>' line (name and id on one line; the
    stdout log). Segmenting stderr by 'setting up run' mislabelled rows (fixed 2026-11-21 before the report)."""
    out = {}
    pat = re.compile(r'View run (i47[A-Z]+_\w+?_(?:none2|none|target)_s\d+_\d+) at: https://wandb\.ai/nagisa/st3d/runs/([a-z0-9]+)')
    for j in jobs:
        for log in (TOOLS / 'logs').glob('output_%s_*.txt' % j):
            text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text(errors='replace').replace('\r', '\n'))
            for m in pat.finditer(text):
                out.setdefault(m.group(1), m.group(2))
    return out


def rule_a(paths):
    out = {}
    for p in paths:
        if Path(p).exists():
            out.update({(r['stats'], r['row'], r['seed']): r for r in json.load(open(p)) if 'rule_A' in r})
    return out


def collect(jobs):
    ids, ra = wandb_ids(jobs), rule_a(RULE_A)
    res = {}
    for part, row, seed, job, epoch, stats, cfg, _ in m47.ROWS:
        t = m47.tag(part, row, seed, job, stats)
        got = read_log((m47.result_pkl(part, row, seed, job, epoch, stats, cfg).parent, 'log_adabn_*.txt', None))
        r = dict(part=part, row=row, seed=seed, job=job, stats=stats, wandb=ids.get(t), log=got)
        if part == 'P':
            a = ra.get((stats, row, seed))
            r['val'] = dict(bev07=a['rule_A'][0], d307=a['rule_A'][1], bev05=a['rule_A'][2],
                            as_scored=a['as_scored']) if a else None
        else:
            r['val'] = got
        res[t] = r
    return res


def value(res, part, row, seed, stats, key='bev07'):
    hit = [r for r in res.values() if (r['part'], r['row'], r['seed'], r['stats']) == (part, row, seed, stats)]
    return hit[0]['val'].get(key) if hit and hit[0]['val'] else None


def contrast(res, part, a, b, stats, key='bev07'):
    c = []
    for s in (666, 667):
        x, y = value(res, part, a, s, stats, key), value(res, part, b, s, stats, key)
        c.append(x - y if x is not None and y is not None else None)
    vd, mean, spread = verdict(c)
    return dict(seeds=c, mean=mean, spread=spread, verdict=vd)


def classify(no, ada):
    """The item 47 AdaBN reading of a structure_rev read '+' without AdaBN."""
    if not no['verdict'].startswith('READ "+"'):
        return 'no reverse structure term' if no['verdict'] != 'pending' else 'pending'
    if ada['verdict'] == 'pending':
        return 'pending'
    if ada['verdict'] == 'n.d.' or ada['mean'] <= 0.5 * no['mean']:
        return 'mostly statistics'
    if ada['verdict'].startswith('READ "+"') and ada['mean'] >= 2.0 / 3.0 * no['mean']:
        return 'mostly weights'
    return 'partly'


def checks(res):
    out = []
    for s in (666, 667):  # 2: P none on Waymo S' against 31713
        v = value(res, 'W', 'P', s, 'none')
        out.append(('2 W P none s%d vs 31713 %.2f' % (s, W_OWN_LATTICE[s]), v, None if v is None else abs(v - W_OWN_LATTICE[s]) <= 0.3))
    for part in ('K', 'W', 'P'):  # 3: P on its own lattice under AdaBN
        for s in (666, 667):
            a, n = value(res, part, 'P', s, 'target'), value(res, part, 'P', s, 'none')
            d = None if a is None or n is None else a - n
            out.append(('3 dP %s s%d (target - none)' % (part, s), d, None if d is None else abs(d) <= 1.0))
    a, n = value(res, 'W', 'P', 666, 'none2'), value(res, 'W', 'P', 666, 'none')  # 4: Waymo redraw
    d = None if a is None or n is None else a - n
    out.append(('4 W P s666 none2 - none', d, None if d is None else abs(d) <= 0.3))
    for (part, row), banked in B_IN_RUN.items():  # 5: B none against the in-run values
        for s, b in zip((666, 667), banked):
            v = value(res, part, row, s, 'none')
            out.append(('5 %s %s none s%d vs in-run %.2f' % (part, row, s, b), v, None if v is None else abs(v - b) <= 0.05))
    for t, r in res.items():  # 6: statistics line
        st = (r['log'] or {}).get('stats') if r['log'] else None
        if r['log'] is None:
            continue
        ok = (st is not None and st['frames'] == '1000') if r['stats'] == 'target' else st is None
        if not ok:
            out.append(('6 stats line %s' % t, st, False))
    return out


def fmt(x, signed=False):
    return '-' if x is None else (('%+.2f' if signed else '%.2f') % x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jobs', nargs='+', default=[])
    p.add_argument('--json', default=None)
    args = p.parse_args()
    res = collect(args.jobs)
    print('## rows (Car BEV / 3D @0.7, BEV @0.5; Pandar64 rule A)')
    for t, r in res.items():
        v = r['val'] or {}
        print('%-30s %7s %7s %7s  %s' % (t, fmt(v.get('bev07')), fmt(v.get('d307')), fmt(v.get('bev05')), r['wandb'] or '-'))
    summary = {}
    print('\n## contrasts')
    for key in ('bev07', 'bev05'):
        for part in ('K', 'W', 'P', 'BW', 'BP'):
            for name, a, b in (('structure_rev P - L', 'P', 'L'), ('P - C1', 'P', 'C1'), ('P - O', 'P', 'O')):
                if part.startswith('B') and b == 'O':
                    continue
                c = {st: contrast(res, part, a, b, st, key) for st in ('none', 'target')}
                line = '%-5s %-3s %-20s ' % (key, part, name) + '  '.join(
                    '%s: %s / %s mean %s spread %s %s' % (st, fmt(c[st]['seeds'][0], True), fmt(c[st]['seeds'][1], True),
                                                         fmt(c[st]['mean'], True), fmt(c[st]['spread']), c[st]['verdict'])
                    for st in ('none', 'target'))
                if name.startswith('structure') and not part.startswith('B') and key == 'bev07':
                    c['reading'] = classify(c['none'], c['target'])
                    line += '  -> %s' % c['reading']
                print(line)
                summary['%s %s %s' % (key, part, name)] = c
    group = [summary['bev07 %s structure_rev P - L' % p].get('reading') for p in ('K', 'W', 'P')]
    gv = 'pending' if 'pending' in group else ('consistent: %s' % group[0] if len(set(group)) == 1 else 'mixed')
    print('\ngroup verdict (K, W, P): %s %s' % (gv, group))
    print('\n## procedure checks')
    ck = checks(res)
    for name, v, ok in ck:
        print('%-45s %-10s %s' % (name, fmt(v, True) if isinstance(v, float) else v, {True: 'pass', False: 'FLAG', None: 'pending'}[ok]))
    if args.json:
        json.dump(dict(rows=res, contrasts=summary, group=gv, checks=ck), open(args.json, 'w'), indent=1, default=str)
        print('wrote', args.json)


if __name__ == '__main__':
    main()
