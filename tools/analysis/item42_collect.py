"""Collect the item 42a / 42a' results (experiments_md 20261108_05; T05 §5 item 42, design fixed 2026-11-08 22:40 JST) and apply
the fixed readings. Reads only logs and rescoring outputs (seconds; fine on the master node).

    python3 analysis/item42_collect.py --jobs <eval job ids> [--rule_a <rescore txt>] [--json out.json]

Per row (analysis/make_item42_configs.py ROWS): the AdaBN statistics line (split, frames), Car AP_R40 BEV / 3D at IoU 0.7
(easy / moderate / hard), BEV at IoU 0.5, from the row's own log in its output dir; Pandar64 rows read rule A from the CPU
rescoring file; the eval W&B run id from the eval jobs' logs. Values without AdaBN are the banked readings (T05 items 26, 33,
34, 36, 37), listed in NO_ADABN.

Readings (T05 item 42a, verbatim rule): per target and seed, structure = L - P, pattern = C1 - P, density = O - C1; two seeds,
mean >= 2 with the same sign -> read; per-seed contrast spread > 3 -> sign only.
"""
import argparse
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / 'analysis'))
import make_item42_configs as m42  # noqa: E402

# Car BEV AP_R40 at IoU 0.7 as read before (target's usual scoring), by (part, row) -> (seed 666, seed 667)
NO_ADABN = {
    ('K', 'O'): (80.33, 78.79), ('K', 'P'): (19.77, 22.73), ('K', 'L'): (68.09, 51.86), ('K', 'C1'): (58.21, 66.16),
    ('K', 'R1'): (71.04, 49.50), ('K', 'R2'): (76.42, 76.29),
    ('W', 'O'): (65.63, 65.62), ('W', 'P'): (19.60, 10.28), ('W', 'L'): (55.40, 54.31), ('W', 'C1'): (57.78, 58.92),
    ('P', 'O'): (81.67, 81.85), ('P', 'P'): (26.36, 29.13), ('P', 'L'): (62.22, 64.20), ('P', 'C1'): (62.43, 64.72),
    ('L', 'O'): (76.60, 76.66), ('L', 'P'): (44.81, 37.94), ('L', 'L'): (65.49, 66.06), ('L', 'C1'): (64.02, 64.82),
    ('A', 'L'): (68.09, 51.86),  # 42a': full val, as trained
}
STATS = re.compile(r'\[adabn\] statistics from (\d+) of (\d+) (\w+) frames of (\w+) \(stride (\d+)\)')


def ap_block(lines, head):
    """(bev easy, mod, hard), (3d easy, mod, hard) of the first block whose header is `head`."""
    for i, line in enumerate(lines):
        if line.strip() == head:
            bev = d3 = None
            for nxt in lines[i + 1:i + 6]:
                if nxt.startswith('bev  AP:'):
                    bev = tuple(float(v) for v in nxt.split(':', 1)[1].split(','))
                if nxt.startswith('3d   AP:'):
                    d3 = tuple(float(v) for v in nxt.split(':', 1)[1].split(','))
            return bev, d3
    return None, None


def row_log(part, row, seed, job, epoch, cfg):
    d = m42.result_pkl(part, row, seed, job, epoch, cfg).parent
    logs = sorted(d.glob('log_adabn_*.txt' if part != 'A' else 'log_eval_*.txt'))
    return logs[-1] if logs else None


def wandb_ids(jobs):
    out = {}
    for j in jobs:
        for log in (TOOLS / 'logs').glob('error_%s_*.txt' % j):
            text = log.read_text(errors='replace').replace('\r', '\n')
            # one segment per wandb.init; the first item-42 tag after it names the row (adabn_eval's run name, or the
            # eval_tag test.py logs among its arguments: eval_checkpoint.sh passes no run name)
            for seg in text.split('wandb: setting up run ')[1:]:
                rid = seg.split('\n', 1)[0].strip()
                m = re.search(r'\b(i42ap?_[A-Z]_\w+?_s\d+_\d+)\b', seg)
                if m and m.group(1) not in out:
                    out[m.group(1)] = rid
    return out


def rule_a(path):
    """label -> (as_scored bev, 3d, rule_A bev, 3d) from analysis/pandaset_eval_protocols.py's table."""
    out = {}
    if not path or not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        m = re.match(r'^(P \S+ s\d+ \(job \d+\))\s+([\d.]+) /\s+([\d.]+)\s+([\d.]+) /\s+([\d.]+)', line)
        if m:
            out[m.group(1)] = tuple(float(m.group(k)) for k in range(2, 6))
    return out


def verdict(c):
    """Two seeds: mean >= 2 with the same sign -> read; per-seed spread > 3 -> sign only."""
    if any(v is None for v in c):
        return 'pending', None, None
    mean, spread = (c[0] + c[1]) / 2, abs(c[0] - c[1])
    same = (c[0] > 0) == (c[1] > 0) and c[0] != 0 and c[1] != 0
    if same and abs(mean) >= 2:
        v = '+' if mean > 0 else '-'
        return ('READ "%s"%s' % (v, ', in sign only' if spread > 3 else '')), mean, spread
    return 'n.d.', mean, spread


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jobs', nargs='+', default=[])
    p.add_argument('--rule_a', default=None)
    p.add_argument('--json', default=None)
    args = p.parse_args()
    ids, ra = wandb_ids(args.jobs), rule_a(args.rule_a)
    res = {}
    for part, row, seed, job, epoch, cfg, _ in m42.ROWS:
        t = m42.tag(part, row, seed, job)
        r = dict(part=part, row=row, seed=seed, job=job, epoch=epoch, cfg=cfg, wandb=ids.get(t))
        log = row_log(part, row, seed, job, epoch, cfg)
        if log is not None:
            lines = log.read_text(errors='replace').replace('\r', '\n').splitlines()
            m = STATS.search('\n'.join(lines))
            r['stats'] = dict(zip(('frames', 'of', 'split', 'dataset', 'stride'), m.groups())) if m else None
            r['bev'], r['d3'] = ap_block(lines, 'Car AP_R40@0.70, 0.70, 0.70:')
            r['bev05'], _ = ap_block(lines, 'Car AP_R40@0.70, 0.50, 0.50:')
            r['log'] = str(log)
        lab = '%s %s s%d (job %d)' % (part, row, seed, job)
        if part == 'P' and lab in ra:
            r['as_scored'], r['rule_a'] = ra[lab][:2], ra[lab][2:]
        # the reading metric: Pandar64 rule A BEV; every other target BEV moderate
        if part == 'P':
            r['read'] = r['rule_a'][0] if r.get('rule_a') else None
        else:
            r['read'] = r['bev'][1] if r.get('bev') else None
        r['no_adabn'] = NO_ADABN[(part, row)][0 if seed == 666 else 1]
        res[t] = r

    def val(part, row, seed):
        hit = [r for r in res.values() if (r['part'], r['row'], r['seed']) == (part, row, seed)]
        return hit[0]['read'] if hit else None

    def base(part, row, seed):
        return NO_ADABN[(part, row)][0 if seed == 666 else 1]

    print('| target | row | seed | job | eval W&B | stats (frames of split) | AdaBN BEV mod (P: rule A) | 3D mod | BEV@0.5 mod | '
          'no AdaBN | change |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for t, r in res.items():
        st = r.get('stats')
        print('| %s | %s | %d | %d | %s | %s | %s | %s | %s | %.2f | %s |' % (
            r['part'], r['row'], r['seed'], r['job'], r['wandb'] or '-',
            '%s of %s %s' % (st['frames'], st['of'], st['split']) if st else '-',
            '%.2f' % r['read'] if r['read'] is not None else '-',
            '%.2f' % (r['rule_a'][1] if r['part'] == 'P' and r.get('rule_a') else r['d3'][1]) if r.get('d3') else '-',
            '%.2f' % r['bev05'][1] if r.get('bev05') else '-', r['no_adabn'],
            '%+.2f' % (r['read'] - r['no_adabn']) if r['read'] is not None else '-'))
    print()
    contrasts = {}
    for part in ('K', 'W', 'P', 'L'):
        for name, (a, b) in (('structure', ('L', 'P')), ('pattern', ('C1', 'P')), ('density', ('O', 'C1'))):
            for mode, f in (('adabn', val), ('no_adabn', base)):
                c = tuple((f(part, a, s) - f(part, b, s)) if f(part, a, s) is not None and f(part, b, s) is not None
                          else None for s in (666, 667))
                v, mean, spread = verdict(c)
                contrasts[(part, name, mode)] = dict(seeds=c, mean=mean, spread=spread, verdict=v)
    for part, name, (a, b) in (('K', 'lines R2 - R1', ('R2', 'R1')), ('K', 'density O - R3', ('O', 'C1'))):
        for mode, f in (('adabn', val), ('no_adabn', base)):
            c = tuple((f(part, a, s) - f(part, b, s)) if f(part, a, s) is not None and f(part, b, s) is not None
                      else None for s in (666, 667))
            v, mean, spread = verdict(c)
            contrasts[(part, name, mode)] = dict(seeds=c, mean=mean, spread=spread, verdict=v)
    print('| target | contrast | AdaBN 666 / 667 | mean | spread | verdict | no AdaBN 666 / 667 | mean | spread | verdict |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for (part, name, mode), c in contrasts.items():
        if mode != 'adabn':
            continue
        n = contrasts[(part, name, 'no_adabn')]
        fmt = lambda x: '%+.2f' % x if x is not None else '-'  # noqa: E731
        print('| %s | %s | %s / %s | %s | %s | %s | %s / %s | %s | %s | %s |' % (
            part, name, fmt(c['seeds'][0]), fmt(c['seeds'][1]), fmt(c['mean']),
            '%.2f' % c['spread'] if c['spread'] is not None else '-', c['verdict'],
            fmt(n['seeds'][0]), fmt(n['seeds'][1]), fmt(n['mean']), '%.2f' % n['spread'], n['verdict']))
    # 42a': L_K on its own-count val, no AdaBN
    ap = [val('A', 'L', s) for s in (666, 667)]
    if None not in ap:
        sp = abs(ap[0] - ap[1])
        print("\n42a': L_K on its own-count KITTI val: %.2f / %.2f, spread %.2f -> %s" % (
            ap[0], ap[1], sp, 'L_K extrapolates as R1 / R3' if sp <= 3 else 'reported as is'))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(rows=res, contrasts={'|'.join(k): v for k, v in contrasts.items()}), open(args.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
