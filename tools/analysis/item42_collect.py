"""Collect the item 42a / 42a' results (experiments_md 20261108_05; T05 §5 item 42, design fixed 2026-11-08 22:40 JST) and apply
the fixed readings, at IoU 0.7 (the reading) and IoU 0.5 BEV (beside, round-5 review), with and without AdaBN. Reads only logs
and rescoring outputs (seconds; fine on the master node).

    python3 analysis/item42_collect.py --jobs <eval job ids> [--rule_a /home/koyama/data/item42a/rule_a_iou.json] [--json out]

Per row (analysis/make_item42_configs.py ROWS): the AdaBN statistics line (split, frames) and Car AP_R40 BEV / 3D at IoU 0.7 and
BEV at IoU 0.5, moderate, from the row's own log in its output dir; the same numbers WITHOUT AdaBN from the log of the
evaluation the banked value came from (NO_ADABN_LOG; its IoU 0.7 BEV must equal the banked NO_ADABN value within 0.01, checked);
Pandar64 rows read rule A for both from analysis/item42_rule_a.py's JSON; the eval W&B ids from the eval jobs' logs.

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

OUT = TOOLS.parent / 'output'
W = Path('/home/koyama/data/wandb')
# Car BEV AP_R40 at IoU 0.7 as banked (target's usual scoring), by (part, row) -> (seed 666, seed 667)
NO_ADABN = {
    ('K', 'O'): (80.33, 78.79), ('K', 'P'): (19.77, 22.73), ('K', 'L'): (68.09, 51.86), ('K', 'C1'): (58.21, 66.16),
    ('K', 'R1'): (71.04, 49.50), ('K', 'R2'): (76.42, 76.29),
    ('W', 'O'): (65.63, 65.62), ('W', 'P'): (19.60, 10.28), ('W', 'L'): (55.40, 54.31), ('W', 'C1'): (57.78, 58.92),
    ('P', 'O'): (81.67, 81.85), ('P', 'P'): (26.36, 29.13), ('P', 'L'): (62.22, 64.20), ('P', 'C1'): (62.43, 64.72),
    ('L', 'O'): (76.60, 76.66), ('L', 'P'): (44.81, 37.94), ('L', 'L'): (65.49, 66.06), ('L', 'C1'): (64.02, 64.82),
    ('A', 'L'): (68.09, 51.86),  # 42a': full val, as trained
}


def _train(run, epoch):  # in-run evaluation in the training log, block of that epoch
    return (W / run / 'files', 'log_train_*.txt', epoch)


def _ev(rel):  # an eval-only job's own log (test.py)
    return (OUT / rel, 'log_eval_*.txt', None)


_S2 = 'da-ieee-access-analysis/factorswap-s2/'
_WW = 'da-ieee-access/centerpoint-sourceonly-waymo2waymo/20260922_sourceonly/eval/epoch_7/val/'
_L40 = 'da-ieee-access-analysis/factorswap-lyft40/centerpoint-factorswap-lyft40-evalonly-'
NO_ADABN_LOG = {  # (part, row, seed) -> where the banked no-AdaBN value was printed (Pandar64: item42_rule_a.py instead)
    ('K', 'O', 666): _train('run-20261001_005323-06iwywhs', 152), ('K', 'O', 667): _train('run-20261105_115743-7l8uog7k', 152),
    ('K', 'P', 666): _train('run-20261108_020021-0i5ai4hn', 152), ('K', 'P', 667): _train('run-20261108_020023-1ya9lozy', 152),
    ('K', 'L', 666): _train('run-20261108_020415-90shlq10', 152), ('K', 'L', 667): _train('run-20261108_020416-nqzlmau3', 152),
    ('K', 'C1', 666): _train('run-20261103_153741-1udhncb3', 152), ('K', 'C1', 667): _train('run-20261106_122650-c1onqx95', 152),
    ('K', 'R1', 666): _train('run-20261102_211349-tpkt5qno', 152), ('K', 'R1', 667): _train('run-20261106_122650-2tbnxqib', 152),
    ('K', 'R2', 666): _train('run-20261102_211414-oh2qkjth', 152), ('K', 'R2', 667): _train('run-20261106_122649-kq6cuzer', 152),
    ('W', 'O', 666): _train('run-20261104_145730-026efrl5', 7), ('W', 'O', 667): _train('run-20261105_115835-d85729y1', 7),
    ('W', 'P', 666): _ev(_WW + 'item26_P_27807'), ('W', 'P', 667): _ev(_WW + 'item26_P_27840'),
    ('W', 'L', 666): _ev(_WW + 'item37a_LW_31823'), ('W', 'L', 667): _ev(_WW + 'item37a_LW_31824'),
    ('W', 'C1', 666): _ev(_WW + 'item26_C1_27850'), ('W', 'C1', 667): _ev(_WW + 'item26_C1_28278'),
    ('L', 'O', 666): _train('run-20261106_145000-1r0c9h3f', 30), ('L', 'O', 667): _train('run-20261107_131329-jrz4p4d9', 30),
    ('L', 'P', 666): _ev(_L40 + 'p-on-lyft40/20261106_fs34/eval/epoch_30/val/fs34_P_30312'),
    ('L', 'P', 667): _ev(_L40 + 'p-on-lyft40/20261106_fs34/eval/epoch_30/val/fs34_P_30363'),
    ('L', 'L', 666): _ev(_L40 + 'l-on-lyft40/20261108_fs37/eval/epoch_30/val/fs37_LL40_s666'),
    ('L', 'L', 667): _ev(_L40 + 'l-on-lyft40/20261108_fs37/eval/epoch_30/val/fs37_LL40_s667'),
    ('L', 'C1', 666): _ev(_L40 + 'c1-on-lyft40/20261106_fs34/eval/epoch_30/val/fs34_C1_30313'),
    ('L', 'C1', 667): _ev(_L40 + 'c1-on-lyft40/20261106_fs34/eval/epoch_30/val/fs34_C1_30364'),
}
NO_ADABN_LOG[('A', 'L', 666)], NO_ADABN_LOG[('A', 'L', 667)] = NO_ADABN_LOG[('K', 'L', 666)], NO_ADABN_LOG[('K', 'L', 667)]
# the S2 source-only control (nuScenes oracle 25743 / 31818 / 32207, epoch 20) on KITTI val: item 33f (no AdaBN), item 39 B1
_C = 'da-ieee-access/centerpoint-sourceonly-nuscenes2kitti/'
CONTROL = {
    666: dict(no=_ev(_C + '20261107_i33f/eval/epoch_20/val/i33f_kitti_s666'), no_wandb='adabx71e',
              ada=(OUT / _C / 'i39_adabn_s666/eval/epoch_20/i39_adabn_s666', 'log_adabn_*.txt', None), ada_wandb='zjczud17'),
    667: dict(no=_ev(_C + '20261107_i33f/eval/epoch_20/val/i33f_kitti_s667'), no_wandb='w4d5wls8',
              ada=(OUT / _C / 'i39_adabn_s667/eval/epoch_20/i39_adabn_s667', 'log_adabn_*.txt', None), ada_wandb='gwoittgl'),
    668: dict(no=_ev(_C + '20261107_i33f/eval/epoch_20/val/i33f_kitti_s668'), no_wandb='s8baki8r',
              ada=(OUT / _C / 'i39_adabn_s668/eval/epoch_20/i39_adabn_s668', 'log_adabn_*.txt', None), ada_wandb='ri8q517s'),
}
STATS = re.compile(r'\[adabn\] statistics from (\d+) of (\d+) (\w+) frames of (\w+) \(stride (\d+)\)')
H07, H05 = 'Car AP_R40@0.70, 0.70, 0.70:', 'Car AP_R40@0.70, 0.50, 0.50:'


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


def read_log(spec):
    """dict(bev07, d307, bev05 (moderate), bev07_emh, log) from (dir, glob, epoch-or-None)."""
    d, pattern, epoch = spec
    for log in sorted(Path(d).glob(pattern), reverse=True):  # a 2-GPU run writes one log per rank; the one with the block
        got = _read_one(log, epoch)
        if got is not None:
            return got
    return None


def _read_one(log, epoch):
    lines = log.read_text(errors='replace').replace('\r', '\n').splitlines()
    if epoch is not None:  # the in-run evaluation block of that epoch
        s = [i for i, l in enumerate(lines) if re.search(r'EPOCH %d of \w+ EVALUATION start' % epoch, l)]
        e = [i for i, l in enumerate(lines) if re.search(r'EPOCH %d of \w+ EVALUATION end' % epoch, l)]
        if not s or not e:
            return None
        lines = lines[s[-1]:e[-1] + 1]
    bev, d3 = ap_block(lines, H07)
    bev05, _ = ap_block(lines, H05)
    if bev is None:
        return None
    m = STATS.search('\n'.join(lines))
    return dict(bev07=bev[1], d307=d3[1] if d3 else None, bev05=bev05[1] if bev05 else None, bev07_emh=bev,
                stats=dict(zip(('frames', 'of', 'split', 'dataset', 'stride'), m.groups())) if m else None,
                log=str(log))


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


def rule_a_rows(paths):
    """(adabn, row, seed) -> {'as_scored': (bev07, 3d07, bev05), 'rule_A': (...)} from item42_rule_a.py's JSON file(s)."""
    out = {}
    for path in ([paths] if isinstance(paths, str) else (paths or [])):
        if Path(path).exists():
            out.update({(r['adabn'], r['row'], r['seed']): r for r in json.load(open(path)) if 'rule_A' in r})
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


def collect(jobs, rule_a_path):
    ids, ra = wandb_ids(jobs), rule_a_rows(rule_a_path)
    res, problems = {}, []
    for part, row, seed, job, epoch, cfg, _ in m42.ROWS:
        t = m42.tag(part, row, seed, job)
        r = dict(part=part, row=row, seed=seed, job=job, epoch=epoch, cfg=cfg, wandb=ids.get(t))
        d = m42.result_pkl(part, row, seed, job, epoch, cfg).parent
        got = read_log((d, 'log_adabn_*.txt' if part != 'A' else 'log_eval_*.txt', None))
        if part == 'P':
            a, n = ra.get((True, row, seed)), ra.get((False, row, seed))
            r['ada'] = dict(bev07=a['rule_A'][0], d307=a['rule_A'][1], bev05=a['rule_A'][2],
                            as_scored=a['as_scored'], stats=got['stats'] if got else None,
                            as_scored_log=got) if a else (dict(stats=got['stats'], as_scored_log=got) if got else None)
            r['no'] = dict(bev07=n['rule_A'][0], d307=n['rule_A'][1], bev05=n['rule_A'][2], as_scored=n['as_scored']) if n else None
        else:
            r['ada'] = got
            r['no'] = read_log(NO_ADABN_LOG[(part, row, seed)])
        banked = NO_ADABN[(part, row)][0 if seed == 666 else 1]
        r['banked'] = banked
        if r['no'] and r['no'].get('bev07') is not None and abs(r['no']['bev07'] - banked) > 0.01:
            problems.append('%s: no-AdaBN log gives %.4f, banked %.2f' % (t, r['no']['bev07'], banked))
        res[t] = r
    return res, problems


def contrasts(res):
    def v(part, row, seed, mode, key):
        hit = [r for r in res.values() if (r['part'], r['row'], r['seed']) == (part, row, seed)]
        x = hit[0][mode] if hit else None
        return x.get(key) if x else None
    out = {}
    defs = [('structure', 'L', 'P'), ('pattern', 'C1', 'P'), ('density', 'O', 'C1')]
    for key in ('bev07', 'bev05'):
        for mode in ('ada', 'no'):
            for part in ('K', 'W', 'P', 'L'):
                for name, a, b in defs + ([('lines R2 - R1', 'R2', 'R1'), ('density O - R3', 'O', 'C1')] if part == 'K' else []):
                    c = []
                    for s in (666, 667):
                        x, y = v(part, a, s, mode, key), v(part, b, s, mode, key)
                        c.append(x - y if x is not None and y is not None else None)
                    vd, mean, spread = verdict(c)
                    out[(key, mode, part, name)] = dict(seeds=c, mean=mean, spread=spread, verdict=vd)
    return out


def fmt(x, signed=False):
    return '-' if x is None else (('%+.2f' if signed else '%.2f') % x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jobs', nargs='+', default=[])
    p.add_argument('--rule_a', nargs='+', default=['/home/koyama/data/item42a/rule_a_iou.json'])
    p.add_argument('--json', default=None)
    args = p.parse_args()
    res, problems = collect(args.jobs, args.rule_a)
    for x in problems:
        print('CHECK FAILED:', x)
    print('| target | row | seed | job | eval W&B | AdaBN statistics | AdaBN BEV@0.7 | 3D@0.7 | BEV@0.5 | no AdaBN BEV@0.7 | '
          '3D@0.7 | BEV@0.5 | change BEV@0.7 | change BEV@0.5 |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for t, r in res.items():
        a, n = r['ada'] or {}, r['no'] or {}
        st = a.get('stats')
        d07 = (a['bev07'] - n['bev07']) if a.get('bev07') is not None and n.get('bev07') is not None else None
        d05 = (a['bev05'] - n['bev05']) if a.get('bev05') is not None and n.get('bev05') is not None else None
        print('| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['part'], r['row'], r['seed'], r['job'], r['wandb'] or '-',
            '%s of %s %s' % (st['frames'], st['of'], st['split']) if st else ('none (42a′)' if r['part'] == 'A' else '-'),
            fmt(a.get('bev07')), fmt(a.get('d307')), fmt(a.get('bev05')), fmt(n.get('bev07')), fmt(n.get('d307')),
            fmt(n.get('bev05')), fmt(d07, True), fmt(d05, True)))
    cs = contrasts(res)
    for key in ('bev07', 'bev05'):
        print('\n%s: | target | contrast | AdaBN 666 / 667 | mean | spread | verdict | no AdaBN 666 / 667 | mean | spread | verdict |'
              % key)
        for (k, mode, part, name), c in cs.items():
            if k != key or mode != 'ada':
                continue
            n = cs[(key, 'no', part, name)]
            print('| %s | %s | %s / %s | %s | %s | %s | %s / %s | %s | %s | %s |' % (
                part, name, fmt(c['seeds'][0], True), fmt(c['seeds'][1], True), fmt(c['mean'], True), fmt(c['spread']),
                c['verdict'], fmt(n['seeds'][0], True), fmt(n['seeds'][1], True), fmt(n['mean'], True), fmt(n['spread']),
                n['verdict']))
    # P_K against the S2 source-only control (three seeds), with and without AdaBN
    print('\nP_K vs S2 control:')
    ctl = {s: dict(no=read_log(c['no']), ada=read_log(c['ada'])) for s, c in CONTROL.items()}
    pk = {s: res[m42.tag('K', 'P', s, j)] for s, j in ((666, 32283), (667, 32284))}
    for key in ('bev07', 'bev05'):
        for mode in ('no', 'ada'):
            cv = [ctl[s][mode][key] if ctl[s][mode] else None for s in (666, 667, 668)]
            pv = [(pk[s][mode] or {}).get(key) for s in (666, 667)]
            cm = sum(cv) / 3 if None not in cv else None
            pm = sum(pv) / 2 if None not in pv else None
            print('%s %s: control %s mean %s | P_K %s mean %s | P_K - control %s' % (
                key, mode, ' / '.join(fmt(x) for x in cv), fmt(cm), ' / '.join(fmt(x) for x in pv), fmt(pm),
                fmt(pm - cm if pm is not None and cm is not None else None, True)))
    ap = [(res[m42.tag('A', 'L', s, j)]['ada'] or {}).get('bev07') for s, j in ((666, 32289), (667, 32290))]
    ap5 = [(res[m42.tag('A', 'L', s, j)]['ada'] or {}).get('bev05') for s, j in ((666, 32289), (667, 32290))]
    if None not in ap:
        sp = abs(ap[0] - ap[1])
        print("\n42a': L_K on its own-count KITTI val: BEV@0.7 %.2f / %.2f, spread %.2f -> %s; BEV@0.5 %s / %s" % (
            ap[0], ap[1], sp, 'L_K extrapolates as R1 / R3' if sp <= 3 else 'reported as is', fmt(ap5[0]), fmt(ap5[1])))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(rows=res, contrasts={'|'.join(k): v for k, v in cs.items()}, control=ctl, problems=problems),
                  open(args.json, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
