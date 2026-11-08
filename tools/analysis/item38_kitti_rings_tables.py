"""Markdown tables of experiments_md 20261003_05 §5 item 38 from item38_kitti_rings.py's JSON.

    python item38_kitti_rings_tables.py RINGS_AP.json DENSITY.json > tables.md
"""
import json
import sys
from collections import defaultdict

import numpy as np

W = 'https://wandb.ai/nagisa/st3d/runs/'


def fmt(v, d=1):
    return 'n/a' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{d}f}'


def main():
    ap, den = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    rings = ['%g-%g' % (lo, hi) for lo, hi in ap['rings']]
    rows = ap['rows']

    print('### Validation: pooled [0, inf) through the ring path against the logged value\n')
    print('| group | row | seed | job | W&B run | logged BEV mod | pooled here | difference | to the digit | read (<= 0.02) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['group']} | {r['row']} | {r['seed']} | {r['job']} | [{r['wandb']}]({W}{r['wandb']}) | "
              f"{r['logged_bev']:.4f} | {r['pooled']['bev']:.4f} | {r['diff']:+.4f} | "
              f"{'yes' if r['reproduced_exact'] else 'no'} | {'yes' if r['reproduced'] else '**NO**'} |")

    print('\n### Per-ring Car BEV AP, every run (IoU 0.7; IoU 0.5 in brackets)\n')
    print('| group | row | seed | W&B run | ' + ' | '.join(rings) + ' |')
    print('|---|---|---|---|' + '---|' * len(rings))
    for r in rows:
        cells = [f"{fmt(g['bev'])} ({fmt(g['bev05'])})" for g in r['rings']]
        print(f"| {r['group']} | {r['row']} | {r['seed']} | [{r['wandb']}]({W}{r['wandb']}) | " + ' | '.join(cells) + ' |')

    print('\n### Per-ring mean and spread (max - min) over seeds, and loss against the oracle mean\n')
    groups = defaultdict(list)
    for r in rows:
        if r['reproduced']:
            groups[(r['group'], r['row'])].append(r)
    o = np.array([[g['bev'] for g in r['rings']] for r in groups[('swap', 'O')]])
    o_mean = o.mean(0)
    print('| group | row | seeds | ' + ' | '.join(f'{x} mean (spread) / loss' for x in rings) + ' | pooled mean |')
    print('|---|---|---|' + '---|' * len(rings) + '---|')
    for (grp, row), runs in groups.items():
        c = np.array([[np.nan if g['bev'] is None else g['bev'] for g in r['rings']] for r in runs])
        m, sp = np.nanmean(c, 0), np.nanmax(c, 0) - np.nanmin(c, 0)
        pooled = np.mean([r['pooled']['bev'] for r in runs])
        cells = [f"{fmt(mi)} ({fmt(si) if len(runs) > 1 else '-'}) / {fmt(oi - mi) if grp == 'swap' or grp == 'transfer' else '-'}"
                 for mi, si, oi in zip(m, sp, o_mean)]
        print(f"| {grp} | {row} | {'/'.join(str(r['seed']) for r in runs)} | " + ' | '.join(cells) + f' | {pooled:.2f} |')

    print('\n### GT per ring (moderate-valid Car, KITTI val)\n')
    print('| ' + ' | '.join(rings) + ' |'); print('|' + '---|' * len(rings))
    print('| ' + ' | '.join(str(g['n_gt']) for g in rows[0]['rings']) + ' |')

    print('\n### Points per moderate-valid Car box per ring (median; mean; share of boxes with no point)\n')
    print('| cloud | ' + ' | '.join(rings) + ' |'); print('|---|' + '---|' * len(rings))
    full = den['table']['full KITTI']
    for name, t in den['table'].items():
        cells = [f"{x['median']:.0f} ({x['median'] / f['median']:.2f}x); {x['mean']:.0f}; {100 * x['empty_share']:.0f}%"
                 for x, f in zip(t, full)]
        print(f'| {name} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
