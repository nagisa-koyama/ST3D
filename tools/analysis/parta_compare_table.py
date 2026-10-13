"""Part A table for an IA-SSD oracle against a CenterPoint oracle (experiments_md 20261011_07 §16 / §21.1 / §22.1): the 11
summary box errors as change in BEV / 3D at each IoU bar, and the checks the pre-declarations name (rank agreement, absolute
3D agreement, BEV blindness to z, shrink vs enlarge, the loose-bar residue). Reads the jsonl of
`oracle_box_error_multiiou.py` / `oracle_box_error_sensitivity.py`; no model, no data.

    python analysis/parta_compare_table.py --ia logs/box_error/X_car.jsonl --cp logs/box_error/Y.jsonl --cls Car
"""
import argparse
import json

CAR_ARMS = [('size x0.90', 'sz_iso0.90'), ('size x1.10', 'sz_iso1.10'), ('length x1.10', 'sz_l1.10'),
            ('height x1.10', 'sz_h1.10'), ('size noise 10%', 'szn0.10'), ('heading +10 deg', 'yaw_b10'),
            ('heading noise 10 deg', 'yaw_n10'), ('0.5 m along', 'lon0.5'), ('0.2 m across', 'lat0.2'),
            ('0.2 m vertical', 'z0.2'), ('position noise sigma 0.2 m (xy)', 'pxy0.2')]
PED_ARMS = [('size x0.90', 'sz_iso0.90'), ('size x1.10', 'sz_iso1.10'), ('0.2 m along', 'lon0.2'),
            ('0.2 m across', 'lat0.2'), ('0.5 m across', 'lat0.5'), ('0.2 m vertical', 'z0.2'),
            ('position noise sigma 0.2 m', 'pxy0.2')]
Z_ONLY = ('z0.1', 'z0.2', 'z0.5', 'pz0.1', 'pz0.2', 'sz_h0.90', 'sz_h1.10')
SIZE_HEADING_VERTICAL = ('sz_', 'szn', 'yaw_', 'z0', 'pz')


def load(path):
    return {r['arm']: r for r in map(json.loads, open(path))}


def val(r, m, t):
    """AP from either tool's record: the multi-IoU tool's '<m>_moderate@<t>', or the single-IoU tool's (CenterPoint Car,
    IoU 0.7 only) '<m>_rule_A' (PandaSet), '<m>_moderate' (KITTI) or '<m>' (Waymo, nuScenes)."""
    for k in (f'{m}_moderate@{t}', f'{m}_rule_A', f'{m}_moderate', m):
        if k in r and (k.endswith(f'@{t}') or abs(t - 0.7) < 1e-9):
            return r[k]
    raise KeyError(f'{m} @{t}')


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ia', required=True)
    ap.add_argument('--cp', required=True)
    ap.add_argument('--cls', default='Car', choices=['Car', 'Pedestrian'])
    args = ap.parse_args()
    ia, cp = load(args.ia), load(args.cp)
    arms = CAR_ARMS if args.cls == 'Car' else PED_ARMS
    bar = 0.7 if args.cls == 'Car' else 0.5
    ths = sorted({float(k.split('@')[1]) for k in ia['base'] if k.startswith('bev_moderate@')}, reverse=True)
    d = lambda rows, a, m, t: val(rows[a], m, t) - val(rows['base'], m, t)
    print('base IA-SSD ' + ' | '.join(f"@{t}: {ia['base'][f'bev_moderate@{t}']:.2f} / {ia['base'][f'3d_moderate@{t}']:.2f}" for t in ths))
    print(f"base CenterPoint @{bar}: {val(cp['base'], 'bev', bar):.2f} / {val(cp['base'], '3d', bar):.2f}")
    print(f'| error | CenterPoint @{bar} | ' + ' | '.join(f'IA-SSD @{t}' for t in ths) + ' |')
    for name, a in arms:
        cells = [f'{d(cp, a, "bev", bar):+.1f} / {d(cp, a, "3d", bar):+.1f}'] + \
                [f'{d(ia, a, "bev", t):+.1f} / {d(ia, a, "3d", t):+.1f}' for t in ths]
        print(f'| {name} | ' + ' | '.join(cells) + ' |')
    keys = [a for _, a in arms]
    for m in ('3d', 'bev'):
        x = [d(ia, a, m, bar) for a in keys]
        y = [d(cp, a, m, bar) for a in keys]
        diffs = [abs(p - q) for p, q in zip(x, y)]
        print(f'{m}: Spearman {spearman(x, y):.3f}; |IA − CP| mean {sum(diffs) / len(diffs):.1f}, '
              f'<= 5 on {sum(v <= 5 for v in diffs)} of {len(diffs)}; per arm ' + ', '.join(f'{v:.1f}' for v in diffs))
    for name, a in arms:
        if a in ('z0.2', 'pz0.2', 'lat0.2', 'lat0.5', 'lon0.2', 'pxy0.2'):
            b = val(ia['base'], 'bev', bar)
            print(f'share of base BEV @{bar}, {name}: IA-SSD {-d(ia, a, "bev", bar) / b * 100:.0f}%, '
                  f'CenterPoint {-d(cp, a, "bev", bar) / val(cp["base"], "bev", bar) * 100:.0f}%')
    zmax = max(abs(d(ia, a, 'bev', t)) for a in Z_ONLY if a in ia for t in ths)
    print(f'max |ΔBEV| over z / height-only arms, every bar: {zmax:.4f}')
    print(f'shrink vs enlarge, Δ3D @{bar}: x0.90 {d(ia, "sz_iso0.90", "3d", bar):+.1f}, x1.10 {d(ia, "sz_iso1.10", "3d", bar):+.1f}')
    loose = [t for t in ths if abs(t - 0.3) < 1e-9 or (args.cls == 'Pedestrian' and abs(t - 0.25) < 1e-9)]
    for t in loose:
        shv = [a for a in ia if a.startswith(SIZE_HEADING_VERTICAL)]
        worst = sorted(((min(d(ia, a, 'bev', t), d(ia, a, '3d', t)), a) for a in shv))[:4]
        summ = [a for a in keys if a.startswith(SIZE_HEADING_VERTICAL)]
        wsum = min(min(d(ia, a, 'bev', t), d(ia, a, '3d', t)) for a in summ)
        print(f'@{t}: worst size / heading / vertical arm among the summary arms {wsum:+.2f}; over all such arms '
              + ', '.join(f'{a} {v:+.2f}' for v, a in worst))


if __name__ == '__main__':
    main()
