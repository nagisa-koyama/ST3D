"""Item 48 (experiments_md 20261003_05 §5 item 48, design fixed 2026-11-21 20:09 JST, commit 23a5224 of experiments_md, before any
number): where by range the KITTI structure term's AdaBN residual sits. CPU only.

    python3 analysis/item48_rings.py manifest fwd OUT.json   # item 42a AdaBN predictions + the same checkpoints without AdaBN
    python3 analysis/item48_rings.py manifest rev OUT.json   # item 47's 16 KITTI rows on S' val (none / target)
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --workers 4               # per-ring scoring (item 38's tool)
    python3 analysis/item48_rings.py read AP_fwd.json [AP_rev.json]                        # the fixed readings

Readings (verbatim rule of T05 item 48): per seed and ring b, forward s(b) = L(b) - P(b), r(b) = s(b) / L(b); reverse s(b) =
P(b) - L(b), r(b) = s(b) / P(b). Near = mean r over 0-10 and 10-20 m; far = mean r over 30-40 and 40-70.4 m. "Residual far": at
both seeds far >= 2 x max(near, 0) and mean s over the far rings >= 2 AP; "residual near": the mirror; otherwise "over the whole
range". A row whose pooled value differs from its logged value by more than 0.02 AP is not read (item 38's rule).

Tests: tests/test_item48_rings.py.
"""
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'analysis'))

ARMS = ('O', 'P', 'L', 'C1')
ITEM38_NAME = {'O': 'O', 'P': 'P_K', 'L': 'L_K', 'C1': 'R3'}  # C1_K = R3 (item 42a)
NEAR, FAR = (0, 1), (3, 4)  # ring indices of item 38's RINGS: 0-10, 10-20 | 30-40, 40-70.4 m


def manifest_fwd():
    import make_item42_configs as m42
    import item38_kitti_rings as i38
    from item42_collect import read_log
    rows = []
    for part, row, seed, job, epoch, cfg, _ in m42.ROWS:
        if part != 'K' or row not in ARMS:
            continue
        pkl = m42.result_pkl(part, row, seed, job, epoch, cfg)
        got = read_log((pkl.parent, 'log_adabn_*.txt', None))
        rows.append(dict(group='fwd', stats='target', row=row, seed=seed, job=job, result=str(pkl),
                         logged_bev=got['bev07'] if got else None,
                         status='ok' if pkl.exists() and got else 'missing'))
        src = [r for r in i38.ROWS if r[0] == 'swap' and r[1] == ITEM38_NAME[row] and r[2] == seed]
        assert len(src) == 1, (row, seed, src)
        r = i38.resolve(*src[0])
        if r['status'] == 'ok' and not Path(r['result']).exists():
            r['status'] = 'result.pkl missing'
        rows.append(dict(r, group='fwd', stats='none', row=row))
    return rows


def manifest_rev():
    import make_item47_configs as m47
    from item42_collect import read_log
    rows = []
    for part, row, seed, job, epoch, stats, cfg, _ in m47.ROWS:
        if part != 'K':
            continue
        pkl = m47.result_pkl(part, row, seed, job, epoch, stats, cfg)
        got = read_log((pkl.parent, 'log_adabn_*.txt', None))
        rows.append(dict(group='rev', stats=stats, row=row, seed=seed, job=job, result=str(pkl),
                         logged_bev=got['bev07'] if got else None,
                         status='ok' if pkl.exists() and got else 'missing'))
    return rows


def ring_values(ap_rows, group, stats, row, seed):
    hit = [r for r in ap_rows if (r['group'], r['stats'], r['row'], r['seed']) == (group, stats, row, seed)]
    if len(hit) != 1 or not hit[0].get('reproduced'):
        return None
    return [g['bev'] for g in hit[0]['rings']]


def per_seed(a, b):
    """(s per ring, r per ring) for the arm on its own lattice `a` against the other `b`."""
    s = [x - y for x, y in zip(a, b)]
    r = [si / x if x else None for si, x in zip(s, a)]
    return s, r


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def classify(seeds):
    """seeds: [(s per ring, r per ring)] for both seeds -> reading (T05 item 48)."""
    if len(seeds) != 2 or any(x is None for x in seeds):
        return 'not read'
    far_ok = near_ok = True
    for s, r in seeds:
        near, far = mean([r[i] for i in NEAR]), mean([r[i] for i in FAR])
        if near is None or far is None:
            return 'not read'
        far_ok &= far >= 2 * max(near, 0.0) and mean([s[i] for i in FAR]) >= 2.0
        near_ok &= near >= 2 * max(far, 0.0) and mean([s[i] for i in NEAR]) >= 2.0
    if far_ok:
        return 'residual far'
    if near_ok:
        return 'residual near'
    return 'over the whole range'


def reading(ap_rows, group, stats):
    own, other = ('L', 'P') if group == 'fwd' else ('P', 'L')
    seeds = []
    for seed in (666, 667):
        a, b = ring_values(ap_rows, group, stats, own, seed), ring_values(ap_rows, group, stats, other, seed)
        seeds.append(per_seed(a, b) if a is not None and b is not None else None)
    return seeds, classify(seeds)


def cmd_read(paths):
    ap_rows, rings = [], None
    for p in paths:
        d = json.load(open(p))
        ap_rows += d['rows']
        rings = d['rings']
    bad = [r for r in ap_rows if not r.get('reproduced')]
    for r in bad:
        print('NOT REPRODUCED (not read): %s %s %s s%d pooled %.4f logged %s' % (
            r['group'], r['stats'], r['row'], r['seed'], r['pooled']['bev'], r['logged_bev']))
    names = ['%g-%g' % tuple(x) for x in rings]
    print('rings', names)
    for r in sorted(ap_rows, key=lambda r: (r['group'], r['stats'], ARMS.index(r['row']), r['seed'])):
        print('%-3s %-6s %-2s s%d pooled %6.2f | %s | n_gt %s' % (
            r['group'], r['stats'], r['row'], r['seed'], r['pooled']['bev'],
            ' '.join('%6.2f' % g['bev'] if g['bev'] is not None else '     -' for g in r['rings']),
            '/'.join(str(g['n_gt']) for g in r['rings'])))
    out = {}
    for group in ('fwd', 'rev'):
        none_seeds = None
        for stats in ('none', 'target'):
            if not any(r['group'] == group and r['stats'] == stats for r in ap_rows):
                continue
            seeds, v = reading(ap_rows, group, stats)
            out[(group, stats)] = v
            print('\n%s %s (%s): %s%s' % (group, stats, 'L - P, / L' if group == 'fwd' else 'P - L, / P', v,
                                          '  <- the reading' if (group, stats) == ('fwd', 'target') else ''))
            for k, sd in enumerate(seeds):
                if sd is None:
                    print('  seed %d: missing' % (666, 667)[k])
                    continue
                s, r = sd
                print('  seed %d s %s | r %s | near r %.3f far r %.3f' % (
                    (666, 667)[k], ' '.join('%+6.2f' % x for x in s),
                    ' '.join('%+.3f' % x if x is not None else '-' for x in r),
                    mean([r[i] for i in NEAR]), mean([r[i] for i in FAR])))
            if stats == 'none':
                none_seeds = seeds
            elif none_seeds is not None:
                for k in range(2):
                    if seeds[k] is None or none_seeds[k] is None:
                        continue
                    share = ['%+.2f' % (1 - a / b) if b else '-' for a, b in zip(seeds[k][0], none_seeds[k][0])]
                    print('  seed %d share AdaBN removes per ring: %s' % ((666, 667)[k], ' '.join(share)))
    return out


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == 'manifest':
        rows = manifest_fwd() if sys.argv[2] == 'fwd' else manifest_rev()
        json.dump(rows, open(sys.argv[3], 'w'), indent=1)
        for r in rows:
            print(r['group'], r['stats'], r['row'], r['seed'], r['job'], r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if len(sys.argv) >= 3 and sys.argv[1] == 'read':
        cmd_read(sys.argv[2:])
        return
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
