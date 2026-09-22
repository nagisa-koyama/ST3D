"""Aggregate analysis/throughput_results/*.tsv into the three-lever conclusion table.

Usage: python analysis/summarize_throughput.py [results_dir]
"""
import glob
import re
import sys
from collections import defaultdict

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else 'analysis/throughput_results'


def parse_line(line):
    if not line.startswith('RESULT'):
        return None
    d = {}
    for tok in line.split()[1:]:
        k, v = tok.split('=', 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        d[k] = v
    return d


def main():
    rows = defaultdict(list)  # cfg -> list of dicts
    for f in sorted(glob.glob('%s/*.tsv' % RESULTS_DIR)):
        for line in open(f):
            d = parse_line(line.strip())
            if d:
                rows[d['cfg']].append(d)

    for cfg, combos in sorted(rows.items()):
        print('=== %s (%d combos) ===' % (cfg, len(combos)))
        print('%-4s %-8s %8s %10s %9s %8s/%-8s %7s %9s' % (
            'bs', 'workers', 'it/s', 'samples/s', 'data-wait', 'peak-alloc', 'peak-rsv',
            'hours', 'iters'))
        for r in sorted(combos, key=lambda r: (r['workers'], r['bs'])):
            print('%-4d %-8d %8.2f %10.1f %8.2f%% %8.1f/%-8.1f %7.1f %9d' % (
                r['bs'], r['workers'], r['it_s'], r['samples_s'], 100 * r['data_frac'],
                r['peak_alloc_gb'], r['peak_reserved_gb'], r['hours'], r['iters']))

        base = min(combos, key=lambda r: (r['bs'], r['workers']))
        best = max(combos, key=lambda r: r['samples_s'])
        # Effect of workers alone at the SMALLEST batch size (isolates the dataloader lever).
        bs0 = base['bs']
        by_workers = sorted([r for r in combos if r['bs'] == bs0], key=lambda r: r['workers'])
        # Effect of batch size alone at the SMALLEST worker count (isolates the batch lever).
        w0 = base['workers']
        by_bs = sorted([r for r in combos if r['workers'] == w0], key=lambda r: r['bs'])

        print('--- workers effect @ bs=%d ---' % bs0)
        for r in by_workers:
            print('  workers=%-3d samples/s=%7.1f (%.2fx base) data-wait=%.2f%%' % (
                r['workers'], r['samples_s'], r['samples_s'] / base['samples_s'], 100 * r['data_frac']))
        print('--- batch effect @ workers=%d ---' % w0)
        for r in by_bs:
            print('  bs=%-3d samples/s=%7.1f (%.2fx base) hours=%.1f' % (
                r['bs'], r['samples_s'], r['samples_s'] / base['samples_s'], r['hours']))

        max_ok_bs = max(r['bs'] for r in combos)
        print('largest batch that did NOT OOM: %d  (peak reserved %.1f GB)' % (
            max_ok_bs, max(r['peak_reserved_gb'] for r in combos if r['bs'] == max_ok_bs)))
        print('BEST single-GPU: bs=%d workers=%d -> %.1fx base, %.1f h, bound=%s' % (
            best['bs'], best['workers'], best['samples_s'] / base['samples_s'], best['hours'],
            'DATALOADER' if best['data_frac'] > 0.25 else 'GPU'))
        print()


if __name__ == '__main__':
    main()
