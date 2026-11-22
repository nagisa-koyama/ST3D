"""Item 53 D2 read (experiments_md 20261003_05 §5; ANALYSIS): Car AP of IA-SSD A1 with BN statistics re-estimated per layer
group, against none, by the fixed rule: BEV@0.7 moderate >= 0.5 x none at both seeds "collapse gone", < 0.5 x none at both
"collapse remains", otherwise "mixed". 3D@0.7 and BEV@0.5 are descriptive. Full AdaBN under the new tag must reproduce
E1c's (b2) value within 0.5 (procedure check).

    python3 analysis/item53_d2_read.py [--json OUT]
"""
import argparse
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'output/da-ieee-access-pointrcnn/iassd-sourceonly-nuscenes2kitti-fovtar'
GROUPS = ('bb', 'head', 'nobox', 'conf', 'noconf', 'novote')


def car_ap(text):
    """(BEV@0.7, 3D@0.7, BEV@0.5) moderate R40 from an eval log: the LAST block of each kind."""
    def block(head):
        hits = re.findall(re.escape(head) + r'\s*\nbbox AP:[^\n]*\nbev  AP:([\d.]+), ([\d.]+), ([\d.]+)\n3d   AP:([\d.]+), '
                          r'([\d.]+), ([\d.]+)', text)
        assert hits, 'no %s block' % head
        return hits[-1]
    a, b = block('Car AP_R40@0.70, 0.70, 0.70:'), block('Car AP_R40@0.70, 0.50, 0.50:')
    return float(a[1]), float(a[4]), float(b[1])


def read_dir(d):
    logs = sorted(glob.glob(str(d / 'log_adabn_*.txt')))
    assert logs, 'no log in %s' % d
    return car_ap(Path(logs[-1]).read_text())


def reading(ratios):
    if all(r >= 0.5 for r in ratios):
        return 'collapse gone'
    if all(r < 0.5 for r in ratios):
        return 'collapse remains'
    return 'mixed'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--json', default=None)
    p.add_argument('--root', default=str(ROOT))
    args = p.parse_args()
    root = Path(args.root)
    out = {'seeds': {}, 'readings': {}}
    for s in (1, 2):
        ep = 'eval/epoch_20'
        r = {'none': read_dir(root / ('b2_s%d' % s) / ep / 'stats_none_mix1_n1000'),
             'full_b2': read_dir(root / ('b2_s%d' % s) / ep / 'stats_target_mix1_n1000'),
             'full': read_dir(root / ('i53_s%d' % s) / ep / 'stats_target_mix1_n1000')}
        for g in GROUPS:
            r[g] = read_dir(root / ('i53_s%d' % s) / ep / ('stats_target_mix1_n1000_layers-%s' % g))
        r['procedure_check'] = abs(r['full'][0] - r['full_b2'][0]) <= 0.5
        out['seeds'][s] = r
    for g in GROUPS + ('full',):
        ratios = [out['seeds'][s][g][0] / out['seeds'][s]['none'][0] for s in (1, 2)]
        out['readings'][g] = {'ratio_to_none': ratios, 'reading': reading(ratios)}
    print('%-8s %s' % ('group', '   '.join('seed %d BEV0.7 / 3D0.7 / BEV0.5 (x none)' % s for s in (1, 2))))
    for g in ('none', 'full_b2', 'full') + GROUPS:
        cells = []
        for s in (1, 2):
            v = out['seeds'][s][g]
            cells.append('%6.2f / %6.2f / %6.2f (%.2f)' % (v + (v[0] / out['seeds'][s]['none'][0],)))
        print('%-8s %s   %s' % (g, '   '.join(cells), out['readings'].get(g, {}).get('reading', '')))
    print('procedure check (full within 0.5 of b2):', [out['seeds'][s]['procedure_check'] for s in (1, 2)])
    if args.json:
        json.dump(out, open(args.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
