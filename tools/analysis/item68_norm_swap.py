"""Item 68 (experiments_md 20261003_05 §5 item 68, design fixed 2026-11-28 16:42 JST, experiments_md commit a7ad7e1, before any
number): LayerNorm / GroupNorm in place of every BatchNorm on nuScenes -> KITTI, against item 41a's BN rows. UDA-LEGAL rows;
this script reads KITTI val labels for evaluation only.

    python3 analysis/item68_norm_swap.py manifest KITTI.json SOURCE.json   # resolve every row from its job's log
    python3 analysis/item38_kitti_rings.py ap KITTI.json AP.json --real-kernel --workers 4
    python3 analysis/item68_norm_swap.py read AP.json SOURCE.json

Rows: LN / GN at seeds 666-668 (KITTI val epoch 20, the twin config), BN_none / BN_ada (item 41a, epoch 20, without and
with AdaBN), and each training row's in-run nuScenes val (SOURCE.json, not ring-scored).
Readings (T05 item 68), per arm v: d = mean(v) - reference mean, with the references fixed in the design (BN 38.15, BN +
AdaBN 65.82): "v beats BN" (d >= 5 and every seed above), "level" (|d| < 3), "v is below BN" (d <= -5 and every seed
below), otherwise "no verdict"; the source flag if mean nuScenes val < 44.94 - 3. Nothing is read unless every scored row
reproduces its logged value and the BN rows reproduce the design's numbers within 0.05.
Tests: tests/test_item68_norm_swap.py.
"""
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))

SEEDS = (666, 667, 668)
ARMS = ('LN', 'GN')
# (arm, seed) -> (training job, evaluation job)
JOBS = {('LN', 666): (37409, 37410), ('LN', 667): (37411, 37412), ('LN', 668): (37413, 37414),
        ('GN', 666): (37416, 37417), ('GN', 667): (37418, 37419), ('GN', 668): (37420, 37421)}
BN = {666: (32373, 32374), 667: (32411, 32413), 668: (32434, 32435)}          # item 41a: row, evaluations
REF = {'BN_none': (35.48, 34.86, 44.12), 'BN_ada': (63.00, 66.73, 67.73)}     # the design's numbers, seeds 666-668
REF_MEAN = {'BN_none': 38.15, 'BN_ada': 65.82}
SOURCE_MEAN, SOURCE_MARGIN = 44.94, 3.0
REF_TOL = 0.05
WORDS = ('%s beats BN', 'level', '%s is below BN', 'no verdict')
FAR = (2, 3)
OUT = Path('/home/koyama/data/item68')


def kitti_specs():
    """(group, row, seed, job, log substring) for item38_kitti_rings.resolve."""
    out = []
    for (v, s), (_, e) in sorted(JOBS.items()):
        out.append(('item68', v, s, e, 'i68_kitti_%s_s%d' % (v.lower(), s)))
    for s, (_, e) in sorted(BN.items()):
        out.append(('item68', 'BN_none', s, e, 'i41_kitti_1gpu_s%d_e20' % s))
        out.append(('item68', 'BN_ada', s, e, 'i41_adabn_1gpu_s%d' % s))
    return out


def source_specs():
    out = [('item68', v + '_src', s, t, None) for (v, s), (t, _) in sorted(JOBS.items())]
    return out + [('item68', 'BN_src', s, t, None) for s, (t, _) in sorted(BN.items())]


def manifest():
    import item38_kitti_rings as m38
    kitti = [m38.resolve(*r) for r in kitti_specs()]
    for r in kitti:
        if r['status'] == 'ok' and not Path(r['result']).exists():
            r['status'] = 'result.pkl missing'
    return kitti, [m38.resolve(*r) for r in source_specs()]


def _vals(rows, name, key=lambda r: r['pooled']['bev']):
    got = {r['seed']: r for r in rows if r['row'] == name}
    return [key(got[s]) if s in got else None for s in SEEDS]


def word(vals, ref, arm):
    m = sum(vals) / len(vals)
    d = m - ref
    if d >= 5.0 and all(x > ref for x in vals):
        return d, WORDS[0] % arm
    if abs(d) < 3.0:
        return d, WORDS[1]
    if d <= -5.0 and all(x < ref for x in vals):
        return d, WORDS[2] % arm
    return d, WORDS[3]


def checks(rows):
    bad = [(r['row'], r['seed']) for r in rows if not r.get('reproduced')]
    ref_bad = []
    for name, want in REF.items():
        for s, w, got in zip(SEEDS, want, _vals(rows, name)):
            if got is None or abs(got - w) > REF_TOL:
                ref_bad.append((name, s, got, w))
    return bad, ref_bad


def readings(rows, source):
    bad, ref_bad = checks(rows)
    out = {'not_reproduced': bad, 'references_off': ref_bad, 'arms': {}}
    for v in ARMS:
        vals = _vals(rows, v)
        src = [r.get('logged_bev') for s in SEEDS for r in source if r['row'] == v + '_src' and r['seed'] == s]
        a = {'kitti': vals, 'source': src}
        if bad or ref_bad or any(x is None for x in vals):
            a.update(H_none='not read', H_ada='not read')
        else:
            a['d_none'], a['H_none'] = word(vals, REF_MEAN['BN_none'], v)
            a['d_ada'], a['H_ada'] = word(vals, REF_MEAN['BN_ada'], v)
        a['source_flag'] = (len(src) == 3 and all(x is not None for x in src)
                            and sum(src) / 3 < SOURCE_MEAN - SOURCE_MARGIN)
        out['arms'][v] = a
    return out


def cmd_read(ap_path, source_path):
    rows = json.load(open(ap_path))['rows']
    source = json.load(open(source_path))
    for r in rows:
        print('%-8s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    for r in source:
        print('%-8s s%d nuScenes val (in-run, own input) %s  job %s  W&B %s' % (r['row'], r['seed'], r.get('logged_bev'),
                                                                               r['job'], r.get('wandb')))
    out = readings(rows, source)
    if out['not_reproduced'] or out['references_off']:
        print('\n== NOT READ: not reproduced %s; references off %s' % (out['not_reproduced'], out['references_off']))
    for v, a in out['arms'].items():
        print('\n== %s: KITTI %s, nuScenes val %s%s' % (v, a['kitti'], a['source'],
                                                         '  [the swap costs the source model]' if a['source_flag'] else ''))
        print('  H_none (against BN %.2f): %s (d %s)' % (REF_MEAN['BN_none'], a['H_none'], a.get('d_none')))
        print('  H_ada  (against BN + AdaBN %.2f): %s (d %s)' % (REF_MEAN['BN_ada'], a['H_ada'], a.get('d_ada')))
    print('\n-- descriptive: pooled 3D, far BEV (20-40 m) and the three-seed spread')
    for name in ARMS + ('BN_none', 'BN_ada'):
        bev = _vals(rows, name)
        far = _vals(rows, name, key=lambda r: sum(r['rings'][i]['bev'] for i in FAR) / len(FAR))
        d3 = _vals(rows, name, key=lambda r: r['pooled']['3d'])
        if any(x is None for x in bev):
            print('  %-8s -' % name)
            continue
        print('  %-8s BEV %s (spread %.2f) | 3D %s | far %s (mean %.2f)' % (
            name, ' / '.join('%.2f' % x for x in bev), max(bev) - min(bev), ' / '.join('%.2f' % x for x in d3),
            ' / '.join('%.2f' % x for x in far), sum(far) / 3))
    json.dump(out, open(Path(ap_path).with_name('item68_readings.json'), 'w'), indent=1, default=float)


def main():
    a = sys.argv[1:]
    if a[:1] == ['manifest'] and len(a) == 3:
        kitti, source = manifest()
        json.dump(kitti, open(a[1], 'w'), indent=1)
        json.dump(source, open(a[2], 'w'), indent=1)
        for r in kitti + source:
            print(r['row'], r['seed'], r['job'], r['status'], r.get('wandb'), r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in kitti) else 1)
    if a[:1] == ['read'] and len(a) == 3:
        return cmd_read(a[1], a[2])
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
