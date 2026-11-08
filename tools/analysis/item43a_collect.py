"""Collect the item 43a rows (experiments_md 20261003_05 §5 item 43; scripts/_item43a_evals_20261109.sh) from the eval jobs'
stderr logs: per row Car AP_R40 at IoU 0.7 (BEV / 3D) and at IoU 0.5 (BEV / 3D), all objects, and the W&B run; then the
contributions and readings fixed in the design. ANALYSIS; reads logs only.

    python analysis/item43a_collect.py logs/error_<job>_<name>.txt [...]

Readings (T05 §5 item 43, verbatim): contribution = O_N - row, paired by seed for KITTI's depth (27468 with 25743, 27804 with
31818); Waymo's depth against 25743. A contribution is read if >= 3 at one seed or >= 2 as a two-seed mean with the same
sign. Own-input check: the accumulated model on its own input at or above O_N's own-input level (O_N on single-sweep val,
the same seed's), else "trained but not transferable".
"""
import re
import sys

BANKED = {'s1_25743': 44.9825, 's1_31818': 44.9091}      # 25768 / 31818 epoch 20, as logged


def rows(paths):
    out = {}
    for p in paths:
        text = open(p, errors='replace').read().replace('\r', '\n')
        for m in re.finditer(r'=== ROW (\S+) job (\d+).*?===\n(.*?)=== END \1 exit (\d+)', text, re.S):
            tag, job, body, code = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            r = dict(job=job, exit=code, log=p)
            a = re.search(r'Car AP_R40@0\.70, 0\.70, 0\.70:\nbbox AP:[^\n]*\nbev  AP:([\d.]+)[^\n]*\n3d   AP:([\d.]+)', body)
            b = re.search(r'Car AP_R40@0\.70, 0\.50, 0\.50:\nbbox AP:[^\n]*\nbev  AP:([\d.]+)[^\n]*\n3d   AP:([\d.]+)', body)
            w = re.search(r'wandb\.ai/nagisa/st3d/runs/([a-z0-9]+)', body)
            if a:
                r.update(bev=float(a.group(1)), d3=float(a.group(2)))
            if b:
                r.update(bev05=float(b.group(1)), d305=float(b.group(2)))
            r['wandb'] = w.group(1) if w else None
            out[tag] = r
    return out


def fmt(r, k):
    return '%.2f' % r[k] if r and k in r else 'n/a'


def main(paths):
    R = rows(paths)
    print('| row | job | Car BEV / 3D @0.7 | BEV / 3D @0.5 | W&B |')
    print('|---|---|---|---|---|')
    for tag, r in R.items():
        print('| %s | %d | %s / %s | %s / %s | %s |' % (tag, r['job'], fmt(r, 'bev'), fmt(r, 'd3'), fmt(r, 'bev05'),
                                                     fmt(r, 'd305'), r['wandb']))
    for tag, v in BANKED.items():
        if tag in R and 'bev' in R[tag]:
            print('rescore %s: %.2f against banked %.2f (diff %+.2f)' % (tag, R[tag]['bev'], v, R[tag]['bev'] - v))
    on = {666: R.get('s1_25743', {}).get('bev', BANKED['s1_25743']), 667: R.get('s1_31818', {}).get('bev', BANKED['s1_31818'])}
    pairs = [('KITTI depth', 666, 's1_27468', 'own_27468'), ('KITTI depth', 667, 's1_27804', 'own_27804'),
             ('Waymo depth 30/20', 666, 's1_27185', 'own_27185'), ('A4 depth 22/17 (beside)', 666, 's1_29354', 'own_29354')]
    terms = {}
    for name, seed, s1, own in pairs:
        if s1 not in R or 'bev' not in R[s1]:
            continue
        c = on[seed] - R[s1]['bev']
        terms.setdefault(name, []).append(c)
        flag = ''
        if own in R and 'bev' in R[own]:
            flag = 'own input %.2f vs O_N %.2f -> %s' % (R[own]['bev'], on[seed], 'transferable-level' if R[own]['bev'] >= on[seed]
                                                         else 'TRAINED BUT NOT TRANSFERABLE')
        print('%s seed %d: O_N %.2f - %s %.2f = %+.2f; %s' % (name, seed, on[seed], s1, R[s1]['bev'], c, flag))
    for name, cs in terms.items():
        mean = sum(cs) / len(cs)
        same_sign = all(x > 0 for x in cs) or all(x < 0 for x in cs)
        one = [abs(x) >= 3 for x in cs]
        two = len(cs) >= 2 and abs(mean) >= 2 and same_sign
        print('%s: per seed %s, mean %+.2f; >= 3 at one seed: %s; two-seed mean >= 2 with the same sign: %s' % (
            name, ['%+.2f' % x for x in cs], mean, one, two))


if __name__ == '__main__':
    main(sys.argv[1:])
