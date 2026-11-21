"""Item 52 (experiments_md 20261003_05 §5 item 52; handover from the paper session 2026-11-22, the user's question 「UADA は
feature representation の汎化を狙っているが、BN 統計の後に残った重みの誤差に効くのか」): UADA3D against source-only at
half target statistics on nuScenes->KITTI, by range. Analysis rows; existing predictions only (no training).

    python3 analysis/item52_uada_rings.py manifest OUT.json
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --workers 4 --real-kernel   # GPU node, singularity --nv
    python3 analysis/item52_uada_rings.py read AP.json

Rows (adabn_eval outputs of 2026-10-04, one seed; the checkpoint of each directory is checked against its log):
  U     UADA3D 27174 (hw280h27) epoch 20, its own statistics (stats none)          logged 66.63
  M     source-only 25743 (wfdcq75s) epoch 20, target statistics mixed 0.5          logged 69.86
  S0    25743, stats none                                                           logged 41.57
  S1    25743, full target statistics                                               logged 65.26
  Usrc  27174, statistics re-estimated on the source                                logged 39.32
  Utgt  27174, full target statistics                                               logged  9.56

Readings (fixed in T05 item 52 before any per-ring number): d(b) = U(b) - M(b), Car BEV AP_R40 at IoU 0.7, KITTI moderate,
camera-FOV filter, rings 0-10 / 10-20 / 20-30 / 30-40 / 40-70.4 m by GT box centre (item 38's ring view).
  primary  D_res = mean d over 20-30 and 30-40 m (where item 48 put the forward AdaBN residual): >= +3 "UADA3D ahead where
           the residual sits", <= -3 "UADA3D behind there", otherwise "level there".
  also     near = mean d over 0-10 and 10-20 m, far = mean d over 30-40 and 40-70.4 m, each by the same +-3 rule.
  A row whose pooled value differs from its logged value by more than 0.02 is not read (item 38's rule); one seed.
Tests: tests/test_item52_uada_rings.py.
"""
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'analysis'))

OUT = TOOLS.parent / 'output' / 'da-ieee-access'
ROWS = [  # name, job, W&B training run of the checkpoint, directory
    ('U', 27174, 'hw280h27', 'centerpoint-uada3d-nuscenes2kitti/adabn/eval/epoch_20/stats_none_mix1_n1000'),
    ('M', 25743, 'wfdcq75s', 'centerpoint-sourceonly-nuscenes2kitti/adabn/eval/epoch_20/stats_target_mix0.5_n1000'),
    ('S0', 25743, 'wfdcq75s', 'centerpoint-sourceonly-nuscenes2kitti/adabn/eval/epoch_20/stats_none_mix1_n1000'),
    ('S1', 25743, 'wfdcq75s', 'centerpoint-sourceonly-nuscenes2kitti/adabn/eval/epoch_20/stats_target_mix1_n1000'),
    ('Usrc', 27174, 'hw280h27', 'centerpoint-uada3d-nuscenes2kitti/adabn/eval/epoch_20/stats_source_mix1_n1000'),
    ('Utgt', 27174, 'hw280h27', 'centerpoint-uada3d-nuscenes2kitti/adabn/eval/epoch_20/stats_target_mix1_n1000'),
]
LOGGED = {'U': 66.63, 'M': 69.86, 'S0': 41.57, 'S1': 65.26, 'Usrc': 39.32, 'Utgt': 9.56}  # as quoted (20261004_01)
NEAR, RES, FAR = (0, 1), (2, 3), (3, 4)  # ring indices: 0-10, 10-20 | 20-30, 30-40 | 30-40, 40-70.4 m
BAR = 3.0


def ckpt_of_log(log):
    """The checkpoint line of an adabn_eval log ('ckpt   /storage/wandb/run-<ts>-<id>/files/ckpt/checkpoint_epoch_N.pth')."""
    for line in open(log, errors='replace').read().replace('\r', '\n').splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-2] == 'ckpt' and parts[-1].endswith('.pth'):
            return parts[-1]
    return None


def manifest():
    from item42_collect import read_log
    rows = []
    for name, job, run, rel in ROWS:
        d = OUT / rel
        logs = sorted(d.glob('log_adabn_*.txt'))
        got = read_log((d, 'log_adabn_*.txt', None))
        ck = ckpt_of_log(logs[-1]) if logs else None
        status = 'ok'
        if len(logs) != 1:
            status = 'expected one log, found %d' % len(logs)
        elif not (d / 'result.pkl').exists():
            status = 'result.pkl missing'
        elif ck is None or ('-%s/' % run) not in ck or not ck.endswith('checkpoint_epoch_20.pth'):
            status = 'checkpoint %s is not %s epoch 20' % (ck, run)
        elif got is None or abs(got['bev07'] - LOGGED[name]) > 0.005:
            status = 'logged %s, quoted %.2f' % (None if got is None else got['bev07'], LOGGED[name])
        rows.append(dict(group='i52', row=name, seed=666, job=job, wandb=run, ckpt=ck, result=str(d / 'result.pkl'),
                         logged_bev=got['bev07'] if got else None, status=status))
    return rows


def mean(xs):
    return sum(xs) / len(xs)


def classify(x, where):
    if x >= BAR:
        return 'UADA3D ahead %s' % where
    if x <= -BAR:
        return 'UADA3D behind %s' % where
    return 'level %s' % where


def readings(ap):
    """{'d': per-ring U - M, 'res' / 'near' / 'far': (mean, reading)}, or None when U or M is not reproduced."""
    by = {r['row']: r for r in ap['rows']}
    if any(k not in by or not by[k]['reproduced'] for k in ('U', 'M')):
        return None
    d = [u['bev'] - m['bev'] for u, m in zip(by['U']['rings'], by['M']['rings'])]
    res, near, far = mean([d[i] for i in RES]), mean([d[i] for i in NEAR]), mean([d[i] for i in FAR])
    return dict(d=d, res=(res, classify(res, 'where the residual sits')), near=(near, classify(near, 'near')),
                far=(far, classify(far, 'far')))


def cmd_read(path):
    ap = json.load(open(path))
    print('kernel', ap['kernel'], 'rings', ap['rings'])
    for r in ap['rows']:
        print('%-5s pooled %6.2f logged %s %s | %s | n_gt %s' % (
            r['row'], r['pooled']['bev'], r['logged_bev'], 'ok' if r['reproduced'] else 'NOT REPRODUCED',
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '/'.join(str(g['n_gt']) for g in r['rings'])))
    by = {r['row']: r for r in ap['rows']}
    for a, b in (('U', 'S0'), ('M', 'S0'), ('S1', 'S0'), ('U', 'Usrc')):
        if a in by and b in by:
            print('%-4s - %-4s %s' % (a, b, ' '.join('%+6.2f' % (x['bev'] - y['bev']) for x, y in zip(by[a]['rings'], by[b]['rings']))))
    rd = readings(ap)
    if rd is None:
        print('not read: U or M missing or not reproduced')
        return None
    print('d = U - M  %s' % ' '.join('%+6.2f' % x for x in rd['d']))
    for k in ('res', 'near', 'far'):
        print('%-4s %+6.2f -> %s' % (k, rd[k][0], rd[k][1]))
    return rd


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'manifest' and len(sys.argv) == 3:
        rows = manifest()
        for r in rows:
            print(r['row'], r['job'], r['status'], r['logged_bev'], r['ckpt'])
        json.dump(rows, open(sys.argv[2], 'w'), indent=1)
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    elif cmd == 'read' and len(sys.argv) == 3:
        cmd_read(sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
