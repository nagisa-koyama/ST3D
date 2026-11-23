"""The S3 readings of experiments_md episodes/20261116_01 §8.10, from the saved confusion matrices.

Every number is cone point mIoU over the declared 16 classes (the 17 PandaSet classes minus barrier, which has no
ground truth in the cone) at epoch 391, from the `cone` matrix that tools/test_seg.py writes (shard_*.json, summed over
shards) or that train_seg.py writes in-run (eval_epoch_391.json). Rows are ground truth, columns predictions
(pcseg.losses_metrics.ConfusionMatrix).

  U2  the oracles per seed (each on its own val);
  U3  each rescore's diagonal against the run's in-run evaluation (points equal; |dmIoU| within T3's bound);
  U4  the P1 gaps per seed, gap = oracle_target - source model on target val, mean and sample SD per direction;
  terms (when the arm outputs exist): term_a1 / a2 / a3 / a4, term_b1, the render cost, b2's correction, the reliance
      forms, and shares (term mean / gap mean of its direction, a ratio of means; terms are never summed).
      A term is READABLE when |mean| >= 2.0 and all three seed-paired values carry its sign.

Nothing here chooses anything: every row is ANALYSIS (target labels).

    cd <worktree>/tools && python3 analysis/pcseg_s3_readings.py --out /home/koyama/data/pcseg/s3_readings/<name>.json

Missing inputs are reported as missing, never filled in.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '../..'))

ROOT = '/home/koyama/data/pcseg/test_seg'
SEEDS = (666, 667, 668)
DROPPED = ('barrier',)
T3_BOUND = 1e-6                      # T3 measured |dmIoU| 1.1e-7 / 3.9e-7 across GPU generations
READABLE = 2.0
VAL = {'64': 'spunet-pandaset64', 'gt': 'spunet-pandasetgt'}


# --- matrices -----------------------------------------------------------------------------------------------------------

def class_names(cfg_file):
    import train_seg
    from pcseg.class_map import build_class_map
    return list(build_class_map(train_seg.load_cfg(cfg_file).DATA.CLASS_MAP).names)


def cone16(mat, names, dropped=DROPPED):
    """Mean IoU over every class but `dropped`. Each kept class must have ground truth (else ValueError): the declared
    set is fixed, so a class without ground truth is a fault to look at, not a class to skip silently."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (len(names), len(names)):
        raise ValueError('matrix %s for %d classes' % (mat.shape, len(names)))
    keep = [i for i, n in enumerate(names) if n not in dropped]
    if len(keep) != len(names) - len(dropped):
        raise ValueError('dropped classes %s not all in %s' % (dropped, names))
    gt = mat.sum(1)
    missing = [names[i] for i in keep if gt[i] == 0]
    if missing:
        raise ValueError('declared classes without ground truth: %s' % missing)
    tp = np.diag(mat)
    iou = tp / (mat.sum(0) + gt - tp)
    return float(np.mean(iou[keep]))


def load_test_seg(d):
    """Summed cone matrix and metadata of a test_seg output directory; None if absent or incomplete."""
    files = sorted(glob.glob(os.path.join(d, 'shard_*_of_*.json')))
    if not files:
        return None
    parts = [json.load(open(f)) for f in files]
    n = parts[0]['num_shards']
    if sorted(p['shard'] for p in parts) != list(range(n)) or any(p['num_shards'] != n for p in parts):
        raise ValueError('%s: shards %s of %d' % (d, [p['shard'] for p in parts], n))
    if len({p['ckpt'] for p in parts}) != 1 or len({p['cfg_file'] for p in parts}) != 1:
        raise ValueError('%s: shards from different checkpoints or configs' % d)
    mat = sum(np.asarray(p['plain']['matrices']['cone'], dtype=np.int64) for p in parts)
    return {'dir': d, 'cone': mat, 'ckpt': parts[0]['ckpt'], 'cfg_file': parts[0]['cfg_file'],
            'epoch': parts[0]['epoch'], 'commit': parts[0].get('commit'), 'frames': sum(p['frames_in_shard'] for p in parts)}


def load_inrun(ckpt):
    """The in-run evaluation beside a checkpoint: <run>/files/ckpt/checkpoint_epoch_391.pth -> <run>/files/eval_epoch_391.json."""
    files = os.path.dirname(os.path.dirname(ckpt))
    ep = os.path.basename(ckpt).replace('checkpoint_epoch_', '').replace('.pth', '')
    f = os.path.join(files, 'eval_epoch_%s.json' % ep)
    if not os.path.isfile(f):
        return None
    d = json.load(open(f))
    return {'file': f, 'cone': np.asarray(d['matrices']['cone'], dtype=np.int64), 'run': files.split('-')[-1].split('/')[0]}


# --- where each number lives (the §8.10 layout) -------------------------------------------------------------------------

def cross_dir(seed, model, val):
    base = 'p1_cross_391' if seed == 666 else 'p1_cross_391_s%d' % seed
    return os.path.join(ROOT, base, 'model_%s_on_%s' % (model, VAL[val].replace('spunet-', '')))


def arm_dir(arm, seed, cfg):
    return os.path.join(ROOT, 's3_%s_s%d' % (arm, seed), cfg)


def reliance_dir(model, seed, cfg):
    return os.path.join(ROOT, 's3_reliance', 'oracle_%s_s%d' % (model, seed), cfg)


# --- statistics ---------------------------------------------------------------------------------------------------------

def summary(values):
    """Mean, sample SD and readability of seed-paired values (in mIoU points); None entries make it incomplete."""
    v = [x for x in values if x is not None]
    out = {'values': values, 'n': len(v)}
    if len(v) < len(values) or not v:
        out['complete'] = False
        return out
    m = float(np.mean(v))
    out.update({'complete': True, 'mean': m, 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else float('nan'),
                'readable': bool(abs(m) >= READABLE and all(np.sign(x) == np.sign(m) and x != 0 for x in v))})
    return out


def diff(a, b):
    return None if a is None or b is None else 100.0 * (a - b)


# --- main ---------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    names = class_names('cfgs/pcseg/spunet-pandaset64.yaml')
    assert names == class_names('cfgs/pcseg/spunet-pandasetgt.yaml'), 'the two sensors map to different class orders'
    res = {'names': names, 'dropped': list(DROPPED), 'inputs': {}, 'u3': [], 'missing': []}
    cache = {}

    def score(d):
        if d not in cache:
            t = load_test_seg(d)
            cache[d] = None if t is None else (t, cone16(t['cone'], names))
            if t is None:
                res['missing'].append(d)
            else:
                res['inputs'][d] = {'miou': cache[d][1], 'ckpt': t['ckpt'], 'cfg_file': t['cfg_file'],
                                    'epoch': t['epoch'], 'commit': t['commit'], 'frames': t['frames'],
                                    'points': int(t['cone'].sum())}
        return None if cache[d] is None else cache[d][1]

    def u3(d):
        """Rescore diagonal against the in-run evaluation of the same checkpoint."""
        if score(d) is None:
            return
        t = cache[d][0]
        r = load_inrun(t['ckpt'])
        row = {'dir': d, 'ckpt': t['ckpt']}
        if r is None:
            row['inrun'] = 'missing'
        else:
            a, b = cone16(t['cone'], names), cone16(r['cone'], names)
            row.update({'run': r['run'], 'points_rescore': int(t['cone'].sum()), 'points_inrun': int(r['cone'].sum()),
                        'points_moved': int(np.abs(t['cone'] - r['cone']).sum() // 2), 'miou_rescore': a,
                        'miou_inrun': b, 'abs_dmiou': abs(a - b),
                        'pass': bool(int(t['cone'].sum()) == int(r['cone'].sum()) and abs(a - b) <= T3_BOUND)})
        res['u3'].append(row)

    # U2 / U4: the 2 x 2 per seed
    x = {s: {(m, v): score(cross_dir(s, m, v)) for m in ('64', 'gt') for v in ('64', 'gt')} for s in SEEDS}
    for s in SEEDS:
        for m in ('64', 'gt'):
            u3(cross_dir(s, m, m))
    res['u2'] = {'oracle_64_on_64': summary([diff(x[s][('64', '64')], 0) for s in SEEDS]),
                 'oracle_gt_on_gt': summary([diff(x[s][('gt', 'gt')], 0) for s in SEEDS]),
                 'model_64_on_gt': summary([diff(x[s][('64', 'gt')], 0) for s in SEEDS]),
                 'model_gt_on_64': summary([diff(x[s][('gt', '64')], 0) for s in SEEDS])}
    res['u4'] = {'gap_64_to_gt': summary([diff(x[s][('gt', 'gt')], x[s][('64', 'gt')]) for s in SEEDS]),
                 'gap_gt_to_64': summary([diff(x[s][('64', '64')], x[s][('gt', '64')]) for s in SEEDS])}

    # terms, on the target's unperturbed val
    a = {(arm, s, v): score(arm_dir(arm, s, VAL[v])) for s in SEEDS
         for arm, v in [('a1', 'gt'), ('a2', 'gt'), ('a3', '64'), ('a4', '64'), ('b0', 'gt'), ('b0', '64'),
                        ('b1', 'gt'), ('b1', '64')]}
    for s in SEEDS:
        for arm, v in [('a1', 'gt'), ('a2', 'gt'), ('a3', '64'), ('a4', '64'), ('b0', 'gt'), ('b1', 'gt')]:
            u3(arm_dir(arm, s, VAL[v]))
    og = {s: x[s][('gt', 'gt')] for s in SEEDS}
    o6 = {s: x[s][('64', '64')] for s in SEEDS}
    terms = {'term_a1': ([diff(og[s], a[('a1', s, 'gt')]) for s in SEEDS], 'gap_64_to_gt'),
             'term_a2': ([diff(og[s], a[('a2', s, 'gt')]) for s in SEEDS], 'gap_64_to_gt'),
             'term_a3': ([diff(o6[s], a[('a3', s, '64')]) for s in SEEDS], 'gap_gt_to_64'),
             'term_a4': ([diff(o6[s], a[('a4', s, '64')]) for s in SEEDS], 'gap_gt_to_64'),
             'term_b1': ([diff(a[('b0', s, 'gt')], a[('b1', s, 'gt')]) for s in SEEDS], 'gap_64_to_gt')}
    res['terms'] = {}
    for k, (vals, gap) in terms.items():
        t = summary(vals)
        g = res['u4'][gap]
        if t.get('complete') and g.get('complete'):
            t['share_of'] = gap
            t['share'] = t['mean'] / g['mean']
        res['terms'][k] = t
    res['readings'] = {'render_cost': summary([diff(og[s], a[('b0', s, 'gt')]) for s in SEEDS]),
                       'b2_correction': summary([diff(a[('b1', s, '64')], a[('b0', s, '64')]) for s in SEEDS])}

    # reliance forms: model(unperturbed) - model(on the arm's -reliance config)
    rel = {}
    for arm, model in [('a1-density', 'gt'), ('a2-intensity', 'gt'), ('a3-cone', '64'), ('a4-intensity', '64'),
                       ('b1-lines', 'gt')]:
        cfg = 'spunet-p1-%s-reliance' % arm
        rel['oracle_' + arm] = summary([diff(x[s][(model, model)], score(reliance_dir(model, s, cfg))) for s in SEEDS])
    rel['b0_b1-lines'] = summary([diff(a[('b0', s, 'gt')], score(arm_dir('b0', s, 'spunet-p1-b1-lines-reliance')))
                                  for s in SEEDS])
    res['reliance'] = rel

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    fmt = lambda t: ('incomplete (%d of %d)' % (t['n'], len(t['values'])) if not t.get('complete') else
                     'mean %.2f  sd %.2f  [%s]%s%s' % (t['mean'], t['sd'], ', '.join('%.2f' % v for v in t['values']),
                                                     '  readable' if t.get('readable') else '',
                                                     ('  share %.3f of %s' % (t['share'], t['share_of'])
                                                      if 'share' in t else '')))
    for sec in ('u2', 'u4', 'terms', 'readings', 'reliance'):
        print('==', sec)
        for k, t in res[sec].items():
            print('  %-24s %s' % (k, fmt(t)))
    print('== u3')
    for r in res['u3']:
        print('  ', {k: r[k] for k in r if k not in ('ckpt',)})
    print('== missing: %d directories' % len(res['missing']))


if __name__ == '__main__':
    main()
