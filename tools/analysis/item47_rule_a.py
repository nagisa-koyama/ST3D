"""Rule-A (and as-scored) rescoring of the item 47 Pandar64 rows at IoU 0.7 and IoU 0.5 BEV (experiments_md 20261003_05 §5
item 47, design fixed 2026-11-21 19:30 JST), as item 42a's Pandar64 rows were (analysis/item42_rule_a.py, whose scoring
functions are used unchanged: pandaset_eval_protocols.score with the official evaluator's printed string kept).

Rows: the 16 Pandar64 rows of analysis/make_item47_configs.py (O, P, L, C1 x seeds 666 / 667 x stats none / target), scored on
S' val (spin, 360 deg, device 0). Ground truth is the target's unthinned label set (gt_with_counts(0)), the same for every arm.

    python3 analysis/item47_rule_a.py --out /home/koyama/data/item47/rule_a_<stats>.json [--stats none|target]   # Slurm CPU job

The JSON is rewritten after every scored row, so a job that hits its time limit keeps what it finished. Exits 1 if a
result.pkl is missing.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'analysis'))


def rows(stats=None):
    import make_item47_configs as m47
    return [dict(row=row, seed=seed, job=job, stats=st, pkl=str(m47.result_pkl(part, row, seed, job, epoch, st, cfg)))
            for part, row, seed, job, epoch, st, cfg, _ in m47.ROWS if part == 'P' and (stats is None or st == stats)]


def label_of(r):
    return '%s %s s%d (job %d)' % (r['stats'], r['row'], r['seed'], r['job'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--stats', choices=['none', 'target'], default=None)
    p.add_argument('--dry', action='store_true', help='list the rows and whether each result.pkl exists, then stop')
    args = p.parse_args()
    rs = rows(args.stats)
    if args.dry:
        for r in rs:
            print(label_of(r), os.path.exists(r['pkl']), r['pkl'])
        return
    import _init_path  # noqa: F401
    import pandaset_eval_protocols as pep
    from kitti_eval_cpu import load_official_eval
    from item42_rule_a import PROTOCOLS, score_both
    os.chdir(TOOLS)
    official = load_official_eval()
    gt = pep.gt_with_counts(0)
    print('%-34s %-10s %8s %8s %8s' % ('row', 'protocol', 'BEV@0.7', '3D@0.7', 'BEV@0.5'))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for r in rs:
        label = label_of(r)
        if not os.path.exists(r['pkl']):
            print('%-34s missing %s' % (label, r['pkl']), flush=True)
            r['missing'] = True
            continue
        dets = pickle.load(open(r['pkl'], 'rb'))
        for proto in PROTOCOLS:
            r[proto] = score_both(pep.score, gt, dets, False, proto, official)
            print('%-34s %-10s %8.2f %8.2f %8.2f' % ((label, proto) + r[proto]), flush=True)
        json.dump([x for x in rs if all(pr in x for pr in PROTOCOLS)], open(args.out, 'w'), indent=1)
    print('wrote', args.out)
    if any(r.get('missing') for r in rs):
        sys.exit(1)


if __name__ == '__main__':
    main()
