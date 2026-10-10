"""ANALYSIS (experiments_md 20261011_01): the PandaSet counterpart of oracle_box_error_sensitivity.py.

Same 34 arms (size / heading / position errors on the oracle's own predicted `boxes_lidar`), but scored with
`pandaset_eval_protocols.score`, i.e. under PandaSet's adopted rule A (and, for reference, as scored), because the
dataset's own evaluation() scores GT that no detector can reach. The base arm must reproduce the ledger value
(20261003_09 §2b). CPU only: run it as a Slurm CPU job.

    python analysis/oracle_box_error_pandaset.py --result <result.pkl> --device 0 --cone 0 --label pandaset_spin_oracle \
        --out logs/box_error/pandaset_spin_oracle.jsonl [--arms all] [--workers 8]
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent)); sys.path.insert(0, str(TOOLS / 'analysis'))
import oracle_box_error_sensitivity as B  # noqa: E402
import pandaset_eval_protocols as P  # noqa: E402  (imports _init_path first)

PROTOCOLS = ('rule_A', 'as_scored')
_STATE = {}


def score_arm(name):
    kind, params = _STATE['arms'][name]
    t0 = time.time()
    dets = B.perturb_result(_STATE['dets'], kind, params)
    res = {'label': _STATE['label'], 'arm': name, 'kind': kind, 'params': params}
    for proto in PROTOCOLS:
        bev, d3 = P.score(_STATE['gt'], dets, _STATE['cone'], proto, _STATE['official'])
        res[f'bev_{proto}'] = float(bev); res[f'3d_{proto}'] = float(d3)
    res['seconds'] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--result', required=True)
    ap.add_argument('--device', type=int, required=True, help='target sensor: 0 spin (Pandar64), 1 flash (PandarGT)')
    ap.add_argument('--cone', type=int, default=0)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--arms', default='all')
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()
    arms = B.build_arms()
    chosen = list(arms) if args.arms == 'all' else args.arms.split(',')
    assert all(c in arms for c in chosen)
    out = Path(args.out)
    done = {json.loads(l)['arm'] for l in out.read_text().splitlines()} if out.exists() else set()
    chosen = [c for c in chosen if c not in done]
    print(f'{args.label}: {len(chosen)} arms to score ({len(done)} done)', flush=True)
    os.chdir(TOOLS)
    official = P.load_official_eval()
    dets = pickle.load(open(args.result, 'rb'))
    gt = P.gt_with_counts(args.device)
    assert len(dets) == len(gt), (len(dets), len(gt))
    _STATE.update(arms=arms, dets=dets, gt=gt, cone=bool(args.cone), official=official, label=args.label)

    def record(r):
        with open(out, 'a') as f:
            f.write(json.dumps(r) + '\n')
        print(f"{r['label']} {r['arm']:<12} rule A BEV {r['bev_rule_A']:.2f} 3D {r['3d_rule_A']:.2f} | "
              f"as scored BEV {r['bev_as_scored']:.2f} 3D {r['3d_as_scored']:.2f} ({r['seconds']} s)", flush=True)

    if args.workers > 1:
        with mp.get_context('fork').Pool(args.workers) as pool:
            for r in pool.imap_unordered(score_arm, chosen):
                record(r)
    else:
        for c in chosen:
            record(score_arm(c))


if __name__ == '__main__':
    main()
