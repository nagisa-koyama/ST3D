"""Compare a checkpoint's plain test_seg point matrix with the training run's own in-run evaluation of the same epoch
(experiments_md episodes/20261116_01 §8.4: a two-GPU run's sharded evaluation against a one-GPU rescore).

The bound is §5's for spconv's run-to-run nondeterminism: at most 100 points in a different cell and |delta mIoU| < 1e-4,
with the same point total. "Points moved" is sum |A - B| / 2, so a point that changes cell counts once.

  python analysis/pcseg_compare_matrices.py --cfg cfgs/pcseg/<x>.yaml --run_eval <run dir>/eval_epoch_<N>.json \
      --test <test_seg out>/shard_000_of_001.json --out <dir>/compare.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/
import _init_path  # noqa: F401,E402
import train_seg  # noqa: E402
from pcseg import engine  # noqa: E402
from pcseg.class_map import build_class_map  # noqa: E402

MAX_MOVED, MAX_DMIOU = 100, 1e-4


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--run_eval', required=True, help="the training run's eval_epoch_<N>.json")
    p.add_argument('--test', required=True, help="test_seg's shard or merged json (its 'plain' block)")
    p.add_argument('--out', required=True)
    args = p.parse_args()
    names = build_class_map(train_seg.load_cfg(args.cfg).DATA.CLASS_MAP).names
    run, test = json.load(open(args.run_eval)), json.load(open(args.test))
    a = np.asarray(run['matrices']['point'], dtype=np.int64)
    b = np.asarray(test['plain']['matrices']['point'], dtype=np.int64)
    ma = engine.metrics_from_matrix(a, names, 'p')['p/miou']
    mb = engine.metrics_from_matrix(b, names, 'p')['p/miou']
    moved = int(np.abs(a - b).sum() // 2)
    res = {'run_eval': args.run_eval, 'test': args.test, 'run_frames': run['val/frames'],
           'test_frames': test['plain']['frames'], 'run_points': int(a.sum()), 'test_points': int(b.sum()),
           'points_moved': moved, 'miou_run': ma, 'miou_test': mb, 'dmiou': mb - ma,
           'test_gpu': test.get('gpu'), 'test_commit': test.get('commit')}
    res['PASS'] = bool(res['run_frames'] == res['test_frames'] and res['run_points'] == res['test_points']
                       and moved <= MAX_MOVED and abs(mb - ma) < MAX_DMIOU)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print('matrix comparison:', 'PASS' if res['PASS'] else 'FAIL')


if __name__ == '__main__':
    main()
