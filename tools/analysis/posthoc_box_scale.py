"""Output-space correction at TEST time: rescale every predicted box, re-score with the official evaluator.

The question it answers (experiments_md/20261004_01): how much of what a training-time size prior
(ROS centred on a ratio) buys can be had by multiplying the predicted l / w / h by the same ratio after
the fact, on a checkpoint that never saw the prior. If most of it, output-space correction is a free
post-processing step that composes with any input- or feature-level method; if little, the prior has to
act through training (the network regresses partial observations toward its training size prior).

Each box keeps its CENTRE (as random_object_scaling scales about the centre); KITTI's `location` is the
bottom centre in the rect camera frame (y down), so it moves by (h - h') / 2. `bbox` (2D) is left alone:
it only enters the difficulty filter through its height, which a few-percent 3D scale barely moves.

    python posthoc_box_scale.py <label>=<result.pkl> [...] --scale iso0.933=0.933,0.933,0.933 ...

A scale is `<name>=l,w,h` (lidar order). `base=1,1,1` is always scored first and must reproduce the
logged AP (the CPU evaluator matches it to the printed digit on BEV). Also printed: the mean l / w / h of
confident predictions (score >= 0.5) before scaling, to read the residual size bias against.
"""
import argparse
import copy
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_eval_cpu import INFOS, evaluate, load_official_eval  # noqa: E402


def scale_result(dets, l, w, h):
    out = []
    for d in dets:
        d = copy.deepcopy(d)
        if len(d['name']):
            dims = d['dimensions'].astype(np.float64)  # camera order l, h, w
            new = dims * np.array([l, h, w])
            d['location'] = d['location'].astype(np.float64).copy()
            d['location'][:, 1] -= (dims[:, 1] - new[:, 1]) / 2.0  # keep the centre: bottom moves up/down
            d['dimensions'] = new.astype(np.float32)
            if 'boxes_lidar' in d and len(d['boxes_lidar']):
                d['boxes_lidar'] = d['boxes_lidar'].copy()
                d['boxes_lidar'][:, 3:6] *= np.array([l, w, h])
        out.append(d)
    return out


def confident_mean_size(dets, thresh=0.5):
    dims = [d['dimensions'][(d['name'] == 'Car') & (d['score'] >= thresh)] for d in dets if len(d['name'])]
    dims = np.concatenate([x for x in dims if len(x)], axis=0)
    return dims[:, 0].mean(), dims[:, 2].mean(), dims[:, 1].mean(), len(dims)  # l, w, h


def gt_mean_size():
    infos = pickle.load(open(INFOS, 'rb'))
    dims = np.concatenate([i['annos']['dimensions'][i['annos']['name'] == 'Car'] for i in infos], axis=0)
    return dims[:, 0].mean(), dims[:, 2].mean(), dims[:, 1].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', nargs='+', help='<label>=<result.pkl>')
    parser.add_argument('--scale', action='append', default=[], help='<name>=l,w,h (lidar order)')
    parser.add_argument('--label_oracle', action='store_true',
                        help='also score l/w/h scaled to KITTI val GT mean / confident-prediction mean '
                             '(READS TARGET LABELS: a reference, never a method)')
    parser.add_argument('--tmpdir', default=None)
    args = parser.parse_args()
    official = load_official_eval()
    scales = [('base', (1.0, 1.0, 1.0))]
    for s in args.scale:
        name, vals = s.split('=', 1)
        scales.append((name, tuple(float(v) for v in vals.split(','))))
    gt = gt_mean_size() if args.label_oracle else None
    if gt is not None:
        print(f'KITTI val GT Car mean l / w / h: {gt[0]:.3f} / {gt[1]:.3f} / {gt[2]:.3f}')
    tmpdir = Path(args.tmpdir or tempfile.mkdtemp())
    for item in args.results:
        label, path = item.split('=', 1)
        dets = pickle.load(open(path, 'rb'))
        ml, mw, mh, n = confident_mean_size(dets)
        print(f'{label}: confident Car predictions (n={n}) mean l / w / h {ml:.3f} / {mw:.3f} / {mh:.3f}', flush=True)
        arms = list(scales)
        if gt is not None:
            arms.append(('label-oracle', (gt[0] / ml, gt[1] / mw, gt[2] / mh)))
        for name, (l, w, h) in arms:
            tmp = tmpdir / f'{label}_{name}.pkl'
            pickle.dump(scale_result(dets, l, w, h), open(tmp, 'wb'))
            _, ap = evaluate(str(tmp), official=official)
            print(f'  {name:14s} x({l:.3f},{w:.3f},{h:.3f})  BEV {ap["Car_bev/easy_R40"]:.2f} / '
                  f'{ap["Car_bev/moderate_R40"]:.2f} / {ap["Car_bev/hard_R40"]:.2f}   3D '
                  f'{ap["Car_3d/easy_R40"]:.2f} / {ap["Car_3d/moderate_R40"]:.2f} / {ap["Car_3d/hard_R40"]:.2f}',
                  flush=True)
            tmp.unlink()


if __name__ == '__main__':
    main()
