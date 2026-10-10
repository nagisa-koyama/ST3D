"""ANALYSIS (experiments_md 20261010_11): how much AP does an oracle lose to errors in its OWN predicted boxes?

An oracle's `result.pkl` (its predictions on its own val set) is re-scored after ONE controlled error is added to
every predicted box: size, heading (yaw) or position, as a constant bias or as zero-mean per-box noise. Scores,
classes and the number of boxes are untouched, so this isolates box geometry from detection. It is the box-space
counterpart of the point-cloud manipulations in 20261010_01 (those degrade the INPUT; this degrades the OUTPUT).

The scoring path is the dataset's own `evaluation()` (so the base arm must reproduce the logged AP), with only the
rotated-IoU kernel replaced by pcdet's CPU op, as in `kitti_eval_cpu.py`. It needs no GPU and is meant to run as a
Slurm CPU job, never on the master node.

    python analysis/oracle_box_error_sensitivity.py --cfg cfgs/.../centerpoint-sourceonly-nuscenes.yaml \
        --result .../result.pkl --label nuscenes_oracle --out out.jsonl [--arms base,sz_iso1.10,...] [--workers 8]

Boxes are `boxes_lidar` = (x, y, z, l, w, h, heading) in the LiDAR frame. Size errors keep the box CENTRE (as
`posthoc_box_scale.py` does). `lon` / `lat` move the centre along / across the box's own heading, `z` vertically.
Target labels are read only by the scoring itself: this is analysis, and no method setting is chosen from it.
"""
import argparse
import copy
import json
import multiprocessing as mp
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED = 0


def build_arms():
    """name -> (kind, params). `base` is the unperturbed control and must reproduce the logged AP."""
    arms = {'base': ('base', {})}
    for s in (0.90, 0.95, 1.05, 1.10, 1.20):
        arms[f'sz_iso{s:.2f}'] = ('size', {'scale': (s, s, s)})
    for dim, idx in (('l', 0), ('w', 1), ('h', 2)):
        for s in (0.90, 1.10):
            sc = [1.0, 1.0, 1.0]; sc[idx] = s
            arms[f'sz_{dim}{s:.2f}'] = ('size', {'scale': tuple(sc)})
    for sig in (0.05, 0.10, 0.20):
        arms[f'szn{sig:.2f}'] = ('size_noise', {'sigma': sig})
    for deg in (2, 5, 10, 20):
        arms[f'yaw_b{deg}'] = ('yaw_bias', {'deg': float(deg)})
    for deg in (2, 5, 10, 20):
        arms[f'yaw_n{deg}'] = ('yaw_noise', {'sigma_deg': float(deg)})
    for ax in ('lon', 'lat', 'z'):
        for m in (0.2, 0.5):
            arms[f'{ax}{m:.1f}'] = ('pos_bias', {'axis': ax, 'm': m})
    for m in (0.1, 0.2, 0.5):
        arms[f'pxy{m:.1f}'] = ('pos_noise', {'sigma_xy': m, 'sigma_z': 0.0})
    for m in (0.1, 0.2):
        arms[f'pz{m:.1f}'] = ('pos_noise', {'sigma_xy': 0.0, 'sigma_z': m})
    return arms


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def perturb_boxes(boxes, kind, params, rng):
    """Return a perturbed COPY of `boxes` [N, 7+] (x, y, z, l, w, h, heading, ...). Empty input is returned empty."""
    out = np.array(boxes, dtype=np.float64, copy=True)
    n = len(out)
    if n == 0 or kind == 'base':
        return out.astype(np.asarray(boxes).dtype)
    if kind == 'size':
        out[:, 3:6] *= np.asarray(params['scale'], dtype=np.float64)
    elif kind == 'size_noise':
        out[:, 3:6] *= np.exp(rng.normal(0.0, params['sigma'], size=(n, 3)))
    elif kind == 'yaw_bias':
        out[:, 6] = wrap_angle(out[:, 6] + np.deg2rad(params['deg']))
    elif kind == 'yaw_noise':
        out[:, 6] = wrap_angle(out[:, 6] + rng.normal(0.0, np.deg2rad(params['sigma_deg']), size=n))
    elif kind == 'pos_bias':
        h = out[:, 6]
        if params['axis'] == 'lon':
            out[:, 0] += params['m'] * np.cos(h); out[:, 1] += params['m'] * np.sin(h)
        elif params['axis'] == 'lat':
            out[:, 0] += -params['m'] * np.sin(h); out[:, 1] += params['m'] * np.cos(h)
        elif params['axis'] == 'z':
            out[:, 2] += params['m']
        else:
            raise ValueError(params['axis'])
    elif kind == 'pos_noise':
        out[:, 0:2] += rng.normal(0.0, params['sigma_xy'], size=(n, 2)) if params['sigma_xy'] else 0.0
        out[:, 2] += rng.normal(0.0, params['sigma_z'], size=n) if params['sigma_z'] else 0.0
    else:
        raise ValueError(kind)
    return out.astype(np.asarray(boxes).dtype)


def perturb_result(det_annos, kind, params, seed=SEED):
    """Deep copy of the detections with only `boxes_lidar` perturbed (one rng per arm, frames in file order)."""
    rng = np.random.default_rng(seed)
    out = []
    for d in det_annos:
        d = copy.deepcopy(d)
        if 'boxes_lidar' in d and len(d['boxes_lidar']):
            d['boxes_lidar'] = perturb_boxes(d['boxes_lidar'], kind, params, rng)
        out.append(d)
    return out


def install_cpu_iou():
    """Replace the CUDA rotated-IoU module before eval.py is imported by the dataset's evaluation()."""
    from kitti_eval_cpu import rotate_iou_cpu
    name = 'pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou'
    stub = types.ModuleType(name)
    stub.rotate_iou_gpu_eval = rotate_iou_cpu
    sys.modules[name] = stub


_STATE = {}


def score_arm(name):
    kind, params = _STATE['arms'][name]
    t0 = time.time()
    dets = perturb_result(_STATE['dets'], kind, params)
    _, ap = _STATE['dataset'].evaluation(dets, _STATE['classes'], eval_metric='kitti', output_path=None)
    res = {'label': _STATE['label'], 'arm': name, 'kind': kind, 'params': params,
           'bev': float(ap['Car_bev/moderate_R40']), '3d': float(ap['Car_3d/moderate_R40']),
           'bev_easy': float(ap['Car_bev/easy_R40']), 'seconds': round(time.time() - t0, 1)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--result', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True, help='jsonl, appended per arm; arms already in it are skipped')
    ap.add_argument('--arms', default='all')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--classes', default='Car')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    arms = build_arms()
    if args.list:
        print('\n'.join(arms)); return
    chosen = list(arms) if args.arms == 'all' else args.arms.split(',')
    assert all(c in arms for c in chosen), [c for c in chosen if c not in arms]
    out = Path(args.out)
    done = {json.loads(l)['arm'] for l in out.read_text().splitlines()} if out.exists() else set()
    chosen = [c for c in chosen if c not in done]
    print(f'{args.label}: {len(chosen)} arms to score ({len(done)} done)', flush=True)

    import _init_path  # noqa: F401,E402  (before any pcdet import)
    install_cpu_iou()
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    cfg_from_yaml_file(args.cfg, cfg)
    eval_cfg = cfg.DATA_CONFIG_TAR if cfg.get('DATA_CONFIG_TAR', None) else cfg.DATA_CONFIG
    logger = common_utils.create_logger('/dev/null', rank=0)
    test_set, _, _ = build_dataloader(dataset_cfg=eval_cfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                      workers=0, logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY', None))
    dets = pickle.load(open(args.result, 'rb'))
    assert len(dets) == len(test_set.infos), (len(dets), len(test_set.infos))
    _STATE.update(arms=arms, dets=dets, dataset=test_set, classes=args.classes.split(','), label=args.label)

    def record(r):
        with open(out, 'a') as f:
            f.write(json.dumps(r) + '\n')
        print(f"{r['label']} {r['arm']:<12} BEV {r['bev']:.2f}  3D {r['3d']:.2f}  ({r['seconds']} s)", flush=True)

    if args.workers > 1:
        with mp.get_context('fork').Pool(args.workers) as pool:
            for r in pool.imap_unordered(score_arm, chosen):
                record(r)
    else:
        for c in chosen:
            record(score_arm(c))


if __name__ == '__main__':
    main()
