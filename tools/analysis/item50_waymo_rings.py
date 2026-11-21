"""Item 50 (experiments_md 20261003_05 §5 item 50, design fixed 2026-11-21 21:58 JST, experiments_md commit cfce639, before any
number): the nuScenes->Waymo pseudo-labelling gain by range, split into item 46's components, both seeds.

Scoring: the Waymo dataset's own `kitti` evaluation path (waymo_dataset.evaluation -> kitti_eval: the same map_name_to_kitti,
kitti_utils.transform_annotations_to_kitti_format on detections and ground truth, the official evaluator, Car), restricted to a
ring as kitti_eval_cpu.ring_view does for KITTI: Car GT whose box centre lies outside the ring becomes Van (the evaluator's
ignore path for Car), detections outside it are dropped. Range = the box centre's x-y distance in the LiDAR frame. The ring
(0, inf) must reproduce each row's logged value within 0.02 (else the row is not read). "All objects" regime: the fabricated 2D
boxes put every object in every difficulty, so moderate = easy = hard.

    python3 analysis/item50_waymo_rings.py score OUT.json [--real-kernel] [--workers 4]   # GPU node for --real-kernel
    python3 analysis/item50_waymo_rings.py read OUT.json

Rows: T 27471 (shared), E 33102 / E2 33264, PL 27704 / PL2 28563, each `none` and `src`, epoch 20; PL2 `src` from the rerun
i46_P2nw_src_r (T05 item 50 amendment). Components and readings: analysis/item49_pl_rings.py's (O, G, S, L per ring; near
0-20 m, far 30 m and beyond). Tests: tests/test_item50_waymo_rings.py. ANALYSIS: Waymo labels score only.
"""
import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'analysis'))

CFG = 'cfgs/da-ieee-access/centerpoint-gblobs-accum-legaldepth-st3d-nuscenes2waymo.yaml'
OUTD = TOOLS.parent / 'output' / 'da-ieee-access' / 'centerpoint-gblobs-accum-legaldepth-st3d-nuscenes2waymo'
RINGS = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, float('inf'))]
TAGS = {666: {'T': 'Tnw', 'E': 'Enw', 'PL': 'Pnw'}, 667: {'T': 'Tnw', 'E': 'E2nw', 'PL': 'P2nw'}}
JOB = {666: {'T': 27471, 'E': 33102, 'PL': 27704}, 667: {'T': 27471, 'E': 33264, 'PL': 28563}}
MAP = {'Vehicle': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist', 'Sign': 'Sign', 'Car': 'Car'}  # waymo_dataset.py
ITEM46 = {666: {'O': 0.0137, 'G': 1.9280, 'S': -0.3772, 'L': 2.5145}, 667: {'G': 2.0967, 'S': -0.2354, 'L': 2.2523}}
TOL = 0.02


def result_dir(row, stats, seed):
    t = 'i46_%s_%s' % (TAGS[seed][row], stats)
    if (row, stats, seed) == ('PL', 'src', 667):
        t += '_r'  # amendment: the rerun after the node03 timeout
    return OUTD / t / 'eval' / 'epoch_20' / t


def rows():
    out = []
    for seed in (666, 667):
        for row in ('T', 'E', 'PL'):
            if seed == 667 and row == 'T':
                continue  # shared
            for stats in ('none', 'src'):
                out.append(dict(seed=seed, row=row, stats=stats, job=JOB[seed][row], dir=str(result_dir(row, stats, seed))))
    return out


def ring_filter(dt, gt, dxy, gxy, lo, hi):
    """Copies of one frame's KITTI-format detections and ground truth restricted to lo <= range < hi."""
    dt, gt = copy.deepcopy(dt), copy.deepcopy(gt)
    names = np.array(gt['name'], dtype=object)
    assert len(names) == len(gxy), 'every GT row has a LiDAR box on Waymo'
    if len(gxy):
        r = np.hypot(gxy[:, 0], gxy[:, 1])
        names[(names == 'Car') & ~((r >= lo) & (r < hi))] = 'Van'
    gt['name'] = names.astype(str)
    nd = len(dt['name'])
    keep = (np.hypot(dxy[:, 0], dxy[:, 1]) >= lo) & (np.hypot(dxy[:, 0], dxy[:, 1]) < hi) if nd else np.zeros(0, bool)
    for k, v in dt.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == nd:
            dt[k] = v[keep]
    return dt, gt


def kitti_view(det_annos, gt_annos):
    """waymo_dataset.evaluation's kitti_eval conversion, with the LiDAR x-y of every box kept for the rings."""
    from pcdet.datasets.kitti import kitti_utils
    det, gt = copy.deepcopy(det_annos), copy.deepcopy(gt_annos)
    dxy = [np.asarray(d['boxes_lidar'])[:, :2] if len(d['name']) else np.zeros((0, 2)) for d in det]
    gxy = [np.asarray(g['gt_boxes_lidar'])[:, :2] if len(g['name']) else np.zeros((0, 2)) for g in gt]
    kitti_utils.transform_annotations_to_kitti_format(det, map_name_to_kitti=MAP)
    kitti_utils.transform_annotations_to_kitti_format(gt, map_name_to_kitti=MAP, info_with_fakelidar=False)
    return det, gt, dxy, gxy


_W = {}


def _init(real_kernel, annos_path):
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    import _init_path  # noqa: F401
    import kitti_eval_cpu
    _W['official'] = kitti_eval_cpu.load_official_eval(real_kernel=real_kernel)
    _W['gt'] = pickle.load(open(annos_path, 'rb'))


def _score(r):
    from item42_collect import read_log
    official, (frame_ids, gt_annos) = _W['official'], _W['gt']
    got = read_log((r['dir'], 'log_adabn_*.txt', None))
    det_annos = pickle.load(open(Path(r['dir']) / 'result.pkl', 'rb'))
    assert [d['frame_id'] for d in det_annos] == frame_ids, '%s: result.pkl frames do not align with the infos' % r['dir']
    det, gt, dxy, gxy = kitti_view(det_annos, gt_annos)
    out = []
    for lo, hi in [(0.0, float('inf'))] + RINGS:
        dts, gts = zip(*[ring_filter(d, g, a, b, lo, hi) for d, g, a, b in zip(det, gt, dxy, gxy)])
        n_gt = int(sum(official.clean_data(g, d, 0, 1)[0] for g, d in zip(gts, dts)))
        _, ap = official.get_official_eval_result(list(gts), list(dts), ['Car'])
        out.append(dict(lo=lo, hi=hi, n_gt=n_gt, bev=float(ap['Car_bev/moderate_R40'])))
    logged = got['bev07'] if got else None
    diff = None if logged is None else out[0]['bev'] - logged
    print('%s %s s%d pooled %.4f logged %s' % (r['row'], r['stats'], r['seed'], out[0]['bev'], logged), flush=True)
    return dict(r, pooled=out[0], rings=out[1:], logged_bev=logged, diff=diff,
                reproduced=bool(diff is not None and abs(diff) <= TOL))


def cmd_score(args):
    import multiprocessing as mp
    from easydict import EasyDict
    import _init_path  # noqa: F401
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import __all__ as DATASETS
    from pcdet.utils import common_utils
    os.chdir(TOOLS)
    cfg = cfg_from_yaml_file(CFG, EasyDict())
    block = cfg.DATA_CONFIG_TAR
    ds = DATASETS[block.DATASET](dataset_cfg=block, class_names=cfg.CLASS_NAMES, root_path=None, training=False,
                                 logger=common_utils.create_logger(), model_ontology=cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None))
    annos_path = Path(args.out).with_suffix('.gt.pkl')
    pickle.dump(([info['frame_id'] for info in ds.infos], [info['annos'] for info in ds.infos]), open(annos_path, 'wb'))
    print('val frames %d (%s)' % (len(ds.infos), block.get('SAMPLED_INTERVAL', None)), flush=True)
    rs = rows()
    with mp.get_context('spawn').Pool(args.workers, initializer=_init, initargs=(args.real_kernel, str(annos_path))) as pool:
        out = pool.map(_score, rs, chunksize=1)
    json.dump({'rings': RINGS, 'kernel': 'cuda' if args.real_kernel else 'cpu', 'rows': out}, open(args.out, 'w'), indent=1,
              default=str)
    print('wrote', args.out)


def cmd_read(path):
    import item49_pl_rings as m49
    d = json.load(open(path))
    rs = d['rows']
    for r in rs:
        print('%-2s %-4s s%d pooled %6.2f logged %s %s | %s | n_gt %s' % (
            r['row'], r['stats'], r['seed'], r['pooled']['bev'], r['logged_bev'], 'ok' if r['reproduced'] else 'NOT REPRODUCED',
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '/'.join(str(g['n_gt']) for g in r['rings'])))
    readings = {}
    for seed in (666, 667):
        pick = lambda row, st: [x for x in rs if x['row'] == row and x['stats'] == st and x['seed'] == (666 if row == 'T' else seed)]  # noqa: E731
        sel = {(row, st): pick(row, st) for row in ('T', 'E', 'PL') for st in ('none', 'src')}
        if any(len(v) != 1 or not v[0]['reproduced'] for v in sel.values()):
            print('seed %d: not read (a row is missing or not reproduced)' % seed)
            continue
        v = {k: [g['bev'] for g in x[0]['rings']] for k, x in sel.items()}
        p = {k: [x[0]['pooled']['bev']] for k, x in sel.items()}
        c, cp = m49.components(v), m49.components(p)
        print('\nseed %d' % seed)
        readings[seed] = {}
        for k in ('O', 'G', 'S', 'L', 'PL-T'):
            ref = ITEM46[seed].get(k)
            chk = '' if ref is None else '  pooled check %s (item 46 %+.4f)' % ('pass' if abs(cp[k][0] - ref) <= 0.05 else 'FLAG', ref)
            readings[seed][k] = m49.classify(c[k])
            print('%-4s pooled %+6.2f | %s | near %+.2f far %+.2f -> %s%s' % (
                k, cp[k][0], ' '.join('%+6.2f' % x for x in c[k]), m49.mean([c[k][i] for i in m49.NEAR]),
                m49.mean([c[k][i] for i in m49.FAR]), readings[seed][k], chk))
    if len(readings) == 2:
        print('\ntwo-seed readings:')
        for k in ('O', 'G', 'S', 'L', 'PL-T'):
            a, b = readings[666][k], readings[667][k]
            print('  %-4s %s' % (k, a if a == b else 'seeds disagree (%s / %s)' % (a, b)))
    return readings


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score'); s.add_argument('out'); s.add_argument('--real-kernel', action='store_true')
    s.add_argument('--workers', type=int, default=4)
    r = sub.add_parser('read'); r.add_argument('path')
    args = p.parse_args()
    if args.cmd == 'score':
        cmd_score(args)
    else:
        cmd_read(args.path)


if __name__ == '__main__':
    main()
