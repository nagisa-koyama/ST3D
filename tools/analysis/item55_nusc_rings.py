"""Item 55 (experiments_md 20261003_05 §5 item 55, design fixed 2026-11-26 22:53 JST, experiments_md commit 86a0173, before any
number): where by range the lattice-thinning gain sits on Waymo->nuScenes and PandaSet->nuScenes. ANALYSIS (nuScenes labels score
only).

    python3 analysis/item55_nusc_rings.py score OUT.json [--real-kernel] [--workers 4]   # GPU node for --real-kernel
    python3 analysis/item55_nusc_rings.py read OUT.json

Rows: item 47 part B (BW / BP: P, L, C1 x seeds 666 / 667 x none / target; make_item47_configs.ROWS, saved predictions of 34801 /
34802) and, descriptive, the nuScenes oracle 31818 (seed 667, epoch 20). Scoring: NuScenesDataset.kitti_eval's own conversion is
captured (its call of the official evaluator is intercepted), then the official evaluator scores Car per ring ("all objects":
the fabricated 2D boxes put every object in every difficulty). Ring = the box centre's x-y distance in the LiDAR frame, from the
converted location (x = -y_lidar, z = x_lidar); item 50's view: Car GT outside the ring becomes Van, detections outside it are
dropped. A row whose pooled value differs from its logged value by more than 0.02 is not read.

Readings (T05 item 55): per pair, seed and stats, s(b) = P(b) - C1(b) and P(b) - L(b), r(b) = s(b) / P(b); item 48's rule
(item48_rings.classify): "gain far" / "gain near" / "over the whole range". `none` is the reading. Tests:
tests/test_item55_nusc_rings.py.
"""
import argparse
import copy
import importlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'analysis'))

RINGS = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, float('inf'))]
TOL = 0.02
PAIRS = {'BW': 'Waymo->nuScenes', 'BP': 'PandaSet->nuScenes'}
ORACLE = dict(part='oracle', row='O', seed=667, job=31818, stats='none', epoch=20,
              cfg='cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml')
_EVAL_MODULE = 'pcdet.datasets.kitti.kitti_object_eval_python.eval'


def rows():
    import make_item47_configs as m47
    import item38_kitti_rings as i38
    from item42_collect import read_log
    out = []
    for part, row, seed, job, epoch, stats, cfg, _ in m47.ROWS:
        if part not in PAIRS:
            continue
        pkl = m47.result_pkl(part, row, seed, job, epoch, stats, cfg)
        got = read_log((pkl.parent, 'log_adabn_*.txt', None))
        out.append(dict(part=part, row=row, seed=seed, job=job, epoch=epoch, stats=stats, cfg=cfg, result=str(pkl),
                        logged_bev=got['bev07'] if got else None, status='ok' if pkl.exists() and got else 'missing'))
    r = i38.resolve('oracle', 'O', ORACLE['seed'], ORACLE['job'], 'epoch_%d' % ORACLE['epoch'])
    out.append(dict(ORACLE, result=r.get('result'), logged_bev=r.get('logged_bev'), wandb=r.get('wandb'),
                    status='ok' if r['status'] == 'ok' and Path(r['result']).exists() else r['status']))
    return out


def capture(ds, det_annos, class_names):
    """(gt, dt) in KITTI format exactly as ds.kitti_eval hands them to the official evaluator (that call is intercepted)."""
    mod = importlib.import_module(_EVAL_MODULE)
    saved, box = mod.get_official_eval_result, {}

    def grab(gt_annos, dt_annos, current_classes, **kwargs):
        box.update(gt=gt_annos, dt=dt_annos, classes=current_classes)
        return '', {}

    mod.get_official_eval_result = grab
    try:
        ds.kitti_eval(copy.deepcopy(det_annos), copy.deepcopy(ds.infos), class_names)
    finally:
        mod.get_official_eval_result = saved
    return box['gt'], box['dt']


def xy_range(anno):
    loc = np.asarray(anno.get('location', np.zeros((0, 3)))).reshape(-1, 3)
    return np.hypot(loc[:, 0], loc[:, 2])   # location = (-y_lidar, -z_lidar, x_lidar)


def ring_view(gt, dt, lo, hi):
    """Copies of one frame's KITTI-format GT and detections restricted to lo <= range < hi (item 50's view)."""
    gt, dt = copy.deepcopy(gt), copy.deepcopy(dt)
    names = np.array(gt['name'], dtype=object)
    gr = xy_range(gt)
    assert len(gr) == len(names), 'every GT row carries a location'
    if len(names):
        names[(names == 'Car') & ~((gr >= lo) & (gr < hi))] = 'Van'
    gt['name'] = names.astype(str)
    nd = len(dt['name'])
    dr = xy_range(dt)
    keep = (dr >= lo) & (dr < hi) if nd else np.zeros(0, bool)
    for k, v in dt.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == nd:
            dt[k] = v[keep]
    return gt, dt


_W = {}


def _init(real_kernel):
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    import _init_path  # noqa: F401
    import kitti_eval_cpu
    os.chdir(TOOLS)
    _W['official'] = kitti_eval_cpu.load_official_eval(real_kernel=real_kernel)   # CPU: installs the rotate_iou stub first
    _W['ds'] = {}


def _dataset(cfg_path):
    if cfg_path not in _W['ds']:
        from easydict import EasyDict
        from pcdet.config import cfg_from_yaml_file
        from pcdet.datasets import __all__ as DATASETS
        from pcdet.utils import common_utils
        cfg = cfg_from_yaml_file(cfg_path, EasyDict())
        block = cfg.get('DATA_CONFIG_TAR', None) or cfg.DATA_CONFIG
        assert block.DATASET == 'NuScenesDataset', (cfg_path, block.DATASET)
        shift = block.get('SHIFT_COOR', None)
        assert shift is None or (shift[0] == 0 and shift[1] == 0), 'the rings assume no x-y shift'
        ds = DATASETS[block.DATASET](dataset_cfg=block, class_names=cfg.CLASS_NAMES, root_path=None, training=False,
                                     logger=common_utils.create_logger(),
                                     model_ontology=cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None))
        _W['ds'][cfg_path] = (ds, list(cfg.CLASS_NAMES))
    return _W['ds'][cfg_path]


def _score(r):
    official = _W['official']
    ds, class_names = _dataset(r['cfg'])
    det_annos = pickle.load(open(r['result'], 'rb'))
    assert len(det_annos) == len(ds.infos), '%s: %d predictions for %d frames' % (r['result'], len(det_annos), len(ds.infos))
    stems = [Path(i['lidar_path']).stem for i in ds.infos]
    assert all(d.get('frame_id', s) == s for d, s in zip(det_annos, stems)), '%s: frames do not align' % r['result']
    gt, dt = capture(ds, det_annos, class_names)
    out = []
    for lo, hi in [(0.0, float('inf'))] + RINGS:
        gts, dts = zip(*[ring_view(g, d, lo, hi) for g, d in zip(gt, dt)])
        n_gt = int(sum(official.clean_data(g, d, 0, 1)[0] for g, d in zip(gts, dts)))
        _, ap = official.get_official_eval_result(list(gts), list(dts), ['Car'])
        out.append(dict(lo=lo, hi=hi, n_gt=n_gt, bev=float(ap['Car_bev/moderate_R40'])))
    diff = None if r['logged_bev'] is None else out[0]['bev'] - r['logged_bev']
    print('%s %s %s s%d pooled %.4f logged %s' % (r['part'], r['row'], r['stats'], r['seed'], out[0]['bev'], r['logged_bev']),
          flush=True)
    return dict(r, pooled=out[0], rings=out[1:], diff=diff, reproduced=bool(diff is not None and abs(diff) <= TOL))


def cmd_score(args):
    import multiprocessing as mp
    rs = rows()
    bad = [r for r in rs if r['status'] != 'ok']
    assert not bad, bad
    with mp.get_context('spawn').Pool(args.workers, initializer=_init, initargs=(args.real_kernel,)) as pool:
        out = pool.map(_score, rs, chunksize=1)
    json.dump({'rings': RINGS, 'kernel': 'cuda' if args.real_kernel else 'cpu', 'rows': out}, open(args.out, 'w'), indent=1,
              default=str)
    print('wrote', args.out)


def ring_values(rs, part, row, seed, stats):
    hit = [x for x in rs if (x['part'], x['row'], x['seed'], x['stats']) == (part, row, seed, stats)]
    if len(hit) != 1 or not hit[0].get('reproduced'):
        return None
    return [g['bev'] for g in hit[0]['rings']]


NAMES = {'residual far': 'gain far', 'residual near': 'gain near'}


def reading(rs, part, other, stats):
    import item48_rings as m48
    seeds = []
    for seed in (666, 667):
        a, b = ring_values(rs, part, 'P', seed, stats), ring_values(rs, part, other, seed, stats)
        seeds.append(m48.per_seed(a, b) if a is not None and b is not None else None)
    v = m48.classify(seeds)
    return seeds, NAMES.get(v, v)


def cmd_read(path):
    import item48_rings as m48
    d = json.load(open(path))
    rs = d['rows']
    for r in rs:
        print('%-6s %-2s %-6s s%d pooled %6.2f logged %s %s | %s | n_gt %s' % (
            r['part'], r['row'], r['stats'], r['seed'], r['pooled']['bev'], r['logged_bev'],
            'ok' if r['reproduced'] else 'NOT REPRODUCED', ' '.join('%6.2f' % g['bev'] for g in r['rings']),
            '/'.join(str(g['n_gt']) for g in r['rings'])))
    out = {}
    for part, pair in PAIRS.items():
        for other in ('C1', 'L'):
            for stats in ('none', 'target'):
                seeds, v = reading(rs, part, other, stats)
                out[(part, other, stats)] = v
                print('\n%s P - %s %s: %s%s' % (pair, other, stats, v, '  <- the reading' if stats == 'none' else ''))
                for k, sd in enumerate(seeds):
                    if sd is None:
                        print('  seed %d: not read' % (666, 667)[k])
                        continue
                    s, r = sd
                    print('  seed %d s %s | r %s | near r %.3f far r %.3f' % (
                        (666, 667)[k], ' '.join('%+6.2f' % x for x in s), ' '.join('%+.3f' % x if x is not None else '-' for x in r),
                        m48.mean([r[i] for i in m48.NEAR]), m48.mean([r[i] for i in m48.FAR])))
    orc = ring_values(rs, 'oracle', 'O', 667, 'none')
    if orc is not None:
        print('\noracle 31818 per ring: %s' % ' '.join('%.2f' % x for x in orc))
        for part, pair in PAIRS.items():
            for seed in (666, 667):
                p = ring_values(rs, part, 'P', seed, 'none')
                if p is not None:
                    print('  %s P s%d / oracle: %s' % (pair, seed, ' '.join('%.2f' % (a / b) if b else '-' for a, b in zip(p, orc))))
    return out


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
