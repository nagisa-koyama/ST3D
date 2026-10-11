"""ANALYSIS (experiments_md 20261011_06): oracle box-error sensitivity for any class at several IoU thresholds.

Same perturbation arms as oracle_box_error_sensitivity.py (+ three small position arms for small objects), but the
evaluator is called once with a LIST of IoU thresholds (the IoU matrices are computed once), so one run gives the
class's AP at the KITTI bar (Pedestrian 0.5) and at looser bars. The dataset's own evaluation() / the PandaSet rule-A
scorer / the KITTI evaluator are reused unchanged; only `get_official_eval_result` is replaced by a thin wrapper around
`do_eval`. CPU only (Slurm CPU job).

    python analysis/oracle_box_error_multiiou.py --dataset waymo|nuscenes|pandaset|kitti --cls Pedestrian --thresholds 0.5,0.25,0.1 \
        --result R --label L --out O.jsonl [--cfg C] [--device 0 --cone 0] [--arms all] [--workers 8]
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent)); sys.path.insert(0, str(TOOLS / 'analysis'))
import oracle_box_error_sensitivity as B  # noqa: E402

CLS_IDX = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
EXTRA = {'lon0.1': ('pos_bias', {'axis': 'lon', 'm': 0.1}), 'lat0.1': ('pos_bias', {'axis': 'lat', 'm': 0.1}),
         'z0.1': ('pos_bias', {'axis': 'z', 'm': 0.1})}
_STATE = {}
LAST = {}


def make_wrapper(official, cls, thresholds):
    def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
        mo = np.zeros([len(thresholds), 3, 1]); mo[:, :, 0] = np.asarray(thresholds)[:, None]
        out = official.do_eval(gt_annos, dt_annos, [CLS_IDX[cls]], mo, False)
        bev, d3 = out[5], out[6]  # R40 [class, difficulty, threshold]
        ap = {}
        for k, t in enumerate(thresholds):
            for d, lvl in enumerate(('easy', 'moderate', 'hard')):
                ap[f'bev_{lvl}@{t}'] = float(bev[0, d, k]); ap[f'3d_{lvl}@{t}'] = float(d3[0, d, k])
        LAST.clear(); LAST.update(ap)
        first = thresholds[0]
        ap.update({f'{cls}_bev/moderate_R40': ap[f'bev_moderate@{first}'], f'{cls}_3d/moderate_R40': ap[f'3d_moderate@{first}']})
        return '', ap
    return types_namespace(get_official_eval_result, official)


def types_namespace(fn, official):
    import types
    ns = types.SimpleNamespace(get_official_eval_result=fn)
    return ns


def score_arm(name):
    kind, params = _STATE['arms'][name]
    t0 = time.time()
    dets = B.perturb_result(_STATE['dets'], kind, params)
    ds = _STATE['dataset_name']
    cls, th = _STATE['cls'], _STATE['thresholds']
    if ds in ('waymo', 'nuscenes'):
        _STATE['dataset'].evaluation(dets, [cls], eval_metric='kitti', output_path=None)
    elif ds == 'pandaset':
        _STATE['P'].score(_STATE['gt'], dets, _STATE['cone'], 'rule_A', _STATE['wrapper'], cls)
    else:  # kitti
        OK = _STATE['OK']
        if name != 'base':
            dets = [OK.rebuild_camera_fields(d, _STATE['calib'][d['frame_id']]) for d in dets]
        by_id = {d['frame_id']: d for d in dets}
        gt = [copy.deepcopy(i['annos']) for i in _STATE['infos']]
        dt = [copy.deepcopy(by_id[i['point_cloud']['lidar_idx']]) for i in _STATE['infos']]
        _STATE['wrapper'].get_official_eval_result(gt, dt, [cls])
    res = {'label': _STATE['label'], 'arm': name, 'cls': cls, 'seconds': round(time.time() - t0, 1)}
    res.update(LAST)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['waymo', 'nuscenes', 'pandaset', 'kitti'])
    ap.add_argument('--cls', default='Pedestrian')
    ap.add_argument('--thresholds', default='0.5,0.25,0.1')
    ap.add_argument('--result', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cfg', default=None)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--cone', type=int, default=0)
    ap.add_argument('--arms', default='all')
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()
    arms = B.build_arms(); arms.update(EXTRA)
    chosen = list(arms) if args.arms == 'all' else args.arms.split(',')
    assert all(c in arms for c in chosen)
    out = Path(args.out)
    done = {json.loads(l)['arm'] for l in out.read_text().splitlines()} if out.exists() else set()
    chosen = [c for c in chosen if c not in done]
    thresholds = [float(x) for x in args.thresholds.split(',')]
    print(f'{args.label} {args.cls}: {len(chosen)} arms to score ({len(done)} done), IoU {thresholds}', flush=True)
    dets = pickle.load(open(args.result, 'rb'))
    _STATE.update(arms=arms, dets=dets, dataset_name=args.dataset, cls=args.cls, thresholds=thresholds, label=args.label)
    if args.dataset in ('waymo', 'nuscenes'):
        import _init_path  # noqa: F401
        B.install_cpu_iou()
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import build_dataloader
        from pcdet.utils import common_utils
        import pcdet.datasets.kitti.kitti_object_eval_python.eval as E
        E.get_official_eval_result = make_wrapper(E, args.cls, thresholds).get_official_eval_result
        cfg_from_yaml_file(args.cfg, cfg)
        eval_cfg = cfg.DATA_CONFIG_TAR if cfg.get('DATA_CONFIG_TAR', None) else cfg.DATA_CONFIG
        test_set, _, _ = build_dataloader(dataset_cfg=eval_cfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                          logger=common_utils.create_logger('/dev/null', rank=0), training=False,
                                          model_ontology=cfg.get('ONTOLOGY', None))
        assert len(dets) == len(test_set.infos)
        _STATE['dataset'] = test_set
    elif args.dataset == 'pandaset':
        os.chdir(TOOLS)
        import pandaset_eval_protocols as P
        official = P.load_official_eval()
        _STATE.update(P=P, gt=P.gt_with_counts(args.device), cone=bool(args.cone), wrapper=make_wrapper(official, args.cls, thresholds))
        assert len(dets) == len(_STATE['gt'])
    else:
        import kitti_eval_cpu as K
        import oracle_box_error_kitti as OK
        from pcdet.utils import calibration_kitti
        official = K.load_official_eval()
        infos = pickle.load(open(K.INFOS, 'rb'))
        calib = {d['frame_id']: calibration_kitti.Calibration(OK.CALIB / f"{d['frame_id']}.txt") for d in dets}
        _STATE.update(OK=OK, infos=infos, calib=calib, wrapper=make_wrapper(official, args.cls, thresholds))
        assert len(dets) == len(infos)

    def record(r):
        with open(out, 'a') as f:
            f.write(json.dumps(r) + '\n')
        print(r['label'], f"{r['arm']:<10}", ' | '.join(f"@{t}: BEV {r[f'bev_moderate@{t}']:.1f} 3D {r[f'3d_moderate@{t}']:.1f}" for t in thresholds),
              f"({r['seconds']} s)", flush=True)

    if args.workers > 1:
        with mp.get_context('fork').Pool(args.workers) as pool:
            for r in pool.imap_unordered(score_arm, chosen):
                record(r)
    else:
        for c in chosen:
            record(score_arm(c))


if __name__ == '__main__':
    main()
