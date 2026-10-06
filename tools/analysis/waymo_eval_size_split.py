"""Re-score stored nuScenes -> Waymo predictions with large Vehicle GT (trucks / buses) IGNORED or removed.

Waymo's single 'Vehicle' class is scored as 'Car' here (waymo_dataset.kitti_eval), so its trucks and
buses are required of a model whose nuScenes source never labelled them as Car (nuScenes truck -> 'Truck',
bus -> 'Misc', both outside CLASS_NAMES, i.e. background). The user's hypothesis (2026-10-06): that is
what drags nuScenes -> Waymo (GBlobs + accumulation 51.64 against the Waymo oracle 65.92). Waymo has no
vehicle sub-type, so length is the proxy: Vehicle GT length p50 4.65 m, p90 5.53, p99 9.96; 10.2% are
longer than 5.5 m, 4.8% longer than 6 m, 2.2% longer than 7 m (val, every 5th frame).

Protocols (Car AP_R40 at IoU 0.7, the three difficulty columns identical on Waymo):
  as_scored     nothing changed (reproduces the logged AP: the validation of this tool)
  ign>L         GT longer than L m IGNORED (KITTI's mechanism: occluded = 3, beyond every difficulty, so an
                ignored box is neither required nor penalised and a detection matching it is not a FP)
  rm>L          GT longer than L m REMOVED, and predictions longer than L m removed too (both sides)
Run on the master node in the container (CPU; the rotated-IoU kernel is swapped as in kitti_eval_cpu.py):
    python analysis/waymo_eval_size_split.py --rows_json rows.json [--lengths 5.5 6 7]
rows.json = [[label, result.pkl], ...]. Analysis only: a manipulation of the evaluation GT, never a method.
"""
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent)); sys.path.insert(0, str(TOOLS / 'analysis'))
import _init_path  # noqa: F401,E402
from pcdet.datasets.kitti import kitti_utils  # noqa: E402
from kitti_eval_cpu import load_official_eval  # noqa: E402

INFOS = TOOLS.parent / 'data/waymo/waymo_infos_val.pkl'
MAP = {'Vehicle': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist', 'Sign': 'Sign', 'Car': 'Car'}


def score(gt_infos, dets, protocol, L, official):
    g_annos, d_annos = [], []
    for info, d in zip(gt_infos, dets):
        a = info['annos']
        g = {'name': a['name'].copy(), 'gt_boxes_lidar': a['gt_boxes_lidar'].copy(),
             'difficulty': a.get('difficulty', np.zeros(len(a['name']), int)).copy()}
        d = {k: copy.deepcopy(d[k]) for k in ('boxes_lidar', 'name', 'score')}
        length = g['gt_boxes_lidar'][:, 3] if len(g['name']) else np.zeros(0)
        big = (g['name'] == 'Vehicle') & (length > L)
        if protocol == 'rm':
            keep = ~big
            g = {k: v[keep] for k, v in g.items()}
            dk = ~((d['name'] == 'Car') & (d['boxes_lidar'][:, 3] > L)) if len(d['name']) else np.zeros(0, bool)
            d = {k: v[dk] for k, v in d.items()}
            ign = np.zeros(len(g['name']), bool)
        elif protocol == 'ign':
            ign = big
        else:
            ign = np.zeros(len(g['name']), bool)
        g['ignore'] = ign
        g_annos.append(g); d_annos.append(d)
    kitti_utils.transform_annotations_to_kitti_format(d_annos, map_name_to_kitti=MAP)
    kitti_utils.transform_annotations_to_kitti_format(g_annos, map_name_to_kitti=MAP, info_with_fakelidar=False)
    for g in g_annos:
        g['occluded'] = np.where(g.pop('ignore'), 3, 0)
    _, ap = official.get_official_eval_result(g_annos, d_annos, ['Car'])
    return ap['Car_bev/moderate_R40'], ap['Car_3d/moderate_R40']


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--rows_json', required=True)
    p.add_argument('--lengths', nargs='+', type=float, default=[5.5, 6.0, 7.0])
    p.add_argument('--protocols', nargs='+', default=['as_scored', 'ign', 'rm'])
    args = p.parse_args()
    rows = json.load(open(args.rows_json))
    infos = pickle.load(open(INFOS, 'rb'))
    by_id = {i['frame_id']: i for i in infos}
    official = load_official_eval()
    print('row | protocol | L | Car BEV / 3D moderate R40', flush=True)
    for label, pkl in rows:
        dets = pickle.load(open(pkl, 'rb'))
        gt = [by_id[d['frame_id']] for d in dets]
        for proto in args.protocols:
            for L in ([None] if proto == 'as_scored' else args.lengths):
                t = time.time()
                bev, d3 = score(gt, dets, proto, L if L is not None else 1e9, official)
                print(f'{label} | {proto} | {L} | {bev:.2f} / {d3:.2f}   ({time.time() - t:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
