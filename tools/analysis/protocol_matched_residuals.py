"""Residual to the oracle under MATCHED evaluation protocols, per target (experiments_md/20261008_04). ANALYSIS.

Hypothesis (e): every pair's residual sits in sparse and far boxes, and only the nuScenes / Waymo targets score
them. A KITTI-target row is scored at KITTI moderate (2D height >= 25 px, i.e. about <= 43 m for a 1.5 m-tall car
at f = 721 px; occlusion <= 1; truncation <= 0.3; camera FOV). A nuScenes / Waymo target is scored "all objects"
(fabricated 2D box, nothing ignored). This tool re-scores stored predictions with the OFFICIAL KITTI evaluator on
CPU (`kitti_eval_cpu.load_official_eval`, only the IoU kernel swapped) under:

  as_scored  the logged protocol, reproduced first (KITTI: moderate; nuScenes / Waymo: all objects)
  P1         "moderate-like": KITTI = official moderate; nuScenes / Waymo = GT centre beyond 43 m IGNORED (occluded
             = 3, KITTI's mechanism: not deleted, a matching detection is neither TP nor FP), predictions centred
             beyond 43 m removed. Pre-declared primary.
  P1_50      as P1 at 50 m (sensitivity)
  P1_pts10   as P1, and GT with fewer than 10 stored points also ignored (sensitivity; nuScenes `num_lidar_pts`,
             Waymo `num_points_in_gt` - different definitions, see the report)
  P2         "all objects": nuScenes / Waymo = as scored; KITTI = every labelled Car scored (no height / occlusion /
             truncation filter, DontCare and Van still ignored as the evaluator does, camera-FOV labels and the
             FOV prediction filter as stored)
  ring_a_b   protocol-free per-ring view: GT outside [a, b) m ignored, predictions outside removed; KITTI uses the
             P2 ("all") difficulty so its far ring exists at all

Distance is the box centre's ground-plane range from the sensor (nuScenes / Waymo lidar frame; KITTI camera frame,
hypot(x, z), ~0.27 m from the lidar). Car only, AP_R40 at IoU 0.7, the moderate column (all three columns are
identical on nuScenes / Waymo targets).

Target labels are read to EVALUATE and DIAGNOSE only, never to choose a setting.

    python analysis/protocol_matched_residuals.py run --group kitti|nuscenes|waymo|all --workers 10 --out DIR
    python analysis/protocol_matched_residuals.py report --out DIR
"""
import argparse
import copy
import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent)); sys.path.insert(0, str(TOOLS / 'analysis'))

W = '/home/koyama/data/wandb/'
O = str(TOOLS.parent / 'output' / 'da-ieee-access') + '/'
DATA = TOOLS.parent / 'data'

# pair, label, job, W&B run, target, result.pkl, logged Car BEV / 3D (moderate R40)
ROWS = [
    ('S2 nuScenes->KITTI', 'oracle (z-shift, ceiling)', '26843', '06iwywhs', 'kitti',
     W + 'run-20261001_005323-06iwywhs/files/eval/eval_with_train/epoch_152/val/result.pkl', (80.33, 70.59)),
    ('S2 nuScenes->KITTI', 'control, 1 sweep', '26265', 'frpjh5ah', 'kitti',
     O + 'centerpoint-sourceonly-nuscenes2kitti/20260922_sourceonly/eval/epoch_20/val/control_no_accum_no_corr/result.pkl',
     (41.57, 22.58)),
    ('S2 nuScenes->KITTI', 'accumulation 15/10 (legal teacher)', '27468', 'o3l96a3p', 'kitti',
     W + 'run-20261006_004144-o3l96a3p/files/eval/eval_with_train/epoch_20/val/result.pkl', (73.70, 46.82)),
    ('S2 nuScenes->KITTI', 'GBlobs + accumulation', '27471', 'jguwly5u', 'kitti',
     W + 'run-20261006_024401-jguwly5u/files/eval/eval_with_train/epoch_20/val/result.pkl', (75.67, 47.93)),
    ('S2 nuScenes->KITTI', '+ pseudo-labelling (27468 teacher)', '27566', '2tjw5vuv', 'kitti',
     W + 'run-20261006_223005-2tjw5vuv/files/eval/eval_with_train/epoch_20/val/result.pkl', (76.66, 48.25)),
    ('S2 nuScenes->KITTI', '+ pseudo-labelling (GBlobs teacher), best legal', '27612', '2r2evjtv', 'kitti',
     W + 'run-20261007_162630-2r2evjtv/files/eval/eval_with_train/epoch_20/val/result.pkl', (78.09, 47.42)),

    ('nuScenes->Waymo', 'oracle (Waymo -> Waymo)', '27179', '1v6bbjhc', 'waymo',
     O + 'centerpoint-sourceonly-waymo2waymo/20260922_sourceonly/eval/epoch_7/val/waymo_eval/result.pkl', (65.92, 51.77)),
    ('nuScenes->Waymo', 'control, 1 sweep', '27178', '4shhqj77', 'waymo',
     O + 'centerpoint-sourceonly-nuscenes2waymo/20260922_sourceonly/eval/epoch_20/val/waymo_eval/result.pkl', (2.20, 0.84)),
    ('nuScenes->Waymo', 'accumulation 15/10 (27468)', '27590', 'd966jeko', 'waymo',
     O + 'centerpoint-accum-legaldepth-nuscenes2waymo/20260922_sourceonly/eval/epoch_20/val/waymo_eval/result.pkl',
     (48.30, 23.47)),
    ('nuScenes->Waymo', 'displacement-spread accumulation (27705)', '27802', 'twdm3q5p', 'waymo',
     O + 'centerpoint-accum-legaldepth-disp-nuscenes2waymo/20261008_disp/eval/epoch_20/val/waymo_ep20/result.pkl',
     (47.17, 23.85)),
    ('nuScenes->Waymo', 'GBlobs + accumulation (27471)', '27596', '416ymjgg', 'waymo',
     O + 'centerpoint-gblobs-sourceonly-nuscenes2waymo/20260922_sourceonly/eval/epoch_20/val/'
         'gblobs_accum_legaldepth_on_waymo/result.pkl', (51.58, 28.35)),
    ('nuScenes->Waymo', '+ pseudo-labelling on Waymo (27704), best legal', '27803', '5hzcs4e4', 'waymo',
     O + 'centerpoint-gblobs-accum-legaldepth-st3d-nuscenes2waymo/legal_waymo_pl/eval/epoch_20/val/waymo_ep20/result.pkl',
     (55.66, 32.82)),

    ('nuScenes target', 'oracle (nuScenes -> nuScenes)', '25768', '0vzzq1rn', 'nuscenes',
     O + 'centerpoint-sourceonly-nuscenes/20260922_sourceonly/eval/epoch_20/val/recover_ep20/result.pkl', (44.98, 26.63)),
    ('S1 Lyft->nuScenes', 'control', '25835', 'ldb35c2o', 'nuscenes',
     W + 'run-20260924_111041-ldb35c2o/files/eval/eval_with_train/epoch_30/val/result.pkl', (23.17, 10.16)),
    ('S1 Lyft->nuScenes', 'GBlobs', '25723', 'qyerdcln', 'nuscenes',
     W + 'run-20260922_131445-qyerdcln/files/eval/eval_with_train/epoch_30/val/result.pkl', (31.43, 16.30)),
    ('S1 Lyft->nuScenes', 'AdaBN teacher + DSNorm pseudo-labelling, best legal', '27594', 'lubwlxht', 'nuscenes',
     W + 'run-20261007_073442-lubwlxht/files/eval/eval_with_train/epoch_30/val/result.pkl', (33.39, 15.43)),
    ('Waymo->nuScenes', 'control', '26398', 'y6p61n0x', 'nuscenes',
     W + 'run-20260928_082639-y6p61n0x/files/eval/eval_with_train/epoch_7/val/result.pkl', (27.67, 12.28)),
    ('Waymo->nuScenes', 'per-bin thinning, best legal', '27599', 'puo3m3qt', 'nuscenes',
     W + 'run-20261007_074706-puo3m3qt/files/eval/eval_with_train/epoch_7/val/result.pkl', (30.63, 13.52)),
    ('Waymo->nuScenes', 'GBlobs', '27598', '6zzdwr6f', 'nuscenes',
     W + 'run-20261007_073836-6zzdwr6f/files/eval/eval_with_train/epoch_7/val/result.pkl', (28.86, 15.16)),
]
RINGS = [(0, 20), (20, 43), (43, 75)]
PROTOCOLS = ['as_scored', 'P1', 'P1_50', 'P1_pts10', 'P2'] + [f'ring_{a}_{b}' for a, b in RINGS]

# KITTI evaluator difficulty thresholds: (MIN_HEIGHT, MAX_OCCLUSION, MAX_TRUNCATION) per difficulty index
OFFICIAL = ([40, 25, 25], [0, 1, 2], [0.15, 0.3, 0.5])
ALL = ([-1, -1, -1], [3, 3, 3], [1.0, 1.0, 1.0])  # every labelled box; occluded 4 is our ignore marker there

_OFFICIAL_EVAL = None
_GT = {}


def official_eval():
    global _OFFICIAL_EVAL
    if _OFFICIAL_EVAL is None:
        import _init_path  # noqa: F401
        from kitti_eval_cpu import load_official_eval
        _OFFICIAL_EVAL = load_official_eval()
        _OFFICIAL_EVAL._orig_clean_data = _OFFICIAL_EVAL.clean_data
    return _OFFICIAL_EVAL


def set_thresholds(mod, th):
    """Swap eval.py's clean_data for one with these thresholds (eval.py looks the name up at call time)."""
    min_h, max_occ, max_trunc = th

    def clean_data(gt_anno, dt_anno, current_class, difficulty):
        CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']
        dc_bboxes, ignored_gt, ignored_dt = [], [], []
        current_cls_name = CLASS_NAMES[current_class].lower()
        num_valid_gt = 0
        for i in range(len(gt_anno['name'])):
            bbox = gt_anno['bbox'][i]
            gt_name = gt_anno['name'][i].lower()
            height = bbox[3] - bbox[1]
            if gt_name == current_cls_name:
                valid_class = 1
            elif current_cls_name == 'pedestrian' and gt_name == 'person_sitting':
                valid_class = 0
            elif current_cls_name == 'car' and gt_name == 'van':
                valid_class = 0
            else:
                valid_class = -1
            ignore = (gt_anno['occluded'][i] > max_occ[difficulty] or gt_anno['truncated'][i] > max_trunc[difficulty]
                      or height <= min_h[difficulty])
            if valid_class == 1 and not ignore:
                ignored_gt.append(0); num_valid_gt += 1
            elif valid_class == 0 or (ignore and valid_class == 1):
                ignored_gt.append(1)
            else:
                ignored_gt.append(-1)
            if gt_anno['name'][i] == 'DontCare':
                dc_bboxes.append(gt_anno['bbox'][i])
        for i in range(len(dt_anno['name'])):
            valid_class = 1 if dt_anno['name'][i].lower() == current_cls_name else -1
            height = abs(dt_anno['bbox'][i, 3] - dt_anno['bbox'][i, 1])
            if height < min_h[difficulty]:
                ignored_dt.append(1)
            elif valid_class == 1:
                ignored_dt.append(0)
            else:
                ignored_dt.append(-1)
        return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes

    mod.clean_data = mod._orig_clean_data if th is OFFICIAL else clean_data


def subset(anno, keep):
    n = len(keep)
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in anno.items()}


# ---------------------------------------------------------------- per-target GT and KITTI-format conversion

def kitti_gt(dets):
    infos = {i['point_cloud']['lidar_idx']: i for i in pickle.load(open(DATA / 'kitti/kitti_infos_val.pkl', 'rb'))}
    gt = [copy.deepcopy(infos[d['frame_id']]['annos']) for d in dets]
    for g in gt:
        g['_range'] = np.hypot(g['location'][:, 0], g['location'][:, 2])
        g['_pts'] = np.full(len(g['name']), 10 ** 6)
    return gt


def kitti_dets(dets):
    out = []
    for d in dets:
        d = copy.deepcopy(d)
        d['_range'] = np.hypot(d['location'][:, 0], d['location'][:, 2]) if len(d['name']) else np.zeros(0)
        out.append(d)
    return out


def waymo_gt_and_dets(dets):
    from pcdet.datasets.kitti import kitti_utils
    if 'waymo' not in _GT:
        infos = pickle.load(open(DATA / 'waymo/waymo_infos_val.pkl', 'rb'))[::5]  # SAMPLED_INTERVAL test 5
        _GT['waymo'] = infos
    infos = _GT['waymo']
    assert len(infos) == len(dets) and all(i['frame_id'] == d['frame_id'] for i, d in zip(infos, dets)), \
        'Waymo predictions are not in the infos order'
    gt = [copy.deepcopy(i['annos']) for i in infos]
    dt = copy.deepcopy(dets)
    m = {'Vehicle': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist', 'Sign': 'Sign', 'Car': 'Car'}
    kitti_utils.transform_annotations_to_kitti_format(dt, map_name_to_kitti=m)
    kitti_utils.transform_annotations_to_kitti_format(gt, map_name_to_kitti=m, info_with_fakelidar=False)
    for g in gt:
        b = g['gt_boxes_lidar']; g['_range'] = np.hypot(b[:, 0], b[:, 1]) if len(b) else np.zeros(0)
        g['_pts'] = np.asarray(g['num_points_in_gt'])
    for d in dt:
        b = d['boxes_lidar']; d['_range'] = np.hypot(b[:, 0], b[:, 1]) if len(b) else np.zeros(0)
    return gt, dt


def nuscenes_kitti_format(annos, is_gt):
    """nuscenes_dataset.kitti_eval's transform, verbatim except the inert GT_FILTER branch."""
    m = {'car': 'Car', 'Car': 'Car', 'pedestrian': 'Pedestrian', 'Pedestrian': 'Pedestrian', 'truck': 'Truck',
         'motorcycle': 'Cyclist', 'bicycle': 'Cyclist', 'Cyclist': 'Cyclist'}
    for anno in annos:
        if 'name' not in anno:
            anno['name'] = anno.pop('gt_names')
        anno['name'] = np.array([m.get(n, 'Person_sitting') for n in anno['name']])
        b = (anno['boxes_lidar'] if 'boxes_lidar' in anno else anno['gt_boxes']).copy()
        anno['_range'] = np.hypot(b[:, 0], b[:, 1]) if len(b) else np.zeros(0)
        n = len(anno['name'])
        anno['bbox'] = np.zeros((n, 4)); anno['bbox'][:, 2:4] = 50
        anno['truncated'] = np.zeros(n); anno['occluded'] = np.zeros(n)
        if len(b):
            b[:, 2] -= b[:, 5] / 2
            anno['location'] = np.stack([-b[:, 1], -b[:, 2], b[:, 0]], 1)
            anno['dimensions'] = b[:, 3:6][:, [0, 2, 1]]
            anno['rotation_y'] = -b[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-b[:, 1], b[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)


def nuscenes_gt_and_dets(dets):
    if 'nuscenes' not in _GT:
        _GT['nuscenes'] = pickle.load(open(DATA / 'nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl', 'rb'))
    infos = _GT['nuscenes']
    assert len(infos) == len(dets) and all(i['token'] == d['metadata']['token'] for i, d in zip(infos, dets)), \
        'nuScenes predictions are not in the infos order'
    gt = [copy.deepcopy(i) for i in infos]
    dt = copy.deepcopy(dets)
    nuscenes_kitti_format(dt, False)
    nuscenes_kitti_format(gt, True)
    for g, i in zip(gt, infos):
        g['_pts'] = np.asarray(i['num_lidar_pts'])
    return gt, dt


def prepare(target, path):
    dets = pickle.load(open(path, 'rb'))
    if target == 'kitti':
        return kitti_gt(dets), kitti_dets(dets)
    if target == 'waymo':
        return waymo_gt_and_dets(dets)
    return nuscenes_gt_and_dets(dets)


# ---------------------------------------------------------------- protocols

def apply_protocol(target, protocol, gt, dt):
    """(gt, dt, thresholds) for one protocol; GT is ignored by occlusion, predictions are removed."""
    if protocol in ('as_scored', 'P1', 'P1_50', 'P1_pts10') and target == 'kitti':
        return gt, dt, OFFICIAL  # KITTI moderate is the logged protocol and P1 by definition
    if protocol == 'P2':
        if target != 'kitti':
            return gt, dt, OFFICIAL  # all objects already
        gt = [dict(g, occluded=np.where(g['name'] == 'DontCare', g['occluded'], 0)) for g in gt]
        return gt, dt, ALL
    if protocol == 'as_scored':
        return gt, dt, OFFICIAL
    if protocol.startswith('P1'):
        r = 50.0 if protocol == 'P1_50' else 43.0
        lo, hi, min_pts = 0.0, r, (10 if protocol == 'P1_pts10' else 0)
        th, marker = OFFICIAL, 3
    else:
        _, a, b = protocol.split('_')
        lo, hi, min_pts = float(a), float(b), 0
        th, marker = (ALL, 4) if target == 'kitti' else (OFFICIAL, 3)
    g2, d2 = [], []
    for g, d in zip(gt, dt):
        g = dict(g)
        out = (g['_range'] < lo) | (g['_range'] >= hi) | (g['_pts'] < min_pts)
        occ = g['occluded'].copy()
        if target == 'kitti':
            dc = g['name'] == 'DontCare'
            occ = np.where(dc, occ, 0)  # the "all" difficulty for the ring view
            occ = np.where(out & ~dc, marker, occ)
        else:
            occ = np.where(out, marker, occ)
        g['occluded'] = occ
        keep = (d['_range'] >= lo) & (d['_range'] < hi)
        g2.append(g); d2.append(subset(d, keep))
    return g2, d2, th


def score_task(task):
    pair, label, job, wandb, target, path, logged = task['row']
    out_json = Path(task['out']) / f"{job}_{task['protocol']}.json"
    if out_json.exists():
        return json.load(open(out_json))
    os.chdir(TOOLS)
    mod = official_eval()
    gt, dt = prepare(target, path)
    g, d, th = apply_protocol(target, task['protocol'], gt, dt)
    set_thresholds(mod, th)
    strip = lambda a: {k: v for k, v in a.items() if not k.startswith('_')}
    _, ap = mod.get_official_eval_result([strip(x) for x in g], [strip(x) for x in d], ['Car'])
    res = {'job': job, 'protocol': task['protocol'], 'bev': ap['Car_bev/moderate_R40'], 'd3': ap['Car_3d/moderate_R40'],
           'n_gt_scored': int(sum(((x['name'] == 'Car') & (x['occluded'] <= (1 if th is OFFICIAL else 3))).sum()
                                  for x in g))}
    json.dump(res, open(out_json, 'w'))
    print(f"{pair:20s} {job} {task['protocol']:12s} BEV {res['bev']:6.2f} 3D {res['d3']:6.2f}", flush=True)
    return res


def canonical(target, protocol):
    """KITTI's P1 variants ARE its official moderate; nuScenes / Waymo's P2 IS their as-scored number."""
    if target == 'kitti' and protocol.startswith('P1'):
        return 'as_scored'
    if target != 'kitti' and protocol == 'P2':
        return 'as_scored'
    return protocol


def tasks_for(group, out):
    rows = [r for r in ROWS if group == 'all' or r[4] == group]
    return [{'row': r, 'protocol': p, 'out': out} for r in rows for p in PROTOCOLS if canonical(r[4], p) == p]


def report(out):
    res = {}
    for f in Path(out).glob('*.json'):
        r = json.load(open(f)); res[(r['job'], r['protocol'])] = r
    for pair, label, job, wandb, target, path, logged in ROWS:
        for p in PROTOCOLS:
            if (job, p) not in res and (job, canonical(target, p)) in res:
                res[(job, p)] = res[(job, canonical(target, p))]
    print('## Reproduction (as_scored vs logged, Car BEV / 3D)')
    for pair, label, job, wandb, target, path, logged in ROWS:
        r = res.get((job, 'as_scored'))
        if r:
            print(f'| {pair} | {label} | {job} | {logged[0]:.2f} / {logged[1]:.2f} | {r["bev"]:.2f} / {r["d3"]:.2f} | '
                  f'{r["bev"] - logged[0]:+.2f} / {r["d3"] - logged[1]:+.2f} | {wandb} |')
    print('\n## All protocols, Car BEV / 3D')
    print('| pair | row | job | ' + ' | '.join(PROTOCOLS) + ' | W&B |')
    for pair, label, job, wandb, target, path, logged in ROWS:
        cells = []
        for p in PROTOCOLS:
            r = res.get((job, p))
            cells.append(f'{r["bev"]:.2f} / {r["d3"]:.2f}' if r else '-')
        print(f'| {pair} | {label} | {job} | ' + ' | '.join(cells) + f' | {wandb} |')
    print('\n## GT scored (Car, not ignored) per protocol')
    for pair, label, job, wandb, target, path, logged in ROWS:
        if 'oracle' in label:
            print(f'| {pair} | ' + ' | '.join(f'{p}: {res[(job, p)]["n_gt_scored"]}' for p in PROTOCOLS
                                             if (job, p) in res) + ' |')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'report'])
    ap.add_argument('--group', default='all', choices=['all', 'kitti', 'waymo', 'nuscenes'])
    ap.add_argument('--only_protocols', nargs='*', default=None)
    ap.add_argument('--only_jobs', nargs='*', default=None)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.mode == 'report':
        return report(a.out)
    tasks = tasks_for(a.group, a.out)
    if a.only_protocols:
        tasks = [t for t in tasks if t['protocol'] in a.only_protocols]
    if a.only_jobs:
        tasks = [t for t in tasks if t['row'][2] in a.only_jobs]
    missing = [t['row'][5] for t in tasks if not os.path.exists(t['row'][5])]
    assert not missing, f'missing result.pkl: {sorted(set(missing))}'
    # largest first, so a long Waymo task does not start last
    tasks.sort(key=lambda t: {'waymo': 0, 'nuscenes': 1, 'kitti': 2}[t['row'][4]])
    if a.workers > 1:
        with Pool(a.workers, maxtasksperchild=4) as pool:
            for _ in pool.imap_unordered(score_task, tasks):
                pass
    else:
        for t in tasks:
            score_task(t)


if __name__ == '__main__':
    main()
