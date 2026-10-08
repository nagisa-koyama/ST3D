"""Where the nuScenes-TARGET residual lives, and whether trucks / buses cost a Waymo-trained model (CPU).

Reads nuScenes-target eval `result.pkl` files and `nuscenes_infos_10sweeps_val.pkl`; no GPU, no retraining.
ANALYSIS ONLY: nuScenes val labels are read to diagnose, never to choose a setting
(memory/repo/uda_legality_rule.md).

Part 1 reuses waymo_target_residual's re-implementation of the family's protocol (KITTI evaluator, "all
objects" regime, Car only, BEV IoU 0.7, greedy matching by score, 40-point AP): per-range AP, recall by GT
attribute (range, points in box, length, azimuth), matched-box size errors. The points-in-box attribute is
nuScenes' own `num_lidar_pts` (keyframe sweep, every return, before the loader's ego removal), the same
population for every row. The pooled number should land within ~1.5 of the logged `Car AP_R40@0.70`.

Part 2 runs the OFFICIAL evaluator on CPU (kitti_eval_cpu.load_official_eval) under two protocols:
  as_scored    the family's nuScenes kitti_eval mapping (car -> Car, truck -> Truck, the rest -> Person_sitting,
               which the Car evaluation does not consider: a detection on a truck is a FALSE POSITIVE);
  veh_ignore   truck / bus / construction_vehicle / trailer GT become Car boxes with occluded = 3, i.e. KITTI's
               ignore mechanism (a detection matched to one is neither TP nor FP), as pandaset_eval_protocols does.
The difference is what a model that was trained with those classes as Car (Waymo's Vehicle) pays for firing on
them under nuScenes' narrower 'car'. A Lyft-trained model (Lyft labels trucks separately) is the control.

    python analysis/nuscenes_target_residual.py <label>=<result.pkl> [...] [--rings 0,10,20,30,40,50,75]
"""
import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent)); sys.path.insert(0, str(TOOLS / 'analysis'))
import _init_path  # noqa: F401,E402
import waymo_target_residual as W  # noqa: E402
from kitti_eval_cpu import load_official_eval  # noqa: E402

INFOS = TOOLS.parent / 'data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl'
OTHER_VEHICLES = ('truck', 'bus', 'construction_vehicle', 'trailer', 'emergency')
# nuscenes_dataset.kitti_eval's map; everything else becomes 'Person_sitting', which the Car evaluation ignores
# in the sense of "not a GT at all" (clean_data valid_class -1): a detection on it stays a false positive.
MAP_AS_SCORED = {'car': 'Car', 'Car': 'Car', 'pedestrian': 'Pedestrian', 'Pedestrian': 'Pedestrian', 'truck': 'Truck',
                 'motorcycle': 'Cyclist', 'bicycle': 'Cyclist', 'Cyclist': 'Cyclist'}


def frame_id(info):
    return Path(info['lidar_path']).stem          # '<...>__LIDAR_TOP__<ts>.pcd', as result.pkl stores it


def load_infos():
    return pickle.load(open(INFOS, 'rb'))


def gt_for_breakdown(infos):
    gt = {}
    for inf in infos:
        m = inf['gt_names'] == 'car'
        gt[frame_id(inf)] = dict(boxes=np.asarray(inf['gt_boxes'])[m, :7].astype(np.float32),
                                 npts=np.asarray(inf['num_lidar_pts'])[m])
    return gt


def to_kitti(annos, is_gt):
    """nuscenes_dataset.kitti_eval.transform_to_kitti_format, with an optional per-box 'ignore' for GT."""
    for anno in annos:
        names = anno.pop('gt_names') if 'gt_names' in anno else anno['name']
        anno['name'] = np.array([MAP_AS_SCORED.get(n, 'Person_sitting') if not (is_gt and ig) else 'Car'
                                 for n, ig in zip(names, anno.pop('ignore', np.zeros(len(names), bool)))])
        b = (anno['boxes_lidar'] if 'boxes_lidar' in anno else anno['gt_boxes']).copy()[:, :7]
        anno['bbox'] = np.zeros((len(anno['name']), 4)); anno['bbox'][:, 2:4] = 50
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = anno.pop('occluded', np.zeros(len(anno['name'])))
        if len(b):
            b[:, 2] -= b[:, 5] / 2
            anno['location'] = np.stack([-b[:, 1], -b[:, 2], b[:, 0]], 1)
            anno['dimensions'] = b[:, 3:6][:, [0, 2, 1]]
            anno['rotation_y'] = -b[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-b[:, 1], b[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3)); anno['rotation_y'] = anno['alpha'] = np.zeros(0)
    return annos


def official_scores(infos, path, official):
    dets = {d['frame_id']: d for d in pickle.load(open(path, 'rb'))}
    out = {}
    for protocol in ('as_scored', 'veh_ignore'):
        g_annos, d_annos = [], []
        for inf in infos:
            names = np.asarray(inf['gt_names'])
            ign = np.isin(names, OTHER_VEHICLES) if protocol == 'veh_ignore' else np.zeros(len(names), bool)
            g_annos.append({'gt_boxes': np.asarray(inf['gt_boxes'])[:, :7].copy(), 'gt_names': names.copy(),
                            'ignore': ign, 'occluded': np.where(ign, 3, 0)})
            d = dets[frame_id(inf)]
            d_annos.append({k: copy.deepcopy(d[k]) for k in ('boxes_lidar', 'name', 'score')})
        _, ap = official.get_official_eval_result(to_kitti(g_annos, True), to_kitti(d_annos, False), ['Car'])
        out[protocol] = (ap['Car_bev/moderate_R40'], ap['Car_3d/moderate_R40'])
    return out


def fp_on_other_vehicles(infos, path, thr=0.3):
    """Confident (>= SCORE_MIN) Car predictions unmatched to a car GT at 0.7 that overlap a truck-class GT (BEV IoU >= thr)."""
    dets = {d['frame_id']: d for d in pickle.load(open(path, 'rb'))}
    n_unm = n_on = n_other = 0
    for inf in infos:
        names = np.asarray(inf['gt_names']); gb = np.asarray(inf['gt_boxes'])[:, :7].astype(np.float32)
        d = dets[frame_id(inf)]; m = (d['name'] == 'Car') & (d['score'] >= W.SCORE_MIN)
        db = d['boxes_lidar'][m].astype(np.float32); ds = d['score'][m]
        dt_m, _ = W.greedy_match(gb[names == 'car'], db, ds, 0.7)
        unm = db[dt_m < 0]; other = gb[np.isin(names, OTHER_VEHICLES)]
        n_unm += len(unm); n_other += len(other)
        if len(unm) and len(other):
            n_on += int((W.bev_iou(unm, other).max(1) >= thr).sum())
    return n_unm, n_on, n_other


def report(outs, rings):
    print('\n## Pooled Car BEV AP_R40 (re-implementation; compare with the logged value)')
    print('| model | IoU 0.7 | IoU 0.5 | confident (>= %.1f) predictions | of which unmatched at 0.7 |' % W.SCORE_MIN); print('|---|---|---|---|---|')
    for o in outs:
        print('| %s | %.2f | %.2f | %d | %.1f%% |' % (o['label'], o['ap_0.7'], o['ap_0.5'], o['dt_conf'], 100 * o['fp_conf'] / max(o['dt_conf'], 1)))
    print('\n## Car BEV AP_R40 at IoU 0.7 per ring of GT / prediction centre range (m)')
    hdr = ['%g-%g' % (rings[k], rings[k + 1]) for k in range(len(rings) - 1)]
    print('| model | ' + ' | '.join(hdr) + ' |'); print('|---' * (len(hdr) + 1) + '|')
    for o in outs:
        print('| %s | ' % o['label'] + ' | '.join('%.1f' % v for v in o['ring_ap']) + ' |')
    print('| GT boxes | ' + ' | '.join(str(n) for n in outs[0]['ring_n']) + ' |')
    print('\n## Recall of Car GT at score >= %.1f, BEV IoU 0.7, by GT attribute (hits / GT)' % W.SCORE_MIN)
    for key, bins, name in [('ring', hdr, 'range'), ('len', ['<4.5 m', '4.5-6 m', '>6 m'], 'GT length'),
                            ('pts', ['0 pts', '1-9', '10-49', '50-199', '>=200'], 'points in GT (num_lidar_pts, keyframe)'),
                            ('az', ['front +-45', 'sides', 'rear'], 'azimuth')]:
        print('\n%s: | model | ' % name + ' | '.join(bins) + ' |'); print('|---' * (len(bins) + 1) + '|')
        for o in outs:
            print('| %s | ' % o['label'] + ' | '.join('%.2f' % (h / max(n, 1)) for h, n in o['att'][key]) + ' |')
        print('| GT | ' + ' | '.join('%d' % n for _, n in outs[0]['att'][key]) + ' |')
    print('\n## Matched boxes (score >= %.1f, BEV IoU >= 0.5): median errors and what fixing one term buys in 3D' % W.SCORE_MIN)
    print('| model | matched | dl | dw | dh | dz bottom | BEV IoU | 3D IoU | share 3D >= 0.7 | size set to GT | bottom set to GT |'); print('|---' * 11 + '|')
    for o in outs:
        e = o['errs']
        print('| %s | %d | %+.2f | %+.2f | %+.2f | %+.2f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (
            o['label'], len(e), np.median(e[:, 1]), np.median(e[:, 2]), np.median(e[:, 3]), np.median(e[:, 4]),
            np.median(e[:, 5]), np.median(e[:, 6]), (e[:, 6] >= 0.7).mean(), (e[:, 7] >= 0.7).mean(), (e[:, 8] >= 0.7).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs', nargs='+'); ap.add_argument('--rings', default='0,10,20,30,40,50,75')
    ap.add_argument('--no-official', action='store_true'); ap.add_argument('--official-only', action='store_true')
    a = ap.parse_args(); rings = [float(x) for x in a.rings.split(',')]
    infos = load_infos(); gt = gt_for_breakdown(infos)
    runs = [r.split('=', 1) for r in a.runs]
    if not a.official_only:
        outs = [W.analyse(label, path, gt=gt, rings=rings) for label, path in runs]
        report(outs, rings)
    if a.no_official:
        return
    official = load_official_eval()
    print('\n## Official evaluator (CPU): truck / bus / construction_vehicle / trailer GT as IGNORE regions')
    print('| model | as scored BEV / 3D | other-vehicle GT ignored BEV / 3D | gain | confident unmatched Car dets | on an other-vehicle GT (IoU >= 0.3) |')
    print('|---|---|---|---|---|---|')
    for label, path in runs:
        s = official_scores(infos, path, official); n_unm, n_on, n_other = fp_on_other_vehicles(infos, path)
        print('| %s | %.2f / %.2f | %.2f / %.2f | %+.2f / %+.2f | %d | %d (%.1f%%) |' % (
            label, *s['as_scored'], *s['veh_ignore'], s['veh_ignore'][0] - s['as_scored'][0], s['veh_ignore'][1] - s['as_scored'][1],
            n_unm, n_on, 100 * n_on / max(n_unm, 1)))
    print('other-vehicle GT boxes in val: %d' % n_other)


if __name__ == '__main__':
    main()
