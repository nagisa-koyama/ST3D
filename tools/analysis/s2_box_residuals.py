"""Box residuals of KITTI-target Car predictions, and an isotropic post-hoc rescale scored by the official evaluator.

Why (experiments_md 20261003_05 §5 item 42b, report 20261108_06): the S2 method row (nuScenes -> KITTI, jobs 27566 /
28283) is level with the oracle in BEV but ~22 AP below it in 3D at IoU 0.7. This measures, on the saved KITTI-val
predictions, how the matched boxes differ from their GT in size and height, and re-scores the same predictions with
every box scaled isotropically about its centre (by 1 / 1.165, the R6 ratio) to see whether the 3D gap is a box-scale
term. ANALYSIS: it reads KITTI val labels to score; nothing here selects or feeds a method.

Matching rule (stated in the report): per frame, the GT set is the official evaluator's moderate-valid Car GT, i.e.
`clean_data(gt, dt, class 0 = Car, difficulty 1 = moderate)` returns ignored_gt == 0 (2D height > 25 px, occlusion
<= 1, truncation <= 0.3; easy boxes included, Van / DontCare excluded). The predictions are every Car prediction in
result.pkl (any score, as scored: the in-run evaluation already applied SCORE_THRESH and, for a KITTI target, the
camera-FOV filter, and removed SHIFT_COOR). BEV IoU is pcdet's rotated BEV IoU on the KITTI LiDAR boxes
(`boxes_lidar` for predictions, `gt_boxes_lidar` for GT). Pairs are taken one-to-one, greedily by highest IoU first,
and kept when BEV IoU >= 0.5.

Residuals per matched pair, in the KITTI LiDAR frame (z up; box z is the centre): l, w, h ratios predicted / GT; the
z-centre offset z_p - z_g; the box-bottom offset (z_p - h_p / 2) - (z_g - h_g / 2), metres. Reported as median and
IQR (25th / 75th percentiles).

Rescoring arms (Car AP_R40, moderate, IoU 0.7 and 0.5, BEV and 3D, the official evaluator on CPU via
kitti_eval_cpu.load_official_eval):
- `base`: the predictions as saved (must reproduce the logged AP; the CPU kernel matches BEV to the printed digit).
- `iso_centre`: l, w, h all x s about the box centre. KITTI's camera `location` is the box BOTTOM centre (y down), so
  it moves by (h - h') / 2; `boxes_lidar` keeps its centre. The 2D `bbox` is re-projected from the scaled 3D box with
  the frame's calibration (as generate_prediction_dicts builds it): it enters the evaluator only through the
  detection's 2D height in the difficulty filter.
- `iso_centre_bbox_kept`: as `iso_centre` with the saved 2D bbox (the posthoc_box_scale.py convention), beside.
- `iso_bottom`: l, w, h all x s with the box BOTTOM kept fixed (camera `location` unchanged; LiDAR centre z moves
  down by (h - h') / 2), bbox re-projected. Optional extra.

    python analysis/s2_box_residuals.py --out <json> NAME=<result.pkl> [NAME=<result.pkl> ...] [--scale 0.858369]
"""
import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_eval_cpu import INFOS, TOOLS, load_official_eval  # noqa: E402  (imports _init_path first)
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu  # noqa: E402

CALIB_DIR = TOOLS.parent / 'data/kitti/training/calib'
MATCH_IOU = 0.5
R6_RATIO = 1.165


def match_one_to_one(iou, thresh=MATCH_IOU):
    """Greedy one-to-one matching by highest IoU first: list of (pred index, gt index), IoU >= thresh."""
    iou = np.array(iou, dtype=np.float64, copy=True)
    pairs = []
    if iou.size == 0:
        return pairs
    while True:
        p, g = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[p, g] < thresh:
            break
        pairs.append((int(p), int(g)))
        iou[p, :] = -1.0
        iou[:, g] = -1.0
    return pairs


def pair_residuals(pred, gt):
    """Residuals of matched (N, 7) LiDAR boxes (x, y, z_centre, l, w, h, heading): dict of (N,) arrays."""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    return {
        'l_ratio': pred[:, 3] / gt[:, 3],
        'w_ratio': pred[:, 4] / gt[:, 4],
        'h_ratio': pred[:, 5] / gt[:, 5],
        'dz_centre': pred[:, 2] - gt[:, 2],
        'dz_bottom': (pred[:, 2] - pred[:, 5] / 2.0) - (gt[:, 2] - gt[:, 5] / 2.0),
    }


def summarise(values):
    v = np.asarray(values, np.float64)
    if len(v) == 0:
        return {'median': None, 'q25': None, 'q75': None, 'iqr': None}
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return {'median': float(med), 'q25': float(q25), 'q75': float(q75), 'iqr': float(q75 - q25)}


def valid_car_gt_index(official, gt_anno, dt_anno):
    """Indices (into gt_boxes_lidar) of the evaluator's moderate-valid Car GT."""
    _, ignored_gt, _, _ = official.clean_data(gt_anno, dt_anno, 0, 1)
    ignored_gt = np.asarray(ignored_gt)
    n = len(gt_anno['gt_boxes_lidar'])
    assert all(nm == 'DontCare' for nm in np.asarray(gt_anno['name'])[n:]), 'GT rows past gt_boxes_lidar must be DontCare'
    idx = np.flatnonzero(ignored_gt == 0)
    assert (idx < n).all()
    return idx


def residuals_of(dets_by_id, infos, official, thresh=MATCH_IOU):
    res = {k: [] for k in ('l_ratio', 'w_ratio', 'h_ratio', 'dz_centre', 'dz_bottom', 'iou_bev', 'score')}
    n_gt = n_matched = 0
    for info in infos:
        gt_anno, dt = info['annos'], dets_by_id[info['point_cloud']['lidar_idx']]
        gi = valid_car_gt_index(official, gt_anno, dt)
        n_gt += len(gi)
        pi = np.flatnonzero(np.asarray(dt['name']) == 'Car')
        if len(gi) == 0 or len(pi) == 0:
            continue
        G = np.asarray(gt_anno['gt_boxes_lidar'][gi], np.float32)
        P = np.asarray(dt['boxes_lidar'][pi], np.float32)
        iou = boxes_bev_iou_cpu(P, G)
        pairs = match_one_to_one(iou, thresh)
        if not pairs:
            continue
        pp, gg = np.array([p for p, _ in pairs]), np.array([g for _, g in pairs])
        r = pair_residuals(P[pp], G[gg])
        for k, v in r.items():
            res[k].append(v)
        res['iou_bev'].append(np.asarray(iou)[pp, gg])
        res['score'].append(np.asarray(dt['score'])[pi][pp])
        n_matched += len(pairs)
    res = {k: np.concatenate(v) if v else np.zeros(0) for k, v in res.items()}
    return n_gt, n_matched, res


def _calib(frame_id, cache={}):
    if frame_id not in cache:
        from pcdet.utils import calibration_kitti
        cache[frame_id] = calibration_kitti.Calibration(CALIB_DIR / f'{frame_id}.txt')
    return cache[frame_id]


def project_bbox(d, info):
    from pcdet.utils import box_utils
    cam = np.concatenate([d['location'], d['dimensions'], d['rotation_y'][:, None]], 1).astype(np.float32)
    return box_utils.boxes3d_kitti_camera_to_imageboxes(cam, _calib(info['point_cloud']['lidar_idx']),
                                                        image_shape=info['image']['image_shape'])


def scale_det(d, s, keep='centre', info=None, reproject=True):
    """One frame's detections with l, w, h x s about the box centre ('centre') or with the bottom fixed ('bottom')."""
    d = copy.deepcopy(d)
    if not len(d['name']):
        return d
    dims = d['dimensions'].astype(np.float64)  # camera order l, h, w
    new = dims * s
    loc = d['location'].astype(np.float64).copy()
    lid = d['boxes_lidar'].astype(np.float64).copy()
    if keep == 'centre':
        loc[:, 1] -= (dims[:, 1] - new[:, 1]) / 2.0   # camera y points down: the bottom rises when h shrinks
    elif keep == 'bottom':
        lid[:, 2] -= (lid[:, 5] - lid[:, 5] * s) / 2.0  # LiDAR z up: the centre drops when h shrinks
    else:
        raise ValueError(keep)
    lid[:, 3:6] *= s
    d['dimensions'] = new.astype(np.float32)
    d['location'] = loc.astype(np.float32)
    d['boxes_lidar'] = lid.astype(np.float32)
    if reproject:
        d['bbox'] = project_bbox(d, info)
    return d


def scale_det_lidar(d, scale_lwh, dz, info):
    """One frame's detections with LiDAR l, w, h x scale_lwh about the centre and the centre z + dz, then converted to
    the camera frame with the frame's calibration (as generate_prediction_dicts does) and the 2D box re-projected."""
    from pcdet.utils import box_utils
    d = copy.deepcopy(d)
    if not len(d['name']):
        return d
    lid = d['boxes_lidar'].astype(np.float64).copy()
    lid[:, 3:6] *= np.asarray(scale_lwh, np.float64)
    lid[:, 2] += dz
    lid = lid.astype(np.float32)
    cam = box_utils.boxes3d_lidar_to_kitti_camera(lid, _calib(info['point_cloud']['lidar_idx']))
    d['boxes_lidar'] = lid
    d['location'] = cam[:, 0:3].astype(np.float32)
    d['dimensions'] = cam[:, 3:6].astype(np.float32)
    d['rotation_y'] = cam[:, 6].astype(np.float32)
    d['bbox'] = project_bbox(d, info)
    return d


def parse_r40(text, cls='Car'):
    """{'0.7': {'bev': (e, m, h), '3d': ...}, '0.5': ...} from the evaluator's printed AP_R40 blocks for `cls`."""
    lines = text.splitlines()
    out = {}
    for key, header in (('0.7', f'{cls} AP_R40@0.70, 0.70, 0.70:'), ('0.5', f'{cls} AP_R40@0.70, 0.50, 0.50:')):
        i = lines.index(header)
        block = {}
        for ln in lines[i + 1:i + 5]:
            tag, vals = ln.split(':', 1)
            block[tag.split()[0]] = tuple(float(v) for v in vals.split(','))
        out[key] = {'bev': block['bev'], '3d': block['3d']}
    return out


def score(official, infos, dets):
    gt = [copy.deepcopy(i['annos']) for i in infos]
    text, _ = official.get_official_eval_result(gt, dets, ['Car'])
    return parse_r40(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', nargs='+', help='<label>=<result.pkl>')
    parser.add_argument('--scale', type=float, default=1.0 / R6_RATIO)
    parser.add_argument('--out', required=True)
    parser.add_argument('--no-rescore', action='store_true')
    parser.add_argument('--arms', default='base,iso_centre,iso_centre_bbox_kept,iso_bottom',
                        help='comma list of arms to score; also measured_lwh, measured_lwh_z (report 20261108_06 '
                             'amendment 1: each run divided by its OWN median l / w / h ratios about the centre, '
                             'and also lowered by its own median z-centre offset; target-label medians, analysis only)')
    args = parser.parse_args()
    wanted = args.arms.split(',')
    official = load_official_eval()
    infos = pickle.load(open(INFOS, 'rb'))
    out = {'scale': args.scale, 'match_iou': MATCH_IOU, 'n_frames': len(infos), 'runs': {}}
    for item in args.results:
        label, path = item.split('=', 1)
        dets_by_id = {d['frame_id']: d for d in pickle.load(open(path, 'rb'))}
        n_gt, n_m, res = residuals_of(dets_by_id, infos, official)
        row = {'result_pkl': path, 'n_valid_car_gt': n_gt, 'n_matched': n_m,
               'residuals': {k: summarise(v) for k, v in res.items()},
               'residuals_score_ge_0.3': {k: summarise(v[res['score'] >= 0.3]) for k, v in res.items()},
               'n_matched_score_ge_0.3': int((res['score'] >= 0.3).sum())}
        print(f'{label}: valid Car GT {n_gt}, matched {n_m}', flush=True)
        for k, v in row['residuals'].items():
            print(f'  {k:10s} median {v["median"]:+.4f}  IQR [{v["q25"]:+.4f}, {v["q75"]:+.4f}]', flush=True)
        if not args.no_rescore:
            base = [dets_by_id[i['point_cloud']['lidar_idx']] for i in infos]
            # the re-projection reproduces the saved 2D boxes on the unscaled predictions
            dev = max((np.abs(project_bbox(d, i) - d['bbox']).max() if len(d['name']) else 0.0)
                      for d, i in zip(base, infos))
            row['bbox_reprojection_max_abs_dev_px'] = float(dev)
            med = {k: row['residuals'][k]['median'] for k in ('l_ratio', 'w_ratio', 'h_ratio', 'dz_centre')}
            inv = (1.0 / med['l_ratio'], 1.0 / med['w_ratio'], 1.0 / med['h_ratio'])
            row['measured_arm_factors'] = {'scale_lwh': inv, 'dz_measured_lwh_z': -med['dz_centre']}
            makers = {
                'base': lambda: base,
                'iso_centre': lambda: [scale_det(d, args.scale, 'centre', i) for d, i in zip(base, infos)],
                'iso_centre_bbox_kept': lambda: [scale_det(d, args.scale, 'centre', i, reproject=False)
                                                 for d, i in zip(base, infos)],
                'iso_bottom': lambda: [scale_det(d, args.scale, 'bottom', i) for d, i in zip(base, infos)],
                'measured_lwh': lambda: [scale_det_lidar(d, inv, 0.0, i) for d, i in zip(base, infos)],
                'measured_lwh_z': lambda: [scale_det_lidar(d, inv, -med['dz_centre'], i) for d, i in zip(base, infos)],
            }
            row['ap_r40_moderate'] = {}
            for name in wanted:
                dets = makers[name]()
                ap = score(official, infos, dets)
                row['ap_r40_moderate'][name] = {iou: {m: v[m][1] for m in ('bev', '3d')} for iou, v in ap.items()}
                row.setdefault('ap_r40_all', {})[name] = ap
                print(f'  {name:22s} IoU0.7 BEV {ap["0.7"]["bev"][1]:.4f} 3D {ap["0.7"]["3d"][1]:.4f} | '
                      f'IoU0.5 BEV {ap["0.5"]["bev"][1]:.4f} 3D {ap["0.5"]["3d"][1]:.4f}', flush=True)
        out['runs'][label] = row
        json.dump(out, open(args.out, 'w'), indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
