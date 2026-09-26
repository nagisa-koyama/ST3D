"""Per-box geometry errors of KITTI-target Car predictions, and what each one costs in 3D IoU.

Why this exists: on nuScenes -> KITTI the accumulation + global-correction row gains +29.5 BEV
moderate but only +17.3 3D, and stays far below the oracle on 3D. This separates the candidate
causes - a vertical offset (e.g. a SHIFT_COOR mismatch), a height error, or a length/width bias -
by matching each prediction to its GT and replacing one error at a time with the true value.

Works in the KITTI CAMERA frame straight from an eval `result.pkl` and `kitti_infos_val.pkl`, so
it needs no calibration and no GPU: `location` is the box BOTTOM centre, y points DOWN,
`dimensions` is (l, h, w), a box spans y in [loc_y - h, loc_y]. Predictions in result.pkl have
already had SHIFT_COOR removed by kitti_dataset.generate_prediction_dicts, exactly as scored.

The counterfactuals that substitute per-box GT are DIAGNOSTICS, not methods; the "median bias"
row uses target GT too (one constant per dimension) and is not UDA-legal either. The output is a
share of matched GT boxes, not AP.

    python analysis/kitti_target_box_errors.py control=<result.pkl> treated=<result.pkl> ...
"""
import pickle
import sys

import numpy as np
from shapely.geometry import Polygon

INFOS = '../data/kitti/kitti_infos_val.pkl'
SCORE_MIN = 0.3      # confident predictions only; the question is geometry, not ranking
MATCH_BEV_IOU = 0.5  # loose gate, so near-miss BEV matches are included in the decomposition
RANGE_BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 80)]


def bev_poly(b):
    x, z, l, w, ry = b[0], b[2], b[3], b[5], b[6]
    c, s = np.cos(ry), np.sin(ry)
    corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    return Polygon(corners @ np.array([[c, s], [-s, c]]).T + np.array([x, z]))


def ious(p, g):
    """(BEV IoU, 3D IoU) of two (x, y_bottom, z, l, h, w, ry) camera-frame boxes."""
    pb, gb = bev_poly(p), bev_poly(g)
    inter = pb.intersection(gb).area
    if inter <= 0:
        return 0.0, 0.0
    overlap = max(0.0, min(p[1], g[1]) - max(p[1] - p[4], g[1] - g[4]))
    v_i = inter * overlap
    return inter / (pb.area + gb.area - inter), v_i / (pb.area * p[4] + gb.area * g[4] - v_i)


def moderate_car_gt(annos):
    # KITTI 'moderate' eligibility: 2D height >= 25 px, occlusion <= 1, truncation <= 0.3
    h2d = annos['bbox'][:, 3] - annos['bbox'][:, 1]
    m = (annos['name'] == 'Car') & (h2d >= 25) & (annos['occluded'] <= 1) & (annos['truncated'] <= 0.3)
    return np.concatenate([annos['location'][m], annos['dimensions'][m], annos['rotation_y'][m, None]], 1)


def match(result_path, gt_by_id):
    """Greedy by score: each confident Car prediction takes its best-BEV-IoU unused GT."""
    pairs = []
    for d in pickle.load(open(result_path, 'rb')):
        G = moderate_car_gt(gt_by_id[d['frame_id']])
        keep = (d['name'] == 'Car') & (d['score'] >= SCORE_MIN)
        P = np.concatenate([d['location'][keep], d['dimensions'][keep], d['rotation_y'][keep, None]], 1)
        P = P[np.argsort(-d['score'][keep])]
        used = np.zeros(len(G), bool)
        for p in P:
            best, bj = 0.0, -1
            for j, g in enumerate(G):
                if not used[j] and np.hypot(p[0] - g[0], p[2] - g[2]) < 4:
                    b = ious(p, g)[0]
                    if b > best:
                        best, bj = b, j
            if bj >= 0 and best >= MATCH_BEV_IOU:
                used[bj] = True
                pairs.append((p, G[bj]))
    return pairs


def report(name, pairs):
    P = np.array([p for p, _ in pairs]); G = np.array([g for _, g in pairs])
    D = P - G
    med = np.median(D, 0)
    rng = np.hypot(G[:, 0], G[:, 2])
    bottom_up = -D[:, 1]                                   # + = prediction's bottom is HIGHER
    centre_up = -((P[:, 1] - P[:, 4] / 2) - (G[:, 1] - G[:, 4] / 2))

    def subst(fn):
        out = []
        for p, g in pairs:
            q = p.copy(); fn(q, g); out.append(ious(q, g)[1])
        return np.array(out)

    arms = {
        'BEV IoU (= 3D with perfect vertical)': np.array([ious(p, g)[0] for p, g in pairs]),
        'as-is': np.array([ious(p, g)[1] for p, g in pairs]),
        'height -> GT': subst(lambda q, g: q.__setitem__(4, g[4])),
        'bottom -> GT': subst(lambda q, g: q.__setitem__(1, g[1])),
        'length+width -> GT': subst(lambda q, g: (q.__setitem__(3, g[3]), q.__setitem__(5, g[5]))),
        'l, h, w -> GT': subst(lambda q, g: q.__setitem__(slice(3, 6), g[3:6])),
        'one median offset per dim': subst(lambda q, g: (q.__setitem__(1, q[1] - med[1]),
                                                         q.__setitem__(slice(3, 6), q[3:6] - med[3:6]))),
    }
    q = lambda a: f'median {np.median(a):+.3f}  p10/p90 {np.percentile(a, 10):+.3f}/{np.percentile(a, 90):+.3f}'
    print(f'\n=== {name}: {len(pairs)} matched moderate Car GT (score >= {SCORE_MIN}, BEV IoU >= {MATCH_BEV_IOU})')
    print(f'  bottom (+ = pred higher)   {q(bottom_up)}')
    print(f'  centre z (+ = pred higher) {q(centre_up)}')
    print(f'  height                     {q(D[:, 4])}')
    print(f'  length                     {q(D[:, 3])}')
    print(f'  width                      {q(D[:, 5])}')
    print('  share passing 3D IoU >= 0.7:')
    for k, v in arms.items():
        print(f'    {k:38s} {np.mean(v >= 0.7):.3f}')
    print('  by GT range:')
    for lo, hi in RANGE_BINS:
        m = (rng >= lo) & (rng < hi)
        if m.sum() > 10:
            print(f'    {lo:2d}-{hi:2d} m  n={m.sum():5d}  bottom {np.median(bottom_up[m]):+.3f}'
                  f'  height {np.median(D[m, 4]):+.3f}  3D-pass {np.mean(arms["as-is"][m] >= 0.7):.2f}')


if __name__ == '__main__':
    gt_by_id = {i['point_cloud']['lidar_idx']: i['annos'] for i in pickle.load(open(INFOS, 'rb'))}
    for arg in sys.argv[1:]:
        name, path = arg.split('=', 1)
        report(name, match(path, gt_by_id))
