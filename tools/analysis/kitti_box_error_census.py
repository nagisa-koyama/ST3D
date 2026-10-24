"""Per-component box errors of MATCHED KITTI-target Car predictions, and which component fails IoU 0.7.

Why (experiments_md 20261024_04 item 4; 20261011_07 §9.4): on nuScenes -> KITTI under the camera-FOV crop, IA-SSD
A2 (accumulation) is 7.0 BEV below A1 at IoU 0.7 but +1.5 at IoU 0.5, which was read as "box fit, not detection".
This census says which component of the fit differs: centre (along / across the GT heading), size (l, w, h),
yaw, and vertical position.

Built on `kitti_target_box_errors.py` (the census the S2 reports used, 20260926_03, T07 §2.5), which is NOT edited:
its camera-frame geometry (`bev_poly`, `ious`), its moderate-Car GT set (`moderate_car_gt`) and its greedy match
(`match`: confident Car predictions by score, each takes its best-BEV-IoU unused GT within 4 m, kept if the BEV
IoU >= the gate). Its two module constants (score floor, match gate) are set from the arguments here, defaults
unchanged (0.3 and 0.5). Added: the centre error split along / across the GT heading, the yaw error, the share
of matched boxes failing BEV and 3D IoU 0.7, and per-component substitution arms for the BEV 0.7 failures.

Camera frame (KITTI result.pkl, SHIFT_COOR already removed): x right, y DOWN, z forward; `location` is the box
BOTTOM centre; `dimensions` is (l, h, w); the box's length axis points along (cos ry, -sin ry) in (x, z), as
`bev_poly` builds it.

The substitution arms put one GT component into each prediction: DIAGNOSTICS on target labels (analysis), not
methods; nothing here selects anything.

    python analysis/kitti_box_error_census.py --out <json> NAME=<result.pkl> [NAME=<result.pkl> ...]
        [--score-min 0.3] [--match-iou 0.5] [--infos ../data/kitti/kitti_infos_val.pkl]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kitti_target_box_errors as kte  # noqa: E402

RANGE_BINS = kte.RANGE_BINS
COMPONENTS = ('along', 'across', 'centre_bev', 'centre_up', 'bottom_up', 'dl', 'dw', 'dh', 'yaw', 'yaw_pi')


def wrap(a, period):
    """Angle wrapped into [-period / 2, period / 2)."""
    return (np.asarray(a) + period / 2) % period - period / 2


def pair_errors(p, g):
    """Errors of one matched pair of (x, y_bottom, z, l, h, w, ry) camera-frame boxes, prediction minus GT.

    along / across: BEV centre offset projected on the GT's length axis (cos ry, -sin ry) and its normal
    (sin ry, cos ry). centre_up / bottom_up: + = prediction higher (camera y points down). yaw: wrapped to
    [-pi, pi); yaw_pi: wrapped to [-pi/2, pi/2), the part a box IoU sees (a box is symmetric under +pi).
    """
    dx, dz = p[0] - g[0], p[2] - g[2]
    c, s = np.cos(g[6]), np.sin(g[6])
    return {
        'along': dx * c - dz * s,
        'across': dx * s + dz * c,
        'centre_bev': float(np.hypot(dx, dz)),
        'centre_up': -((p[1] - p[4] / 2) - (g[1] - g[4] / 2)),
        'bottom_up': -(p[1] - g[1]),
        'dl': p[3] - g[3], 'dh': p[4] - g[4], 'dw': p[5] - g[5],
        'yaw': float(wrap(p[6] - g[6], 2 * np.pi)),
        'yaw_pi': float(wrap(p[6] - g[6], np.pi)),
    }


def substitute(p, g, what):
    """The prediction with one component replaced by the GT's (diagnostic)."""
    q = np.array(p, dtype=float)
    if what == 'centre_xz':
        q[[0, 2]] = g[[0, 2]]
    elif what == 'yaw':
        q[6] = g[6]
    elif what == 'lw':
        q[[3, 5]] = g[[3, 5]]
    elif what == 'vertical':          # bottom and height: the 3D-only components
        q[[1, 4]] = g[[1, 4]]
    elif what == 'lhw':
        q[3:6] = g[3:6]
    elif what == 'all':
        q = np.array(g, dtype=float)
    else:
        raise ValueError(what)
    return q


SUBST = ('centre_xz', 'yaw', 'lw', 'vertical', 'lhw', 'all')


def q3(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return None
    return {'median': float(np.median(a)), 'p10': float(np.percentile(a, 10)), 'p90': float(np.percentile(a, 90)),
            'mean': float(np.mean(a)), 'mean_abs': float(np.mean(np.abs(a)))}


def census(pairs):
    """Summary of matched pairs [(pred, gt), ...]: per-component errors, IoU 0.7 failure shares, substitution arms."""
    if not pairs:
        raise ValueError('no matched pairs')
    E = {k: np.array([pair_errors(p, g)[k] for p, g in pairs]) for k in COMPONENTS}
    iou = np.array([kte.ious(p, g) for p, g in pairs])          # (n, 2): BEV, 3D
    bev, d3 = iou[:, 0], iou[:, 1]
    G = np.array([g for _, g in pairs])
    rng = np.hypot(G[:, 0], G[:, 2])
    out = {
        'n_matched': len(pairs),
        'errors': {k: q3(v) for k, v in E.items()},
        'abs_yaw_pi_deg': q3(np.degrees(np.abs(E['yaw_pi']))),
        'share_abs_yaw_pi_over_10deg': float(np.mean(np.abs(E['yaw_pi']) > np.radians(10))),
        'share_heading_flipped': float(np.mean(np.abs(E['yaw']) > np.pi / 2)),
        'ratio_l': q3(np.array([p[3] / g[3] for p, g in pairs])),
        'ratio_w': q3(np.array([p[5] / g[5] for p, g in pairs])),
        'ratio_h': q3(np.array([p[4] / g[4] for p, g in pairs])),
        'bev_iou': q3(bev), 'iou_3d': q3(d3),
        'share_fail_bev_0.7': float(np.mean(bev < 0.7)),
        'share_fail_3d_0.7': float(np.mean(d3 < 0.7)),
        'share_fail_bev_0.5': float(np.mean(bev < 0.5)),  # 0 by construction at the default gate
        'share_fail_3d_0.5': float(np.mean(d3 < 0.5)),
        'subst': {},
        'by_range': [],
    }
    for w in SUBST:
        s = np.array([kte.ious(substitute(p, g, w), g) for p, g in pairs])
        out['subst'][w] = {'share_fail_bev_0.7': float(np.mean(s[:, 0] < 0.7)),
                           'share_fail_3d_0.7': float(np.mean(s[:, 1] < 0.7))}
    for lo, hi in RANGE_BINS:
        m = (rng >= lo) & (rng < hi)
        if m.sum() == 0:
            continue
        out['by_range'].append({
            'range': [lo, hi], 'n': int(m.sum()),
            'share_fail_bev_0.7': float(np.mean(bev[m] < 0.7)), 'share_fail_3d_0.7': float(np.mean(d3[m] < 0.7)),
            **{f'median_{k}': float(np.median(E[k][m])) for k in ('along', 'across', 'dl', 'dw', 'dh', 'bottom_up')},
            'median_abs_yaw_pi_deg': float(np.degrees(np.median(np.abs(E['yaw_pi'][m])))),
        })
    return out


def gt_counts(result_path, gt_by_id):
    """Moderate Car GT boxes over the frames of one result.pkl (the denominator of recall at the gate)."""
    dets = pickle.load(open(result_path, 'rb'))
    return len(dets), int(sum(len(kte.moderate_car_gt(gt_by_id[d['frame_id']])) for d in dets))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('rows', nargs='+', metavar='NAME=RESULT_PKL')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--score-min', type=float, default=kte.SCORE_MIN)
    p.add_argument('--match-iou', type=float, default=kte.MATCH_BEV_IOU)
    p.add_argument('--infos', default=kte.INFOS)
    args = p.parse_args()
    kte.SCORE_MIN, kte.MATCH_BEV_IOU = args.score_min, args.match_iou   # read by kte.match at call time
    gt_by_id = {i['point_cloud']['lidar_idx']: i['annos'] for i in pickle.load(open(args.infos, 'rb'))}
    result = {'score_min': args.score_min, 'match_bev_iou': args.match_iou, 'infos': args.infos, 'rows': {}}
    for arg in args.rows:
        name, path = arg.split('=', 1)
        frames, n_gt = gt_counts(path, gt_by_id)
        row = census(kte.match(path, gt_by_id))
        row.update({'result_pkl': path, 'frames': frames, 'moderate_car_gt': n_gt,
                    'matched_share_of_gt': row['n_matched'] / n_gt})
        result['rows'][name] = row
        e = row['errors']
        print(f"[{name}] frames {frames}, moderate Car GT {n_gt}, matched {row['n_matched']} "
              f"({row['matched_share_of_gt']:.3f}); fail BEV 0.7 {row['share_fail_bev_0.7']:.3f}, "
              f"3D 0.7 {row['share_fail_3d_0.7']:.3f}; median along {e['along']['median']:+.3f} across "
              f"{e['across']['median']:+.3f} dl {e['dl']['median']:+.3f} dw {e['dw']['median']:+.3f} "
              f"dh {e['dh']['median']:+.3f} bottom_up {e['bottom_up']['median']:+.3f} |yaw_pi| "
              f"{row['abs_yaw_pi_deg']['median']:.2f} deg", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, 'w'), indent=1)
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
