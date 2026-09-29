"""Car BEV AP_R40 per range ring from a KITTI-target result.pkl, on CPU.

The official evaluator (kitti_object_eval_python) compiles a numba.cuda kernel at import and cannot
run on the master node, and it has no range breakdown. This reimplements its Car / moderate /
BEV IoU 0.7 protocol closely enough to reproduce the pooled number (validated: see the report),
then applies it per ring of box-centre range:

  GT: class Car counted; Van, DontCare and Car above the difficulty (difficulty > 1 = harder than
      moderate) are IGNORED - a detection matched to them is neither TP nor FP. GT outside the ring
      is ignored too (the Waymo-style range breakdown).
  DT: class Car; a detection whose 2D box is under 25 px high is ignored (official MIN_HEIGHT);
      detections outside the ring are dropped. Greedy matching by score at BEV IoU >= 0.7.
  AP: 40-point interpolated precision-recall (R40).

    python per_range_ap.py <label>=<result.pkl> [...]  [--rings 0,20,40,60,80]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu  # noqa: E402
import torch  # noqa: E402

INFOS = TOOLS.parent / 'data/kitti/kitti_infos_val.pkl'
MIN_HEIGHT = 25.0
IOU = 0.7


def frame_gt(info):
    a = info['annos']
    n = len(a['gt_boxes_lidar'])
    names = a['name'][:n]; diff = a['difficulty'][:n]
    valid = (names == 'Car') & (diff >= 0) & (diff <= 1)
    ignored = ((names == 'Car') & ~valid) | (names == 'Van')
    keep = valid | ignored
    return a['gt_boxes_lidar'][keep].astype(np.float32), valid[keep]


def frame_dt(res):
    m = res['name'] == 'Car'
    boxes = res['boxes_lidar'][m].astype(np.float32); scores = res['score'][m]
    h = res['bbox'][m][:, 3] - res['bbox'][m][:, 1]
    return boxes, scores, h >= MIN_HEIGHT


def match(gt, gt_valid, dt, sc, dt_valid, lo, hi):
    """Returns (scores of TP, scores of FP, number of valid GT) inside the ring [lo, hi)."""
    rg = np.linalg.norm(gt[:, :2], axis=1) if len(gt) else np.zeros(0)
    rd = np.linalg.norm(dt[:, :2], axis=1) if len(dt) else np.zeros(0)
    in_ring_gt = (rg >= lo) & (rg < hi)
    gt_valid = gt_valid & in_ring_gt          # GT outside the ring counts as ignored
    keep_dt = (rd >= lo) & (rd < hi)
    dt, sc, dt_valid = dt[keep_dt], sc[keep_dt], dt_valid[keep_dt]
    n_gt = int(gt_valid.sum())
    if len(dt) == 0:
        return [], [], n_gt
    order = np.argsort(-sc); dt, sc, dt_valid = dt[order], sc[order], dt_valid[order]
    iou = boxes_bev_iou_cpu(torch.from_numpy(dt), torch.from_numpy(gt)).numpy() if len(gt) else np.zeros((len(dt), 0))
    assigned = np.zeros(len(gt), bool); tp, fp = [], []
    for i in range(len(dt)):
        if not dt_valid[i]:
            continue                            # ignored detection: neither TP nor FP
        cand = np.where(~assigned & (iou[i] >= IOU))[0] if len(gt) else np.zeros(0, int)
        if len(cand) == 0:
            fp.append(sc[i]); continue
        j = cand[np.argmax(iou[i][cand])]
        assigned[j] = True
        if gt_valid[j]:
            tp.append(sc[i])
        # matched an ignored GT: neither
    return tp, fp, n_gt


def ap_r40(tp, fp, n_gt):
    if n_gt == 0:
        return float('nan')
    s = np.concatenate([np.array(tp), np.array(fp)]); lab = np.concatenate([np.ones(len(tp)), np.zeros(len(fp))])
    o = np.argsort(-s); lab = lab[o]
    ctp = np.cumsum(lab); cfp = np.cumsum(1 - lab)
    rec = ctp / n_gt; prec = ctp / np.maximum(ctp + cfp, 1)
    ap = 0.0
    for r in np.linspace(1 / 40, 1, 40):
        m = rec >= r
        ap += prec[m].max() if m.any() else 0.0
    return 100.0 * ap / 40


def evaluate(result_path, rings):
    infos = {i['point_cloud']['lidar_idx']: i for i in pickle.load(open(INFOS, 'rb'))}
    results = pickle.load(open(result_path, 'rb'))
    out = {}
    for lo, hi in rings:
        tp, fp, n = [], [], 0
        for res in results:
            gt, gv = frame_gt(infos[res['frame_id']]); dt, sc, dv = frame_dt(res)
            t, f, k = match(gt, gv, dt, sc, dv, lo, hi); tp += t; fp += f; n += k
        out[(lo, hi)] = (ap_r40(tp, fp, n), n)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs', nargs='+'); ap.add_argument('--rings', default='0,20,40,60,80')
    a = ap.parse_args()
    edges = [float(x) for x in a.rings.split(',')]
    rings = [(0, 1e9)] + list(zip(edges[:-1], edges[1:]))
    header = ['run', 'pooled'] + ['%d-%d' % (lo, hi) for lo, hi in rings[1:]]
    print(' | '.join('%-14s' % h for h in header))
    for r in a.runs:
        label, path = r.split('=', 1)
        res = evaluate(path, rings)
        cells = ['%.2f (n=%d)' % res[k] for k in rings]
        print(' | '.join(['%-14s' % label] + ['%-14s' % c for c in cells]))


if __name__ == '__main__':
    main()
