"""Where the nuScenes -> Waymo residual lives: per-range AP, recall vs box fit, recall by GT attribute (CPU).

Reads Waymo-target eval `result.pkl` files and `waymo_infos_val.pkl`; no GPU, no retraining. ANALYSIS ONLY:
it reads Waymo labels to diagnose, never to choose a setting (memory/repo/uda_legality_rule.md).

Protocol reproduced: the family scores Waymo val with the KITTI evaluator under the "all objects" regime
(fabricated 2D box, so no difficulty filter; Vehicle -> Car; GT never range-filtered). Car only, BEV IoU 0.7
(and 0.5 for the box-fit question), greedy matching by score, 40-point interpolated AP. The pooled number
should land within ~1.5 of the logged `Car AP_R40@0.70` (the same re-implementation gap as gates/per_range_ap.py).

    python analysis/waymo_target_residual.py oracle=<result.pkl> legal=<result.pkl> ... [--rings 0,10,20,30,40,50,75]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402
import torch  # noqa: E402
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu  # noqa: E402

INFOS = TOOLS.parent / 'data/waymo/waymo_infos_val.pkl'
SCORE_MIN = 0.3        # for the recall-by-attribute and box-error tables (confident predictions)
LEN_BINS = [(0, 4.5), (4.5, 6.0), (6.0, 100)]
PTS_BINS = [(0, 1), (1, 10), (10, 50), (50, 200), (200, 10**9)]


def bev_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return boxes_bev_iou_cpu(torch.from_numpy(a).float(), torch.from_numpy(b).float()).numpy()


def iou3d_from_bev(a, b, bev):
    """3D IoU given BEV IoU, for boxes (x, y, z_centre, l, w, h, yaw): vertical overlap scales the intersection."""
    za0, za1 = a[2] - a[5] / 2, a[2] + a[5] / 2
    zb0, zb1 = b[2] - b[5] / 2, b[2] + b[5] / 2
    ov = max(0.0, min(za1, zb1) - max(za0, zb0))
    area_a, area_b = a[3] * a[4], b[3] * b[4]
    inter_bev = bev * (area_a + area_b) / (1.0 + bev)  # I/(A+B-I) = iou  ->  I = iou (A+B) / (1+iou)
    inter = inter_bev * ov
    union = area_a * a[5] + area_b * b[5] - inter
    return inter / max(union, 1e-9)


def ap_r40(tp_scores, fp_scores, n_gt):
    if n_gt == 0:
        return float('nan')
    s = np.concatenate([tp_scores, fp_scores]); is_tp = np.concatenate([np.ones(len(tp_scores)), np.zeros(len(fp_scores))])
    o = np.argsort(-s); is_tp = is_tp[o]
    tp = np.cumsum(is_tp); fp = np.cumsum(1 - is_tp)
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1)
    out = []
    for r in np.linspace(0, 1, 41)[1:]:
        m = rec >= r
        out.append(prec[m].max() if m.any() else 0.0)
    return 100 * float(np.mean(out))


def greedy_match(gt, dt, sc, thr):
    """Returns for each dt: matched gt index or -1 (greedy by score); and for each gt: matched dt index or -1."""
    o = np.argsort(-sc); iou = bev_iou(dt, gt)
    gt_taken = -np.ones(len(gt), dtype=int); dt_m = -np.ones(len(dt), dtype=int)
    for i in o:
        if len(gt) == 0:
            break
        row = iou[i].copy(); row[gt_taken >= 0] = -1
        j = int(row.argmax())
        if row[j] >= thr:
            gt_taken[j] = i; dt_m[i] = j
    return dt_m, gt_taken


def load_gt():
    infos = pickle.load(open(INFOS, 'rb'))
    gt = {}
    for inf in infos:
        a = inf['annos']; fid = inf['frame_id'] if 'frame_id' in inf else inf['point_cloud']['lidar_sequence'] + '_%03d' % inf['point_cloud']['sample_idx']
        m = a['name'] == 'Vehicle'
        gt[fid] = dict(boxes=a['gt_boxes_lidar'][m].astype(np.float32), npts=np.asarray(a['num_points_in_gt'])[m])
    return gt


def analyse(label, path, gt, rings):
    res = pickle.load(open(path, 'rb'))
    per_ring = {k: ([], [], 0) for k in range(len(rings) - 1)}
    pooled = {0.7: ([], [], 0), 0.5: ([], [], 0)}
    # attribute recall at SCORE_MIN, BEV IoU 0.7
    att = dict(len=np.zeros((len(LEN_BINS), 2)), pts=np.zeros((len(PTS_BINS), 2)), az=np.zeros((3, 2)), ring=np.zeros((len(rings) - 1, 2)))
    errs = []  # per matched box: (range, dl, dw, dh, dz_bottom, bev_iou, iou3d, iou3d_size_fixed, iou3d_z_fixed)
    n_fp_conf = 0; n_dt_conf = 0
    for r in res:
        g = gt[r['frame_id']]; gb, gp = g['boxes'], g['npts']
        m = r['name'] == 'Car'; db = r['boxes_lidar'][m].astype(np.float32); ds = r['score'][m]
        rg = np.linalg.norm(gb[:, :2], axis=1); rd = np.linalg.norm(db[:, :2], axis=1)
        for thr in (0.7, 0.5):
            dt_m, _ = greedy_match(gb, db, ds, thr)
            tp, fp, n = pooled[thr]; pooled[thr] = (tp + [ds[dt_m >= 0]], fp + [ds[dt_m < 0]], n + len(gb))
        for k in range(len(rings) - 1):
            lo, hi = rings[k], rings[k + 1]
            gsel = (rg >= lo) & (rg < hi); dsel = (rd >= lo) & (rd < hi)
            dt_m, _ = greedy_match(gb[gsel], db[dsel], ds[dsel], 0.7)
            tp, fp, n = per_ring[k]; per_ring[k] = (tp + [ds[dsel][dt_m >= 0]], fp + [ds[dsel][dt_m < 0]], n + int(gsel.sum()))
        # confident predictions: recall by attribute (0.7) and box errors (matched at 0.5)
        c = ds >= SCORE_MIN; dbc, dsc = db[c], ds[c]
        dt_m, gt_m = greedy_match(gb, dbc, dsc, 0.7)
        n_fp_conf += int((dt_m < 0).sum()); n_dt_conf += len(dbc)
        hit = gt_m >= 0
        az = np.degrees(np.arctan2(gb[:, 1], gb[:, 0]))
        az_bin = np.where(np.abs(az) <= 45, 0, np.where(np.abs(az) >= 135, 2, 1))
        for i, (lo, hi) in enumerate(LEN_BINS):
            s = (gb[:, 3] >= lo) & (gb[:, 3] < hi); att['len'][i] += [hit[s].sum(), s.sum()]
        for i, (lo, hi) in enumerate(PTS_BINS):
            s = (gp >= lo) & (gp < hi); att['pts'][i] += [hit[s].sum(), s.sum()]
        for i in range(3):
            s = az_bin == i; att['az'][i] += [hit[s].sum(), s.sum()]
        for k in range(len(rings) - 1):
            s = (rg >= rings[k]) & (rg < rings[k + 1]); att['ring'][k] += [hit[s].sum(), s.sum()]
        dt_m5, _ = greedy_match(gb, dbc, dsc, 0.5)
        iou = bev_iou(dbc, gb)
        for i in np.where(dt_m5 >= 0)[0]:
            j = dt_m5[i]; p, q = dbc[i], gb[j]; b = float(iou[i, j])
            i3 = iou3d_from_bev(p, q, b)
            p_size = p.copy(); p_size[3:6] = q[3:6]; b_size = float(bev_iou(p_size[None], q[None])[0, 0])
            p_z = p.copy(); p_z[2] = q[2] - q[5] / 2 + p[5] / 2  # same bottom as GT
            errs.append((np.linalg.norm(q[:2]), p[3] - q[3], p[4] - q[4], p[5] - q[5], (p[2] - p[5] / 2) - (q[2] - q[5] / 2),
                         b, i3, iou3d_from_bev(p_size, q, b_size), iou3d_from_bev(p_z, q, b)))
    out = {'label': label}
    for thr in (0.7, 0.5):
        tp, fp, n = pooled[thr]; out['ap_%.1f' % thr] = ap_r40(np.concatenate(tp), np.concatenate(fp), n)
    out['ring_ap'] = [ap_r40(np.concatenate(tp), np.concatenate(fp), n) for tp, fp, n in per_ring.values()]
    out['ring_n'] = [n for _, _, n in per_ring.values()]
    out['att'] = att; out['errs'] = np.array(errs); out['fp_conf'] = n_fp_conf; out['dt_conf'] = n_dt_conf
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs', nargs='+'); ap.add_argument('--rings', default='0,10,20,30,40,50,75')
    a = ap.parse_args(); rings = [float(x) for x in a.rings.split(',')]
    gt = load_gt()
    outs = [analyse(*r.split('=', 1), gt=gt, rings=rings) for r in a.runs]
    print('\n## Pooled Car BEV AP_R40 (re-implementation; compare with the logged value)')
    print('| model | IoU 0.7 | IoU 0.5 | confident (>= %.1f) predictions | of which unmatched at 0.7 |' % SCORE_MIN); print('|---|---|---|---|---|')
    for o in outs:
        print('| %s | %.2f | %.2f | %d | %.1f%% |' % (o['label'], o['ap_0.7'], o['ap_0.5'], o['dt_conf'], 100 * o['fp_conf'] / max(o['dt_conf'], 1)))
    print('\n## Car BEV AP_R40 at IoU 0.7 per ring of GT / prediction centre range (m)')
    hdr = ['%g-%g' % (rings[k], rings[k + 1]) for k in range(len(rings) - 1)]
    print('| model | ' + ' | '.join(hdr) + ' |'); print('|---' * (len(hdr) + 1) + '|')
    for o in outs:
        print('| %s | ' % o['label'] + ' | '.join('%.1f' % v for v in o['ring_ap']) + ' |')
    print('| GT boxes | ' + ' | '.join(str(n) for n in outs[0]['ring_n']) + ' |')
    print('\n## Recall of Car GT at score >= %.1f, BEV IoU 0.7, by GT attribute (hits / GT)' % SCORE_MIN)
    for key, bins, name in [('ring', hdr, 'range'), ('len', ['<4.5 m', '4.5-6 m', '>6 m'], 'GT length'),
                            ('pts', ['0 pts', '1-9', '10-49', '50-199', '>=200'], 'points in GT (stored count)'),
                            ('az', ['front +-45', 'sides', 'rear'], 'azimuth')]:
        print('\n%s: | model | ' % name + ' | '.join(bins) + ' |'); print('|---' * (len(bins) + 1) + '|')
        for o in outs:
            print('| %s | ' % o['label'] + ' | '.join('%.2f' % (h / max(n, 1)) for h, n in o['att'][key]) + ' |')
        print('| GT | ' + ' | '.join('%d' % n for _, n in outs[0]['att'][key]) + ' |')
    print('\n## Matched boxes (score >= %.1f, BEV IoU >= 0.5): median errors and what fixing one term buys in 3D' % SCORE_MIN)
    print('| model | matched | dl | dw | dh | dz bottom | BEV IoU | 3D IoU | share 3D >= 0.7 | size set to GT | bottom set to GT |'); print('|---' * 11 + '|')
    for o in outs:
        e = o['errs']
        print('| %s | %d | %+.2f | %+.2f | %+.2f | %+.2f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (
            o['label'], len(e), np.median(e[:, 1]), np.median(e[:, 2]), np.median(e[:, 3]), np.median(e[:, 4]),
            np.median(e[:, 5]), np.median(e[:, 6]), (e[:, 6] >= 0.7).mean(), (e[:, 7] >= 0.7).mean(), (e[:, 8] >= 0.7).mean()))
    print('\nper range (median dl / dz bottom / share 3D >= 0.7):')
    for o in outs:
        e = o['errs']; row = []
        for k in range(len(rings) - 1):
            s = (e[:, 0] >= rings[k]) & (e[:, 0] < rings[k + 1])
            row.append('%+.2f / %+.2f / %.2f (n=%d)' % (np.median(e[s, 1]), np.median(e[s, 4]), (e[s, 6] >= 0.7).mean(), s.sum()) if s.any() else '-')
        print('| %s | ' % o['label'] + ' | '.join(row) + ' |')


if __name__ == '__main__':
    main()
