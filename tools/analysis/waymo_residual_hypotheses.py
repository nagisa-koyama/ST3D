"""Three cheap tests on the nuScenes -> Waymo residual, from existing Waymo-val result.pkl files (CPU, ANALYSIS).

H4  range vs point count: recall at score >= 0.3 per range ring AND per points-in-box bin. If the far-ring deficit
    against the oracle persists at equal point count, range itself (training coverage) matters.
H5  object count and score calibration: TP / FP score distributions and the score at which precision = 0.5 per ring;
    AP and recall stratified by the frame's Car GT count; predictions per frame vs GT per frame.
H7  source pitch: matched-box bottom-z error by azimuth sector (ahead / sides / behind) and ring, plus its spread.

Waymo labels are read to diagnose only (memory/repo/uda_legality_rule.md). Protocol as waymo_target_residual.py:
Vehicle -> Car, greedy BEV matching by score, IoU 0.7 (0.5 for the box-error table).

    python analysis/waymo_residual_hypotheses.py label=<result.pkl> ...
"""
import sys
from pathlib import Path
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from waymo_target_residual import load_gt, greedy_match, bev_iou, ap_r40, SCORE_MIN  # noqa: E402

RINGS = [0, 20, 43, 60, 75.2]
PTS = [(10, 50), (50, 200), (200, 10 ** 9)]
CNT = [(0, 11), (11, 26), (26, 41), (41, 10 ** 9)]


def frame_id(r):
    return r['frame_id']


def analyse(path, gt):
    res = pickle.load(open(path, 'rb'))
    rec = np.zeros((len(RINGS) - 1, len(PTS), 2))          # H4: hits / gt
    tp_sc = [[] for _ in RINGS[:-1]]; fp_sc = [[] for _ in RINGS[:-1]]   # H5(i), all predictions, per DT ring
    cnt = {k: ([], [], 0, [0, 0]) for k in range(len(CNT))}            # H5(ii): tp scores, fp scores, n gt, (hits@0.3, gt)
    pf = []                                                             # H5(iii): (n_gt, n_dt_conf)
    dz = []                                                             # H7: (ring idx, sector, dz_bottom, dl, dw)
    for r in res:
        g = gt[frame_id(r)]; gb, gp = g['boxes'], g['npts']
        m = r['name'] == 'Car'; db = r['boxes_lidar'][m].astype(np.float32); ds = r['score'][m]
        rg = np.linalg.norm(gb[:, :2], axis=1); rd = np.linalg.norm(db[:, :2], axis=1)
        # all-prediction matching at 0.7 for the score tables and the count strata
        dt_m, gt_m = greedy_match(gb, db, ds, 0.7)
        kd = np.clip(np.searchsorted(RINGS, rd, side='right') - 1, 0, len(RINGS) - 2)
        for k in range(len(RINGS) - 1):
            s = kd == k
            tp_sc[k].append(ds[s & (dt_m >= 0)]); fp_sc[k].append(ds[s & (dt_m < 0)])
        c = ds >= SCORE_MIN
        _, gt_mc = greedy_match(gb, db[c], ds[c], 0.7)
        hit = gt_mc >= 0
        ci = next(i for i, (lo, hi) in enumerate(CNT) if lo <= len(gb) < hi)
        tp, fp, n, h = cnt[ci]
        cnt[ci] = (tp + [ds[dt_m >= 0]], fp + [ds[dt_m < 0]], n + len(gb), [h[0] + int(hit.sum()), h[1] + len(gb)])
        pf.append((len(gb), int(c.sum()), int(hit.sum())))
        # H4
        kg = np.clip(np.searchsorted(RINGS, rg, side='right') - 1, 0, len(RINGS) - 2)
        for k in range(len(RINGS) - 1):
            for i, (lo, hi) in enumerate(PTS):
                s = (kg == k) & (gp >= lo) & (gp < hi)
                rec[k, i] += [hit[s].sum(), s.sum()]
        # H7: confident predictions matched at 0.5
        dt_m5, _ = greedy_match(gb, db[c], ds[c], 0.5)
        dbc = db[c]
        az = np.degrees(np.arctan2(gb[:, 1], gb[:, 0]))
        sec = np.where(np.abs(az) <= 45, 0, np.where(np.abs(az) >= 135, 2, 1))
        for i in np.where(dt_m5 >= 0)[0]:
            j = dt_m5[i]; p, q = dbc[i], gb[j]
            dz.append((kg[j], sec[j], (p[2] - p[5] / 2) - (q[2] - q[5] / 2), p[3] - q[3], p[4] - q[4]))
    out = dict(rec=rec, pf=np.array(pf), dz=np.array(dz))
    out['tp_sc'] = [np.concatenate(x) if x else np.zeros(0) for x in tp_sc]
    out['fp_sc'] = [np.concatenate(x) if x else np.zeros(0) for x in fp_sc]
    out['cnt'] = {k: (ap_r40(np.concatenate(tp), np.concatenate(fp), n) if n else float('nan'), n, h)
                  for k, (tp, fp, n, h) in cnt.items()}
    return out


def p50_score(tp, fp):
    """Smallest score t with precision(score >= t) >= 0.5 (nan if never)."""
    s = np.concatenate([tp, fp]); lab = np.concatenate([np.ones(len(tp)), np.zeros(len(fp))])
    o = np.argsort(-s); lab = lab[o]; s = s[o]
    prec = np.cumsum(lab) / np.arange(1, len(lab) + 1)
    ok = np.where(prec >= 0.5)[0]
    return float(s[ok[-1]]) if len(ok) else float('nan')


def main():
    args = [a.split('=', 1) for a in sys.argv[1:]]
    gt = load_gt()
    R = {lab: analyse(p, gt) for lab, p in args}
    labels = [l for l, _ in args]
    ring_names = ['%d-%d' % (RINGS[k], RINGS[k + 1]) for k in range(len(RINGS) - 1)]
    print('## H4: recall of Car GT at score >= 0.3, BEV IoU 0.7, by ring x points-in-box (stored count)')
    print('| model | ring | ' + ' | '.join('%d-%s' % (lo, hi - 1 if hi < 10 ** 9 else '') for lo, hi in PTS) + ' | GT per bin |')
    print('|---|---|' + '---|' * (len(PTS) + 1))
    for lab in labels:
        rec = R[lab]['rec']
        for k, rn in enumerate(ring_names):
            print(f'| {lab} | {rn} | ' + ' | '.join('%.2f' % (rec[k, i, 0] / max(rec[k, i, 1], 1)) for i in range(len(PTS)))
                  + ' | ' + ' / '.join('%d' % rec[k, i, 1] for i in range(len(PTS))) + ' |')
    print('\n## H5(i): score calibration per prediction ring (all predictions, matched at 0.7)')
    print('| model | ring | TP median score | FP median score | FP p90 | score at precision 0.5 | TP / FP counts |')
    print('|---|---|---|---|---|---|---|')
    for lab in labels:
        for k, rn in enumerate(ring_names):
            tp, fp = R[lab]['tp_sc'][k], R[lab]['fp_sc'][k]
            print(f'| {lab} | {rn} | {np.median(tp) if len(tp) else float("nan"):.3f} | {np.median(fp) if len(fp) else float("nan"):.3f} | '
                  f'{np.percentile(fp, 90) if len(fp) else float("nan"):.3f} | {p50_score(tp, fp):.3f} | {len(tp)} / {len(fp)} |')
    print('\n## H5(ii): Car BEV AP_R40 (IoU 0.7) and recall@0.3 by the frame\'s Car GT count')
    print('| model | ' + ' | '.join('%d-%s GT' % (lo, hi - 1 if hi < 10 ** 9 else '') for lo, hi in CNT) + ' |')
    print('|---|' + '---|' * len(CNT))
    for lab in labels:
        print(f'| {lab} | ' + ' | '.join('%.1f / %.2f (n=%d)' % (R[lab]['cnt'][k][0], R[lab]['cnt'][k][2][0] / max(R[lab]['cnt'][k][2][1], 1),
                                                              R[lab]['cnt'][k][1]) for k in range(len(CNT))) + ' |')
    print('\n## H5(iii): confident predictions (>= 0.3) per frame against GT per frame')
    print('| model | ' + ' | '.join('%d-%s GT: dt/gt, hits/gt' % (lo, hi - 1 if hi < 10 ** 9 else '') for lo, hi in CNT) + ' | frames |')
    print('|---|' + '---|' * (len(CNT) + 1))
    for lab in labels:
        pf = R[lab]['pf']; cells = []
        for lo, hi in CNT:
            s = (pf[:, 0] >= lo) & (pf[:, 0] < hi)
            cells.append('%.2f, %.2f' % (pf[s, 1].sum() / max(pf[s, 0].sum(), 1), pf[s, 2].sum() / max(pf[s, 0].sum(), 1)))
        print(f'| {lab} | ' + ' | '.join(cells) + f' | {len(pf)} |')
    print('\n## H7: matched-box (score >= 0.3, IoU 0.5) bottom-z error, median [IQR] in m, by GT sector x ring')
    print('| model | sector | ' + ' | '.join(ring_names) + ' |')
    print('|---|---|' + '---|' * len(ring_names))
    for lab in labels:
        dz = R[lab]['dz']
        for si, sn in enumerate(['ahead', 'sides', 'behind']):
            cells = []
            for k in range(len(ring_names)):
                s = (dz[:, 0] == k) & (dz[:, 1] == si); v = dz[s, 2]
                cells.append('%+.2f [%.2f] n=%d' % (np.median(v), np.percentile(v, 75) - np.percentile(v, 25), len(v)) if len(v) > 20 else '-')
            print(f'| {lab} | {sn} | ' + ' | '.join(cells) + ' |')
    print('\nH7 companion: median dl / dw by sector (all rings)')
    print('| model | ahead dl / dw | sides | behind |')
    print('|---|---|---|---|')
    for lab in labels:
        dz = R[lab]['dz']
        print(f'| {lab} | ' + ' | '.join('%+.2f / %+.2f' % (np.median(dz[dz[:, 1] == si, 3]), np.median(dz[dz[:, 1] == si, 4])) for si in range(3)) + ' |')


if __name__ == '__main__':
    main()
