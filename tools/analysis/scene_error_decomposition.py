"""ENV E1: foreground vs background errors on SCENE (labelled analysis; experiments_md 20261024_02).

Design FIXED in experiments_md T05 §5, entry "User 10-24 ~02:40", item 4 (commit 3cddbb7); applied here unchanged:
S->B (the S->S model on Boston val) against the B->B oracle on Boston val; the control pair B->S against S->S on
Singapore val. Per model, Car, at the metric's own matching (BEV IoU 0.7): background FPs (max BEV IoU with any GT
car < 0.1), localisation FPs (0.1 <= IoU < 0.7, duplicates included), missed GT (no detection at IoU >= 0.1).
dAP_bkg / dAP_loc / dAP_miss = AP with that category removed - AP; E_x = dAP_x(cross-city) - dAP_x(in-city oracle).
Reading on S->B: background-led if E_bkg >= 2 and E_bkg >= 2 max(E_miss, E_loc); foreground-led if E_miss >= 2 and
E_miss >= 2 max(E_bkg, E_loc); localisation-led likewise; mixed if two are >= 2 otherwise; unresolved if none is
>= 2. Control: on B->S every E_x is expected < 2, else the decomposition is flagged unreliable.

Sanity gate (fixed with the design): for each model, the Car BEV R40 IoU 0.7 recomputed from the saved predictions
must reproduce the logged value to within 0.05 BEFORE any decomposition is computed; otherwise exit 3.

    singularity exec --bind /home/koyama/data/:/storage <sif> python3 analysis/scene_error_decomposition.py \\
        --model 'B->B=<cfg>=<result.pkl>=<logged>' ... --pair 'S->B:B->B' --pair 'B->S:S->S' --out <json>

Run from ST3D/tools. CPU only. Reported beside: recall at IoU 0.1 and 0.7 by range and by points per GT box,
background FPs per frame at score >= 0.3, GT cars per frame.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kitti_frame_decomposition as kfd  # noqa: E402  (imports _init_path first)

GATE_TOL = 0.05
RANGE_BINS = [(0, 20), (20, 40), (40, 60), (60, np.inf)]          # m, [lo, hi), BEV centre distance from LIDAR_TOP
POINT_BINS = [(0, 0), (1, 4), (5, 9), (10, 19), (20, 49), (50, np.inf)]  # [lo, hi], num_lidar_pts (keyframe sweep)
BKG_SCORE = 0.3
CATS = ('bkg', 'miss', 'loc')


def e1_reading(E):
    """The fixed reading, literally. E = {'bkg': .., 'miss': .., 'loc': ..}."""
    label = {'bkg': 'background-led', 'miss': 'foreground-led', 'loc': 'localisation-led'}
    for x in CATS:
        others = [E[y] for y in CATS if y != x]
        if E[x] >= 2 and E[x] >= 2 * max(others):
            return label[x]
    n2 = sum(E[x] >= 2 for x in CATS)
    if n2 >= 2:
        return 'mixed'
    if n2 == 0:
        return 'unresolved'
    return 'not covered by the fixed reading (exactly one E >= 2, without the 2x margin)'


def binned_recall(values, hit, bins, inclusive):
    """Recall per bin: [lo, hi) for ranges (inclusive=False), [lo, hi] for integer point counts (inclusive=True)."""
    rows = []
    for lo, hi in bins:
        sel = (values >= lo) & ((values <= hi) if inclusive else (values < hi))
        n = int(sel.sum())
        name = (f'{lo}-{hi}' if np.isfinite(hi) else f'>={lo}') if not (inclusive and lo == hi) else f'{lo}'
        rows.append(dict(bin=name, n=n, recall=float(hit[sel].mean()) if n else float('nan')))
    return rows


def describe(fr, n_frames):
    gt, dt = fr.gt, fr.dt
    rng_m = gt['range']
    pts = gt['points']
    r01 = gt['best_iou'] >= kfd.CATEGORY_IOU
    r07 = gt['matched07'].astype(bool)
    cat = dt['category']
    dup = (cat == 2) & (dt['max_iou'] > kfd.MIN_OVERLAP)
    out = dict(
        frames=n_frames, gt_cars=int(len(gt['frame'])), gt_cars_per_frame=len(gt['frame']) / n_frames,
        missed=int((~r01).sum()), missed_share=float((~r01).mean()),
        car_dets=int(len(cat)), car_dets_per_frame=len(cat) / n_frames,
        all_dets=dict(tp=int((cat == 0).sum()), bkg_fp=int((cat == 1).sum()), loc_fp=int((cat == 2).sum()),
                      loc_fp_duplicates=int(dup.sum()), loc_fp_mislocalised=int(((cat == 2) & ~dup).sum())),
        bkg_fp_per_frame_score_ge_03=float(((cat == 1) & (dt['score'] >= BKG_SCORE)).sum() / n_frames),
        loc_fp_per_frame_score_ge_03=float(((cat == 2) & (dt['score'] >= BKG_SCORE)).sum() / n_frames),
        recall01=float(r01.mean()), recall07=float(r07.mean()),
        recall_by_range={'0.1': binned_recall(rng_m, r01, RANGE_BINS, False),
                         '0.7': binned_recall(rng_m, r07, RANGE_BINS, False)},
        recall_by_points={'0.1': binned_recall(pts, r01, POINT_BINS, True),
                          '0.7': binned_recall(pts, r07, POINT_BINS, True)},
        min_saved_score=float(dt['score'].min()) if len(cat) else float('nan'),
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', action='append', required=True, help="'<label>=<cfg>=<result.pkl>=<logged AP>'")
    ap.add_argument('--pair', action='append', required=True, help="'<cross label>:<in-city oracle label>' (first = read)")
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    official = kfd.load_official_eval()
    models = {}
    for spec in a.model:
        label, cfg, pkl, logged = spec.split('=')
        models[label] = dict(cfg=cfg, pkl=pkl, logged=float(logged))

    # ---- 1. sanity gate, before anything else is computed --------------------------------------------------------
    print('== sanity gate: Car BEV AP_R40 @0.7 recomputed from the saved predictions vs logged (tolerance 0.05) ==',
          flush=True)
    gate_ok = True
    for label, m in models.items():
        t0 = time.time()
        cap = kfd.capture_eval_annos(m['cfg'], m['pkl'], official)
        m['cap'] = cap
        m['recomputed'] = cap['car_bev_r40']
        diff = m['recomputed'] - m['logged']
        ok = abs(diff) <= GATE_TOL
        gate_ok &= ok
        print(f'{label:5s} frames {len(cap["gt"]):5d}  recomputed {m["recomputed"]:.4f}  logged {m["logged"]:.4f}  '
              f'diff {diff:+.4f}  {"PASS" if ok else "FAIL"}  ({time.time() - t0:.0f} s; {m["pkl"]})', flush=True)
    if not gate_ok:
        print('SANITY GATE FAILED: no decomposition is computed (design: STOP and report).', flush=True)
        sys.exit(3)
    print('sanity gate: ALL PASS', flush=True)

    # ---- 2. per-model decomposition ------------------------------------------------------------------------------
    result = dict(gate={k: dict(recomputed=v['recomputed'], logged=v['logged'], pkl=v['pkl'], cfg=v['cfg'])
                        for k, v in models.items()}, models={}, pairs={})
    for label, m in models.items():
        t0 = time.time()
        cap = m['cap']
        meta = dict(range=[np.linalg.norm(np.asarray(g['gt_boxes'])[:, :2], axis=1) if len(g['gt_boxes'])
                           else np.zeros(0) for g in cap['gt']],
                    points=[np.asarray(g['num_lidar_pts']) for g in cap['gt']])
        fr = kfd.CarBevFrames(cap['gt'], cap['dt'], official, gt_meta=meta, verify=True)
        base = fr.ap()
        assert abs(base - m['recomputed']) < 1e-6, (label, base, m['recomputed'])
        aps = dict(AP=base, AP_no_bkg=fr.ap(drop='bkg'), AP_no_loc=fr.ap(drop='loc'),
                   AP_no_miss=fr.ap(drop_missed=True))
        # official cross-checks: the two removals that are deletions
        checks = {}
        for what, key in (('bkg', 'AP_no_bkg'), ('miss', 'AP_no_miss')):
            g, d = kfd.annos_without(cap['gt'], cap['dt'], fr, what)
            checks[what] = kfd.official_car_bev_ap(g, d, official)
            assert abs(checks[what] - aps[key]) < 1e-6, (label, what, checks[what], aps[key])
        delta = dict(bkg=aps['AP_no_bkg'] - base, loc=aps['AP_no_loc'] - base, miss=aps['AP_no_miss'] - base)
        desc = describe(fr, len(cap['gt']))
        result['models'][label] = dict(ap=aps, delta=delta, official_deletion_checks=checks,
                                       n_state_checks=fr.n_state_checks, **desc)
        print(f'\n== {label}: AP {base:.2f}  dAP_bkg {delta["bkg"]:+.2f}  dAP_loc {delta["loc"]:+.2f}  '
              f'dAP_miss {delta["miss"]:+.2f}  (official deletion checks: no_bkg {checks["bkg"]:.4f} = '
              f'{aps["AP_no_bkg"]:.4f}, no_miss {checks["miss"]:.4f} = {aps["AP_no_miss"]:.4f}; '
              f'{fr.n_state_checks} step counts equal to compute_statistics_jit; {time.time() - t0:.0f} s)', flush=True)
        print(f'   frames {desc["frames"]}, GT cars {desc["gt_cars"]} ({desc["gt_cars_per_frame"]:.2f}/frame), missed '
              f'{desc["missed"]} ({100 * desc["missed_share"]:.1f}%), Car dets {desc["car_dets_per_frame"]:.1f}/frame; '
              f'all saved dets: TP {desc["all_dets"]["tp"]}, bkg FP {desc["all_dets"]["bkg_fp"]}, loc FP '
              f'{desc["all_dets"]["loc_fp"]} (duplicates {desc["all_dets"]["loc_fp_duplicates"]}, mislocalised '
              f'{desc["all_dets"]["loc_fp_mislocalised"]}); bkg FP/frame at score >= 0.3: '
              f'{desc["bkg_fp_per_frame_score_ge_03"]:.2f}; recall@0.1 {desc["recall01"]:.3f}, @0.7 '
              f'{desc["recall07"]:.3f}', flush=True)
        for kind in ('recall_by_range', 'recall_by_points'):
            for iou in ('0.1', '0.7'):
                cells = '  '.join(f'{r["bin"]}: {r["recall"]:.3f} (n={r["n"]})' for r in desc[kind][iou])
                print(f'   {kind} @{iou}: {cells}', flush=True)

    # ---- 3. excess and the fixed reading ---------------------------------------------------------------------------
    for i, spec in enumerate(a.pair):
        cross, oracle = spec.split(':')
        dc, do = result['models'][cross]['delta'], result['models'][oracle]['delta']
        E = {x: dc[x] - do[x] for x in CATS}
        if i == 0:
            verdict = e1_reading(E)
        else:
            flag = [x for x in CATS if E[x] >= 2]
            verdict = ('control as expected: every E_x < 2' if not flag else
                       f'control FAILS (E >= 2 for {", ".join(flag)}): decomposition flagged unreliable')
        result['pairs'][f'{cross} vs {oracle}'] = dict(E=E, verdict=verdict, role='read' if i == 0 else 'control')
        print(f'\n== {cross} vs {oracle} ({"reading" if i == 0 else "control"}): E_bkg {E["bkg"]:+.2f}  '
              f'E_loc {E["loc"]:+.2f}  E_miss {E["miss"]:+.2f}  -> {verdict}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(result, f, indent=1, default=float)
    print(f'\nwrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
