"""ANALYSIS (experiments_md 20261011_04): does a box-size / localisation effect survive a looser IoU threshold?

KITTI-metric Car AP is scored at a BEV / 3D IoU of 0.7. That bar is tight enough that a 5-10% box-size error moves it by
tens of points (20261011_01). This re-scores stored KITTI-val predictions at IoU 0.7, 0.5, 0.3 and 0.1 in ONE evaluator
call (the IoU matrices are computed once, only the matching changes), for
  (A) the nuScenes -> KITTI rows that differ in a box-size PRIOR (Car ROS interval, a test-time x0.933 shrink) and in
      accumulation, and
  (B) the KITTI in-domain oracle with one controlled box error added (size, heading, position).
CPU only, KITTI GT (real labels, difficulty levels unchanged). The 2D-bbox bar stays at 0.7 and is not reported.

    python analysis/iou_threshold_sensitivity.py --out logs/box_error/iou_threshold.jsonl [--workers 8]
"""
import argparse
import copy
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent)); sys.path.insert(0, str(TOOLS / 'analysis'))
import kitti_eval_cpu as K  # noqa: E402  (imports _init_path first)
import oracle_box_error_sensitivity as B  # noqa: E402
import oracle_box_error_kitti as OK  # noqa: E402
from posthoc_box_scale import scale_result  # noqa: E402

W = '/home/koyama/data/wandb/'
O = str(TOOLS.parent / 'output' / 'da-ieee-access') + '/'
EV = '/files/eval/eval_with_train/epoch_%d/val/result.pkl'
THRESHOLDS = [0.7, 0.5, 0.3, 0.1]
ORACLE = W + 'run-20261001_005323-06iwywhs' + EV % 152
# group, label, result.pkl, post-processing: None | ('scale', s) | ('arm', arm name)
ROWS = [
    ('A', '26265 control: nuScenes single sweep, ROS [0.85,1.20]', O + 'centerpoint-sourceonly-nuscenes2kitti/20260922_sourceonly/eval/epoch_20/val/control_no_accum_no_corr/result.pkl', None),
    ('A', '26388 accumulation 15, ROS [0.85,1.20]', W + 'run-20260928_050622-exbb9chu' + EV % 20, None),
    ('A', '26388 accumulation, test-time x0.933', W + 'run-20260928_050622-exbb9chu' + EV % 20, ('scale', 0.933)),
    ('A', '26536 accumulation 15, ROS [0.75,1.00] (size prior)', W + 'run-20260929_092722-zpndk3rh' + EV % 20, None),
    ('A', '26814 accumulation, ROS [0.81,1.06]', W + 'run-20261001_022243-2da6oz6e' + EV % 20, None),
    ('A', '27468 legal teacher (Boston 15 / Singapore 10)', W + 'run-20261006_004144-o3l96a3p' + EV % 20, None),
    ('A', '27468 teacher, test-time x0.933', W + 'run-20261006_004144-o3l96a3p' + EV % 20, ('scale', 0.933)),
    ('B', 'KITTI oracle 26843 (z-shift)', ORACLE, None),
    ('B', 'KITTI oracle 26717 (no ROS)', W + 'run-20260930_132344-vezunseu' + EV % 152, None),
    ('B', 'KITTI oracle 26912 (ROS [0.75,1.00])', W + 'run-20261001_134345-2g08c9pb' + EV % 152, None),
] + [('B', f'KITTI oracle 26843 + box error {a}', ORACLE, ('arm', a))
     for a in ('sz_iso0.90', 'sz_iso0.95', 'sz_iso1.05', 'sz_iso1.10', 'yaw_b5', 'yaw_b10', 'lon0.2', 'lon0.5', 'lat0.2', 'z0.2')]
_STATE = {}


def load_dets(path):
    if path not in _STATE['cache']:
        _STATE['cache'][path] = pickle.load(open(path, 'rb'))
    return _STATE['cache'][path]


def run_row(row):
    group, label, path, post = row
    t0 = time.time()
    dets = load_dets(path)
    if post is not None and post[0] == 'scale':
        s = post[1]
        dets = scale_result(dets, s, s, s)
    elif post is not None:
        kind, params = _STATE['arms'][post[1]]
        dets = B.perturb_result(dets, kind, params)
        dets = [OK.rebuild_camera_fields(d, _STATE['calib'][d['frame_id']]) for d in dets]
    by_id = {d['frame_id']: d for d in dets}
    gt = [copy.deepcopy(i['annos']) for i in _STATE['infos']]
    dt = [copy.deepcopy(by_id[i['point_cloud']['lidar_idx']]) for i in _STATE['infos']]
    mo = np.zeros([len(THRESHOLDS), 3, 1]); mo[:, 0, 0] = 0.7; mo[:, 1, 0] = THRESHOLDS; mo[:, 2, 0] = THRESHOLDS
    out = _STATE['official'].do_eval(gt, dt, [0], mo, False)
    bev, d3 = out[5], out[6]  # R40, [class, difficulty, threshold]
    res = {'group': group, 'label': label, 'post': post, 'seconds': round(time.time() - t0, 1)}
    for k, t in enumerate(THRESHOLDS):
        for d, lvl in enumerate(('easy', 'moderate', 'hard')):
            res[f'bev_{lvl}@{t}'] = float(bev[0, d, k]); res[f'3d_{lvl}@{t}'] = float(d3[0, d, k])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()
    from pcdet.utils import calibration_kitti
    out = Path(args.out)
    done = {json.loads(l)['label'] for l in out.read_text().splitlines()} if out.exists() else set()
    rows = [r for r in ROWS if r[1] not in done]
    missing = [r[2] for r in rows if not Path(r[2]).exists()]
    assert not missing, missing
    print(f'{len(rows)} rows to score ({len(done)} done)', flush=True)
    infos = pickle.load(open(K.INFOS, 'rb'))
    calib = {i['point_cloud']['lidar_idx']: calibration_kitti.Calibration(OK.CALIB / f"{i['point_cloud']['lidar_idx']}.txt")
             for i in infos}
    _STATE.update(official=K.load_official_eval(), infos=infos, calib=calib, arms=B.build_arms(), cache={})

    def record(r):
        with open(out, 'a') as f:
            f.write(json.dumps(r) + '\n')
        print(r['label'], ' | '.join(f"@{t}: BEV {r[f'bev_moderate@{t}']:.1f} 3D {r[f'3d_moderate@{t}']:.1f}" for t in THRESHOLDS),
              f"({r['seconds']} s)", flush=True)

    if args.workers > 1:
        with mp.get_context('fork').Pool(args.workers) as pool:
            for r in pool.imap_unordered(run_row, rows):
                record(r)
    else:
        for r in rows:
            record(run_row(r))


if __name__ == '__main__':
    main()
