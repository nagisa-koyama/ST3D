"""ANALYSIS (experiments_md 20261011_01): the KITTI counterpart of oracle_box_error_sensitivity.py.

KITTI's evaluator reads the CAMERA-frame fields of each detection (location, dimensions, rotation_y), not
`boxes_lidar`. So every perturbed arm perturbs `boxes_lidar` with the shared 34-arm definitions and then rebuilds the
camera fields from it with the frame's own calibration, exactly as `KittiDataset.generate_prediction_dicts` does. The 2D
`bbox` is left alone (it only enters through the detection-ignore height). Two controls guard the conversion:
`base` keeps the stored fields; `roundtrip` rebuilds them from the UNCHANGED boxes and must equal `base` to the printed
digit. KITTI is a REAL-label target: easy / moderate / hard differ (unlike the fabricated-2D-box regime of the other
datasets), and the camera-FOV prediction filter was already applied when the result was written. CPU only.

    python analysis/oracle_box_error_kitti.py --result <result.pkl> --label kitti_oracle --out logs/box_error/kitti_oracle.jsonl
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
import oracle_box_error_sensitivity as B  # noqa: E402
import kitti_eval_cpu as K  # noqa: E402  (imports _init_path first)
from pcdet.utils import box_utils, calibration_kitti  # noqa: E402

CALIB = TOOLS.parent / 'data/kitti/training/calib'
_STATE = {}


def rebuild_camera_fields(det, calib):
    """Rewrite location / dimensions / rotation_y from `boxes_lidar` (same conversion as generate_prediction_dicts)."""
    det = copy.copy(det)
    if len(det['boxes_lidar']):
        cam = box_utils.boxes3d_lidar_to_kitti_camera(np.asarray(det['boxes_lidar'], np.float64), calib)
        det['location'] = cam[:, 0:3]; det['dimensions'] = cam[:, 3:6]; det['rotation_y'] = cam[:, 6]
    return det


def score_arm(name):
    kind, params = _STATE['arms'][name]
    t0 = time.time()
    dets = B.perturb_result(_STATE['dets'], kind, params)
    if name != 'base':
        dets = [rebuild_camera_fields(d, _STATE['calib'][d['frame_id']]) for d in dets]
    by_id = {d['frame_id']: d for d in dets}
    gt, dt = [], []
    for info in _STATE['infos']:
        gt.append(copy.deepcopy(info['annos'])); dt.append(copy.deepcopy(by_id[info['point_cloud']['lidar_idx']]))
    _, ap = _STATE['official'].get_official_eval_result(gt, dt, ['Car'])
    r = {'label': _STATE['label'], 'arm': name, 'kind': kind, 'params': params, 'seconds': round(time.time() - t0, 1)}
    for lvl in ('easy', 'moderate', 'hard'):
        r[f'bev_{lvl}'] = float(ap[f'Car_bev/{lvl}_R40']); r[f'3d_{lvl}'] = float(ap[f'Car_3d/{lvl}_R40'])
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--result', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--arms', default='all')
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()
    arms = B.build_arms(); arms['roundtrip'] = ('base', {})
    chosen = list(arms) if args.arms == 'all' else args.arms.split(',')
    assert all(c in arms for c in chosen)
    out = Path(args.out)
    done = {json.loads(l)['arm'] for l in out.read_text().splitlines()} if out.exists() else set()
    chosen = [c for c in chosen if c not in done]
    print(f'{args.label}: {len(chosen)} arms to score ({len(done)} done)', flush=True)
    official = K.load_official_eval()
    dets = pickle.load(open(args.result, 'rb'))
    infos = pickle.load(open(K.INFOS, 'rb'))
    assert len(dets) == len(infos), (len(dets), len(infos))
    calib = {d['frame_id']: calibration_kitti.Calibration(CALIB / f"{d['frame_id']}.txt") for d in dets}
    _STATE.update(arms=arms, dets=dets, infos=infos, calib=calib, official=official, label=args.label)

    def record(r):
        with open(out, 'a') as f:
            f.write(json.dumps(r) + '\n')
        print(f"{r['label']} {r['arm']:<12} BEV e/m/h {r['bev_easy']:.2f} / {r['bev_moderate']:.2f} / {r['bev_hard']:.2f}  "
              f"3D mod {r['3d_moderate']:.2f}  ({r['seconds']} s)", flush=True)

    if args.workers > 1:
        with mp.get_context('fork').Pool(args.workers) as pool:
            for r in pool.imap_unordered(score_arm, chosen):
                record(r)
    else:
        for c in chosen:
            record(score_arm(c))


if __name__ == '__main__':
    main()
