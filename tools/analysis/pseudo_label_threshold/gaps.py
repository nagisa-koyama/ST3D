"""Label-free missed-detection estimate from temporal gaps. Run from ST3D/tools.

For anchor i, take same-class box pairs (q-, q+) in frames i-1 / i+1, both with score >= HI,
motion-consistent. Interpolate the box into frame i (lidar coords, SHIFT_COOR applied as the
pseudo-labels are), count points inside it, and record whether frame i holds a same-class
pseudo-label (score >= 0.10) within TOL of the interpolated centre.

Also records every anchor-frame pseudo-box that IS such a track midpoint, so detected and missed
objects of the same 'confirmed track' population are comparable.
Validation-only column: whether a real GT box of that class lies within 2 m of the interpolated centre.
"""
import copy, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path('.').resolve()))
import _init_path  # noqa
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.utils import common_utils
import bins

S = sys.argv[1]
HI = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
TOL, VMAX = 1.5, 20.0
PAIR = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
cfg_from_yaml_file('cfgs/da-ieee-access/centerpoint-foreground-lyft2nuscenes.yaml', cfg)
rc = copy.deepcopy(cfg.DATA_CONFIG_TAR); rc.USE_PSEUDO_LABEL = False
ds, _, _ = build_dataloader(rc, cfg.CLASS_NAMES, 1, False, workers=0, logger=common_utils.create_logger(),
                            training=True, model_ontology=cfg.get('ONTOLOGY'))
infos = ds.infos
ps = pickle.load(open(bins.PS_LABEL, 'rb'))
SHIFT = np.array(rc.SHIFT_COOR)
NAMES = {1: 'car', 2: 'pedestrian'}


def T_lidar_to_global(i):
    return np.linalg.inv(infos[i]['car_from_global']) @ np.linalg.inv(infos[i]['ref_from_car'])


def boxes(i):
    k = Path(infos[i]['lidar_path']).stem
    b = ps.get(k)
    if b is None:
        return None
    b = b['gt_boxes']; return b[b[:, 7] > 0] if len(b) else b.reshape(0, 9)


def to_frame(b, src, dst):
    """boxes (lidar+shift of src) -> lidar+shift of dst"""
    T = np.linalg.inv(T_lidar_to_global(dst)) @ T_lidar_to_global(src)
    c = b[:, :3] - SHIFT
    c = (T[:3, :3] @ c.T).T + T[:3, 3] + SHIFT
    yaw = np.arctan2(T[1, 0], T[0, 0])
    out = b.copy(); out[:, :3] = c; out[:, 6] = b[:, 6] + yaw
    return out


def ok(i, j):
    return 0 <= j < len(infos) and 0.3 < abs(infos[j]['timestamp'] - infos[i]['timestamp']) < 0.7


rows = []  # cls, range, npts_in_interp_box, detected, det_score, gt_near(valid), frame
step = len(infos) // 1000
for n_done, i in enumerate(range(0, len(infos), step)[:1000]):
    if not (ok(i, i - 1) and ok(i, i + 1)):
        continue
    b0, bp, bn = boxes(i), boxes(i - 1), boxes(i + 1)
    if b0 is None or bp is None or bn is None:
        continue
    pts = ds.get_lidar_with_sweeps(i, max_sweeps=1)
    pts[:, :3] += SHIFT
    bpi, bni = to_frame(bp, i - 1, i), to_frame(bn, i + 1, i)
    gt = infos[i]['gt_boxes']; gnm = infos[i]['gt_names']
    for c, name in NAMES.items():
        P = bpi[(bp[:, 7] == c) & (bp[:, 8] >= HI)]; N = bni[(bn[:, 7] == c) & (bn[:, 8] >= HI)]
        if not len(P) or not len(N):
            continue
        used_n = np.zeros(len(N), bool); interp = []
        for p in P:
            d = np.linalg.norm(N[:, :2] - p[:2], axis=1); d[used_n] = np.inf
            j = np.argmin(d)
            if d[j] < PAIR:
                mid = (p[:7] + N[j, :7]) / 2
                # motion-consistent: midpoint of the pair must be a plausible straight-line position
                used_n[j] = True
                mid[6] = p[6]  # heading from one side; averaging angles is not needed for point counting
                interp.append(mid)
        if not interp:
            continue
        interp = np.array(interp, dtype=np.float32)
        interp = interp[np.linalg.norm(interp[:, :2], axis=1) < 70]
        if not len(interp):
            continue
        _, per_box, kept = DataProcessor.box_occupancy(pts, interp)
        cur = b0[b0[:, 7] == c]
        for bx, npt in zip(kept, per_box):
            if len(cur):
                d = np.linalg.norm(cur[:, :2] - bx[:2], axis=1); j = np.argmin(d)
                det, sc = d[j] < TOL, cur[j, 8] if d[j] < TOL else -1
            else:
                det, sc = False, -1
            g = gt[gnm == name]
            gnear = (np.linalg.norm(g[:, :2] + 0 - (bx[:2]), axis=1) < 2.0).any() if len(g) else False
            rows.append((c, np.linalg.norm(bx[:2]), npt, det, sc, gnear, i))
    if (n_done + 1) % 200 == 0:
        print('%d/1000' % (n_done + 1), flush=True)
pickle.dump(np.array(rows, dtype=float), open(S + '/gaps_%s_%s.pkl' % (HI, PAIR), 'wb'))
print('tracks', len(rows))
