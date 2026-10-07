"""UDA-legal height alignment between a source and a target: the offset is computed, never chosen.

Why (user, 2026-10-07): "今後の手法へ活用ために、センサ高さのオフセット計算は UDA-legal にしておきたい" - the sensor-height
offset must be a label-free rule, so a future method can use it. So the offset is a FUNCTION of two inputs only:
  - the target's TRAIN-split point clouds (never val, never a label), and
  - the sensors' published extrinsics (a spec, legal by the 2026-10-05 convention),
and it is evaluated once per pre-declared rule. Sweeping offsets and keeping the best one on val is NOT this tool.

Measurement, per dataset (per platform where a dataset has two): for each TRAIN frame (deterministic stride), the
second-lowest return per 1 m BEV cell within 4-40 m, a RANSAC plane z = a x + b y + c, least-squares refit on the
inliers (ground_tilt.py's estimator). g = median over frames of c, the ground height under the vehicle, in the
loader's RAW frame (before SHIFT_COOR). s = the top lidar's height in the same raw frame, from the extrinsic: 0 for
the datasets whose clouds are in the sensor frame (KITTI, nuScenes, Lyft), 2.184 m for Waymo (vehicle frame; TOP
extrinsic read from the tfrecord with tools/waymo_calib.py, constant across sequences).

Two rules for the target's SHIFT_COOR z, given the source config's shift S_src (the model frame = raw + SHIFT_COOR):
  G (ground alignment, the METHOD rule, declared 2026-10-07 before any result): the target ground lands where the
    source ground sits in the model frame:   S_tgt = g_src + S_src - g_tgt
    This is decision D's definition of SHIFT_COOR (each dataset's ground from its own train clouds) made relative
    to the trained frame.
  S (sensor alignment, ANALYSIS arm for the mount-height hypothesis): the target's top lidar lands at the source
    lidar's model-frame height:              S_tgt = s_src + S_src - s_tgt
    A rigid shift cannot reproduce a higher sensor's viewing geometry (incidence angles, occlusion); S tests only
    whether the frame ORIGIN's height matters.

    python analysis/ground_height_offset.py --source nuscenes --target waymo --src_shift 1.75 --tgt_shift 0.0
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ground_tilt import fit_plane, lowest_per_cell  # noqa: E402

ROOT = Path('/home/koyama/code/ST3D/data')
# top-lidar height in each loader's RAW frame, from the published / shipped extrinsic (spec, label-free)
SENSOR_Z = {'kitti': 0.0, 'nuscenes': 0.0, 'lyft': 0.0, 'waymo': 2.184}


def train_frames(dataset, n):
    """(points xyz in the loader's raw frame, platform) for n TRAIN frames at a deterministic stride."""
    if dataset == 'nuscenes':
        infos = pickle.load(open(ROOT / 'nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl', 'rb'))
        for i in np.linspace(0, len(infos) - 1, n).astype(int):
            info = infos[i]
            pts = np.fromfile(ROOT / 'nuscenes/v1.0-trainval' / info['lidar_path'], dtype=np.float32).reshape(-1, 5)
            pts = pts[~((np.abs(pts[:, 0]) < 1.5) & (np.abs(pts[:, 1]) < 1.5))]  # the loader's remove_ego_points
            yield pts[:, :3], info['lidar_path'].split('/')[-1][:4]  # n008 Boston / n015 Singapore
    elif dataset == 'lyft':
        infos = pickle.load(open(ROOT / 'lyft/trainval/lyft_infos_train.pkl', 'rb'))
        for i in np.linspace(0, len(infos) - 1, n).astype(int):
            info = infos[i]
            raw = np.fromfile(ROOT / 'lyft/trainval' / info['lidar_path'], dtype=np.float32)
            host = info['lidar_path'].split('/')[-1].split('_')[0]
            yield raw[: (len(raw) // 5) * 5].reshape(-1, 5)[:, :3], ('64-beam' if host in ('host-a101', 'host-a102') else '40-beam')
    elif dataset == 'kitti':
        infos = pickle.load(open(ROOT / 'kitti/kitti_infos_train.pkl', 'rb'))
        for i in np.linspace(0, len(infos) - 1, n).astype(int):
            idx = infos[i]['point_cloud']['lidar_idx']
            yield np.fromfile(ROOT / f'kitti/training/velodyne/{idx}.bin', dtype=np.float32).reshape(-1, 4)[:, :3], 'KITTI'
    elif dataset == 'waymo':
        train = sorted(l.strip().replace('.tfrecord', '') for l in open(ROOT / 'waymo/ImageSets/train.txt'))
        seqs = [ROOT / 'waymo/waymo_processed_data' / s for s in train]
        seqs = [s for s in seqs if s.is_dir()]
        for s in np.array(seqs)[np.linspace(0, len(seqs) - 1, min(n, len(seqs))).astype(int)]:
            f = sorted(Path(s).glob('*.npy'))
            if f:
                yield np.load(f[len(f) // 2])[:, :3], 'Waymo'
    else:
        raise ValueError(dataset)


def ground(dataset, n):
    """{platform: (median ground height c, IQR, frames)} in the raw frame."""
    per = {}
    for pts, plat in train_frames(dataset, n):
        res = fit_plane(lowest_per_cell(pts))
        if res is not None:
            per.setdefault(plat, []).append(res[0][2])
    return {k: (float(np.median(v)), float(np.percentile(v, 75) - np.percentile(v, 25)), len(v)) for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, choices=list(SENSOR_Z))
    ap.add_argument('--target', required=True, choices=list(SENSOR_Z))
    ap.add_argument('--src_shift', type=float, required=True, help="the source config's SHIFT_COOR z")
    ap.add_argument('--tgt_shift', type=float, required=True, help="the target config's current SHIFT_COOR z")
    ap.add_argument('--frames', type=int, default=400)
    a = ap.parse_args()

    gs, gt = ground(a.source, a.frames), ground(a.target, a.frames)
    print(f'TRAIN-split ground height under the vehicle, raw frame (median [IQR], frames):')
    for name, g in ((a.source, gs), (a.target, gt)):
        for plat, (c, iqr, k) in sorted(g.items()):
            print(f'  {name:9s} {plat:10s} g = {c:+.3f} [{iqr:.3f}]  n={k}   sensor above ground {SENSOR_Z[name] - c:.3f} m')
    # a two-platform source is pooled with equal platform weight: one SHIFT_COOR serves both in its config
    g_src = float(np.mean([v[0] for v in gs.values()]))
    g_tgt = float(np.mean([v[0] for v in gt.values()]))
    s_src, s_tgt = SENSOR_Z[a.source], SENSOR_Z[a.target]
    G = g_src + a.src_shift - g_tgt
    S = s_src + a.src_shift - s_tgt
    print(f'\nsource ground in the model frame: {g_src + a.src_shift:+.3f}; source lidar in the model frame: {s_src + a.src_shift:+.3f}')
    print(f'rule G (ground alignment, method rule): target SHIFT_COOR z = {G:+.3f}  (config now {a.tgt_shift:+.3f}, '
          f'change {G - a.tgt_shift:+.3f})')
    print(f'rule S (sensor alignment, analysis):    target SHIFT_COOR z = {S:+.3f}  (config now {a.tgt_shift:+.3f}, '
          f'change {S - a.tgt_shift:+.3f})')


if __name__ == '__main__':
    main()
