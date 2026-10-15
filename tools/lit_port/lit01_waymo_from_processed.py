"""Replacement for LiT's lit_01_extract_scene.py (Waymo branch) that reads OUR processed Waymo data instead of tfrecords
through TensorFlow (experiments_md 20261015_02 §1). Runs INSIDE the LiT container, with LiT's code on PYTHONPATH:

    python -I lit_port/lit01_waymo_from_processed.py --infos <waymo_infos_train.pkl> --data <waymo_processed_data> \
        --calib <top_calib.pkl> --out <scene_dir> [--scene_index i | --scene_names a,b] [--skip_existing]

It writes one `Scene` pickle per sequence, in LiT's format (`Scene.save`), with only the TOP lidar (LiT's reconstruction
and simulation read lidar 0 only: lit_02 `enabled_lidars=(0,)`, `Scene.extract_fg` "only the top lidar is used").

Conventions, each checked against the loader of this repo (checklist rule 5):
- points: the processed `NNNN.npy` is (N, 7) = x, y, z, intensity, elongation, NLZ flag, vehicle frame, lidars stored in
  name order with TOP first; `info['num_points_of_each_lidar'][0]` points are TOP. The slice happens BEFORE the NLZ
  filter, as `WaymoDataset.get_lidar` does; NLZ == -1 is kept (no-label-zone points are dropped). Intensity gets the same
  tanh as the loader. No SHIFT_COOR is applied here (LiT works in the vehicle frame; the detector's loader adds it later).
- pose: `info['pose']` is vehicle -> global (the loader uses its inverse for the global -> vehicle step).
- lidar_to_vehicle_poses[0]: the segment's TOP extrinsic (top_calib.pkl), lidar -> vehicle.
- boxes: `annos['gt_boxes_lidar'][:, :7]` in the vehicle frame, label column 7 = 1 Vehicle, 2 Pedestrian, 3 Cyclist
  (LiT's Waymo order); other classes (Sign) are dropped. `object_ids` = Waymo's track id, as str.
- float32 everywhere (LiT's `BaseContainer.save` asserts it).
No label of any target split is read: this is the SOURCE.
"""
import argparse
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

LABEL = {'Vehicle': 1, 'Pedestrian': 2, 'Cyclist': 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infos', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scene_index', type=int, default=None)
    ap.add_argument('--scene_names', default=None)
    ap.add_argument('--skip_existing', action='store_true')
    ap.add_argument('--max_frames', type=int, default=None, help='smoke only')
    a = ap.parse_args()

    from lit.containers.frame import Frame
    from lit.containers.scene import Scene

    with open(a.infos, 'rb') as f:
        infos = pickle.load(f)
    with open(a.calib, 'rb') as f:
        calib = pickle.load(f)
    seqs = OrderedDict()
    for inf in infos:
        seqs.setdefault(inf['point_cloud']['lidar_sequence'], []).append(inf)
    names = list(seqs)
    if a.scene_names:
        names = [n for n in a.scene_names.split(',')]
    elif a.scene_index is not None:
        names = [names[a.scene_index]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = out / f'{name}.pkl'
        if a.skip_existing and path.exists():
            print('skip', name)
            continue
        L2V = np.asarray(calib[name]['extrinsic'], np.float64).reshape(4, 4).astype(np.float32)
        scene = Scene(scene_name=name)
        frames = sorted(seqs[name], key=lambda i: i['point_cloud']['sample_idx'])
        assert [i['point_cloud']['sample_idx'] for i in frames] == list(range(len(frames))), \
            f'{name}: sample_idx is not 0..N-1, LiT appends frames sequentially'
        if a.max_frames:
            frames = frames[:a.max_frames]
        for k, inf in enumerate(frames):
            raw = np.load(Path(a.data) / name / ('%04d.npy' % inf['point_cloud']['sample_idx']))
            counts = np.asarray(inf['num_points_of_each_lidar'], np.int64)
            assert counts.sum() == len(raw), f'{name} {k}: per-lidar counts do not match the file'
            top = raw[:counts[0]]
            top = top[top[:, 5] == -1][:, :5].astype(np.float32)
            top[:, 3] = np.tanh(top[:, 3])
            an = inf['annos']
            keep = [i for i, n in enumerate(an['name']) if n in LABEL]
            if keep:
                b = np.asarray(an['gt_boxes_lidar'], np.float32)[keep, :7]
                lab = np.array([LABEL[an['name'][i]] for i in keep], np.float32)[:, None]
                boxes = np.concatenate([b, lab], 1).astype(np.float32)
                ids = [str(an['obj_ids'][i]) for i in keep]
            else:
                boxes, ids = np.zeros((0, 8), np.float32), []
            scene.append_frame(Frame(scene_name=name, frame_index=k,
                                     frame_pose=np.asarray(inf['pose'], np.float32).reshape(4, 4),
                                     lidar_to_vehicle_poses=[L2V], num_points_of_each_lidar=[len(top)],
                                     local_points=top, local_bboxes=boxes, object_ids=ids), check_valid=True)
        scene.save(path)
        print(f'{name}: {len(scene)} frames -> {path}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
