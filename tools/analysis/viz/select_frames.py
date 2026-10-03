"""Choose the common frames every run's figure shows (experiments_md/20261003_02 §4).

Per frame set: frames with >= 1 Car and >= 1 Pedestrian label, each holding >= MIN_PTS points of the
STORED sweep, centre inside the detection grid; taken from distinct drives, by a seeded draw. Real
labels decide only which frames are shown. Writes viz_frames.json next to this file; run from tools/
in the container (CPU):

    python analysis/viz/select_frames.py [set ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_common as vc  # noqa: E402

from pcdet.config import cfg_from_yaml_file  # noqa: E402

OUT = Path(__file__).resolve().parent / 'viz_frames.json'
SEED, MIN_PTS, GRID = 0, 5, 75.2
# name: (dataset config, training split?, how many). Two-platform sources split 2 + 1 (§4).
SETS = {
    'kitti/val': ('cfgs/da-ieee-access/da_kitti_dataset.yaml', False, 3),
    'kitti/train': ('cfgs/da-ieee-access/da_kitti_dataset.yaml', True, 3),
    'nuscenes/val': ('cfgs/da-ieee-access/da_nuscenes_dataset.yaml', False, 3),
    'nuscenes/train/n008': ('cfgs/da-ieee-access/da_nuscenes_n008_dataset.yaml', True, 2),
    'nuscenes/train/n015': ('cfgs/da-ieee-access/da_nuscenes_n015_dataset.yaml', True, 1),
    'lyft/train/40': ('cfgs/da-ieee-access/da_lyft40_dataset.yaml', True, 2),
    'lyft/train/64': ('cfgs/da-ieee-access/da_lyft64_dataset.yaml', True, 1),
}
LYFT_ROOT = Path('/home/koyama/code/ST3D/data/lyft/trainval')


def lyft_front_images(lidar_paths):
    """lidar file -> CAM_FRONT image of the same sample, via the SDK tables (no SDK needed)."""
    sd = json.load(open(LYFT_ROOT / 'data' / 'sample_data.json'))
    cal = {c['token']: c['sensor_token'] for c in json.load(open(LYFT_ROOT / 'data' / 'calibrated_sensor.json'))}
    chan = {s['token']: s['channel'] for s in json.load(open(LYFT_ROOT / 'data' / 'sensor.json'))}
    want = set(lidar_paths)
    sample_of = {r['filename']: r['sample_token'] for r in sd if r['filename'] in want}
    tokens = set(sample_of.values())
    front = {r['sample_token']: r['filename'] for r in sd
             if r['sample_token'] in tokens and chan[cal[r['calibrated_sensor_token']]] == 'CAM_FRONT'}
    return {p: str(LYFT_ROOT / front[sample_of[p]]) if p in sample_of and sample_of[p] in front else None
            for p in lidar_paths}


def select(name, cfg_path, training, k):
    dc = EasyDict()
    cfg_from_yaml_file(cfg_path, dc)
    ds = vc.build_dataset(dc, ['Car', 'Pedestrian', 'Cyclist'], training, 'kitti')
    rng = np.random.RandomState(SEED)
    I = vc.infos(ds)
    order = rng.permutation(len(I))
    chosen, groups, tried = [], set(), 0
    for i in order:
        info = I[i]
        boxes, names = vc.raw_labels(ds, info)
        inside = (np.abs(boxes[:, 0]) < GRID) & (np.abs(boxes[:, 1]) < GRID) if len(boxes) else np.zeros(0, bool)
        if not ((names[inside] == vc.CAR).any() and (names[inside] == vc.PED).any()):
            continue
        g = vc.frame_group(ds, info)
        if g in groups:
            continue
        tried += 1
        n = vc.points_per_box(vc.raw_points(ds, info), boxes)
        ok = inside & (n >= MIN_PTS)
        n_car, n_ped = int((ok & (names == vc.CAR)).sum()), int((ok & (names == vc.PED)).sum())
        if n_car and n_ped:
            chosen.append(dict(fid=vc.frame_id(ds, info), group=str(g), image=vc.image_path(ds, info), yaw=vc.forward_yaw(ds, info),
                               n_car=n_car, n_ped=n_ped))
            groups.add(g)
            if len(chosen) == k:
                break
    assert len(chosen) == k, '%s: only %d qualifying frames' % (name, len(chosen))
    if vc.dataset_kind(ds) == 'lyft':
        imgs = lyft_front_images([c['fid'] for c in chosen])
        for c in chosen:
            c['image'] = imgs[c['fid']]
    print('%-22s %s (point-checked %d candidates)' % (name, [(c['fid'][-40:], c['n_car'], c['n_ped']) for c in chosen], tried))
    return dict(config=cfg_path, training=training, frames=chosen)


if __name__ == '__main__':
    names = sys.argv[1:] or list(SETS)
    out = json.load(open(OUT)) if OUT.exists() else {}
    for n in names:
        out[n] = select(n, *SETS[n])
    json.dump(out, open(OUT, 'w'), indent=1)
    print('wrote', OUT)
