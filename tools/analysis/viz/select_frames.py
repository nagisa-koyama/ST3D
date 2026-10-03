"""Choose the common frames every run's figure shows (experiments_md/20261003_02 §4).

Per frame set, one frame per RANGE BAND (near 0-20 m, mid 20-40 m, far 40-60 m): a frame qualifies
for a band when it holds >= 1 Car and >= 1 Pedestrian label IN FRONT of the vehicle (box centre within
+-FRONT_DEG of its heading, so both are in the front image) with centre range inside the band, each
holding >= MIN_PTS points of the STORED sweep. Frames come from distinct drives, by a seeded draw. If
no frame in a band reaches MIN_PTS, the band is retried at MIN_PTS_FALLBACK and the frame records it.
Real labels decide only which frames are shown. (Rule updated by the user 2026-10-03 16:30: "car and
pedestrian in front, distributed across the range"; it replaced "anywhere in the grid".) Writes viz_frames.json next to this file; run from tools/
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
SEED, MIN_PTS, MIN_PTS_FALLBACK, FRONT_DEG = 0, 5, 3, 35.0
NEAR, MID, FAR = (0, 20), (20, 40), (40, 60)
# name: (dataset config, training split?, range bands). Two-platform sources: near + mid from the
# majority platform, far from the other (§4).
SETS = {
    'kitti/val': ('cfgs/da-ieee-access/da_kitti_dataset.yaml', False, [NEAR, MID, FAR]),
    'kitti/train': ('cfgs/da-ieee-access/da_kitti_dataset.yaml', True, [NEAR, MID, FAR]),
    'nuscenes/val': ('cfgs/da-ieee-access/da_nuscenes_dataset.yaml', False, [NEAR, MID, FAR]),
    'nuscenes/train/n008': ('cfgs/da-ieee-access/da_nuscenes_n008_dataset.yaml', True, [NEAR, MID]),
    'nuscenes/train/n015': ('cfgs/da-ieee-access/da_nuscenes_n015_dataset.yaml', True, [FAR]),
    'lyft/train/40': ('cfgs/da-ieee-access/da_lyft40_dataset.yaml', True, [NEAR, MID]),
    'lyft/train/64': ('cfgs/da-ieee-access/da_lyft64_dataset.yaml', True, [FAR]),
    'waymo/val': ('cfgs/da-ieee-access/da_waymo_dataset.yaml', False, [NEAR, MID, FAR]),
    'waymo/train': ('cfgs/da-ieee-access/da_waymo_dataset.yaml', True, [NEAR, MID, FAR]),
}
# PandaSet's two sensors share timestamps: one selection, valid on BOTH clouds, written to both keys.
JOINT = {
    'pandaset/val': (('cfgs/da-ieee-access/da_pandaset_spin_dataset.yaml', 'spin'),
                     ('cfgs/da-ieee-access/da_pandaset_flash_dataset.yaml', 'flash'), False, [NEAR, MID, FAR]),
    'pandaset/train': (('cfgs/da-ieee-access/da_pandaset_spin_dataset.yaml', 'spin'),
                       ('cfgs/da-ieee-access/da_pandaset_flash_dataset.yaml', 'flash'), True, [NEAR, MID, FAR]),
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


def front_in_band(ds, info, band):
    """Mask of labels in front of the vehicle with centre range inside `band`, and the names."""
    boxes, names = vc.raw_labels(ds, info)
    if not len(boxes):
        return boxes, names, np.zeros(0, bool)
    az = np.degrees(np.arctan2(boxes[:, 1], boxes[:, 0])) - vc.forward_yaw(ds, info)
    az = (az + 180.0) % 360.0 - 180.0
    r = np.linalg.norm(boxes[:, :2], axis=1)
    return boxes, names, (np.abs(az) <= FRONT_DEG) & (r >= band[0]) & (r < band[1])


def pick(ds, I, order, band, groups, min_pts, also=()):
    """`also`: further datasets over the SAME infos (PandaSet's other sensor) that must qualify too."""
    tried = 0
    for i in order:
        info = I[i]
        boxes, names, m = front_in_band(ds, info, band)
        if not ((names[m] == vc.CAR).any() and (names[m] == vc.PED).any()):
            continue
        g = vc.frame_group(ds, info)
        if g in groups:
            continue
        tried += 1
        ok = m & (vc.points_per_box(vc.raw_points(ds, info), boxes) >= min_pts)
        for other in also:
            ok &= vc.points_per_box(vc.raw_points(other, vc.infos(other)[i]), boxes) >= min_pts
        n_car, n_ped = int((ok & (names == vc.CAR)).sum()), int((ok & (names == vc.PED)).sum())
        if n_car and n_ped:
            r = np.linalg.norm(boxes[ok, :2], axis=1)
            return dict(fid=vc.frame_id(ds, info), group=str(g), image=vc.image_path(ds, info),
                        yaw=vc.forward_yaw(ds, info), band=list(band), min_pts=min_pts, n_car=n_car,
                        n_ped=n_ped, ranges=[round(float(x), 1) for x in sorted(r)]), tried
    return None, tried


def select(name, cfg_path, training, bands):
    dc = EasyDict()
    cfg_from_yaml_file(cfg_path, dc)
    ds = vc.build_dataset(dc, ['Car', 'Pedestrian', 'Cyclist'], training, 'kitti')
    I = vc.infos(ds)
    order = np.random.RandomState(SEED).permutation(len(I))
    chosen, groups, tried = [], set(), 0
    for band in bands:
        c, t = pick(ds, I, order, band, groups, MIN_PTS)
        tried += t
        if c is None:
            c, t = pick(ds, I, order, band, groups, MIN_PTS_FALLBACK)
            tried += t
        assert c is not None, '%s: no frame with a Car and a Pedestrian in front at %s m' % (name, band)
        chosen.append(c); groups.add(c['group'])
    if vc.dataset_kind(ds) == 'waymo':
        for c in chosen:
            seq, idx = c['fid'].rsplit('/', 1)
            c['image'] = vc.waymo_front_image(seq, int(idx))
    if vc.dataset_kind(ds) == 'lyft':
        imgs = lyft_front_images([c['fid'] for c in chosen])
        for c in chosen:
            c['image'] = imgs[c['fid']]
    print('%-20s %s (point-checked %d)' % (name, ['%s %s m, %d car %d ped >=%d pts' % (
        c['fid'][-28:], c['band'], c['n_car'], c['n_ped'], c['min_pts']) for c in chosen], tried))
    return dict(config=cfg_path, training=training, frames=chosen)


def select_joint(name, a, b, training, bands):
    dss = []
    for cfg_path, _ in (a, b):
        dc = EasyDict()
        cfg_from_yaml_file(cfg_path, dc)
        dss.append(vc.build_dataset(dc, ['Car', 'Pedestrian', 'Cyclist'], training, 'kitti'))
    I = vc.infos(dss[0])
    assert [vc.frame_id(dss[0], x) for x in I] == [vc.frame_id(dss[1], x) for x in vc.infos(dss[1])], 'sensor infos differ'
    order = np.random.RandomState(SEED).permutation(len(I))
    chosen, groups = [], set()
    for band in bands:
        c, _ = pick(dss[0], I, order, band, groups, MIN_PTS, also=dss[1:])
        if c is None:
            c, _ = pick(dss[0], I, order, band, groups, MIN_PTS_FALLBACK, also=dss[1:])
        assert c is not None, '%s: no frame at %s m valid on both sensors' % (name, band)
        chosen.append(c); groups.add(c['group'])
    print('%-20s %s (both sensors)' % (name, ['%s %s m, %d car %d ped >=%d pts' % (
        c['fid'], c['band'], c['n_car'], c['n_ped'], c['min_pts']) for c in chosen]))
    return {'%s/%s' % (name, plat): dict(config=cfg_path, training=training, frames=chosen) for cfg_path, plat in (a, b)}


if __name__ == '__main__':
    names = sys.argv[1:] or list(SETS) + list(JOINT)
    out = json.load(open(OUT)) if OUT.exists() else {}
    for n in names:
        if n in JOINT:
            out.update(select_joint(n, *JOINT[n]))
            continue
        out[n] = select(n, *SETS[n])
    json.dump(out, open(OUT, 'w'), indent=1)
    print('wrote', OUT)
