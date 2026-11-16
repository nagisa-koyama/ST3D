"""pcseg nuScenes-lidarseg (G0, experiments_md episodes/20261116_01 §5): the 16-class map against two independent
sources, the -1 sentinel, the frame index against the devkit's split, and the reader over >= 40 frames.

The map is checked against
- the official nuscenes-devkit fine -> coarse table, BY NAME (transcribed below from python-sdk/nuscenes/eval/lidarseg/
  utils.py LidarsegClassMapper, file at c44366d 2021-08-03), joined to lidarseg indices through the index -> fine-name
  table (transcribed below, and checked against the category.json on disk);
- Pointcept's index-based learning_map, parsed from its source on disk (pointcept/datasets/nuscenes.py).
"""
import ast
import os

import numpy as np
import pytest

from pcseg import IGNORE
from pcseg.class_map import build_class_map, NUSCENES_LIDARSEG16_NAMES

LIDARSEG = '/home/koyama/data/nuscenes_lidarseg_v1_0'
NUSCENES = '/home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval'
INDEX = '/home/koyama/data/pcseg/nuscenes_lidarseg_index_v1.json'
POINTCEPT_NUSCENES = '/home/koyama/data/released_repos/Pointcept/pointcept/datasets/nuscenes.py'

# lidarseg category index -> fine name (v1.0-trainval/category.json; also TTYD datasets/nuscenes.py class_mapping_ns).
INDEX_TO_FINE = {
    0: 'noise', 1: 'animal', 2: 'human.pedestrian.adult', 3: 'human.pedestrian.child',
    4: 'human.pedestrian.construction_worker', 5: 'human.pedestrian.personal_mobility',
    6: 'human.pedestrian.police_officer', 7: 'human.pedestrian.stroller', 8: 'human.pedestrian.wheelchair',
    9: 'movable_object.barrier', 10: 'movable_object.debris', 11: 'movable_object.pushable_pullable',
    12: 'movable_object.trafficcone', 13: 'static_object.bicycle_rack', 14: 'vehicle.bicycle', 15: 'vehicle.bus.bendy',
    16: 'vehicle.bus.rigid', 17: 'vehicle.car', 18: 'vehicle.construction', 19: 'vehicle.emergency.ambulance',
    20: 'vehicle.emergency.police', 21: 'vehicle.motorcycle', 22: 'vehicle.trailer', 23: 'vehicle.truck',
    24: 'flat.driveable_surface', 25: 'flat.other', 26: 'flat.sidewalk', 27: 'flat.terrain', 28: 'static.manmade',
    29: 'static.other', 30: 'static.vegetation', 31: 'vehicle.ego',
}
# nuscenes-devkit LidarsegClassMapper.get_fine2coarse (None = its 'ignore') and get_coarse2idx (1..16; 0 = ignore).
DEVKIT_FINE2COARSE = {
    'noise': None, 'human.pedestrian.adult': 'pedestrian', 'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': None, 'human.pedestrian.stroller': None, 'human.pedestrian.personal_mobility': None,
    'human.pedestrian.police_officer': 'pedestrian', 'human.pedestrian.construction_worker': 'pedestrian',
    'animal': None, 'vehicle.car': 'car', 'vehicle.motorcycle': 'motorcycle', 'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus', 'vehicle.bus.rigid': 'bus', 'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle', 'vehicle.emergency.ambulance': None,
    'vehicle.emergency.police': None, 'vehicle.trailer': 'trailer', 'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone', 'movable_object.pushable_pullable': None,
    'movable_object.debris': None, 'static_object.bicycle_rack': None, 'flat.driveable_surface': 'driveable_surface',
    'flat.sidewalk': 'sidewalk', 'flat.terrain': 'terrain', 'flat.other': 'other_flat', 'static.manmade': 'manmade',
    'static.vegetation': 'vegetation', 'static.other': None, 'vehicle.ego': None,
}
DEVKIT_COARSE2IDX = {
    'barrier': 1, 'bicycle': 2, 'bus': 3, 'car': 4, 'construction_vehicle': 5, 'motorcycle': 6, 'pedestrian': 7,
    'traffic_cone': 8, 'trailer': 9, 'truck': 10, 'driveable_surface': 11, 'other_flat': 12, 'sidewalk': 13,
    'terrain': 14, 'manmade': 15, 'vegetation': 16,
}


def test_nuscenes16_equals_the_devkit_by_name():
    cm = build_class_map('nuscenes_lidarseg16')
    assert cm.names == sorted(DEVKIT_COARSE2IDX, key=DEVKIT_COARSE2IDX.get)     # model index = devkit index - 1
    raw = np.arange(32)
    want = [IGNORE if DEVKIT_FINE2COARSE[INDEX_TO_FINE[i]] is None
            else DEVKIT_COARSE2IDX[DEVKIT_FINE2COARSE[INDEX_TO_FINE[i]]] - 1 for i in raw]
    assert cm(raw).tolist() == want
    assert cm.names == NUSCENES_LIDARSEG16_NAMES


@pytest.mark.skipif(not os.path.isfile(os.path.join(LIDARSEG, 'v1.0-trainval', 'category.json')),
                    reason='nuScenes-lidarseg absent')
def test_index_to_fine_table_equals_category_json():
    import json
    cats = json.load(open(os.path.join(LIDARSEG, 'v1.0-trainval', 'category.json')))
    assert {c['index']: c['name'] for c in cats} == INDEX_TO_FINE


def _pointcept_learning_map():
    tree = ast.parse(open(POINTCEPT_NUSCENES).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'get_learning_map':
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    out = {}
                    for k, v in zip(sub.keys, sub.values):
                        out[k.value] = IGNORE if isinstance(v, ast.Name) else v.value      # ignore_index -> -1
                    return out
    raise AssertionError('get_learning_map not found')


@pytest.mark.skipif(not os.path.isfile(POINTCEPT_NUSCENES), reason='Pointcept source absent')
def test_nuscenes16_equals_pointcept_learning_map():
    lm = _pointcept_learning_map()
    assert sorted(lm) == list(range(32))
    assert build_class_map('nuscenes_lidarseg16')(np.arange(32)).tolist() == [lm[i] for i in range(32)]


def test_nuscenes16_sentinel_and_range():
    cm = build_class_map('nuscenes_lidarseg16')
    assert cm(np.array([0, 31, 29, 17])).tolist() == [IGNORE, IGNORE, IGNORE, 3]
    for bad in ([32], [-1], [255]):
        with pytest.raises(ValueError):
            cm(np.array(bad))


_have_data = all(os.path.exists(p) for p in (INDEX, LIDARSEG, NUSCENES))


@pytest.mark.skipif(not _have_data, reason='nuScenes data or the pcseg index absent')
def test_index_matches_the_devkit_split():
    from nuscenes.utils import splits
    from pcseg.nuscenes_seg import load_index
    idx = load_index(INDEX)
    assert idx['meta']['counts'] == {'train': 28130, 'val': 6019}
    assert sorted({r[3] for r in idx['train']}) == sorted(splits.train)
    assert sorted({r[3] for r in idx['val']}) == sorted(splits.val)
    toks = [r[0] for k in ('train', 'val') for r in idx[k]]
    assert len(set(toks)) == len(toks)
    assert all(r[1].startswith('samples/LIDAR_TOP/') and r[2] == 'lidarseg/v1.0-trainval/%s_lidarseg.bin' % r[0]
               for k in ('train', 'val') for r in idx[k])


@pytest.mark.skipif(not _have_data, reason='nuScenes data or the pcseg index absent')
def test_reader_over_48_frames():
    """Points are the raw .pcd.bin (x, y, z unchanged, intensity / 255), labels are the raw lidarseg bytes through the
    map, one per point, for 24 frames spread over each split."""
    from pcseg.nuscenes_seg import load_index, read_frame
    idx = load_index(INDEX)
    cm = build_class_map('nuscenes_lidarseg16')
    n = 0
    for split in ('train', 'val'):
        rows = idx[split]
        for i in np.linspace(0, len(rows) - 1, 24).round().astype(int):
            tok, lidar_rel, label_rel, _ = rows[i]
            pts, raw = read_frame(NUSCENES, LIDARSEG, lidar_rel, label_rel)
            ref = np.fromfile(os.path.join(NUSCENES, lidar_rel), dtype=np.float32).reshape(-1, 5)
            lab = np.fromfile(os.path.join(LIDARSEG, label_rel), dtype=np.uint8)
            assert np.array_equal(pts[:, :3], ref[:, :3])
            assert np.allclose(pts[:, 3], ref[:, 3] / 255) and pts[:, 3].min() >= 0 and pts[:, 3].max() <= 1
            assert np.array_equal(raw, lab.astype(np.int64))
            seg = cm(raw)                                            # raises on an undeclared index
            assert ((seg >= 0) | (seg == IGNORE)).all() and seg.max() < 16
            n += 1
    assert n >= 40


@pytest.mark.skipif(not _have_data, reason='nuScenes data or the pcseg index absent')
def test_dataset_samples_train_and_eval():
    from pcseg.nuscenes_seg import NuScenesSeg
    cfg = {'ROOT': NUSCENES, 'LIDARSEG_ROOT': LIDARSEG, 'INDEX': INDEX, 'SHIFT_COOR': [0.0, 0.0, 0.0],
           'CLASS_MAP': 'nuscenes_lidarseg16', 'GRID_SIZE': 0.05,
           'AUG': {'ROTATE_Z_PROB': 0.5, 'SCALE': [0.9, 1.1], 'FLIP_PROB': 0.5, 'JITTER': [0.005, 0.02]}}
    ev = NuScenesSeg(dict(cfg, SPLIT='val'), False)
    tr = NuScenesSeg(dict(cfg, SPLIT='train'), True)
    assert len(ev) == 6019 and len(tr) == 28130 and len(ev.sequences) == 150
    s = ev[0]
    coord, _, seg, key = ev.load(0)
    assert key == ev.frames[0][0]
    assert len(s['origin_segment']) == len(coord) and len(s['inverse']) == len(coord)
    assert np.array_equal(s['origin_coord'], coord)
    assert (s['segment'][s['inverse']] == seg).mean() > 0.97            # a voxel's label is one of its points'
    assert np.array_equal(ev[0]['grid_coord'], s['grid_coord'])          # eval voxel draw is fixed per frame
    t = tr[0]
    assert t['feat'].shape[1] == 4 and t['feat'].dtype == np.float32 and 'inverse' not in t
