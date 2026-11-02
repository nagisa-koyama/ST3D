"""TRAIN_FACTOR_SWAP on Waymo (pcdet/datasets/waymo/waymo_factor_swap.py and its hook in waymo_dataset.__getitem__): each
arm changes exactly its factor, is fixed per frame, reads MODEL class names (Waymo's raw 'Vehicle' is not one), and the
key is inert when absent and at evaluation. The FACTOR-SWAP N->W rows (experiments_md 20261003_05 §5 item 24, Waymo
port) rely on it. Sentinel values (bug_prevention_checklist rule 1): IGNORE_CLASS_LABEL -99 placeholders for non-model
classes ('Sign' -> 'Misc') and ignored boxes, through the real prepare_data. CPU only.
"""
import copy
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.datasets.waymo import waymo_factor_swap as wfs  # noqa: E402
from pcdet.utils.common_utils import IGNORE_CLASS_LABEL  # noqa: E402
from pcdet.utils.ontology_mapping import get_ontology_mapping  # noqa: E402

NAMES = ['Car', 'Pedestrian', 'Cyclist']
W2K = get_ontology_mapping('waymo', 'kitti')
PIVOT = [1.43, 0.0, 2.184]


def _frame():
    """One Waymo-like frame: raw names with 'Sign' boxes BEFORE class boxes (the order that misaligned gt_classes)."""
    names = np.array(['Sign', 'Vehicle', 'Vehicle', 'Sign', 'Pedestrian', 'Cyclist'])
    boxes = np.array([[3, 8, 1.0, 0.5, 0.5, 1.0, 0],
                      [50, 0, 0.8, 4.5, 2.0, 1.6, 0.3],
                      [10, 2, 0.8, 4.6, 2.1, 1.7, 0.1],
                      [12, -6, 1.0, 0.5, 0.5, 1.0, 0],
                      [15, -3, 0.9, 0.8, 0.8, 1.8, 0],
                      [20, 4, 0.9, 1.8, 0.7, 1.7, 0]], dtype=np.float32)
    rng = np.random.default_rng(3)
    pts = [np.c_[b[:3] + rng.uniform(-0.3, 0.3, (20, 3)) * np.r_[b[3:5], b[5]] / 2,
                 np.zeros((20, 2))] for b in boxes]
    ground = np.c_[rng.uniform(-60, 60, (3000, 2)), np.zeros((3000, 1)), np.zeros((3000, 2))]
    points = np.concatenate(pts + [ground]).astype(np.float32)
    return names, boxes, points


# ---------------------------------------------------------------- arm selection and loader guards
def test_exactly_one_arm_and_known_keys_only():
    with pytest.raises(AssertionError):
        wfs.arm_of(EasyDict({}))
    with pytest.raises(AssertionError):
        wfs.arm_of(EasyDict({'PITCH_DEG': 1.0, 'PIVOT': PIVOT, 'GT_DIM_SCALE': {'Car': 1.1}}))
    with pytest.raises(AssertionError):                       # KITTI-only arms are refused, not silently ignored
        wfs.arm_of(EasyDict({'RANGE_KEEP': {'EDGES': [0, 1], 'RATE': [1]}}))
    with pytest.raises(AssertionError):                       # the pitch needs its pivot
        wfs.arm_of(EasyDict({'PITCH_DEG': 1.0}))
    with pytest.raises(AssertionError):
        wfs.arm_of(EasyDict({'REGION_THIN': {'SIDE': 'foreground', 'KEEP': 0.5}}))
    assert wfs.arm_of(EasyDict({'PITCH_DEG': [1.0, 2.0], 'PITCH_PROB': [0.5, 0.5], 'PIVOT': PIVOT})) == 'PITCH_DEG'


@pytest.mark.parametrize('extra', [{'MAX_SWEEPS': 2}, {'USE_PSEUDO_LABEL': True}, {'RING_PATTERN': {'X': 1}},
                                   {'TRAIN_TOP_THIN': {'X': 1}}, {'LIT_TRANSLATED': {'DIR': 'x'}},
                                   {'BEAM_DROP': {'X': 1}}, {'BEAM_DISTILL': {'X': 1}}, {'FOV_POINTS_ONLY': True}])
def test_loader_combinations_are_refused(extra):
    dcfg = EasyDict({'TRAIN_FACTOR_SWAP': {'GT_DIM_SCALE': {'Car': 1.1}}, **extra})
    with pytest.raises(AssertionError):
        wfs.check_dataset_cfg(dcfg, NAMES)


def test_raw_dataset_names_in_the_config_are_refused():
    """GT_DIM_SCALE / GT_IGNORE_BY_RANGE are keyed by MODEL class: a 'Vehicle' key would match nothing, silently."""
    wfs.check_dataset_cfg(EasyDict({'TRAIN_FACTOR_SWAP': {'GT_DIM_SCALE': {'Car': 1.1}}, 'MAX_SWEEPS': 1}), NAMES)
    with pytest.raises(AssertionError):
        wfs.check_dataset_cfg(EasyDict({'TRAIN_FACTOR_SWAP': {'GT_DIM_SCALE': {'Vehicle': 1.1}}}), NAMES)
    with pytest.raises(AssertionError):
        wfs.check_dataset_cfg(EasyDict({'TRAIN_FACTOR_SWAP': {
            'GT_IGNORE_BY_RANGE': {'CLASS': 'Vehicle', 'EDGES': [0, 1000], 'KEEP': [1.0]}}}), NAMES)


def test_model_names_follow_the_waymo_to_kitti_map():
    names, _, _ = _frame()
    assert wfs.model_names(names, W2K).tolist() == ['Misc', 'Car', 'Car', 'Misc', 'Pedestrian', 'Cyclist']
    assert wfs.model_names(np.array([], dtype='<U7'), W2K).shape == (0,)
    assert wfs.model_names(names, None).tolist() == names.tolist()


# ---------------------------------------------------------------- PITCH_DEG
def test_pitch_about_the_pivot():
    cfg = EasyDict({'PITCH_DEG': 2.0, 'PIVOT': PIVOT})
    piv = np.array([PIVOT + [0.7, 0.1]], np.float32)
    assert np.allclose(wfs.apply_points(piv, cfg, 'f0'), piv, atol=1e-6)          # the sensor stays put
    ground = np.array([[x, y, 0.0, 0.3, 0.2] for x in (5.0, 25.0, 45.0) for y in (-3.0, 4.0)], np.float32)
    out = wfs.apply_points(ground, cfg, 'f0')
    slope = np.degrees(np.arctan2(out[2::2, 2] - out[0:-2:2, 2], out[2::2, 0] - out[0:-2:2, 0]))
    assert np.allclose(slope, 2.0, atol=1e-4)                                      # flat ground rises ahead by d
    assert out[4, 2] > out[0, 2] > 0                                               # ahead rises (sensor 2.18 m up)
    assert np.array_equal(out[:, 1], ground[:, 1]) and np.array_equal(out[:, 3:], ground[:, 3:])   # y, features
    back = wfs.apply_points(np.array([[-20.0, 0, 0, 0, 0]], np.float32), cfg, 'f0')
    assert back[0, 2] < wfs.apply_points(np.array([[1.43, 0, 0, 0, 0]], np.float32), cfg, 'f0')[0, 2]


def test_pitch_moves_box_centres_with_the_points_and_keeps_them_upright():
    cfg = EasyDict({'PITCH_DEG': 1.5, 'PIVOT': PIVOT})
    names, boxes, _ = _frame()
    boxes9 = np.c_[boxes, np.full((len(boxes), 2), 7.0, np.float32)]          # extra columns (e.g. velocity)
    ob = wfs.apply_boxes(boxes9, wfs.model_names(names, W2K), cfg, 'f1')
    centres = wfs.apply_points(np.c_[boxes[:, :3], np.zeros((len(boxes), 2))].astype(np.float32), cfg, 'f1')
    assert np.allclose(ob[:, :3], centres[:, :3], atol=1e-5)
    assert np.array_equal(ob[:, 3:], boxes9[:, 3:])                               # size, yaw, extra columns
    assert not np.allclose(ob[:, :3], boxes[:, :3])


def test_pitch_mixture_draws_one_angle_per_frame_for_points_and_boxes():
    cfg = EasyDict({'PITCH_DEG': [2.18, 1.24], 'PITCH_PROB': [0.558, 0.442], 'PIVOT': PIVOT})
    pt = np.array([[21.43, 0.0, 2.184, 0, 0]], np.float32)                      # 20 m ahead of the pivot, level
    bx = np.array([[21.43, 0.0, 2.184, 4.0, 1.8, 1.5, 0.0]], np.float32)
    angles = []
    for i in range(400):
        fid = 'segment-x_%03d' % i
        zp = wfs.apply_points(pt, cfg, fid)[0, 2]
        zb = wfs.apply_boxes(bx, np.array(['Car']), cfg, fid)[0, 2]
        assert zp == pytest.approx(zb, abs=1e-6)
        assert wfs.apply_points(pt, cfg, fid)[0, 2] == zp                         # fixed per frame
        angles.append(np.degrees(np.arcsin((zp - 2.184) / 20.0)))
    angles = np.round(angles, 2)
    assert set(np.unique(angles)) == {2.18, 1.24}
    assert abs(np.mean(angles == 2.18) - 0.558) < 0.08


# ---------------------------------------------------------------- GT_DIM_SCALE
def test_dim_scale_uses_model_names_and_keeps_the_bottom():
    cfg = EasyDict({'GT_DIM_SCALE': {'Car': 1.1, 'Pedestrian': 0.95}})
    names, boxes, points = _frame()
    ob = wfs.apply_boxes(boxes, wfs.model_names(names, W2K), cfg, 'f2')
    f = np.array([1.0, 1.1, 1.1, 1.0, 0.95, 1.0], np.float32)                   # Sign, Car, Car, Sign, Ped, Cyc
    assert np.allclose(ob[:, 3:6], boxes[:, 3:6] * f[:, None])
    assert np.allclose(ob[:, 2] - ob[:, 5] / 2, boxes[:, 2] - boxes[:, 5] / 2, atol=1e-6)     # bottom fixed
    assert np.array_equal(ob[:, [0, 1, 6]], boxes[:, [0, 1, 6]])
    assert np.array_equal(wfs.apply_points(points, cfg, 'f2'), points)          # points untouched
    raw = wfs.apply_boxes(boxes, names, cfg, 'f2')                               # the trap: raw names scale nothing
    assert np.array_equal(raw[1:3], boxes[1:3])


# ---------------------------------------------------------------- GT_IGNORE_BY_RANGE
IGN = EasyDict({'GT_IGNORE_BY_RANGE': {'CLASS': 'Car', 'EDGES': [0, 30, 1000], 'KEEP': [1.0, 0.0]}})


def test_ignore_labels_from_model_names_only_touch_the_chosen_class():
    names, boxes, points = _frame()
    d = wfs.finish({'points': points, 'gt_boxes': boxes, 'gt_names': names, 'num_points_in_gt': None},
                   IGN, NAMES, W2K, 'f3')
    # Sign -> Misc placeholder (removed by the class filter), far Car -> -99, near Car 1, Pedestrian 2, Cyclist 3
    assert d['gt_classes'].tolist() == [IGNORE_CLASS_LABEL, IGNORE_CLASS_LABEL, 1, IGNORE_CLASS_LABEL, 2, 3]
    assert d['gt_scores'].tolist() == [1.0] * 6
    assert np.array_equal(d['points'], points) and np.array_equal(d['gt_boxes'], boxes)
    keep_all = EasyDict({'GT_IGNORE_BY_RANGE': {'CLASS': 'Car', 'EDGES': [0, 1000], 'KEEP': [1.0]}})
    d1 = wfs.finish({'points': points, 'gt_boxes': boxes, 'gt_names': names}, keep_all, NAMES, W2K, 'f3')
    assert d1['gt_classes'].tolist() == [IGNORE_CLASS_LABEL, 1, 1, IGNORE_CLASS_LABEL, 2, 3]
    # The trap the model-name mapping avoids: with raw names every Vehicle would become -99 even at KEEP = 1.
    from pcdet.datasets.kitti.kitti_factor_swap import ignore_labels
    raw, _ = ignore_labels(boxes, names, NAMES, keep_all, 'f3')
    assert raw.tolist()[1:3] == [IGNORE_CLASS_LABEL, IGNORE_CLASS_LABEL]


def test_ignore_fraction_and_determinism():
    cfg = EasyDict({'GT_IGNORE_BY_RANGE': {'CLASS': 'Car', 'EDGES': [0, 1000], 'KEEP': [0.3]}})
    boxes = np.tile(np.array([[20.0, 0, 0, 4, 2, 1.5, 0]], np.float32), (2000, 1))
    names = np.array(['Vehicle'] * 2000)
    a = wfs.finish({'points': np.zeros((0, 5), np.float32), 'gt_boxes': boxes, 'gt_names': names}, cfg, NAMES, W2K,
                   's_001')['gt_classes']
    b = wfs.finish({'points': np.zeros((0, 5), np.float32), 'gt_boxes': boxes, 'gt_names': names}, cfg, NAMES, W2K,
                   's_001')['gt_classes']
    assert np.array_equal(a, b)
    assert abs(np.mean(a == 1) - 0.3) < 0.04 and set(np.unique(a)) == {1, IGNORE_CLASS_LABEL}


def _template():
    """The real prepare_data for a Waymo dataset under the 'kitti' model ontology, augmentation and processor off."""
    cfg = EasyDict({
        'POINT_CLOUD_RANGE': [-75.2, -75.2, -2, 75.2, 75.2, 4],
        'POINT_FEATURE_ENCODING': {'encoding_type': 'absolute_coordinates_encoding',
                                   'used_feature_list': ['x', 'y', 'z'],
                                   'src_feature_list': ['x', 'y', 'z', 'intensity', 'elongation']},
        'ONTOLOGY': 'waymo', 'DATA_PROCESSOR': [], 'MIN_POINTS_OF_GT': 1,
        'DATA_AUGMENTOR': {'DISABLE_AUG_LIST': [], 'AUG_CONFIG_LIST': []},
    })
    return DatasetTemplate(dataset_cfg=cfg, class_names=list(NAMES), training=True, root_path=Path('/tmp'),
                           logger=mock.MagicMock(), model_ontology='kitti')


def _prepared(ds, swap, frame_id='seg_000'):
    names, boxes, points = _frame()
    d = {'points': points, 'gt_boxes': boxes.copy(), 'gt_names': names.copy(), 'frame_id': frame_id,
         'num_points_in_gt': np.full(len(boxes), 20, np.int32)}
    if swap is not None:
        d = wfs.finish(d, swap, NAMES, ds.map_ontology_dataset_to_model, frame_id)
    return ds.prepare_data(d)


def test_ignore_route_through_prepare_data_stays_aligned_with_sign_boxes_first():
    """Bug-prevention rule 1 against the real consumer: KEEP = 1 must reproduce the plain route's labels exactly, with
    'Sign' boxes (non-model class) BEFORE class boxes; KEEP 0 beyond 30 m turns only the far Car into -99."""
    ds = _template()
    base = _prepared(ds, None)['gt_boxes']
    keep_all = EasyDict({'GT_IGNORE_BY_RANGE': {'CLASS': 'Car', 'EDGES': [0, 1000], 'KEEP': [1.0]}})
    same = _prepared(ds, keep_all)['gt_boxes']
    assert base[:, 7].tolist() == [1, 1, 2, 3]
    assert np.array_equal(same, base)
    far = _prepared(ds, IGN)['gt_boxes']
    assert far[:, 7].tolist() == [IGNORE_CLASS_LABEL, 1, 2, 3]
    assert np.array_equal(far[:, :7], base[:, :7])


def test_ignore_boxes_survive_the_processor_recount():
    """drop_empty_gt_boxes (RS's processor recount) keeps -99 boxes whatever they hold."""
    from pcdet.datasets.processor.data_processor import DataProcessor
    proc = DataProcessor([EasyDict({'NAME': 'drop_empty_gt_boxes', 'MIN_POINTS': 1})],
                         point_cloud_range=np.array([-75.2, -75.2, -2, 75.2, 75.2, 4]), training=True,
                         num_point_features=3)
    boxes = np.array([[10, 0, 1, 4, 2, 1.5, 0, IGNORE_CLASS_LABEL], [20, 0, 1, 4, 2, 1.5, 0, 1],
                      [30, 0, 1, 4, 2, 1.5, 0, 1]], np.float32)
    pts = np.array([[30, 0, 1]], np.float32)
    out = proc.forward({'points': pts, 'gt_boxes': boxes})
    assert out['gt_boxes'][:, 7].tolist() == [IGNORE_CLASS_LABEL, 1]


# ---------------------------------------------------------------- REGION_THIN and the differential recount
def test_region_thin_recount_drops_only_the_boxes_the_arm_emptied():
    boxes = np.array([[10, 0, 0, 2, 2, 2, 0], [20, 0, 0, 2, 2, 2, 0], [30, 0, 0, 2, 2, 2, 0]], np.float32)
    names = np.array(['Vehicle', 'Vehicle', 'Vehicle'])
    one = np.array([[10, 0, 0, 0, 0]], np.float32)                               # box 0 holds one point
    many = np.c_[np.random.default_rng(0).uniform(19.5, 20.5, (40, 1)), np.zeros((40, 4))].astype(np.float32)
    bg = np.c_[np.random.default_rng(1).uniform(40, 70, (500, 1)), np.zeros((500, 4))].astype(np.float32)
    points = np.concatenate([one, many, bg])                                     # box 2: stored 7, no point here
    stored = np.array([3, 40, 7], np.int64)
    emptied = []
    for i in range(40):
        d = wfs.finish({'points': points, 'gt_boxes': boxes, 'gt_names': names, 'num_points_in_gt': stored.copy()},
                       EasyDict({'REGION_THIN': {'SIDE': 'foreground', 'KEEP': 0.5}, 'BOX_MARGIN': 0.25}),
                       NAMES, W2K, 'seg_%03d' % i)
        n = d['num_points_in_gt']
        assert n[2] == 7                                  # empty before the arm: the oracle's stored count stands
        assert n[1] == 40                                 # thinned, not emptied: stored count stands
        assert n[0] in (0, 3)
        emptied.append(n[0] == 0)
        assert np.sum(d['points'][:, 0] > 35) == 500      # background untouched
    assert 0 < np.mean(emptied) < 1                       # the 1-point box is emptied in about half the frames
    d = wfs.finish({'points': points, 'gt_boxes': boxes, 'gt_names': names, 'num_points_in_gt': stored.copy()},
                   EasyDict({'REGION_THIN': {'SIDE': 'background', 'KEEP': 0.5}, 'BOX_MARGIN': 0.25}),
                   NAMES, W2K, 'seg_000')
    assert d['num_points_in_gt'].tolist() == [3, 40, 7]  # background thinning never empties a box
    assert np.sum(d['points'][:, 0] > 35) == 250 and np.sum(d['points'][:, 0] < 35) == 41


def test_recount_emptied_without_a_stored_count():
    assert wfs.recount_emptied(None, [2, 0], [0, 0]).tolist() == [0, 0]
    assert wfs.recount_emptied([5, 6, 7], [2, 0, 1], [0, 0, 1]).tolist() == [0, 6, 7]


# ---------------------------------------------------------------- the hook in WaymoDataset.__getitem__
def _hooked_dataset(dataset_cfg, training=True):
    from pcdet.datasets.waymo.waymo_dataset import WaymoDataset
    names, boxes, points = _frame()
    info = {'point_cloud': {'lidar_sequence': 'seg', 'sample_idx': 4}, 'frame_id': 'seg_004',
            'num_points_of_each_lidar': [len(points), 0, 0, 0, 0],
            'annos': {'name': np.r_[names, ['unknown']],
                      'gt_boxes_lidar': np.r_[boxes, [[1, 1, 1, 1, 1, 1, 0]]].astype(np.float32),
                      'num_points_in_gt': np.r_[np.full(len(boxes), 20), [0]].astype(np.int32),
                      'difficulty': np.zeros(len(boxes) + 1, np.int32)}}
    ds = object.__new__(WaymoDataset)
    ds.dataset_cfg = EasyDict(dataset_cfg)
    ds.training = training
    ds.class_names = list(NAMES)
    ds.map_ontology_dataset_to_model = W2K
    ds._merge_all_iters_to_one_epoch = False
    ds._ring_labels_from_order = False
    ds.ring_pattern_cfg = None
    ds.train_top_thin_cfg = None
    ds.lit_cfg = None
    ds.infos = [info]
    ds.get_lidar_with_sweeps = lambda info_: points.copy()
    seen = {}

    def prepare_data(data_dict):
        seen.update(copy.deepcopy(data_dict))
        return data_dict
    ds.prepare_data = prepare_data
    return ds, seen, (names, boxes, points)


BASE_DCFG = {'SHIFT_COOR': [0.0, 0.0, 0.0], 'MAX_SWEEPS': 1}


def test_hook_absent_key_and_evaluation_are_inert():
    ds, base, _ = _hooked_dataset(BASE_DCFG)
    ds[0]
    for cfg, training in ((BASE_DCFG, True),
                          ({**BASE_DCFG, 'TRAIN_FACTOR_SWAP': {'PITCH_DEG': 2.0, 'PIVOT': PIVOT}}, False),
                          ({**BASE_DCFG, 'TRAIN_FACTOR_SWAP': IGN}, False),
                          ({**BASE_DCFG, 'TRAIN_FACTOR_SWAP': {'REGION_THIN': {'SIDE': 'background', 'KEEP': 0.5},
                                                               'BOX_MARGIN': 0.25}}, False)):
        ds2, seen, _ = _hooked_dataset(cfg, training=training)
        ds2[0]
        assert set(seen) == set(base)
        for k in base:
            assert np.array_equal(np.asarray(seen[k]), np.asarray(base[k])), k


def test_hook_pitch_rotates_points_and_boxes_and_keeps_the_stored_count():
    swap = {'PITCH_DEG': 2.0, 'PIVOT': PIVOT}
    ds, seen, (names, boxes, points) = _hooked_dataset({**BASE_DCFG, 'TRAIN_FACTOR_SWAP': swap})
    ds[0]
    assert np.allclose(seen['points'], wfs.apply_points(points, EasyDict(swap), 'seg_004'))
    assert np.allclose(seen['gt_boxes'], wfs.apply_boxes(boxes, wfs.model_names(names, W2K), EasyDict(swap), 'seg_004'))
    assert seen['num_points_in_gt'].tolist() == [20] * len(boxes)
    assert seen['gt_names'].tolist() == names.tolist()                       # raw names: prepare_data maps them


def test_hook_dim_scale_and_ignore_and_region_thin():
    ds, seen, (names, boxes, points) = _hooked_dataset({**BASE_DCFG, 'TRAIN_FACTOR_SWAP': {
        'GT_DIM_SCALE': {'Car': 1.2}}})
    ds[0]
    assert np.allclose(seen['gt_boxes'][1:3, 3:6], boxes[1:3, 3:6] * 1.2)
    assert np.array_equal(seen['gt_boxes'][[0, 3, 4, 5]], boxes[[0, 3, 4, 5]]) and np.array_equal(seen['points'], points)
    ds, seen, _ = _hooked_dataset({**BASE_DCFG, 'TRAIN_FACTOR_SWAP': IGN})
    ds[0]
    assert seen['gt_classes'].tolist() == [IGNORE_CLASS_LABEL, IGNORE_CLASS_LABEL, 1, IGNORE_CLASS_LABEL, 2, 3]
    ds, seen, _ = _hooked_dataset({**BASE_DCFG, 'TRAIN_FACTOR_SWAP': {
        'REGION_THIN': {'SIDE': 'foreground', 'KEEP': 0.5}, 'BOX_MARGIN': 0.25}})
    ds[0]
    from pcdet.datasets.osens2_utils import enlarge_boxes, in_any_box
    pool = int(in_any_box(points, enlarge_boxes(boxes, 0.25)).sum())             # 120 object points + ground inside
    assert pool >= 120 and len(seen['points']) == len(points) - (pool - int(round(pool * 0.5)))
    assert seen['num_points_in_gt'].tolist() == [20] * len(boxes)                # 10 left in each box: none emptied


def test_hook_refuses_accumulation():
    ds, _, _ = _hooked_dataset({'SHIFT_COOR': [0, 0, 0], 'MAX_SWEEPS': 3,
                                'TRAIN_FACTOR_SWAP': {'GT_DIM_SCALE': {'Car': 1.2}}})
    with pytest.raises(AssertionError):
        ds[0]


# ---------------------------------------------------------------- the row configs
ROWS = {
    'rs-toponly': {'LIDAR_INDICES', 'DATA_PROCESSOR'},
    'r5-pitch': {'TRAIN_FACTOR_SWAP'},
    'r6-labelsize': {'TRAIN_FACTOR_SWAP'},
    'r7-labelrange': {'TRAIN_FACTOR_SWAP'},
    'r8-fg050': {'TRAIN_FACTOR_SWAP'},
    'r9-bg050': {'TRAIN_FACTOR_SWAP'},
}


def _load(path, monkeypatch):
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(Path(__file__).resolve().parent.parent / 'tools')
    return cfg_from_yaml_file(path, EasyDict())


@pytest.mark.parametrize('row', sorted(ROWS))
def test_row_config_differs_from_the_oracle_in_its_one_key(row, monkeypatch):
    base = _load('cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml', monkeypatch)
    cfg = _load('cfgs/da-ieee-access-analysis/factorswap-n2w/centerpoint-factorswap-n2w-%s.yaml' % row, monkeypatch)
    for key in ('POINT_CLOUD_RANGE', 'DATA_PROCESSOR', 'DATA_AUGMENTOR', 'POINT_FEATURE_ENCODING', 'DATA_SPLIT'):
        assert cfg.DATA_CONFIG.get(key, None) is not None, key           # resolved to a complete block
    differ = {k for k in set(cfg.DATA_CONFIG) | set(base.DATA_CONFIG)
              if k != '_BASE_CONFIG_' and cfg.DATA_CONFIG.get(k, None) != base.DATA_CONFIG.get(k, None)}
    assert differ == ROWS[row]
    assert cfg.DATA_CONFIG_TAR == base.DATA_CONFIG_TAR                    # evaluation untouched
    assert cfg.MODEL == base.MODEL and cfg.OPTIMIZATION == base.OPTIMIZATION and cfg.CLASS_NAMES == base.CLASS_NAMES
    if 'TRAIN_FACTOR_SWAP' in differ:
        wfs.check_dataset_cfg(cfg.DATA_CONFIG, cfg.CLASS_NAMES)
    else:
        assert list(cfg.DATA_CONFIG.LIDAR_INDICES) == [0]
        names = [p.NAME for p in cfg.DATA_CONFIG.DATA_PROCESSOR]
        assert names == ['mask_points_and_boxes_outside_range', 'shuffle_points', 'drop_empty_gt_boxes',
                         'transform_points_to_voxels']
        stripped = [p for p in cfg.DATA_CONFIG.DATA_PROCESSOR if p.NAME != 'drop_empty_gt_boxes']
        assert stripped == list(base.DATA_CONFIG.DATA_PROCESSOR)
