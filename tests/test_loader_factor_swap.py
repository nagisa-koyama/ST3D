"""TRAIN_FACTOR_SWAP on PandaSet and Lyft (pcdet/datasets/loader_factor_swap.py and its hooks in pandaset_dataset /
lyft_dataset __getitem__), for the factor-swap rows of experiments_md 20261003_05 §5 item 34. Each arm changes exactly
its factor, is fixed per FRAME KEY (PandaSet: sequence + frame, since frame ids repeat across sequences), reads MODEL
class names, keeps the oracle's box set where it moves points or redraws boxes, and the key is inert when absent and at
evaluation. Sentinel values (bug_prevention_checklist rule 1): IGNORE_CLASS_LABEL -99 for non-model classes and ignored
boxes, through the real prepare_data (class mask, augmentor mask, drop_empty_gt_boxes) and into CenterHead's targets;
the pseudo-label route (negative labels) is refused with the key. CPU only.
"""
import copy
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets import loader_factor_swap as lfs  # noqa: E402
from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.datasets.osens2_utils import enlarge_boxes, in_any_box  # noqa: E402
from pcdet.datasets.waymo import waymo_factor_swap as wfs  # noqa: E402
from pcdet.utils.common_utils import IGNORE_CLASS_LABEL as IGN  # noqa: E402
from pcdet.utils.ontology_mapping import get_ontology_mapping  # noqa: E402

NAMES = ['Car', 'Pedestrian', 'Cyclist']
P2K = get_ontology_mapping('pandaset', 'kitti')
L2K = get_ontology_mapping('lyft', 'kitti')
PIVOT = [0.0, 0.0, 1.88]
SHIFT = {'pandaset': 0.30, 'lyft': 1.87}


def _frame(loader):
    """One frame in each loader's RAW vocabulary, with non-model-class boxes BEFORE class boxes (the order that
    misaligned gt_classes before ST3D 12b5147): PandaSet 'Pickup Truck' -> Truck, 'Cones' -> Misc; Lyft 'truck' ->
    Truck, 'bus' -> Misc. Boxes in the raw frame (z of the box centre ~ ground + dz / 2 below the sensor for Lyft)."""
    if loader == 'pandaset':
        names = np.array(['Pickup Truck', 'Car', 'Car', 'Cones', 'Pedestrian with Object', 'Bicycle'])
        z0 = 0.0
    else:
        names = np.array(['truck', 'car', 'car', 'bus', 'pedestrian', 'bicycle'])
        z0 = -1.8
    boxes = np.array([[3, 8, z0 + 1.0, 5.0, 2.0, 2.0, 0],
                      [50, 0, z0 + 0.8, 4.5, 2.0, 1.6, 0.3],
                      [10, 2, z0 + 0.8, 4.6, 2.1, 1.7, 0.1],
                      [12, -6, z0 + 0.3, 0.4, 0.4, 0.6, 0],
                      [15, -3, z0 + 0.9, 0.8, 0.8, 1.8, 0],
                      [20, 4, z0 + 0.9, 1.8, 0.7, 1.7, 0]], dtype=np.float32)
    rng = np.random.default_rng(3)
    pts = [np.c_[b[:3] + rng.uniform(-0.3, 0.3, (20, 3)) * b[3:6], np.zeros((20, 1))] for b in boxes]
    ground = np.c_[rng.uniform(-60, 60, (3000, 2)), np.full((3000, 1), z0), np.zeros((3000, 1))]
    points = np.concatenate(pts + [ground]).astype(np.float32)
    return names, boxes, points


# ---------------------------------------------------------------- arm selection and loader guards
def test_arms_per_loader():
    pitch = EasyDict({'PITCH_DEG': 1.0, 'PIVOT': PIVOT})
    assert lfs.arm_of(pitch, 'pandaset') == 'PITCH_DEG'
    assert lfs.arm_of(pitch, 'lyft') == 'PITCH_DEG'             # since item 37d (Lyft R5); item 34 refused it
    for loader in ('pandaset', 'lyft'):
        with pytest.raises(AssertionError):
            lfs.arm_of(EasyDict({}), loader)
        with pytest.raises(AssertionError):                    # two arms
            lfs.arm_of(EasyDict({'GT_DIM_SCALE': {'Car': 1.1}, 'REGION_THIN': {'SIDE': 'background', 'KEEP': 0.5},
                                 'BOX_MARGIN': 0.25}), loader)
        with pytest.raises(AssertionError):                    # KITTI-only arm
            lfs.arm_of(EasyDict({'RANGE_KEEP': {'EDGES': [0, 1], 'RATE': [1]}}), loader)
        with pytest.raises(AssertionError):                    # REGION_THIN needs its margin
            lfs.arm_of(EasyDict({'REGION_THIN': {'SIDE': 'background', 'KEEP': 0.5}}), loader)


@pytest.mark.parametrize('loader,extra', [
    ('pandaset', {'MAX_SWEEPS': 2}), ('pandaset', {'USE_PSEUDO_LABEL': True}), ('pandaset', {'RING_PATTERN': {'X': 1}}),
    ('pandaset', {'RING_FOV_CUT': {'X': 1}}), ('pandaset', {'ROW_EXTRUSION': {'X': 1}}),
    ('lyft', {'MAX_SWEEPS': 2}), ('lyft', {'USE_PSEUDO_LABEL': True}), ('lyft', {'RING_PATTERN': {'X': 1}}),
    ('lyft', {'LEVEL_COOR': {'X': 1}}), ('lyft', {'BEAM_DROP': {'X': 1}})])
def test_loader_combinations_are_refused(loader, extra):
    """USE_PSEUDO_LABEL is the route that carries NEGATIVE labels (the pseudo-label ignore band): never mixed in."""
    dcfg = EasyDict({'TRAIN_FACTOR_SWAP': {'GT_DIM_SCALE': {'Car': 1.1}}, **extra})
    with pytest.raises(AssertionError):
        lfs.check_dataset_cfg(dcfg, NAMES, loader)


@pytest.mark.parametrize('loader,raw', [('pandaset', 'Pickup Truck'), ('lyft', 'car')])
def test_raw_dataset_names_in_the_config_are_refused(loader, raw):
    lfs.check_dataset_cfg(EasyDict({'TRAIN_FACTOR_SWAP': {'GT_DIM_SCALE': {'Car': 1.1}}, 'MAX_SWEEPS': 1}), NAMES, loader)
    with pytest.raises(AssertionError):
        lfs.check_dataset_cfg(EasyDict({'TRAIN_FACTOR_SWAP': {'GT_DIM_SCALE': {raw: 1.1}}}), NAMES, loader)
    with pytest.raises(AssertionError):
        lfs.check_dataset_cfg(EasyDict({'TRAIN_FACTOR_SWAP': {
            'GT_IGNORE_BY_RANGE': {'CLASS': raw, 'EDGES': [0, 1000], 'KEEP': [1.0]}}}), NAMES, loader)
    with pytest.raises(AssertionError):
        lfs.check_dataset_cfg(EasyDict({'TRAIN_FACTOR_SWAP': {'GT_DIM_SCALE': {'Car': 1.1}}}),
                              ['kitti:Car', 'kitti:Pedestrian'], loader)


# ---------------------------------------------------------------- the arms on an input_dict (after SHIFT_COOR)
def _input(loader):
    names, boxes, points = _frame(loader)
    s = np.array([0, 0, SHIFT[loader]], np.float32)
    return {'points': points + np.r_[s, 0].astype(np.float32), 'gt_boxes': boxes + np.r_[s, [0] * 4].astype(np.float32),
            'gt_names': names.copy(), 'frame_id': 'f'}


def test_pitch_rotates_points_and_box_centres_about_the_pivot_together():
    d = _input('pandaset')
    cfg = EasyDict({'PITCH_DEG': 2.0, 'PIVOT': PIVOT})
    before = lfs.oracle_counts(d['points'], d['gt_boxes'])
    out = lfs.apply(copy.deepcopy(d), cfg, NAMES, P2K, '014/05', 'pandaset')
    assert np.allclose(out['points'], wfs.apply_points(d['points'], cfg, '014/05'))
    assert np.allclose(out['gt_boxes'][:, :3], wfs.apply_points(np.c_[d['gt_boxes'][:, :3], np.zeros(6)].astype(
        np.float32), cfg, '014/05')[:, :3], atol=1e-5)
    assert np.array_equal(out['gt_boxes'][:, 3:], d['gt_boxes'][:, 3:])       # upright: size and yaw kept
    assert out['num_points_in_gt'].tolist() == before.tolist()               # the ORACLE's count, not a recount
    # sign: flat ground 20 m ahead of the pivot rises by d (the ground rises ahead), behind it falls
    g = np.array([[20.0, 0, 0.0, 0], [-20.0, 0, 0.0, 0]], np.float32)
    gp = wfs.apply_points(g, cfg, 'x')
    assert gp[0, 2] > 0 > gp[1, 2]
    assert np.degrees(np.arctan2(gp[0, 2] - gp[1, 2], gp[0, 0] - gp[1, 0])) == pytest.approx(2.0, abs=1e-4)
    assert np.array_equal(gp[:, 1], g[:, 1])
    # the pivot itself stays put
    piv = np.array([PIVOT + [0.5]], np.float32)
    assert np.allclose(wfs.apply_points(piv, cfg, 'x'), piv, atol=1e-6)


def test_pitch_count_is_the_oracle_s_even_when_a_point_leaves_its_box():
    d = _input('pandaset')
    b = d['gt_boxes'][1]                                                      # the 50 m car, yaw 0.3
    c, s_ = np.cos(b[6]), np.sin(b[6])
    fx, fz = b[3] / 2 - 0.05, b[5] / 2 - 0.02                                 # its front top edge, inside the box
    edge = np.array([[b[0] + c * fx, b[1] + s_ * fx, b[2] + fz, 0]], np.float32)
    rest = d['points'][~in_any_box(d['points'], d['gt_boxes'][1:2])]
    d['points'] = np.concatenate([edge, rest])                                # the car keeps ONE point, at its edge
    before = lfs.oracle_counts(d['points'], d['gt_boxes'])
    assert before[1] == 1
    out = lfs.apply(copy.deepcopy(d), EasyDict({'PITCH_DEG': 3.0, 'PIVOT': PIVOT}), NAMES, P2K, 'k', 'pandaset')
    assert lfs.oracle_counts(out['points'], out['gt_boxes'])[1] == 0          # pitched, the corner point falls out
    assert out['num_points_in_gt'][1] == 1                                    # but the oracle trains on this box


def test_pitch_mixture_one_angle_per_frame_key():
    cfg = EasyDict({'PITCH_DEG': [1.97, 1.03], 'PITCH_PROB': [0.558, 0.442], 'PIVOT': PIVOT})
    pt = np.array([[20.0, 0.0, 1.88, 0]], np.float32)
    bx = np.array([[20.0, 0.0, 1.88, 4.0, 1.8, 1.5, 0.0]], np.float32)
    angles = []
    for seq in range(40):
        for fr in range(10):
            key = '%03d/%02d' % (seq, fr)
            zp = wfs.apply_points(pt, cfg, key)[0, 2]
            zb = wfs.apply_boxes(bx, np.array(['Car']), cfg, key)[0, 2]
            assert zp == pytest.approx(zb, abs=1e-6) and wfs.apply_points(pt, cfg, key)[0, 2] == zp
            angles.append(np.degrees(np.arcsin((zp - 1.88) / 20.0)))
    angles = np.round(angles, 2)
    assert set(np.unique(angles)) == {1.97, 1.03}
    assert abs(np.mean(angles == 1.97) - 0.558) < 0.08


@pytest.mark.parametrize('loader,m2k', [('pandaset', P2K), ('lyft', L2K)])
def test_dim_scale_uses_model_names_keeps_the_bottom_and_the_oracle_count(loader, m2k):
    d = _input(loader)
    before = lfs.oracle_counts(d['points'], d['gt_boxes'])
    out = lfs.apply(copy.deepcopy(d), EasyDict({'GT_DIM_SCALE': {'Car': 0.5, 'Pedestrian': 1.2}}), NAMES, m2k, 'k',
                    loader)
    f = np.array([1.0, 0.5, 0.5, 1.0, 1.2, 1.0], np.float32)              # Truck, Car, Car, Misc, Ped, Cyclist
    assert np.allclose(out['gt_boxes'][:, 3:6], d['gt_boxes'][:, 3:6] * f[:, None])
    assert np.allclose(out['gt_boxes'][:, 2] - out['gt_boxes'][:, 5] / 2, d['gt_boxes'][:, 2] - d['gt_boxes'][:, 5] / 2,
                       atol=1e-5)
    assert np.array_equal(out['gt_boxes'][:, [0, 1, 6]], d['gt_boxes'][:, [0, 1, 6]])
    assert np.array_equal(out['points'], d['points'])
    assert out['num_points_in_gt'].tolist() == before.tolist()
    raw = wfs.apply_boxes(d['gt_boxes'], d['gt_names'], EasyDict({'GT_DIM_SCALE': {'Car': 0.5}}), 'k')
    if loader == 'lyft':                                                      # raw 'car' would scale nothing
        assert np.array_equal(raw[1:3], d['gt_boxes'][1:3])


IGN_CFG = EasyDict({'GT_IGNORE_BY_RANGE': {'CLASS': 'Car', 'EDGES': [0, 30, 1000], 'KEEP': [1.0, 0.0]}})
KEEP1 = EasyDict({'GT_IGNORE_BY_RANGE': {'CLASS': 'Car', 'EDGES': [0, 1000], 'KEEP': [1.0]}})


@pytest.mark.parametrize('loader,m2k', [('pandaset', P2K), ('lyft', L2K)])
def test_ignore_labels_from_model_names(loader, m2k):
    d = _input(loader)
    out = lfs.apply(copy.deepcopy(d), IGN_CFG, NAMES, m2k, 'k', loader)
    # Truck -> -99 placeholder (removed by the class filter), far Car -> -99, near Car 1, Misc placeholder, Ped 2, Cyc 3
    assert out['gt_classes'].tolist() == [IGN, IGN, 1, IGN, 2, 3]
    assert out['gt_scores'].tolist() == [1.0] * 6
    assert np.array_equal(out['points'], d['points']) and np.array_equal(out['gt_boxes'], d['gt_boxes'])
    out1 = lfs.apply(copy.deepcopy(d), KEEP1, NAMES, m2k, 'k', loader)
    assert out1['gt_classes'].tolist() == [IGN, 1, 1, IGN, 2, 3]


@pytest.mark.parametrize('loader,m2k', [('pandaset', P2K), ('lyft', L2K)])
def test_region_thin_background_and_its_identity(loader, m2k):
    d = _input(loader)
    cfg = EasyDict({'REGION_THIN': {'SIDE': 'background', 'KEEP': 0.5}, 'BOX_MARGIN': 0.25})
    out = lfs.apply(copy.deepcopy(d), cfg, NAMES, m2k, '014/05', loader)
    inside = in_any_box(d['points'], enlarge_boxes(d['gt_boxes'], 0.25))
    inside_out = in_any_box(out['points'], enlarge_boxes(d['gt_boxes'], 0.25))
    assert np.array_equal(out['points'][inside_out], d['points'][inside])      # in-box points: same, same order
    assert (~inside_out).sum() == int(round((~inside).sum() * 0.5))
    assert lfs.oracle_counts(out['points'], d['gt_boxes']).tolist() == lfs.oracle_counts(d['points'],
                                                                                         d['gt_boxes']).tolist()
    assert 'num_points_in_gt' not in out and np.array_equal(out['gt_boxes'], d['gt_boxes'])
    one = lfs.apply(copy.deepcopy(d), EasyDict({'REGION_THIN': {'SIDE': 'background', 'KEEP': 1.0},
                                                'BOX_MARGIN': 0.25}), NAMES, m2k, '014/05', loader)
    assert np.array_equal(one['points'], d['points'])                         # KEEP 1: the identity
    # seeded by the frame KEY: PandaSet frame 05 of two sequences draws differently, the same key identically
    other = lfs.apply(copy.deepcopy(d), cfg, NAMES, m2k, '050/05', loader)
    again = lfs.apply(copy.deepcopy(d), cfg, NAMES, m2k, '014/05', loader)
    assert np.array_equal(again['points'], out['points']) and not np.array_equal(other['points'], out['points'])


# ---------------------------------------------------------------- through the REAL prepare_data, augmentor and head
def _template(loader, processor=(), aug=()):
    cfg = EasyDict({
        'POINT_CLOUD_RANGE': [-75.2, -75.2, -2, 75.2, 75.2, 4],
        'POINT_FEATURE_ENCODING': {'encoding_type': 'absolute_coordinates_encoding', 'used_feature_list': ['x', 'y', 'z'],
                                   'src_feature_list': ['x', 'y', 'z', 'intensity']},
        'ONTOLOGY': loader, 'DATA_PROCESSOR': list(processor), 'MIN_POINTS_OF_GT': 1,
        'DATA_AUGMENTOR': {'DISABLE_AUG_LIST': [], 'AUG_CONFIG_LIST': list(aug)},
    })
    return DatasetTemplate(dataset_cfg=cfg, class_names=list(NAMES), training=True, root_path=Path('/tmp'),
                           logger=mock.MagicMock(), model_ontology='kitti')


SCALING = [EasyDict({'NAME': 'random_object_scaling', 'SCALE_UNIFORM_NOISE': {'Car': [0.85, 1.2],
                                                                                'Pedestrian': [0.8, 1.25]}})]


def _prepared(ds, loader, swap, key='014/05'):
    d = _input(loader)
    if swap is not None:
        d = lfs.apply(d, swap, NAMES, ds.map_ontology_dataset_to_model, key, loader)
    np.random.seed(0)
    return ds.prepare_data(d)


@pytest.mark.parametrize('loader', ['pandaset', 'lyft'])
@pytest.mark.parametrize('aug', [(), SCALING])
def test_ignore_route_stays_aligned_through_prepare_data(loader, aug):
    """Bug-prevention rules 1 / 8 against the real consumer: KEEP 1 reproduces the plain route's boxes AND labels
    exactly with non-model boxes first (through the augmentor's class mask); KEEP 0 beyond 30 m turns only the far Car
    into -99."""
    ds = _template(loader, aug=aug)
    base = _prepared(ds, loader, None)['gt_boxes']
    assert base[:, 7].tolist() == [1, 1, 2, 3]
    same = _prepared(ds, loader, KEEP1)['gt_boxes']
    assert np.array_equal(same, base)
    far = _prepared(ds, loader, IGN_CFG)['gt_boxes']
    assert far[:, 7].tolist() == [IGN, 1, 2, 3] and np.array_equal(far[:, :7], base[:, :7])


@pytest.mark.parametrize('loader', ['pandaset', 'lyft'])
def test_identity_settings_reproduce_the_oracle_through_prepare_data(loader):
    ds = _template(loader, aug=SCALING)
    base = _prepared(ds, loader, None)
    idents = [EasyDict({'GT_DIM_SCALE': {'Car': 1.0, 'Pedestrian': 1.0}}), KEEP1,
              EasyDict({'REGION_THIN': {'SIDE': 'background', 'KEEP': 1.0}, 'BOX_MARGIN': 0.25})]
    if loader == 'pandaset':
        idents.append(EasyDict({'PITCH_DEG': 0.0, 'PIVOT': PIVOT}))
    for swap in idents:
        out = _prepared(ds, loader, swap)
        assert np.array_equal(out['points'], base['points']), swap
        assert np.array_equal(out['gt_boxes'], base['gt_boxes']), swap
        assert out['gt_names'].tolist() == base['gt_names'].tolist(), swap


def test_dim_scale_trains_on_the_oracle_box_set_even_when_a_box_is_emptied():
    ds = _template('lyft')
    base = _prepared(ds, 'lyft', None)['gt_boxes']
    out = _prepared(ds, 'lyft', EasyDict({'GT_DIM_SCALE': {'Car': 0.05}}))['gt_boxes']   # shrinks the cars to ~nothing
    assert len(out) == len(base) and out[:, 7].tolist() == base[:, 7].tolist()


def test_ignore_boxes_survive_the_processor_recount_and_never_become_positives():
    """Lyft-40's processor recount (drop_empty_gt_boxes) keeps a -99 box whatever it holds; CenterHead gives it no
    peak and masks its footprint (the consumer of the sentinel)."""
    proc = [EasyDict({'NAME': 'drop_empty_gt_boxes', 'MIN_POINTS': 1})]
    ds = _template('lyft', processor=proc)
    d = _input('lyft')
    d['points'] = d['points'][20:]                                            # the far (50 m) car now holds no point
    d = lfs.apply(d, IGN_CFG, NAMES, L2K, 'k', 'lyft')                         # the arm's gt_classes / gt_scores
    assert d['gt_classes'].tolist() == [IGN, IGN, 1, IGN, 2, 3]
    d['num_points_in_gt'] = np.array([20, 20, 20, 20, 20, 20])               # zero-point filter keeps all
    out = ds.prepare_data(d)
    assert out['gt_boxes'][:, 7].tolist() == [IGN, 1, 2, 3]                  # -99 kept though empty
    from test_center_head_ignore_regions import _head
    head = _head()
    head.point_cloud_range = np.array([-75.2, -75.2, -2, 75.2, 75.2, 4], np.float32)
    gt = torch.tensor(out['gt_boxes'][None], dtype=torch.float32)
    t = head.assign_targets(gt, feature_map_size=[376, 376])
    hm, mask = t['heatmaps'][0][0], t['heatmap_masks'][0][0]
    assert int(hm.eq(1).sum()) == 3                                           # Car, Ped, Cyc positives only
    b = out['gt_boxes'][0]
    r, c = int((b[1] + 75.2) / 0.4), int((b[0] + 75.2) / 0.4)
    assert hm[:, r, c].max() < 1 and mask[r, c] == 0


# ---------------------------------------------------------------- the hooks in the real __getitem__
def _pandaset(dataset_cfg, training=True):
    from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset
    names, boxes, points = _frame('pandaset')
    ds = object.__new__(PandasetDataset)
    ds.dataset_cfg = EasyDict(dataset_cfg)
    ds.training = training
    ds.class_names = list(NAMES)
    ds.map_ontology_dataset_to_model = P2K
    ds.pandaset_infos = [{'sequence': '014', 'frame_idx': 5}]
    pose = {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'heading': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}}
    ds._get_pose = lambda info: pose
    ds._get_points_with_sweeps = lambda info, pose_: points.copy()
    ds._get_annotations = lambda info, pose_: (boxes.copy(), names.copy(), 0.0)
    seen = {}

    def prepare_data(data_dict):
        seen.update(copy.deepcopy(data_dict))
        return data_dict
    ds.prepare_data = prepare_data
    return ds, seen


def _lyft(dataset_cfg, training=True):
    from pcdet.datasets.lyft.lyft_dataset import LyftDataset
    names, boxes, points = _frame('lyft')
    ds = object.__new__(LyftDataset)
    ds.dataset_cfg = EasyDict(dataset_cfg)
    ds.training = training
    ds.class_names = list(NAMES)
    ds.map_ontology_dataset_to_model = L2K
    ds._merge_all_iters_to_one_epoch = False
    ds.ring_pattern_cfg = None
    ds.infos = [{'lidar_path': 'lidar/host-a004_lidar1_1234.bin', 'token': 't',
                 'gt_boxes': boxes.astype(np.float64), 'gt_names': names}]
    ds.get_lidar_with_sweeps = lambda index, max_sweeps=1: points.copy()
    ds.level_rotation_for = lambda stem: None
    seen = {}

    def prepare_data(data_dict):
        seen.update(copy.deepcopy(data_dict))
        return data_dict
    ds.prepare_data = prepare_data
    return ds, seen


BUILD = {'pandaset': _pandaset, 'lyft': _lyft}
KEYS = {'pandaset': '014/05', 'lyft': 'host-a004_lidar1_1234'}


def _dcfg(loader, swap=None):
    d = {'SHIFT_COOR': [0.0, 0.0, SHIFT[loader]], 'MAX_SWEEPS': 1}
    if swap is not None:
        d['TRAIN_FACTOR_SWAP'] = swap
    return d


@pytest.mark.parametrize('loader', ['pandaset', 'lyft'])
def test_hook_absent_key_and_evaluation_are_inert(loader):
    ds, base = BUILD[loader](_dcfg(loader))
    ds[0]
    swaps = [{'GT_DIM_SCALE': {'Car': 1.2}}, dict(IGN_CFG), {'REGION_THIN': {'SIDE': 'background', 'KEEP': 0.5},
                                                              'BOX_MARGIN': 0.25}]
    if loader == 'pandaset':
        swaps.append({'PITCH_DEG': 2.0, 'PIVOT': PIVOT})
    for swap in swaps:
        ds2, seen = BUILD[loader](_dcfg(loader, swap), training=False)
        ds2[0]
        assert set(seen) == set(base)
        for k in base:
            assert np.array_equal(np.asarray(seen[k]), np.asarray(base[k])), k


@pytest.mark.parametrize('loader', ['pandaset', 'lyft'])
def test_hook_applies_after_shift_with_the_frame_key_and_pops_the_count(loader):
    m2k = P2K if loader == 'pandaset' else L2K
    ds, base = BUILD[loader](_dcfg(loader))
    ds[0]
    swaps = [{'GT_DIM_SCALE': {'Car': 1.2}}, dict(IGN_CFG), {'REGION_THIN': {'SIDE': 'background', 'KEEP': 0.5},
                                                              'BOX_MARGIN': 0.25}]
    if loader == 'pandaset':
        swaps.append({'PITCH_DEG': [2.0, 1.0], 'PITCH_PROB': [0.5, 0.5], 'PIVOT': PIVOT})
    for swap in swaps:
        ds2, seen = BUILD[loader](_dcfg(loader, swap))
        out = ds2[0]
        expect = lfs.apply({k: copy.deepcopy(v) for k, v in base.items()}, EasyDict(swap), NAMES, m2k, KEYS[loader],
                           loader)
        assert set(seen) == set(expect)
        for k in expect:
            assert np.array_equal(np.asarray(seen[k]), np.asarray(expect[k])), (swap, k)
        assert 'num_points_in_gt' not in out                                 # popped after prepare_data


def test_hook_refuses_accumulation():
    ds, _ = _pandaset({**_dcfg('pandaset', {'GT_DIM_SCALE': {'Car': 1.2}}), 'MAX_SWEEPS': 3})
    with pytest.raises(AssertionError):
        ds[0]


def test_hook_pitches_lyft_points_and_boxes_together():
    """Item 37d: the Lyft hook applies PITCH_DEG (item 34 refused it): points and box centres rotate by the same angle
    about PIVOT, box sizes and the box set unchanged."""
    ds0, base = _lyft(_dcfg('lyft'))
    ds0[0]
    ds, seen = _lyft(_dcfg('lyft', {'PITCH_DEG': -3.0, 'PIVOT': [0, 0, SHIFT['lyft']]}))
    ds[0]
    assert len(seen['gt_boxes']) == len(base['gt_boxes'])
    assert np.allclose(seen['gt_boxes'][:, 3:], base['gt_boxes'][:, 3:])
    assert not np.allclose(seen['points'][:, :3], base['points'][:, :3])
    pv = np.array([0.0, 0.0, SHIFT['lyft']])
    ang = lambda p: np.degrees(np.arctan2(p[:, 2] - pv[2], p[:, 0] - pv[0]))   # noqa: E731
    far = np.hypot(base['points'][:, 0], base['points'][:, 2] - pv[2]) > 2.0
    d = (ang(seen['points'][far, :3].astype(np.float64)) - ang(base['points'][far, :3].astype(np.float64)) + 180) % 360 - 180
    assert np.allclose(d, -3.0, atol=1e-3)


# ---------------------------------------------------------------- the row configs (item 34)
A34 = 'cfgs/da-ieee-access-analysis/'
ORACLES = {'pandaset': ('cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml', ['DATA_CONFIG']),
           'lyft64': (A34 + 'factorswap-lyft64/centerpoint-factorswap-lyft64-oracle.yaml', ['DATA_CONFIG']),
           'lyft40': (A34 + 'factorswap-lyft40/centerpoint-factorswap-lyft40-oracle.yaml', ['DATA_CONFIGS', 'LYFT_40BEAM'])}
ROWS34 = [('pandaset', 'r5-pitch', 'PITCH_DEG'), ('pandaset', 'r6-labelsize', 'GT_DIM_SCALE'),
          ('pandaset', 'r7-labelrange', 'GT_IGNORE_BY_RANGE'), ('pandaset', 'r9-bg050', 'REGION_THIN'),
          ('lyft64', 'r6-labelsize', 'GT_DIM_SCALE'), ('lyft64', 'r7-labelrange', 'GT_IGNORE_BY_RANGE'),
          ('lyft64', 'r9-bg050', 'REGION_THIN'), ('lyft40', 'r6-labelsize', 'GT_DIM_SCALE'),
          ('lyft40', 'r7-labelrange', 'GT_IGNORE_BY_RANGE'), ('lyft40', 'r9-bg050', 'REGION_THIN')]


def _load(path, monkeypatch):
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(Path(__file__).resolve().parent.parent / 'tools')
    return cfg_from_yaml_file(path, EasyDict())


def _block(cfg, path):
    for k in path:
        cfg = cfg[k]
    return cfg


@pytest.mark.parametrize('target,row,arm', ROWS34)
def test_row_config_differs_from_its_oracle_in_its_one_key(target, row, arm, monkeypatch):
    base_path, blk = ORACLES[target]
    base = _load(base_path, monkeypatch)
    cfg = _load(A34 + 'factorswap-%s/centerpoint-factorswap-%s-%s.yaml' % (target, target, row), monkeypatch)
    b, c = _block(base, blk), _block(cfg, blk)
    for key in ('POINT_CLOUD_RANGE', 'DATA_PROCESSOR', 'DATA_AUGMENTOR', 'POINT_FEATURE_ENCODING', 'DATA_SPLIT'):
        assert c.get(key, None) is not None, key                         # resolved to a complete block
    differ = {k for k in set(c) | set(b) if k != '_BASE_CONFIG_' and c.get(k, None) != b.get(k, None)}
    assert differ == {'TRAIN_FACTOR_SWAP'}
    assert lfs.arm_of(c.TRAIN_FACTOR_SWAP, 'pandaset' if target == 'pandaset' else 'lyft') == arm
    lfs.check_dataset_cfg(c, cfg.CLASS_NAMES, 'pandaset' if target == 'pandaset' else 'lyft')
    assert cfg.DATA_CONFIG_TAR == base.DATA_CONFIG_TAR                    # evaluation untouched
    assert cfg.MODEL == base.MODEL and cfg.OPTIMIZATION == base.OPTIMIZATION and cfg.CLASS_NAMES == base.CLASS_NAMES
    swap = c.TRAIN_FACTOR_SWAP
    if arm == 'GT_IGNORE_BY_RANGE':
        keep = np.asarray(swap.GT_IGNORE_BY_RANGE.KEEP)
        assert swap.GT_IGNORE_BY_RANGE.CLASS == 'Car' and len(swap.GT_IGNORE_BY_RANGE.EDGES) == len(keep) + 1
        assert keep.max() == 1.0 and keep.min() > 0.0
    if arm == 'PITCH_DEG':
        assert list(swap.PIVOT) == [0.0, 0.0, 1.88] and abs(sum(swap.PITCH_PROB) - 1.0) < 1e-6


def test_lyft64_density_row_is_the_c1_analog_s_operators(monkeypatch):
    base = _load(ORACLES['lyft64'][0], monkeypatch)
    cfg = _load(A34 + 'factorswap-lyft64/centerpoint-factorswap-lyft64-d-density.yaml', monkeypatch)
    c1 = _load('cfgs/da-ieee-access/centerpoint-global-recount-lyft40-lyft2nuscenes.yaml', monkeypatch)
    differ = {k for k in set(cfg.DATA_CONFIG) | set(base.DATA_CONFIG)
              if k != '_BASE_CONFIG_' and cfg.DATA_CONFIG.get(k, None) != base.DATA_CONFIG.get(k, None)}
    assert differ == {'DATA_PROCESSOR', 'POINT_CALIB_TAR', 'HIST_DIST_LYFT_PATH', 'HIST_DIST_NUSCENES_PATH'}
    assert list(cfg.DATA_CONFIG.DATA_PROCESSOR) == list(c1.DATA_CONFIGS.LYFT_40BEAM.DATA_PROCESSOR)
    assert not cfg.DATA_CONFIG.get('HIST_DIST_ON_THE_FLY', False) and cfg.DATA_CONFIG.POINT_CALIB_TAR == 'nuscenes'
    assert cfg.DATA_CONFIG_TAR == base.DATA_CONFIG_TAR and cfg.MODEL == base.MODEL
