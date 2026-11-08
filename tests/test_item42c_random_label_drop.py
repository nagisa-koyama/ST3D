"""TRAIN_FACTOR_SWAP.GT_IGNORE_RANDOM, the random label-drop twin R7r of the label-range swap R7 (experiments_md
20261003_05 §5 item 42c; report 20261108_07). The arm sets each box of CLASS to IGNORE_CLASS_LABEL (-99) with
probability 1 - KEEP whatever its range, by R7's route (gt_classes / gt_scores from MODEL names, in waymo_factor_swap.finish
and loader_factor_swap.apply), with one uniform per box from the frame key and the arm. bug_prevention_checklist rules
1 / 4 / 8 / 9: the -99 sentinel through the real prepare_data (augmentor class mask, non-model boxes first) and into
CenterHead, where it must be neither a positive nor a negative; KEEP 1 is the identity; the draw ignores the global numpy
state (both training seeds see one IGNORE set) and the range. Every test here fails on the code before the arm. CPU only.
"""
import copy
import json
import os
import sys
import zlib
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pcdet.datasets import loader_factor_swap as lfs  # noqa: E402
from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.datasets.kitti import kitti_factor_swap as kfs  # noqa: E402
from pcdet.datasets.waymo import waymo_factor_swap as wfs  # noqa: E402
from pcdet.utils.common_utils import IGNORE_CLASS_LABEL as IGN  # noqa: E402
from pcdet.utils.ontology_mapping import get_ontology_mapping  # noqa: E402

import test_loader_factor_swap as TL  # noqa: E402  (helpers only: _frame, _input, _template, _pandaset, _lyft)
import test_waymo_factor_swap as TW  # noqa: E402  (helpers only: _frame, _hooked_dataset)

NAMES = ['Car', 'Pedestrian', 'Cyclist']
W2K = get_ontology_mapping('waymo', 'kitti')
M2K = {'pandaset': TL.P2K, 'lyft': TL.L2K}


def rnd(keep, cls='Car'):
    return EasyDict({'GT_IGNORE_RANDOM': {'CLASS': cls, 'KEEP': keep}})


# ---------------------------------------------------------------- the arm is registered and validated
def test_arm_is_known_on_waymo_pandaset_and_lyft():
    assert wfs.arm_of(rnd(0.5)) == 'GT_IGNORE_RANDOM'
    assert lfs.arm_of(rnd(0.5), 'pandaset') == 'GT_IGNORE_RANDOM'
    assert lfs.arm_of(rnd(0.5), 'lyft') == 'GT_IGNORE_RANDOM'


@pytest.mark.parametrize('bad', [
    {'GT_IGNORE_RANDOM': {'CLASS': 'Car'}},                                    # no KEEP
    {'GT_IGNORE_RANDOM': {'KEEP': 0.5}},                                       # no CLASS
    {'GT_IGNORE_RANDOM': {'CLASS': 'Car', 'KEEP': 0.5, 'EDGES': [0, 1000]}},   # R7's keys do not belong here
    {'GT_IGNORE_RANDOM': {'CLASS': 'Car', 'KEEP': [0.5]}},                     # one probability, not a list
    {'GT_IGNORE_RANDOM': {'CLASS': 'Car', 'KEEP': 1.5}},
    {'GT_IGNORE_RANDOM': {'CLASS': 'Car', 'KEEP': -0.1}},
    {'GT_IGNORE_RANDOM': {'CLASS': ['Car'], 'KEEP': 0.5}},
    {'GT_IGNORE_RANDOM': {'CLASS': 'Car', 'KEEP': 0.5},                        # two arms
     'GT_IGNORE_BY_RANGE': {'CLASS': 'Car', 'EDGES': [0, 1000], 'KEEP': [0.5]}},
])
def test_malformed_configs_are_refused(bad):
    with pytest.raises(AssertionError):
        wfs.arm_of(EasyDict(bad))
    for loader in ('pandaset', 'lyft'):
        with pytest.raises(AssertionError):
            lfs.arm_of(EasyDict(bad), loader)


def test_raw_dataset_names_are_refused_and_model_names_pass():
    wfs.check_dataset_cfg(EasyDict({'TRAIN_FACTOR_SWAP': rnd(0.5), 'MAX_SWEEPS': 1}), NAMES)
    with pytest.raises(AssertionError):
        wfs.check_dataset_cfg(EasyDict({'TRAIN_FACTOR_SWAP': rnd(0.5, 'Vehicle'), 'MAX_SWEEPS': 1}), NAMES)
    for loader, raw in (('pandaset', 'Pickup Truck'), ('lyft', 'car')):
        lfs.check_dataset_cfg(EasyDict({'TRAIN_FACTOR_SWAP': rnd(0.5), 'MAX_SWEEPS': 1}), NAMES, loader)
        with pytest.raises(AssertionError):
            lfs.check_dataset_cfg(EasyDict({'TRAIN_FACTOR_SWAP': rnd(0.5, raw), 'MAX_SWEEPS': 1}), NAMES, loader)
        with pytest.raises(AssertionError):                                   # pseudo-label route never mixed in
            lfs.check_dataset_cfg(EasyDict({'TRAIN_FACTOR_SWAP': rnd(0.5), 'USE_PSEUDO_LABEL': True}), NAMES, loader)


def test_kitti_does_not_take_the_arm():
    """Item 42c twins R7 on Waymo, Pandar64 and Lyft-64 only; KITTI's arm list is unchanged and refuses the key alone."""
    with pytest.raises(AssertionError):
        kfs.arm_of(rnd(0.5))


# ---------------------------------------------------------------- labels: only CLASS, -99, seeded by frame + box + arm
def test_waymo_labels_touch_only_the_class_and_keep_1_is_the_identity():
    names, boxes, points = TW._frame()            # Sign, Vehicle (50 m), Vehicle (10 m), Sign, Pedestrian, Cyclist
    plain = [IGN, 1, 1, IGN, 2, 3]                 # R7 at KEEP 1 and the plain route give this (test_waymo_factor_swap)
    d = wfs.finish({'points': points, 'gt_boxes': boxes, 'gt_names': names}, rnd(1.0), NAMES, W2K, 'f3')
    assert d['gt_classes'].tolist() == plain and d['gt_scores'].tolist() == [1.0] * 6
    assert np.array_equal(d['points'], points) and np.array_equal(d['gt_boxes'], boxes)
    d0 = wfs.finish({'points': points, 'gt_boxes': boxes, 'gt_names': names}, rnd(0.0), NAMES, W2K, 'f3')
    assert d0['gt_classes'].tolist() == [IGN, IGN, IGN, IGN, 2, 3]
    dp = wfs.finish({'points': points, 'gt_boxes': boxes, 'gt_names': names}, rnd(0.0, 'Pedestrian'), NAMES, W2K, 'f3')
    assert dp['gt_classes'].tolist() == [IGN, 1, 1, IGN, IGN, 3]


@pytest.mark.parametrize('loader', ['pandaset', 'lyft'])
def test_loader_labels_touch_only_the_class_and_keep_1_is_the_identity(loader):
    d = TL._input(loader)                          # Truck, Car (50 m), Car (10 m), Misc, Pedestrian, Cyclist
    out1 = lfs.apply(copy.deepcopy(d), rnd(1.0), NAMES, M2K[loader], 'k', loader)
    assert out1['gt_classes'].tolist() == [IGN, 1, 1, IGN, 2, 3]
    out0 = lfs.apply(copy.deepcopy(d), rnd(0.0), NAMES, M2K[loader], 'k', loader)
    assert out0['gt_classes'].tolist() == [IGN, IGN, IGN, IGN, 2, 3]
    assert np.array_equal(out0['points'], d['points']) and np.array_equal(out0['gt_boxes'], d['gt_boxes'])
    assert 'num_points_in_gt' not in out0                                     # prepare_data counts, as for R7


def test_draw_is_one_uniform_per_box_from_the_frame_key_and_the_arm():
    n, keep = 400, 0.6
    names = np.array(['Car'] * n)
    got, _ = wfs.random_ignore_labels(names, NAMES, rnd(keep), 'seg_17_004')
    u = np.random.default_rng(zlib.crc32(b'factorswap/seg_17_004/GT_IGNORE_RANDOM')).random(n)
    assert got.tolist() == np.where(u >= keep, IGN, 1).tolist()
    # box i's draw does not depend on the boxes after it (one stream, in box order)
    head, _ = wfs.random_ignore_labels(names[:150], NAMES, rnd(keep), 'seg_17_004')
    assert head.tolist() == got[:150].tolist()
    # independent of the global numpy state: both training seeds see one IGNORE set
    np.random.seed(666)
    a, _ = wfs.random_ignore_labels(names, NAMES, rnd(keep), 'seg_17_004')
    np.random.seed(667)
    b, _ = wfs.random_ignore_labels(names, NAMES, rnd(keep), 'seg_17_004')
    assert np.array_equal(a, b) and np.array_equal(a, got)
    # another frame draws another set; R7's stream (another arm name) is not this one
    other, _ = wfs.random_ignore_labels(names, NAMES, rnd(keep), 'seg_17_005')
    assert not np.array_equal(other, got)
    boxes = np.tile(np.array([[20.0, 0, 0, 4, 2, 1.5, 0]], np.float32), (n, 1))
    r7, _ = kfs.ignore_labels(boxes, names, NAMES, EasyDict({'GT_IGNORE_BY_RANGE': {
        'CLASS': 'Car', 'EDGES': [0, 1000], 'KEEP': [keep]}}), 'seg_17_004')
    assert not np.array_equal(r7, got)


def test_kept_share_is_keep_and_flat_over_range_while_r7_tilts():
    """The twin's whole point: the same expected count as R7, none of its range selectivity. 300 frames of 100 Car boxes
    spread uniformly over 0-100 m; R7 here is Pandar64's R7 KEEP per 10 m bin."""
    edges = [0, 10, 20, 30, 40, 50, 60, 70, 1000]
    r7_keep = [0.9922, 1.0, 0.7975, 0.5583, 0.3433, 0.2053, 0.117, 0.0633]
    r7cfg = EasyDict({'GT_IGNORE_BY_RANGE': {'CLASS': 'Car', 'EDGES': edges, 'KEEP': r7_keep}})
    rng = np.random.default_rng(0)
    k = 0.4484
    kept_r, kept_7, rad = [], [], []
    for f in range(300):
        r = rng.uniform(0, 100, 100)
        th = rng.uniform(-np.pi, np.pi, 100)
        boxes = np.c_[r * np.cos(th), r * np.sin(th), np.zeros((100, 1)), np.tile([4, 2, 1.5, 0], (100, 1))]
        names = np.array(['Car'] * 100)
        kept_r.append(wfs.random_ignore_labels(names, NAMES, rnd(k), 'f%03d' % f)[0] == 1)
        kept_7.append(kfs.ignore_labels(boxes, names, NAMES, r7cfg, 'f%03d' % f)[0] == 1)
        rad.append(r)
    kept_r, kept_7, rad = np.concatenate(kept_r), np.concatenate(kept_7), np.concatenate(rad)
    assert abs(kept_r.mean() - k) < 0.01
    b = np.clip(np.searchsorted(edges, rad, side='right') - 1, 0, len(edges) - 2)
    per_r = np.array([kept_r[b == i].mean() for i in range(len(edges) - 1)])
    per_7 = np.array([kept_7[b == i].mean() for i in range(len(edges) - 1)])
    se = np.sqrt(k * (1 - k) / np.bincount(b))
    assert np.all(np.abs(per_r - k) < 4 * se), per_r                        # flat
    assert per_7.max() - per_7.min() > 0.8 and per_r.max() - per_r.min() < 0.1


# ---------------------------------------------------------------- through the REAL prepare_data and augmentor (rule 8)
SCALING = TL.SCALING


def _waymo_template(aug=()):
    cfg = EasyDict({
        'POINT_CLOUD_RANGE': [-75.2, -75.2, -2, 75.2, 75.2, 4],
        'POINT_FEATURE_ENCODING': {'encoding_type': 'absolute_coordinates_encoding', 'used_feature_list': ['x', 'y', 'z'],
                                   'src_feature_list': ['x', 'y', 'z', 'intensity', 'elongation']},
        'ONTOLOGY': 'waymo', 'DATA_PROCESSOR': [], 'MIN_POINTS_OF_GT': 1,
        'DATA_AUGMENTOR': {'DISABLE_AUG_LIST': [], 'AUG_CONFIG_LIST': list(aug)},
    })
    return DatasetTemplate(dataset_cfg=cfg, class_names=list(NAMES), training=True, root_path=Path('/tmp'),
                           logger=mock.MagicMock(), model_ontology='kitti')


def _waymo_prepared(ds, swap, frame_id='seg_000'):
    names, boxes, points = TW._frame()
    d = {'points': points, 'gt_boxes': boxes.copy(), 'gt_names': names.copy(), 'frame_id': frame_id,
         'num_points_in_gt': np.full(len(boxes), 20, np.int32)}
    if swap is not None:
        d = wfs.finish(d, swap, NAMES, ds.map_ontology_dataset_to_model, frame_id)
    np.random.seed(0)
    return ds.prepare_data(d)


def _loader_prepared(ds, loader, swap, key='014/05'):
    d = TL._input(loader)
    if swap is not None:
        d = lfs.apply(d, swap, NAMES, ds.map_ontology_dataset_to_model, key, loader)
    np.random.seed(0)
    return ds.prepare_data(d)


def _expected_after_filter(key, keep):
    """The prepared labels predicted from the draw: after the class filter the boxes are Car (50 m), Car (10 m), Ped,
    Cyc, i.e. raw indices 1, 2, 4, 5 of the six-box frame."""
    u = np.random.default_rng(zlib.crc32(('factorswap/%s/GT_IGNORE_RANDOM' % key).encode())).random(6)
    return [IGN if u[1] >= keep else 1, IGN if u[2] >= keep else 1, 2, 3]


@pytest.mark.parametrize('route', ['waymo', 'pandaset', 'lyft'])
@pytest.mark.parametrize('aug', [(), SCALING])
def test_identity_and_real_keep_through_prepare_data(route, aug):
    """KEEP 1 reproduces the plain route's points, boxes AND labels exactly, with non-model boxes BEFORE class boxes
    (the order that misaligned gt_classes before ST3D 12b5147) and the augmentor's class mask; at a real KEEP the points
    and box geometry are the plain route's and only Car labels turn into -99, exactly the draws that say so."""
    if route == 'waymo':
        ds = _waymo_template(aug)
        prep = lambda swap: _waymo_prepared(ds, swap)                         # noqa: E731
        key = 'seg_000'
    else:
        ds = TL._template(route, aug=aug)
        prep = lambda swap: _loader_prepared(ds, route, swap)                 # noqa: E731
        key = '014/05'
    base = prep(None)
    assert base['gt_boxes'][:, 7].tolist() == [1, 1, 2, 3]
    same = prep(rnd(1.0))
    assert np.array_equal(same['gt_boxes'], base['gt_boxes']) and np.array_equal(same['points'], base['points'])
    assert same['gt_names'].tolist() == base['gt_names'].tolist()
    for keep in (0.0, 0.3, 0.7):
        out = prep(rnd(keep))
        assert np.array_equal(out['points'], base['points'])
        assert np.array_equal(out['gt_boxes'][:, :7], base['gt_boxes'][:, :7])
        assert out['gt_boxes'][:, 7].tolist() == _expected_after_filter(key, keep), keep
    assert prep(rnd(0.0))['gt_boxes'][:, 7].tolist() == [IGN, IGN, 2, 3]


# ---------------------------------------------------------------- the consumer: CenterHead (rules 1 / 4)
def test_ignored_cars_are_neither_positives_nor_negatives_in_centerhead():
    """KEEP 0: both Cars arrive as -99. CenterHead gives them no peak and no regression target, masks their footprint in
    every class channel, and the heatmap loss does not change whatever the network predicts there (not a negative)."""
    from test_center_head_ignore_regions import _head
    ds = TL._template('lyft', aug=SCALING)
    out = _loader_prepared(ds, 'lyft', rnd(0.0))
    gt_boxes = out['gt_boxes']
    assert gt_boxes[:, 7].tolist() == [IGN, IGN, 2, 3]
    head = _head()
    head.point_cloud_range = np.array([-75.2, -75.2, -2, 75.2, 75.2, 4], np.float32)
    gt = torch.tensor(gt_boxes[None], dtype=torch.float32)
    t = head.assign_targets(gt, feature_map_size=[376, 376])
    hm, mask = t['heatmaps'][0], t['heatmap_masks'][0]
    assert int(hm[0, 0].eq(1).sum()) == 0                                    # no Car positive
    assert int(hm[0, 1].eq(1).sum()) == 1 and int(hm[0, 2].eq(1).sum()) == 1  # Ped, Cyc kept
    assert int(t['masks'][0].sum()) == 2                                     # regression targets: Ped and Cyc only
    assert mask is not None
    for b in gt_boxes[:2]:
        r, c = int((b[1] + 75.2) / 0.4), int((b[0] + 75.2) / 0.4)
        assert hm[0, :, r, c].max() < 1 and mask[0, r, c] == 0
    ignored = mask[0] == 0
    assert int(ignored.sum()) > 0
    pred = torch.full_like(hm, 0.1)
    loss = head.hm_loss_func(pred, hm, mask=mask)
    pred2 = pred.clone()
    pred2[0, 0][ignored] = 0.95                                              # a confident Car on the ignored footprint
    assert torch.equal(head.hm_loss_func(pred2, hm, mask=mask), loss)
    assert not torch.equal(head.hm_loss_func(pred2, hm, mask=None), head.hm_loss_func(pred, hm, mask=None))


# ---------------------------------------------------------------- the hooks in the real __getitem__
def test_waymo_hook_applies_the_arm_in_training_and_is_inert_at_evaluation():
    base_cfg = {'SHIFT_COOR': [0.0, 0.0, 0.0], 'MAX_SWEEPS': 1}
    ds, base, (names, boxes, points) = TW._hooked_dataset(base_cfg)
    ds[0]
    ds, seen, _ = TW._hooked_dataset({**base_cfg, 'TRAIN_FACTOR_SWAP': rnd(0.5)})
    ds[0]
    want, _ = wfs.random_ignore_labels(wfs.model_names(names, W2K), NAMES, rnd(0.5), 'seg_004')
    assert seen['gt_classes'].tolist() == want.tolist()
    assert np.array_equal(seen['points'], base['points']) and np.array_equal(seen['gt_boxes'], base['gt_boxes'])
    ds, seen, _ = TW._hooked_dataset({**base_cfg, 'TRAIN_FACTOR_SWAP': rnd(0.5)}, training=False)
    ds[0]
    assert set(seen) == set(base)
    for k in base:
        assert np.array_equal(np.asarray(seen[k]), np.asarray(base[k])), k


@pytest.mark.parametrize('loader', ['pandaset', 'lyft'])
def test_loader_hooks_apply_the_arm_with_the_frame_key_and_are_inert_at_evaluation(loader):
    ds, base = TL.BUILD[loader](TL._dcfg(loader))
    ds[0]
    ds, seen = TL.BUILD[loader](TL._dcfg(loader, dict(rnd(0.5))))
    out = ds[0]
    names = wfs.model_names(base['gt_names'], M2K[loader])
    want, _ = wfs.random_ignore_labels(names, NAMES, rnd(0.5), TL.KEYS[loader])
    assert seen['gt_classes'].tolist() == want.tolist()
    assert np.array_equal(seen['points'], base['points']) and np.array_equal(seen['gt_boxes'], base['gt_boxes'])
    assert 'num_points_in_gt' not in out
    ds, seen = TL.BUILD[loader](TL._dcfg(loader, dict(rnd(0.5))), training=False)
    ds[0]
    assert set(seen) == set(base)
    for k in base:
        assert np.array_equal(np.asarray(seen[k]), np.asarray(base[k])), k


# ---------------------------------------------------------------- the three R7r configs
A = 'cfgs/da-ieee-access-analysis/'
ROWS = {   # target: (R7 config, R7r config, loader, KEEP, stats JSON, its key)
    'waymo': (A + 'factorswap-n2w/centerpoint-factorswap-n2w-r7-labelrange.yaml',
              A + 'factorswap-n2w/centerpoint-factorswap-n2w-r7r-labelrandom.yaml', 'waymo', 0.5647,
              '/home/koyama/data/factorswap_n2w/stats.json', 'kept_share_of_waymo_cars'),
    'pandaset': (A + 'factorswap-pandaset/centerpoint-factorswap-pandaset-r7-labelrange.yaml',
                 A + 'factorswap-pandaset/centerpoint-factorswap-pandaset-r7r-labelrandom.yaml', 'pandaset', 0.4484,
                 '/home/koyama/data/factorswap_pandaset/stats.json', 'kept_share_of_target_cars'),
    'lyft64': (A + 'factorswap-lyft64/centerpoint-factorswap-lyft64-r7-labelrange.yaml',
               A + 'factorswap-lyft64/centerpoint-factorswap-lyft64-r7r-labelrandom.yaml', 'lyft', 0.6400,
               '/home/koyama/data/factorswap_lyft/stats_lyft64.json', 'kept_share_of_target_cars'),
}


def _load(path, monkeypatch):
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(Path(__file__).resolve().parent.parent / 'tools')
    return cfg_from_yaml_file(path, EasyDict())


@pytest.mark.parametrize('target', sorted(ROWS))
def test_r7r_config_is_r7_with_the_arm_swapped_and_keep_is_r7_s_kept_share(target, monkeypatch):
    r7_path, r7r_path, loader, keep, stats, key = ROWS[target]
    r7, r7r = _load(r7_path, monkeypatch), _load(r7r_path, monkeypatch)
    for k in ('POINT_CLOUD_RANGE', 'DATA_PROCESSOR', 'DATA_AUGMENTOR', 'POINT_FEATURE_ENCODING', 'DATA_SPLIT'):
        assert r7r.DATA_CONFIG.get(k, None) is not None, k                  # resolved to a complete block
    differ = {k for k in set(r7r.DATA_CONFIG) | set(r7.DATA_CONFIG)
              if k != '_BASE_CONFIG_' and r7r.DATA_CONFIG.get(k, None) != r7.DATA_CONFIG.get(k, None)}
    assert differ == {'TRAIN_FACTOR_SWAP'}
    assert r7r.DATA_CONFIG.TRAIN_FACTOR_SWAP == {'GT_IGNORE_RANDOM': {'CLASS': 'Car', 'KEEP': keep}}
    assert r7r.DATA_CONFIG_TAR == r7.DATA_CONFIG_TAR                         # evaluation untouched
    assert r7r.MODEL == r7.MODEL and r7r.OPTIMIZATION == r7.OPTIMIZATION and r7r.CLASS_NAMES == r7.CLASS_NAMES
    assert r7r.get('ONTOLOGY', None) == r7.get('ONTOLOGY', None)
    if loader == 'waymo':
        assert wfs.arm_of(r7r.DATA_CONFIG.TRAIN_FACTOR_SWAP) == 'GT_IGNORE_RANDOM'
        wfs.check_dataset_cfg(r7r.DATA_CONFIG, r7r.CLASS_NAMES)
    else:
        assert lfs.arm_of(r7r.DATA_CONFIG.TRAIN_FACTOR_SWAP, loader) == 'GT_IGNORE_RANDOM'
        lfs.check_dataset_cfg(r7r.DATA_CONFIG, r7r.CLASS_NAMES, loader)
    if os.access(stats, os.R_OK):
        assert round(json.load(open(stats))['label_range'][key], 4) == keep
