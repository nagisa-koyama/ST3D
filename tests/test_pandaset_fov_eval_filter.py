"""Evaluating PandarGT only where PandarGT can see.

PandaSet labels the full 360 degrees whichever device is loaded, and `cuboids.sensor_id` does not
restrict them. PandarGT is a forward flash lidar covering about +-30 degrees, so most of its GT
boxes contain no points at all - and the zero-point GT filter in dataset.py is guarded by
`if self.training:`, so at evaluation they survive and score as false negatives.

Measured on 25 val frames, Car boxes: 3,126 boxes, 2,507 of them empty (80.2%). A 60 degree cone
keeps 923 (29.5%) of which 64.8% are non-empty - roughly tripling the achievable recall ceiling.

This is the same correction KITTI gets, with the asymmetry reversed: KITTI's labels are narrower
than its sensor so PREDICTIONS get cut; here the labels are wider so the GT does.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcdet.datasets.pandaset.pandaset_dataset import (  # noqa: E402
    filter_annos_to_fov, fov_mask)


def box(x, y, z=0.0):
    return [x, y, z, 4.0, 2.0, 2.0, 0.0]


def test_a_box_straight_ahead_is_kept():
    assert fov_mask(np.array([box(20.0, 0.0)]), 60.0, 0.0).tolist() == [True]


def test_a_box_behind_the_vehicle_is_dropped():
    assert fov_mask(np.array([box(-20.0, 0.0)]), 60.0, 0.0).tolist() == [False]


def test_the_cone_edge_is_the_half_angle():
    """60 degrees is the FULL angle, so the boundary sits at 30, not 60."""
    inside = box(10.0, 10.0 * np.tan(np.radians(29.0)))
    outside = box(10.0, 10.0 * np.tan(np.radians(31.0)))
    assert fov_mask(np.array([inside, outside]), 60.0, 0.0).tolist() == [True, False]


def test_it_is_symmetric_about_the_heading():
    y = 10.0 * np.tan(np.radians(20.0))
    assert fov_mask(np.array([box(10.0, y), box(10.0, -y)]), 60.0, 0.0).tolist() == [True, True]


def test_an_empty_frame_does_not_blow_up():
    """rotate_points_along_z is not safe on a zero-row array, so the wrapper must short-circuit."""
    assert fov_mask(np.zeros((0, 7)), 60.0, 0.0).shape == (0,)


def test_a_wider_cone_keeps_at_least_as_much():
    boxes = np.array([box(10.0, 10.0 * np.tan(np.radians(a))) for a in range(-80, 81, 5)])
    counts = [fov_mask(boxes, d, 0.0).sum() for d in (30.0, 60.0, 90.0, 180.0)]
    assert counts == sorted(counts) and counts[0] < counts[-1]


def test_every_per_box_array_is_filtered_together():
    annos = [{'gt_boxes': np.array([box(20.0, 0.0), box(-20.0, 0.0), box(30.0, 1.0)]),
              'gt_names': np.array(['Car', 'Car', 'Pedestrian'])}]
    kept, total = filter_annos_to_fov(annos, 'gt_boxes', ('gt_boxes', 'gt_names'), 60.0, 0.0)
    assert (kept, total) == (2, 3)
    assert len(annos[0]['gt_boxes']) == 2
    assert annos[0]['gt_names'].tolist() == ['Car', 'Pedestrian']


def test_predictions_are_filtered_on_their_own_key():
    annos = [{'boxes_lidar': np.array([box(20.0, 0.0), box(-20.0, 0.0)]),
              'name': np.array(['Car', 'Car']), 'score': np.array([0.9, 0.8]),
              'pred_labels': np.array([1, 1])}]
    kept, total = filter_annos_to_fov(annos, 'boxes_lidar',
                                      ('boxes_lidar', 'name', 'score', 'pred_labels'), 60.0, 0.0)
    assert (kept, total) == (1, 2)
    assert annos[0]['score'].tolist() == [0.9] and len(annos[0]['pred_labels']) == 1


def test_a_fixed_width_field_is_NOT_filtered_by_length_coincidence():
    """`pose` is 7 numbers and a frame can hold 7 boxes; only listed keys may be touched."""
    annos = [{'gt_boxes': np.array([box(20.0, 0.0)] * 6 + [box(-20.0, 0.0)]),
              'gt_names': np.array(['Car'] * 7),
              'pose': np.arange(7.0)}]
    filter_annos_to_fov(annos, 'gt_boxes', ('gt_boxes', 'gt_names'), 60.0, 0.0)
    assert annos[0]['pose'].tolist() == list(range(7)), 'pose must survive intact'
    assert len(annos[0]['gt_boxes']) == 6


def test_a_missing_box_key_is_skipped_rather_than_raising():
    annos = [{'gt_names': np.array(['Car'])}]
    assert filter_annos_to_fov(annos, 'gt_boxes', ('gt_boxes',), 60.0, 0.0) == (0, 0)


def test_the_flash_config_enables_it_and_the_spin_config_does_not():
    root = Path(__file__).resolve().parent.parent / 'tools/cfgs/da-ieee-access'
    flash = (root / 'da_pandaset_flash_dataset.yaml').read_text(encoding='utf-8')
    spin = (root / 'da_pandaset_spin_dataset.yaml').read_text(encoding='utf-8')
    assert 'EVAL_FOV_FILTER: True' in flash, 'the flash sensor is the one with the coverage gap'
    assert 'EVAL_FOV_DEGREE: 60.0' in flash
    assert 'EVAL_FOV_FILTER' not in spin, 'Pandar64 sees 360 degrees and needs no cone'


def test_the_evaluation_path_actually_consults_the_key():
    src = (Path(__file__).resolve().parent.parent
           / 'pcdet/datasets/pandaset/pandaset_dataset.py').read_text(encoding='utf-8')
    assert "self.dataset_cfg.get('EVAL_FOV_FILTER', False)" in src
    i_filter = src.index("EVAL_FOV_FILTER', False")
    i_eval = src.index('get_official_eval_result')
    assert i_filter < i_eval, 'the filter must run before the AP computation, not after'
