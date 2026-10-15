"""LIT_TRANSLATED: the opt-in training hook that serves LiT's translated frames (pcdet/datasets/waymo/waymo_lit.py,
experiments_md 20261015_02). Sentinel values of the pipeline (checklist rule 1) are run through the consumer the hook
feeds, the zero-point GT filter in DatasetTemplate.prepare_data:
- negative class label / IGNORE_CLASS_LABEL (-99): a box kept as an ignore region keeps its label, and is dropped like any
  other box when the translated cloud leaves it empty;
- features=None (x, y, z-only input): the translated (N, 3) cloud is padded to the stored 5 columns, so a source config with
  used_feature_list [x, y, z] still sees a defined array;
- an empty translated cloud and an empty box list do not crash the filter;
- a missing / mislabelled / non-finite frame is an error, never a fallback to the stored cloud.
Runs on CPU.
"""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.datasets.waymo.waymo_lit import load_translated  # noqa: E402
from pcdet.utils.common_utils import IGNORE_CLASS_LABEL  # noqa: E402

NAMES = ['Car', 'Pedestrian', 'Cyclist']


def _write(root, seq, idx, pts, frame_index=None):
    d = Path(root) / seq
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / ('%04d.npz' % idx), frame_index=idx if frame_index is None else frame_index,
                        frame_pose=np.eye(4, dtype=np.float32), local_points=np.asarray(pts, np.float32),
                        local_bboxes=np.zeros((0, 7), np.float32), incident_angles=np.zeros(len(pts), np.float32))


def test_padded_to_the_stored_five_columns(tmp_path):
    _write(tmp_path, 's', 3, [[1, 2, 3], [4, 5, 6]])
    p = load_translated(tmp_path, 's', 3)
    assert p.shape == (2, 5) and p.dtype == np.float32
    assert np.array_equal(p[:, :3], [[1, 2, 3], [4, 5, 6]]) and not p[:, 3:].any()


def test_empty_cloud_is_legal(tmp_path):
    _write(tmp_path, 's', 0, np.zeros((0, 3)))
    assert load_translated(tmp_path, 's', 0).shape == (0, 5)


@pytest.mark.parametrize('case', ['missing', 'index', 'nan', 'shape'])
def test_bad_frames_raise_not_fall_back(tmp_path, case):
    if case == 'index':
        _write(tmp_path, 's', 1, [[0, 0, 0]], frame_index=2)
    elif case == 'nan':
        _write(tmp_path, 's', 1, [[np.nan, 0, 0]])
    elif case == 'shape':
        _write(tmp_path, 's', 1, np.zeros((3, 4)))
    with pytest.raises((FileNotFoundError, ValueError)):
        load_translated(tmp_path, 's', 1)


def _dataset(ignore_other=True):
    cfg = EasyDict({
        'POINT_CLOUD_RANGE': [-75.2, -75.2, -2, 75.2, 75.2, 4],
        'POINT_FEATURE_ENCODING': {'encoding_type': 'absolute_coordinates_encoding',
                                   'used_feature_list': ['x', 'y', 'z'],
                                   'src_feature_list': ['x', 'y', 'z', 'intensity', 'elongation']},
        'ONTOLOGY': 'kitti', 'DATA_PROCESSOR': [],
        'DATA_AUGMENTOR': {'DISABLE_AUG_LIST': [], 'AUG_CONFIG_LIST': []},
    })
    if ignore_other:
        cfg['IGNORE_OTHER_CLASSES'] = True
    return DatasetTemplate(dataset_cfg=cfg, class_names=list(NAMES), training=True, root_path=Path('/tmp'),
                           logger=mock.MagicMock())


def _frame(points, num_points=None):
    names = np.array(['Car', 'Truck', 'Pedestrian', 'Cyclist'])
    boxes = np.array([[5, 0, 0, 4, 2, 1.5, 0], [10, 3, 0, 9, 2.6, 3.2, 0], [20, 5, 0, 0.8, 0.8, 1.7, 0],
                      [30, -5, 0, 1.8, 0.6, 1.7, 0]], dtype=np.float32)
    return {'points': points, 'gt_boxes': boxes, 'gt_names': names, 'num_points_in_gt': num_points, 'frame_id': '0'}


def test_recount_drops_empty_boxes_and_keeps_ignore_labels(tmp_path):
    # points only inside the Car (x ~ 5) and the Truck (x ~ 10); the Pedestrian and Cyclist boxes are left empty
    pts = np.array([[5, 0, 0], [5.5, 0.2, 0.1], [10, 3, 0], [11, 3, 0.5]], np.float32)
    _write(tmp_path, 's', 0, pts)
    out = _dataset().prepare_data(_frame(load_translated(tmp_path, 's', 0), num_points=None))
    assert list(out['gt_names']) == ['Car', 'Truck']
    assert list(out['gt_boxes'][:, 7]) == [1, IGNORE_CLASS_LABEL]       # the Truck stays a class-less ignore region


def test_empty_cloud_and_empty_boxes_do_not_crash(tmp_path):
    _write(tmp_path, 's', 0, np.zeros((0, 3)))
    out = _dataset().prepare_data(_frame(load_translated(tmp_path, 's', 0), num_points=None))
    assert len(out['gt_boxes']) == 0
    fr = _frame(np.array([[5, 0, 0]], np.float32), num_points=None)
    fr.update(gt_boxes=np.zeros((0, 7), np.float32), gt_names=np.array([], dtype='<U3'))
    assert len(_dataset().prepare_data(fr)['gt_boxes']) == 0


def test_stored_count_would_keep_the_empty_boxes():
    """Why the hook sets num_points_in_gt to None: the stored all-lidar count keeps boxes the translation emptied."""
    pts = np.zeros((0, 5), np.float32)
    out = _dataset().prepare_data(_frame(pts, num_points=np.array([30, 40, 12, 9], np.int32)))
    assert len(out['gt_boxes']) == 4


def test_hook_serves_the_translated_cloud_without_the_stored_file(tmp_path):
    from pcdet.datasets.waymo.waymo_dataset import WaymoDataset
    _write(tmp_path, 'seq', 7, [[1, 1, 1]])
    ds = WaymoDataset.__new__(WaymoDataset)
    ds.lit_cfg = EasyDict({'DIR': str(tmp_path)})
    ds.training = True
    ds.dataset_cfg = EasyDict({})
    info = {'point_cloud': {'lidar_sequence': 'seq', 'sample_idx': 7}}
    assert np.array_equal(ds.get_lidar_with_sweeps(info)[:, :3], [[1, 1, 1]])
    ds.training = False                      # evaluation never uses translated frames: it reaches the stored-file path
    with pytest.raises(Exception):
        ds.get_lidar_with_sweeps(info)
