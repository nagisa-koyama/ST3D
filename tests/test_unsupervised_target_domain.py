"""A DA target marked UNSUPERVISED must be loaded raw: no augmentor, no GT requirement.

UADA3D-native reads `UNSUPERVISED` (its `pcdet/datasets/dataset.py:36`) and uses it in three
places: it builds no `DataAugmentor`, it skips the whole GT-requiring branch of `prepare_data`,
and it skips the empty-GT resample. ST3D's port never implemented the key. It had NINE configs
setting it and ZERO Python references, so it read as a dead key while `adaptive_train.py` built
the target loader with `training=True` - and the target therefore got an augmentor and ran the
full branch, with `random_object_scaling` resizing the TARGET's objects using the TARGET's GT
boxes. That is a use of target labels, so the migrated `*-rospm-C` rows were not UDA-legal and
did not match the method they were meant to baseline.

All augmentation is dropped for such a domain, not only the object-level ones, because
UADA3D-native constructs no augmentor at all - matching the paper rather than keeping the
label-free world-level augmentations.
"""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402


def _cfg(unsupervised):
    cfg = EasyDict({
        'POINT_CLOUD_RANGE': [-75.2, -75.2, -2, 75.2, 75.2, 4],
        'POINT_FEATURE_ENCODING': {
            'encoding_type': 'absolute_coordinates_encoding',
            'used_feature_list': ['x', 'y', 'z'],
            'src_feature_list': ['x', 'y', 'z', 'intensity'],
        },
        'ONTOLOGY': 'kitti',
        'DATA_PROCESSOR': [],
        'DATA_AUGMENTOR': {
            'DISABLE_AUG_LIST': [],
            'AUG_CONFIG_LIST': [
                {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']},
            ],
        },
    })
    if unsupervised:
        cfg['UNSUPERVISED'] = True
    return cfg


def _build(unsupervised):
    return DatasetTemplate(
        dataset_cfg=_cfg(unsupervised), class_names=['Car'],
        training=True, root_path=Path("/tmp"), logger=mock.MagicMock(),
    )


def test_unsupervised_domain_builds_no_augmentor():
    assert _build(True).data_augmentor is None, (
        'an unsupervised target must get no augmentor - UADA3D-native builds none'
    )


def test_supervised_domain_still_builds_one():
    """The guard must be opt-in: every existing config leaves UNSUPERVISED unset."""
    assert _build(False).data_augmentor is not None


def test_flag_defaults_to_false():
    assert _build(False).unsupervised is False
    assert _build(True).unsupervised is True


def test_prepare_data_does_not_augment_an_unsupervised_domain():
    """The GT-requiring branch - filter, assert, class mask, augmentor - must be skipped."""
    ds = _build(True)
    ds.data_augmentor = mock.MagicMock()   # would raise if the branch were entered
    data_dict = {
        'points': np.random.rand(64, 3).astype(np.float32),
        'gt_boxes': np.zeros((0, 7), dtype=np.float32),
        'gt_names': np.array([], dtype='<U10'),
    }
    ds.prepare_data(data_dict)
    ds.data_augmentor.forward.assert_not_called()


def test_prepare_data_still_augments_a_supervised_domain():
    ds = _build(False)
    ds.data_augmentor = mock.MagicMock()
    ds.data_augmentor.forward.side_effect = lambda data_dict: data_dict
    data_dict = {
        'points': np.random.rand(64, 3).astype(np.float32),
        'gt_boxes': np.zeros((0, 7), dtype=np.float32),
        'gt_names': np.array([], dtype='<U10'),
        'num_points_in_gt': np.array([], dtype=np.int32),
    }
    ds.prepare_data(data_dict)
    ds.data_augmentor.forward.assert_called_once()
