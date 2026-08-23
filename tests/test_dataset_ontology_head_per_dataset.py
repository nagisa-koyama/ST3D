"""
Regression test for pcdet/datasets/dataset.py's `DatasetTemplate.__init__()` branch selection
between the "head_per_dataset" multi-head ("dataset:class"-formatted `CLASS_NAMES`) setup and the
single-ontology cross-mapping setup.

Bug found 2026-08-23: the two branches were checked in the wrong order:

    if model_ontology is not None and self.dataset_ontology is not None \\
            and model_ontology != self.dataset_ontology:
        ...  # single-ontology cross-mapping (e.g. waymo -> kitti)
    elif class_names is not None and ":" in class_names[0]:
        ...  # multi-head "dataset:class" handling

Multi-dataset "head_per_dataset" configs pass the literal string `'head_per_dataset'` as
`model_ontology` (see e.g. `tools/cfgs/.../domain_attention_head_per_dataset/*.yaml`'s
`SELF_TRAIN.MODEL_TEACHER.ONTOLOGY` / top-level `ONTOLOGY`, consumed by
`tools/train.py`/`tools/test.py`), while each real per-dataset config's own `ONTOLOGY` field is
the dataset's native ontology (e.g. `'lyft'`, `'kitti'`). Since `'head_per_dataset' != 'lyft'`
(both non-None), the FIRST branch above always won for any "dataset:class"-formatted CLASS_NAMES
list combined with `model_ontology == 'head_per_dataset'` unless a caller special-cased
`model_ontology` to `None` (as `tools/train.py` does, but only for the *source* dataloader, not
target/eval).

That first branch calls `get_ontology_mapping(dataset_ontology, 'head_per_dataset')`, whose
`map_<dataset>_to_head_per_dataset` dictionaries are keyed by *plain* (non-prefixed) class names
and (separately) are miswritten to always emit `'waymo:...'` regardless of the input dataset
ontology (see `pcdet/utils/ontology_mapping.py`, all five `map_*_to_head_per_dataset` dicts,
explicitly marked "Defined for compatibility with the rest of the code but not used"). Using this
as `map_ontology_model_to_dataset` to remap "dataset:class"-formatted `class_names` either
KeyErrors (the dict has no `"lyft:car"`-style keys) or silently renames GT boxes to the wrong
dataset prefix, causing every GT box for that dataset to be filtered out downstream in
`prepare_data()`'s `n in self.class_names` mask -- silent, total training/eval data loss for that
dataset, with no error raised.

Fix: check for "dataset:class"-formatted `class_names` FIRST. That branch already handles
`head_per_dataset` correctly and generically (by filtering to classes whose prefix matches this
dataset's own `ONTOLOGY`), so the single-ontology cross-mapping branch must only run when
`class_names` is NOT in the "dataset:class" format.
"""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.utils.ontology_mapping import get_ontology_mapping  # noqa: E402


class _FakeDataProcessor:
    def __init__(self, *args, **kwargs):
        self.grid_size = np.array([1, 1, 1])
        self.voxel_size = np.array([1.0, 1.0, 1.0])

    def forward(self, data_dict):
        return data_dict


class _FakePointFeatureEncoder:
    def __init__(self, *args, **kwargs):
        self.num_point_features = 4

    def forward(self, data_dict):
        return data_dict


def _make_dataset_cfg(ontology):
    return EasyDict({
        'DATA_PATH': '/tmp',
        'POINT_CLOUD_RANGE': [0, -40, -3, 70.4, 40, 1],
        'POINT_FEATURE_ENCODING': {},
        'DATA_PROCESSOR': [],
        'ONTOLOGY': ontology,
    })


def _build_dataset(class_names, dataset_ontology, model_ontology):
    with mock.patch('pcdet.datasets.dataset.PointFeatureEncoder', _FakePointFeatureEncoder), \
         mock.patch('pcdet.datasets.dataset.DataProcessor', _FakeDataProcessor):
        return DatasetTemplate(
            dataset_cfg=_make_dataset_cfg(dataset_ontology),
            class_names=class_names,
            training=False,
            root_path=Path('/tmp'),
            logger=EasyDict({'info': lambda *a, **k: None}),
            model_ontology=model_ontology,
        )


def test_head_per_dataset_multihead_names_take_precedence_over_ontology_mapping():
    dataset = _build_dataset(
        class_names=['lyft:car', 'lyft:pedestrian', 'kitti:Car', 'kitti:Pedestrian'],
        dataset_ontology='lyft',
        model_ontology='head_per_dataset',
    )
    # The buggy single-ontology cross-mapping branch must NOT be taken.
    assert dataset.map_ontology_dataset_to_model is None
    assert dataset.map_ontology_model_to_dataset is None
    # dataset_class_names must be filtered to this dataset's own prefixed classes, unmodified.
    assert dataset.dataset_class_names == ['lyft:car', 'lyft:pedestrian']


def test_head_per_dataset_multihead_names_for_kitti_dataset():
    dataset = _build_dataset(
        class_names=['lyft:car', 'lyft:pedestrian', 'kitti:Car', 'kitti:Pedestrian'],
        dataset_ontology='kitti',
        model_ontology='head_per_dataset',
    )
    assert dataset.map_ontology_dataset_to_model is None
    assert dataset.dataset_class_names == ['kitti:Car', 'kitti:Pedestrian']


def test_single_ontology_cross_mapping_still_works_for_non_prefixed_class_names():
    # Legitimate case: no "dataset:class" prefixes, real (non-'head_per_dataset') ontologies that
    # differ between model and dataset -- the cross-mapping branch must still run as before.
    dataset = _build_dataset(
        class_names=['Car', 'Pedestrian'],
        dataset_ontology='waymo',
        model_ontology='kitti',
    )
    assert dataset.map_ontology_dataset_to_model == get_ontology_mapping('waymo', 'kitti')
    assert dataset.map_ontology_model_to_dataset == get_ontology_mapping('kitti', 'waymo')
    expected = [get_ontology_mapping('kitti', 'waymo')[label] for label in ['Car', 'Pedestrian']]
    assert dataset.dataset_class_names == expected


def test_mismatched_ontology_raises_clear_error_instead_of_index_error():
    # Misconfiguration guard: if a dataset's own ONTOLOGY doesn't match the prefix of ANY class in
    # a "dataset:class"-formatted CLASS_NAMES list (e.g. a typo, or a dataset missing from
    # CLASS_NAMES entirely), dataset_class_names would end up empty. Previously this crashed with
    # an uninformative `IndexError: list index out of range` from `dataset_class_names[-1]`; it
    # must now raise a clear AssertionError explaining the mismatch instead.
    with pytest.raises(AssertionError, match="No class in .* matches this dataset's ONTOLOGY"):
        _build_dataset(
            class_names=['lyft:car', 'lyft:pedestrian', 'kitti:Car', 'kitti:Pedestrian'],
            dataset_ontology='nuscenes',  # not present in any CLASS_NAMES prefix
            model_ontology='head_per_dataset',
        )


def test_head_per_dataset_sentinel_reaching_cross_mapping_branch_raises():
    # Guard against a regression of the original bug: if class_names does NOT use "dataset:class"
    # formatting but model_ontology or dataset_ontology is literally the 'head_per_dataset'
    # sentinel, the cross-mapping branch must refuse to run rather than silently use the buggy
    # map_*_to_head_per_dataset dictionaries in ontology_mapping.py.
    with pytest.raises(AssertionError, match="Refusing to use single-ontology cross-mapping"):
        _build_dataset(
            class_names=['Car'],  # no colon -> would otherwise reach the cross-mapping branch
            dataset_ontology='waymo',
            model_ontology='head_per_dataset',
        )
