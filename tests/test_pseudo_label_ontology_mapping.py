"""prepare_data must not map pseudo-labelled names a second time, and a bad map must fail early.

Job 25940 ran 20 minutes - through the whole pseudo-label generation pass - and died inside the
foreground calibration:

    point_calibration.py:143  compute_foreground_histograms -> dataset[idx]
    dataset.py:376            map_ontology_dataset_to_model[...]
    KeyError: 'Car'

`fill_pseudo_labels` builds gt_names from `self.class_names` - the MODEL vocabulary - while
`map_ontology_dataset_to_model` is keyed by the DATASET vocabulary ('car', 'bicycle', ...). The
foreground calibration is the first thing in this repo that ever reads a pseudo-labelled target
back through prepare_data, so nothing had exercised the combination before.

These tests drive prepare_data over a pseudo-labelled target directly, which is what should have
been written before the fifth GPU launch of that row.
"""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402

MODEL_NAMES = ['Car', 'Pedestrian', 'Cyclist']


def _build(training=True, use_pseudo_label=False):
    cfg = EasyDict({
        'ONTOLOGY': 'nuscenes',
        'POINT_CLOUD_RANGE': [-75.2, -75.2, -2, 75.2, 75.2, 4],
        'POINT_FEATURE_ENCODING': {
            'encoding_type': 'absolute_coordinates_encoding',
            'used_feature_list': ['x', 'y', 'z'],
            'src_feature_list': ['x', 'y', 'z', 'intensity'],
        },
        'DATA_PROCESSOR': [],
        'DATA_AUGMENTOR': {'DISABLE_AUG_LIST': [], 'AUG_CONFIG_LIST': []},
    })
    if use_pseudo_label:
        cfg['USE_PSEUDO_LABEL'] = True
    return DatasetTemplate(dataset_cfg=cfg, class_names=MODEL_NAMES, training=training,
                           root_path=Path('/tmp'), logger=mock.MagicMock(), model_ontology='kitti')


def _dict(names):
    return {'points': np.random.rand(32, 3).astype(np.float32),
            'gt_boxes': np.zeros((len(names), 7), dtype=np.float32),
            'gt_names': np.array(names),
            'num_points_in_gt': np.ones((len(names),), dtype=np.int32)}


def test_pseudo_labelled_names_are_not_mapped_again():
    """The exact shape of job 25940: model-vocabulary names on a training pseudo-label target."""
    ds = _build(training=True, use_pseudo_label=True)
    out = ds.prepare_data(_dict(['Car', 'Pedestrian']))
    assert out is not None


def test_dataset_vocabulary_is_still_mapped_when_not_using_pseudo_labels():
    """The normal path must be untouched: 'car' -> 'Car'."""
    ds = _build(training=True, use_pseudo_label=False)
    ds.prepare_data(_dict(['car', 'pedestrian']))


def test_eval_pass_over_a_pseudo_label_dataset_still_maps():
    """USE_PSEUDO_LABEL is set but training is False, so fill_pseudo_labels did NOT run and the
    names are real dataset vocabulary. The guard must key on both, not on the flag alone."""
    ds = _build(training=False, use_pseudo_label=True)
    ds.prepare_data(_dict(['car']))


def test_an_unmapped_name_says_which_vocabulary_it_expected():
    """A bare KeyError names only the missing key, which sent 25940's diagnosis to the wrong layer."""
    ds = _build(training=True, use_pseudo_label=False)
    with pytest.raises(KeyError) as e:
        ds.prepare_data(_dict(['Car']))
    msg = str(e.value)
    assert 'already been mapped' in msg, msg
    assert 'ontology map' in msg, msg


def test_construction_refuses_a_map_that_does_not_cover_the_class_names():
    """Early assertion: a mapping problem must surface in seconds on CPU, not minutes into a GPU run."""
    with mock.patch('pcdet.datasets.dataset.get_ontology_mapping',
                    side_effect=[{'car': 'Car'}, {'Car': 'car', 'Pedestrian': 'pedestrian',
                                                  'Cyclist': 'bicycle'}]):
        with pytest.raises(AssertionError, match='does not cover'):
            _build()


def test_construction_refuses_a_map_that_is_not_round_trip_consistent():
    """A class trained under one label and scored under another would otherwise pass silently."""
    with mock.patch('pcdet.datasets.dataset.get_ontology_mapping',
                    side_effect=[{'car': 'Pedestrian', 'pedestrian': 'Pedestrian',
                                  'bicycle': 'Cyclist'},
                                 {'Car': 'car', 'Pedestrian': 'pedestrian', 'Cyclist': 'bicycle'}]):
        with pytest.raises(AssertionError, match='round-trip'):
            _build()


# --- the other mapping regimes -----------------------------------------------------------------
#
# This repo has THREE, and only one of them was broken. Pinning all three so the next change to
# prepare_data cannot fix one and break another - which is the failure mode this file exists for.
#
#   A  head_per_dataset / multi-head : CLASS_NAMES carry 'dataset:label' prefixes.
#                                      map_ontology_dataset_to_model is deliberately None (the
#                                      reverse maps are known-broken), and prepare_data's
#                                      multi-head block prefixes plain names instead.
#   B  single-ontology cross-mapping : plain names, model_ontology != dataset ontology. The
#                                      regime job 25940 died in.
#   C  same ontology                 : no mapping at all.

def _build_regime(model_ontology, dataset_ontology, class_names, training=True,
                  use_pseudo_label=False):
    cfg = EasyDict({
        'ONTOLOGY': dataset_ontology,
        'POINT_CLOUD_RANGE': [-75.2, -75.2, -2, 75.2, 75.2, 4],
        'POINT_FEATURE_ENCODING': {
            'encoding_type': 'absolute_coordinates_encoding',
            'used_feature_list': ['x', 'y', 'z'],
            'src_feature_list': ['x', 'y', 'z', 'intensity'],
        },
        'DATA_PROCESSOR': [],
        'DATA_AUGMENTOR': {'DISABLE_AUG_LIST': [], 'AUG_CONFIG_LIST': []},
    })
    if use_pseudo_label:
        cfg['USE_PSEUDO_LABEL'] = True
    return DatasetTemplate(dataset_cfg=cfg, class_names=class_names, training=training,
                           root_path=Path('/tmp'), logger=mock.MagicMock(),
                           model_ontology=model_ontology)


def test_regime_A_head_per_dataset_leaves_the_dataset_to_model_map_unset():
    """The multi-head branch sets only model->dataset; the reverse maps are known-broken."""
    ds = _build_regime('head_per_dataset', 'nuscenes',
                       ['nuscenes:car', 'nuscenes:pedestrian'])
    assert ds.map_ontology_dataset_to_model is None, (
        'the multi-head branch must leave this None - prepare_data prefixes names instead'
    )


def test_regime_A_early_assertion_does_not_misfire():
    """The construction-time round-trip check is guarded on the map existing, so a regime that
    legitimately has no dataset->model map must still build."""
    _build_regime('head_per_dataset', 'lyft', ['lyft:car', 'lyft:pedestrian'])


def test_regime_A_prefixes_plain_gt_names():
    ds = _build_regime('head_per_dataset', 'kitti', ['kitti:Car', 'kitti:Pedestrian'])
    out = ds.prepare_data(_dict(['Car']))
    assert out is not None


def test_regime_A_with_pseudo_labels_keeps_already_prefixed_names():
    """fill_pseudo_labels builds names from class_names, which are ALREADY prefixed here, and the
    multi-head block keeps any name containing ':'. So this regime was never exposed to the double
    mapping - pin that it stays that way."""
    ds = _build_regime('head_per_dataset', 'nuscenes',
                       ['nuscenes:car', 'nuscenes:pedestrian'], use_pseudo_label=True)
    ds.prepare_data(_dict(['nuscenes:car']))


def test_regime_C_same_ontology_needs_no_mapping():
    ds = _build_regime('kitti', 'kitti', MODEL_NAMES)
    assert ds.map_ontology_dataset_to_model is None
    ds.prepare_data(_dict(['Car', 'Pedestrian']))


def test_regime_C_with_pseudo_labels_passes_names_through():
    ds = _build_regime('kitti', 'kitti', MODEL_NAMES, use_pseudo_label=True)
    ds.prepare_data(_dict(['Car']))


@pytest.mark.parametrize('use_pseudo_label', [False, True])
def test_regime_B_multi_source_each_source_maps_from_its_own_ontology(use_pseudo_label):
    """A multi-source run builds one dataset per source, each with its OWN dataset ontology against
    the shared model ontology. The two must not interfere - and the pseudo-label flag belongs to
    the target, so a source carrying it must still behave."""
    for dataset_ontology, native in [('nuscenes', 'car'), ('lyft', 'car'), ('waymo', 'Vehicle')]:
        ds = _build_regime('kitti', dataset_ontology, MODEL_NAMES,
                           use_pseudo_label=use_pseudo_label)
        names = MODEL_NAMES[:1] if use_pseudo_label else [native]
        ds.prepare_data(_dict(names))


def test_regime_A_construction_names_what_the_map_is_missing():
    """A bare KeyError at construction says only the key; say the direction and the options too.

    This must hit the CROSS-MAP sub-path: when some CLASS_NAMES already carry this dataset's own
    prefix, dataset_class_names is filtered directly and no map is consulted. The map is only used
    when NO class matches - i.e. a lyft loader under a nuScenes-prefixed multi-head model.
    """
    with mock.patch('pcdet.datasets.dataset.get_ontology_mapping',
                    return_value={'nuscenes:car': 'car'}):
        with pytest.raises(AssertionError, match='does not cover'):
            _build_regime('head_per_dataset', 'lyft',
                          ['nuscenes:car', 'nuscenes:pedestrian'])
