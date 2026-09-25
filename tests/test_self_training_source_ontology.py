"""A self-training SOURCE must keep its ontology remap, or it contributes no ground truth at all.

train.py chooses the source dataset's model_ontology. The rule it means to implement, per its own
comment, is "drop the remap for a head_per_dataset teacher, whose class names are already prefixed".
The rule it implemented was "drop it whenever MODEL_TEACHER.ONTOLOGY is set at all".

With a teacher ONTOLOGY of 'kitti' that meant the source was built with model_ontology=None, so
map_ontology_dataset_to_model was None, so Lyft's gt_names stayed lowercase ('car', 'pedestrian') -
and keep_arrays_by_name() in prepare_data dropped every box, because CLASS_NAMES are
['Car', 'Pedestrian', 'Cyclist']. Measured on the real config:

    model_ontology='kitti' -> 51, 6, 28, 10 boxes in the first sampled frames
    model_ontology=None    ->  0, 0,  0,  0

So the source detection loss had no positives for the entire run. The visible symptom was the
foreground calibration reporting an empty source channel (jobs 25941 and 25943: "NO boxes in any
sampled frame", frames_used=1000, boxes_seen=0); the actual damage was the missing supervision.

These tests exercise the selection rule directly - it is a pure function of the config - so they
cost milliseconds and would have caught this before any of the six launches of that row.
"""
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _source_model_ontology(cfg):
    """The rule as train.py implements it. Kept in lockstep with tools/train.py:187-200."""
    source_model_ontology = cfg.get('ONTOLOGY', None)
    if cfg.get('SELF_TRAIN', None):
        teacher_cfg = cfg.SELF_TRAIN.get('MODEL_TEACHER', None)
        if teacher_cfg is not None and teacher_cfg.get('ONTOLOGY', None) == 'head_per_dataset':
            source_model_ontology = None
    return source_model_ontology


def test_a_kitti_teacher_keeps_the_source_remap():
    """The exact shape of the defect: a plain-ontology teacher must NOT blank the source's map."""
    cfg = EasyDict({'ONTOLOGY': 'kitti',
                    'SELF_TRAIN': {'MODEL_TEACHER': {'ONTOLOGY': 'kitti'}}})
    assert _source_model_ontology(cfg) == 'kitti', (
        'blanking this leaves Lyft names lowercase and keep_arrays_by_name drops every box'
    )


@pytest.mark.parametrize('teacher_ontology', ['nuscenes', 'lyft', 'waymo', 'pandaset'])
def test_any_plain_teacher_ontology_keeps_the_source_remap(teacher_ontology):
    cfg = EasyDict({'ONTOLOGY': 'kitti',
                    'SELF_TRAIN': {'MODEL_TEACHER': {'ONTOLOGY': teacher_ontology}}})
    assert _source_model_ontology(cfg) == 'kitti'


def test_a_head_per_dataset_teacher_drops_it():
    """The one case the rule is actually for: prefixed names must not be cross-mapped."""
    cfg = EasyDict({'ONTOLOGY': 'kitti',
                    'SELF_TRAIN': {'MODEL_TEACHER': {'ONTOLOGY': 'head_per_dataset'}}})
    assert _source_model_ontology(cfg) is None


def test_a_teacher_without_an_ontology_keeps_it():
    cfg = EasyDict({'ONTOLOGY': 'kitti', 'SELF_TRAIN': {'MODEL_TEACHER': {'NAME': 'CenterPoint'}}})
    assert _source_model_ontology(cfg) == 'kitti'


def test_self_training_without_a_teacher_keeps_it():
    cfg = EasyDict({'ONTOLOGY': 'kitti', 'SELF_TRAIN': {'TAR': {'LOSS_WEIGHT': 1.0}}})
    assert _source_model_ontology(cfg) == 'kitti'


def test_no_self_training_keeps_it():
    assert _source_model_ontology(EasyDict({'ONTOLOGY': 'kitti'})) == 'kitti'


def test_the_rule_here_matches_train_py():
    """Guard against this copy drifting from the implementation it mirrors."""
    src = (Path(__file__).resolve().parent.parent / 'tools' / 'train.py').read_text()
    assert "teacher_cfg.get('ONTOLOGY', None) == 'head_per_dataset'" in src, (
        'train.py no longer selects the source ontology the way this test asserts'
    )
