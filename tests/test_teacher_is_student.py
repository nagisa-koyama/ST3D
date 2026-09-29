"""SELF_TRAIN.TEACHER_IS_STUDENT: the student generates its own pseudo-labels."""
import os
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools')); sys.path.insert(0, str(ROOT))
import _init_path  # noqa: F401,E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from train_utils.train_st_utils import student_is_teacher  # noqa: E402

FAMILY = 'cfgs/da-ieee-access/'


@pytest.fixture
def in_tools_dir():
    prev = os.getcwd(); os.chdir(ROOT / 'tools')
    try:
        yield
    finally:
        os.chdir(prev)


def _flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + str(k) + '.'))
        else:
            out[prefix + str(k)] = v
    return out


def test_off_by_default_and_refuses_the_single_pass_flag():
    assert student_is_teacher(EasyDict()) is False
    assert student_is_teacher(EasyDict(TEACHER_IS_STUDENT=True)) is True
    with pytest.raises(AssertionError, match='contradict'):
        student_is_teacher(EasyDict(TEACHER_IS_STUDENT=True, FROZEN_TEACHER_SINGLE_PASS=True))


def test_unfrozen_row_differs_from_the_frozen_one_in_the_teacher_policy_only(in_tools_dir):
    a = _flat(cfg_from_yaml_file(FAMILY + 'centerpoint-foreground-v2-lyft2nuscenes.yaml', EasyDict()))
    b = _flat(cfg_from_yaml_file(FAMILY + 'centerpoint-foreground-v2-unfrozen-lyft2nuscenes.yaml', EasyDict()))
    diffs = sorted(k for k in set(a) | set(b) if not k.endswith('_BASE_CONFIG_') and str(a.get(k)) != str(b.get(k)))
    assert diffs == ['SELF_TRAIN.FROZEN_TEACHER_SINGLE_PASS', 'SELF_TRAIN.TEACHER_IS_STUDENT']
    cfg = cfg_from_yaml_file(FAMILY + 'centerpoint-foreground-v2-unfrozen-lyft2nuscenes.yaml', EasyDict())
    assert student_is_teacher(cfg.SELF_TRAIN)
    assert cfg.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED and cfg.SELF_TRAIN.MEMORY_ENSEMBLE.MEMORY_VOTING.ENABLED
    assert cfg.SELF_TRAIN.UPDATE_PSEUDO_LABEL_INTERVAL == 2


def test_train_py_skips_the_teacher_and_refuses_a_teacher_checkpoint():
    src = (ROOT / 'tools/train.py').read_text(encoding='utf-8')
    i = src.index('student_is_teacher(cfg.SELF_TRAIN)')
    block = src[i:i + 1600]
    assert 'model_teacher = None' in block
    assert 'args.pretrained_model_teacher is None' in block
    assert 'args.pretrained_model is not None' in block
    # the frozen-teacher branch still exists for every other config
    assert "elif cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('MODEL_TEACHER', None)" in src
