"""Item 51's configs resolve to their reference rows plus exactly the stated changes (experiments_md 20261003_05 §5 item 51).

D0 = 25743's centerpoint-sourceonly-nuscenes.yaml + TRAIN_BEAM_UPSAMPLE; S (ST3D++, nuScenes -> KITTI) = 28650's
centerpoint-st3d-genuine-legal-nuscenes2kitti.yaml + SASD / DSNorm / SEP_LOSS_WEIGHTS / ST3D++ target augmentation /
CDA off / VOTE_ON_EMPTY off, without the MODEL_TEACHER block (unused under TEACHER_IS_STUDENT). CPU only.
"""
import os
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS.parent))


def resolve(rel):
    from pcdet.config import cfg_from_yaml_file
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        return cfg_from_yaml_file(rel, EasyDict())
    finally:
        os.chdir(cwd)


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        if k == '_BASE_CONFIG_':
            continue  # where a block came from, not a setting
        key = prefix + str(k)
        if isinstance(v, dict):
            out.update(flat(v, key + '.'))
        else:
            out[key] = v
    return out


def diff(a, b):
    fa, fb = flat(a), flat(b)
    return {k: (fa.get(k), fb.get(k)) for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}


def test_d0_is_25743_plus_the_up_sampling_only():
    d = diff(resolve('cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml'),
             resolve('cfgs/da-ieee-access/item51/centerpoint-dts-d0-sourceonly-nuscenes.yaml'))
    assert d == {'DATA_CONFIG.TRAIN_BEAM_UPSAMPLE.BEAM_PROB': (None, 0.5),
                 'DATA_CONFIG.TRAIN_BEAM_UPSAMPLE.PHI_THRESHOLD': (None, 0.03)}


def test_st3dpp_is_28650_plus_the_stated_changes():
    base = resolve('cfgs/da-ieee-access/centerpoint-st3d-genuine-legal-nuscenes2kitti.yaml')
    s = resolve('cfgs/da-ieee-access/item51/centerpoint-st3dpp-nuscenes2kitti.yaml')
    d = {k: v for k, v in diff(base, s).items() if not k.startswith('SELF_TRAIN.MODEL_TEACHER.')}
    aug = 'DATA_CONFIG_TAR.DATA_AUGMENTOR.AUG_CONFIG_LIST'
    assert d.pop(aug)[1] == [
        {'NAME': 'random_object_scaling', 'SCALE_UNIFORM_NOISE': [0.9, 1.1]},
        {'NAME': 'random_object_rotation', 'ROT_PROB': 1.0, 'ROT_UNIFORM_NOISE': [-0.78539816, 0.78539816]},
        {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x', 'y']},
        {'NAME': 'random_world_rotation', 'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]},
        {'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [0.95, 1.05]}]
    assert d == {
        'SELF_TRAIN.DSNORM': (None, True),
        'SELF_TRAIN.SRC.USE_DATA': (False, True),
        'SELF_TRAIN.SRC.USE_GRAD': (False, True),
        'SELF_TRAIN.SRC.SEP_LOSS_WEIGHTS.HM_CLASS_WEIGHTS': (None, [0.0, 1.0, 1.0]),
        'SELF_TRAIN.SRC.SEP_LOSS_WEIGHTS.LOC_WEIGHT': (None, 1.0),
        'SELF_TRAIN.PROG_AUG.ENABLED': (True, False),
        'SELF_TRAIN.MEMORY_ENSEMBLE.MEMORY_VOTING.VOTE_ON_EMPTY': (True, False),
    }
    assert 'MODEL_TEACHER' not in s.SELF_TRAIN
    assert s.CLASS_NAMES == ['Car', 'Pedestrian', 'Cyclist'] and s.DATA_CONFIG.MAX_SWEEPS == 1


@pytest.mark.parametrize('rel', ['cfgs/da-ieee-access/item51/centerpoint-st3dpp-nuscenes2kitti.yaml'])
def test_sep_weights_are_in_centerhead_format_and_cover_the_head(rel):
    s = resolve(rel)
    sep = s.SELF_TRAIN.SRC.SEP_LOSS_WEIGHTS
    assert isinstance(sep, dict) and len(sep.HM_CLASS_WEIGHTS) == sum(len(c) for c in s.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD)
    assert 'SEP_LOSS_WEIGHTS' not in s.SELF_TRAIN.TAR
