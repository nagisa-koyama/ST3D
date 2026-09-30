"""The two KITTI-oracle variants differ from centerpoint-sourceonly-kitti2kitti in the object-scaling augmentation only."""
import os
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcdet.config import cfg_from_yaml_file  # noqa: E402

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


def _aug(cfg, name):
    return next(a for a in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST if a['NAME'] == name)


def test_noros_disables_scaling_and_changes_nothing_else(in_tools_dir):
    base = cfg_from_yaml_file(FAMILY + 'centerpoint-sourceonly-kitti2kitti.yaml', EasyDict())
    v = cfg_from_yaml_file(FAMILY + 'centerpoint-sourceonly-kitti2kitti-noros.yaml', EasyDict())
    a, b = _flat(base), _flat(v)
    diffs = sorted(k for k in set(a) | set(b) if not k.endswith('_BASE_CONFIG_') and str(a.get(k)) != str(b.get(k)))
    assert diffs == ['DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST']
    assert 'random_object_scaling' in v.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST
    assert v.OPTIMIZATION.NUM_EPOCHS == 152 and v.DATA_CONFIG_TAR.TEST.BOX_FILTER.FOV_FILTER is True


def test_rosshrink_changes_the_car_interval_only(in_tools_dir):
    base = cfg_from_yaml_file(FAMILY + 'centerpoint-sourceonly-kitti2kitti.yaml', EasyDict())
    v = cfg_from_yaml_file(FAMILY + 'centerpoint-sourceonly-kitti2kitti-rosshrink.yaml', EasyDict())
    assert _aug(base, 'random_object_scaling')['SCALE_UNIFORM_NOISE']['Car'] == [0.85, 1.20]
    assert _aug(v, 'random_object_scaling')['SCALE_UNIFORM_NOISE']['Car'] == [0.75, 1.00]
    assert _aug(v, 'random_object_scaling')['SCALE_UNIFORM_NOISE']['Pedestrian'] == [0.80, 1.25]
    assert [x['NAME'] for x in v.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST] == \
        [x['NAME'] for x in base.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST]
    a, b = _flat(base), _flat(v)
    diffs = sorted(k for k in set(a) | set(b) if not k.endswith('_BASE_CONFIG_') and not k.startswith('DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST') and str(a.get(k)) != str(b.get(k)))
    assert not diffs, diffs
