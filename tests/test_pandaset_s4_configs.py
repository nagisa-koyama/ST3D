"""The PandaSet S3/S4 completion rows (experiments_md/20261001_03): each differs from its parent only as
its header says, and no target block inherits the flash cone filter by accident.

The configs inherit from the spin -> flash row, whose TARGET block carries EVAL_FOV_FILTER (the flash
cone); nearest-definition merging would leak it into a spin target unless the child states it, so the
resolved values are what is asserted here, never the YAML text.
"""
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcdet.config import cfg_from_yaml_file  # noqa: E402

TOOLS = ROOT / 'tools'
D = 'cfgs/da-ieee-access/'
T1 = 'cfgs/da-ieee-access-tier1/'


@pytest.fixture(autouse=True)
def in_tools_dir(monkeypatch):
    monkeypatch.chdir(TOOLS)


def _load(name):
    cfg = EasyDict()
    cfg_from_yaml_file(name, cfg)
    return cfg


def _cone(block):
    return (block.get('EVAL_FOV_FILTER', False), block.get('EVAL_FOV_DEGREE'), block.get('EVAL_FOV_HEADING'))


def _strip(block, *keys):
    return {k: v for k, v in block.items() if k not in keys}


def test_s4_control_is_the_s3_control_with_the_datasets_swapped():
    s3, s4 = _load(D + 'centerpoint-pandaset-spin2flash.yaml'), _load(D + 'centerpoint-pandaset-flash2spin.yaml')
    assert s4.DATA_CONFIG.LIDAR_DEVICE == 1 and s4.DATA_CONFIG.MAX_SWEEPS == 1
    assert not s4.DATA_CONFIG.get('HIST_DIST_ON_THE_FLY', False)
    assert s4.DATA_CONFIG_TAR.LIDAR_DEVICE == 0
    assert _cone(s4.DATA_CONFIG_TAR)[0] is False, 'the flash cone filter leaked into the spin target'
    assert s4.MODEL == s3.MODEL and s4.OPTIMIZATION == s3.OPTIMIZATION
    assert s4.DATA_CONFIG.DATA_AUGMENTOR == s3.DATA_CONFIG.DATA_AUGMENTOR
    assert s4.DATA_CONFIG.DATA_PROCESSOR == s3.DATA_CONFIG.DATA_PROCESSOR
    # the source cone keys are inert in training, but the source must be the flash sensor
    assert s4.DATA_CONFIG.SHIFT_COOR == s3.DATA_CONFIG.SHIFT_COOR


@pytest.mark.parametrize('name, device, cone', [
    ('centerpoint-pandaset-flash2spin-incone.yaml', 0, (True, 60.0, 0.0)),
    ('centerpoint-pandaset-flash2flash.yaml', 1, (True, 60.0, 0.0)),
    ('centerpoint-pandaset-spin2spin.yaml', 0, (False, None, None)),
    ('centerpoint-pandaset-spin2spin-incone.yaml', 0, (True, 60.0, 0.0)),
    ('centerpoint-accum-global-pandaset-flash2spin-incone.yaml', 0, (True, 60.0, 0.0)),
])
def test_eval_only_targets(name, device, cone):
    cfg = _load(D + name)
    assert cfg.DATA_CONFIG_TAR.LIDAR_DEVICE == device
    got = _cone(cfg.DATA_CONFIG_TAR)
    assert got[0] is cone[0]
    if cone[0]:
        assert got[1:] == cone[1:]


@pytest.mark.parametrize('child, parent', [
    ('centerpoint-pandaset-flash2spin-incone.yaml', 'centerpoint-pandaset-flash2spin.yaml'),
    ('centerpoint-pandaset-flash2flash.yaml', 'centerpoint-pandaset-flash2spin.yaml'),
    ('centerpoint-pandaset-spin2spin.yaml', 'centerpoint-pandaset-spin2flash.yaml'),
    ('centerpoint-accum-global-pandaset-flash2spin-incone.yaml', 'centerpoint-accum-global-pandaset-flash2spin.yaml'),
])
def test_eval_only_configs_keep_the_trained_model_and_source(child, parent):
    c, p = _load(D + child), _load(D + parent)
    assert c.MODEL == p.MODEL and c.CLASS_NAMES == p.CLASS_NAMES
    assert c.DATA_CONFIG == p.DATA_CONFIG


def test_s4_method_adds_only_the_cone_restricted_correction():
    ctrl, corr = _load(D + 'centerpoint-pandaset-flash2spin.yaml'), _load(D + 'centerpoint-conecorr-pandaset-flash2spin.yaml')
    s, c = corr.DATA_CONFIG, ctrl.DATA_CONFIG
    assert s.HIST_DIST_ON_THE_FLY is True and s.HIST_DIST_FOV_DEGREE == 60.0 and s.HIST_DIST_FOV_HEADING == 0.0
    assert s.get('HIST_DIST_TARGET_SPLIT', 'train') == 'train'
    names = [p.NAME for p in s.DATA_PROCESSOR]
    assert names == ['mask_points_and_boxes_outside_range', 'shuffle_points', 'sample_points_hist_based',
                     'transform_points_to_voxels']
    assert [p for p in s.DATA_PROCESSOR if p.NAME != 'sample_points_hist_based'] == list(c.DATA_PROCESSOR)
    hist_keys = [k for k in s if k.startswith('HIST_DIST_')]
    # _BASE_CONFIG_ survives resolution as a bookkeeping key only where a block named one itself
    assert _strip(s, 'DATA_PROCESSOR', '_BASE_CONFIG_', *hist_keys) == _strip(c, 'DATA_PROCESSOR', '_BASE_CONFIG_')
    assert corr.DATA_CONFIG_TAR == ctrl.DATA_CONFIG_TAR
    assert corr.MODEL == ctrl.MODEL and corr.OPTIMIZATION == ctrl.OPTIMIZATION


@pytest.mark.parametrize('name', ['centerpoint-pandaset-flash2spin', 'centerpoint-conecorr-pandaset-flash2spin'])
def test_tier1_proxies_change_only_the_epochs(name):
    full, proxy = _load(D + name + '.yaml'), _load(T1 + name + '-15ep.yaml')
    assert proxy.OPTIMIZATION.NUM_EPOCHS == 15
    assert _strip(proxy.OPTIMIZATION, 'NUM_EPOCHS') == _strip(full.OPTIMIZATION, 'NUM_EPOCHS')
    assert proxy.DATA_CONFIG == full.DATA_CONFIG and proxy.DATA_CONFIG_TAR == full.DATA_CONFIG_TAR
    # the S3 proxies it is read beside run the same 15 epochs
    assert _load(T1 + 'centerpoint-pandaset-spin2flash-15ep.yaml').OPTIMIZATION.NUM_EPOCHS == 15
