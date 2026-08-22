"""
Regression test for pcdet/config.py's `_BASE_CONFIG_` merge behavior.

Covers a real bug found 2026-08-23 (see experiments_md/20260823_01_merge_new_config_base_override_bug.md):
`merge_new_config` used to let a `_BASE_CONFIG_` file's values silently clobber a child yaml's
own overrides for any scalar key both defined, instead of the child taking precedence.
"""
import sys
from pathlib import Path

import yaml
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.config import cfg_from_yaml_file  # noqa: E402


def _write_yaml(path, data):
    with open(path, 'w') as f:
        yaml.safe_dump(data, f)


def test_child_scalar_overrides_base_scalar(tmp_path):
    base_path = tmp_path / 'base.yaml'
    child_path = tmp_path / 'child.yaml'

    _write_yaml(base_path, {
        'DATASET': 'WaymoDataset',
        'DATA_PATH': '../data/waymo',
        'INFO_WITH_FAKELIDAR': True,
        'SOME_BASE_ONLY_KEY': 'base_value',
    })
    _write_yaml(child_path, {
        'CLASS_NAMES': ['Vehicle'],
        'DATA_CONFIG': {
            '_BASE_CONFIG_': str(base_path),
            'DATA_PATH': '../data/waymo_child_override',
            'INFO_WITH_FAKELIDAR': False,
        },
    })

    cfg = EasyDict()
    cfg_from_yaml_file(str(child_path), cfg)

    # Child's own values must win over the base's values for shared scalar keys.
    assert cfg.DATA_CONFIG.INFO_WITH_FAKELIDAR is False
    assert cfg.DATA_CONFIG.DATA_PATH == '../data/waymo_child_override'
    # Keys only defined in the base must still be inherited.
    assert cfg.DATA_CONFIG.SOME_BASE_ONLY_KEY == 'base_value'


def test_child_partial_nested_override_keeps_other_base_subkeys(tmp_path):
    base_path = tmp_path / 'base.yaml'
    child_path = tmp_path / 'child.yaml'

    _write_yaml(base_path, {
        'NESTED': {'A': 1, 'B': 2},
    })
    _write_yaml(child_path, {
        'DATA_CONFIG': {
            '_BASE_CONFIG_': str(base_path),
            'NESTED': {'A': 100},
        },
    })

    cfg = EasyDict()
    cfg_from_yaml_file(str(child_path), cfg)

    # Child overrides the subkey it sets...
    assert cfg.DATA_CONFIG.NESTED.A == 100
    # ...but subkeys only defined in the base are still inherited.
    assert cfg.DATA_CONFIG.NESTED.B == 2
