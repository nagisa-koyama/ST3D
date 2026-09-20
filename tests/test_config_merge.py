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


# ---------------------------------------------------------------------------
# Multi-level `_BASE_CONFIG_` chains.
#
# `de9f9d7` (2026-08-23) fixed child-wins precedence but replaced a recursive merge with a
# non-recursive `_fill_missing_from_base`, so a base file's OWN `_BASE_CONFIG_` was copied
# across as an ordinary key and never opened. Chains then expanded exactly one level and the
# whole second hop vanished. Found 2026-09-20, fixed 2026-09-21 - see
# experiments_md/20260921_01_base_config_recursion_regression_fix.md.
# ---------------------------------------------------------------------------


def test_multi_level_base_chain_is_fully_expanded(tmp_path):
    """A base that itself declares `_BASE_CONFIG_` must be followed, not copied verbatim."""
    grandparent = tmp_path / 'preprocess.yaml'
    parent = tmp_path / 'dataset.yaml'
    child = tmp_path / 'experiment.yaml'

    _write_yaml(grandparent, {
        'POINT_CLOUD_RANGE': [-75.2, -75.2, -2, 75.2, 75.2, 4],
        'DATA_PROCESSOR': [{'NAME': 'sample_points_hist_based'}],
    })
    _write_yaml(parent, {
        '_BASE_CONFIG_': str(grandparent),
        'DATASET': 'LyftDataset',
        'DATA_PATH': '../data/lyft',
    })
    _write_yaml(child, {
        'DATA_CONFIGS': {
            'LYFT_CONFIG': {
                '_BASE_CONFIG_': str(parent),
                'CLASS_NAMES': ['lyft:car'],
            },
        },
    })

    cfg = EasyDict()
    cfg_from_yaml_file(str(child), cfg)
    block = cfg.DATA_CONFIGS.LYFT_CONFIG

    # Hop 1 (the parent) - was never in doubt.
    assert block.DATASET == 'LyftDataset'
    # Hop 2 (the grandparent) - silently lost by the 2026-08-23..2026-09-21 implementation.
    assert block.POINT_CLOUD_RANGE == [-75.2, -75.2, -2, 75.2, 75.2, 4]
    assert block.DATA_PROCESSOR == [{'NAME': 'sample_points_hist_based'}]
    # The child's own contribution survives.
    assert block.CLASS_NAMES == ['lyft:car']


def test_nearest_definition_wins_across_three_levels(tmp_path):
    """Precedence must stay child > parent > grandparent at every hop."""
    grandparent = tmp_path / 'gp.yaml'
    parent = tmp_path / 'p.yaml'
    child = tmp_path / 'c.yaml'

    _write_yaml(grandparent, {'A': 'from_grandparent', 'B': 'from_grandparent',
                              'C': 'from_grandparent'})
    _write_yaml(parent, {'_BASE_CONFIG_': str(grandparent), 'B': 'from_parent',
                         'C': 'from_parent'})
    _write_yaml(child, {'DATA_CONFIG': {'_BASE_CONFIG_': str(parent), 'C': 'from_child'}})

    cfg = EasyDict()
    cfg_from_yaml_file(str(child), cfg)

    assert cfg.DATA_CONFIG.A == 'from_grandparent'   # only the grandparent defines it
    assert cfg.DATA_CONFIG.B == 'from_parent'        # parent overrides grandparent
    assert cfg.DATA_CONFIG.C == 'from_child'         # child overrides both


def test_nested_dicts_merge_across_three_levels(tmp_path):
    """Per-key merging of nested dicts must also survive the second hop."""
    grandparent = tmp_path / 'gp.yaml'
    parent = tmp_path / 'p.yaml'
    child = tmp_path / 'c.yaml'

    _write_yaml(grandparent, {'NESTED': {'A': 1, 'B': 2, 'C': 3}})
    _write_yaml(parent, {'_BASE_CONFIG_': str(grandparent), 'NESTED': {'B': 20}})
    _write_yaml(child, {'DATA_CONFIG': {'_BASE_CONFIG_': str(parent), 'NESTED': {'C': 300}}})

    cfg = EasyDict()
    cfg_from_yaml_file(str(child), cfg)

    assert cfg.DATA_CONFIG.NESTED.A == 1      # grandparent only
    assert cfg.DATA_CONFIG.NESTED.B == 20     # parent overrides grandparent
    assert cfg.DATA_CONFIG.NESTED.C == 300    # child overrides both


def test_resolved_config_records_its_own_base_path(tmp_path):
    """`_BASE_CONFIG_` is bookkeeping: a block keeps the path it actually declared."""
    grandparent = tmp_path / 'gp.yaml'
    parent = tmp_path / 'p.yaml'
    child = tmp_path / 'c.yaml'

    _write_yaml(grandparent, {'A': 1})
    _write_yaml(parent, {'_BASE_CONFIG_': str(grandparent), 'B': 2})
    _write_yaml(child, {'DATA_CONFIG': {'_BASE_CONFIG_': str(parent)}})

    cfg = EasyDict()
    cfg_from_yaml_file(str(child), cfg)

    assert cfg.DATA_CONFIG._BASE_CONFIG_ == str(parent)


def test_cyclic_base_chain_raises(tmp_path):
    """A chain that loops must fail loudly instead of recursing forever."""
    import pytest

    a = tmp_path / 'a.yaml'
    b = tmp_path / 'b.yaml'
    child = tmp_path / 'c.yaml'

    _write_yaml(a, {'_BASE_CONFIG_': str(b), 'X': 1})
    _write_yaml(b, {'_BASE_CONFIG_': str(a), 'Y': 2})
    _write_yaml(child, {'DATA_CONFIG': {'_BASE_CONFIG_': str(a)}})

    cfg = EasyDict()
    with pytest.raises(ValueError, match='Cyclic _BASE_CONFIG_ chain'):
        cfg_from_yaml_file(str(child), cfg)
