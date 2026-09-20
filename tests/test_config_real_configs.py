"""Oracle test: the repo's own multi-level configs must resolve to complete dataset blocks.

The `tmp_path`-based tests in test_config_merge.py pin the merge *semantics*. This file pins
the thing that actually broke in production: a `da-MIRU2025`-family config reaches its dataset
through `DATA_CONFIGS.<NAME>._BASE_CONFIG_` -> a dataset yaml with its own `_BASE_CONFIG_` ->
a preprocessing yaml. Between `de9f9d7` (2026-08-23) and 2026-09-21 that second hop was never
opened, so `DATA_PROCESSOR` and `POINT_CLOUD_RANGE` silently vanished and the density
correction (`sample_points_hist_based`) was not applied - while the config still "loaded fine".

See experiments_md/20260921_01_base_config_recursion_regression_fix.md.
"""
import os
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

TOOLS_DIR = Path(__file__).resolve().parent.parent / 'tools'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.config import cfg_from_yaml_file  # noqa: E402


@pytest.fixture
def in_tools_dir():
    """`_BASE_CONFIG_` paths are relative to tools/, which is where train.py is always run."""
    prev = os.getcwd()
    os.chdir(TOOLS_DIR)
    try:
        yield
    finally:
        os.chdir(prev)


def _dataset_blocks(cfg):
    for name in ('DATA_CONFIG', 'DATA_CONFIG_TAR'):
        if name in cfg:
            yield name, cfg[name]
    for name, blk in cfg.get('DATA_CONFIGS', {}).items():
        yield 'DATA_CONFIGS.%s' % name, blk


# (config, expect_density_correction) - one representative per affected family.
MULTI_LEVEL_CONFIGS = [
    ('cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2nuscenes_car_ped_default.yaml', False),
    ('cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2nuscenes_car_ped_point_calibrated.yaml', True),
    ('cfgs/da-MIRU2025/second_old_anchor_basebev_multi_nuscenes2kitti_car_ped_point_label_calibrated.yaml', True),
    ('cfgs/da-post-MIRU2025/second_old_anchor_st3d_basebev_multi_lyft2nuscenes_dann_source_target_car_ped_point_label_calibrated.yaml', True),
]


@pytest.mark.parametrize('cfg_file,expect_hist', MULTI_LEVEL_CONFIGS)
def test_multi_level_config_resolves_complete_dataset_blocks(cfg_file, expect_hist, in_tools_dir):
    assert Path(cfg_file).exists(), 'config moved or renamed: %s' % cfg_file

    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)

    blocks = list(_dataset_blocks(cfg))
    assert blocks, 'no dataset blocks found in %s' % cfg_file

    hist_seen = False
    for label, blk in blocks:
        # These live in the SECOND hop of the base chain - the one the regression dropped.
        assert 'DATA_PROCESSOR' in blk, '%s.DATA_PROCESSOR missing in %s' % (label, cfg_file)
        assert 'POINT_CLOUD_RANGE' in blk, '%s.POINT_CLOUD_RANGE missing in %s' % (label, cfg_file)
        names = [step.get('NAME') for step in blk['DATA_PROCESSOR']]
        assert 'transform_points_to_voxels' in names, \
            '%s has no voxelization step in %s' % (label, cfg_file)
        if 'sample_points_hist_based' in names:
            hist_seen = True

    assert hist_seen == expect_hist, (
        'density correction presence mismatch for %s: expected %s, got %s'
        % (cfg_file, expect_hist, hist_seen))
