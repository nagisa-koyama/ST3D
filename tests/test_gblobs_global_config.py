"""centerpoint-gblobs-global-lyft2nuscenes must be its parent plus the GBlobs VFE and nothing else.

The row exists to answer one question - are the global density correction and GBlobs additive on
Lyft -> nuScenes? - so its resolved config has to differ from centerpoint-global-lyft2nuscenes in
MODEL.VFE alone, and its VFE block has to be the one the GBlobs source-only row used (or the
comparison against 31.43 measures a second variable). Both parents have drifted from siblings
before (experiments_md/20260922_04 §1.7), which is why this is pinned rather than assumed.
"""
import os
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(ROOT))

from pcdet.config import cfg_from_yaml_file  # noqa: E402

FAMILY = 'cfgs/da-ieee-access/'
CHILD = FAMILY + 'centerpoint-gblobs-global-lyft2nuscenes.yaml'
PARENT = FAMILY + 'centerpoint-global-lyft2nuscenes.yaml'
GBLOBS = FAMILY + 'centerpoint-gblobs-sourceonly-lyft.yaml'


@pytest.fixture
def in_tools_dir():
    prev = os.getcwd()
    os.chdir(TOOLS)
    try:
        yield
    finally:
        os.chdir(prev)


def _flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = prefix + str(k)
        if isinstance(v, dict):
            out.update(_flat(v, key + '.'))
        else:
            out[key] = v
    return out


def _load(path):
    return cfg_from_yaml_file(path, EasyDict())


def test_differs_from_the_global_correction_row_in_the_vfe_only(in_tools_dir):
    child, parent = _flat(_load(CHILD)), _flat(_load(PARENT))
    diffs = sorted(k for k in set(child) | set(parent)
                   if not k.endswith('_BASE_CONFIG_') and not k.startswith('MODEL.VFE.')
                   and str(child.get(k)) != str(parent.get(k)))
    assert not diffs, diffs


def test_vfe_block_is_the_gblobs_rows(in_tools_dir):
    child, gblobs = _load(CHILD), _load(GBLOBS)
    assert dict(child.MODEL.VFE) == dict(gblobs.MODEL.VFE)
    assert child.MODEL.VFE.NAME == 'GBlobsVFE'


def test_correction_and_two_platform_split_are_inherited(in_tools_dir):
    child = _load(CHILD)
    assert set(child.DATA_CONFIGS) == {'LYFT_40BEAM', 'LYFT_64BEAM'}
    for name, blk in child.DATA_CONFIGS.items():
        assert blk.HIST_DIST_ON_THE_FLY is True, name
        assert any(p['NAME'] == 'sample_points_hist_based' for p in blk.DATA_PROCESSOR), name
    assert child.OPTIMIZATION.BATCH_SIZE_PER_GPU == 3     # global 6 on the 2-GPU launcher
    assert child.OPTIMIZATION.NUM_EPOCHS == 30
