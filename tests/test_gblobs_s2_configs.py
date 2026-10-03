"""GBlobs on S2 (experiments_md/20260928_05 Phase 3.2): each row is its parent with the voxel encoder
swapped to GBlobsVFE and nothing else, resolved through the repo's own loader."""
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pcdet.config import cfg_from_yaml_file  # noqa: E402

D = 'cfgs/da-ieee-access/'
GBLOBS = {'NAME': 'GBlobsVFE', 'RELATIVE_DISTANCE': True, 'COVARIANCE_ONLY': False, 'COVARIANCE_SCALE': 1.0}


@pytest.fixture(autouse=True)
def in_tools_dir(monkeypatch):
    monkeypatch.chdir(ROOT / 'tools')


def _load(name):
    cfg = EasyDict()
    cfg_from_yaml_file(D + name, cfg)
    return cfg


@pytest.mark.parametrize('child, parent', [
    ('centerpoint-gblobs-sourceonly-nuscenes2kitti.yaml', 'centerpoint-sourceonly-nuscenes2kitti.yaml'),
    ('centerpoint-gblobs-accum-nuscenes2kitti.yaml', 'centerpoint-accum-nuscenes2kitti.yaml'),
])
def test_only_the_voxel_encoder_changes(child, parent):
    c, p = _load(child), _load(parent)
    assert dict(c.MODEL.VFE) == GBLOBS
    assert {k: v for k, v in c.MODEL.items() if k != 'VFE'} == {k: v for k, v in p.MODEL.items() if k != 'VFE'}
    for key in ('DATA_CONFIG', 'DATA_CONFIG_TAR', 'OPTIMIZATION', 'CLASS_NAMES'):
        assert c.get(key) == p.get(key), key
    if 'DATA_CONFIGS' in p:
        assert c.DATA_CONFIGS == p.DATA_CONFIGS


def test_the_vfe_matches_the_s1_gblobs_row():
    s1 = _load('centerpoint-gblobs-sourceonly-lyft.yaml')
    assert dict(s1.MODEL.VFE) == GBLOBS
