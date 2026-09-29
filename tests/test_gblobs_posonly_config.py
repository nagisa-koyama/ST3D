"""GBlobs position-only: the covariance block is zeroed, nothing else changes, and the VFE honours it."""
import os
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools')); sys.path.insert(0, str(ROOT))
import _init_path  # noqa: F401,E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.models.backbones_3d.vfe.gblobs_vfe import GBlobsVFE  # noqa: E402

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


def test_differs_from_gblobs_row_in_covariance_scale_only(in_tools_dir):
    a = _flat(cfg_from_yaml_file(FAMILY + 'centerpoint-gblobs-sourceonly-lyft.yaml', EasyDict()))
    b = _flat(cfg_from_yaml_file(FAMILY + 'centerpoint-gblobs-posonly-sourceonly-lyft.yaml', EasyDict()))
    diffs = sorted(k for k in set(a) | set(b) if not k.endswith('_BASE_CONFIG_') and str(a.get(k)) != str(b.get(k)))
    assert diffs == ['MODEL.VFE.COVARIANCE_SCALE']
    assert float(b['MODEL.VFE.COVARIANCE_SCALE']) == 0.0


def test_vfe_zeroes_the_covariance_and_keeps_the_relative_position_and_width():
    cfg = EasyDict(RELATIVE_DISTANCE=True, COVARIANCE_ONLY=False, COVARIANCE_SCALE=0.0)
    vfe = GBlobsVFE(cfg, num_point_features=3, voxel_size=[0.1, 0.1, 0.15],
                    point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4])
    assert vfe.get_output_feature_dim() == 12                 # stem stays 12 -> 16
    torch.manual_seed(0)
    voxels = torch.randn(5, 10, 3) * 0.03
    out = vfe({'voxels': voxels, 'voxel_num_points': torch.full((5,), 10, dtype=torch.long),
               'voxel_coords': torch.zeros(5, 4, dtype=torch.long)})['voxel_features']
    assert out.shape == (5, 12)
    assert torch.all(out[:, 3:] == 0)                         # covariance block zeroed
    assert not torch.all(out[:, :3] == 0)                     # relative position kept
