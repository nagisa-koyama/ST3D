"""The learned-sampler rows (experiments_md/20261003_04 section 6) differ from their parents only by the
sampler step, resolved through the repo's loader."""
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pcdet.config import cfg_from_yaml_file  # noqa: E402


@pytest.fixture(autouse=True)
def in_tools_dir(monkeypatch):
    monkeypatch.chdir(ROOT / 'tools')


def _load(name):
    cfg = EasyDict(); cfg_from_yaml_file(name, cfg); return cfg


def _sources(cfg):
    return cfg.get('DATA_CONFIGS') or {'DATA_CONFIG': cfg.DATA_CONFIG}


@pytest.mark.parametrize('child, parent', [
    ('cfgs/da-ieee-access/centerpoint-accum-rosext-learned-nuscenes2kitti.yaml', 'cfgs/da-ieee-access/centerpoint-accum-rosext-nuscenes2kitti.yaml'),
    ('cfgs/da-ieee-access/centerpoint-accum-learned-pandaset-spin2flash.yaml', 'cfgs/da-ieee-access/centerpoint-accum-pandaset-spin2flash.yaml'),
])
def test_only_the_sampler_step_is_added(child, parent):
    c, p = _load(child), _load(parent)
    assert c.MODEL == p.MODEL and c.OPTIMIZATION == p.OPTIMIZATION and c.DATA_CONFIG_TAR == p.DATA_CONFIG_TAR
    for key, dc in _sources(c).items():
        dp = _sources(p)[key]
        names = [s.NAME for s in dc.DATA_PROCESSOR]
        assert names == ['mask_points_and_boxes_outside_range', 'shuffle_points', 'sample_points_learned', 'transform_points_to_voxels']
        assert 'sample_points_hist_based' not in names
        assert [dict(s) for s in dc.DATA_PROCESSOR if s.NAME != 'sample_points_learned'] == [dict(s) for s in dp.DATA_PROCESSOR]
        assert {k: v for k, v in dc.items() if k != 'DATA_PROCESSOR'} == {k: v for k, v in dp.items() if k != 'DATA_PROCESSOR'}
        assert 'HIST_DIST_FOV_DEGREE' not in dc and not dc.get('HIST_DIST_ON_THE_FLY', False), 'no per-pair density key may remain'
        step = [s for s in dc.DATA_PROCESSOR if s.NAME == 'sample_points_learned'][0]
        assert step.WEIGHTS.endswith('.npz')
