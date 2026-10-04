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
        # ABSOLUTE and outside the repo: run_sourceonly_2gpu.sh binds a snapshot that excludes output/ over
        # the repo path, so a repo-relative weights path does not exist inside the job (27257 died on it).
        assert step.WEIGHTS.startswith('/home/koyama/data/'), step.WEIGHTS


L0_S2 = ('cfgs/da-ieee-access/centerpoint-accum-rosext-learnedinit-nuscenes2kitti.yaml',
         'cfgs/da-ieee-access/centerpoint-accum-rosext-learned-nuscenes2kitti.yaml')


def test_l0_only_swaps_in_the_initial_weights():
    c, p = _load(L0_S2[0]), _load(L0_S2[1])
    assert c.MODEL == p.MODEL and c.OPTIMIZATION == p.OPTIMIZATION and c.DATA_CONFIG_TAR == p.DATA_CONFIG_TAR
    for key, dc in _sources(c).items():
        dp = _sources(p)[key]
        assert {k: v for k, v in dc.items() if k != 'DATA_PROCESSOR'} == {k: v for k, v in dp.items() if k != 'DATA_PROCESSOR'}
        for sc, sp in zip(dc.DATA_PROCESSOR, dp.DATA_PROCESSOR):
            if sc.NAME == 'sample_points_learned':
                assert sc.WEIGHTS == sp.WEIGHTS.replace('.npz', '_init.npz')
            else:
                assert dict(sc) == dict(sp)


@pytest.mark.parametrize('child, parent', [
    # S1: the rule's step is SWAPPED for the sampler, and the on-the-fly calibration switched off.
    ('cfgs/da-ieee-access/centerpoint-learned-lyft2nuscenes.yaml', 'cfgs/da-ieee-access/centerpoint-global-lyft2nuscenes.yaml'),
    ('cfgs/da-ieee-access/centerpoint-st3d-learned-lyft2nuscenes.yaml', 'cfgs/da-ieee-access/centerpoint-st3d-global-lyft2nuscenes.yaml'),
])
def test_s1_only_the_rule_step_is_swapped(child, parent):
    c, p = _load(child), _load(parent)
    assert c.MODEL == p.MODEL and c.OPTIMIZATION == p.OPTIMIZATION and c.DATA_CONFIG_TAR == p.DATA_CONFIG_TAR
    assert c.get('SELF_TRAIN') == p.get('SELF_TRAIN')
    for key, dc in _sources(c).items():
        dp = _sources(p)[key]
        assert [s.NAME for s in dc.DATA_PROCESSOR] == [
            'sample_points_learned' if s.NAME == 'sample_points_hist_based' else s.NAME for s in dp.DATA_PROCESSOR]
        assert [dict(s) for s in dc.DATA_PROCESSOR if s.NAME != 'sample_points_learned'] == \
               [dict(s) for s in dp.DATA_PROCESSOR if s.NAME != 'sample_points_hist_based']
        assert dc.HIST_DIST_ON_THE_FLY is False and dp.HIST_DIST_ON_THE_FLY is True
        skip = ('DATA_PROCESSOR', 'HIST_DIST_ON_THE_FLY', '_BASE_CONFIG_')  # the parent keeps its resolved base path as a key
        assert {k: v for k, v in dc.items() if k not in skip} == {k: v for k, v in dp.items() if k not in skip}
        assert not dc.get('HIST_DIST_FOREGROUND_FROM_PSEUDO_LABELS', False)
        step = [s for s in dc.DATA_PROCESSOR if s.NAME == 'sample_points_learned'][0]
        assert step.WEIGHTS.startswith('/home/koyama/data/samplers/s1_') and step.WEIGHTS.endswith('.npz')


def test_l4_is_the_s2_headline_plus_the_sampler_and_its_own_car_cut():
    c = _load('cfgs/da-ieee-access/centerpoint-accum-st3d-learned-nuscenes2kitti.yaml')
    p = _load('cfgs/da-ieee-access/centerpoint-accum-st3d-nuscenes2kitti.yaml')
    l3 = _load('cfgs/da-ieee-access/centerpoint-accum-rosext-learned-nuscenes2kitti.yaml')
    assert c.MODEL == p.MODEL and c.OPTIMIZATION == p.OPTIMIZATION and c.DATA_CONFIG_TAR == p.DATA_CONFIG_TAR
    assert {k: v for k, v in c.SELF_TRAIN.items() if k != 'SCORE_THRESH'} == {k: v for k, v in p.SELF_TRAIN.items() if k != 'SCORE_THRESH'}
    assert list(c.SELF_TRAIN.SCORE_THRESH[1:]) == list(p.SELF_TRAIN.SCORE_THRESH[1:])
    assert 0.05 < float(c.SELF_TRAIN.SCORE_THRESH[0]) < 0.9
    for key, dc in _sources(c).items():
        assert dc == _sources(l3)[key], 'L4 sources must be exactly L3 (the teacher) sources'
