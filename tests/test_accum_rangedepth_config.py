"""centerpoint-accum-rangedepth-nuscenes2kitti = the shrinking-ROS accumulation row (26536) + a per-range depth schedule."""
import os
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
import _init_path  # noqa: F401,E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets.nuscenes.nuscenes_dataset import sweep_min_range  # noqa: E402

FAMILY = 'cfgs/da-ieee-access/'
PARENT, CHILD = 'centerpoint-accum-rosshrink-nuscenes2kitti.yaml', 'centerpoint-accum-rangedepth-nuscenes2kitti.yaml'


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


def test_one_variable_from_the_shrinking_ros_accumulation_row(in_tools_dir):
    a = _flat(cfg_from_yaml_file(FAMILY + PARENT, EasyDict())); b = _flat(cfg_from_yaml_file(FAMILY + CHILD, EasyDict()))
    allowed_suffix = ('.MAX_SWEEPS', '.ACCUMULATION_DEPTH_BY_RANGE')
    diffs = sorted(k for k in set(a) | set(b) if not k.endswith('_BASE_CONFIG_')
                   and not k.endswith(allowed_suffix) and k != 'OPTIMIZATION.NUM_WORKERS'
                   and str(a.get(k)) != str(b.get(k)))
    assert not diffs, diffs


def test_schedule_is_consistent_with_max_sweeps_and_motion_compensation(in_tools_dir):
    cfg = cfg_from_yaml_file(FAMILY + CHILD, EasyDict())
    for name, blk in cfg.DATA_CONFIGS.items():
        sched = blk.ACCUMULATION_DEPTH_BY_RANGE
        assert max(n for _, n in sched) == blk.MAX_SWEEPS, name
        assert sched[0][0] == 0, name                         # the near field is covered
        assert blk.GT_BOXES_MOTION_COMPENSATION is True, name
        assert sweep_min_range(sched, blk.MAX_SWEEPS) is None  # nothing asks for more than MAX_SWEEPS
        assert sweep_min_range(sched, 1) == 0.0                # the second frame is used everywhere
    assert cfg.DATA_CONFIGS.NUSCENES_N008.MAX_SWEEPS == 30 and cfg.DATA_CONFIGS.NUSCENES_N015.MAX_SWEEPS == 20


def test_inherits_the_shrinking_car_interval(in_tools_dir):
    cfg = cfg_from_yaml_file(FAMILY + CHILD, EasyDict())
    for name, blk in cfg.DATA_CONFIGS.items():
        ros = next(a for a in blk.DATA_AUGMENTOR.AUG_CONFIG_LIST if a['NAME'] == 'random_object_scaling')
        assert ros['SCALE_UNIFORM_NOISE']['Car'] == [0.75, 1.00], name
