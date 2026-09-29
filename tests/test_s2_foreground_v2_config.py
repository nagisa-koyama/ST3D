"""centerpoint-accum-foreground-v2-nuscenes2kitti = its v1 parent + the v2 recipe, and nothing else.

Same pin as test_config_real_configs.test_foreground_v2_differs_from_v1_only_in_thresholds_and_class_set
for the Lyft pair: the S2 row's difference from the never-run v1 row must be the Car-only channel
and the per-class cuts, so the comparison against 26388 / 25780 measures the correction alone.
"""
import os
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcdet.config import cfg_from_yaml_file  # noqa: E402

V1 = 'cfgs/da-ieee-access/centerpoint-accum-foreground-nuscenes2kitti.yaml'
V2 = 'cfgs/da-ieee-access/centerpoint-accum-foreground-v2-nuscenes2kitti.yaml'


@pytest.fixture
def in_tools_dir():
    prev = os.getcwd()
    os.chdir(ROOT / 'tools')
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


def test_v2_differs_from_v1_only_in_cuts_and_channel_classes(in_tools_dir):
    v1, v2 = cfg_from_yaml_file(V1, EasyDict()), cfg_from_yaml_file(V2, EasyDict())
    a, b = _flat(v1), _flat(v2)
    allowed = {'SELF_TRAIN.SCORE_THRESH', 'SELF_TRAIN.NEG_THRESH'}
    diffs = sorted(k for k in set(a) | set(b) if not k.endswith('_BASE_CONFIG_')
                   and k not in allowed and not k.endswith('.HIST_DIST_FOREGROUND_CLASSES')
                   and str(a.get(k)) != str(b.get(k)))
    assert not diffs, diffs
    assert set(v2.DATA_CONFIGS) == {'NUSCENES_N008', 'NUSCENES_N015'}
    for name, blk in v2.DATA_CONFIGS.items():
        assert list(blk.HIST_DIST_FOREGROUND_CLASSES) == ['Car'], name
        assert blk.HIST_DIST_FOREGROUND_FROM_PSEUDO_LABELS is True, name
        assert blk.MAX_SWEEPS == 15 and blk.GT_BOXES_MOTION_COMPENSATION is True, name
    assert v2.SELF_TRAIN.SCORE_THRESH == [0.30, 0.18, 0.17]
    assert all(s > n for s, n in zip(v2.SELF_TRAIN.SCORE_THRESH, v2.SELF_TRAIN.NEG_THRESH))
    assert v2.DATA_CONFIG_TAR.USE_PSEUDO_LABEL is True
    assert v2.DATA_CONFIG_TAR.TEST.BOX_FILTER.FOV_FILTER is True
    assert v2.SELF_TRAIN.EPOCH_FOLLOWS == 'source'
