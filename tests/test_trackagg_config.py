"""The (c1) row differs from 27834 in exactly one variable (experiments_md 20261010_04 §5.1 item 5): the resolved
trackagg configs equal the z-buffer configs except for TRACK_AGG in the two nuScenes source blocks."""
import os

import pytest
from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
D = 'cfgs/da-ieee-access/centerpoint-accum-legaldepth-'


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(flat(v, key))
        else:
            out[key] = repr(v)
    return out


def resolved(path):
    cwd = os.getcwd()
    try:
        os.chdir(TOOLS)
        c = EasyDict()
        cfg_from_yaml_file(path, c)
        return c
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize('target', ['nuscenes2kitti', 'nuscenes2waymo'])
def test_trackagg_is_zbuffer_plus_track_agg_only(target):
    a = flat(resolved(f'{D}zbuffer-{target}.yaml'))
    b = flat(resolved(f'{D}trackagg-{target}.yaml'))
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    diff.discard('_BASE_CONFIG_')
    assert diff and all(k.split('.')[0] == 'DATA_CONFIGS' and k.split('.')[2] == 'TRACK_AGG' for k in diff), sorted(diff)
    assert {k.split('.')[1] for k in diff} == {'NUSCENES_N008', 'NUSCENES_N015'}
    c = resolved(f'{D}trackagg-{target}.yaml')
    for key in ('NUSCENES_N008', 'NUSCENES_N015'):
        t = c.DATA_CONFIGS[key].TRACK_AGG
        assert t.ICP and t.VISIBILITY and t.SENSOR_Z == 2.184
        assert t.CACHE_DIR.startswith('/home/koyama/data/') and t.RAYDROP.startswith('/home/koyama/data/')
        assert c.DATA_CONFIGS[key].get('LEVEL_COOR', None) is None
        assert [p.NAME for p in c.DATA_CONFIGS[key].DATA_PROCESSOR][2] == 'sample_points_learned'
