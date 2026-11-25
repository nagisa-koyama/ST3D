"""The viewpoint render kind gt_on_gtrays_p64pos of tools/analysis/pcseg_render_cache.py (experiments_md episodes/20261116_01
§8.11, arm v1): PandarGT's own rays cast from the Pandar64's position.

CPU only, synthetic points and an identity pose track:
- the kind differs from gt_on_gtrays (b0's cache) ONLY in the ray origin: same lattice, firing times, real-sensor cells;
- cast from the PandarGT's own origin, a frame renders back onto itself (the identity), and the viewpoint render's
  points lie on the template's directions, measured from the Pandar64's position;
- an unknown kind raises instead of falling back to gt_on_gtrays.
The >= 40-frame identity check of the unchanged kind against the built cache runs as a CPU job (§8.11).
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tools', 'analysis'))

from pcseg.render import angles_deg, to_sensor  # noqa: E402

MODELS = {'pandargt': {'origin': [1.22, 0.0, 1.573]}, 'pandar64': {'origin': [0.782, 0.001, 1.860]}}


def _builder():
    spec = importlib.util.spec_from_file_location(
        'pcseg_render_cache_mod', os.path.join(REPO, 'tools', 'analysis', 'pcseg_render_cache.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Track:
    """World = ego at every time (a parked car)."""

    def ego_at(self, xyz, times):
        return np.asarray(xyz, dtype=np.float64)

    def world_from_ego_at(self, xyz, times):
        return np.asarray(xyz, dtype=np.float64)

    def ego_static(self, xyz, frame):
        return np.asarray(xyz, dtype=np.float64)


def _frame(n=3000, seed=0):
    """A PandarGT-like frame: points 4-60 m ahead in a 60 x 20 deg window, one per direction, times by azimuth."""
    rs = np.random.default_rng(seed)
    az = rs.uniform(-30, 30, n)
    el = rs.uniform(-13, 7, n)
    r = rs.uniform(4, 60, n)
    a, e = np.radians(az), np.radians(el)
    q = np.stack([r * np.cos(e) * np.cos(a), r * np.cos(e) * np.sin(a), r * np.sin(e)], 1)
    xyz = q + np.array(MODELS['pandargt']['origin'])
    t = 100.0 + (az + 30) / 60 * 0.1
    return xyz, t


def test_viewpoint_kind_differs_from_b0s_kind_only_in_the_ray_origin():
    m = _builder()
    xyz, t = _frame()
    a = m.target_model('gt_on_gtrays', _Track(), 10, xyz, t, MODELS, None)
    b = m.target_model('gt_on_gtrays_p64pos', _Track(), 10, xyz, t, MODELS, None)
    La, Oa, Ra, ta, t0a, qa, ea = a
    Lb, Ob, Rb, tb, t0b, qb, eb = b
    assert np.array_equal(La.az_el, Lb.az_el) and La.tol == Lb.tol          # the same rays
    assert np.array_equal(Ra, Rb) and np.array_equal(Ra, np.eye(3))       # the attitude is not changed
    assert t0a == t0b and np.array_equal(qa, qb)                            # same start time, same real-sensor cells
    az, el, _ = angles_deg(qa[::7])
    assert np.array_equal(ta(az, el), tb(az, el))                           # same firing time per direction
    assert np.allclose(Oa, MODELS['pandargt']['origin']) and np.allclose(Ob, MODELS['pandar64']['origin'])
    assert ea == {} and np.allclose(eb['cast_origin'], MODELS['pandar64']['origin'])


def test_identity_from_the_own_origin_and_rays_from_the_pandar64_position():
    m = _builder()
    xyz, t = _frame()
    tr = _Track()
    L, O, R, time_of, t0, q, _ = m.target_model('gt_on_gtrays', tr, 10, xyz, t, MODELS, None)
    cell, rng, cells, win, out = m.render_frame(tr, 10, L, O, R, time_of, t, xyz)
    assert len(cells) == len(xyz)                                           # every ray keeps its own point
    assert np.allclose(out, xyz[win], atol=1e-9)
    L, O, R, time_of, t0, q, _ = m.target_model('gt_on_gtrays_p64pos', tr, 10, xyz, t, MODELS, None)
    cell, rng, cells, win, out = m.render_frame(tr, 10, L, O, R, time_of, t, xyz)
    assert 0 < len(cells) < len(xyz)                                        # a moved origin leaves rays empty
    d = out - np.array(MODELS['pandar64']['origin'])
    assert np.allclose(d / np.linalg.norm(d, axis=1, keepdims=True), L.rays(cells), atol=1e-9)
    src = xyz[win] - np.array(MODELS['pandar64']['origin'])
    assert np.allclose(np.linalg.norm(d, axis=1), np.linalg.norm(src, axis=1), atol=1e-9)   # range from the Pandar64
    _, _, r_src = angles_deg(to_sensor(xyz[win], np.array(MODELS['pandar64']['origin']), np.eye(3)))
    assert np.all(r_src > 0)


def test_unknown_kind_raises():
    m = _builder()
    xyz, t = _frame(200)
    with pytest.raises(ValueError):
        m.target_model('gt_on_gtrays_typo', _Track(), 10, xyz, t, MODELS, None)
    assert 'gt_on_gtrays_p64pos' in m.KINDS


def test_cache_compare_finds_a_changed_array_and_a_missing_frame(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'pcseg_render_cache_compare', os.path.join(REPO, 'tools', 'analysis', 'pcseg_render_cache_compare.py'))
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    rec = {'src_off': np.array([0, -1], np.int8), 'src_idx': np.array([3, 4], np.int32),
           'xyz': np.zeros((2, 3), np.float32), 'n_src': np.array([5, 6, 7], np.int32), 'mixed': np.array([0, 1], bool)}
    for d in ('old', 'new'):
        (tmp_path / d / '035').mkdir(parents=True)
        np.savez(tmp_path / d / '035' / '30.npz', **rec)
    assert c.compare(str(tmp_path / 'new'), str(tmp_path / 'old')) == {'frames': 1, 'differ': []}
    np.savez(tmp_path / 'new' / '035' / '30.npz', **dict(rec, src_idx=np.array([3, 5], np.int32)))
    assert c.compare(str(tmp_path / 'new'), str(tmp_path / 'old'))['differ'] == ['035/30.npz: src_idx differs']
    np.savez(tmp_path / 'new' / '035' / '31.npz', **rec)
    assert any('absent' in d for d in c.compare(str(tmp_path / 'new'), str(tmp_path / 'old'))['differ'])
