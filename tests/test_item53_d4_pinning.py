"""Item 53 D1b / D4 per-box pinning geometry and readings (tools/analysis/item53_d4_pinning.py). CPU only, synthetic boxes.

KITTI camera frame, (x, y_bottom, z, l, h, w, ry), y DOWN. The pinned boxes here are built from the KITTI devkit's own
corner convention (roty: length along (cos ry, 0, -sin ry), width along (sin ry, 0, cos ry)), independently of the tool,
so a wrong heading sign in the tool fails these tests.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'analysis'))
pytest.importorskip('shapely')
import item53_d4_pinning as m  # noqa: E402


def pinned(g, dl, dw, dh):
    """g grown by dl, dw, dh with the BEV corner nearest along the camera ray and the roof fixed (devkit axes)."""
    c, s = np.cos(g[6]), np.sin(g[6])
    e_l, e_w = np.array([c, -s]), np.array([s, c])
    u = np.array([g[0], g[2]]) / np.hypot(g[0], g[2])
    ctr = np.array([g[0], g[2]])
    corners = [ctr + a * g[3] / 2 * e_l + b * g[5] / 2 * e_w for a in (-1, 1) for b in (-1, 1)]
    near = min(corners, key=lambda q: q @ u)
    sl, sw = np.sign((ctr - near) @ e_l), np.sign((ctr - near) @ e_w)     # from the near corner into the box
    new = near + sl * (g[3] + dl) / 2 * e_l + sw * (g[5] + dw) / 2 * e_w
    p = g.copy()
    p[0], p[2] = new
    p[3], p[5], p[4] = g[3] + dl, g[5] + dw, g[4] + dh
    p[1] = g[1] + dh                                                      # bottom lower by dh (y down), roof fixed
    return p


@pytest.mark.parametrize('ry', [0.0, 0.3, 1.2, -2.0, np.pi / 2, 3.0])
@pytest.mark.parametrize('xz', [(0.0, 20.0), (5.0, 20.0), (-8.0, 12.0)])
def test_a_box_pinned_to_its_near_faces_moves_by_exactly_the_pinned_prediction(ry, xz):
    g = np.array([xz[0], 1.6, xz[1], 3.9, 1.53, 1.6, ry])
    p = pinned(g, 0.6, 0.25, 0.2)
    dr, x, c, s = m.ray_geometry(p, g)
    assert dr == pytest.approx(x, abs=1e-9) and x > 0
    assert c * c + s * s == pytest.approx(1.0)
    b_up, t_up, dh = m.vertical(p, g)
    assert b_up == pytest.approx(-0.2) and t_up == pytest.approx(0.0, abs=1e-12) and dh == pytest.approx(0.2)


def test_end_on_and_side_on_take_length_and_width():
    g = np.array([0.0, 1.6, 20.0, 3.9, 1.53, 1.6, np.pi / 2])            # length along the ray (end-on)
    assert m.ray_geometry(pinned(g, 0.6, 0.2, 0.0), g)[1] == pytest.approx(0.3)
    g[6] = 0.0                                                           # length across the ray (side-on)
    assert m.ray_geometry(pinned(g, 0.6, 0.2, 0.0), g)[1] == pytest.approx(0.1)


def _pairs(rng, kind, n=300):
    out = {}
    for i in range(n):
        g = np.array([rng.uniform(-10, 10), 1.6, rng.uniform(8, 40), 3.9, 1.53, 1.6, rng.uniform(-np.pi, np.pi)])
        dl, dw, dh = rng.normal(0.6, 0.3), rng.normal(0.25, 0.1), rng.normal(0.2, 0.05)
        if kind == 'pinned':
            p = pinned(g, dl, dw, dh)
        else:                                                            # grows about its centre, middle-anchored
            p = g.copy(); p[3] += dl; p[5] += dw; p[4] += dh; p[1] += dh / 2
        p[0] += rng.normal(0, 0.05); p[2] += rng.normal(0, 0.05)
        out[('f%d' % i, 0)] = (p, g)
    return out


def test_pinning_reads_pinned_boxes_and_centred_growth_apart():
    rng = np.random.RandomState(0)
    a = m.pinning(_pairs(rng, 'pinned'))
    assert a['front_back']['b'] == pytest.approx(1.0, abs=0.1) and a['front_back']['r'] > 0.8
    assert m.reading3_one(a['front_back']) == 'pinned to the near faces'
    assert a['up_down']['f_down'] == pytest.approx(1.0, abs=0.05) and m.reading4_one(a['up_down']) == 'pinned to the roof'
    c = m.pinning(_pairs(rng, 'centred'))
    assert abs(c['front_back']['b']) < 0.3 and m.reading3_one(c['front_back']) == 'not pinned'
    assert m.reading4_one(c['up_down']) == 'grows about its middle'


def test_reading_boundaries():
    assert m.reading3_one({'b': 1.0, 'r': 0.29}) == 'not pinned'
    assert m.reading3_one({'b': 0.49, 'r': 0.9}) == 'not pinned'
    assert m.reading3_one({'b': 0.5, 'r': 0.3}) == 'pinned to the near faces'
    assert m.reading3_one({'b': 1.6, 'r': 0.5}) == 'pushed beyond the pinning'
    assert m.reading3_one({'b': float('nan'), 'r': float('nan')}) == 'not pinned'
    ud = lambda dh, f: {'median_dh': dh, 'f_down': f}  # noqa: E731
    assert m.reading4_one(ud(0.04, 1.0)) == 'no height growth'
    assert m.reading4_one(ud(0.2, 0.75)) == 'pinned to the roof'
    assert m.reading4_one(ud(0.2, 0.25)) == 'pinned to the ground'
    assert m.reading4_one(ud(0.2, 0.5)) == 'grows about its middle'
    assert m.agree(['a', 'a']) == 'a' and m.agree(['a', 'b']).startswith('mixed')


def test_reproduces_needs_the_same_sign_at_both_seeds():
    full = [{c: {'reading': 'shifted', 'shift': 0.12} for c in m.COMPARED}] * 2
    same = [{c: {'reading': 'shifted', 'shift': 0.2} for c in m.COMPARED}] * 2
    flip = [dict(same[0], log_l={'reading': 'shifted', 'shift': -0.2}), same[1]]
    assert all(m.reproduces(same, full).values())
    r = m.reproduces(flip, full)
    assert r['log_l'] is False and r['log_w'] is True


def test_provenance_tolerance_and_exact_count():
    row = lambda n, l: {'matched': n, 'length': l, 'width': 0.06, 'height': 0.0, 'pass_3d_0.7': 0.61}  # noqa: E731
    printed = {'none': (4781, 0.14, 0.06, -0.00, 0.61)}
    assert m.provenance({'none': row(4781, 0.144)}, printed)['none'] is True
    assert m.provenance({'none': row(4781, 0.146)}, printed)['none'] is False
    assert m.provenance({'none': row(4780, 0.14)}, printed)['none'] is False
