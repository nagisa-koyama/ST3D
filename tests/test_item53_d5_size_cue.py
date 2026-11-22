"""Item 53 D5 size-cue slopes, pull fractions and readings (tools/analysis/item53_d5_size_cue.py). CPU only, synthetic.

Camera-frame boxes (x, y, z, l, h, w, ry): l is index 3, h 4, w 5. A "pulled" arm mixes the none prediction with a fixed
size in log space (weight 0.7); a "uniform" arm multiplies every size by exp(0.12). The uniform arm also moves the median
toward the larger source size, which is why reading 6 alone must not state a pull.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'analysis'))
pytest.importorskip('shapely')
import item53_d5_size_cue as m  # noqa: E402

NUSC = {'median': {'l': 4.6, 'w': 1.92, 'h': 1.68}}


def _arms(kind, n_frames=400, per=4, seed=0):
    rng = np.random.RandomState(seed)
    n = n_frames * per
    g = np.zeros((n, 7)); g[:, 0] = 1; g[:, 2] = 20
    g[:, 3] = np.exp(rng.normal(np.log(3.9), 0.08, n))
    g[:, 4] = np.exp(rng.normal(np.log(1.5), 0.05, n))
    g[:, 5] = np.exp(rng.normal(np.log(1.62), 0.04, n))
    p0 = g.copy()
    for i in (3, 4, 5):
        p0[:, i] = np.exp(0.9 * np.log(g[:, i]) + 0.1 * np.log(g[:, i]).mean() + rng.normal(0, 0.02, n) + 0.03)
    p1 = p0.copy()
    src = {3: 4.63, 4: 1.74, 5: 1.96}
    for i in (3, 4, 5):
        if kind == 'pulled':
            p1[:, i] = np.exp(0.3 * np.log(p0[:, i]) + 0.7 * np.log(src[i]))
        else:
            p1[:, i] = p0[:, i] * np.exp(0.12)
    frames = np.repeat(['f%04d' % j for j in range(n_frames)], per)
    return g, p0, p1, frames


def test_a_pulled_arm_loses_the_cue_and_moves_to_the_source():
    g, p0, p1, fr = _arms('pulled')
    s = m.point_stats(g, p0, p1, NUSC['median'])
    ci = m.bootstrap(g, p0, p1, fr, NUSC['median'], 200)
    for d in ('l', 'w'):
        assert s[d]['rho'] == pytest.approx(0.3, abs=0.01)
        assert m.reading5_one(s[d], ci[d]) == 'size cue lost'
        assert m.reading6_one(s[d], ci[d]) == 'moves to the source size'
    assert m.combined(['size cue lost'] * 4, ['moves to the source size'] * 4) == m.PULL


def test_a_uniform_inflation_keeps_the_cue_but_also_moves_toward_the_source():
    g, p0, p1, fr = _arms('uniform')
    s = m.point_stats(g, p0, p1, NUSC['median'])
    ci = m.bootstrap(g, p0, p1, fr, NUSC['median'], 200)
    r5 = [m.reading5_one(s[d], ci[d]) for d in ('l', 'w')]
    r6 = [m.reading6_one(s[d], ci[d]) for d in ('l', 'w')]
    assert all(r == 'cue kept, uniform inflation' for r in r5)
    assert s['l']['rho'] == pytest.approx(1.0, abs=1e-6)
    assert all(r == 'moves to the source size' for r in r6)                  # the stated limit of reading 6
    assert m.combined(r5 * 2, r6 * 2) == m.UNIFORM


def test_the_bootstrap_resamples_whole_frames():
    frames = np.array(['b', 'a', 'b', 'c', 'a', 'b'])
    groups = m.frame_groups(frames)
    rng = np.random.RandomState(1)
    for _ in range(20):
        idx = m.frame_resample(groups, rng)
        picked = frames[idx]
        for f in set(picked):
            assert (picked == f).sum() % (frames == f).sum() == 0           # every draw brings all its boxes


def test_reading_bands_and_guard():
    s = lambda rho, b0=0.8: {'rho': rho, 'b_none': b0}  # noqa: E731
    ci = lambda lo, hi, b0lo=0.5: {'rho': [lo, hi], 'b_none': [b0lo, 1.0]}  # noqa: E731
    assert m.reading5_one(s(0.5), ci(0.3, 0.79)) == 'size cue lost'
    assert m.reading5_one(s(0.5), ci(0.3, 0.8)) == 'unresolved'
    assert m.reading5_one(s(0.8), ci(0.51, 1.0)) == 'cue kept, uniform inflation'
    assert m.reading5_one(s(0.65), ci(0.55, 0.75)) == 'intermediate'
    assert m.reading5_one(s(0.3, b0=0.19), ci(0.1, 0.4)) == 'no cue in none to lose'
    assert m.reading5_one(s(0.3), ci(0.1, 0.4, b0lo=0.0)) == 'no cue in none to lose'
    f = lambda v, lo, hi: ({'f': v}, {'f': [lo, hi]})  # noqa: E731
    assert m.reading6_one(*f(0.7, 0.5, 0.9)) == 'moves to the source size'
    assert m.reading6_one(*f(0.7, 0.49, 0.9)) == 'partly'
    assert m.reading6_one(*f(0.3, 0.1, 0.5)) == 'does not move'
    assert m.combined(['size cue lost'] * 3 + ['intermediate'], ['moves to the source size'] * 4).startswith('as measured')


def test_nuscenes_sizes_take_car_with_points_only():
    infos = [{'gt_names': np.array(['car', 'truck', 'car', 'car']),
              'gt_boxes': np.array([[0, 0, 0, 4.0, 1.9, 1.6, 0], [0, 0, 0, 9, 3, 3, 0], [0, 0, 0, 5.0, 2.0, 1.8, 0],
                                    [0, 0, 0, 20, 20, 20, 0]], float),
              'num_lidar_pts': np.array([5, 5, 1, 0])}]
    r = m.nuscenes_car_sizes(infos)
    assert r['n'] == 2 and r['median']['l'] == pytest.approx(4.5) and r['median']['w'] == pytest.approx(1.95)
    assert r['median']['h'] == pytest.approx(1.7)
