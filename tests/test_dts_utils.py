"""DTS operations ported to CenterPoint (pcdet/utils/dts_utils.py; experiments_md 20261003_05 §5 item 51). CPU only."""
import numpy as np

from pcdet.utils import dts_utils as d


def ring_point(phi, theta, r, *extra):
    return [np.cos(theta) * np.cos(phi) * r, np.cos(theta) * np.sin(phi) * r, np.sin(theta) * r, *extra]


def two_rings(phis0, phis1, th0=np.radians(-10.0), th1=np.radians(-8.667), r0=10.0, r1=12.0):
    pts = [ring_point(p, th0, r0, 0.2, 0) for p in phis0] + [ring_point(p, th1, r1, 0.6, 1) for p in phis1]
    return np.array(pts, dtype=np.float32)


def test_zero_probability_is_the_identity_and_draws_one_number_per_pair():
    pts = two_rings([0.1, 0.2], [0.1, 0.2])
    calls = []
    out = d.ring_upsample(pts, pts[:, 4], 0.0, 0.03, n_rings=32, rand=lambda: calls.append(1) or 0.5)
    assert out is pts and len(calls) == 31  # DTS draws np.random.rand() once per pair, in order


def test_midpoint_in_phi_theta_r_and_extra_columns_averaged():
    pts = two_rings([1.00, 2.00], [1.01, 2.50])  # second pair 0.5 rad apart: not formed
    out = d.ring_upsample(pts, pts[:, 4], 1.0, 0.03, n_rings=2)
    assert out.shape == (5, 5) and np.array_equal(out[:4], pts)  # inputs kept, first, in order
    phi, theta, r = d.polar(out[4:])
    assert np.allclose(phi, 1.005, atol=1e-5) and np.allclose(theta, np.radians(-9.3335), atol=1e-5)
    assert np.allclose(r, 11.0, atol=1e-4) and np.allclose(out[4, 3:], [0.4, 0.5])
    assert out.dtype == pts.dtype


def test_nearest_azimuth_partner_and_threshold_is_strict():
    pts = two_rings([1.000], [0.980, 1.029, 1.100])
    out = d.ring_upsample(pts, pts[:, 4], 1.0, 0.03, n_rings=2)
    assert len(out) == 5 and np.isclose(d.polar(out[4:])[0][0], 0.9900, atol=1e-5)  # 0.980 is nearer (0.020 vs 0.029)
    out = d.ring_upsample(two_rings([1.0], [1.04]), np.array([0, 1]), 1.0, 0.03, n_rings=2)
    assert len(out) == 2  # 0.04 >= 0.03: no pair


def test_no_pair_across_the_azimuth_seam():
    pts = two_rings([0.001], [2 * np.pi - 0.001])  # 0.002 rad apart on the circle, ~2 pi apart without wrap (DTS)
    assert len(d.ring_upsample(pts, pts[:, 4], 1.0, 0.03, n_rings=2)) == 2


def test_only_adjacent_rings_pair_and_empty_rings_are_skipped():
    pts = np.array([ring_point(1.0, -0.20, 10, 0, 0), ring_point(1.0, -0.15, 10, 0, 2)], dtype=np.float32)
    out = d.ring_upsample(pts, pts[:, 4], 1.0, 0.03, n_rings=3)  # ring 1 empty: neither (0,1) nor (1,2) forms
    assert len(out) == 2


def test_per_pair_probabilities_select_pairs_in_order():
    pts = np.array([ring_point(1.0, -0.20 + 0.02 * i, 10, 0, i) for i in range(4)], dtype=np.float32)
    draws = iter([0.9, 0.1, 0.9])  # pair (0,1) not drawn, (1,2) drawn, (2,3) not
    out = d.ring_upsample(pts, pts[:, 4], 0.5, 0.03, n_rings=4, rand=lambda: next(draws))
    assert len(out) == 5 and np.isclose(out[4, 4], 1.5)


def test_rings_for_a_many_point_frame_match_a_brute_force_reference():
    rng = np.random.default_rng(0)
    rows = []
    for ring in range(4):
        for p in rng.uniform(0, 2 * np.pi, 300):
            rows.append(ring_point(p, -0.3 + 0.023 * ring, rng.uniform(3, 60), rng.uniform(0, 1), ring))
    pts = np.array(rows, dtype=np.float32)
    out = d.ring_upsample(pts, pts[:, 4], 1.0, 0.03, n_rings=4)
    phi, th, r = d.polar(pts)
    ref = 0
    for i in range(3):
        a, b = np.flatnonzero(pts[:, 4] == i), np.flatnonzero(pts[:, 4] == i + 1)
        ref += int((np.abs(phi[a, None] - phi[None, b]).min(1) < 0.03).sum())  # DTS's dense argmin
    assert len(out) - len(pts) == ref
