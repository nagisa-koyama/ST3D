"""CPU tests for pcdet/datasets/lyft/lyft_rings.py (experiments_md 20261010_07): de-skew of a motion-compensated
laser-major scan, ring recovery, RING_PATTERN thinning, and the edge cases a loader can hand it."""
import numpy as np
import pytest

from pcdet.datasets.lyft import lyft_rings as lr

PERIOD, REF = 0.1, 1.0


def synthetic_scan(n_lasers=10, el_top=5.0, gap=1.0, sign=1.0, a0=6.0, step=0.2, v=(12.0, 0.5, 0.1),
                   w=(0.02, -0.03, 0.05), seed=0, drop=0.1):
    """A laser-major scan as Lyft stores it: laser k (top first) sweeps once from azimuth a0 in direction `sign`;
    ranges random; then every point is MOTION-COMPENSATED to the end of the rotation (the inverse of lr.deskew).
    Returns (stored xyz, true ring per point, true firing-frame xyz)."""
    rng = np.random.default_rng(seed)
    pts, ring = [], []
    for k in range(n_lasers):
        el = np.radians(el_top - k * gap)
        az = a0 + sign * np.arange(0.0, 360.0, step)
        az = az[rng.random(len(az)) > drop]                  # dropped returns
        r = rng.uniform(8.0, 60.0, len(az))
        a = np.radians(az)
        pts.append(np.stack([r * np.cos(el) * np.cos(a), r * np.cos(el) * np.sin(a), r * np.sin(el)], 1))
        ring.append(np.full(len(az), k))
    fire = np.concatenate(pts)
    ring = np.concatenate(ring)
    dt = lr.firing_offset(lr.azimuth_deg(fire), sign, a0, PERIOD, REF)
    theta = np.outer(dt, np.asarray(w))                       # forward: the inverse of deskew, first order
    stored = fire + np.cross(theta, fire) + np.outer(dt, np.asarray(v))
    T = np.eye(4)                                             # previous sweep 0.2 s earlier, as in the infos
    T[:3, 3] = -np.asarray(v) * 0.2
    wz = np.asarray(w) * 0.2
    T[:3, :3] = np.eye(3) + np.array([[0, -(-wz[2]), (-wz[1])], [(-wz[2]), 0, -(-wz[0])], [-(-wz[1]), (-wz[0]), 0]])
    info = {'sweeps': [{'transform_matrix': T, 'time_lag': 0.2}]}
    return stored, ring, fire, info


def test_ego_motion_reads_velocity_and_rates_back():
    _, _, _, info = synthetic_scan()
    v, w = lr.ego_motion(info)
    np.testing.assert_allclose(v, [12.0, 0.5, 0.1], atol=1e-9)
    np.testing.assert_allclose(w, [0.02, -0.03, 0.05], atol=1e-6)
    v, w = lr.ego_motion(info, yaw_only=True)
    np.testing.assert_allclose(w, [0.0, 0.0, 0.05], atol=1e-6)


@pytest.mark.parametrize('sweeps', [None, [], [{'transform_matrix': None, 'time_lag': 0.2}],
                                    [{'transform_matrix': np.eye(4), 'time_lag': 0.0}]])
def test_ego_motion_without_a_usable_sweep_is_none(sweeps):
    assert lr.ego_motion({'sweeps': sweeps}) is None


def test_deskew_undoes_the_compensation():
    stored, ring, fire, info = synthetic_scan()
    q = lr.deskew_frame(stored, lr.ego_motion(info), 1.0, 6.0, PERIOD, REF, iters=2)
    err = np.abs(lr.elevation_deg(q) - lr.elevation_deg(fire))
    assert np.percentile(err, 99) < 0.01
    # points fired just after the sweep start whose compensated azimuth crossed back over it read one rotation late
    # until the phase is unwrapped per ring: the full recovery removes them
    q2, got, lev = lr.deskew_and_rings(stored, lr.ego_motion(info), 1.0, 6.0, PERIOD, REF)
    assert np.max(np.abs(lr.elevation_deg(q2) - lr.elevation_deg(fire))) < 0.01
    assert np.mean(got == ring) > 0.999
    # without it the compensation moves elevation by far more than a 1-deg ring gap's half on near points
    assert np.max(np.abs(lr.elevation_deg(stored) - lr.elevation_deg(fire))) > 0.5


@pytest.mark.parametrize('sign,a0', [(1.0, 6.0), (-1.0, -1.65)])
def test_rings_recovered_exactly_after_deskew(sign, a0):
    stored, ring, _, info = synthetic_scan(sign=sign, a0=a0)
    q, got, lev = lr.deskew_and_rings(stored, lr.ego_motion(info), sign, a0, PERIOD, REF)
    assert len(lev) == 10
    assert np.mean(got == ring) > 0.999
    assert np.all(np.diff(lev) < 0)                           # top laser first


def test_rings_without_motion_are_unchanged_input():
    stored, ring, fire, _ = synthetic_scan(v=(0, 0, 0), w=(0, 0, 0))
    q = lr.deskew_frame(stored, None, 1.0, 6.0, PERIOD, REF)
    np.testing.assert_array_equal(q, stored)
    got, lev = lr.rings_from_deskewed(q, 1.0, 6.0)
    assert len(lev) == 10 and np.mean(got == ring) > 0.999


@pytest.mark.parametrize('n', [0, 1, 5])
def test_tiny_clouds(n):
    xyz = np.random.default_rng(0).normal(size=(n, 3)) * 10
    q, ring, lev = lr.deskew_and_rings(xyz, (np.zeros(3), np.zeros(3)), 1.0, 0.0, PERIOD, REF)
    assert len(ring) == n
    keep, kept = lr.ring_pattern_mask(xyz, ring, lev, 1.33, 0.332, 0.3, 0.7)
    assert keep.shape == (n,)


def test_pattern_keeps_rings_near_the_lattice_and_one_point_per_bin():
    stored, ring, _, info = synthetic_scan(n_lasers=24, el_top=7.0, gap=0.33, drop=0.0)
    q = lr.deskew_frame(stored, lr.ego_motion(info), 1.0, 6.0, PERIOD, REF)
    got, lev = lr.rings_from_deskewed(q, 1.0, 6.0)
    keep, kept = lr.ring_pattern_mask(stored, got, lev, 1.33, 0.332, 0.25, 0.5)
    gaps = np.abs(np.diff(np.sort(lev[kept])))
    assert 1.0 < np.median(gaps) < 1.7                        # the lattice's 1.33 within one 0.33 ring step
    assert set(np.unique(got[keep])) == set(kept)
    # one point per kept ring per 0.332 deg bin: at a 0.2 deg source step about 60% of a kept ring survives
    per_ring = np.array([np.sum(keep & (got == k)) / np.sum(got == k) for k in kept])
    assert np.all((per_ring > 0.5) & (per_ring < 0.7))
