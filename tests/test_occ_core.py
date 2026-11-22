"""pcocc's data-free parts (experiments_md 20261121_02 §5 checks 2-3; bug_prevention_checklist rules 1-2):
the ray traversal against an exact slab-test reference, the majority vote and its sentinels, the grid encoding.
"""
import numpy as np
import pytest

from pcseg import IGNORE
from pcocc import EMPTY, IGNORE_VOXEL
from pcocc.grid import Grid, PANDASET_GRID, encode, scoring_labels
from pcocc.raycast import traverse
from pcocc.vote import vote


# --- grid -----------------------------------------------------------------------------------------------------------

def test_pandaset_grid_is_semantickitti_volume_relative_to_ground():
    g = PANDASET_GRID
    assert g.shape == (256, 256, 32) and g.voxel == 0.2
    np.testing.assert_allclose(g.lo, [0.0, -25.6, -0.57])
    np.testing.assert_allclose(g.hi, [51.2, 25.6, 5.83])
    # 0.27 m below the ground (-0.30), 6.13 m above it, as SemanticKITTI's [-2.0, 4.4) about a sensor 1.73 m up
    np.testing.assert_allclose([-0.30 - g.lo[2], g.hi[2] + 0.30], [0.27, 6.13])


def test_grid_index_bounds():
    g = Grid((0, 0, 0), 0.5, (4, 3, 2))
    ijk, inside = g.index([[0, 0, 0], [1.99, 1.49, 0.99], [2.0, 0, 0], [-1e-9, 0, 0], [0.25, 0.75, 0.6]])
    assert inside.tolist() == [True, True, False, False, True]
    assert ijk[1].tolist() == [3, 2, 1] and ijk[4].tolist() == [0, 1, 1]


def test_encode_rejects_sentinel():
    assert encode(np.array([0, 16])).tolist() == [1, 17]
    with pytest.raises(ValueError):
        encode(np.array([0, IGNORE]))


def test_scoring_labels_folds_unknown_into_ignore():
    label = np.array([EMPTY, 3, IGNORE_VOXEL, EMPTY], dtype=np.uint8)
    unknown = np.array([True, False, False, False])
    assert scoring_labels(label, unknown).tolist() == [IGNORE_VOXEL, 3, IGNORE_VOXEL, EMPTY]


# --- vote -----------------------------------------------------------------------------------------------------------

def test_vote_majority_tie_and_ignore():
    #            voxel 0: 2 x class 4, 1 x class 1 -> 4
    #            voxel 1: 1 x class 5, 1 x class 2 -> tie -> lowest = 2
    #            voxel 2: only IGNORE -> IGNORE_VOXEL
    #            voxel 3: 3 x IGNORE, 1 x class 7 -> 7 (ignore never votes)
    flat = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3])
    cls = np.array([4, 4, 1, 5, 2, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, 7])
    label, occ, stats = vote(flat, cls, 17, 6)
    assert label.tolist() == [5, 3, IGNORE_VOXEL, 8, EMPTY, EMPTY]
    assert occ.tolist() == [True, True, True, True, False, False]
    assert stats['voting_points'] == 6 and stats['ignore_only'] == 1


def test_vote_order_independent():
    rng = np.random.default_rng(0)
    flat = rng.integers(0, 50, 5000)
    cls = rng.integers(-1, 17, 5000)
    cls[cls == -1] = IGNORE
    a = vote(flat, cls, 17, 50)[0]
    p = rng.permutation(5000)
    b = vote(flat[p], cls[p], 17, 50)[0]
    assert np.array_equal(a, b)
    # brute force
    for v in range(50):
        m = (flat == v) & (cls != IGNORE)
        if not (flat == v).any():
            assert a[v] == EMPTY
        elif not m.any():
            assert a[v] == IGNORE_VOXEL
        else:
            cnt = np.bincount(cls[m], minlength=17)
            assert a[v] == int(np.argmax(cnt)) + 1           # argmax takes the first (lowest) maximum


@pytest.mark.parametrize('bad', [-2, 17, -99])
def test_vote_rejects_out_of_range_class(bad):
    with pytest.raises(ValueError):
        vote(np.array([0, 1]), np.array([3, bad]), 17, 4)


def test_vote_rejects_out_of_range_voxel():
    with pytest.raises(ValueError):
        vote(np.array([0, 4]), np.array([3, 3]), 17, 4)
    with pytest.raises(ValueError):
        vote(np.array([-1]), np.array([3]), 17, 4)


# --- ray traversal --------------------------------------------------------------------------------------------------

def _reference(o, e, shape):
    """Every voxel whose box the segment o->e (grid units) crosses for a positive length, except the end voxel."""
    lo = np.maximum(np.floor(np.minimum(o, e)).astype(int), 0)
    hi = np.minimum(np.floor(np.maximum(o, e)).astype(int), np.array(shape) - 1)
    out = set()
    if np.any(lo > hi):
        return out
    d = e - o
    end = tuple(np.floor(e).astype(int))
    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            for k in range(lo[2], hi[2] + 1):
                s_in, s_out = 0.0, 1.0
                ok = True
                for a, c in enumerate((i, j, k)):
                    if d[a] == 0:
                        if not (c <= o[a] < c + 1):
                            ok = False
                        continue
                    t0, t1 = (c - o[a]) / d[a], (c + 1 - o[a]) / d[a]
                    s_in, s_out = max(s_in, min(t0, t1)), min(s_out, max(t0, t1))
                if ok and s_out - s_in > 1e-9 and (i, j, k) != end:
                    out.add((i, j, k))
    return out


def _run(o, e, g):
    m, _ = traverse(np.array([o]) * g.voxel + g.lo, np.array([e]) * g.voxel + g.lo, g)
    return set(map(tuple, np.argwhere(m)))


def test_traverse_matches_exact_reference_random_rays():
    g = Grid((-1.0, 2.0, 0.5), 0.2, (12, 10, 8))
    rng = np.random.default_rng(1)
    shape = np.array(g.shape)
    n_checked = 0
    for r in range(600):
        o = rng.uniform(-3, shape + 3)                     # starts inside or outside
        e = o + rng.normal(0, 7, 3)
        if r % 7 == 0:                                    # axis-parallel in one or two axes
            ax = rng.choice(3, size=rng.integers(1, 3), replace=False)
            e[ax] = o[ax]
        ref = _reference(o, e, g.shape)
        got = _run(o, e, g)
        assert got == ref, (r, o, e, sorted(got ^ ref))
        n_checked += len(ref)
    assert n_checked > 1000


def test_traverse_end_voxel_and_degenerate_rays():
    g = Grid((0, 0, 0), 1.0, (5, 5, 5))
    # along x from voxel 0 to a point in voxel 3: voxels 0, 1, 2 traversed, 3 (the point's) not
    assert _run([0.5, 0.5, 0.5], [3.5, 0.5, 0.5], g) == {(0, 0, 0), (1, 0, 0), (2, 0, 0)}
    # zero length, and a ray inside one voxel: nothing
    assert _run([1.5, 1.5, 1.5], [1.5, 1.5, 1.5], g) == set()
    assert _run([1.2, 1.5, 1.5], [1.8, 1.5, 1.5], g) == set()
    # ray from outside ending outside, passing through
    assert _run([-2.5, 2.5, 2.5], [7.5, 2.5, 2.5], g) == {(i, 2, 2) for i in range(5)}
    # ray entirely outside
    assert _run([-2.5, 2.5, 2.5], [-0.5, 2.5, 2.5], g) == set()
    assert _run([-2.5, 7.5, 2.5], [7.5, 7.5, 2.5], g) == set()
    # ray ending outside after entering: every voxel to the boundary
    assert _run([3.5, 3.5, 3.5], [3.5, 3.5, 9.0], g) == {(3, 3, 3), (3, 3, 4)}


def test_traverse_accumulates_and_counts():
    g = Grid((0, 0, 0), 1.0, (5, 5, 5))
    o = np.array([[0.5, 0.5, 0.5], [0.5, 4.5, 0.5], [-9, -9, -9]])
    e = np.array([[4.5, 0.5, 0.5], [0.5, 0.5, 0.5], [-8, -8, -8]])
    m, st = traverse(o, e, g)
    assert st == {'rays': 3, 'rays_in_grid': 2, 'voxel_visits': 8}
    assert m.sum() == 8                                   # x 0..3 and y 4..1; each end voxel excluded
    m2, _ = traverse(o[:1], e[:1], g, out=m.copy())
    assert np.array_equal(m2, m)


# --- movers: box frame and tracks -----------------------------------------------------------------------------------

def test_box_frame_matches_devkit_corners():
    from pandaset.geometry import center_box_to_corners
    from pcocc.movers import to_box, from_box, inside
    rng = np.random.default_rng(2)
    for _ in range(20):
        c, d, yaw = rng.normal(0, 30, 3), rng.uniform(0.5, 6, 3), rng.uniform(-4, 4)
        corners = center_box_to_corners([*c, *d, yaw])
        q = to_box(corners, c, yaw)
        np.testing.assert_allclose(np.abs(q), np.broadcast_to(d / 2, q.shape), atol=1e-9)
        assert inside(q * (1 - 1e-9), d).all() and not inside(q * 1.01, d).any()
        np.testing.assert_allclose(from_box(q, c, yaw), corners, atol=1e-9)


class _Cub:
    """A stand-in for movers.Cuboids with one uuid."""
    def __init__(self, times, centres, yaws):
        n = len(times)
        self.uid = np.zeros(n, dtype=np.int64)
        self.in_track = np.ones(n, dtype=bool)
        self.time = np.asarray(times, dtype=np.float64)
        self.centre = np.asarray(centres, dtype=np.float64)
        self.yaw = np.asarray(yaws, dtype=np.float64)


def test_track_interpolation_extrapolation_and_yaw_wrap():
    from pcocc.movers import Track
    tr = Track(_Cub([1.0, 0.0, 2.0], [[1, 0, 0], [0, 0, 0], [3, 0, 0]], [3.0, 3.1, -3.0]), 0)   # unsorted on purpose
    c, yaw = tr.at([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, -0.5])
    np.testing.assert_allclose(c[:, 0], [0, 0.5, 1, 2, 3, 4, -0.5])
    # 3.1 -> 3.0 is -0.1; 3.0 -> -3.0 the short way is +0.283 (through pi), never -6.0
    short = 2 * np.pi - 6.0
    expect = np.array([3.1, 3.05, 3.0, 3.0 + short / 2, 3.0 + short])
    np.testing.assert_allclose(np.angle(np.exp(1j * (yaw[:5] - expect))), 0, atol=1e-12)    # equal as angles
    assert np.all(np.abs(yaw) <= np.pi)
    one = Track(_Cub([5.0], [[1, 2, 3]], [0.3]), 0)
    c1, y1 = one.at([0.0, 9.0])
    np.testing.assert_allclose(c1, [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_allclose(y1, [0.3, 0.3])


def test_track_excludes_out_of_track_cuboids():
    from pcocc.movers import Track
    cub = _Cub([0.0, 1.0, 2.0], [[0, 0, 0], [50, 0, 0], [2, 0, 0]], [0, 0, 0])
    cub.in_track[1] = False
    c, _ = Track(cub, 0).at([1.0])
    np.testing.assert_allclose(c, [[1, 0, 0]])


class _Scans:
    def __init__(self, xyz, raw, frame=0, dev=0, t=0.0):
        n = len(xyz)
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.raw = np.asarray(raw)
        self.frame = np.full(n, frame, dtype=np.int16)
        self.dev = np.full(n, dev, dtype=np.int8)
        self.t = np.full(n, t)

    def __len__(self):
        return len(self.t)


class _OneMover:
    """movers.Cuboids stand-in: one moving 4 x 2 x 1.5 m box at the origin in frame 0, sensor -1."""
    def __init__(self):
        self.stationary = np.array([False])
        self.uid = np.array([0])
        self.frame = np.array([0])
        self.sensor = np.array([-1])
        self.centre = np.zeros((1, 3))
        self.yaw = np.zeros(1)
        self.dims = np.array([[4.0, 2.0, 1.5]])
        self.time = np.zeros(1)
        self.in_track = np.ones(1, dtype=bool)

    def devices(self, row):
        return (0, 1)


def test_mover_shell_takes_movable_points_only():
    from pcocc.movers import assign_movers
    #        inside        car point 0.15 m out   road point 0.15 m out   car point 0.4 m out    far car point
    xyz = [[0, 0, 0], [0, 1.15, 0], [0, -1.15, -0.5], [2.4, 0, 0], [9, 9, 0]]
    raw = [13, 13, 7, 13, 13]
    sc = _Scans(xyz, raw)
    usable = np.ones(len(xyz), dtype=bool)
    m0, box0, _ = assign_movers(_OneMover(), sc, usable, shell=0.0)
    assert m0.tolist() == [0, -1, -1, -1, -1]
    m2, box2, _ = assign_movers(_OneMover(), sc, usable, shell=0.2)
    assert m2.tolist() == [0, 0, -1, -1, -1]
    np.testing.assert_allclose(box2[1], [0, 1.15, 0])
    m5, _, _ = assign_movers(_OneMover(), sc, usable, shell=0.5)
    assert m5.tolist() == [0, 0, -1, 0, -1]
    with pytest.raises(AssertionError):
        assign_movers(_OneMover(), sc, usable, shell=2.5)
