"""CPU tests for pcdet/datasets/kitti/kitti_rings.py (experiments_md 20261011_02): ring recovery from the stored order
of a raw HDL-64E-like scan, every EVAL_RING_THIN mode, and the edge cases the KITTI loader can hand it."""
import numpy as np
import pytest

from pcdet.datasets.kitti import kitti_rings as kr

# KITTI's measured profile: 64 lasers, 0.3 deg apart above -7.6 deg, 0.5 deg below (two laser blocks)
ELEV = np.concatenate([2.0 - 0.3 * np.arange(32), (2.0 - 0.3 * 31) - 0.5 * np.arange(1, 33)])


def kitti_like_scan(seed=0, drop=0.15, step=0.18, offsets=0.0):
    """Laser-major: laser k (top first) turns once from azimuth 0 counter-clockwise (the measured structure: each ring
    ends where the azimuth passes 0 again), with dropped returns and random ranges."""
    rng = np.random.default_rng(seed)
    pts, ring = [], []
    for k, el in enumerate(np.radians(ELEV)):
        off = rng.uniform(0.0, offsets) if offsets else 0.0
        az = (off + np.arange(0.01, 360.0, step) + 180.0) % 360.0 - 180.0
        az = az[rng.random(len(az)) > drop]
        r = rng.uniform(5.0, 70.0, len(az))
        a = np.radians(az)
        pts.append(np.stack([r * np.cos(el) * np.cos(a), r * np.cos(el) * np.sin(a), r * np.sin(el),
                             rng.random(len(az))], 1))
        ring.append(np.full(len(az), k))
    return np.concatenate(pts).astype(np.float32), np.concatenate(ring)


def test_rings_recovered_exactly():
    pts, ring = kitti_like_scan()
    got, lev = kr.rings_from_order(pts[:, :3])
    assert len(lev) == 64
    assert np.mean(got == ring) > 0.999
    np.testing.assert_allclose(lev, ELEV, atol=0.01)


@pytest.mark.parametrize('n', [0, 1, 5])
@pytest.mark.parametrize('mode', ['rows', 'random', 'cols', 'random_cols', 'azbin', 'random_azbin', 'pattern'])
def test_tiny_clouds(n, mode):
    pts = np.random.default_rng(0).normal(size=(n, 4)).astype(np.float32) * 10
    cfg = {'MODE': mode, 'STRIDE': 2, 'AZ_RES_DEG': 0.332, 'SPACING_DEG': 1.33}
    keep = kr.eval_ring_thin_mask(pts, cfg, np.random.default_rng(0))
    assert keep.shape == (n,)
    assert kr.eval_ring_thin_points(pts, cfg, np.random.default_rng(0)).shape[1] == 4


@pytest.mark.parametrize('stride', [2, 3, 4])
def test_rows_and_their_count_matched_control(stride):
    pts, ring = kitti_like_scan()
    rows = kr.eval_ring_thin_mask(pts, {'MODE': 'rows', 'STRIDE': stride}, np.random.default_rng(1))
    assert set(np.unique(ring[rows])) == set(range(0, 64, stride))      # whole lines kept, the rest gone
    rand = kr.eval_ring_thin_mask(pts, {'MODE': 'random', 'STRIDE': stride}, np.random.default_rng(1))
    assert rand.sum() == rows.sum()
    assert len(np.unique(ring[rand])) == 64                             # the control keeps every line


@pytest.mark.parametrize('stride', [2, 4])
def test_cols_and_their_count_matched_control(stride):
    pts, ring = kitti_like_scan()
    cols = kr.eval_ring_thin_mask(pts, {'MODE': 'cols', 'STRIDE': stride}, np.random.default_rng(1))
    per_ring = np.bincount(ring[cols], minlength=64) / np.bincount(ring, minlength=64)
    assert np.all(np.abs(per_ring - 1.0 / stride) < 0.02)               # every line keeps 1 / stride of its returns
    rand = kr.eval_ring_thin_mask(pts, {'MODE': 'random_cols', 'STRIDE': stride}, np.random.default_rng(1))
    assert rand.sum() == cols.sum()


def test_azbin_one_point_per_line_per_bin_and_its_control():
    pts, ring = kitti_like_scan(drop=0.0)
    keep = kr.eval_ring_thin_mask(pts, {'MODE': 'azbin', 'AZ_RES_DEG': 0.332}, np.random.default_rng(0))
    b = np.floor((kr.azimuth_deg(pts[keep, :3]) + 180.0) / 0.332).astype(int)
    assert len(np.unique(ring[keep] * 10000 + b)) == keep.sum()          # never two points in one (line, bin)
    assert abs(keep.sum() / (64 * 360.0 / 0.332) - 1.0) < 0.02           # and every (line, bin) occupied
    rand = kr.eval_ring_thin_mask(pts, {'MODE': 'random_azbin', 'AZ_RES_DEG': 0.332}, np.random.default_rng(0))
    assert rand.sum() == keep.sum()


def test_pattern_keeps_lines_at_the_lattice_spacing():
    pts, ring = kitti_like_scan(drop=0.0)
    keep = kr.eval_ring_thin_mask(pts, {'MODE': 'pattern', 'SPACING_DEG': 1.33, 'AZ_RES_DEG': 0.332},
                                  np.random.default_rng(3))
    kept = np.unique(ring[keep])
    gaps = np.abs(np.diff(np.sort(ELEV[kept])))
    assert 1.0 <= np.median(gaps) <= 1.6                                 # 1.33 within one 0.3 / 0.5 deg ring step
    assert 18 <= len(kept) <= 22                                         # 26.8 deg of FOV / 1.33


def test_seeded_cut_is_reproducible():
    pts, _ = kitti_like_scan()
    for mode in ('random', 'random_cols', 'random_azbin', 'pattern'):
        cfg = {'MODE': mode, 'STRIDE': 2, 'AZ_RES_DEG': 0.332, 'SPACING_DEG': 1.33}
        a = kr.eval_ring_thin_mask(pts, cfg, np.random.default_rng(7))
        b = kr.eval_ring_thin_mask(pts, cfg, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)


def test_unknown_mode_raises():
    pts, _ = kitti_like_scan()
    with pytest.raises(ValueError):
        kr.eval_ring_thin_mask(pts, {'MODE': 'columns'}, np.random.default_rng(0))
