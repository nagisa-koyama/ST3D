"""Item 56's two halves of the S' cut (experiments_md 20261003_05 §5 item 56, design fixed 2026-11-27 00:02 JST):
kitti_rings `pattern_rows` (the rings `pattern` keeps, all their points) and `pattern_az` (every ring at `pattern`'s azimuth
step), and the four eval configs. `pattern` itself must be unchanged; pattern = pattern_rows AND pattern_az exactly.
The last two tests read 40 KITTI val frames when the data are mounted (identity check over >= 40 frames, checklist rule 8)."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from pcdet.datasets.kitti import kitti_rings as kr
from pcdet.datasets.lyft.lyft_rings import ring_pattern_mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_kitti_rings import kitti_like_scan  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
CFG = {'SPACING_DEG': 1.33, 'AZ_RES_DEG': 0.332}


def masks(pts, seed):
    return {m: kr.eval_ring_thin_mask(pts, dict(CFG, MODE=m), np.random.default_rng(seed))
            for m in ('pattern', 'pattern_rows', 'pattern_az')}


def old_pattern(pts, seed):
    """`pattern` as it stood before item 56: ring_pattern_mask with the two draws as its arguments."""
    ring, lev = kr.rings_from_order(pts[:, :3])
    rng = np.random.default_rng(seed)
    return ring_pattern_mask(pts[:, :3], ring, lev, 1.33, 0.332, rng.random(), rng.random())[0]


@pytest.mark.parametrize('seed', [0, 3, 11])
def test_pattern_is_the_intersection_of_its_halves(seed):
    pts, ring = kitti_like_scan(seed=seed, drop=0.1)
    m = masks(pts, seed)
    np.testing.assert_array_equal(m['pattern'], old_pattern(pts, seed))
    np.testing.assert_array_equal(m['pattern'], m['pattern_rows'] & m['pattern_az'])
    assert set(np.unique(ring[m['pattern_rows']])) == set(np.unique(ring[m['pattern']]))   # the same rings, whole
    rows_ring = np.isin(ring, np.unique(ring[m['pattern']]))
    np.testing.assert_array_equal(m['pattern_rows'], rows_ring)
    assert len(np.unique(ring[m['pattern_az']])) == 64                                     # every ring kept
    per_ring = np.bincount(ring[m['pattern_az']], minlength=64)
    n_bins = round(360.0 / 0.332)
    assert per_ring.max() <= n_bins and per_ring.min() >= 0.85 * n_bins                    # one point per (ring, bin), bins filled


@pytest.mark.parametrize('n', [0, 1, 5])
@pytest.mark.parametrize('mode', ['pattern_rows', 'pattern_az'])
def test_tiny_clouds(n, mode):
    pts = np.random.default_rng(0).normal(size=(n, 4)).astype(np.float32) * 10
    assert kr.eval_ring_thin_mask(pts, dict(CFG, MODE=mode), np.random.default_rng(0)).shape == (n,)


def test_configs_change_only_the_mode():
    import os
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        for row in ('o', 'p'):
            base = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/item47/centerpoint-item47-kitti-%s-on-nuscenes-lattice.yaml'
                                      % row, EasyDict())
            for arm in ('rows', 'az'):
                c = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/item56/centerpoint-item56-kitti-%s-on-nuscenes-%s.yaml'
                                       % (row, arm), EasyDict())
                assert dict(c.DATA_CONFIG_TAR.EVAL_FACTOR_SWAP.RING_THIN) == dict(CFG, MODE='pattern_' + arm)
                c.DATA_CONFIG_TAR.EVAL_FACTOR_SWAP.RING_THIN.MODE = 'pattern'
                assert c.pop('_BASE_CONFIG_').endswith('item47-kitti-%s-on-nuscenes-lattice.yaml' % row)
                assert c == {k: v for k, v in base.items() if k != '_BASE_CONFIG_'}             # nothing else differs
    finally:
        os.chdir(cwd)


def _real_frames(k=40):
    infos = TOOLS.parent / 'data/kitti/kitti_infos_val.pkl'
    if not infos.exists():
        pytest.skip('KITTI infos not mounted')
    velo = TOOLS.parent / 'data/kitti/training/velodyne'
    for info in pickle.load(open(infos, 'rb'))[:k]:
        fid = info['point_cloud']['lidar_idx']
        yield fid, np.fromfile(velo / ('%s.bin' % fid), dtype=np.float32).reshape(-1, 4)


def test_on_real_frames_the_halves_intersect_to_s_prime_through_the_loader_path():
    from pcdet.datasets.kitti import kitti_factor_swap as fs
    n = 0
    for fid, pts in _real_frames():
        ring, lev = kr.rings_from_order(pts[:, :3])
        rng = fs.frame_rng(fid, 'RING_THIN')
        old = ring_pattern_mask(pts[:, :3], ring, lev, 1.33, 0.332, rng.random(), rng.random())[0]
        got = {m: fs.apply_points(pts, {'RING_THIN': dict(CFG, MODE=m)}, fid) for m in ('pattern', 'pattern_rows', 'pattern_az')}
        np.testing.assert_array_equal(got['pattern'], pts[old])                      # S' unchanged by item 56's edit
        rows = kr.eval_ring_thin_mask(pts, dict(CFG, MODE='pattern_rows'), fs.frame_rng(fid, 'RING_THIN'))
        az = kr.eval_ring_thin_mask(pts, dict(CFG, MODE='pattern_az'), fs.frame_rng(fid, 'RING_THIN'))
        np.testing.assert_array_equal(old, rows & az)
        np.testing.assert_array_equal(got['pattern_rows'], pts[rows])
        np.testing.assert_array_equal(got['pattern_az'], pts[az])
        assert len(got['pattern']) < len(got['pattern_rows']) < len(pts) and len(got['pattern']) < len(got['pattern_az']) < len(pts)
        n += 1
    assert n == 40
