"""`DataProcessor.sample_points_by_voxels`: IA-SSD's published Waymo point budget, ported 2026-10-13 (experiments_md
20261011_07 §13). Voxelise, keep one point per occupied voxel, then `sample_points` to NUM_POINTS.

Sentinel cases: an empty cloud, a cloud with fewer occupied voxels than the budget (padding), every point in one voxel,
points outside the range, the voxel cap, the 'mean_vfe' variant, NUM_POINTS -1, determinism under a seed, and pickling
(DDP spawns the loader workers, so the processor must pickle after its generator exists).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402

PCR = np.array([-10.0, -10.0, -2.0, 10.0, 10.0, 4.0], dtype=np.float32)
VS = [0.5, 0.5, 0.5]


def _proc(budget=64, cap=(10000, 10000), sample_type='raw', training=True):
    cfg = EasyDict(NAME='sample_points_by_voxels', SAMPLE_TYPE=sample_type, VOXEL_SIZE=VS, MAX_POINTS_PER_VOXEL=5,
                   MAX_NUMBER_OF_VOXELS={'train': cap[0], 'test': cap[1]}, NUM_POINTS={'train': budget, 'test': budget})
    return DataProcessor([cfg], point_cloud_range=PCR, training=training, num_point_features=3)


def _voxel_ids(p):
    return [tuple(v) for v in np.floor((p[:, :3] - PCR[:3]) / np.array(VS)).astype(int)]


def _cloud(n, seed=0, spread=9.0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(-spread, spread, n), rng.uniform(-spread, spread, n), rng.uniform(-1.5, 3.5, n)], 1).astype(np.float32)


def test_one_point_per_voxel_then_the_budget():
    pts = _cloud(5000)
    np.random.seed(0)
    out = _proc(budget=256).forward({'points': pts.copy()})['points']
    assert out.shape == (256, 3)
    ids = _voxel_ids(out)
    assert len(set(ids)) == len(ids)                         # at most one point per voxel
    src = {tuple(p) for p in pts}
    assert all(tuple(p) in src for p in out)                 # 'raw' keeps original points


def test_fewer_voxels_than_budget_pads_with_duplicates():
    pts = _cloud(40, seed=1)
    n_vox = len(set(_voxel_ids(pts)))
    np.random.seed(0)
    out = _proc(budget=128).forward({'points': pts.copy()})['points']
    assert out.shape == (128, 3)
    assert len(np.unique(out, axis=0)) == n_vox               # every occupied voxel once, the rest duplicates


def test_all_points_in_one_voxel():
    pts = (np.array([[1.1, 1.1, 1.1]], np.float32) + np.random.default_rng(2).uniform(0, 0.3, (50, 3))).astype(np.float32)
    assert len(set(_voxel_ids(pts))) == 1
    np.random.seed(0)
    out = _proc(budget=16).forward({'points': pts.copy()})['points']
    assert out.shape == (16, 3) and len(np.unique(out, axis=0)) == 1


def test_empty_cloud_and_all_out_of_range_stay_empty():
    out = _proc().forward({'points': np.zeros((0, 3), np.float32)})['points']
    assert out.shape == (0, 3)
    far = np.full((20, 3), 50.0, np.float32)                  # outside the range: no voxel
    out = _proc().forward({'points': far})['points']
    assert out.shape[0] == 0


def test_points_outside_the_range_are_never_kept():
    pts = np.concatenate([_cloud(500, seed=3), np.full((50, 3), 30.0, np.float32)])
    np.random.seed(0)
    out = _proc(budget=64).forward({'points': pts.copy()})['points']
    assert np.all((out[:, :3] >= PCR[:3]) & (out[:, :3] < PCR[3:]))


def test_voxel_cap_limits_distinct_points():
    pts = _cloud(5000, seed=4)
    np.random.seed(0)
    out = _proc(budget=500, cap=(100, 100)).forward({'points': pts.copy()})['points']
    assert out.shape == (500, 3) and len(np.unique(out, axis=0)) <= 100


def test_mean_vfe_keeps_the_voxel_mean():
    a = np.array([[1.1, 1.1, 1.1], [1.3, 1.3, 1.3]], np.float32)  # one voxel
    out = _proc(budget=1, sample_type='mean_vfe').forward({'points': a.copy()})['points']
    assert np.allclose(out, [[1.2, 1.2, 1.2]], atol=1e-6)


def test_budget_minus_one_is_a_no_op():
    pts = _cloud(100, seed=5)
    out = _proc(budget=-1).forward({'points': pts.copy()})['points']
    assert np.array_equal(out, pts)


def test_deterministic_under_a_seed():
    pts = _cloud(3000, seed=6)
    outs = []
    for _ in range(2):
        np.random.seed(123)
        outs.append(_proc(budget=300).forward({'points': pts.copy()})['points'])
    assert np.array_equal(outs[0], outs[1])


def test_pickles_after_the_generator_exists():
    proc = _proc(budget=32)
    np.random.seed(0)
    proc.forward({'points': _cloud(200, seed=7)})
    assert proc._sample_voxel_generator is not None
    clone = pickle.loads(pickle.dumps(proc))
    assert clone._sample_voxel_generator is None
    np.random.seed(0)
    assert clone.forward({'points': _cloud(200, seed=7)})['points'].shape == (32, 3)


def test_mode_specific_voxel_cap():
    pts = _cloud(5000, seed=8)
    np.random.seed(0)
    out = _proc(budget=500, cap=(1000, 50), training=False).forward({'points': pts.copy()})['points']
    assert len(np.unique(out, axis=0)) <= 50                  # the 'test' cap applies in evaluation mode
