"""Voxel grid sampling with an index and an inverse map; Pointcept's GridSample (MIT, Copyright (c) 2023 Pointcept,
pointcept/datasets/transform.py at 77f87ab), cut to what pcseg uses: fnv hash, 'train' mode, grid_coord, inverse.

Returns indices only; the caller moves every per-point array (coord, strength, segment) by them, so a label can never be
separated from its point.
"""
import numpy as np


def fnv_hash_vec(arr):
    """FNV64-1A, as Pointcept."""
    assert arr.ndim == 2
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def grid_sample(coord, grid_size, rng):
    """One random representative point per occupied voxel.

    Returns (idx_unique, grid_coord, inverse):
      idx_unique  [M]  indices into coord of the representatives (the sampled points);
      grid_coord  [M, 3] their integer voxel coordinates, the grid's minimum at 0;
      inverse     [N]  for every input point, the row in idx_unique of its voxel's representative.
    rng is a numpy Generator; a frame-keyed rng makes the draw reproducible.
    """
    coord = np.asarray(coord)
    assert coord.ndim == 2 and coord.shape[1] == 3 and len(coord) > 0
    scaled = coord / np.asarray(grid_size)
    grid = np.floor(scaled).astype(np.int64)
    grid -= grid.min(0)
    key = fnv_hash_vec(grid)
    idx_sort = np.argsort(key, kind='stable')
    key_sort = key[idx_sort]
    _, inv_sorted, count = np.unique(key_sort, return_inverse=True, return_counts=True)
    start = np.cumsum(np.insert(count, 0, 0)[:-1])
    idx_select = start + rng.integers(0, count.max(), count.size) % count
    idx_unique = idx_sort[idx_select]
    inverse = np.empty(len(coord), dtype=np.int64)
    inverse[idx_sort] = inv_sorted
    return idx_unique, grid[idx_unique], inverse
