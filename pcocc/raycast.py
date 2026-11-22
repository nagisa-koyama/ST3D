"""Measured-ray traversal (experiments_md 20261121_02 §3 "visibility").

Each ray runs from its sensor's origin at the point's capture time to the point. Every voxel the segment passes
through is TRAVERSED, except the voxel holding the point itself, which the point occupies (or, for a mover point that
is moved to its frame-t box, which is left to other rays). Exact voxel traversal (Amanatides & Woo 1987), with the
segment first clipped to the grid by the slab method, so rays that miss the grid cost one test.
"""
import numpy as np
import numba as nb


@nb.njit(cache=False, inline='always')
def _clamp(i, n):
    if i < 0:
        return 0
    if i >= n:
        return n - 1
    return i


@nb.njit(cache=False)
def _traverse_one(o0, o1, o2, e0, e1, e2, nx, ny, nz, out):
    """o, e in GRID units ((p - lo) / voxel). Marks out[i, j, k] = 1 on every traversed voxel; returns their number."""
    d0 = e0 - o0
    d1 = e1 - o1
    d2 = e2 - o2
    s_in = 0.0
    s_out = 1.0
    # slab clip against [0, n) on each axis
    if d0 == 0.0:
        if o0 < 0.0 or o0 >= nx:
            return 0
    else:
        a = (0.0 - o0) / d0
        b = (nx - o0) / d0
        if a > b:
            a, b = b, a
        s_in = max(s_in, a)
        s_out = min(s_out, b)
    if d1 == 0.0:
        if o1 < 0.0 or o1 >= ny:
            return 0
    else:
        a = (0.0 - o1) / d1
        b = (ny - o1) / d1
        if a > b:
            a, b = b, a
        s_in = max(s_in, a)
        s_out = min(s_out, b)
    if d2 == 0.0:
        if o2 < 0.0 or o2 >= nz:
            return 0
    else:
        a = (0.0 - o2) / d2
        b = (nz - o2) / d2
        if a > b:
            a, b = b, a
        s_in = max(s_in, a)
        s_out = min(s_out, b)
    if s_in >= s_out:
        return 0
    p0 = o0 + s_in * d0
    p1 = o1 + s_in * d1
    p2 = o2 + s_in * d2
    i = _clamp(int(np.floor(p0)), nx)
    j = _clamp(int(np.floor(p1)), ny)
    k = _clamp(int(np.floor(p2)), nz)
    ei = int(np.floor(e0))
    ej = int(np.floor(e1))
    ek = int(np.floor(e2))
    inf = np.inf
    if d0 > 0.0:
        si, ti, dti = 1, s_in + (i + 1 - p0) / d0, 1.0 / d0
    elif d0 < 0.0:
        si, ti, dti = -1, s_in + (p0 - i) / (-d0), -1.0 / d0
    else:
        si, ti, dti = 0, inf, inf
    if d1 > 0.0:
        sj, tj, dtj = 1, s_in + (j + 1 - p1) / d1, 1.0 / d1
    elif d1 < 0.0:
        sj, tj, dtj = -1, s_in + (p1 - j) / (-d1), -1.0 / d1
    else:
        sj, tj, dtj = 0, inf, inf
    if d2 > 0.0:
        sk, tk, dtk = 1, s_in + (k + 1 - p2) / d2, 1.0 / d2
    elif d2 < 0.0:
        sk, tk, dtk = -1, s_in + (p2 - k) / (-d2), -1.0 / d2
    else:
        sk, tk, dtk = 0, inf, inf
    count = 0
    while True:
        if i == ei and j == ej and k == ek:
            break
        out[i, j, k] = 1
        count += 1
        if ti < tj:
            if ti < tk:
                if ti >= s_out:
                    break
                i += si
                ti += dti
                if i < 0 or i >= nx:
                    break
            else:
                if tk >= s_out:
                    break
                k += sk
                tk += dtk
                if k < 0 or k >= nz:
                    break
        else:
            if tj < tk:
                if tj >= s_out:
                    break
                j += sj
                tj += dtj
                if j < 0 or j >= ny:
                    break
            else:
                if tk >= s_out:
                    break
                k += sk
                tk += dtk
                if k < 0 or k >= nz:
                    break
    return count


@nb.njit(cache=False, parallel=True)
def _traverse_all(o, e, lo, inv_voxel, nx, ny, nz, out):
    n = o.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for r in nb.prange(n):
        counts[r] = _traverse_one((o[r, 0] - lo[0]) * inv_voxel, (o[r, 1] - lo[1]) * inv_voxel,
                                  (o[r, 2] - lo[2]) * inv_voxel, (e[r, 0] - lo[0]) * inv_voxel,
                                  (e[r, 1] - lo[1]) * inv_voxel, (e[r, 2] - lo[2]) * inv_voxel, nx, ny, nz, out)
    return counts.sum(), (counts > 0).sum()


def traverse(origins, ends, grid, out=None):
    """origins, ends [N, 3] metres in the grid's frame. Returns (traversed [X, Y, Z] bool, stats).

    Pass `out` (uint8 [X, Y, Z]) to accumulate into an existing mask."""
    origins = np.ascontiguousarray(origins, dtype=np.float64)
    ends = np.ascontiguousarray(ends, dtype=np.float64)
    assert origins.shape == ends.shape and origins.ndim == 2 and origins.shape[1] == 3
    if out is None:
        out = np.zeros(grid.shape, dtype=np.uint8)
    assert out.dtype == np.uint8 and out.shape == grid.shape
    nx, ny, nz = grid.shape
    visits, rays_in = _traverse_all(origins, ends, grid.lo, 1.0 / grid.voxel, nx, ny, nz, out)
    return out, {'rays': int(len(origins)), 'rays_in_grid': int(rays_in), 'voxel_visits': int(visits)}
