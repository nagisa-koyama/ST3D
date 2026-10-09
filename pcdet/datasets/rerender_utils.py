"""Spec-conditioned re-rendering onto a target sensor's lattice (experiments_md 20261009_01).

The TARGET's published lattice - beam inclinations, columns per revolution, mount height - is placed at the SOURCE
ego pose as a virtual sensor. The accumulated source becomes that sensor's range image: every point goes to the cell
of its nearest inclination and its azimuth column, and each cell keeps its NEAREST return (z-buffer). Empty cells
between two filled cells of the same column that lie on one continuous surface are filled by linear interpolation in
row index (no extrapolation past an edge). Every filled cell is back-projected at its lattice ray. Target point clouds
and labels are never read: the lattice is the target's published spec.

Coordinates: the source ego frame with SHIFT_COOR applied (ground at z ~ 0); the virtual sensor sits at (0, 0, h).
Rows are indexed in ASCENDING inclination inside this module.
"""
import functools
import pickle

import numpy as np

WAYMO_TOP_CALIB = '/home/koyama/data/waymo_top_calib/top_calib.pkl'


@functools.lru_cache(maxsize=4)
def _waymo_top_median(path=WAYMO_TOP_CALIB):
    d = pickle.load(open(path, 'rb'))
    inc = np.array([np.sort(np.asarray(v['inclinations'], np.float64).ravel()) for v in d.values()])
    return np.median(inc, 0)


def lattice_spec(name, mount_height=None):
    """Target lattice: ascending inclinations (rad), columns per revolution, mount height above ground (m).

    waymo_top: the element-wise median of the TOP inclination vectors of all segments in WAYMO_TOP_CALIB (1,000
    segments, 105 distinct vectors, all within 0.38 deg of the median), 2,650 columns, 2.184 m.
    hdl32e: nuScenes' HDL-32E, 32 rows uniformly from -30.67 to +10.67 deg, 0.332-deg columns (1,084), 1.84 m."""
    if name == 'waymo_top':
        thetas, n_cols, h = _waymo_top_median(), 2650, 2.184
    elif name == 'hdl32e':
        thetas, n_cols, h = np.radians(np.linspace(-30.67, 10.67, 32)), int(round(360.0 / 0.332)), 1.84
    else:
        raise ValueError(f'unknown TARGET_LATTICE {name!r} (waymo_top | hdl32e)')
    return dict(thetas=np.asarray(thetas, np.float64), n_cols=int(n_cols),
                height=float(h if mount_height is None else mount_height))


def to_lattice(xyz, height, thetas, n_cols):
    """(row, col, range, valid) of every point about the virtual origin (0, 0, height). Row = nearest inclination
    (ascending index); points outside the vertical field of view (beyond half a row gap past either end) are invalid."""
    d = np.asarray(xyz, np.float64)[:, :3] - np.array([0.0, 0.0, height])
    rp = np.hypot(d[:, 0], d[:, 1]); rng = np.hypot(rp, d[:, 2])
    el = np.arctan2(d[:, 2], np.maximum(rp, 1e-6)); az = np.arctan2(d[:, 1], d[:, 0])
    n_rows = len(thetas)
    if n_rows == 1:
        row = np.zeros(len(d), np.int64); valid = np.ones(len(d), bool)
    else:
        k = np.clip(np.searchsorted(thetas, el), 1, n_rows - 1)
        row = np.where(np.abs(el - thetas[k - 1]) <= np.abs(el - thetas[k]), k - 1, k)
        lo = thetas[0] - (thetas[1] - thetas[0]) / 2; hi = thetas[-1] + (thetas[-1] - thetas[-2]) / 2
        valid = (el >= lo) & (el <= hi)
    col = np.clip(np.floor((az + np.pi) / (2 * np.pi) * n_cols).astype(np.int64), 0, n_cols - 1)
    valid &= rng > 1e-3
    return row, col, rng, valid


def zbuffer(row, col, rng, valid, n_rows, n_cols):
    """Range image (inf = empty) and the index of the point each filled cell keeps (-1 = empty): the NEAREST return."""
    img = np.full((n_rows, n_cols), np.inf); idx = np.full((n_rows, n_cols), -1, np.int64)
    sel = np.nonzero(valid)[0]
    if len(sel):
        cell = row[sel] * n_cols + col[sel]
        o = sel[np.lexsort((rng[sel], cell))]; c = row[o] * n_cols + col[o]
        first = np.ones(len(o), bool); first[1:] = c[1:] != c[:-1]
        keep = o[first]
        img.flat[row[keep] * n_cols + col[keep]] = rng[keep]; idx.flat[row[keep] * n_cols + col[keep]] = keep
    return img, idx


def _neighbours(filled):
    """Per cell, the nearest filled row strictly below (-1 if none) and strictly above (n_rows if none), per column."""
    n_rows = filled.shape[0]; r = np.arange(n_rows)[:, None]
    below = np.where(filled, r, -1); below = np.maximum.accumulate(below, axis=0)
    below = np.vstack([np.full((1, filled.shape[1]), -1), below[:-1]])
    above = np.where(filled, r, n_rows); above = np.minimum.accumulate(above[::-1], axis=0)[::-1]
    above = np.vstack([above[1:], np.full((1, filled.shape[1]), n_rows)])
    return below, above


def fill_continuous(img, k_rows=4, dr_max=0.3, thetas=None, span_max_deg=None, h_fill=False):
    """Fill empty cells lying on one continuous surface; returns (filled image, which cells were filled, and for each
    filled cell the row of its lower neighbour - the source of its non-xyz attributes).

    Declared rule (20261009_01 §2.3): an empty cell is filled when the nearest filled cells BELOW and ABOVE it in the
    same column both exist, each within k_rows rows, and their ranges agree within dr_max; the range is linear in row
    index between them. Variant: span_max_deg instead of k_rows - both neighbours exist and their inclinations differ
    by at most span_max_deg (needs thetas). h_fill: afterwards, a still-empty cell whose left and right neighbours
    (same row, one column away, wrapping) are filled with ranges within dr_max gets their mean (off by default)."""
    n_rows, n_cols = img.shape
    filled = np.isfinite(img)
    below, above = _neighbours(filled)
    r = np.arange(n_rows)[:, None]
    ok = ~filled & (below >= 0) & (above < n_rows)
    bl = np.clip(below, 0, n_rows - 1); ab = np.clip(above, 0, n_rows - 1)
    cols = np.broadcast_to(np.arange(n_cols), img.shape)
    rb = img[bl, cols]; ra = img[ab, cols]
    if span_max_deg is None:
        ok &= (r - below <= k_rows) & (above - r <= k_rows)
    else:
        ok &= (thetas[ab] - thetas[bl]) <= np.radians(span_max_deg)
    with np.errstate(invalid='ignore'):
        ok &= np.abs(ra - rb) <= dr_max
    out = img.copy(); src = np.full(img.shape, -1, np.int64)
    if ok.any():
        w = ((r - below) / np.maximum(above - below, 1))[ok]
        out[ok] = rb[ok] + w * (ra[ok] - rb[ok])
        src[ok] = bl[ok]
    new = ok.copy()
    if h_fill:
        f2 = np.isfinite(out); left = np.roll(out, 1, axis=1); right = np.roll(out, -1, axis=1)
        okh = ~f2 & np.isfinite(left) & np.isfinite(right)
        with np.errstate(invalid='ignore'):
            okh &= np.abs(left - right) <= dr_max
        out[okh] = ((left + right) / 2)[okh]; src[okh] = r.repeat(n_cols, 1)[okh]; new |= okh
    return out, new, src


def from_lattice(img, height, thetas, n_cols):
    """xyz of every filled cell, back-projected at the cell's ray (inclination thetas[row], column centre azimuth);
    returns (xyz, rows, cols)."""
    rows, cols = np.nonzero(np.isfinite(img))
    rr = img[rows, cols]; th = thetas[rows]; az = (cols + 0.5) / n_cols * 2 * np.pi - np.pi
    xyz = np.stack([rr * np.cos(th) * np.cos(az), rr * np.cos(th) * np.sin(az), height + rr * np.sin(th)], 1)
    return xyz, rows, cols


def rerender(points, spec, fill='rule', k_rows=4, dr_max=0.3, span_max_deg=None, h_fill=False):
    """Re-render an (N, C) cloud onto the target lattice. Columns beyond xyz are carried from the kept point (a z-buffer
    cell) or from the lower neighbour's kept point (a filled cell). Returns (points (M, C), filled mask (M,))."""
    points = np.asarray(points)
    if len(points) == 0:
        return points[:0].copy(), np.zeros(0, bool)
    thetas, n_cols, h = spec['thetas'], spec['n_cols'], spec['height']
    row, col, rng, valid = to_lattice(points[:, :3], h, thetas, n_cols)
    img, idx = zbuffer(row, col, rng, valid, len(thetas), n_cols)
    if fill == 'rule':
        img2, new, src = fill_continuous(img, k_rows, dr_max, thetas, span_max_deg, h_fill)
    elif fill == 'none':
        img2, new, src = img, np.zeros(img.shape, bool), np.full(img.shape, -1, np.int64)
    else:
        raise ValueError(f'unknown FILL {fill!r} (rule | none)')
    xyz, rows, cols = from_lattice(img2, h, thetas, n_cols)
    is_new = new[rows, cols]
    pidx = idx[rows, cols]
    pidx[is_new] = idx[src[rows, cols][is_new], cols[is_new]]
    out = points[np.maximum(pidx, 0)].copy()
    out[:, :3] = xyz.astype(points.dtype)
    return out, is_new
