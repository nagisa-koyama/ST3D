"""Index-returning z-buffer re-rendering for pcseg (S3b prototype; experiments_md episodes/20261116_01 §8.9).

pcseg's operator rule (§4): an operator returns a mask, an index or a transform, never a new cloud without provenance.
A re-render makes NEW points, so it returns, for every output point, the index of the input point it came from (the
z-buffer winner, `src`), or -1 for a point the renderer made up (a filled cell). Labels follow by `carry_labels`:
a winner's label, -1 for a filled point, and an input label of -1 (pcseg.IGNORE) stays -1.

Frames. PandaSet points are stored in WORLD coordinates, motion-compensated point by point: each point sits where its
own firing put it, with the pose AT ITS TIME (measured: per-point pose interpolation sharpens the Pandar64 rings from
0.32 to 0.056 deg median absolute deviation at 0-10 m on a frame at 18 m/s, and gives one PandarGT origin at every
speed; tools/analysis/pcseg_render_sensor_model.py). So a ray's origin is the sensor at the ray's firing time, not at
the frame's pose time. `PoseTrack` gives the normative ego frame (x forward, y left, z up, no SHIFT_COOR) at a frame's
pose time - identical to pcseg.pandaset_seg.read_frame - or at any time (position linear, rotation slerp).

A sensor is (origin, rotation) in that ego frame plus a lattice of rays in the SENSOR frame:
- `RingLattice`: a spinning sensor, rows = an elevation table, columns = azimuth bins (Pandar64);
- `TemplateLattice`: rays measured from data, one direction each (PandarGT, whose lines are curved and come in three
  overlapping sectors).
"""
import json
import os

import numpy as np

IGNORE = -1
# PandaSet ego axes (x right, y forward, z up) -> normative (x forward, y left, z up); the same swap as
# pcseg.pandaset_seg._ego_from_world.
SWAP = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


# --- poses ----------------------------------------------------------------------------------------------------------

class PoseTrack:
    """A sequence's lidar poses (lidar/poses.json) with their times (lidar/timestamps.json)."""

    def __init__(self, root, seq):
        from scipy.spatial.transform import Rotation
        with open(os.path.join(root, seq, 'lidar', 'poses.json')) as f:
            poses = json.load(f)
        with open(os.path.join(root, seq, 'lidar', 'timestamps.json')) as f:
            self.times = np.asarray(json.load(f), dtype=np.float64)
        self.rot = Rotation.from_quat([[p['heading'][k] for k in 'xyzw'] for p in poses])
        self.pos = np.array([[p['position'][k] for k in 'xyz'] for p in poses], dtype=np.float64)

    def _at(self, times):
        from scipy.spatial.transform import Slerp
        t = np.clip(np.asarray(times, dtype=np.float64), self.times[0], self.times[-1])
        rot = Slerp(self.times, self.rot)(t)
        pos = np.stack([np.interp(t, self.times, self.pos[:, k]) for k in range(3)], axis=1)
        return rot, pos

    def ego_static(self, xyz_world, frame):
        """World -> normative ego with the frame's pose (what read_frame returns)."""
        q = self.rot[frame].inv().apply(np.asarray(xyz_world, dtype=np.float64) - self.pos[frame])
        return q @ SWAP.T

    def ego_at(self, xyz_world, times):
        """World -> normative ego with the pose at each point's time (one time per point, or one for all)."""
        xyz_world = np.asarray(xyz_world, dtype=np.float64)
        times = np.broadcast_to(np.asarray(times, dtype=np.float64), (len(xyz_world),))
        rot, pos = self._at(times)
        return rot.inv().apply(xyz_world - pos) @ SWAP.T

    def world_from_ego_at(self, xyz_ego, times):
        xyz_ego = np.asarray(xyz_ego, dtype=np.float64)
        times = np.broadcast_to(np.asarray(times, dtype=np.float64), (len(xyz_ego),))
        rot, pos = self._at(times)
        return rot.apply(xyz_ego @ SWAP) + pos


# --- sensor geometry ------------------------------------------------------------------------------------------------

def to_sensor(xyz_ego, origin, rot):
    """Ego -> sensor frame: q = rot^T (p - origin); `rot` maps sensor axes into the ego frame."""
    return (np.asarray(xyz_ego, dtype=np.float64) - np.asarray(origin, dtype=np.float64)) @ np.asarray(rot)


def from_sensor(q, origin, rot):
    return np.asarray(q, dtype=np.float64) @ np.asarray(rot).T + np.asarray(origin, dtype=np.float64)


def angles_deg(q):
    """(azimuth deg, elevation deg, range m) of sensor-frame vectors (x forward, y left, z up)."""
    q = np.asarray(q, dtype=np.float64)
    rp = np.hypot(q[:, 0], q[:, 1])
    return np.degrees(np.arctan2(q[:, 1], q[:, 0])), np.degrees(np.arctan2(q[:, 2], rp)), np.linalg.norm(q, axis=1)


def unit_from_angles(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=1)


# --- lattices -------------------------------------------------------------------------------------------------------

class RingLattice:
    """Rows: an elevation table (deg, sensor frame); columns: azimuth bins of `az_res_deg` starting at `az_phase_deg`.
    Optional per-row azimuth offsets `row_daz_deg` (a laser's column inside a firing block) are subtracted before
    binning. A point is invalid beyond half a row gap past either end of the table and, with `el_tol_deg`, farther
    than that from its row's elevation (a beam footprint; without it a cell spans half the gap to each neighbouring
    row, up to 3 deg at the Pandar64's bottom rows). Cell id = row * n_cols + col."""

    def __init__(self, el_deg, az_res_deg, az_phase_deg=0.0, row_daz_deg=None, el_tol_deg=None):
        el = np.asarray(el_deg, dtype=np.float64)
        self.order = np.argsort(el)                     # table index of each ascending row
        self.el = el[self.order]
        self.daz = (np.zeros(len(el)) if row_daz_deg is None else np.asarray(row_daz_deg, np.float64))[self.order]
        self.az_res = float(az_res_deg)
        self.phase = float(az_phase_deg)
        self.n_rows = len(el)
        self.n_cols = int(round(360.0 / self.az_res))
        self.n_cells = self.n_rows * self.n_cols
        self.el_tol = None if el_tol_deg is None else float(el_tol_deg)

    def assign(self, q):
        """(cell id, -1 = outside; range) of sensor-frame points."""
        az, el, rng = angles_deg(q)
        t = self.el
        k = np.clip(np.searchsorted(t, el), 1, self.n_rows - 1)
        row = np.where(np.abs(el - t[k - 1]) <= np.abs(el - t[k]), k - 1, k)
        lo, hi = t[0] - (t[1] - t[0]) / 2, t[-1] + (t[-1] - t[-2]) / 2
        valid = (el >= lo) & (el <= hi) & (rng > 1e-3)
        if self.el_tol is not None:
            valid &= np.abs(el - t[row]) <= self.el_tol
        col = np.floor(((az - self.daz[row] - self.phase + 180.0) % 360.0) / self.az_res).astype(np.int64)
        col = np.minimum(col, self.n_cols - 1)
        cell = np.where(valid, row * self.n_cols + col, -1)
        return cell, rng

    def rays(self, cell):
        """Unit ray of each cell: the row's elevation, the bin centre's azimuth (plus the row's offset)."""
        cell = np.asarray(cell, dtype=np.int64)
        row, col = cell // self.n_cols, cell % self.n_cols
        az = (col + 0.5) * self.az_res - 180.0 + self.phase + self.daz[row]
        return unit_from_angles(az, self.el[row])

    def table_row(self, cell):
        """Index into the ORIGINAL elevation table (e.g. the Pandar64 channel) of each cell."""
        return self.order[np.asarray(cell, dtype=np.int64) // self.n_cols]

    def fill_vertical(self, cells, rng, k_rows=4, dr_max=0.3):
        """The declared continuity rule of rerender_utils.fill_continuous (20261009_01 §2.3), without h_fill: an empty
        cell is filled when the nearest occupied cells below and above it in its column both exist within k_rows rows
        and their ranges agree within dr_max; the range is linear in row index. Returns (new cells, their ranges)."""
        img = np.full((self.n_rows, self.n_cols), np.inf)
        img.flat[np.asarray(cells, dtype=np.int64)] = rng
        filled = np.isfinite(img)
        r = np.arange(self.n_rows)[:, None]
        below = np.maximum.accumulate(np.where(filled, r, -1), axis=0)
        below = np.vstack([np.full((1, self.n_cols), -1), below[:-1]])
        above = np.minimum.accumulate(np.where(filled, r, self.n_rows)[::-1], axis=0)[::-1]
        above = np.vstack([above[1:], np.full((1, self.n_cols), self.n_rows)])
        ok = ~filled & (below >= 0) & (above < self.n_rows) & (r - below <= k_rows) & (above - r <= k_rows)
        cols = np.broadcast_to(np.arange(self.n_cols), img.shape)
        rb = img[np.clip(below, 0, self.n_rows - 1), cols]
        ra = img[np.clip(above, 0, self.n_rows - 1), cols]
        with np.errstate(invalid='ignore'):
            ok &= np.abs(ra - rb) <= dr_max
        w = (r - below) / np.maximum(above - below, 1)
        new = np.flatnonzero(ok.ravel())
        with np.errstate(invalid='ignore'):                # inf - inf in cells that are not filled
            val = (rb + w * (ra - rb)).ravel()[new]
        return new.astype(np.int64), val


class TemplateLattice:
    """Rays measured from data: (K, 2) azimuth, elevation in deg (sensor frame). A point goes to its nearest ray in
    the local tangent plane (azimuth scaled by cos elevation) if that ray is within `tol_deg`, else it is invalid.
    Cell id = ray index."""

    def __init__(self, az_el_deg, tol_deg):
        from scipy.spatial import cKDTree
        self.az_el = np.asarray(az_el_deg, dtype=np.float64)
        self.tol = float(tol_deg)
        self.n_cells = len(self.az_el)
        self._tree = cKDTree(self._plane(self.az_el[:, 0], self.az_el[:, 1]))

    @staticmethod
    def _plane(az, el):
        return np.stack([az * np.cos(np.radians(el)), el], axis=1)

    def nearest(self, az, el):
        d, i = self._tree.query(self._plane(az, el))
        return i.astype(np.int64), d

    def assign(self, q):
        az, el, rng = angles_deg(q)
        i, d = self.nearest(az, el)
        return np.where((d <= self.tol) & (rng > 1e-3), i, -1), rng

    def rays(self, cell):
        a = self.az_el[np.asarray(cell, dtype=np.int64)]
        return unit_from_angles(a[:, 0], a[:, 1])


# --- the z-buffer ---------------------------------------------------------------------------------------------------

def zbuffer(cell, rng):
    """Occupied cells (ascending) and, for each, the index of the input point it keeps: the NEAREST one; equal ranges
    go to the lower index. Points with cell < 0 never win."""
    cell = np.asarray(cell, dtype=np.int64)
    rng = np.asarray(rng, dtype=np.float64)
    sel = np.flatnonzero(cell >= 0)
    if len(sel) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    o = sel[np.lexsort((sel, rng[sel], cell[sel]))]
    first = np.ones(len(o), dtype=bool)
    first[1:] = cell[o[1:]] != cell[o[:-1]]
    win = o[first]
    return cell[win], win


def render(q, lattice, fill=False, **fill_kw):
    """Re-render sensor-frame points q (N, 3) on `lattice`.

    Returns a dict:
      cell [M]  lattice cell of each output point;
      src  [M]  index into q of the point it came from (the z-buffer winner), -1 for a filled point;
      rng  [M]  range (m);
      q    [M, 3] the output points in the sensor frame, on the cell's ray at that range.
    `fill=True` (RingLattice only) adds the vertical continuity fill; filled points carry src = -1."""
    cell, rng = lattice.assign(q)
    return render_cells(cell, rng, lattice, fill, **fill_kw)


def render_cells(cell, rng, lattice, fill=False, **fill_kw):
    """`render` from cells already assigned (e.g. by `assign_time_aware`); `rng` is each input point's range."""
    cell = np.asarray(cell, dtype=np.int64)
    rng = np.asarray(rng, dtype=np.float64)
    cells, win = zbuffer(cell, rng)
    out_cell, out_src, out_rng = cells, win, rng[win]
    if fill:
        new, new_rng = lattice.fill_vertical(cells, rng[win], **fill_kw)
        out_cell = np.concatenate([cells, new])
        out_src = np.concatenate([win, np.full(len(new), -1, np.int64)])
        out_rng = np.concatenate([rng[win], new_rng])
    q_out = lattice.rays(out_cell) * out_rng[:, None] if len(out_cell) else np.zeros((0, 3))
    return {'cell': out_cell, 'src': out_src, 'rng': out_rng, 'q': q_out}


def carry_labels(labels, src):
    """Label of every rendered point: the source point's label, IGNORE (-1) for a filled point (src < 0). A source
    label of -1 stays -1."""
    labels = np.asarray(labels)
    src = np.asarray(src, dtype=np.int64)
    out = np.full(len(src), IGNORE, dtype=labels.dtype if labels.dtype.kind == 'i' else np.int64)
    ok = src >= 0
    out[ok] = labels[src[ok]]
    return out


# --- time-aware rendering of world points ---------------------------------------------------------------------------

def sensor_coords_at(track, xyz_world, times, origin, rot):
    return to_sensor(track.ego_at(xyz_world, times), origin, rot)


def assign_time_aware(track, xyz_world, lattice, origin, rot, time_of_direction, t0, n_iter=2):
    """Cell and sensor-frame coordinates of world points as the sensor would see them at the firing time of the ray
    they fall on. `time_of_direction(az_deg, el_deg)` is the frame's firing time per direction (e.g. the time of the
    nearest real return); start from time t0 and iterate n_iter times. Returns (cell, q, times)."""
    times = np.full(len(xyz_world), float(t0))
    q = sensor_coords_at(track, xyz_world, times, origin, rot)
    for _ in range(n_iter):
        az, el, _ = angles_deg(q)
        times = time_of_direction(az, el)
        q = sensor_coords_at(track, xyz_world, times, origin, rot)
    cell, _ = lattice.assign(q)
    return cell, q, times


def nearest_time(az_deg, el_deg, t):
    """A `time_of_direction` for assign_time_aware: the time of the nearest of a frame's real returns (sensor-frame
    azimuth / elevation in deg, times in s), in the tangent plane."""
    from scipy.spatial import cKDTree
    tree = cKDTree(TemplateLattice._plane(np.asarray(az_deg, np.float64), np.asarray(el_deg, np.float64)))
    t = np.asarray(t, dtype=np.float64)

    def time_of_direction(az, el):
        _, i = tree.query(TemplateLattice._plane(az, el))
        return t[i]
    return time_of_direction
