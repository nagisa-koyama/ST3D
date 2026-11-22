"""Moving objects: which points belong to a mover, their box-relative coordinates, and where they go in frame t
(experiments_md 20261121_02 §3 "movers" and §5.1 item 2, refined 2026-11-22).

- A mover is a cuboid `uuid` with `stationary` False (checked constant over a sequence on 097 / 043 / 011 / 003).
- Device d's points use the cuboids with `cuboids.sensor_id` in {-1, d}.
- A cuboid's TIME is the median capture time of the points of its own device(s) inside it in its frame (the frame time
  when it holds none): moving objects in the overlap carry one cuboid per sensor, linked as siblings, about 0.3 m
  apart, so each follows its own sensor's capture.
- A point's box pose is interpolated at the point's own time between the two cuboids of its uuid whose times bracket
  it: position linear, yaw the short way; beyond the ends the end segment extrapolates; one cuboid is held fixed.
  PandarGT cuboids (sensor_id 1) of a STALE frame (census) stay out of the track: their frame's points are a copy of
  the neighbour frame's scan, so they cannot time them.
- Membership and box coordinates use that interpolated pose; the box frame is the devkit's center_box_to_corners
  (yaw about world z; |x| <= dx / 2, |y| <= dy / 2, |z| <= dz / 2).
- Placement at frame t: at the uuid's own frame-t cuboid ('own'), or, for PandarGT points in GU ('p64'), at the
  frame-t cuboid of the uuid's Pandar64 sibling, falling back to its own (counted).
"""
import os

import numpy as np
import pandas as pd

from pcseg.pandaset_seg import N_FRAMES

PANDAR64, PANDARGT = 0, 1


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def to_box(xyz, centre, yaw):
    """World -> box frame, per point (centre [N, 3], yaw [N]) or one box."""
    d = np.asarray(xyz, dtype=np.float64) - centre
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)


def from_box(q, centre, yaw):
    q = np.asarray(q, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1], q[:, 2]], axis=1) + centre


def inside(q, dims):
    return np.all(np.abs(q) <= np.asarray(dims) / 2, axis=1)


class Cuboids:
    """Every cuboid of a sequence, one row each: frame, uid (int code of uuid), sensor, sibling uid (-1 none),
    stationary, centre [3], yaw, dims [3]."""

    def __init__(self, root, seq, frame_times, frames=None):
        frames = range(N_FRAMES) if frames is None else frames
        rows = []
        for f in frames:
            c = pd.read_pickle(os.path.join(root, seq, 'annotations', 'cuboids', '%02d.pkl.gz' % f))
            c = c.assign(frame=f)
            rows.append(c)
        df = pd.concat(rows, ignore_index=True)
        if df.groupby(['frame', 'uuid']).size().max() > 1:
            raise ValueError('%s: a uuid appears twice in one frame' % seq)
        codes, uniq = pd.factorize(df['uuid'])
        self.uuid_names = list(uniq)
        code_of = {u: i for i, u in enumerate(uniq)}
        self.frame = df['frame'].to_numpy(dtype=np.int64)
        self.uid = codes.astype(np.int64)
        self.sensor = df['cuboids.sensor_id'].to_numpy().astype(np.int64)
        sib = df['cuboids.sibling_id'].to_numpy()
        self.sibling = np.array([code_of.get(s, -1) if s != '-' else -1 for s in sib], dtype=np.int64)
        self.sibling_unknown = int(sum(1 for s in sib if s != '-' and s not in code_of))
        self.stationary = df['stationary'].to_numpy().astype(bool)
        self.centre = df[['position.x', 'position.y', 'position.z']].to_numpy(dtype=np.float64)
        self.yaw = df['yaw'].to_numpy(dtype=np.float64)
        self.dims = df[['dimensions.x', 'dimensions.y', 'dimensions.z']].to_numpy(dtype=np.float64)
        self.label = df['label'].to_numpy()
        self.frame_times = np.asarray(frame_times, dtype=np.float64)
        # per-uuid constancy the design relies on
        for name, arr in (('sensor_id', self.sensor), ('stationary', self.stationary), ('sibling', self.sibling)):
            per = pd.Series(arr).groupby(self.uid).nunique()
            if (per > 1).any():
                raise ValueError('%s: %s varies within a uuid for %d uuids' % (seq, name, int((per > 1).sum())))
        self.time = self.frame_times[self.frame].copy()      # replaced by `time_cuboids`
        self.in_track = np.ones(len(self.frame), dtype=bool)
        self.row_at = {(int(f), int(u)): i for i, (f, u) in enumerate(zip(self.frame, self.uid))}

    def __len__(self):
        return len(self.frame)

    def devices(self, row):
        s = self.sensor[row]
        return (PANDAR64, PANDARGT) if s == -1 else (int(s),)


def _by_frame(frame):
    order = np.argsort(frame, kind='stable')
    bounds = np.searchsorted(frame[order], np.arange(N_FRAMES + 1))
    return order, bounds


def _near(xyz, centre, dims, margin):
    r = 0.5 * np.hypot(dims[0], dims[1]) + margin
    return (np.abs(xyz[:, 0] - centre[0]) <= r) & (np.abs(xyz[:, 1] - centre[1]) <= r)


def time_cuboids(cub, scans, usable):
    """Set cub.time from the points inside each mover cuboid (as annotated) and cub.in_track (stale-frame PandarGT
    cuboids out). `usable` [N]: points that may time a cuboid (not stale). Returns diagnostics."""
    order, bounds = _by_frame(scans.frame)
    n_timed = n_both = 0
    for r in np.flatnonzero(~cub.stationary):
        f = cub.frame[r]
        if cub.sensor[r] == PANDARGT and f in scans.stale_frames[PANDARGT]:
            cub.in_track[r] = False
            continue
        idx = order[bounds[f]:bounds[f + 1]]
        idx = idx[usable[idx] & np.isin(scans.dev[idx], cub.devices(r))]
        idx = idx[_near(scans.xyz[idx], cub.centre[r], cub.dims[r], 0.0)]
        q = to_box(scans.xyz[idx], cub.centre[r], cub.yaw[r])
        m = inside(q, cub.dims[r])
        if m.any():
            cub.time[r] = float(np.median(scans.t[idx[m]]))
            n_timed += 1
            if cub.sensor[r] == -1 and len(np.unique(scans.dev[idx[m]])) > 1:
                n_both += 1
    return {'movers_timed_by_points': n_timed, 'shared_movers_with_both_devices': n_both}


class Track:
    """One uuid's box pose over time, from its in-track cuboids sorted by time."""

    def __init__(self, cub, uid):
        rows = np.flatnonzero((cub.uid == uid) & cub.in_track)
        rows = rows[np.argsort(cub.time[rows], kind='stable')]
        self.t = cub.time[rows]
        self.c = cub.centre[rows]
        self.yaw = np.unwrap(cub.yaw[rows]) if len(rows) else cub.yaw[rows]
        # equal times (two cuboids with no points at... never: frame times differ) are kept strictly increasing
        if len(self.t) > 1 and np.any(np.diff(self.t) <= 0):
            raise ValueError('uuid %d: cuboid times not increasing' % uid)

    def at(self, times):
        t = np.asarray(times, dtype=np.float64)
        n = len(self.t)
        if n == 0:
            raise ValueError('empty track')
        if n == 1:
            return np.broadcast_to(self.c[0], (len(t), 3)).copy(), np.full(len(t), self.yaw[0])
        j = np.clip(np.searchsorted(self.t, t, side='right'), 1, n - 1)   # segment [j-1, j]; ends extrapolate
        i = j - 1
        s = (t - self.t[i]) / (self.t[j] - self.t[i])
        c = self.c[i] + s[:, None] * (self.c[j] - self.c[i])
        yaw = self.yaw[i] + s * (self.yaw[j] - self.yaw[i])
        return c, wrap(yaw)


def assign_movers(cub, scans, usable, margin=2.0):
    """Per point: mover uid (-1 none) and box coordinates [N, 3] (NaN for non-movers), using each point's pose
    interpolated at its capture time. `usable`: points that may join a mover (not stale). margin (m): pre-filter
    radius beyond the frame cuboid's half-diagonal (>= 100 ms of motion at 20 m/s)."""
    n = len(scans)
    mover = np.full(n, -1, dtype=np.int64)
    box = np.full((n, 3), np.nan)
    best = np.full(n, np.inf)                     # normalised distance to the box centre, to settle overlaps
    tracks = {}
    order, bounds = _by_frame(scans.frame)
    conflicts = 0
    for r in np.flatnonzero(~cub.stationary):
        u = int(cub.uid[r])
        if u not in tracks:
            tracks[u] = Track(cub, u)
        if len(tracks[u].t) == 0:
            continue
        f = cub.frame[r]
        idx = order[bounds[f]:bounds[f + 1]]
        idx = idx[usable[idx] & np.isin(scans.dev[idx], cub.devices(r))]
        idx = idx[_near(scans.xyz[idx], cub.centre[r], cub.dims[r], margin)]
        if not len(idx):
            continue
        c, yaw = tracks[u].at(scans.t[idx])
        q = to_box(scans.xyz[idx], c, yaw)
        m = inside(q, cub.dims[r])
        if not m.any():
            continue
        idx, q = idx[m], q[m]
        dist = np.max(np.abs(q) / (cub.dims[r] / 2), axis=1)
        taken = mover[idx] >= 0
        conflicts += int(taken.sum())
        win = dist < best[idx]
        mover[idx[win]] = u
        box[idx[win]] = q[win]
        best[idx[win]] = dist[win]
    return mover, box, {'mover_points': int((mover >= 0).sum()), 'mover_overlap_points': conflicts,
                        'movers': len(tracks)}


def placement_rows(cub, frame_t, mode):
    """{uid: cuboid row in frame_t} where each mover uid's points go. mode 'own', or 'p64' (PandarGT-only uuids go to
    their Pandar64 sibling's frame-t cuboid when it exists). Returns (rows, number of 'p64' fallbacks to own)."""
    rows = {}
    fallbacks = 0
    for u in np.unique(cub.uid[~cub.stationary]):
        u = int(u)
        own = cub.row_at.get((frame_t, u))
        if mode == 'p64':
            r0 = np.flatnonzero(cub.uid == u)[0]
            if cub.sensor[r0] == PANDARGT and cub.sibling[r0] >= 0:
                sib = cub.row_at.get((frame_t, int(cub.sibling[r0])))
                if sib is not None:
                    rows[u] = sib
                    continue
                if own is not None:
                    fallbacks += 1
        if own is not None:
            rows[u] = own
    return rows, fallbacks


def place(cub, rows, mover, box):
    """World positions at frame t of the mover points (mover >= 0), and the mask of those placed (uid present)."""
    out = np.full(box.shape, np.nan)
    placed = np.zeros(len(mover), dtype=bool)
    for u, r in rows.items():
        m = mover == u
        if m.any():
            out[m] = from_box(box[m], cub.centre[r], cub.yaw[r])
            placed |= m
    return out, placed
