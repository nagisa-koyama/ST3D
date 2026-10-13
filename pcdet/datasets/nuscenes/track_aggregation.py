"""Whole-track object aggregation from SOURCE labels, for the nuScenes -> Waymo line gap (experiments_md 20261010_04,
design (c1); operational definitions pre-declared in its §5.1 on 2026-10-13 23:37 JST).

Every nuScenes Car track (`gt_names == 'car'`, i.e. `vehicle.car`, keyed by the devkit's `instance_token`) is built
from the OTHER keyframes of the anchor's scene: each keyframe's in-box points, in that box's local frame, are placed on
the anchor's box of the same track. Three additions on top of the plain transport:
- ICP: a 4-DoF (x, y, z, yaw) refinement of each keyframe's points onto the anchor keyframe's own in-box points;
- visibility: back-face culling against the anchor box, from the virtual target sensor;
- ray-drop: a per-cell drop rate measured on unlabelled Waymo TRAIN clouds, applied to the aggregated points only.
The detector's renderer (27834's z-buffer, `sample_points_learned`) then resolves occlusion.

Source labels and source clouds only; no target label is read here. Box-local frame: origin at the box centre, x along
the heading, z up; box = (x, y, z, dx, dy, dz, yaw) in the keyframe's own LIDAR_TOP frame.
"""
import pickle
from pathlib import Path

import numpy as np

CAR = 'car'
CACHE_VERSION = 1
# the five faces back-face culling can assign a point to: +x, -x, +y, -y, +z (the bottom is never a visible face)
N_FACES = 5


def to_box_local(xyz, box):
    """Points (N, >=3) in the box's local frame (N, 3), float64."""
    d = np.asarray(xyz, np.float64)[:, :3] - np.asarray(box[:3], np.float64)
    c, s = np.cos(box[6]), np.sin(box[6])
    return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c, d[:, 2]], 1)


def from_box_local(local, box):
    """Box-local points (N, 3) back in the box's frame."""
    c, s = np.cos(box[6]), np.sin(box[6])
    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    return np.stack([x * c - y * s + box[0], x * s + y * c + box[1], z + box[2]], 1)


def inside_local(local, dims):
    """Inside the box (bounds inclusive), for box-local points and dims (dx, dy, dz)."""
    return ((np.abs(local[:, 0]) <= dims[0] / 2) & (np.abs(local[:, 1]) <= dims[1] / 2)
            & (np.abs(local[:, 2]) <= dims[2] / 2))


def keyframe_tracks(points, boxes, names, tids):
    """In-box points of every Car track of one keyframe: {tid: (local xyz (n, 3) float64, attrs (n, k))}.

    `points` (N, 3 + k) in the keyframe's LIDAR_TOP frame (ego points already removed); `boxes` (M, >=7) and `names`,
    `tids` (M,) from the info / devkit. Only `names == 'car'` with a track id count; anything else (other classes, a box
    with no instance token) is never a track. A point is claimed by at most one track, the first in sorted tid order,
    so overlapping boxes cannot share a point."""
    out = {}
    if points is None or len(points) == 0 or boxes is None or len(boxes) == 0:
        return out
    pts = np.asarray(points)
    claimed = np.zeros(len(pts), bool)
    cars = sorted((str(t), i) for i, (n, t) in enumerate(zip(names, tids)) if n == CAR and t is not None)
    for tid, i in cars:
        box = np.asarray(boxes[i], np.float64)[:7]
        loc = to_box_local(pts, box)
        m = inside_local(loc, box[3:6]) & ~claimed
        if m.any():
            out[tid] = (loc[m], pts[m, 3:])
            claimed |= m
    return out


def _rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def apply_4dof(local, yaw, t):
    """Rz(yaw) @ p + t for box-local points."""
    return local @ _rz(yaw).T + np.asarray(t, np.float64)


def _trunc_error(tree, tgt_in_src, d_cap):
    d, _ = tree.query(tgt_in_src, distance_upper_bound=d_cap)
    return float(np.mean(np.minimum(d, d_cap)))


def icp_4dof(src, tgt, iters=20, d_far=1.0, d_near=0.5, n_far=5, min_pairs=5, max_t_xy=0.5, max_t_z=0.3,
             max_yaw_deg=5.0, tol=1e-4):
    """Refine a keyframe's transported points `src` onto the anchor's own points `tgt` (both box-local, (n, 3)).

    Pre-declared (20261010_04 §5.1, item 2): correspondences from each target point to its nearest transformed source
    point, within d_far for the first n_far iterations and d_near afterwards; 4-DoF (yaw about the box's vertical axis,
    t in x, y, z), closed form, at most `iters` iterations. Skipped if fewer than min_pairs target points find a
    correspondence. The result is kept only if |t_xy| <= max_t_xy, |t_z| <= max_t_z, |yaw| <= max_yaw_deg, and the
    mean truncated nearest-neighbour distance of the target points (capped at d_far) did not grow.

    Returns (yaw, t (3,), accepted, n_pairs). A rejected fit returns the identity."""
    from scipy.spatial import cKDTree
    ident = (0.0, np.zeros(3), False, 0)
    src = np.asarray(src, np.float64); tgt = np.asarray(tgt, np.float64)
    if len(src) < 1 or len(tgt) < min_pairs:
        return ident
    tree = cKDTree(src)
    e0 = _trunc_error(tree, tgt, d_far)
    yaw, t = 0.0, np.zeros(3)
    n_pairs = 0
    for it in range(iters):
        dmax = d_far if it < n_far else d_near
        q = (tgt - t) @ _rz(yaw)                     # target points in the source frame: Rz(yaw)^T (p - t)
        dist, idx = tree.query(q, distance_upper_bound=dmax)
        m = np.isfinite(dist)
        n_pairs = int(m.sum())
        if n_pairs < min_pairs:
            if it == 0:
                return ident
            break
        A = src[idx[m]]; B = tgt[m]
        a, b = A.mean(0), B.mean(0)
        Ac, Bc = A[:, :2] - a[:2], B[:, :2] - b[:2]
        new_yaw = float(np.arctan2(np.sum(Ac[:, 0] * Bc[:, 1] - Ac[:, 1] * Bc[:, 0]),
                                   np.sum(Ac[:, 0] * Bc[:, 0] + Ac[:, 1] * Bc[:, 1])))
        new_t = b - _rz(new_yaw) @ a
        done = abs(new_yaw - yaw) < tol and np.all(np.abs(new_t - t) < tol)
        yaw, t = new_yaw, new_t
        if done:
            break
    if (np.hypot(t[0], t[1]) > max_t_xy or abs(t[2]) > max_t_z or abs(np.degrees(yaw)) > max_yaw_deg
            or n_pairs < min_pairs):
        return ident
    if _trunc_error(tree, (tgt - t) @ _rz(yaw), d_far) > e0:
        return ident
    return yaw, t, True, n_pairs


def visible_faces(sensor_local, dims):
    """Which of the five faces (+x, -x, +y, -y, +z) face a sensor at `sensor_local` (box-local): the sensor lies on the
    face's outer side."""
    hx, hy, hz = dims[0] / 2, dims[1] / 2, dims[2] / 2
    s = sensor_local
    return np.array([s[0] > hx, s[0] < -hx, s[1] > hy, s[1] < -hy, s[2] > hz])


def nearest_face(local, dims):
    """Index (0..4) of each box-local point's nearest face among +x, -x, +y, -y, +z (pre-declared item 3)."""
    hx, hy, hz = dims[0] / 2, dims[1] / 2, dims[2] / 2
    d = np.stack([hx - local[:, 0], hx + local[:, 0], hy - local[:, 1], hy + local[:, 1], hz - local[:, 2]], 1)
    return np.argmin(d, 1)


def visible_mask(local, dims, sensor_local):
    """Back-face culling: keep a point only if its nearest face faces the sensor."""
    if len(local) == 0:
        return np.zeros(0, bool)
    return visible_faces(sensor_local, dims)[nearest_face(local, dims)]


def aggregate_anchor(scene, a, sensor_lidar, icp=True, visibility=True, tracks=None, return_tids=False):
    """Aggregated Car points for anchor keyframe `a` of a cached scene, in the anchor's LIDAR_TOP frame.

    Each other keyframe j's in-box points of track k (box-local) get the cached ICP correction for (a, k, j) when
    `icp`, are culled against the anchor box when `visibility` (virtual sensor at `sensor_lidar`, anchor lidar frame),
    and are placed on the anchor's box of k. Returns (N, 3 + k) = xyz + the stored attributes (intensity, ring), and,
    with return_tids, the track id of each point. `tracks` limits the tracks used (None = every Car track at the
    anchor)."""
    fa = scene['frames'][a]
    out, ids = [], []
    icp_tab = scene.get('icp', {}) if icp else {}
    for k, box in zip(fa['tids'], fa['boxes']):
        if tracks is not None and k not in tracks:
            continue
        box = np.asarray(box, np.float64)
        s_loc = to_box_local(np.asarray(sensor_lidar, np.float64)[None], box)[0]
        corr = icp_tab.get((a, k))
        corr = {int(r[0]): r[1:] for r in corr} if corr is not None else {}
        for j, fj in enumerate(scene['frames']):
            if j == a or k not in fj['points']:
                continue
            xyz, attr = fj['points'][k]
            loc = np.asarray(xyz, np.float64)
            if j in corr:
                c = corr[j]
                loc = apply_4dof(loc, c[0], c[1:4])
            if visibility:
                m = visible_mask(loc, box[3:6], s_loc)
                loc, attr = loc[m], attr[m]
            if len(loc):
                out.append(np.column_stack([from_box_local(loc, box), np.asarray(attr, np.float64)]))
                ids.extend([k] * len(loc))
    n_attr = 2
    pts = np.concatenate(out) if out else np.zeros((0, 3 + n_attr))
    return (pts, np.array(ids, dtype=object)) if return_tids else pts


def raydrop_keep(xyz, rates, range_edges, elev_split_rad, inclinations, n_cols, sensor_z, rng):
    """Per-cell ray-drop on aggregated points (pre-declared item 4): z-buffer `xyz` alone on the lattice of a virtual
    sensor at (0, 0, sensor_z); each cell holding points is dropped with probability rates[range band of its nearest
    range, elevation band of its row]; every point in a dropped cell is removed. Points outside the lattice's vertical
    field of view are kept (the renderer drops them). rates: (n_range_bands, 2) for [below, at-or-above] the split."""
    from ..processor.point_sampler import lattice_pixels
    keep = np.ones(len(xyz), bool)
    if len(xyz) == 0:
        return keep
    pix, row, rng_ = lattice_pixels(np.asarray(xyz, np.float64), inclinations, n_cols, sensor_z)
    ok = np.nonzero(pix >= 0)[0]
    if not len(ok):
        return keep
    up, inv = np.unique(pix[ok], return_inverse=True)
    rmin = np.full(len(up), np.inf)
    np.minimum.at(rmin, inv, rng_[ok])
    inc_desc = np.sort(np.asarray(inclinations, np.float64))[::-1]       # lattice_pixels rows run top -> bottom
    el = inc_desc[up // n_cols]
    rb = np.clip(np.searchsorted(np.asarray(range_edges, np.float64), rmin, side='right') - 1, 0, len(range_edges) - 2)
    eb = (el >= elev_split_rad).astype(np.int64)
    p = np.asarray(rates, np.float64)[rb, eb]
    drop = rng.random(len(up)) < p
    keep[ok[drop[inv]]] = False
    return keep


def rowrun_keep(xyz, ids, q, inclinations, n_cols, sensor_z, rng):
    """Row-run drop (experiments_md 20261010_04 §5.1, re-gate item 1, pre-declared 2026-10-14 00:34): group the
    aggregated points by (object id, lattice row of a virtual sensor at (0, 0, sensor_z)) and remove each group with
    probability q, independently. Applied to every object, range and sector alike. Points outside the lattice's
    vertical field of view are kept (the renderer drops them)."""
    from ..processor.point_sampler import lattice_pixels
    keep = np.ones(len(xyz), bool)
    if len(xyz) == 0 or q <= 0:
        return keep
    _, row, _ = lattice_pixels(np.asarray(xyz, np.float64), inclinations, n_cols, sensor_z)
    ok = np.nonzero(row >= 0)[0]
    if not len(ok):
        return keep
    _, obj = np.unique(np.asarray(ids)[ok], return_inverse=True)
    groups, inv = np.unique(obj.astype(np.int64) * (len(inclinations) + 1) + row[ok], return_inverse=True)
    drop = rng.random(len(groups)) < q
    keep[ok[drop[inv]]] = False
    return keep


def load_raydrop(path):
    d = np.load(path)
    return dict(rates=d['rates'], range_edges=d['range_edges'], elev_split_rad=float(d['elev_split_rad']))


def save_scene(path, scene):
    """One scene's cache: float16 box-local xyz and uint8 attributes per (keyframe, track); ICP table float32."""
    out = dict(scene)
    out['frames'] = [dict(f, points={k: (np.asarray(x, np.float16), np.asarray(a).astype(np.uint8))
                                     for k, (x, a) in f['points'].items()}) for f in scene['frames']]
    out['icp'] = {k: np.asarray(v, np.float32) for k, v in scene.get('icp', {}).items()}
    out['version'] = CACHE_VERSION
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_scene(path):
    with open(path, 'rb') as f:
        scene = pickle.load(f)
    assert scene.get('version') == CACHE_VERSION, f'{path}: cache version {scene.get("version")} != {CACHE_VERSION}'
    return scene


def build_scene(keyframes, read_points, icp=True, icp_min_points=5):
    """Cache entry for one scene.

    keyframes: time-ordered list of dicts with 'token', 't', 'boxes' (M, >=7, LIDAR_TOP frame), 'names', 'tids'.
    read_points(token) -> (N, 3 + k) points of that keyframe in its LIDAR_TOP frame, ego points removed.
    The ICP table holds, per (anchor index, track) with >= icp_min_points own points, the accepted corrections
    [j, yaw, tx, ty, tz] of every other keyframe j holding that track."""
    frames = []
    for kf in keyframes:
        tr = keyframe_tracks(read_points(kf['token']), kf['boxes'], kf['names'], kf['tids'])
        keep = [i for i, (n, t) in enumerate(zip(kf['names'], kf['tids'])) if n == CAR and t is not None]
        frames.append({'token': kf['token'], 't': float(kf['t']),
                       'tids': [str(kf['tids'][i]) for i in keep],
                       'boxes': np.asarray(kf['boxes'], np.float64)[keep, :7] if keep else np.zeros((0, 7)),
                       'points': tr})
    table = {}
    if icp:
        for a, fa in enumerate(frames):
            for k in fa['tids']:
                if k not in fa['points'] or len(fa['points'][k][0]) < icp_min_points:
                    continue
                own = fa['points'][k][0]
                rows = []
                for j, fj in enumerate(frames):
                    if j == a or k not in fj['points']:
                        continue
                    yaw, t, ok, _ = icp_4dof(fj['points'][k][0], own)
                    if ok:
                        rows.append([j, yaw, t[0], t[1], t[2]])
                if rows:
                    table[(a, k)] = np.array(rows)
    return {'frames': frames, 'icp': table}
