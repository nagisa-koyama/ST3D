"""Temporal propagation of confident pseudo-labels (experiments_md/20261017_02, pre-declared before any run).

Core library, numpy / scipy only, so every piece is CPU-testable (tests/test_pseudo_label_propagation.py). The two
drivers are `pseudo_label_propagation_source.py` (the rule's calibration on labelled nuScenes VAL) and
`pseudo_label_propagation_waymo.py` (label-free counts on Waymo TRAIN, the augmented pseudo-label file, and the
ANALYSIS-ONLY diagnosis on Waymo TRAIN labels).

Conventions (20261017_02 §3):
- A box is [x, y, z, dx, dy, dz, yaw] in its frame; a pseudo-label row is [box(7), label, score]. A NEGATIVE label is an
  ignore region (pseudo-label ignore band, or IGNORE_CLASS_LABEL = -99) and is never a seed, never propagated, never
  turned positive (bug_prevention_checklist rule 1/2: nothing here indexes with a label).
- A frame carries M, its frame -> global 4x4 transform, and t, its time in seconds. A box moves from frame f to frame g
  by T = inv(M_g) @ M_f: centre T.c, yaw + atan2(T[1,0], T[0,0]).
- Seeds are Car rows (label exactly CAR_LABEL) with score >= s_conf.
"""
import numpy as np

CAR_LABEL = 1
RANGE_EDGES = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, np.inf)
RANGE_NAMES = ('0-10', '10-20', '20-30', '30-40', '40-50', '50-75', '>=75')
PC_RANGE_XY = 75.2
DEDUP_IOU = 0.1


# ----------------------------------------------------------------------------------------------- geometry

def yaw_of(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


def transform_boxes(boxes, T):
    """Boxes (N, >=7) moved by the 4x4 T: centre rotated + translated, yaw + the rotation's yaw. Extra columns kept."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if len(boxes) == 0:
        return boxes.reshape(0, boxes.shape[1] if boxes.ndim == 2 else 7).copy()
    out = boxes.copy()
    out[:, :3] = boxes[:, :3] @ T[:3, :3].T + T[:3, 3]
    out[:, 6] = wrap_angle(boxes[:, 6] + yaw_of(T))
    return out


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def bev_corners(box):
    """The four BEV corners of one box, counter-clockwise."""
    x, y, dx, dy, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[dx / 2, dy / 2], [-dx / 2, dy / 2], [-dx / 2, -dy / 2], [dx / 2, -dy / 2]])
    R = np.array([[c, -s], [s, c]])
    return loc @ R.T + np.array([x, y])


def _clip(subject, clip):
    """Sutherland-Hodgman: the part of convex polygon `subject` inside convex CCW polygon `clip`."""
    out = list(subject)
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        inp, out = out, []
        if not inp:
            break
        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12
        def cross(p, q):
            # intersection of segment pq with the line ab
            d1 = np.array(q) - np.array(p)
            d2 = np.array(b) - np.array(a)
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-15:
                return q
            tt = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
            return (p[0] + tt * d1[0], p[1] + tt * d1[1])
        s = inp[-1]
        for e in inp:
            if inside(e):
                if not inside(s):
                    out.append(cross(s, e))
                out.append(e)
            elif inside(s):
                out.append(cross(s, e))
            s = e
    return out


def _area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def bev_iou_matrix(a, b):
    """Rotated BEV IoU between boxes a (N, >=7) and b (M, >=7). Pairs whose centres are too far apart to overlap are 0."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, np.asarray(a).shape[-1] if np.asarray(a).size else 7)
    b = np.asarray(b, dtype=np.float64).reshape(-1, np.asarray(b).shape[-1] if np.asarray(b).size else 7)
    out = np.zeros((len(a), len(b)))
    if len(a) == 0 or len(b) == 0:
        return out
    ra = 0.5 * np.hypot(a[:, 3], a[:, 4])
    rb = 0.5 * np.hypot(b[:, 3], b[:, 4])
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    near = d < (ra[:, None] + rb[None, :])
    ca = {}
    cb = {}
    for i, j in zip(*np.nonzero(near)):
        if i not in ca:
            ca[i] = [tuple(p) for p in bev_corners(a[i])]
        if j not in cb:
            cb[j] = [tuple(p) for p in bev_corners(b[j])]
        inter = _area(_clip(ca[i], cb[j]))
        union = a[i, 3] * a[i, 4] + b[j, 3] * b[j, 4] - inter
        out[i, j] = inter / union if union > 0 else 0.0
    return out


def range_of(boxes):
    boxes = np.asarray(boxes)
    if len(boxes) == 0:
        return np.zeros(0)
    return np.hypot(boxes[:, 0], boxes[:, 1])


def range_bin(r):
    """Index into RANGE_NAMES of BEV centre range r (array)."""
    return np.searchsorted(np.asarray(RANGE_EDGES[1:]), np.asarray(r), side='right')


def in_pc_range(boxes):
    boxes = np.asarray(boxes)
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    return (np.abs(boxes[:, 0]) <= PC_RANGE_XY) & (np.abs(boxes[:, 1]) <= PC_RANGE_XY)


# ----------------------------------------------------------------------------------------------- point evidence

GROUND_MARGIN = 1.5      # m, footprint dilation per side for the local ground estimate
GROUND_PCT = 10.0        # percentile of z in the dilated footprint taken as the local ground
GROUND_MIN_PTS = 10      # fewer -> the box's own bottom is the ground
CLEARANCE = 0.3          # m above the local ground before a point counts as evidence


def count_nonground(points, boxes, tree=None):
    """Points of the frame inside each box's BEV footprint, between local ground + 0.3 m and local ground + dz.

    The local ground is the 10th percentile of z over the footprint dilated by 1.5 m per side, within
    [z_bottom - 2.0, z_top] (fewer than 10 points: z_bottom). 20261017_02 §3.6.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    points = np.asarray(points)
    if len(points) == 0:
        return np.zeros(len(boxes), dtype=np.int64)
    if tree is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(points[:, :2])
    out = np.zeros(len(boxes), dtype=np.int64)
    for i, b in enumerate(boxes):
        x, y, z, dx, dy, dz, yaw = b[:7]
        rq = 0.5 * np.hypot(dx, dy) + GROUND_MARGIN * np.sqrt(2) + 1e-3
        idx = tree.query_ball_point([x, y], rq)
        if not idx:
            continue
        p = points[idx]
        c, s = np.cos(yaw), np.sin(yaw)
        u = (p[:, 0] - x) * c + (p[:, 1] - y) * s
        v = -(p[:, 0] - x) * s + (p[:, 1] - y) * c
        zb, zt = z - dz / 2, z + dz / 2
        dil = (np.abs(u) <= dx / 2 + GROUND_MARGIN) & (np.abs(v) <= dy / 2 + GROUND_MARGIN) & \
              (p[:, 2] >= zb - 2.0) & (p[:, 2] <= zt)
        g0 = np.percentile(p[dil, 2], GROUND_PCT) if dil.sum() >= GROUND_MIN_PTS else zb
        foot = (np.abs(u) <= dx / 2) & (np.abs(v) <= dy / 2)
        out[i] = int((foot & (p[:, 2] >= g0 + CLEARANCE) & (p[:, 2] <= g0 + dz)).sum())
    return out


def k_table(direct_counts, direct_ranges, q, min_per_bin=20):
    """k(q, bin) = max(1, q-quantile (method 'lower') of the direct Car boxes' counts in that bin); a bin with fewer
    than `min_per_bin` boxes uses the quantile pooled over all bins. Returns an array over RANGE_NAMES."""
    counts = np.asarray(direct_counts)
    bins = range_bin(direct_ranges)
    pooled = float(np.quantile(counts, q, method='lower')) if len(counts) else 1.0
    out = np.zeros(len(RANGE_NAMES))
    for k in range(len(RANGE_NAMES)):
        c = counts[bins == k]
        val = float(np.quantile(c, q, method='lower')) if len(c) >= min_per_bin else pooled
        out[k] = max(1.0, val)
    return out


def classify(counts, ranges, ktab):
    """'pos' (n >= k), 'ign' (1 <= n < k) or 'empty' (n == 0) per added box."""
    counts = np.asarray(counts)
    k = np.asarray(ktab)[range_bin(ranges)] if len(counts) else np.zeros(0)
    cls = np.full(len(counts), 'ign', dtype=object)
    cls[counts >= k] = 'pos'
    cls[counts == 0] = 'empty'
    return cls


# ----------------------------------------------------------------------------------------------- matching

def greedy_match(pred_xy, scores, gt_xy, max_dist=1.0, used=None):
    """As source_precision_cut.match: predictions in descending score, each to the nearest unused GT within max_dist.
    Returns (hit per prediction, the updated `used` mask over GT)."""
    pred_xy = np.asarray(pred_xy).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy).reshape(-1, 2)
    hit = np.zeros(len(pred_xy), bool)
    used = np.zeros(len(gt_xy), bool) if used is None else used.copy()
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return hit, used
    for i in np.argsort(-np.asarray(scores), kind='stable'):
        d = np.hypot(*(gt_xy - pred_xy[i]).T)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= max_dist:
            used[j] = True
            hit[i] = True
    return hit, used


def match_to_gt_index(pred_xy, scores, gt_xy, max_dist=1.0):
    """Like greedy_match but returns the matched GT index per prediction (-1 if none)."""
    pred_xy = np.asarray(pred_xy).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy).reshape(-1, 2)
    idx = -np.ones(len(pred_xy), dtype=np.int64)
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return idx
    used = np.zeros(len(gt_xy), bool)
    for i in np.argsort(-np.asarray(scores), kind='stable'):
        d = np.hypot(*(gt_xy - pred_xy[i]).T)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= max_dist:
            used[j] = True
            idx[i] = j
    return idx


# ----------------------------------------------------------------------------------------------- sequences

class Frame:
    """One pseudo-labelled frame of a sequence: id, time (s), frame -> global M (4x4), pseudo-label rows (N, 9)."""

    __slots__ = ('fid', 't', 'M', 'rows')

    def __init__(self, fid, t, M, rows):
        self.fid = fid
        self.t = float(t)
        self.M = np.asarray(M, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.float64)
        self.rows = rows.reshape(-1, 9) if rows.size else np.zeros((0, 9))


def seeds_of(frame, s_conf):
    """Indices of the frame's seeds: label exactly CAR_LABEL (+1) and score >= s_conf. Negative labels never qualify."""
    r = frame.rows
    if len(r) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.nonzero((r[:, 7] == CAR_LABEL) & (r[:, 8] >= s_conf))[0]


def to_global(frame, idx):
    return transform_boxes(frame.rows[idx, :7], frame.M)


def static_clusters(frames, s_conf, d_assoc):
    """Variant S clusters (20261017_02 §3.2). Returns (clusters, member_of) where each cluster is a dict with
    'anchor' (global box, 7), 'score', 'frames' (sorted frame indices), 'members' [(frame index, row index)], and
    'static' (members in >= 2 frames); member_of maps (frame index, row index) -> cluster index."""
    seeds = []
    for fi, f in enumerate(frames):
        idx = seeds_of(f, s_conf)
        if len(idx) == 0:
            continue
        g = to_global(f, idx)
        for k, ri in enumerate(idx):
            seeds.append((f.rows[ri, 8], fi, int(ri), g[k]))
    seeds.sort(key=lambda s: (-s[0], s[1], s[2]))
    clusters = []
    member_of = {}
    anchors = np.zeros((0, 2))
    for score, fi, ri, gbox in seeds:
        best, best_d = -1, np.inf
        if len(clusters):
            d = np.hypot(anchors[:, 0] - gbox[0], anchors[:, 1] - gbox[1])
            for ci in np.argsort(d, kind='stable'):
                if d[ci] > d_assoc:
                    break
                if fi in clusters[ci]['frame_set']:
                    continue
                best, best_d = int(ci), d[ci]
                break
        if best < 0:
            clusters.append({'anchor': gbox.copy(), 'score': float(score), 'members': [(fi, ri)],
                             'frame_set': {fi}})
            anchors = np.vstack([anchors, gbox[None, :2]])
            member_of[(fi, ri)] = len(clusters) - 1
        else:
            clusters[best]['members'].append((fi, ri))
            clusters[best]['frame_set'].add(fi)
            member_of[(fi, ri)] = best
    for c in clusters:
        c['frames'] = sorted(c.pop('frame_set'))
        c['static'] = len(c['frames']) >= 2
    return clusters, member_of


def propagate_static(frames, clusters, H):
    """Candidates of variant S: for each static cluster and each frame g holding no member, with Delta (time to the
    nearest member frame) <= H, the anchor box moved into g, kept if its centre is inside the point-cloud range.
    Returns {frame index: list of dicts(box(7), score, delta, cluster)}."""
    out = {}
    times = np.array([f.t for f in frames])
    for ci, c in enumerate(clusters):
        if not c['static']:
            continue
        mt = times[c['frames']]
        member = set(c['frames'])
        for g, f in enumerate(frames):
            if g in member:
                continue
            delta = float(np.min(np.abs(mt - f.t)))
            if delta > H + 1e-6:
                continue
            box = transform_boxes(c['anchor'][None, :], np.linalg.inv(f.M))[0]
            if not in_pc_range(box[None])[0]:
                continue
            out.setdefault(g, []).append({'box': box, 'score': c['score'], 'delta': delta, 'src': ci})
    return out


def cv_tracks(frames, s_conf, exclude, d_init, d_cv, min_obs=3, max_miss=1):
    """Variant D tracks (20261017_02 §3.3) over the seeds not in `exclude` (a set of (frame index, row index)).
    A track: dict 'obs' = list of (frame index, row index, t, global box(7), score). Only tracks with >= min_obs
    observations are returned."""
    active, done = [], []
    for fi, f in enumerate(frames):
        idx = [ri for ri in seeds_of(f, s_conf) if (fi, int(ri)) not in exclude]
        g = to_global(f, np.asarray(idx, dtype=np.int64)) if idx else np.zeros((0, 7))
        # predictions and gates
        pairs = []
        for ti, tr in enumerate(active):
            last = tr['obs'][-1]
            dt = f.t - last[2]
            if tr['v'] is None:
                p, gate = last[3][:2], d_init
            else:
                p, gate = last[3][:2] + tr['v'] * dt, d_cv
            for si in range(len(idx)):
                d = float(np.hypot(*(g[si, :2] - p)))
                if d <= gate:
                    pairs.append((d, ti, si))
        pairs.sort()
        t_used, s_used = set(), set()
        for d, ti, si in pairs:
            if ti in t_used or si in s_used:
                continue
            t_used.add(ti); s_used.add(si)
            tr = active[ti]
            prev = tr['obs'][-1]
            tr['obs'].append((fi, int(idx[si]), f.t, g[si].copy(), float(f.rows[idx[si], 8])))
            tr['v'] = (g[si, :2] - prev[3][:2]) / max(f.t - prev[2], 1e-6)
            tr['miss'] = 0
        still = []
        for ti, tr in enumerate(active):
            if ti not in t_used:
                tr['miss'] += 1
            (done if tr['miss'] > max_miss else still).append(tr)
        active = still
        for si in range(len(idx)):
            if si not in s_used:
                active.append({'obs': [(fi, int(idx[si]), f.t, g[si].copy(), float(f.rows[idx[si], 8]))],
                               'v': None, 'miss': 0})
    done += active
    return [tr for tr in done if len(tr['obs']) >= min_obs]


def track_box_at(tr, t):
    """The track's global box at time t: linear interpolation inside its span, constant-velocity extrapolation from the
    nearest end outside it; size of the highest-score observation, yaw of the nearest-in-time observation."""
    obs = tr['obs']
    ts = np.array([o[2] for o in obs])
    xyz = np.array([o[3][:3] for o in obs])
    best = obs[int(np.argmax([o[4] for o in obs]))][3]
    near = obs[int(np.argmin(np.abs(ts - t)))][3]
    if t <= ts[0]:
        v = (xyz[1] - xyz[0]) / max(ts[1] - ts[0], 1e-6)
        c = xyz[0] + v * (t - ts[0])
    elif t >= ts[-1]:
        v = (xyz[-1] - xyz[-2]) / max(ts[-1] - ts[-2], 1e-6)
        c = xyz[-1] + v * (t - ts[-1])
    else:
        k = int(np.searchsorted(ts, t))
        w = (t - ts[k - 1]) / max(ts[k] - ts[k - 1], 1e-6)
        c = xyz[k - 1] + w * (xyz[k] - xyz[k - 1])
    return np.array([c[0], c[1], c[2], best[3], best[4], best[5], near[6]])


def propagate_tracks(frames, tracks, H):
    """Candidates of variant D: each track into every frame without an observation, Delta <= H."""
    out = {}
    for ti, tr in enumerate(tracks):
        own = {o[0] for o in tr['obs']}
        ts = np.array([o[2] for o in tr['obs']])
        score = max(o[4] for o in tr['obs'])
        for g, f in enumerate(frames):
            if g in own:
                continue
            delta = float(np.min(np.abs(ts - f.t)))
            if delta > H + 1e-6:
                continue
            box = transform_boxes(track_box_at(tr, f.t)[None, :], np.linalg.inv(f.M))[0]
            if not in_pc_range(box[None])[0]:
                continue
            out.setdefault(g, []).append({'box': box, 'score': score, 'delta': delta, 'src': ti})
    return out


def dedup(cands, existing, H=np.inf, iou_thresh=DEDUP_IOU):
    """Split one frame's candidates (with delta <= H) into added / duplicate-of-existing / duplicate-among-propagated.

    `existing` (N, >=7): every box already labelled in the frame - the direct pseudo-labels of ANY class and sign, plus,
    for variant D, the S-added boxes. Candidates sorted by (delta asc, score desc) go through greedy NMS at iou_thresh.
    Returns (added list, n_dup_existing, n_dup_propagated)."""
    cs = [c for c in cands if c['delta'] <= H + 1e-6]
    if not cs:
        return [], 0, 0
    boxes = np.array([c['box'] for c in cs])
    ex = np.asarray(existing).reshape(-1, np.asarray(existing).shape[-1] if np.asarray(existing).size else 7)
    dup_ex = (bev_iou_matrix(boxes, ex[:, :7]) > iou_thresh).any(axis=1) if len(ex) else np.zeros(len(cs), bool)
    order = sorted([i for i in range(len(cs)) if not dup_ex[i]], key=lambda i: (cs[i]['delta'], -cs[i]['score']))
    kept = []
    n_dup_prop = 0
    for i in order:
        if kept and (bev_iou_matrix(boxes[i:i + 1], boxes[kept]) > iou_thresh).any():
            n_dup_prop += 1
            continue
        kept.append(i)
    return [cs[i] for i in kept], int(dup_ex.sum()), n_dup_prop


def delta_bin(delta):
    return int(max(1, round(float(delta))))


# ----------------------------------------------------------------------------------------------- output

def merge_entry(entry, added_boxes, labels, scores):
    """A pseudo-label entry with rows appended. The original rows and arrays are copied unchanged (bit-identical);
    labels must be +1 or -1 (Car positive / Car ignore)."""
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    assert np.all(np.isin(labels, (1.0, -1.0))), 'added rows are Car positives (+1) or Car ignore regions (-1) only'
    added_boxes = np.asarray(added_boxes, dtype=np.float64).reshape(-1, 7)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    gt = np.asarray(entry['gt_boxes'])
    new_rows = np.concatenate([added_boxes, labels[:, None], scores[:, None]], axis=1).astype(gt.dtype)
    out = dict(entry)
    out['gt_boxes'] = np.concatenate([gt.reshape(-1, 9), new_rows], axis=0)
    n = len(labels)
    for key in ('cls_scores', 'iou_scores'):
        v = entry.get(key, None)
        if v is not None:
            v = np.asarray(v)
            out[key] = np.concatenate([v.reshape(-1), scores.astype(v.dtype)])
    mc = np.asarray(entry.get('memory_counter', np.zeros(len(gt))))
    out['memory_counter'] = np.concatenate([mc.reshape(-1), np.zeros(n, dtype=mc.dtype)])
    tc = entry.get('teacher_classes', None)
    if tc is not None:
        tc = np.asarray(tc)
        wide_enough = tc.dtype.kind == 'U' and tc.dtype.itemsize >= np.dtype('U7').itemsize
        dtype = tc.dtype if wide_enough else object
        names = np.array(['Car' if l > 0 else 'Ignored' for l in labels], dtype=dtype)
        out['teacher_classes'] = np.concatenate([tc.astype(dtype), names])
    return out


def strip_labels(info):
    """The label-free view of a Waymo info: only frame_id, pose and point_cloud (20261017_02 §2)."""
    return {'frame_id': info['frame_id'], 'pose': np.asarray(info['pose']),
            'point_cloud': dict(info['point_cloud'])}


def waymo_get_lidar(npy_path):
    """As WaymoDataset.get_lidar with LIDAR_INDICES absent: columns 0-4 of the frame's .npy, NLZ_flag == -1, tanh
    intensity. Then the POINT_CLOUD_RANGE cut of the target config (SHIFT_COOR 0)."""
    pf = np.load(npy_path)
    pts = pf[:, 0:5][pf[:, 5] == -1].copy()
    pts[:, 3] = np.tanh(pts[:, 3])
    m = (np.abs(pts[:, 0]) <= 75.2) & (np.abs(pts[:, 1]) <= 75.2) & (pts[:, 2] >= -2) & (pts[:, 2] <= 4)
    return pts[m]
