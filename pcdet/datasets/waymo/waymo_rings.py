"""Waymo TOP lidar: laser rows from the stored point ORDER, and thinning to another sensor's scan pattern.

Why rows come from order and not from elevation: the TOP points are motion-compensated per pixel, which moves each
point's apparent inclination by up to a few tenths of a degree - more than Waymo's 0.14 deg row spacing near the
horizon - so assigning a point to the nearest declared inclination is right for only ~56% of points
(experiments_md 20260927_05). But the re-extracted files (tools/analysis/waymo_reextract_all.py) store the range
image's valid pixels ROW-MAJOR, TOP first (`num_points_of_each_lidar[0]` points), so within a row the column index
rises and a new row starts where it falls back. Compensation shifts a point by at most a few columns, so the order
survives. The column comes from the azimuth about the sensor in the VEHICLE's axes (the sensor-frame azimuth plus the
extrinsic yaw, ~148 deg; reconstruct() in waymo_reextract_bench.py builds the pixel azimuth that way).

Row ORDER is the truth; the declared inclinations (SOURCE calibration, cached per segment) only map rows to angles.
A monotone match of each recovered row's median elevation to the declared beams absorbs the two ways the order can
mislead: a row with no valid pixel at all (open sky above the top beams) is invisible in the order, and a row whose
points all lie within a few columns of the seam can merge with its neighbour.

`RING_PATTERN` (WaymoDataset, training only) uses these rows to thin the TOP cloud to a target sensor's published
scan pattern: rows at the target's vertical spacing with a random phase, one point per row per target azimuth bin
(`AZ_RES_DEG: null`: every point of the kept rows, the vertical-only ablation; experiments_md 20261005_01).
"""
import numpy as np

W_TOP = 2650          # TOP range-image width (columns)
SEAM_COLS = 30        # columns either side of the range-image seam where compensation can wrap a point


def top_columns_and_elevations(xyz, extrinsic, width=W_TOP):
    """Per TOP point: (fractional range-image column in [0, width), elevation in radians about the sensor).

    `xyz` in the vehicle frame (the processed files' frame, before any SHIFT_COOR), `extrinsic` the TOP
    sensor -> vehicle transform.
    """
    E = np.asarray(extrinsic, dtype=np.float64)
    Einv = np.linalg.inv(E)
    ps = np.asarray(xyz, dtype=np.float64)[:, :3] @ Einv[:3, :3].T + Einv[:3, 3]
    yaw = np.arctan2(E[1, 0], E[0, 0])
    az = np.arctan2(ps[:, 1], ps[:, 0]) + yaw
    # pixel j's centre maps to j + 0.5, so float rounding at a row's first pixel cannot wrap it to the far side
    col = np.mod(width - (az / np.pi + 1.0) * width / 2.0, width)
    el = np.arctan2(ps[:, 2], np.hypot(ps[:, 0], ps[:, 1]))
    return col, el


def rows_from_order(col, width=W_TOP, seam=SEAM_COLS):
    """Row number (0, 1, ... in order of appearance) of each point of a row-major cloud.

    A new row starts where the column falls by more than half the width. Compensation can wrap a point that sits
    within a few columns of the seam to the other side, which looks like a fall followed by a rise; so a fall only
    starts a new row if the current row already holds a point away from the seam (hysteresis). Costs a loop over the
    ~64-130 candidate falls, not over points.
    """
    n = len(col)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    cand = np.nonzero(np.diff(col) < -width / 2.0)[0] + 1
    away = (col >= seam) & (col < width - seam)
    c_away = np.concatenate([[0], np.cumsum(away)])
    starts = [0]
    for c in cand:
        if c_away[c] - c_away[starts[-1]] > 0:
            starts.append(int(c))
    # the last row has no later fall to test it: if it holds only seam points, it is the previous row's wrapped tail
    if len(starts) > 1 and c_away[n] - c_away[starts[-1]] == 0:
        starts.pop()
    rid = np.zeros(n, dtype=np.int64)
    rid[np.asarray(starts[1:], dtype=np.int64)] = 1
    return np.cumsum(rid)


def match_rows_to_beams(row_median_el, inclinations_desc):
    """Monotone assignment of recovered rows, top to bottom, to declared beams, top to bottom.

    When the order yields exactly one row per declared beam, the ORDER is the answer (identity): row medians are only
    a check, and a frame whose medians are perturbed (per-pixel compensation over a bump) must not be re-mapped by
    them. Otherwise minimise the summed |median elevation - declared inclination|, strictly increasing when there are
    no more rows than beams (a skipped beam is a row with no valid pixel), non-decreasing when there are more (a
    spurious split maps both halves to one beam). Returns beam index per recovered row (0 = highest beam).
    """
    m = np.asarray(row_median_el, dtype=np.float64)
    d = np.asarray(inclinations_desc, dtype=np.float64)
    R, B = len(m), len(d)
    if R == 0:
        return np.zeros(0, dtype=np.int64)
    if R == B:
        return np.arange(R, dtype=np.int64)
    strict = R < B
    cost = np.abs(m[:, None] - d[None, :])
    acc = np.empty((R, B)); arg = np.empty((R, B), dtype=np.int64)
    acc[0] = cost[0]
    ar = np.arange(B)
    for r in range(1, R):
        best = np.minimum.accumulate(acc[r - 1])
        # index attaining the running minimum over b' <= b (forward-filled where a new minimum is reached)
        bidx = np.maximum.accumulate(np.where(acc[r - 1] <= best, ar, 0))
        if strict:   # predecessor must be b' <= b - 1
            best = np.concatenate([[np.inf], best[:-1]]); bidx = np.concatenate([[0], bidx[:-1]])
        acc[r] = cost[r] + best
        arg[r] = bidx
    out = np.empty(R, dtype=np.int64)
    out[-1] = int(np.argmin(acc[-1]))
    for r in range(R - 1, 0, -1):
        out[r - 1] = arg[r, out[r]]
    return out


def top_beam_ids(xyz, extrinsic, inclinations, width=W_TOP):
    """Beam index (0 = highest declared beam) of every TOP point, plus diagnostics.

    `xyz` must be the TOP block of a processed file in STORED order, before the NLZ filter or any shuffle.
    """
    inc_desc = np.sort(np.asarray(inclinations, dtype=np.float64))[::-1]
    col, el = top_columns_and_elevations(xyz, extrinsic, width)
    rid = rows_from_order(col, width)
    n_rows = int(rid[-1]) + 1 if len(rid) else 0
    order = np.argsort(rid, kind='stable')
    bounds = np.searchsorted(rid[order], np.arange(n_rows + 1))
    med = np.array([np.median(el[order[bounds[k]:bounds[k + 1]]]) for k in range(n_rows)])
    beam_of_row = match_rows_to_beams(med, inc_desc)
    resid = np.abs(med - inc_desc[beam_of_row]) if n_rows else np.zeros(0)
    diag = {'rows': n_rows, 'strict': bool(np.all(np.diff(beam_of_row) > 0)),
            'max_resid_deg': float(np.degrees(resid.max())) if n_rows else 0.0}
    return beam_of_row[rid], col, el, inc_desc, diag


def ring_pattern_mask(beam, col, inc_desc, spacing_deg, az_res_deg, phase_v, phase_h, width=W_TOP):
    """Keep the rows nearest a vertical lattice of `spacing_deg` (offset `phase_v` in [0, 1) of a spacing) inside
    the source's field of view, and one point per kept row per azimuth bin of `az_res_deg` (offset `phase_h`).
    `az_res_deg` None keeps every point of the kept rows (the vertical lattice alone; `phase_h` unused).

    Within a row the stored order is column order, so the point kept per bin is the first one the scan reached.
    """
    lo, hi = inc_desc.min(), inc_desc.max()
    sp = np.radians(spacing_deg)
    j0 = int(np.floor((lo - phase_v * sp) / sp)) - 1
    j1 = int(np.ceil((hi - phase_v * sp) / sp)) + 1
    targets = (np.arange(j0, j1 + 1) + phase_v) * sp
    targets = targets[(targets >= lo) & (targets <= hi)]
    kept_beams = np.unique(np.argmin(np.abs(inc_desc[None, :] - targets[:, None]), axis=1))
    keep = np.isin(beam, kept_beams)
    if az_res_deg is None:
        return keep, kept_beams
    n_bins = int(round(360.0 / az_res_deg))
    b = np.floor((col / width + phase_h / n_bins) * n_bins).astype(np.int64) % n_bins
    key = beam.astype(np.int64) * n_bins + b
    idx = np.nonzero(keep)[0]
    _, first = np.unique(key[idx], return_index=True)
    out = np.zeros(len(beam), dtype=bool)
    out[idx[first]] = True
    return out, kept_beams


def ring_pattern_points(raw, num_points_of_each_lidar, calib, cfg, rng=None):
    """The training cloud for RING_PATTERN: one stored `.npy` frame -> (N, 5) [x, y, z, tanh(intensity), elongation].

    `raw` is the file as stored (x y z intensity elongation nlz, lidars in name order, TOP first, row-major) and
    `calib` the segment's {'inclinations', 'extrinsic'} for TOP. Rows are recovered on the FULL TOP block, before
    the no-label-zone filter, because the filter removes points from the middle of rows and the order is what
    identifies them; the filter (nlz == -1 kept, as `get_lidar`) is applied to what is kept. Side lidars are dropped
    unless `cfg.TOP_ONLY` is False. `cfg.AZ_RES_DEG` must be present: null switches the azimuth binning off (both
    phases are still drawn, so the vertical phase stream is the same as with binning on).
    """
    rng = np.random if rng is None else rng
    counts = np.asarray(num_points_of_each_lidar).astype(np.int64)
    assert counts.sum() == len(raw), 'per-lidar counts %s do not add up to the %d stored points' % (counts, len(raw))
    top = raw[:counts[0]]
    if len(top):
        beam, col, _, inc_desc, _ = top_beam_ids(top[:, :3], calib['extrinsic'], calib['inclinations'])
        draw = rng.random_sample if hasattr(rng, 'random_sample') else rng.random   # np.random or a Generator
        keep, _ = ring_pattern_mask(beam, col, inc_desc, cfg['SPACING_DEG'], cfg['AZ_RES_DEG'], draw(), draw())
        pts = top[keep]
    else:
        pts = top
    if not cfg.get('TOP_ONLY', True):
        pts = np.concatenate([pts, raw[counts[0]:]])
    pts = pts[pts[:, 5] == -1]
    out = np.array(pts[:, 0:5], dtype=raw.dtype)
    out[:, 3] = np.tanh(out[:, 3])
    return out


def ring_labelled_points(raw, num_points_of_each_lidar, calib):
    """(points (N, 5) as `get_lidar` returns them, ring label per point) for BEAM_DISTILL / BEAM_DROP on Waymo.

    Labels follow `beam_downsample_utils`' convention - ring index in ASCENDING elevation (0 = lowest TOP beam) - and
    are -1 for the side lidars, which `generate_mask` / `random_ring_subset_mask` drop from the thinned stream. Rows
    come from the stored order on the full TOP block; the no-label-zone filter is applied afterwards, to points and
    labels alike.
    """
    counts = np.asarray(num_points_of_each_lidar).astype(np.int64)
    assert counts.sum() == len(raw), 'per-lidar counts %s do not add up to the %d stored points' % (counts, len(raw))
    label = np.full(len(raw), -1, dtype=np.int64)
    n_top = int(counts[0])
    if n_top:
        beam, _, _, inc_desc, _ = top_beam_ids(raw[:n_top, :3], calib['extrinsic'], calib['inclinations'])
        label[:n_top] = len(inc_desc) - 1 - beam
    keep = raw[:, 5] == -1
    out = np.array(raw[keep, 0:5], dtype=raw.dtype)
    out[:, 3] = np.tanh(out[:, 3])
    return out, label[keep]


def eval_top_thin_points(raw, num_points_of_each_lidar, calib, mode, stride=2, rng=None, az_res_deg=None):
    """ANALYSIS: one stored frame with its TOP block thinned (EVAL_TOP_THIN at evaluation, TRAIN_TOP_THIN in training), returned as `get_lidar` returns it.

    `mode` 'rows' keeps the TOP beams whose index (0 = highest declared beam) is a multiple of `stride`, so every
    object loses whole scan lines (lattice-row coverage falls by about 1 / stride). `mode` 'random' is its
    count-matched control: the SAME NUMBER of TOP points, drawn uniformly over the TOP block, so density falls
    as much while every row keeps some points. Side lidars are untouched in both. Rows are recovered on the full
    TOP block in stored order, before the no-label-zone filter (as `ring_pattern_points`); the filter and
    tanh(intensity) follow `get_lidar`. experiments_md 20261009_02 (go / no-go for a learned re-renderer).
    """
    counts = np.asarray(num_points_of_each_lidar).astype(np.int64)
    assert counts.sum() == len(raw), 'per-lidar counts %s do not add up to the %d stored points' % (counts, len(raw))
    n_top = int(counts[0])
    keep = np.ones(len(raw), dtype=bool)
    if n_top:
        beam, col, _, _, _ = top_beam_ids(raw[:n_top, :3], calib['extrinsic'], calib['inclinations'])
        rows = (beam % stride) == 0
        cols = (np.floor(col).astype(np.int64) % stride) == 0   # every STRIDE-th range-image column (col is fractional)
        if mode in ('azbin', 'random_azbin'):
            # one point per line per AZ_RES_DEG azimuth bin (a non-integer step, e.g. nuScenes' 0.332 deg = 2.44 Waymo
            # columns): the first point of each (beam, bin) in stored order (20261010_01 §3)
            assert az_res_deg, 'azbin needs AZ_RES_DEG'
            key = beam.astype(np.int64) * 100000 + np.floor(col * (360.0 / W_TOP) / az_res_deg).astype(np.int64)
            azbin = np.zeros(n_top, dtype=bool)
            azbin[np.unique(key, return_index=True)[1]] = True
        if mode == 'azbin':
            keep[:n_top] = azbin
        elif mode == 'random_azbin':
            rng = np.random.default_rng(0) if rng is None else rng
            sel = np.zeros(n_top, dtype=bool)
            sel[rng.choice(n_top, int(azbin.sum()), replace=False)] = True
            keep[:n_top] = sel
        elif mode == 'rows':
            keep[:n_top] = rows
        elif mode == 'cols':
            keep[:n_top] = cols
        elif mode in ('random', 'random_cols'):
            # count-matched controls: 'random' matches the rows cut (as since 20261009_02), 'random_cols' the cols cut
            rng = np.random.default_rng(0) if rng is None else rng
            sel = np.zeros(n_top, dtype=bool)
            sel[rng.choice(n_top, int((rows if mode == 'random' else cols).sum()), replace=False)] = True
            keep[:n_top] = sel
        else:
            raise ValueError("MODE must be 'rows', 'cols', 'azbin', 'random', 'random_cols' or 'random_azbin', got %r" % (mode,))
    pts = raw[keep]
    pts = pts[pts[:, 5] == -1]
    out = np.array(pts[:, 0:5], dtype=raw.dtype)
    out[:, 3] = np.tanh(out[:, 3])
    return out
