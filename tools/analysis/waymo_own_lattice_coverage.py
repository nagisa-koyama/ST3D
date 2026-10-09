"""Waymo's OWN lattice-row coverage (G1's RC) on its sparse far cars, from the native TOP range images (experiments_md
20261009_01 §9). ANALYSIS: reads Waymo val labels. Calibrates the re-render gate's 0.8 bar.

Frame 50 of every VAL segment (raw tfrecords, pure-numpy reader of waymo_reextract_bench.py). Per Vehicle box at 20-75 m
with 10-199 TOP first-return points (membership on per-pixel-pose-corrected points, which the labels refer to):
- silhouette = native range-image pixels whose ray (TOP's extrinsic origin; the segment's beam inclination; column
  azimuth pi - (c + 0.5) * 2 pi / W minus the extrinsic yaw, as reconstruct() builds it) hits the box;
- a pixel is occluded if its first return lies > 0.5 m in front of the box surface and is not in the box;
- RC = rows holding an in-box first return / rows with >= 1 visible silhouette pixel.
DECIDING lattice: the segment's own inclinations. REPORTED: the 1,000-segment median (every return assigned to its
nearest median row; per median cell the nearest return).

    python analysis/waymo_own_lattice_coverage.py measure <out.npz> [frame=50] [prev=40]
    python analysis/waymo_own_lattice_coverage.py report <out.npz> <gate dir with control_n00*.npz, deep_n00*.npz, waymo_val.npz>
"""
import pickle
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import _init_path  # noqa
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils
from pcdet.datasets import rerender_utils as RR
from waymo_reextract_bench import frames, parse_frame, reconstruct
from lattice_depth_ceiling import sector_of, SECTORS

RAW = Path('/home/koyama/code/ST3D/data/waymo/raw_data')
INFOS = '/home/koyama/code/ST3D/data/waymo/waymo_infos_val.pkl'
W = 2650
OCCL_M = 0.5
RINGS = ((20, 40), (40, 75))


def silhouette_native(box, inc_desc, E):
    """(rows, cols, entry range) of the range-image pixels whose ray from TOP hits the box (vehicle frame)."""
    o = E[:3, 3]; R = E[:3, :3]; yaw = np.arctan2(E[1, 0], E[0, 0])
    cs = (box_utils.boxes_to_corners_3d(box[None, :7].astype(np.float64))[0] - o) @ R   # corners in the sensor frame
    el = np.arctan2(cs[:, 2], np.hypot(cs[:, 0], cs[:, 1])); az = np.arctan2(cs[:, 1], cs[:, 0])
    cen = (box[:3] - o) @ R; az0 = np.arctan2(cen[1], cen[0])
    daz = (az - az0 + np.pi) % (2 * np.pi) - np.pi
    rows = np.nonzero((inc_desc >= el.min() - 0.01) & (inc_desc <= el.max() + 0.01))[0]
    col_of = lambda a: (np.pi - yaw - a) * W / (2 * np.pi) - 0.5
    c1, c2 = col_of(az0 + daz.max()), col_of(az0 + daz.min())
    cols = np.arange(int(np.floor(min(c1, c2))) - 1, int(np.ceil(max(c1, c2))) + 2)
    if not len(rows) or len(cols) > W // 2:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    Rr, Cc = np.meshgrid(rows, cols, indexing='ij'); Rr, Cc = Rr.ravel(), Cc.ravel()
    inc = inc_desc[Rr]; a = np.pi - (Cc + 0.5) * 2 * np.pi / W - yaw
    d = np.stack([np.cos(inc) * np.cos(a), np.cos(inc) * np.sin(a), np.sin(inc)], 1) @ R.T   # vehicle frame
    ch, sh = np.cos(-box[6]), np.sin(-box[6])
    rot = lambda v: np.stack([v[..., 0] * ch - v[..., 1] * sh, v[..., 0] * sh + v[..., 1] * ch, v[..., 2]], -1)
    oo = rot(o - box[:3]); dd = rot(d); e = np.asarray(box[3:6]) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (-e - oo) / dd; t2 = (e - oo) / dd
    t1 = np.where(np.isfinite(t1), t1, -np.inf); t2 = np.where(np.isfinite(t2), t2, np.inf)
    tmin = np.minimum(t1, t2).max(1); tmax = np.maximum(t1, t2).min(1); hit = tmax >= np.maximum(tmin, 0)
    return Rr[hit], Cc[hit] % W, np.maximum(tmin[hit], 0)


def rc_on(rows, cols, tin, rng_img, inbox_img, inbox_rows):
    """RC given a silhouette on some lattice, that lattice's nearest-return range image and in-box masks."""
    r = rng_img[rows, cols]
    occl = (r > 0) & np.isfinite(r) & (r < tin - OCCL_M) & ~inbox_img[rows, cols]
    vis_rows = np.unique(rows[~occl])
    if not len(vis_rows):
        return np.nan, 0
    return np.isin(vis_rows, inbox_rows).mean(), len(vis_rows)


def measure(out, frame_idx=50, prev_idx=40):
    infos = pickle.load(open(INFOS, 'rb'))
    by = {(i['point_cloud']['lidar_sequence'], i['point_cloud']['sample_idx']): i for i in infos}
    seqs = sorted({k[0] for k in by})
    med_asc = RR.lattice_spec('waymo_top')['thetas']; med_desc = med_asc[::-1]
    recs = []
    for si, seq in enumerate(seqs):
        info, info0 = by.get((seq, frame_idx)), by.get((seq, prev_idx))
        if info is None or info0 is None:
            continue
        buf = None
        for k, b in enumerate(frames(RAW / (seq + '.tfrecord'))):
            if k == frame_idx:
                buf = b; break
        if buf is None:
            continue
        pose, cal, lasers = parse_frame(buf)
        ri, rpose = lasers[1]; inc, lo, hi, E = cal[1]
        inc_desc = np.sort(np.asarray(inc, np.float64))[::-1]
        P, _, _, _, valid = reconstruct(ri, cal[1], rpose, pose)
        vr, vc = np.nonzero(valid)
        rng = np.where(valid, ri[..., 0], np.inf)
        # ego motion over the last second, in this frame's vehicle frame
        P50, P40 = np.asarray(info['pose']), np.asarray(info0['pose'])
        d = P50[:3, :3].T @ (P50[:3, 3] - P40[:3, 3]); moving = np.hypot(d[0], d[1]) >= 1.0
        fwd = d[:2] / max(np.hypot(d[0], d[1]), 1e-9) if moving else np.array([1.0, 0.0])
        a = info['annos']; boxes = np.asarray(a['gt_boxes_lidar'], np.float64)[np.asarray(a['name']) == 'Vehicle']
        if not len(boxes):
            continue
        rr = np.hypot(boxes[:, 0], boxes[:, 1]); cand = np.nonzero((rr >= 20) & (rr < 75))[0]
        if not len(cand):
            continue
        inb = roiaware_pool3d_utils.points_in_boxes_cpu(P.astype(np.float32), boxes[cand, :7].astype(np.float32)) > 0
        # median lattice: each native row -> nearest median row; per (median row, col) the nearest return
        mrow = np.argmin(np.abs(inc_desc[:, None] - med_desc[None, :]), 1)
        med_rng = np.full((len(med_desc), W), np.inf); np.minimum.at(med_rng, (mrow[vr], vc), rng[vr, vc])
        for jj, j in enumerate(cand):
            n_top = int(inb[jj].sum())
            if not 10 <= n_top < 200:
                continue
            inbox_img = np.zeros(rng.shape, bool); inbox_img[vr[inb[jj]], vc[inb[jj]]] = True
            rows, cols, tin = silhouette_native(boxes[j], inc_desc, E)
            rc_own, nv_own = rc_on(rows, cols, tin, rng, inbox_img, np.unique(vr[inb[jj]])) if len(rows) else (np.nan, 0)
            mrows, mcols, mtin = silhouette_native(boxes[j], med_desc, E)
            inbox_med = np.zeros(med_rng.shape, bool); inbox_med[mrow[vr[inb[jj]]], vc[inb[jj]]] = True
            rc_med, nv_med = rc_on(mrows, mcols, mtin, med_rng, inbox_med, np.unique(mrow[vr[inb[jj]]])) if len(mrows) else (np.nan, 0)
            recs.append((si, rr[j], sector_of(boxes[j:j + 1, :2], fwd)[0], moving, n_top, rc_own, rc_med, nv_own, nv_med))
        if (si + 1) % 20 == 0:
            print(f'  {si + 1} / {len(seqs)} segments, {len(recs)} cars', flush=True)
    np.savez(out, rows=np.array(recs, float), n_segments=len(seqs))
    print(f'saved {out}: {len(recs)} Vehicle boxes with 10-199 TOP points at 20-75 m')


def report(path, gate_dir):
    Wr = np.load(path)['rows']   # seg, range, sector, moving, n_top, rc_own, rc_med, nvis_own, nvis_med
    gd = Path(gate_dir)
    src = {}
    for depth in ('control', 'deep'):
        for plat in ('n008', 'n015'):
            R = np.load(gd / f'{depth}_{plat}.npz')['rows']   # rerender_gate rows: fi, r, sector, moving, n_src, then A (RC, CC, PPP, pts) ...
            src[(depth, plat)] = R
    mov = Wr[:, 3] > 0
    print(f'Waymo val, frame 50 of {int(np.load(path)["n_segments"])} segments: {len(Wr)} Vehicle boxes (10-199 TOP points, 20-75 m), '
          f'{int(mov.sum())} in moving-ego frames\n')
    print('| ring | sector | Waymo 10-49: boxes / RC own / RC median | Waymo 50-199: boxes / RC own | Boston control / deep RC (A) | '
          'Singapore control / deep RC (A) |')
    print('|---|---|---|---|---|---|')
    c1 = c2 = 0; comps = 0; small = []
    for lo, hi in RINGS:
        for sec in range(3):
            w = Wr[mov & (Wr[:, 1] >= lo) & (Wr[:, 1] < hi) & (Wr[:, 2] == sec)]
            w1 = w[(w[:, 4] >= 10) & (w[:, 4] < 50)]; w2 = w[(w[:, 4] >= 50) & (w[:, 4] < 200)]
            rc_w = np.nanmedian(w1[:, 5]) if len(w1) else np.nan
            if len(w1) < 15:
                small.append(f'{lo}-{hi} {SECTORS[sec]} ({len(w1)})')
            cells = []
            for plat in ('n008', 'n015'):
                vals = []
                for depth in ('control', 'deep'):
                    R = src[(depth, plat)]
                    q = R[(R[:, 3] > 0) & (R[:, 4] >= 10) & (R[:, 4] < 50) & (R[:, 1] >= lo) & (R[:, 1] < hi) & (R[:, 2] == sec)]
                    vals.append(np.median(q[:, 5]) if len(q) else np.nan)
                cells.append(f'{vals[0]:.2f} / {vals[1]:.2f}')
                comps += 1
                c1 += bool(rc_w <= vals[1] + 0.10); c2 += bool(rc_w >= 0.8)
            print(f'| {lo}-{hi} | {SECTORS[sec]} | {len(w1)} / {rc_w:.2f} / {np.nanmedian(w1[:, 6]) if len(w1) else np.nan:.2f} | '
                  f'{len(w2)} / {np.nanmedian(w2[:, 5]) if len(w2) else np.nan:.2f} | ' + ' | '.join(cells) + ' |')
    print(f'\nCriterion 1 (Waymo RC own <= deep rendered RC + 0.10): {c1} of {comps}; criterion 2 (Waymo RC own >= 0.8): {c2} of {comps}')
    print(f'Waymo cells under 15 boxes (10-49 points): {small or "none"}')
    verdict = ('1: the deep render already matches Waymo; the 0.8 bar was uncalibrated' if c1 >= 10 else
               '2: the rows are truly missing' if c2 >= 10 else '3: neither - per cell, no decision')
    print(f'Reading: {verdict}')
    print(f'All moving-ego 10-49-point Waymo cars, 20-75 m: median RC own {np.nanmedian(Wr[mov & (Wr[:, 4] < 50), 5]):.2f}, '
          f'median {np.nanmedian(Wr[mov & (Wr[:, 4] < 50), 6]):.2f}; 50-199: {np.nanmedian(Wr[mov & (Wr[:, 4] >= 50), 5]):.2f}')


if __name__ == '__main__':
    if sys.argv[1] == 'measure':
        measure(sys.argv[2], *(int(a) for a in sys.argv[3:5]))
    else:
        report(sys.argv[2], sys.argv[3])
