"""Prerequisite counts for RR v5 arm (b'), teacher-tracked Waymo objects re-rendered at their far frames (experiments_md
20261018_03 §9; T05 §5, 20261018_04 §3). CPU, run from a git-archive snapshot.

1. `tracks` (label-free): associate teacher 27471's stored Car boxes (score >= 27704's Car SCORE_THRESH) across the
   consecutive SAMPLED frames of each Waymo segment in the world frame (ego poses from the infos): constant-velocity
   prediction, centre gate 2 m per sampled step of horizon (15 m per step for a track with one observation, which has no
   velocity yet), heading gate 30 deg modulo pi; greedy one-to-one by distance; a
   track survives up to GAP_MAX missed sampled frames. Per track: observations (range, frame), interior gap frames (position
   linearly interpolated between the bracketing observations) and a bounded tail after the last observation (constant
   velocity, up to TAIL_MAX sampled frames, inside 75.2 m). Counts near (<= 25 m) and far (>= 45 / >= 60 m) frames.
2. `tracks` on Waymo VAL with the teacher's own detections (27596's result.pkl, every 5th frame) and `--gt`: for frames
   reached by constant-velocity extrapolation from the last NEAR observation, centre / heading error against the GT
   Vehicle box with BEV IoU > 0.3, by gap length; the share without a match (labelled analysis).
3. `coverage` (TRAIN, a fixed-seed sample of near-and-far tracks): the track's teacher-box TOP points aggregated in the
   box frame (relocate_render.to_box_frame + VoxelAccumulator 0.02 m, as the RR donor builder), rendered at each far frame
   through the segment's own TOP lattice; per height band of the silhouette: covered share, real-return rate among covered
   and among uncovered pixels; the same covered share for a random nuScenes RR v2 donor placed in the same box.

    python analysis/rr_track_prereq.py tracks --split train --out <dir> [--workers 8]
    python analysis/rr_track_prereq.py tracks --split val --gt --out <dir>
    python analysis/rr_track_prereq.py coverage --out <dir> [--n_tracks 300]
    python analysis/rr_track_prereq.py report --out <dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS.parent, TOOLS, TOOLS / 'analysis'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _init_path  # noqa: F401,E402

PS_TRAIN = Path('/home/koyama/data/pseudo_labels/jguwly5u_ep20_waymo_train_int10_thr0.0001/ps_label_e0.pkl')
VAL_RESULT = Path('/home/koyama/code/ST3D/output/da-ieee-access/centerpoint-gblobs-sourceonly-nuscenes2waymo/'
                  '20260922_sourceonly/eval/epoch_20/val/gblobs_accum_legaldepth_on_waymo/result.pkl')
DATA = TOOLS.parent / 'data/waymo/waymo_processed_data'
TOP_CALIB = Path('/home/koyama/data/waymo_top_calib/top_calib.pkl')
DB_DIR = Path('/home/koyama/data/relocate_render_20261018/v2')
CAR_THRESH = 0.18            # 27704: cfg.SELF_TRAIN.SCORE_THRESH [0.18, 0.32, 0.38], Car first (error_27704_wymPL.txt)
NEAR_M, FAR_M, FAR2_M, R_MAX = 25.0, 45.0, 60.0, 75.2
GATE_M_PER_STEP, INIT_GATE_M_PER_STEP, HEAD_GATE_DEG, GAP_MAX, TAIL_MAX = 2.0, 15.0, 30.0, 3, 5
STEP_S = {'train': 1.0, 'val': 0.5}
SEED = 20261018
BANDS = (0.0, 0.25, 0.5, 0.75, 1.0)
W_TOP = 2650


# ------------------------------------------------------------------------------------------------ geometry (tested)
def yaw_of(pose):
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def to_world(boxes, pose):
    """(x, y, z, dx, dy, dz, heading) vehicle-frame boxes -> world frame (centre through the 4x4 pose, heading + yaw)."""
    b = np.array(boxes, np.float64).reshape(-1, 7)
    c = np.c_[b[:, :3], np.ones(len(b))] @ np.asarray(pose, np.float64).T
    b[:, :3] = c[:, :3]; b[:, 6] += yaw_of(pose)
    return b


def to_vehicle(boxes, pose):
    b = np.array(boxes, np.float64).reshape(-1, 7)
    inv = np.linalg.inv(np.asarray(pose, np.float64))
    c = np.c_[b[:, :3], np.ones(len(b))] @ inv.T
    b[:, :3] = c[:, :3]; b[:, 6] -= yaw_of(pose)
    return b


def head_diff_mod_pi(a, b):
    d = np.mod(np.asarray(a) - np.asarray(b), np.pi)
    return np.minimum(d, np.pi - d)


def bev_iou(a, b):
    """BEV IoU of two (7,) boxes (shapely)."""
    from shapely.geometry import Polygon
    from box_pattern_visual import box_corners
    pa = Polygon(box_corners(a)[:4, :2]); pb = Polygon(box_corners(b)[:4, :2])
    u = pa.union(pb).area
    return pa.intersection(pb).area / u if u > 0 else 0.0


def associate(frames, gate=GATE_M_PER_STEP, head_gate_deg=HEAD_GATE_DEG, gap_max=GAP_MAX, init_gate=INIT_GATE_M_PER_STEP):
    """frames: list of (step index, world boxes (N, 7), extra (N, k)) in time order. Returns tracks: list of dicts with
    'steps', 'wboxes' (M, 7), 'extra' (M, k). Constant-velocity prediction from the last two observations; gate = `gate` x
    steps of horizon, or `init_gate` x steps for a track with ONE observation (no velocity yet; 15 m/s ~ 54 km/h);
    heading within head_gate_deg modulo pi; greedy one-to-one by distance."""
    tracks, alive = [], []
    for step, wb, ex in frames:
        wb = np.asarray(wb, np.float64).reshape(-1, 7)
        n = len(wb)
        cand = []
        for ti in alive:
            t = tracks[ti]
            h = step - t['steps'][-1]
            if h > gap_max + 1 or n == 0:
                continue
            last = t['wboxes'][-1]
            v = (last[:2] - t['wboxes'][-2][:2]) / (t['steps'][-1] - t['steps'][-2]) if len(t['steps']) > 1 else 0.0
            pred = last[:2] + v * h
            d = np.hypot(wb[:, 0] - pred[0], wb[:, 1] - pred[1])
            gm = (gate if len(t['steps']) > 1 else init_gate) * h
            ok = (d < gm) & (np.degrees(head_diff_mod_pi(wb[:, 6], last[6])) < head_gate_deg)
            for j in np.nonzero(ok)[0]:
                cand.append((d[j], ti, int(j)))
        used_t, used_b = set(), set()
        for d, ti, j in sorted(cand):
            if ti in used_t or j in used_b:
                continue
            used_t.add(ti); used_b.add(j)
            t = tracks[ti]
            t['steps'].append(step); t['wboxes'].append(wb[j]); t['extra'].append(ex[j])
        for j in range(n):
            if j not in used_b:
                tracks.append(dict(steps=[step], wboxes=[wb[j]], extra=[ex[j]]))
        alive = [ti for ti in range(len(tracks)) if step - tracks[ti]['steps'][-1] <= gap_max]
    for t in tracks:
        t['steps'] = np.asarray(t['steps']); t['wboxes'] = np.asarray(t['wboxes']); t['extra'] = np.asarray(t['extra'])
    return tracks


def interior_gaps(track):
    """(step, interpolated world box) for the sampled steps strictly between consecutive observations."""
    out = []
    s, w = track['steps'], track['wboxes']
    for k in range(len(s) - 1):
        for g in range(int(s[k]) + 1, int(s[k + 1])):
            f = (g - s[k]) / (s[k + 1] - s[k])
            b = w[k] * (1 - f) + w[k + 1] * f
            b[6] = w[k][6] + f * ((w[k + 1][6] - w[k][6] + np.pi) % (2 * np.pi) - np.pi)
            out.append((g, b))
    return out


def tail(track, n_steps, last_step):
    """(step, extrapolated world box) after the last observation, constant velocity, up to n_steps and last_step."""
    s, w = track['steps'], track['wboxes']
    if len(s) < 2:
        return []
    v = (w[-1][:2] - w[-2][:2]) / (s[-1] - s[-2])
    out = []
    for h in range(1, n_steps + 1):
        g = int(s[-1]) + h
        if g > last_step:
            break
        b = w[-1].copy(); b[:2] += v * h
        out.append((g, b))
    return out


def extrapolate_from(track, k, h):
    """World box h steps after observation k, constant velocity from observations k-1, k (None with k == 0)."""
    s, w = track['steps'], track['wboxes']
    if k == 0:
        return None
    v = (w[k][:2] - w[k - 1][:2]) / (s[k] - s[k - 1])
    b = w[k].copy(); b[:2] += v * h
    return b


# --------------------------------------------------------------------------------------------------------- data
G = {}


def load_pass(split):
    """{segment: sorted list of (sample_idx, (N, 9) boxes [box7, label, score])} of the teacher pass."""
    by_seg = {}
    if split == 'train':
        d = pickle.load(open(PS_TRAIN, 'rb'))
        for fid, v in d.items():
            seq, idx = fid[:-4], int(fid[-3:])
            by_seg.setdefault(seq, []).append((idx, np.asarray(v['gt_boxes'], np.float64).reshape(-1, 9)))
    else:
        r = pickle.load(open(VAL_RESULT, 'rb'))
        for x in r:
            seq, idx = x['frame_id'][:-4], int(x['frame_id'][-3:])
            lab = np.where(np.asarray(x['name']) == 'Car', 1, 0)
            by_seg.setdefault(seq, []).append((idx, np.c_[np.asarray(x['boxes_lidar'], np.float64).reshape(-1, 7), lab,
                                                           np.asarray(x['score'], np.float64)]))
    for seq in by_seg:
        by_seg[seq].sort(key=lambda t: t[0])
    return by_seg


def seg_infos(seq):
    p = pickle.load(open(glob.glob(str(DATA / seq / '*.pkl'))[0], 'rb'))
    return {int(i['point_cloud']['sample_idx']): i for i in p}


def seg_tracks(task):
    """One segment: tracks of Car boxes >= CAR_THRESH; per-track counts; optional GT extrapolation errors (val)."""
    seq, frames = task
    infos = seg_infos(seq)
    interval = int(np.min(np.diff([f[0] for f in frames]))) if len(frames) > 1 else 1
    steps = [(idx // interval, idx) for idx, _ in frames]
    flist = []
    for (st, idx), (_, bx) in zip(steps, frames):
        m = (bx[:, 7] == 1) & (bx[:, 8] >= CAR_THRESH)
        b = bx[m]
        pose = np.asarray(infos[idx]['pose'], np.float64)
        rng = np.hypot(b[:, 0], b[:, 1])
        flist.append((st, to_world(b[:, :7], pose), np.c_[np.full(len(b), idx), rng, b[:, 8], b[:, :7]]))
    tracks = associate(flist)
    last_step = steps[-1][0]
    pose_of = {st: np.asarray(infos[idx]['pose'], np.float64) for st, idx in steps}
    idx_of = {st: idx for st, idx in steps}
    rows, gt_rows, keep = [], [], []
    for ti, t in enumerate(tracks):
        rng = t['extra'][:, 1]
        gaps = interior_gaps(t); tl = tail(t, TAIL_MAX, last_step)
        gap_r = np.array([np.hypot(*to_vehicle(b, pose_of[g])[0, :2]) for g, b in gaps]) if gaps else np.zeros(0)
        tail_r = np.array([np.hypot(*to_vehicle(b, pose_of[g])[0, :2]) for g, b in tl]) if tl else np.zeros(0)
        tail_r = tail_r[tail_r <= R_MAX]
        rows.append([len(t['steps']), int(t['steps'][-1] - t['steps'][0] + 1), float(rng.min()), float(rng.max()),
                     int((rng <= NEAR_M).sum()), int((rng >= FAR_M).sum()), int((rng >= FAR2_M).sum()),
                     len(gaps), int((gap_r >= FAR_M).sum()), int((gap_r >= FAR2_M).sum()),
                     len(tail_r), int((tail_r >= FAR_M).sum()), int((tail_r >= FAR2_M).sum()), float(t['extra'][:, 2].mean())])
        if (rng <= NEAR_M).any() and ((rng >= FAR_M).any() or (gap_r >= FAR_M).any()):
            keep.append(dict(seq=seq, ti=ti, steps=t['steps'].tolist(), idx=[int(x) for x in t['extra'][:, 0]],
                             rng=rng.tolist(), vboxes=t['extra'][:, 3:10].tolist(), wboxes=t['wboxes'].tolist(),
                             gaps=[(int(g), int(idx_of[g]), to_vehicle(b, pose_of[g])[0].tolist()) for (g, b), r_ in
                                   zip(gaps, gap_r) if r_ >= FAR_M]))
        if G.get('gt') and (rng <= NEAR_M).any():
            k = int(np.nonzero(rng <= NEAR_M)[0][-1])            # the last near observation
            det_steps = set(t['steps'].tolist())
            for h in range(1, 21):
                g = int(t['steps'][k]) + h
                if g > last_step or g not in pose_of:
                    break
                wb = extrapolate_from(t, k, h)
                if wb is None:
                    break
                vb = to_vehicle(wb, pose_of[g])[0]
                r_ = float(np.hypot(vb[0], vb[1]))
                if r_ > R_MAX:
                    break
                an = infos[idx_of[g]]['annos']; names = np.asarray(an['name'])
                gv = np.asarray(an['gt_boxes_lidar'], np.float64).reshape(len(names), -1)[:, :7][names == 'Vehicle'] \
                    if len(names) else np.zeros((0, 7))
                best, bi = 0.0, -1
                if len(gv):
                    near = np.hypot(gv[:, 0] - vb[0], gv[:, 1] - vb[1]) < 8.0
                    for j in np.nonzero(near)[0]:
                        io = bev_iou(vb, gv[j])
                        if io > best:
                            best, bi = io, j
                ce = float(np.hypot(gv[bi, 0] - vb[0], gv[bi, 1] - vb[1])) if bi >= 0 else np.nan
                he = float(np.degrees(head_diff_mod_pi(gv[bi, 6], vb[6]))) if bi >= 0 else np.nan
                gt_rows.append([h, r_, best, ce, he, float(g in det_steps), float(rng[k])])
    return seq, np.array(rows, np.float64).reshape(-1, 14), keep, np.array(gt_rows, np.float64).reshape(-1, 7)


TRACK_COLS = ('n_obs', 'span_steps', 'r_min', 'r_max', 'n_near', 'n_far45', 'n_far60', 'n_gap', 'n_gap45', 'n_gap60',
              'n_tail', 'n_tail45', 'n_tail60', 'score_mean')
GT_COLS = ('gap_steps', 'r_pred', 'best_iou', 'centre_err', 'heading_err', 'teacher_detected', 'r_last_near')


def main_tracks(a):
    by_seg = load_pass(a.split)
    G['gt'] = bool(a.gt)
    tasks = sorted(by_seg.items())
    if a.max_segments:
        tasks = tasks[:a.max_segments]
    print(f'[tracks] {a.split}: {len(tasks)} segments, {sum(len(f) for _, f in tasks)} sampled frames, Car >= {CAR_THRESH}',
          flush=True)
    t0 = time.time(); rows, keep, gt, per_seg = [], [], [], []
    with Pool(a.workers) as pool:
        for n, (seq, r, k, g) in enumerate(pool.imap_unordered(seg_tracks, tasks, chunksize=2)):
            rows.append(r); keep += k; gt.append(g); per_seg.append([len(r), len(k)])
            if n % 100 == 0:
                print(f'[tracks] {n + 1}/{len(tasks)} segments, {time.time() - t0:.0f} s', flush=True)
    rows = np.concatenate(rows); gt = np.concatenate(gt)
    np.savez(Path(a.out) / f'tracks_{a.split}.npz', rows=rows, cols=np.array(TRACK_COLS), per_seg=np.array(per_seg),
             gt=gt, gt_cols=np.array(GT_COLS), step_s=STEP_S[a.split], n_segments=len(tasks))
    pickle.dump(keep, open(Path(a.out) / f'nearfar_tracks_{a.split}.pkl', 'wb'))
    print(f'[tracks] {len(rows)} tracks, {len(keep)} near-and-far, {len(gt)} extrapolated val frames, '
          f'{time.time() - t0:.0f} s', flush=True)


# ------------------------------------------------------------------------------------------------- coverage (3)
def top_block(seq, idx, infos):
    raw = np.load(DATA / seq / ('%04d.npy' % idx))
    n_top = int(infos[idx]['num_points_of_each_lidar'][0])
    return raw[:n_top], raw[:n_top, 5] == -1


def pixels_waymo(xyz, extrinsic, inc_desc):
    """(beam, col) of points by elevation about the TOP in its own axes (for synthetic points without a stored order)."""
    from pcdet.datasets.waymo.waymo_rings import top_columns_and_elevations
    colf, el = top_columns_and_elevations(np.asarray(xyz, np.float64), extrinsic)
    inc = np.asarray(inc_desc)
    k = np.clip(np.searchsorted(-inc, -el), 1, len(inc) - 1)
    beam = np.where(np.abs(el - inc[k - 1]) <= np.abs(el - inc[k]), k - 1, k)
    return beam * W_TOP + (np.floor(colf).astype(np.int64) % W_TOP)


def coverage_track(tr):
    """One near-and-far track: aggregate over its observations; render at each far frame (<= FAR_FRAMES_MAX)."""
    import torch
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.waymo.waymo_rings import top_beam_ids
    from box_pattern_visual import sensor_pose, silhouette_pixels
    torch.set_num_threads(1)
    seq = tr['seq']; infos = seg_infos(seq); cal = G['calib'][seq]
    E = np.asarray(cal['extrinsic'], np.float64); inc_desc = np.sort(np.asarray(cal['inclinations'], np.float64))[::-1]
    pose = sensor_pose('waymo', extrinsic=E)
    t0 = time.time()
    acc = rr.VoxelAccumulator(0.02, 4); n_obs_pts = 0
    sizes = np.asarray(tr['vboxes'])[:, 3:6]
    for idx, vb in zip(tr['idx'], tr['vboxes']):
        top, keep = top_block(seq, idx, infos)
        xyz = top[keep, :3].astype(np.float64)
        inb = rr.points_in_box(xyz, np.asarray(vb, np.float64))
        if inb.any():
            acc.add(np.c_[rr.to_box_frame(xyz[inb], vb), top[keep][inb, 3]]); n_obs_pts += int(inb.sum())
    local = acc.points()
    if len(local) < 50:
        return None
    size_med = np.median(sizes, 0)
    far = [(idx, np.asarray(vb), 1) for idx, vb, r_ in zip(tr['idx'], tr['vboxes'], tr['rng']) if r_ >= FAR_M]
    far += [(idx, np.r_[np.asarray(vb)[:3], size_med, np.asarray(vb)[6]], 0) for _, idx, vb in tr['gaps']]
    far = far[:G['far_max']]
    rng_d = np.random.default_rng(SEED + tr['ti'])
    don = int(rng_d.integers(G['db'].n))
    dl = G['db'].points(don)[:, :3]; dsize = G['db'].size[don]
    out = []
    for idx, fb, detected in far:
        top, keep = top_block(seq, idx, infos)
        beam, colf, _, _, _ = top_beam_ids(top[:, :3], E, cal['inclinations'])
        pix_all = beam * W_TOP + (np.floor(colf).astype(np.int64) % W_TOP)
        xyz = top[:, :3].astype(np.float64)
        test_box = fb.copy()
        if not detected:
            test_box[3:5] += 1.0                           # a predicted box: grown by 0.5 m on each horizontal side
        inb = keep & rr.points_in_box(xyz, test_box)
        real_pix = np.unique(pix_all[inb])
        sil, tin, tout, ent = silhouette_pixels(fb, pose, inc_desc, 'waymo', W_TOP, full=True)
        if not len(sil):
            continue
        agg = rr.from_box_frame(local, fb)
        agg_pix = np.unique(pixels_waymo(agg, E, inc_desc))
        dbox = np.r_[fb[:3], dsize, fb[6]]
        don_pix = np.unique(pixels_waymo(rr.from_box_frame(dl, dbox), E, inc_desc))
        dsil, _, _, dent = silhouette_pixels(dbox, pose, inc_desc, 'waymo', W_TOP, full=True)
        relh = np.clip((ent[:, 2] - (fb[2] - fb[5] / 2)) / fb[5], 0, 1); band = np.minimum((relh * 4).astype(int), 3)
        cov = np.isin(sil, agg_pix); ret = np.isin(sil, real_pix)
        drelh = np.clip((dent[:, 2] - (dbox[2] - dbox[5] / 2)) / dbox[5], 0, 1); dband = np.minimum((drelh * 4).astype(int), 3)
        dcov = np.isin(dsil, don_pix)
        r_ = float(np.hypot(fb[0], fb[1]))
        for k in range(4):
            m = band == k; dm = dband == k
            out.append([tr['ti'], r_, detected, k, m.sum(), (cov & m).sum(), (ret & m).sum(), (cov & ret & m).sum(),
                        dm.sum(), (dcov & dm).sum()])
    return dict(ti=tr['ti'], seq=seq, n_obs=len(tr['idx']), n_agg=len(local), n_raw=n_obs_pts, n_far=len(far),
                rows=np.array(out, np.float64).reshape(-1, 10), secs=time.time() - t0)


COV_COLS = ('ti', 'r', 'detected', 'band', 'n_sil', 'n_cov', 'n_ret', 'n_cov_ret', 'n_sil_donor', 'n_cov_donor')


def main_coverage(a):
    from pcdet.datasets.augmentor import relocate_render as rr
    keep = pickle.load(open(Path(a.out) / 'nearfar_tracks_train.pkl', 'rb'))
    rng = np.random.default_rng(SEED)
    sel = [keep[i] for i in rng.choice(len(keep), min(a.n_tracks, len(keep)), replace=False)]
    G.update(calib=pickle.load(open(TOP_CALIB, 'rb')), far_max=a.far_max,
             db=rr.DonorDB(DB_DIR / 'db_NUSCENES_N008_points.npy', DB_DIR / 'db_NUSCENES_N008_index.npz'))
    print(f'[coverage] {len(sel)} of {len(keep)} near-and-far tracks; donors {G["db"].n}', flush=True)
    t0 = time.time(); res = []
    with Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(coverage_track, sel, chunksize=1)):
            if r is not None:
                res.append(r)
            if n % 25 == 0:
                print(f'[coverage] {n + 1}/{len(sel)} tracks, {time.time() - t0:.0f} s', flush=True)
    rows = np.concatenate([r['rows'] for r in res])
    meta = np.array([[r['ti'], r['n_obs'], r['n_agg'], r['n_raw'], r['n_far'], r['secs']] for r in res])
    np.savez(Path(a.out) / 'coverage_train.npz', rows=rows, cols=np.array(COV_COLS), meta=meta, n_selected=len(sel),
             n_pool=len(keep), wall_s=time.time() - t0, workers=a.workers)
    print(f'[coverage] {len(res)} tracks with an aggregate, {len(rows) // 4} far frames, {time.time() - t0:.0f} s', flush=True)


# ---------------------------------------------------------------------------------------------------------- report
def _q(v, qs=(25, 50, 75)):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [round(float(x), 3) for x in np.percentile(v, qs)] if len(v) else [np.nan] * len(qs)


def main_report(a):
    out = Path(a.out); res = {}
    for split in ('train', 'val'):
        f = out / f'tracks_{split}.npz'
        if not f.exists():
            continue
        z = np.load(f); r = z['rows']; C = {k: j for j, k in enumerate(z['cols'])}
        near = r[:, C['n_near']] >= 1
        far_obs = r[:, C['n_far45']] >= 1; far_any = far_obs | (r[:, C['n_gap45']] >= 1)
        far60_obs = r[:, C['n_far60']] >= 1; far60_any = far60_obs | (r[:, C['n_gap60']] >= 1)
        nf = near & far_any
        d = dict(segments=int(z['n_segments']), step_s=float(z['step_s']), tracks=int(len(r)),
                 tracks_per_segment=_q(z['per_seg'][:, 0]), obs_per_track=_q(r[:, C['n_obs']]),
                 single_obs_share=round(float((r[:, C['n_obs']] == 1).mean()), 3),
                 tracks_near=int(near.sum()),
                 near_and_far45=dict(observed=int((near & far_obs).sum()), observed_or_gap=int(nf.sum()),
                                     per_segment=round(float(nf.sum() / z['n_segments']), 2)),
                 near_and_far60=dict(observed=int((near & far60_obs).sum()), observed_or_gap=int((near & far60_any).sum())),
                 far45_frames_of_those=dict(detected=int(r[nf, C['n_far45']].sum()), gap=int(r[nf, C['n_gap45']].sum()),
                                            tail=int(r[nf, C['n_tail45']].sum()),
                                            detected_per_track=_q(r[nf, C['n_far45']]), gap_per_track=_q(r[nf, C['n_gap45']])),
                 far60_frames_of_those=dict(detected=int(r[near & far60_any, C['n_far60']].sum()),
                                            gap=int(r[near & far60_any, C['n_gap60']].sum()),
                                            tail=int(r[near & far60_any, C['n_tail60']].sum())),
                 near_and_far45_with_a_far_gap=int((nf & (r[:, C['n_gap45']] >= 1)).sum()))
        g = z['gt']
        if len(g):
            GC = {k: j for j, k in enumerate(z['gt_cols'])}
            far = g[:, GC['r_pred']] >= FAR_M
            tab = {}
            for lab, m0 in (('r_pred >= 45 m', far), ('all ranges', np.ones(len(g), bool))):
                tab[lab] = {}
                for lo, hi in ((1, 2), (3, 4), (5, 8), (9, 12), (13, 20)):
                    m = m0 & (g[:, GC['gap_steps']] >= lo) & (g[:, GC['gap_steps']] <= hi)
                    if not m.any():
                        continue
                    mt = m & (g[:, GC['best_iou']] > 0.3)
                    tab[lab][f'{lo * z["step_s"]:g}-{hi * z["step_s"]:g} s'] = dict(
                        n=int(m.sum()), unmatched=round(float(1 - mt.sum() / m.sum()), 3),
                        teacher_detected=round(float(g[m, GC['teacher_detected']].mean()), 3),
                        centre_err_m=_q(g[mt, GC['centre_err']]), heading_err_deg=_q(g[mt, GC['heading_err']]),
                        unmatched_when_not_detected=round(float(1 - (mt & (g[:, GC['teacher_detected']] == 0)).sum() /
                                                                max((m & (g[:, GC['teacher_detected']] == 0)).sum(), 1)), 3))
            d['gt_extrapolation'] = tab
        res[split] = d
    f = out / 'coverage_train.npz'
    if f.exists():
        z = np.load(f); r = z['rows']; C = {k: j for j, k in enumerate(z['cols'])}; meta = z['meta']
        cov = {}
        for lab, m0 in (('all far frames', np.ones(len(r), bool)), ('detected', r[:, C['detected']] == 1),
                        ('gap (predicted box)', r[:, C['detected']] == 0)):
            cov[lab] = {}
            for k in range(4):
                m = m0 & (r[:, C['band']] == k)
                s = r[m].sum(0)
                cov[lab][f'{BANDS[k]:.2f}-{BANDS[k + 1]:.2f}'] = dict(
                    frames=int(m.sum()), covered=round(float(s[C['n_cov']] / max(s[C['n_sil']], 1)), 3),
                    return_rate_covered=round(float(s[C['n_cov_ret']] / max(s[C['n_cov']], 1)), 3),
                    return_rate_uncovered=round(float((s[C['n_ret']] - s[C['n_cov_ret']]) /
                                                      max(s[C['n_sil']] - s[C['n_cov']], 1)), 3),
                    nuscenes_donor_covered=round(float(s[C['n_cov_donor']] / max(s[C['n_sil_donor']], 1)), 3))
        res['coverage'] = dict(tracks=int(len(meta)), selected=int(z['n_selected']), pool=int(z['n_pool']),
                               far_frames=int(len(r) // 4), obs_per_track=_q(meta[:, 1]), agg_points=_q(meta[:, 2]),
                               raw_points=_q(meta[:, 3]), secs_per_track=_q(meta[:, 5]),
                               wall_s=float(z['wall_s']), workers=int(z['workers']), bands=cov)
    json.dump(res, open(out / 'prereq.json', 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['tracks', 'coverage', 'report'])
    ap.add_argument('--out', required=True); ap.add_argument('--split', default='train')
    ap.add_argument('--gt', action='store_true'); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_segments', type=int, default=0); ap.add_argument('--n_tracks', type=int, default=300)
    ap.add_argument('--far_max', type=int, default=10)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {'tracks': main_tracks, 'coverage': main_coverage, 'report': main_report}[a.mode](a)
