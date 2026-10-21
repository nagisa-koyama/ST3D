"""Row extrusion (RX): the transplant library and q from teacher 27471's Waymo TRAIN pass, and the Waymo reference of the
gate (experiments_md 20261021_02; design T05 §5, commits 9a7c1f5 / b479ec7). CPU. No Waymo label is read.

Frames: the stored pass `ps_label_e0.pkl` of /home/koyama/data/pseudo_labels/jguwly5u_ep20_waymo_train_int10_thr0.0001
(27471 epoch 20 on Waymo TRAIN, every 10th frame, score floor 0.0001; boxes in the vehicle frame). Segments split as RR's
targets (20261018_02 §9.2): the sorted segment names permuted with numpy default_rng(0), the first round(0.8 n) the fit set,
the rest held out.

Per frame (the same code for both sets; geometry of 20261018_03 §7-8, box_pattern_visual imported):
- the segment's own TOP inclinations and extrinsic, TOP first returns, rows from the stored order (waymo_rings.top_beam_ids),
  the range-image columns; nearest return per pixel from every TOP return;
- cars = teacher Car, score >= 0.18, length dx < 6 m, BEV centre range hypot(x, y) in [15, 75) m (vehicle frame); per car the
  full and visible silhouette (visible_mask), row_records' rows (relh, visible pixels, hits; hits = pixels holding an in-box
  point of the evaluated cloud, no-label-zone points dropped), the relative view angle about the TOP origin;
- fit set (mode library): q's counts. A pixel of beam b (1-62) is a candidate if beams b - 1 and b + 1 of its column both
  return, with 3D ranges within 0.3 m, and both returns lie outside every teacher Car box with score >= 0.18 (any size,
  visibility) grown 0.5 m in BEV; it counts as returning if it holds a return; bin = the neighbours' mean horizontal range
  about the TOP (0-20 / 20-40 / 40-50 / 50-60 / 60-75 m);
- held-out set (mode reference): the cars (X1, X2) and X3's run-length histograms: object pixels (return > 0.3 m above S;
  S = the frame's RANSAC plane of the TOP returns plus the E5a offset of the teacher Car boxes >= 0.18) at 40-50 / 50-60 /
  60-75.2 m horizontal range about the TOP, inside a reference car / outside every teacher box of any class with score >= 0.18
  grown 0.5 m in BEV.
Library = the fit-set cars with visible / full silhouette >= 0.9, cell = range bin x aspect (side if |relative view angle| in
[45, 135] deg).

    python analysis/rx_build_library.py library   --out /home/koyama/data/row_extrusion_20261021 [--workers 8]
    python analysis/rx_build_library.py reference --out /home/koyama/data/row_extrusion_20261021 [--workers 8]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _init_path  # noqa: F401,E402

TOOLS = Path(__file__).resolve().parents[1]
PASS = Path('/home/koyama/data/pseudo_labels/jguwly5u_ep20_waymo_train_int10_thr0.0001/ps_label_e0.pkl')
CAR_THRESH = 0.18                 # 27704's Car SCORE_THRESH (rule B 0.17 raised to rule A's label-free NEG 0.180), as RR
LONG_M = 6.0
R_LO, R_HI = 15.0, 75.0
VIS_MIN = 0.9
GROW_M = 0.5
Q_DR = 0.3
X3_EDGES = (40.0, 50.0, 60.0, 75.2)
N_RUN = 65                        # run lengths 0..64
CAR_COLS = ('seg', 'frame', 'r', 'alpha_deg', 'n_full', 'n_vis', 'n_pts', 'row_start', 'row_count')
ROW_COLS = ('row', 'n_vis', 'n_hit', 'relh')


def split():
    """(sorted frame ids of the pass, {segment: id}, n_fit): RR's 80 / 20 split (rr_v5_fit.targets)."""
    ps = pickle.load(open(PASS, 'rb'))
    fids = sorted(ps)
    seqs = sorted({f.rsplit('_', 1)[0] for f in fids})
    perm = np.random.default_rng(0).permutation(len(seqs))
    n_fit = int(round(0.8 * len(seqs)))
    seg_id = {seqs[j]: i for i, j in enumerate(perm)}            # seg_id < n_fit: fit set
    return ps, fids, seg_id, n_fit


def x3_bin(rho):
    """[40, 50), [50, 60), [60, 75.2]; -1 outside."""
    r = np.asarray(rho, np.float64)
    out = np.full(r.shape, -1, np.int64)
    out[(r >= X3_EDGES[0]) & (r < X3_EDGES[1])] = 0
    out[(r >= X3_EDGES[1]) & (r < X3_EDGES[2])] = 1
    out[(r >= X3_EDGES[2]) & (r <= X3_EDGES[3])] = 2
    return out


def q_counts(has, rng, rho, ok_pix, dr=Q_DR):
    """q's counts on an (R, C) range image whose rows are adjacent beams: candidates = pixels of rows 1..R-2 whose upper
    and lower neighbours both return, within dr in range, and are both ok (outside the grown boxes); returns (num, den) per
    Q_BINS bin of the neighbours' mean horizontal range (unclamped: beyond 75 m not counted)."""
    from pcdet.datasets.augmentor import row_extrusion as RX
    has = np.asarray(has, bool)
    up, dn = slice(0, -2), slice(2, None)
    with np.errstate(invalid='ignore'):
        cand = has[up] & has[dn] & ok_pix[up] & ok_pix[dn] & (np.abs(rng[up] - rng[dn]) <= dr)
    rmean = (rho[up] + rho[dn]) / 2
    b = RX.bin_index(rmean[cand], RX.Q_BINS, clamp=False)
    ret = has[1:-1][cand]
    m = b >= 0
    nb = len(RX.Q_BINS)
    return (np.bincount(b[m], weights=ret[m].astype(np.float64), minlength=nb),
            np.bincount(b[m], minlength=nb).astype(np.float64))


def x3_hist(has, rng, obj, rho, inside, outside):
    """(3 bins, 2 [inside, outside], N_RUN) run-length histogram of the object pixels (X3)."""
    from pcdet.datasets.augmentor import row_extrusion as RX
    run = RX.run_lengths(has, rng)
    b = x3_bin(rho)
    out = np.zeros((3, 2, N_RUN), np.int64)
    for k, m in ((0, inside), (1, outside)):
        sel = obj & m & (b >= 0)
        np.add.at(out, (b[sel], k, np.minimum(run[sel], N_RUN - 1)), 1)
    return out


G = {}


def waymo_frame(task):
    """One teacher-pass frame (module docstring). Returns dict(cars (k, len(CAR_COLS) - 2), rows list, q (num, den) or
    x3 hist, flags)."""
    import torch
    from scipy.spatial import cKDTree
    from pcdet.datasets.waymo.waymo_rings import top_beam_ids
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.augmentor import row_extrusion as RX
    from source_coverage import in_box_indices
    import box_pattern_visual as BV
    torch.set_num_threads(1)
    fi, seq, sidx, n_top, boxes, cls, score, seg, mode = task
    raw = np.load(G['data_path'] / seq / ('%04d.npy' % sidx))
    cal = G['calib'][seq]
    top = raw[:n_top, :3].astype(np.float64)
    beam, colf, _, inc_desc, _ = top_beam_ids(raw[:n_top, :3], cal['extrinsic'], cal['inclinations'])
    W = BV.W_TOP; R = len(inc_desc)
    pix_all = beam * W + (np.floor(colf).astype(np.int64) % W)
    pose = BV.sensor_pose('waymo', extrinsic=cal['extrinsic'])
    o = pose['origin']
    rng_all = np.linalg.norm(top - o, axis=1)
    npp = BV.nearest_per_pixel(pix_all, rng_all)                  # occlusion from every TOP return (first returns)
    keep = raw[:n_top, 5] == -1                                    # evaluated cloud: no-label-zone points dropped
    P = np.ascontiguousarray(top[keep], np.float32); pk = pix_all[keep]; kidx = np.nonzero(keep)[0]
    tree = cKDTree(P[:, :2]) if len(P) else None
    boxes = np.asarray(boxes, np.float64).reshape(-1, 7)
    is_car = (np.asarray(cls) == 'Car') & (np.asarray(score) >= CAR_THRESH)
    r_box = np.hypot(boxes[:, 0], boxes[:, 1])
    sel = np.nonzero(is_car & (boxes[:, 3] < LONG_M) & (r_box >= R_LO) & (r_box < R_HI))[0]
    idx = in_box_indices(P, boxes[sel], tree) if len(sel) else []
    cars, rows = [], []
    for k, j in enumerate(sel):
        b = boxes[j]
        inbox = np.zeros(n_top, bool); inbox[kidx[idx[k]]] = True
        sil, tin, tout, ent = BV.silhouette_pixels(b, pose, inc_desc, 'waymo', W, full=True)
        n_full = len(np.unique(sil))
        if not n_full:
            continue
        vis = BV.visible_mask(sil, tin, npp, inbox)
        n_vis = len(np.unique(sil[vis]))
        if not n_vis:
            continue
        rr_ = BV.row_records(sil, tin, tout, ent, vis, pk[idx[k]], np.zeros(len(idx[k])), npp, b, W)
        cars.append([seg, fi, float(r_box[j]), BV.relative_view_angle_deg(b, o), n_full, n_vis, len(idx[k])])
        rows.append(rr_[:, [0, 1, 2, 3]].astype(np.float32))
    out = dict(cars=np.array(cars, np.float64).reshape(-1, len(CAR_COLS) - 2), rows=rows, no_plane=False)
    # range image (rows = beams, adjacent beams adjacent)
    up, uidx, urng = npp
    has = np.zeros(R * W, bool); has[up] = True; has = has.reshape(R, W)
    img_rng = np.full(R * W, np.nan); img_rng[up] = urng; img_rng = img_rng.reshape(R, W)
    img_pt = np.full(R * W, -1, np.int64); img_pt[up] = uidx; img_pt = img_pt.reshape(R, W)
    pts = top[np.maximum(img_pt, 0)]
    rho = np.where(has, np.hypot(pts[..., 0] - o[0], pts[..., 1] - o[1]), np.nan)
    if mode == 'library':
        car_b = boxes[is_car]
        grown = np.zeros((R, W), bool)
        for b in car_b:
            grown |= has & rr.points_in_rect(pts.reshape(-1, 3)[:, :2], b, GROW_M).reshape(R, W)
        out['q'] = q_counts(has, img_rng, rho, has & ~grown)
        return out
    # reference: X3
    plane = rr.ground_plane(top, RX.PLANE_RMAX)
    if plane is None:
        out['no_plane'] = True
        out['x3'] = np.zeros((3, 2, N_RUN), np.int64)
        return out
    S = RX.Surface(plane, boxes[is_car])
    flat = pts.reshape(-1, 3)
    s_val = np.full(R * W, np.nan)
    hf = has.ravel()
    s_val[hf] = S(flat[hf, 0], flat[hf, 1])
    with np.errstate(invalid='ignore'):
        obj = has & (flat[:, 2].reshape(R, W) > s_val.reshape(R, W) + RX.GROUND_TOL)
    inside = np.zeros(R * W, bool)
    if len(sel):
        inside[hf] = RX.points_in_boxes(flat[hf], boxes[sel]).any(1)
    inside = inside.reshape(R, W)
    any_b = np.asarray(score) >= CAR_THRESH
    grown = np.zeros(R * W, bool)
    for b in boxes[any_b]:
        grown[hf] |= rr.points_in_rect(flat[hf, :2], b, GROW_M)
    out['x3'] = x3_hist(has, img_rng, obj, rho, inside, has & ~grown.reshape(R, W))
    return out


def tasks_for(ps, fids, seg_id, n_fit, mode):
    infos = pickle.load(open(TOOLS.parent / 'data/waymo/waymo_infos_train.pkl', 'rb'))
    want = set(fids); meta = {}
    for inf in infos:
        pc = inf['point_cloud']; fid = pc['lidar_sequence'] + '_%03d' % pc['sample_idx']
        if fid in want:
            meta[fid] = (pc['lidar_sequence'], int(pc['sample_idx']), int(inf['num_points_of_each_lidar'][0]))
    del infos
    tasks, missing = [], 0
    for fi, f in enumerate(fids):
        seq = f.rsplit('_', 1)[0]
        is_fit = seg_id[seq] < n_fit
        if is_fit != (mode in ('library', 'reference_fit')):
            continue
        if f not in meta:
            missing += 1
            continue
        d = ps[f]
        gb = np.asarray(d['gt_boxes'], np.float64).reshape(-1, 9)
        tasks.append((fi, meta[f][0], meta[f][1], meta[f][2], gb[:, :7].copy(), np.asarray(d['teacher_classes']).copy(),
                      np.asarray(d['cls_scores'], np.float64).copy(), seg_id[seq],
                      'reference' if mode == 'reference_fit' else mode))
    return tasks, missing


def run(a, mode):
    from multiprocessing import Pool
    from pcdet.datasets.augmentor import row_extrusion as RX
    import box_pattern_visual as BV
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ps, fids, seg_id, n_fit = split()
    tasks, missing = tasks_for(ps, fids, seg_id, n_fit, mode)
    del ps
    tasks = tasks[:a.max_frames or None]
    G.update(data_path=TOOLS.parent / 'data/waymo/waymo_processed_data', calib=pickle.load(open(BV.TOP_CALIB, 'rb')))
    print(f'[rx {mode}] {len(fids)} pass frames, {len(seg_id)} segments ({n_fit} fit); {len(tasks)} frames in this set, '
          f'{missing} missing from the infos; {time.time() - t0:.0f} s', flush=True)
    cars, rows, qn, qd = [], [], np.zeros(len(RX.Q_BINS)), np.zeros(len(RX.Q_BINS))
    x3 = np.zeros((3, 2, N_RUN), np.int64); n_noplane = 0
    with Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap(waymo_frame, tasks, chunksize=4)):
            cars.append(res['cars']); rows += res['rows']
            if mode == 'library':
                qn += res['q'][0]; qd += res['q'][1]
            else:
                x3 += res['x3']; n_noplane += int(res['no_plane'])
            if k % 1000 == 0:
                print(f'[rx {mode}] {k + 1}/{len(tasks)} frames, {sum(len(c) for c in cars)} cars, '
                      f'{time.time() - t0:.0f} s', flush=True)
    C = np.concatenate(cars) if cars else np.zeros((0, len(CAR_COLS) - 2))
    cnt = np.array([len(r) for r in rows], np.int64)
    start = np.r_[0, np.cumsum(cnt)[:-1]].astype(np.int64)
    Rw = np.concatenate(rows) if rows else np.zeros((0, 4), np.float32)
    C = np.c_[C, start, cnt]
    meta = dict(pass_path=str(PASS), car_thresh=CAR_THRESH, n_fit_segments=n_fit, n_segments=len(seg_id),
                n_frames=len(tasks), car_cols=np.array(CAR_COLS), row_cols=np.array(ROW_COLS))
    if mode == 'library':
        q = np.where(qd > 0, qn / np.maximum(qd, 1), np.nan)
        vis = C[:, 5] / C[:, 4]
        L = np.nonzero(vis >= VIS_MIN)[0]
        cell = RX.bin_index(C[L, 2], RX.LIB_BINS) * 2 + RX.aspect_side(np.radians(C[L, 3])).astype(np.int64)
        rr_rel, rr_hit, rs, rc = [], [], [], []
        pos = 0
        for j in L:
            s, n = int(C[j, 7]), int(C[j, 8])
            rw = Rw[s:s + n]
            rr_rel.append(rw[:, 3]); rr_hit.append(rw[:, 2] / rw[:, 1]); rs.append(pos); rc.append(n); pos += n
        np.savez(out / 'library.npz', cell=cell, row_start=np.array(rs, np.int64), row_count=np.array(rc, np.int64),
                 row_relh=np.concatenate(rr_rel).astype(np.float64), row_hit_rate=np.concatenate(rr_hit).astype(np.float64),
                 q=q, q_num=qn, q_den=qd, lib_car=C[L], all_cars=C, all_rows=Rw, **meta)
        lib = RX.Library.from_npz(out / 'library.npz').check()
        print(f'[rx library] {len(C)} fit-set cars with a visible silhouette, library (visible >= {VIS_MIN}) {len(L)}; '
              f'per cell (range bin x [end, side]): ' + ', '.join(
                  f'{RX.LIB_BINS[c // 2][0]:g}-{RX.LIB_BINS[c // 2][1]:g} {"side" if c % 2 else "end"} {len(ix)}'
                  for c, ix in enumerate(lib.by_cell)), flush=True)
        print('[rx library] q per bin: ' + ', '.join(
            f'{lo:g}-{hi:g} m {q[k]:.4f} ({int(qn[k])} / {int(qd[k])})' for k, (lo, hi) in enumerate(RX.Q_BINS)), flush=True)
    else:
        np.savez(out / ('reference_fit.npz' if mode == 'reference_fit' else 'reference.npz'), cars=C, rows=Rw, x3=x3,
                 n_noplane=n_noplane, **meta)
        print(f'[rx reference] {len(C)} held-out cars with a visible silhouette; frames without a plane {n_noplane}; '
              f'X3 object pixels inside / outside per bin: '
              + ', '.join(f'{int(x3[b, 0].sum())} / {int(x3[b, 1].sum())}' for b in range(3)), flush=True)
    print(f'[rx {mode}] done in {time.time() - t0:.0f} s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('library', 'reference', 'reference_fit'),
                    help='reference_fit: the reference measurements on the FIT-set segments (experiments_md 20261021_02 §6)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_frames', type=int, default=0)
    a = ap.parse_args()
    run(a, a.mode)


if __name__ == '__main__':
    main()
