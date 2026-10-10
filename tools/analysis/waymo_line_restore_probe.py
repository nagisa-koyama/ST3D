"""How well do label-free baselines restore Waymo's held-out TOP scan lines from an HDL-32E-like subset of them?
(experiments_md 20261010_04, design study for a learned line completer; CPU feasibility probe, no training.)

Native TOP range images of Waymo TRAIN frames (raw tfrecords, pure-numpy reader of waymo_reextract_bench.py), first
return. Input = the TOP rows nearest to nuScenes' HDL-32E elevations (1.333-deg spacing, three lattice phases) inside
TOP's field of view - ~15 of 64 rows; every column kept (lines are the question; columns cost the oracles nothing,
20261010_01 §5.1). Every other row is held out and predicted per column from the kept rows only:
- NR   nearest kept row (in inclination), its range copied (no prediction where that cell has no return);
- LIN  linear in inclination between the kept rows below and above, both with a return (no fallback);
- RULE LIN gated by |r_above - r_below| <= 0.3 m (the re-render rule fill of 20261009_01 on this lattice).
Scored on the held-out cells (label-free): coverage of true returns, |dr|, and predictions in cells with no return.
ANALYSIS (reads Waymo TRAIN labels, never used to choose anything): per Vehicle box at 20-75 m with 10-49 / 50-199 TOP
points, the share of its true in-box rows present in the input, restored, and the predicted in-box rows that are
phantoms (no true in-box return in that row). Predicted points go through the same per-pixel-pose chain as truth.

    python analysis/waymo_line_restore_probe.py <out.npz> [n_segments=40] [frame=20]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import pickle
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import _init_path  # noqa
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from waymo_reextract_bench import frames, parse_frame, reconstruct

ROOT = Path('/home/koyama/code/ST3D/data/waymo')
HDL_DEG = -30.67 + 1.3333 * np.arange(32)
PHASES_DEG = (0.0, 0.444, 0.889)
DR_RULE = 0.3
BANDS = ((0, 20), (20, 40), (40, 75))
METHODS = ('NR', 'LIN', 'RULE')
CELL_STRIDE = 7


def kept_rows(inc_desc, phase_deg):
    el = np.radians(HDL_DEG + phase_deg)
    el = el[(el >= inc_desc.min() - 1e-6) & (el <= inc_desc.max() + 1e-6)]
    return np.unique(np.argmin(np.abs(inc_desc[:, None] - el[None, :]), 0))


def predict(rng, inc_desc, keep):
    """Per method, a range image holding predictions on held-out rows only (0 = no prediction)."""
    H, W = rng.shape
    held = np.setdiff1d(np.arange(H), keep)
    ke = inc_desc[keep]                      # descending, like the rows
    out = {m: np.zeros_like(rng) for m in METHODS}
    for i in held:
        e = inc_desc[i]
        up = keep[ke > e]; dn = keep[ke < e]                 # kept rows above / below in inclination
        a = up[np.argmin(ke[ke > e] - e)] if len(up) else None
        b = dn[np.argmin(e - ke[ke < e])] if len(dn) else None
        cand = [r for r in (a, b) if r is not None]
        nr = min(cand, key=lambda r: abs(inc_desc[r] - e))
        out['NR'][i] = rng[nr]
        if a is not None and b is not None:
            ra, rb = rng[a], rng[b]; ok = (ra > 0) & (rb > 0)
            w = (e - inc_desc[b]) / (inc_desc[a] - inc_desc[b])
            lin = np.where(ok, rb + w * (ra - rb), 0.0)
            out['LIN'][i] = lin
            out['RULE'][i] = np.where(ok & (np.abs(ra - rb) <= DR_RULE), lin, 0.0)
    return out, held


def frame_records(seg, frame_idx):
    infos = pickle.load(open(ROOT / 'waymo_processed_data' / seg / (seg + '.pkl'), 'rb'))
    info = [i for i in infos if i['point_cloud']['sample_idx'] == frame_idx]
    if not info:
        return None
    buf = None
    for k, b in enumerate(frames(ROOT / 'raw_data' / (seg + '.tfrecord'))):
        if k == frame_idx:
            buf = b; break
    if buf is None:
        return None
    pose, cal, lasers = parse_frame(buf)
    ri, rpose = lasers[1]; inc = np.asarray(cal[1][0], np.float64)
    inc_desc = np.sort(inc)[::-1]                             # row 0 = top beam, as reconstruct() orders it
    rng = ri[..., 0].astype(np.float64); valid = rng > 0
    a = info[0]['annos']; boxes = np.asarray(a['gt_boxes_lidar'], np.float64)[np.asarray(a['name']) == 'Vehicle']
    P, _, _, _, _ = reconstruct(ri, cal[1], rpose, pose)
    vr, vc = np.nonzero(valid)
    rr = np.hypot(boxes[:, 0], boxes[:, 1]) if len(boxes) else np.zeros(0)
    cand = np.nonzero((rr >= 20) & (rr < 75))[0]
    inb = (roiaware_pool3d_utils.points_in_boxes_cpu(P.astype(np.float32), boxes[cand, :7].astype(np.float32)) > 0
           if len(cand) else np.zeros((0, len(P)), bool))
    cars = []
    for jj in range(len(cand)):
        n = int(inb[jj].sum())
        if 10 <= n < 200:
            cars.append((cand[jj], n, rr[cand[jj]], np.unique(vr[inb[jj]])))
    cells, objs = [], []
    for ph in PHASES_DEG:
        keep = kept_rows(inc_desc, ph)
        pred, held = predict(rng, inc_desc, keep)
        hr = np.isin(np.arange(rng.shape[0]), held)[:, None] & np.ones_like(valid)
        sub = hr.copy(); sub[:, np.arange(rng.shape[1]) % CELL_STRIDE != 0] = False   # cell statistics on 1 column in 7
        t = rng[sub].astype(np.float32)
        for m in METHODS:
            p = pred[m][sub].astype(np.float32)
            cells.append((ph, METHODS.index(m), len(keep), t, p))
        if not cars:
            continue
        for m in METHODS:
            ri2 = ri.copy(); ri2[..., 0] = np.where(hr, pred[m], 0.0)
            P2, _, _, _, v2 = reconstruct(ri2, cal[1], rpose, pose)
            pr, pc = np.nonzero(v2)
            inb2 = roiaware_pool3d_utils.points_in_boxes_cpu(P2.astype(np.float32),
                                                             boxes[[c[0] for c in cars], :7].astype(np.float32)) > 0
            for jj, (j, n, r, true_rows) in enumerate(cars):
                kept_true = np.intersect1d(true_rows, keep)
                prow = np.unique(pr[inb2[jj]])
                hit = np.intersect1d(prow, true_rows)
                # point-level: predicted in-box points whose cell holds a TRUE in-box return
                pc_in = (pr[inb2[jj]], pc[inb2[jj]])
                true_cell = np.zeros(rng.shape, bool)
                tsel = inb[np.nonzero(cand == j)[0][0]]
                true_cell[vr[tsel], vc[tsel]] = True
                good_pts = true_cell[pc_in].sum(); all_pts = len(pc_in[0])
                objs.append((ph, METHODS.index(m), n, r, len(true_rows), len(kept_true), len(prow), len(hit),
                             all_pts, good_pts))
    return cells, objs


def summarise(cells, objs):
    print('HELD-OUT CELLS (label-free). coverage = share of true returns predicted; void = predictions in no-return cells')
    print('%-5s %-8s %8s %9s %9s %9s %9s %9s' % ('meth', 'band', 'n_true', 'coverage', 'med|dr|', '>0.5m', '>2m', 'void'))
    for mi, m in enumerate(METHODS):
        T = np.concatenate([c[3] for c in cells if c[1] == mi]); Pp = np.concatenate([c[4] for c in cells if c[1] == mi])
        void = ((T <= 0) & (Pp > 0)).sum() / max((Pp > 0).sum(), 1)
        for lo, hi in BANDS + ((0, 1e9),):
            s = (T > lo) & (T <= hi); cov = (Pp[s] > 0).mean(); d = np.abs(Pp[s] - T[s])[Pp[s] > 0]
            print('%-5s %-8s %8d %9.3f %9.3f %9.3f %9.3f %9s' % (m, f'{lo}-{hi}' if hi < 1e9 else 'all', s.sum(), cov,
                  np.median(d), (d > 0.5).mean(), (d > 2).mean(), '%.3f' % void if hi >= 1e9 else ''))
    if not len(objs):
        return
    O = np.array(objs, float)
    print('\nVEHICLES 20-75 m (ANALYSIS: Waymo TRAIN labels). Per car medians over cars x phases.')
    print('in-input = true in-box rows kept / true in-box rows; restored = (kept + predicted-in-box rows hitting a true')
    print('in-box row) / true in-box rows; phantom rows = predicted in-box rows with no true in-box return / predicted;')
    print('phantom pts = predicted in-box points in cells without a true in-box return / predicted in-box points.')
    print('%-5s %-8s %-6s %6s %9s %9s %9s %11s %11s %10s' % ('meth', 'pts', 'band', 'cars', 'truerows', 'in-input',
          'restored', 'phant.rows', 'phant.pts', 'no-kept'))
    for mi, m in enumerate(METHODS):
        for plo, phi in ((10, 50), (50, 200)):
            for lo, hi in ((20, 40), (40, 75)):
                s = (O[:, 1] == mi) & (O[:, 2] >= plo) & (O[:, 2] < phi) & (O[:, 3] >= lo) & (O[:, 3] < hi)
                if not s.any():
                    continue
                o = O[s]; tr = o[:, 4]; kin = o[:, 5] / tr; rest = (o[:, 5] + o[:, 7]) / tr
                with np.errstate(invalid='ignore', divide='ignore'):
                    prow = (o[:, 6] - o[:, 7]) / o[:, 6]; ppts = (o[:, 8] - o[:, 9]) / o[:, 8]
                print('%-5s %-8s %-6s %6d %9.1f %9.2f %9.2f %11.2f %11.2f %10.2f' % (
                    m, f'{plo}-{phi - 1}', f'{lo}-{hi}', s.sum() // len(PHASES_DEG), np.median(tr), np.median(kin),
                    np.median(rest), np.nanmedian(prow), np.nanmedian(ppts), (o[:, 5] == 0).mean()))


def main():
    out = sys.argv[1]; n_seg = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    segs = [s.strip().replace('.tfrecord', '') for s in open(ROOT / 'ImageSets' / 'train.txt') if s.strip()]
    segs = segs[::max(len(segs) // n_seg, 1)][:n_seg]
    cells, objs = [], []
    for k, s in enumerate(segs):
        r = frame_records(s, frame_idx)
        if r is None:
            continue
        cells += r[0]; objs += r[1]
        if (k + 1) % 10 == 0:
            print(f'  {k + 1} / {len(segs)} segments, {len(objs) // len(METHODS)} car-phases', flush=True)
    np.savez_compressed(out, objs=np.array(objs, float),
                        n_keep=np.array([c[2] for c in cells if c[1] == 0]))
    print('kept rows per phase: median %d of 64' % np.median([c[2] for c in cells if c[1] == 0]))
    summarise(cells, objs)


if __name__ == '__main__':
    main()
