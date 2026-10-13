"""G1''s reference, LABEL-FREE (experiments_md 20261010_04 §5.1, option A, pre-declared 2026-10-14 00:10 JST): Waymo's
own lattice-row coverage on its sparse far cars, measured as 20261009_01 §9 measured it, but with a legal teacher's
predicted Car boxes (27704, score >= 0.180) in place of GT, on Waymo TRAIN.

Frames: frame 50 of every 4th sequence of ImageSets/train.txt (raw tfrecords, pure-numpy reader). Moving ego: >= 1 m
between frames 40 and 50, from the infos' poses (the per-sequence infos are read for poses and frame ids only: their
`annos` are dropped on load). Boxes: predictions from analysis/predict_waymo_frames.py. Population: predicted Car boxes
with 10-49 TOP first-return points inside (50-199 reported), centre at 20-75 m planar range. Silhouette, occlusion
(0.5 m) and RC on the segment's OWN TOP inclinations: waymo_own_lattice_coverage.py's functions, unchanged.

    python analysis/waymo_pl_lattice_coverage.py measure <pred result.pkl> <out.npz> [min_score=0.18] [frame=50] [prev=40] [stride=4] [offset=0]
    python analysis/waymo_pl_lattice_coverage.py reference <out.npz> <ref.npz>
"""
import pickle
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import _init_path  # noqa
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from waymo_reextract_bench import frames, parse_frame, reconstruct
from waymo_own_lattice_coverage import silhouette_native, rc_on, RAW, RINGS
from lattice_depth_ceiling import sector_of, SECTORS

ROOT = Path('/home/koyama/code/ST3D/data/waymo')
PROCESSED = ROOT / 'waymo_processed_data'


def poses(seq):
    """{sample_idx: (frame_id, pose 4x4)} of one train sequence; labels are dropped on load."""
    with open(PROCESSED / seq / f'{seq}.pkl', 'rb') as f:
        infos = pickle.load(f)
    out = {}
    for i in infos:
        i.pop('annos', None)
        out[int(i['point_cloud']['sample_idx'])] = (i['frame_id'], np.asarray(i['pose'], np.float64))
    return out


def measure(pred_path, out, min_score=0.18, frame_idx=50, prev_idx=40, stride=4, offset=0):
    preds = {str(p['frame_id']): p for p in pickle.load(open(pred_path, 'rb'))}
    seqs = [s.strip().replace('.tfrecord', '') for s in open(ROOT / 'ImageSets' / 'train.txt') if s.strip()][offset::stride]
    recs, n_frames, n_moving, n_boxes = [], 0, 0, 0
    for si, seq in enumerate(seqs):
        ps = poses(seq)
        if frame_idx not in ps or prev_idx not in ps:
            continue
        fid, P50 = ps[frame_idx]; P40 = ps[prev_idx][1]
        p = preds.get(str(fid))
        if p is None:
            continue
        buf = None
        for k, b in enumerate(frames(RAW / (seq + '.tfrecord'))):
            if k == frame_idx:
                buf = b; break
        if buf is None:
            continue
        n_frames += 1
        pose, cal, lasers = parse_frame(buf)
        ri, rpose = lasers[1]; inc, lo, hi, E = cal[1]
        inc_desc = np.sort(np.asarray(inc, np.float64))[::-1]
        P, _, _, _, valid = reconstruct(ri, cal[1], rpose, pose)
        vr, vc = np.nonzero(valid)
        rng = np.where(valid, ri[..., 0], np.inf)
        d = P50[:3, :3].T @ (P50[:3, 3] - P40[:3, 3]); moving = np.hypot(d[0], d[1]) >= 1.0
        n_moving += int(moving)
        fwd = d[:2] / max(np.hypot(d[0], d[1]), 1e-9) if moving else np.array([1.0, 0.0])
        m = (np.asarray(p['name']) == 'Car') & (np.asarray(p['score']) >= min_score)
        boxes = np.asarray(p['boxes_lidar'], np.float64).reshape(-1, 7)[m]
        scores = np.asarray(p['score'])[m]
        n_boxes += len(boxes)
        if not len(boxes):
            continue
        rr = np.hypot(boxes[:, 0], boxes[:, 1]); cand = np.nonzero((rr >= 20) & (rr < 75))[0]
        if not len(cand):
            continue
        inb = roiaware_pool3d_utils.points_in_boxes_cpu(P.astype(np.float32), boxes[cand, :7].astype(np.float32)) > 0
        for jj, j in enumerate(cand):
            n_top = int(inb[jj].sum())
            if not 10 <= n_top < 200:
                continue
            inbox_img = np.zeros(rng.shape, bool); inbox_img[vr[inb[jj]], vc[inb[jj]]] = True
            rows, cols, tin = silhouette_native(boxes[j], inc_desc, E)
            rc, nv = rc_on(rows, cols, tin, rng, inbox_img, np.unique(vr[inb[jj]])) if len(rows) else (np.nan, 0)
            recs.append((si, rr[j], sector_of(boxes[j:j + 1, :2], fwd)[0], moving, n_top, rc, nv, scores[j]))
        if (si + 1) % 20 == 0:
            print(f'  {si + 1} / {len(seqs)} sequences, {len(recs)} boxes', flush=True)
    np.savez(out, rows=np.array(recs, float), sequences=len(seqs), frames=n_frames, moving_frames=n_moving,
             car_boxes=n_boxes, min_score=min_score, pred=str(pred_path), offset=offset, stride=stride)
    print(f'saved {out}: {n_frames} frames ({n_moving} moving), {n_boxes} Car boxes >= {min_score}, '
          f'{len(recs)} with 10-199 TOP points at 20-75 m')


def reference(path, ref):
    G = np.load(path); R = G['rows']   # seq, range, sector, moving, n_top, rc_own, nvis, score
    mov = R[:, 3] > 0
    cells, rows = [], []
    for lo, hi in RINGS:
        for s in range(3):
            w = R[mov & (R[:, 1] >= lo) & (R[:, 1] < hi) & (R[:, 2] == s)]
            w1 = w[(w[:, 4] >= 10) & (w[:, 4] < 50)]; w2 = w[(w[:, 4] >= 50) & (w[:, 4] < 200)]
            rc = float(np.nanmedian(w1[:, 5])) if len(w1) else np.nan
            cells.append((lo, s, rc)); rows.append((lo, hi, s, len(w1), rc, len(w2),
                                                    float(np.nanmedian(w2[:, 5])) if len(w2) else np.nan))
    off = int(G['offset']) if 'offset' in G.files else 0
    name = (f'Waymo own RC with 27704 Car boxes >= {float(G["min_score"]):.3f}, TRAIN train.txt[{off}::4] frame 50 of '
            f'{int(G["frames"])} sequences ({int(G["moving_frames"])} moving); label-free (20261010_04 §5.1 option A)')
    np.savez(ref, cells=np.array(cells, float), name=name, counts=np.array([r[3] for r in rows]))
    print(name)
    print('| ring | sector | boxes 10-49 | RC own (reference) | boxes 50-199 | RC own 50-199 |')
    print('|---|---|---|---|---|---|')
    for lo, hi, s, n1, rc, n2, rc2 in rows:
        flag = ' (flag: < 15 boxes)' if n1 < 15 else ''
        print(f'| {lo}-{hi} | {SECTORS[s]} | {n1}{flag} | {rc:.2f} | {n2} | {rc2:.2f} |')
    print(f'saved {ref}')


if __name__ == '__main__':
    if sys.argv[1] == 'measure':
        a = sys.argv[2:]
        measure(a[0], a[1], *(float(a[2]),) if len(a) > 2 else (), *(int(x) for x in a[3:7]))
    else:
        reference(sys.argv[2], sys.argv[3])
