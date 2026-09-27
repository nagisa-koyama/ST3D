"""Rebuild the Waymo gt_database from a (re-extracted) processed directory, without a GPU.

    python waymo_rebuild_gt_database.py --processed <dir> --out_tag v2 [--workers 16] [--frames N]

Mirrors WaymoDataset.create_groundtruth_database exactly (see waymo_dataset.py): every
`sampled_interval`-th (10) entry of waymo_infos_train.pkl, points through get_lidar (NLZ points
dropped, intensity -> tanh), each GT box's points with the box centre subtracted, five float32
features per point, file '<sequence>_<sample:04d>_<name>_<i>.bin', and the same db_info dict.
Writes pcdet_gt_database_train_sampled_10_<tag>/ and pcdet_waymo_dbinfos_train_sampled_10_<tag>.pkl
next to the originals; swap names afterwards. Box containment is the CPU equivalent of
points_in_boxes_gpu (yaw-rotated axis-aligned test, half extents).
"""
import argparse, pickle, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path('/home/koyama/code/ST3D/data/waymo')
PROCESSED = None
OUT_DIR = None
USED_CLASSES = None


def get_lidar(seq, idx):
    pf = np.load(PROCESSED / seq / ('%04d.npy' % idx))
    pts, nlz = pf[:, 0:5], pf[:, 5]
    pts = pts[nlz == -1]
    pts[:, 3] = np.tanh(pts[:, 3])
    return pts


def box_index_of_points(pts, boxes):
    """-> index of the box each point falls in, -1 if none (first match wins, as the GPU kernel)."""
    idx = np.full(len(pts), -1, dtype=np.int64)
    for k, b in enumerate(boxes):
        d = pts[:, :2] - b[:2]; c, s = np.cos(-b[6]), np.sin(-b[6])
        x = d[:, 0] * c - d[:, 1] * s; y = d[:, 0] * s + d[:, 1] * c
        m = (idx < 0) & (np.abs(x) < b[3] / 2) & (np.abs(y) < b[4] / 2) & (np.abs(pts[:, 2] - b[2]) < b[5] / 2)
        idx[m] = k
    return idx


def do_frame(info):
    pc = info['point_cloud']; seq, idx = pc['lidar_sequence'], pc['sample_idx']
    pts = get_lidar(seq, idx)
    an = info['annos']; names, diff, boxes = an['name'], an['difficulty'], an['gt_boxes_lidar']
    bi = box_index_of_points(pts, boxes[:, :7].astype(np.float64))
    out = []
    for i in range(len(boxes)):
        if USED_CLASSES is not None and names[i] not in USED_CLASSES:   # as create_groundtruth_database(used_classes=CLASS_NAMES)
            continue
        gp = pts[bi == i].copy(); gp[:, :3] -= boxes[i, :3]
        fn = '%s_%04d_%s_%d.bin' % (seq, idx, names[i], i)
        gp.astype(np.float32).tofile(OUT_DIR / fn)
        out.append({'name': names[i], 'path': str((OUT_DIR / fn).relative_to(ROOT)), 'sequence_name': seq,
                    'sample_idx': idx, 'gt_idx': i, 'box3d_lidar': boxes[i], 'num_points_in_gt': gp.shape[0],
                    'difficulty': diff[i]})
    return out


def _init(processed, out_dir, classes):
    global PROCESSED, OUT_DIR, USED_CLASSES
    PROCESSED, OUT_DIR, USED_CLASSES = Path(processed), Path(out_dir), set(classes)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--processed', required=True); ap.add_argument('--out_tag', default='v2')
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--frames', type=int, default=0); ap.add_argument('--interval', type=int, default=10)
    ap.add_argument('--classes', nargs='+', default=['Vehicle', 'Pedestrian', 'Cyclist'])
    a = ap.parse_args()
    out_dir = ROOT / ('pcdet_gt_database_train_sampled_%d_%s' % (a.interval, a.out_tag)); out_dir.mkdir(exist_ok=True)
    infos = pickle.load(open(ROOT / 'waymo_infos_train.pkl', 'rb'))
    sel = infos[::a.interval]
    if a.frames: sel = sel[:a.frames]
    print('%d frames (interval %d), %d workers -> %s' % (len(sel), a.interval, a.workers, out_dir), flush=True)
    t0 = time.time(); db = {}
    with Pool(a.workers, initializer=_init, initargs=(a.processed, str(out_dir), a.classes)) as pool:
        for n, entries in enumerate(pool.imap(do_frame, sel, chunksize=8)):
            for e in entries: db.setdefault(e['name'], []).append(e)
            if (n + 1) % 1000 == 0: print('  %d/%d  %.0f s' % (n + 1, len(sel), time.time() - t0), flush=True)
    for k, v in db.items(): print('Database %s: %d' % (k, len(v)))
    pickle.dump(db, open(ROOT / ('pcdet_waymo_dbinfos_train_sampled_%d_%s.pkl' % (a.interval, a.out_tag)), 'wb'))
    print('done in %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
