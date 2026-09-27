"""Re-extract every Waymo segment with the validated numpy extractor, in parallel, resumably.

    python waymo_reextract_all.py --out <new processed dir> [--workers 16] [--segments N]

For each tfrecord under data/waymo/raw_data: skip if <out>/<segment>/.done exists; otherwise write
one .npy per frame (same layout as the 2023 files: x y z intensity elongation nlz, lidars in name
order, row-major), then VERIFY against the old processed directory before marking done:
  * frame count equal to the old segment's,
  * on the first, middle and last frame, every side lidar within 5 mm (p90) of the old file
    element-wise (the old side lidars are correct, so this checks the extractor bit for bit),
  * TOP not compared to the old file (the old TOP is the defect) - its check is num_points_in_gt,
    done once per segment on the same three frames against the infos, ratio must exceed 0.8.
A segment failing any check is left without a .done marker and listed in <out>/FAILED.txt.
Writes <out>/<segment>/.done with the stats. See experiments_md/20260927_05.
"""
import argparse, json, os, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from waymo_reextract_bench import frames, parse_frame, reconstruct  # noqa: E402

ROOT = Path('/home/koyama/code/ST3D/data/waymo')
OLD = ROOT / 'waymo_processed_data'
INFOS = None


def load_infos():
    global INFOS
    if INFOS is None:
        d = {}
        for split in ['train', 'val']:
            for i in pickle.load(open(ROOT / ('waymo_infos_%s.pkl' % split), 'rb')):
                d[(i['point_cloud']['lidar_sequence'], i['point_cloud']['sample_idx'])] = i
        INFOS = d
    return INFOS


def points_in_boxes(pts, boxes):
    """CPU box containment without the CUDA extension: rotate into each box frame."""
    out = np.zeros(len(boxes), dtype=np.int64)
    for k, b in enumerate(boxes):
        d = pts[:, :2] - b[:2]; c, s = np.cos(-b[6]), np.sin(-b[6])
        x = d[:, 0] * c - d[:, 1] * s; y = d[:, 0] * s + d[:, 1] * c
        m = (np.abs(x) <= b[3] / 2) & (np.abs(y) <= b[4] / 2) & (np.abs(pts[:, 2] - b[2]) <= b[5] / 2)
        out[k] = m.sum()
    return out


def do_segment(tf_path, out_root):
    seq = Path(tf_path).stem
    out = Path(out_root) / seq
    if (out / '.done').exists():
        return seq, 'skip', None
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); counts = []
    try:
        for k, buf in enumerate(frames(tf_path)):
            pose, cal, lasers = parse_frame(buf)
            parts = []; c = []
            for name in sorted(lasers):
                ri, rpose = lasers[name]
                P, inten, elong, nlz, _ = reconstruct(ri, cal[name], rpose if name == 1 else None, pose)
                parts.append(np.column_stack([P, inten, elong, nlz]).astype(np.float32)); c.append(len(P))
            np.save(out / ('%04d.npy' % k), np.concatenate(parts)); counts.append(c)
        n = len(counts)
        # ---- verification against the old files and the infos
        old_n = len([f for f in os.listdir(OLD / seq) if f.endswith('.npy')]) if (OLD / seq).exists() else -1
        stats = {'frames': n, 'old_frames': old_n, 'seconds': round(time.time() - t0, 1), 'side_p90_m': [], 'top_gt_ratio': []}
        ok = (old_n == n)
        infos = load_infos()
        for k in sorted({0, n // 2, n - 1}):
            new = np.load(out / ('%04d.npy' % k)); c = counts[k]; b = np.cumsum([0] + c)
            if (OLD / seq / ('%04d.npy' % k)).exists():
                old = np.load(OLD / seq / ('%04d.npy' % k))
                if len(old) != len(new):
                    ok = False; stats['side_p90_m'].append(None); continue
                p90 = max(float(np.percentile(np.linalg.norm(old[b[li]:b[li + 1], :3] - new[b[li]:b[li + 1], :3], axis=1), 90)) for li in range(1, len(c)))
                stats['side_p90_m'].append(round(p90, 4)); ok &= p90 < 0.005
            info = infos.get((seq, k))
            if info is not None and 'annos' in info and len(info['annos']['gt_boxes_lidar']):
                an = info['annos']; keep = an['num_points_in_gt'] > 0
                if keep.any():
                    cnt = points_in_boxes(new[:, :3], an['gt_boxes_lidar'][keep].astype(np.float64))
                    r = float(np.median(cnt / an['num_points_in_gt'][keep])); stats['top_gt_ratio'].append(round(r, 3)); ok &= r > 0.8
        if ok:
            (out / '.done').write_text(json.dumps(stats))
            return seq, 'ok', stats
        return seq, 'FAILED', stats
    except Exception as e:  # noqa: BLE001
        return seq, 'ERROR', str(e)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--segments', type=int, default=0); a = ap.parse_args()
    tfs = sorted(str(p) for p in (ROOT / 'raw_data').iterdir() if p.name.endswith('.tfrecord'))
    if a.segments: tfs = tfs[:a.segments]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tfs if not (out / Path(t).stem / '.done').exists()]
    print('%d segments, %d to do, %d workers' % (len(tfs), len(todo), a.workers), flush=True)
    t0 = time.time(); done = 0; failed = []
    with Pool(a.workers) as pool:
        for seq, status, stats in pool.imap_unordered(_worker, [(t, str(out)) for t in todo]):
            done += 1
            if status != 'ok':
                failed.append((seq, status, stats)); (out / 'FAILED.txt').open('a').write('%s\t%s\t%s\n' % (seq, status, stats))
            if done % 10 == 0 or status != 'ok':
                el = time.time() - t0
                print('[%5d/%d] %s %s %s  (%.0f s elapsed, %.1f h left)' % (done, len(todo), status, seq[:40], stats if status != 'ok' else '', el, el / done * (len(todo) - done) / 3600), flush=True)
    print('finished: %d ok, %d not ok, %.1f h' % (done - len(failed), len(failed), (time.time() - t0) / 3600), flush=True)


def _worker(args):
    return do_segment(*args)


if __name__ == '__main__':
    main()
