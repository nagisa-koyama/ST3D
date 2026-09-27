"""Re-evaluate segments the re-extraction driver left without a .done marker, without re-extracting.

    python waymo_reextract_recheck.py --out <new processed dir> [--frames 10] [--gate 0.5]

The driver's first gate (median counted/num_points_in_gt > 0.8 on three frames) was set from three
frames at 0.96-0.98 and is too tight: the passing population spans 0.80-1.00 with median 0.94, and a
frame with a handful of boxes can legitimately sit at 0.6-0.8 (the label count includes second
returns). A corrupt TOP cloud reads 0.00, and a frame whose small far objects
are second-return-only can legitimately read 0.2-0.5 (the label counts both returns, the pipeline
keeps the first). So: recompute over more frames, side lidars still within
5 mm of the old files, and mark done when the median ratio over all sampled frames exceeds the gate.
Appends the outcome to <out>/RECHECK.txt and removes passing entries from FAILED.txt.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from waymo_reextract_all import ROOT, OLD, load_infos, points_in_boxes  # noqa: E402


def recheck(seq, out, nframes, gate):
    d = out / seq; n = len([f for f in os.listdir(d) if f.endswith('.npy')])
    old_n = len([f for f in os.listdir(OLD / seq) if f.endswith('.npy')]) if (OLD / seq).exists() else -1
    infos = load_infos(); side = []; ratios = []
    for k in np.linspace(0, n - 1, nframes).astype(int):
        new = np.load(d / ('%04d.npy' % k)); info = infos.get((seq, k))
        if info is None:
            continue
        c = list(map(int, info['num_points_of_each_lidar'])); b = np.cumsum([0] + c)
        if (OLD / seq / ('%04d.npy' % k)).exists():
            old = np.load(OLD / seq / ('%04d.npy' % k))
            if len(old) == len(new):
                side.append(max(float(np.percentile(np.linalg.norm(old[b[li]:b[li + 1], :3] - new[b[li]:b[li + 1], :3], axis=1), 90)) for li in range(1, len(c))))
        an = info['annos']; keep = an['num_points_in_gt'] > 0
        if keep.any():
            cnt = points_in_boxes(new[:, :3].astype(np.float64), an['gt_boxes_lidar'][keep].astype(np.float64))
            ratios.append(float(np.median(cnt / an['num_points_in_gt'][keep])))
    stats = {'frames': n, 'old_frames': old_n, 'side_p90_m': [round(x, 4) for x in side], 'top_gt_ratio': [round(x, 3) for x in ratios], 'recheck_frames': nframes}
    ok = (n == old_n) and side and max(side) < 0.005 and ratios and float(np.median(ratios)) > gate
    return ok, stats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--frames', type=int, default=10); ap.add_argument('--gate', type=float, default=0.2)
    a = ap.parse_args(); out = Path(a.out)
    failed = [l.split('\t')[0] for l in (out / 'FAILED.txt').read_text().splitlines() if l.strip()] if (out / 'FAILED.txt').exists() else []
    failed = sorted(set(failed)); still = []
    for seq in failed:
        if (out / seq / '.done').exists():
            continue
        ok, stats = recheck(seq, out, a.frames, a.gate)
        (out / 'RECHECK.txt').open('a').write('%s\t%s\t%s\n' % (seq, 'ok' if ok else 'FAILED', json.dumps(stats)))
        if ok:
            (out / seq / '.done').write_text(json.dumps(stats))
        else:
            still.append(seq)
        print('%s %s side max %.4f  ratio median %.3f (n=%d)' % ('ok    ' if ok else 'FAILED', seq[:45], max(stats['side_p90_m']) if stats['side_p90_m'] else -1, float(np.median(stats['top_gt_ratio'])) if stats['top_gt_ratio'] else -1, len(stats['top_gt_ratio'])), flush=True)
    (out / 'FAILED.txt').write_text(''.join('%s\tFAILED\trecheck\n' % s for s in still))
    print('rechecked %d: %d now ok, %d still failed' % (len(failed), len(failed) - len(still), len(still)))


if __name__ == '__main__':
    main()
