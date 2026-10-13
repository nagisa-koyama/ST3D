"""Label-free ray-drop rate of Waymo TOP on above-ground surfaces, from unlabelled TRAIN range images (experiments_md
20261010_04 §5.1, pre-declared item 4). No label is read.

Frame 50 of every 4th TRAIN segment (`ImageSets/train.txt`), native TOP range image, first return (pure-numpy reader of
waymo_reextract_bench.py; points per-pixel-pose corrected into the vehicle frame, ground at z ~ 0). A cell (row r,
column c) is ELIGIBLE when both vertical neighbours (r - 1, c) and (r + 1, c) hold returns whose ranges agree within
0.3 m and whose points are both 0.3-3.0 m above the ground (vehicle-frame z). p(range band, elevation band) is the share
of eligible cells that hold no return. Range band: the neighbours' mean range, 0-20 / 20-40 / 40-75 m (beyond 75 m not
counted). Elevation band: the cell's own beam inclination below -3 deg, or -3 deg and above.

    python analysis/waymo_raydrop_rate.py <out.npz> [stride=4] [frame=50] [workers=8]
"""
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
from waymo_reextract_bench import frames, parse_frame, reconstruct

ROOT = Path('/home/koyama/code/ST3D/data/waymo')
RANGE_EDGES = np.array([0.0, 20.0, 40.0, 75.0])
ELEV_SPLIT_DEG = -3.0
DR_MAX, Z_LO, Z_HI = 0.3, 0.3, 3.0


def frame_counts(seg, frame_idx):
    """(eligible, empty) counts (3 range bands x 2 elevation bands) for one frame, or None if it is missing."""
    buf = None
    for k, b in enumerate(frames(ROOT / 'raw_data' / (seg + '.tfrecord'))):
        if k == frame_idx:
            buf = b; break
    if buf is None:
        return None
    pose, cal, lasers = parse_frame(buf)
    ri, rpose = lasers[1]
    inc = np.asarray(cal[1][0], np.float64)
    H, W = ri.shape[:2]
    inc_desc = np.sort(inc)[::-1] if len(inc) else ((0.5 + np.arange(H)) / H * (cal[1][2] - cal[1][1]) + cal[1][1])[::-1]
    P, _, _, _, valid = reconstruct(ri, cal[1], rpose, pose)
    rng = np.where(valid, ri[..., 0], np.nan)
    z = np.full((H, W), np.nan); z[valid] = P[:, 2]
    up, mid, dn = slice(0, H - 2), slice(1, H - 1), slice(2, H)
    nb = valid[up] & valid[dn]
    with np.errstate(invalid='ignore'):
        elig = (nb & (np.abs(rng[up] - rng[dn]) <= DR_MAX) & (z[up] >= Z_LO) & (z[up] <= Z_HI)
                & (z[dn] >= Z_LO) & (z[dn] <= Z_HI))
        mr = (rng[up] + rng[dn]) / 2
    elig &= mr < RANGE_EDGES[-1]
    rb = np.clip(np.searchsorted(RANGE_EDGES, np.nan_to_num(mr, nan=0.0), side='right') - 1, 0, len(RANGE_EDGES) - 2)
    eb = np.broadcast_to((np.degrees(inc_desc[1:H - 1]) >= ELEV_SPLIT_DEG)[:, None], elig.shape).astype(np.int64)
    empty = ~valid[mid]
    n_el = np.zeros((3, 2), np.int64); n_em = np.zeros((3, 2), np.int64)
    np.add.at(n_el, (rb[elig], eb[elig]), 1)
    np.add.at(n_em, (rb[elig & empty], eb[elig & empty]), 1)
    return n_el, n_em


def main():
    out = sys.argv[1]; stride = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    segs = [s.strip().replace('.tfrecord', '') for s in open(ROOT / 'ImageSets' / 'train.txt') if s.strip()][::stride]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    n_el = np.zeros((3, 2), np.int64); n_em = np.zeros((3, 2), np.int64); used = 0
    per_seg = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap(partial(frame_counts, frame_idx=frame_idx), segs)):
            if r is not None:
                n_el += r[0]; n_em += r[1]; used += 1
                per_seg.append(r[1].sum() / max(r[0].sum(), 1))
            if (i + 1) % 20 == 0:
                print(f'  {i + 1} / {len(segs)} segments', flush=True)
    rates = n_em / np.maximum(n_el, 1)
    np.savez(out, rates=rates, eligible=n_el, empty=n_em, range_edges=RANGE_EDGES,
             elev_split_rad=np.radians(ELEV_SPLIT_DEG), segments=used, frame=frame_idx, stride=stride,
             per_segment_rate=np.array(per_seg))
    print(f'Waymo TRAIN, frame {frame_idx} of {used} segments (every {stride}th): ray-drop on above-ground surfaces')
    print('| range band | elevation < -3 deg: eligible / empty / p | >= -3 deg: eligible / empty / p |')
    print('|---|---|---|')
    for b in range(3):
        print(f'| {RANGE_EDGES[b]:.0f}-{RANGE_EDGES[b + 1]:.0f} m | {n_el[b, 0]} / {n_em[b, 0]} / {rates[b, 0]:.4f} | '
              f'{n_el[b, 1]} / {n_em[b, 1]} / {rates[b, 1]:.4f} |')
    ps = np.array(per_seg)
    print(f'per-segment pooled rate: median {np.median(ps):.4f}, IQR {np.percentile(ps, 25):.4f}-{np.percentile(ps, 75):.4f}')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
