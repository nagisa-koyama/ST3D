"""Waymo TOP lidar in its NATIVE range-image lattice (raw tfrecords, no tensorflow), for a target-pattern statistic
that does not depend on how the processed cloud was compensated (experiments_md 20261007_03 §12; user 2026-10-08:
pixel association depends on ego-motion compensation).

Per frame (TOP, return 1, 64 x 2650): (a) valid-pixel rate per beam row and valid pixels per range ring; (b) the
mis-assignment a compensated point would get if re-projected onto a static lattice: reconstruct each valid pixel
with the per-pixel pose (as the processed cloud is), take it back to the sensor frame with the static extrinsic, and
compare the row / column it lands in with its native row / column, per range ring and per column-time. Label-free
(sensor data only). Training segments.

    python analysis/waymo_native_lattice_stats.py <n_segments> <frames_per_segment> <out.npz>
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from waymo_reextract_bench import frames, parse_frame, rot_rpy  # noqa: E402

ROOT = Path('/home/koyama/code/ST3D/data/waymo/raw_data')
RINGS = [0, 10, 20, 30, 40, 50, 75]
n_seg, n_fr, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
infos_train = sorted(Path('/home/koyama/code/ST3D/data/waymo/ImageSets').glob('train.txt'))
segs = [l.strip().replace('.tfrecord', '') for l in open('/home/koyama/code/ST3D/data/waymo/ImageSets/train.txt')]
segs = segs[::max(1, len(segs) // n_seg)][:n_seg]

valid_row = np.zeros(64); seen_row = 0
ring_valid = np.zeros(len(RINGS) - 1)
dcol_by_ring = [[] for _ in range(len(RINGS) - 1)]; drow_by_ring = [[] for _ in range(len(RINGS) - 1)]
n_frames = 0
for s in segs:
    tf = ROOT / (s + '.tfrecord')
    if not tf.exists():
        continue
    allf = list(frames(tf))
    for buf in allf[::max(1, len(allf) // n_fr)][:n_fr]:
        pose, cal, lasers = parse_frame(buf)
        ri, rpose = lasers[1]
        inc, lo, hi, E = cal[1]
        rng = ri[..., 0]; H, W = rng.shape
        inc_rows = inc[::-1]
        valid = rng > 0
        valid_row += valid.mean(1); seen_row += 1
        az_col = ((np.arange(W, 0, -1) - 0.5) / W * 2 - 1) * np.pi - np.arctan2(E[1, 0], E[0, 0])
        INC, AZ = np.meshgrid(inc_rows, az_col, indexing='ij')
        rows, cols = np.nonzero(valid)
        r = rng[valid]; i = INC[valid]; a = AZ[valid]
        P = np.stack([r * np.cos(i) * np.cos(a), r * np.cos(i) * np.sin(a), r * np.sin(i)], -1) @ E[:3, :3].T + E[:3, 3]
        pp = rpose[valid]; R = rot_rpy(pp[:, 0], pp[:, 1], pp[:, 2])
        G = np.einsum('nij,nj->ni', R, P) + pp[:, 3:6]
        Finv = np.linalg.inv(pose); Pc = G @ Finv[:3, :3].T + Finv[:3, 3]          # compensated, vehicle frame at frame time
        Einv = np.linalg.inv(E); Ps = Pc @ Einv[:3, :3].T + Einv[:3, 3]             # back to the sensor frame (static)
        az2 = np.arctan2(Ps[:, 1], Ps[:, 0]); el2 = np.arctan2(Ps[:, 2], np.hypot(Ps[:, 0], Ps[:, 1]))
        yaw = np.arctan2(E[1, 0], E[0, 0])                                          # inverse of reconstruct()'s column azimuth
        col2 = np.round(W - 0.5 - W * ((az2 + yaw) / np.pi + 1) / 2).astype(int) % W
        dcol = (col2 - cols + W // 2) % W - W // 2
        row2 = np.argmin(np.abs(el2[:, None] - inc_rows[None, :]), axis=1); drow = row2 - rows
        rr = np.hypot(Pc[:, 0], Pc[:, 1])
        for k in range(len(RINGS) - 1):
            m = (rr >= RINGS[k]) & (rr < RINGS[k + 1])
            ring_valid[k] += m.sum()
            sel = np.nonzero(m)[0][::20]
            dcol_by_ring[k].append(dcol[sel]); drow_by_ring[k].append(drow[sel])
        n_frames += 1
print(f'frames {n_frames} from {len(segs)} segments')
vr = valid_row / max(seen_row, 1)
print('valid-pixel rate by beam row (top -> bottom, groups of 8): ' + ' '.join('%.2f' % vr[k:k + 8].mean() for k in range(0, 64, 8)))
print('valid pixels per frame by ring: ' + ', '.join('%d-%d m %.0f' % (RINGS[k], RINGS[k + 1], ring_valid[k] / n_frames) for k in range(len(RINGS) - 1)))
print('| ring | |dcol| p50 / p90 / p99 (columns, 0.136 deg) | share |dcol| >= 1 | |drow| p50 / p90 | share |drow| >= 1 |')
print('|---|---|---|---|---|')
for k in range(len(RINGS) - 1):
    dc = np.abs(np.concatenate(dcol_by_ring[k])); dr = np.abs(np.concatenate(drow_by_ring[k]))
    if len(dc):
        print(f'| {RINGS[k]}-{RINGS[k+1]} m | {np.percentile(dc,50):.0f} / {np.percentile(dc,90):.0f} / {np.percentile(dc,99):.0f} | {(dc>=1).mean():.2f} | {np.percentile(dr,50):.0f} / {np.percentile(dr,90):.0f} | {(dr>=1).mean():.2f} |')
np.savez(out, valid_row=vr, ring_valid=ring_valid / max(n_frames, 1), inclinations=inc_rows)
