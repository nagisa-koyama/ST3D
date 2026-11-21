"""Label-free TRAIN-split statistics for the S3a factor-swap operators of P1 (experiments_md episodes/20261116_01
§8.9; pcseg/factor_ops.py). Reads only the lidar files (annotations/semseg is never opened) of every frame of the
config's TRAIN sequences that carry semseg (pcseg's train set), for BOTH devices of each frame, through pcseg's reader
transform (`factor_ops.points_from_lidar`, tested identical to `pandaset_seg.read_frame`'s points).

Per device it sums over frames:
  hist_density   in-cone points per operator range bin (detection's binning, 0..MAX_DIST in NUM_BINS uniform bins,
                 points beyond MAX_DIST in the last bin), the a1 magnitude;
  hist_int_cone  in-cone counts per (intensity ring, dequantised intensity level of NUM_LEVELS), the a2 / a4 magnitude;
  hist_int_all   the same over all azimuths (diagnostic: does the cone change a ring's distribution?);
  fine_*         1 m planar / 3D range histograms in the cone and over all azimuths, the last bin an overflow
                 (diagnostic, and the a6 share);
and per frame the point counts in the cone and beyond 200 m (Pandar64's spec range) and 300 m (PandarGT's).

Choices, fixed before any measurement (no target result exists for P1):
  - cone: EVAL.CONE of the config (60 deg at heading 0), engine.cone_mask's convention, as the evaluation;
  - density bins: 0..300 m in 1.5 m bins. 300 m is PandarGT's spec range, the larger of the two sensors' (Pandar64
    200 m), so both sensors' returns are binned; 1.5 m is the detection paper's resolution (75 m / 50 bins);
  - intensity rings: detection's 0/10/20/30/40/50/75 m, extended by 100/150/200/300 m because pcseg has no range crop;
    points beyond the last edge are counted in the last ring, as the map applies the last ring to them;
  - dequantisation of the integer intensities over one step (1/255) with a frame-keyed draw, as detection's fit.

Outputs <out>.npz (arrays) and <out>.json (summary and provenance: script and code sha256, branch, commit, frames,
date, job). Run as a Slurm CPU job:
    python3 analysis/pcseg_factor_stats.py --cfg cfgs/pcseg/spunet-pandaset64.yaml \
        --out /home/koyama/data/pcseg/factor_stats/p1_train_stats --workers 4
"""
import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
sys.path.insert(0, REPO)

MAX_DIST = 300.0
NUM_BINS = 200
RING_EDGES = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0)
FINE_MAX = 600                    # 1 m bins 0..600 m, then one overflow bin
STEP = 1 / 255.0
SPEC_RANGE = {0: 200.0, 1: 300.0}  # Hesai spec ranges: Pandar64 200 m, PandarGT 300 m
DEVICES = (0, 1)
CODE_FILES = ('tools/analysis/pcseg_factor_stats.py', 'pcseg/factor_ops.py', 'pcseg/pandaset_seg.py', 'pcseg/engine.py',
              'pcdet/datasets/intensity_calibration.py')


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def git(*args):
    try:
        return subprocess.check_output(['git', '-C', REPO] + list(args), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def provenance(extra=None):
    out = {'date': datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
           'host': socket.gethostname(), 'slurm_job': os.environ.get('SLURM_JOB_ID'),
           'repo': REPO, 'branch': git('rev-parse', '--abbrev-ref', 'HEAD'), 'commit': git('rev-parse', 'HEAD'),
           'uncommitted': git('status', '--porcelain', '--', *CODE_FILES),
           'sha256': {f: sha256(os.path.join(REPO, f)) for f in CODE_FILES}}
    out.update(extra or {})
    return out


def frame_stats(lidar, pose, key, shift, cone):
    """One frame, both devices: a dict of arrays per device (counts are int64)."""
    from pcseg.factor_ops import points_from_lidar, planar_range, range_bins, cone_keep
    from pcseg.pandaset_seg import key_seed
    from pcdet.datasets.intensity_calibration import NUM_LEVELS, normalise
    edges = np.asarray(RING_EDGES)
    n_ring = len(edges) - 1
    out = {}
    for dev in DEVICES:
        pts, _ = points_from_lidar(lidar, dev, pose)
        coord = pts[:, :3] + shift                      # as PandasetSeg.load holds it
        r = planar_range(coord, shift)                  # the operators' range
        r3 = np.linalg.norm(pts[:, :3].astype(np.float64), axis=1)
        inc = cone_keep(coord, shift, *cone)
        rng = np.random.default_rng(key_seed('%s/d%d' % (key, dev), 'factor_stats/intensity_fit'))
        lvl = np.minimum((normalise(pts[:, 3], 1.0, STEP, rng) * NUM_LEVELS).astype(np.int64), NUM_LEVELS - 1)
        ring = np.clip(np.searchsorted(edges, r, side='right') - 1, 0, n_ring - 1)
        cell = ring * NUM_LEVELS + lvl
        fine = np.minimum(np.floor(r).astype(np.int64), FINE_MAX)
        fine3 = np.minimum(np.floor(r3).astype(np.int64), FINE_MAX)
        out[dev] = {
            'hist_density': np.bincount(range_bins(r[inc], MAX_DIST, NUM_BINS), minlength=NUM_BINS),
            'hist_int_cone': np.bincount(cell[inc], minlength=n_ring * NUM_LEVELS).reshape(n_ring, NUM_LEVELS),
            'hist_int_all': np.bincount(cell, minlength=n_ring * NUM_LEVELS).reshape(n_ring, NUM_LEVELS),
            'fine_planar_cone': np.bincount(fine[inc], minlength=FINE_MAX + 1),
            'fine_planar_all': np.bincount(fine, minlength=FINE_MAX + 1),
            'fine_3d_cone': np.bincount(fine3[inc], minlength=FINE_MAX + 1),
            'fine_3d_all': np.bincount(fine3, minlength=FINE_MAX + 1),
            'row': [len(r), int(inc.sum()), int((r[inc] >= 200.0).sum()), int((r3[inc] >= 200.0).sum()),
                    int((r[inc] >= 300.0).sum()), float(r[inc].max()) if inc.any() else 0.0],
        }
    return out


def sequence_stats(job):
    import pandas as pd
    from pcseg.pandaset_seg import N_FRAMES, load_poses, frame_key
    root, seq, shift, cone, limit = job
    poses = load_poses(root, seq)
    acc, rows = {}, []
    for frame in range(N_FRAMES if limit is None else min(limit, N_FRAMES)):
        key = frame_key(seq, frame)
        lidar = pd.read_pickle(os.path.join(root, seq, 'lidar', '%02d.pkl.gz' % frame))
        st = frame_stats(lidar, poses[frame], key, shift, cone)
        for dev, d in st.items():
            for k, v in d.items():
                if k == 'row':
                    rows.append([key, dev] + v)
                else:
                    acc[(dev, k)] = acc.get((dev, k), 0) + v
    return seq, acc, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='a P1 pcseg config (its ROOT, SPLITS.train, SHIFT_COOR, EVAL.CONE)')
    ap.add_argument('--out', required=True, help='output prefix; writes <out>.npz and <out>.json')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit_sequences', type=int, default=None, help='quick checks only')
    ap.add_argument('--limit_frames', type=int, default=None, help='quick checks only')
    a = ap.parse_args()
    os.chdir(TOOLS)
    import train_seg
    from pcseg.pandaset_seg import has_semseg
    cfg = train_seg.load_cfg(a.cfg)
    d = cfg.DATA
    shift = np.asarray(d.SHIFT_COOR, dtype=np.float32)
    cone = (float(cfg.EVAL.CONE.FOV_DEG), float(cfg.EVAL.CONE.get('HEADING_DEG', 0.0)))
    seqs = [s for s in d.SPLITS.train if has_semseg(d.ROOT, s)]
    if a.limit_sequences:
        seqs = seqs[:a.limit_sequences]
    jobs = [(d.ROOT, s, shift, cone, a.limit_frames) for s in seqs]
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.workers) as pool:
            results = pool.map(sequence_stats, jobs, chunksize=1)
    else:
        results = [sequence_stats(j) for j in jobs]
    acc, rows = {}, []
    for seq, ac, rw in results:                     # pool.map keeps the sequence order
        for k, v in ac.items():
            acc[k] = acc.get(k, 0) + v
        rows.extend(rw)
    n_frames = len(rows) // len(DEVICES)
    arrays = {'%s_d%d' % (k, dev): v for (dev, k), v in acc.items()}
    arrays.update(density_max_dist=MAX_DIST, density_num_bins=NUM_BINS, ring_edges=np.asarray(RING_EDGES),
                  intensity_step=STEP, frames=n_frames, cone=np.asarray(cone),
                  frame_keys=np.array([r[0] for r in rows if r[1] == 0]),
                  frame_rows=np.array([[r[1]] + r[2:] for r in rows], dtype=np.float64))
    summary = {'frames_per_device': n_frames, 'sequences': seqs, 'cfg': a.cfg, 'split': 'train',
               'labels_read': 'none (annotations/semseg never opened; has_semseg only checks the directory)',
               'cone': cone, 'shift_coor': shift.tolist(), 'seconds': time.time() - t0,
               'frame_rows_columns': ['device', 'points', 'in_cone', 'in_cone_planar_ge_200m', 'in_cone_3d_ge_200m',
                                      'in_cone_planar_ge_300m', 'in_cone_max_planar_m']}
    fr = arrays['frame_rows']
    for dev in DEVICES:
        m = fr[:, 0] == dev
        pts, cone_n = fr[m, 1].sum(), fr[m, 2].sum()
        share = fr[m, 3] / np.maximum(fr[m, 2], 1)
        summary['device_%d' % dev] = {
            'name': {0: 'Pandar64', 1: 'PandarGT'}[dev], 'points_per_frame': pts / n_frames,
            'in_cone_per_frame': cone_n / n_frames, 'in_cone_share': cone_n / pts,
            'in_cone_planar_ge_200m_share': fr[m, 3].sum() / cone_n, 'in_cone_3d_ge_200m_share': fr[m, 4].sum() / cone_n,
            'in_cone_planar_ge_300m_share': fr[m, 5].sum() / cone_n,
            'per_frame_share_ge_200m_p50_p99_max': [float(np.percentile(share, 50)), float(np.percentile(share, 99)),
                                                    float(share.max())],
            'in_cone_max_planar_m': float(fr[m, 6].max())}
    summary['provenance'] = provenance()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out + '.npz', provenance=json.dumps(summary), **arrays)
    summary['npz_sha256'] = sha256(a.out + '.npz')
    with open(a.out + '.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'sequences'}, indent=1))


if __name__ == '__main__':
    main()
