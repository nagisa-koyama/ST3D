"""Label-free TRAIN intensity statistics of pair P2 and the two intensity tables of its method rows
(experiments_md episodes/20261128_01 §4, §8 CPU 2; pcseg/factor_ops.py INTENSITY).

Reads only the point files of every TRAIN frame of both datasets, never a label file:
  nuScenes       the keyframe LIDAR_TOP .pcd.bin of the index's train split (`nuscenes_seg.read_points`);
  SemanticKITTI  velodyne/<frame>.bin of the train sequences (`semantickitti_seg.read_points`).
Every point counts: both sensors are labelled and scored over 360 degrees, so there is no cone.

Per dataset it sums the histogram of (intensity ring, dequantised intensity level of NUM_LEVELS), with the P1 choices
fixed in 20261116_01 §8.9 and kept unchanged for P2:
  - rings by planar range about the sensor origin: 0/10/20/30/40/50/75/100/150/200/300 m (pcseg_factor_stats.RING_EDGES),
    the last ring taking every point beyond;
  - dequantisation over the stored step with a frame-keyed draw: 1/255 for nuScenes, 0.01 for SemanticKITTI.
It also checks the stored format on every frame: nuScenes intensities are integers / 255 in [0, 1], SemanticKITTI
remissions are multiples of 0.01 in [0, 1]. A frame that breaks it is counted and reported; tables are not written if
any frame does.

Tables (detection's construction, factor_ops.intensity_tables_from_histograms: quantiles at 201 levels, a ring with
< 2000 points on either side takes the all-range map), in the npz format factor_ops reads, with from_sensor / to_sensor:
  p2_intensity_nuscenes_to_semantickitti_rings.npz   M_ns->sk: nuScenes intensities -> SemanticKITTI's response;
  p2_intensity_semantickitti_to_nuscenes_rings.npz   M_sk->ns: the reverse.
Plus p2_train_intensity_stats.npz / .json (histograms, per-ring medians and counts, provenance).

    python3 analysis/pcseg_p2_intensity_tables.py --out_dir /home/koyama/data/pcseg/factor_stats --workers 8
Run as a Slurm CPU job from <worktree>/tools.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
sys.path.insert(0, REPO)
from analysis.pcseg_factor_stats import RING_EDGES, git, sha256  # noqa: E402

MIN_POINTS = 2000
CFGS = {'nuscenes': 'cfgs/pcseg/spunet-p2-nuscenes.yaml', 'semantickitti': 'cfgs/pcseg/spunet-p2-semantickitti.yaml'}
TABLES = {('nuscenes', 'semantickitti'): 'p2_intensity_nuscenes_to_semantickitti_rings.npz',
          ('semantickitti', 'nuscenes'): 'p2_intensity_semantickitti_to_nuscenes_rings.npz'}
CODE_FILES = ('tools/analysis/pcseg_p2_intensity_tables.py', 'tools/analysis/pcseg_factor_stats.py',
              'pcseg/factor_ops.py', 'pcseg/nuscenes_seg.py', 'pcseg/semantickitti_seg.py',
              'pcdet/datasets/intensity_calibration.py')


def provenance(extra=None):
    import datetime
    import socket
    out = {'date': datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
           'host': socket.gethostname(), 'slurm_job': os.environ.get('SLURM_JOB_ID'),
           'repo': REPO, 'branch': git('rev-parse', '--abbrev-ref', 'HEAD'), 'commit': git('rev-parse', 'HEAD'),
           'uncommitted': git('status', '--porcelain', '--', *CODE_FILES),
           'sha256': {f: sha256(os.path.join(REPO, f)) for f in CODE_FILES}}
    out.update(extra or {})
    return out


def format_ok(sensor, v):
    """The stored format: nuScenes integers / 255, SemanticKITTI multiples of 0.01; both in [0, 1]."""
    v = np.asarray(v, dtype=np.float64)
    scale = 255.0 if sensor == 'nuscenes' else 100.0
    return bool(np.all((v >= 0) & (v <= 1)) and np.all(np.abs(v * scale - np.round(v * scale)) < 1e-3))


def frame_hist(sensor, points, key, step):
    """(rings, NUM_LEVELS) counts of one frame's dequantised intensities."""
    from pcdet.datasets.intensity_calibration import NUM_LEVELS, normalise
    from pcseg.factor_ops import planar_range
    from pcseg.pandaset_seg import key_seed
    edges = np.asarray(RING_EDGES)
    n_ring = len(edges) - 1
    r = planar_range(points[:, :3], np.zeros(3, np.float32))
    rng = np.random.default_rng(key_seed('%s/%s' % (sensor, key), 'p2_stats/intensity_fit'))
    lvl = np.minimum((normalise(points[:, 3], 1.0, step, rng) * NUM_LEVELS).astype(np.int64), NUM_LEVELS - 1)
    ring = np.clip(np.searchsorted(edges, r, side='right') - 1, 0, n_ring - 1)
    return np.bincount(ring * NUM_LEVELS + lvl, minlength=n_ring * NUM_LEVELS).reshape(n_ring, NUM_LEVELS), r


def chunk_stats(job):
    sensor, root, items = job
    from pcseg import nuscenes_seg, semantickitti_seg
    step = nuscenes_seg.INTENSITY_STEP if sensor == 'nuscenes' else semantickitti_seg.INTENSITY_STEP
    acc, n, pts, bad, rmax = 0, 0, 0, [], 0.0
    for key, a, b in items:
        if sensor == 'nuscenes':
            p = nuscenes_seg.read_points(root, a)
        else:
            p = semantickitti_seg.read_points(root, a, b)
        if not format_ok(sensor, p[:, 3]):
            bad.append(key)
        h, r = frame_hist(sensor, p, key, step)
        acc = acc + h
        n += 1
        pts += len(p)
        rmax = max(rmax, float(r.max()) if len(r) else 0.0)
    return sensor, acc, n, pts, bad, rmax


def train_items(sensor, cfg, limit):
    d = cfg.DATA
    if sensor == 'nuscenes':
        from pcseg.nuscenes_seg import load_index
        rows = load_index(d.INDEX)['train']
        items = [(r[0], r[1], None) for r in rows]
        root = d.ROOT
    else:
        from pcseg.semantickitti_seg import frame_key, list_frames
        items = [(frame_key(s, f), s, f) for s in d.SPLITS.train for f in list_frames(d.ROOT, s)]
        root = d.ROOT
    return root, (items[:limit] if limit else items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit_frames', type=int, default=None, help='quick checks only: tables are not written')
    a = ap.parse_args()
    os.chdir(TOOLS)
    import train_seg
    from pcseg.factor_ops import intensity_tables_from_histograms
    from pcseg import nuscenes_seg, semantickitti_seg
    from pcdet.datasets.intensity_calibration import quantiles
    steps = {'nuscenes': nuscenes_seg.INTENSITY_STEP, 'semantickitti': semantickitti_seg.INTENSITY_STEP}
    t0 = time.time()
    jobs, n_items = [], {}
    for sensor, cfg_file in CFGS.items():
        cfg = train_seg.load_cfg(cfg_file)
        root, items = train_items(sensor, cfg, a.limit_frames)
        n_items[sensor] = len(items)
        size = max(1, len(items) // (4 * a.workers))
        jobs += [(sensor, root, items[i:i + size]) for i in range(0, len(items), size)]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.workers) as pool:
            results = pool.map(chunk_stats, jobs, chunksize=1)
    else:
        results = [chunk_stats(j) for j in jobs]
    hist = {s: 0 for s in CFGS}
    frames = {s: 0 for s in CFGS}
    points = {s: 0 for s in CFGS}
    bad = {s: [] for s in CFGS}
    rmax = {s: 0.0 for s in CFGS}
    for sensor, h, n, p, b, rm in results:
        hist[sensor] = hist[sensor] + h
        frames[sensor] += n
        points[sensor] += p
        bad[sensor] += b
        rmax[sensor] = max(rmax[sensor], rm)
    for s in CFGS:
        if frames[s] != n_items[s]:
            raise RuntimeError('%s: %d frames read of %d' % (s, frames[s], n_items[s]))
    edges = np.asarray(RING_EDGES, dtype=np.float64)

    def median(h):
        return float(quantiles(h, np.array([0.5]))[0]) if np.sum(h) else float('nan')

    os.makedirs(a.out_dir, exist_ok=True)
    tag = '' if not a.limit_frames else '_limit%d' % a.limit_frames
    summary = {
        'frames': frames, 'points': points, 'format_violations': {s: len(bad[s]) for s in CFGS},
        'format_violation_examples': {s: bad[s][:10] for s in CFGS}, 'max_planar_range_m': rmax,
        'steps': steps, 'ring_edges_m': edges.tolist(), 'labels_read': 'none (point files only)',
        'region': 'all points (360 degrees, no cone)', 'seconds': time.time() - t0,
        'rings': {s: [{'ring_m': [float(edges[k]), float(edges[k + 1])], 'points': int(hist[s][k].sum()),
                       'p50': median(hist[s][k])} for k in range(len(edges) - 1)] for s in CFGS},
        'all_range_p50': {s: median(hist[s].sum(axis=0)) for s in CFGS},
    }
    prov = provenance({'script': os.path.abspath(__file__), 'cfgs': CFGS, 'limit_frames': a.limit_frames})
    np.savez(os.path.join(a.out_dir, 'p2_train_intensity_stats%s.npz' % tag),
             hist_nuscenes=hist['nuscenes'], hist_semantickitti=hist['semantickitti'], ring_edges=edges,
             provenance=json.dumps(prov))
    if not a.limit_frames and not any(bad.values()):
        summary['tables'] = {}
        for (frm, to), name in TABLES.items():
            from_q, to_q, own = intensity_tables_from_histograms(hist[frm], hist[to], MIN_POINTS)
            path = os.path.join(a.out_dir, name)
            if os.path.exists(path):
                raise FileExistsError(path)
            np.savez(path, edges=edges, from_q=from_q, to_q=to_q, from_step=steps[frm], from_sensor=frm, to_sensor=to,
                     own=own, provenance=json.dumps(dict(prov, fitted_on='every TRAIN point of both datasets, '
                                                                        'labels never read')))
            summary['tables'][name] = {'sha256': sha256(path), 'own_map_rings': own.tolist(),
                                       'from_sensor': frm, 'to_sensor': to, 'from_step': steps[frm]}
    else:
        summary['tables'] = 'not written (%s)' % ('quick check' if a.limit_frames else 'format violations')
    summary['provenance'] = prov
    with open(os.path.join(a.out_dir, 'p2_train_intensity_stats%s.json' % tag), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ('frames', 'points', 'format_violations', 'all_range_p50', 'seconds')},
                     indent=1))
    for s in CFGS:
        print(s, [(r['ring_m'], r['points'], round(r['p50'], 4)) for r in summary['rings'][s]])
    print('tables:', summary['tables'] if isinstance(summary['tables'], str) else list(summary['tables']))


if __name__ == '__main__':
    main()
