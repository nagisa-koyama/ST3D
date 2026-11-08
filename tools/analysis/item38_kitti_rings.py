"""Per-range Car AP and per-range density of the KITTI-target rows (experiments_md 20261003_05 §5 item 38). CPU only.

    python item38_kitti_rings.py manifest OUT.json      # resolve each row's result.pkl, W&B run and logged AP from its log
    python item38_kitti_rings.py ap MANIFEST OUT.json   # [0, inf) plus the rings, official evaluator (kitti_eval_cpu)
    python item38_kitti_rings.py density OUT.json       # median points per moderate-valid Car GT box per ring, per cut

`ap` scores [0, inf) through the same ring path first; a row whose pooled BEV moderate differs from its logged value
beyond the printed digit (0.0002) is flagged `reproduced: false` and must not be read. `density` cuts KITTI VAL clouds with each
training row's own frame-seeded `kitti_factor_swap.apply_points`, so the same boxes are counted under every cut.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
LOGS = TOOLS / 'logs'
RINGS = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 70.4)]   # item 38: 50-70.4 had 66 GT, merged
S2 = 'cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-%s.yaml'
CUTS = {'R1 (32 lines)': S2 % 'r1-lines', 'R3 (density)': S2 % 'r3-density',
        'P_K (lattice)': S2 % 'p-pattern', 'L_K (P_K count)': S2 % 'l-latcount'}

# (group, row, seed, job, substring of the "Result is save to" line to use; None = the job's last one)
ROWS = [
    ('swap', 'O', 666, 26843, None), ('swap', 'O', 667, 31817, None),
    ('swap', 'R1', 666, 31409, None), ('swap', 'R1', 667, 31946, None), ('swap', 'R1', 668, 32092, None),
    ('swap', 'R2', 666, 31410, None), ('swap', 'R2', 667, 31947, None),
    ('swap', 'R3', 666, 31414, None), ('swap', 'R3', 667, 31948, None), ('swap', 'R3', 668, 32099, None),
    ('swap', 'R4', 666, 31411, None), ('swap', 'R4', 667, 31949, None),
    ('swap', 'R5', 666, 31415, None),
    ('swap', 'R6', 666, 31416, None), ('swap', 'R6', 667, 31950, None),
    ('swap', 'R7', 666, 31417, None), ('swap', 'R7', 667, 32130, None),
    ('swap', 'R8', 666, 31412, None), ('swap', 'R9', 666, 31413, None),
    ('swap', 'R11', 666, 32141, None), ('swap', 'R11r', 666, 32143, None),
    ('swap', 'P_K', 666, 32283, None), ('swap', 'P_K', 667, 32284, None),
    ('swap', 'L_K', 666, 32289, None), ('swap', 'L_K', 667, 32290, None),
    ('transfer', 'control', 666, 32206, 'i33f_kitti_s666'), ('transfer', 'control', 667, 32206, 'i33f_kitti_s667'),
    ('transfer', 'control', 668, 32208, 'i33f_kitti_s668'),
    ('transfer', 'teacher', 666, 32253, 'i33f_teacher_s666_ep20'),
    ('transfer', 'teacher', 667, 32253, 'i33f_teacher_s667_ep20'),
] + [('own-cut 33d', f'{r} on 32-line val', s, 32203, f'i33d_{r}_s{s}')
     for r, seeds in (('R1', (666, 667, 668)), ('R2', (666, 667)), ('O', (666, 667))) for s in seeds] \
  + [('own-cut 33e', f'{r} on {cut} val', s, 32204, f'i33e_{tag}_{r}_s{s}')
     for cut, tag in (('own-density', 'density'), ('half-way', 'halfway'))
     for r, seeds in (('R3', (666, 667, 668)), ('O', (666, 667))) for s in seeds]


def log_lines(job):
    files = sorted(LOGS.glob(f'error_{job}_*.txt'))
    if not files:
        return None
    return files[0].read_bytes().decode('utf-8', 'replace').replace('\r', '\n').split('\n')


def resolve(group, row, seed, job, match):
    """The row's result.pkl, its W&B run and the logged moderate Car BEV / 3D AP of that evaluation."""
    out = {'group': group, 'row': row, 'seed': seed, 'job': job}
    lines = log_lines(job)
    if lines is None:
        return dict(out, status='no log')
    saves = [i for i, ln in enumerate(lines) if 'Result is save to' in ln and (match is None or match in ln)]
    if not saves:
        return dict(out, status='no evaluation in the log yet')
    i = saves[-1]
    res_dir = lines[i].split('Result is save to ', 1)[1].strip().replace('/storage/', '/home/koyama/data/')
    head = lines[:i]
    runs = [ln for ln in head if 'Run data is saved locally in' in ln]
    blocks = [k for k, ln in enumerate(head) if ln.startswith('Car AP_R40@0.70, 0.70, 0.70')]
    bev = float(head[blocks[-1] + 2].split(':')[1].split(',')[1]) if blocks else None
    d3 = float(head[blocks[-1] + 3].split(':')[1].split(',')[1]) if blocks else None
    epoch = re.search(r'epoch_(\d+)', res_dir)
    return dict(out, status='ok', result=str(Path(res_dir) / 'result.pkl'), epoch=int(epoch.group(1)) if epoch else None,
                wandb=runs[-1].rsplit('-', 1)[1].strip() if runs else None, logged_bev=bev, logged_3d=d3)


def cmd_manifest(args):
    rows = [resolve(*r) for r in ROWS]
    for r in rows:
        if r['status'] == 'ok' and not Path(r['result']).exists():
            r['status'] = 'result.pkl missing'
    json.dump(rows, open(args.out, 'w'), indent=1)
    for r in rows:
        print(r['group'], r['row'], r['seed'], r['job'], r['status'], r.get('wandb'), r.get('logged_bev'))


def _score(row):
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    import kitti_eval_cpu
    rings = kitti_eval_cpu.evaluate_rings(row['result'], [(0.0, float('inf'))] + RINGS)
    pooled = rings[0]
    ok = row['logged_bev'] is not None and abs(pooled['bev'] - row['logged_bev']) < 0.0002
    print(f"{row['row']} {row['seed']} pooled {pooled['bev']:.4f} logged {row['logged_bev']} -> {ok}", flush=True)
    return dict(row, pooled=pooled, rings=rings[1:], reproduced=bool(ok))


def cmd_ap(args):
    rows = [r for r in json.load(open(args.manifest)) if r['status'] == 'ok']
    with mp.get_context('spawn').Pool(args.workers) as pool:
        out = pool.map(_score, rows, chunksize=1)
    json.dump({'rings': RINGS, 'rows': out}, open(args.out, 'w'), indent=1)


def _density_frame(info):
    import _init_path  # noqa: F401
    import torch  # noqa: F401
    from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
    from pcdet.datasets.kitti import kitti_factor_swap
    import kitti_eval_cpu
    official = _density_frame.official
    fid = info['point_cloud']['lidar_idx']
    a = info['annos']
    n = len(a['gt_boxes_lidar'])
    empty = {'name': np.array([]), 'bbox': np.zeros((0, 4)), 'boxes_lidar': np.zeros((0, 7))}
    _, ignored, _, _ = official.clean_data(a, empty, 0, 1)
    valid = np.array(ignored[:n]) == 0
    if not valid.any():
        return {}
    boxes = a['gt_boxes_lidar'][valid].astype(np.float32)
    rng = np.hypot(boxes[:, 0], boxes[:, 1])
    pts = np.fromfile(kitti_eval_cpu.VELODYNE / f'{fid}.bin', dtype=np.float32).reshape(-1, 4)
    out = {'range': rng.tolist()}
    for name, swap in [('full KITTI', None)] + list(_density_frame.cuts.items()):
        p = pts if swap is None else kitti_factor_swap.apply_points(pts, swap, fid)
        out[name] = points_in_boxes_cpu(p[:, :3], boxes).sum(1).tolist()
    return out


def _density_init(cuts):
    import kitti_eval_cpu
    _density_frame.official = kitti_eval_cpu.load_official_eval()
    _density_frame.cuts = cuts


def cmd_density(args):
    import _init_path  # noqa: F401
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    import kitti_eval_cpu
    os.chdir(TOOLS)
    cuts = {}
    for name, path in CUTS.items():
        cfg = cfg_from_yaml_file(path, EasyDict())
        cuts[name] = cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP
    infos = pickle.load(open(kitti_eval_cpu.INFOS, 'rb'))
    with mp.get_context('spawn').Pool(args.workers, initializer=_density_init, initargs=(cuts,)) as pool:
        frames = [f for f in pool.imap(_density_frame, infos, chunksize=16) if f]
    rng = np.concatenate([f['range'] for f in frames])
    table = {}
    for name in ['full KITTI'] + list(cuts):
        counts = np.concatenate([f[name] for f in frames])
        table[name] = [{'lo': lo, 'hi': hi, 'n_boxes': int(m.sum()), 'median': float(np.median(counts[m])),
                        'mean': float(counts[m].mean()), 'empty_share': float((counts[m] == 0).mean())}
                       for lo, hi in RINGS for m in [(rng >= lo) & (rng < hi)]]
    json.dump({'rings': RINGS, 'cuts': {k: str(v) for k, v in cuts.items()}, 'table': table}, open(args.out, 'w'),
              indent=1)
    for name, rows in table.items():
        print(name, ' '.join(f"{r['lo']:g}-{r['hi']:g}: {r['median']:.0f}" for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('manifest'); m.add_argument('out')
    a = sub.add_parser('ap'); a.add_argument('manifest'); a.add_argument('out'); a.add_argument('--workers', type=int, default=8)
    d = sub.add_parser('density'); d.add_argument('out'); d.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    {'manifest': cmd_manifest, 'ap': cmd_ap, 'density': cmd_density}[args.cmd](args)


if __name__ == '__main__':
    main()
