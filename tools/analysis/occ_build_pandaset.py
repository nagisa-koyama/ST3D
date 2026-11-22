"""Build the three PandaSet occupancy GTs (G64, GGT, GU) for whole sequences (experiments_md 20261121_02 §6 steps 2-3).

Writes <out>/<seq>/<frame>.npz (g64_label, g64_unknown, ggt_label, ggt_unknown, gu_label, gu_unknown; uint8 / bool,
256 x 256 x 32, encoding in pcocc/__init__.py), <out>/<seq>/stats.json (per-frame diagnostics and seconds) and
<out>/manifest_<seq or tag>.json (code commit, pinned data hashes, grid, conventions). GT data: analysis / evaluation
only (uda_legality_rule.md). Run as a CPU Slurm job, never on the master node:

  NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK python3 tools/analysis/occ_build_pandaset.py --seqs 097 --out <dir>
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from pcseg.class_map import build_class_map                                            # noqa: E402
from pcseg.pandaset_seg import N_FRAMES, has_semseg                                    # noqa: E402
from pcocc.build import GTS, MOVER_SHELL, RANGE_CUT, SequenceGT                        # noqa: E402
from pcocc.grid import PANDASET_GRID                                                   # noqa: E402
from pcocc.scans import (CENSUS, CENSUS_SHA256, SENSOR_MODELS, SENSOR_MODELS_SHA256,   # noqa: E402
                         sensor_origins, sha256, stale_frames)

ROOT = '/home/koyama/data/pandaset/dataset'


def git(*args):
    return subprocess.run(['git', '-C', REPO, *args], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', nargs='*', type=int, default=None, help='output frames (default: all 80)')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cm = build_class_map('pandaset_scalr17')
    origins = sensor_origins()
    stale = stale_frames()
    head = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain', '--', 'pcocc', 'pcseg', 'tools/analysis/occ_build_pandaset.py')
    manifest = {
        'script': os.path.relpath(os.path.abspath(__file__), REPO), 'script_sha256': sha256(os.path.abspath(__file__)),
        'git_head': head, 'git_branch': git('rev-parse', '--abbrev-ref', 'HEAD'), 'dirty_paths': dirty.splitlines(),
        'root': ROOT, 'seqs': args.seqs, 'frames': args.frames, 'grid': PANDASET_GRID.to_dict(),
        'range_cut_m': list(RANGE_CUT), 'mover_shell_m': MOVER_SHELL, 'class_map': 'pandaset_scalr17',
        'class_names': cm.names,
        'encoding': {'empty': 0, 'class': 'index + 1', 'ignore_voxel': 255, 'unknown': 'separate bool mask'},
        'gts': {g: list(d) for g, d in GTS.items()},
        'data_files': {SENSOR_MODELS: SENSOR_MODELS_SHA256, CENSUS: CENSUS_SHA256},
        'sensor_origins_ego': {int(k): v.tolist() for k, v in origins.items()},
        'design': 'experiments_md episodes/20261121_02 §3-§5.1',
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'numba_threads': os.environ.get('NUMBA_NUM_THREADS'),
        'started': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
    }
    if dirty:
        print('WARNING: uncommitted changes in pcocc / pcseg / this script:\n' + dirty, flush=True)
    name = 'manifest_%s.json' % (args.tag or '_'.join(args.seqs))
    with open(os.path.join(args.out, name), 'w') as f:
        json.dump(manifest, f, indent=1)

    for seq in args.seqs:
        if not has_semseg(ROOT, seq):
            raise SystemExit('%s has no semseg' % seq)
        d = os.path.join(args.out, seq)
        os.makedirs(d, exist_ok=True)
        sg = SequenceGT(ROOT, seq, PANDASET_GRID, cm, origins, stale)
        frames = range(N_FRAMES) if args.frames is None else args.frames
        per = {}
        for t in frames:
            path = os.path.join(d, '%02d.npz' % t)
            if os.path.exists(path) and not args.overwrite:
                raise SystemExit('%s exists (pass --overwrite or use a new --out)' % path)
            out, st = sg.frame(t)
            arrays = {}
            for g, (label, unknown) in out.items():
                arrays[g + '_label'] = label
                arrays[g + '_unknown'] = unknown
            np.savez_compressed(path, **arrays)
            st['stale_pandargt_frame'] = t in sg.scans.stale_frames[1]
            st['bytes'] = os.path.getsize(path)
            per[t] = st
            print('%s/%02d: %.1f s, rays in grid %s, occupied %s, unknown cone %s, %d kB' % (
                seq, t, st['seconds'], [st.get('rays_%d' % k, {}).get('rays_in_grid') for k in (0, 1)],
                [st[g]['occupied'] for g in out], [st[g]['unknown_cone'] for g in out], st['bytes'] // 1000),
                flush=True)
        with open(os.path.join(d, 'stats.json'), 'w') as f:
            json.dump({'seq': seq, 'sequence': sg.stats, 'stale_frames': {k: sorted(v) for k, v in
                                                                          sg.scans.stale_frames.items()},
                       'frames': per}, f, indent=1)
        secs = [per[t]['seconds'] for t in per]
        print('%s done: load %.1f s, frames median %.1f s, max %.1f s, total %.1f min' % (
            seq, sg.stats['load_s'], np.median(secs), np.max(secs), (sg.stats['load_s'] + sum(secs)) / 60), flush=True)
    manifest['finished'] = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    with open(os.path.join(args.out, name), 'w') as f:
        json.dump(manifest, f, indent=1)


if __name__ == '__main__':
    main()
