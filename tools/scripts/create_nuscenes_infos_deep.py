"""Regenerate nuScenes infos at a deeper MAX_SWEEPS, without touching the existing ones.

WHY. A nuScenes info stores the metadata for max_sweeps-1 sweeps per keyframe, and the file is
named for that number - nuscenes_infos_<max_sweeps>sweeps_{train,val}.pkl. The repo ships
10sweeps, so MAX_SWEEPS cannot exceed 10: get_lidar_with_sweeps() indexes info['sweeps'][k] with
no clamp and raises IndexError beyond it. Deeper accumulation needs deeper infos.

BACKWARD COMPATIBLE BY CONSTRUCTION. The max_sweeps value is part of the filename, so this writes
new files beside the old ones and overwrites nothing. Every existing config keeps pointing at
nuscenes_infos_10sweeps_*.pkl and is unaffected. The script refuses to run if its output already
exists, and prints the pre-existing inventory so the before/after is on the record.

NOT the dataset __main__. That also calls create_groundtruth_database(max_sweeps=...), which loads
and concatenates max_sweeps point clouds per sample - at 200 that is ruinous, and gt_sampling is
disabled in every DA config here anyway.

CAVEAT ON DEPTH. fill_trainval_infos pads by repeating sweeps[-1] when a keyframe runs out of
history at the start of its scene, so a frame near a scene boundary gets DUPLICATED sweeps rather
than 200 distinct ones. A nuScenes scene is ~20 s at 20 Hz, i.e. ~400 frames, so most keyframes can
supply 199 genuinely - but early ones cannot, and their accumulation will contain repeats.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _init_path  # noqa: F401,E402  before pcdet, or it resolves to the stale /code/ST3D copy
from pcdet.datasets.nuscenes.nuscenes_dataset import create_nuscenes_info  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_sweeps', type=int, default=200)
    ap.add_argument('--version', type=str, default='v1.0-trainval')
    ap.add_argument('--force', action='store_true', help='overwrite an existing output')
    args = ap.parse_args()

    root = ROOT / 'data' / 'nuscenes'
    out_dir = root / args.version
    print('nuScenes root : %s' % root.resolve(), flush=True)

    print('\nexisting infos BEFORE this run (none of these are modified):', flush=True)
    for f in sorted(out_dir.glob('nuscenes_infos_*.pkl')):
        print('   %-48s %8.1f MB' % (f.name, f.stat().st_size / 1e6), flush=True)

    targets = [out_dir / ('nuscenes_infos_%dsweeps_%s.pkl' % (args.max_sweeps, s))
               for s in ('train', 'val')]
    clash = [t for t in targets if t.exists()]
    if clash and not args.force:
        print('\nREFUSING: %s already exists. Pass --force to overwrite.'
              % ', '.join(t.name for t in clash), flush=True)
        return 1

    print('\ncreating infos at max_sweeps=%d (%d sweeps stored per keyframe)'
          % (args.max_sweeps, args.max_sweeps - 1), flush=True)
    t0 = time.time()
    create_nuscenes_info(version=args.version, data_path=root, save_path=root,
                         max_sweeps=args.max_sweeps)
    print('\ndone in %.1f min' % ((time.time() - t0) / 60), flush=True)

    for t in targets:
        if not t.exists():
            print('   MISSING %s' % t.name, flush=True)
            continue
        with open(t, 'rb') as f:
            infos = pickle.load(f)
        n = [len(i['sweeps']) for i in infos[:200]]
        print('   %-48s %8.1f MB  %6d samples  sweeps/keyframe min %d max %d'
              % (t.name, t.stat().st_size / 1e6, len(infos), min(n), max(n)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
