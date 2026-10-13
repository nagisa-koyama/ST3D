"""Build the whole-track Car aggregation cache (experiments_md 20261010_04 §5.1, item 6). CPU only; source labels only.

One pickle per nuScenes scene (`pcdet/datasets/nuscenes/track_aggregation.py`, `build_scene`): for every keyframe the
in-box points of each Car track (`gt_names == 'car'`, devkit `instance_token`) in box-local coordinates with intensity
and ring, and per (anchor keyframe, track) with >= 5 own points the accepted 4-DoF ICP corrections of every other
keyframe of that track. Plus `index.pkl` = {keyframe token: (scene token, position in the scene)} and `meta.json`.

    python analysis/build_track_agg_cache.py <out_dir> [workers=8] [max_scenes=0 (all)] [info=nuscenes_infos_200sweeps_train.pkl]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import json
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa
from pcdet.datasets.motion_compensation import DevkitSweepCompensator, find_devkit_meta_dir
from pcdet.datasets.nuscenes import track_aggregation as TA

ROOT = Path('/home/koyama/code/ST3D/data/nuscenes/v1.0-trainval')
G = {}


def read_points(token):
    """A keyframe's LIDAR_TOP cloud (x, y, z, intensity, ring), ego points within 1.5 m removed as the loader does."""
    p = np.fromfile(str(ROOT / G['tok2info'][token]['lidar_path']), dtype=np.float32).reshape(-1, 5)
    return p[~((np.abs(p[:, 0]) < 1.5) & (np.abs(p[:, 1]) < 1.5))]


def one_scene(scene):
    comp = G['comp']
    kfs = [{'token': G['frame2tok'][i], 't': comp.frames[i]['t'], 'boxes': comp.frames[i]['boxes'],
            'names': comp.frames[i]['names'], 'tids': comp.frames[i]['tids']} for i in comp.scene_order[scene]]
    t0 = time.time()
    sc = TA.build_scene(kfs, read_points)
    sc['scene'] = scene
    TA.save_scene(G['out'] / f'{scene}.pkl', sc)
    n_pts = sum(len(x) for f in sc['frames'] for x, _ in f['points'].values())
    n_tracks = len({k for f in sc['frames'] for k in f['tids']})
    return scene, [k['token'] for k in kfs], n_tracks, len(sc['icp']), int(sum(len(v) for v in sc['icp'].values())), \
        n_pts, time.time() - t0


def main():
    out = Path(sys.argv[1]); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    max_scenes = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    info_name = sys.argv[4] if len(sys.argv) > 4 else 'nuscenes_infos_200sweeps_train.pkl'
    out.mkdir(parents=True, exist_ok=True)
    infos = pickle.load(open(ROOT / info_name, 'rb'))
    comp = DevkitSweepCompensator(infos, find_devkit_meta_dir(ROOT, 'v1.0-trainval'), classes=None)
    G.update(comp=comp, out=out, tok2info={i['token']: i for i in infos},
             frame2tok={f: t for t, f in comp.tok2frame.items()})
    scenes = sorted(comp.scene_order)
    if max_scenes:
        scenes = scenes[::max(1, len(scenes) // max_scenes)][:max_scenes]
    print(f'{len(infos)} infos, {len(comp.frames)} keyframes, {len(comp.scene_order)} scenes; building {len(scenes)} '
          f'with {workers} workers -> {out}', flush=True)
    t0 = time.time(); index, stats = {}, []
    with Pool(workers) as pool:
        for n, (scene, toks, n_tr, n_icp, n_rows, n_pts, dt) in enumerate(pool.imap_unordered(one_scene, scenes)):
            index.update({t: (scene, p) for p, t in enumerate(toks)})
            stats.append((len(toks), n_tr, n_icp, n_rows, n_pts, dt))
            if (n + 1) % 25 == 0:
                print(f'  {n + 1} / {len(scenes)} scenes, {time.time() - t0:.0f} s', flush=True)
    with open(out / 'index.pkl', 'wb') as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    S = np.array(stats, float)
    sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', '../pcdet/datasets/nuscenes/track_aggregation.py',
                                 'analysis/build_track_agg_cache.py'], capture_output=True, text=True).stdout.strip())
    meta = dict(cache_version=TA.CACHE_VERSION, info=info_name, scenes=len(scenes), keyframes=int(S[:, 0].sum()),
                tracks=int(S[:, 1].sum()), icp_anchor_tracks=int(S[:, 2].sum()), icp_corrections=int(S[:, 3].sum()),
                points=int(S[:, 4].sum()), core_seconds=float(S[:, 5].sum()), wall_seconds=time.time() - t0,
                git=sha, git_dirty_in_these_files=dirty, created=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    print(json.dumps(meta, indent=1))
    print(f'per scene: median {np.median(S[:, 5]):.1f} s, max {S[:, 5].max():.1f} s; '
          f'median {np.median(S[:, 4]):.0f} points; cache {sum(p.stat().st_size for p in out.glob("*.pkl")) / 1e9:.2f} GB')


if __name__ == '__main__':
    main()
