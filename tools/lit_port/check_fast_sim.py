"""Proof that the cached-background ray caster of lit_worker.py gives the SAME output as LiT's per-frame fused-mesh engine
(experiments_md 20261015_02 §7). Runs in the LiT container. For the smoke sequences (bg, fg and the stored LiT-engine output
of job 28677 under <smoke>):
  1. fast engine, ALL frames, against the stored files: local_points, local_bboxes, frame_pose compared with array_equal;
  2. legacy engine re-run on --n_legacy frames against the same files (the stored output is reproducible here);
  3. the scene rebuilt from the per-sequence info pickles (the full run's input) against the smoke scene pickle: points,
     boxes, ids and poses of --n_scene frames.
    python -I lit_port/check_fast_sim.py --smoke /home/koyama/data/lit/smoke --scenes a,b --out <json>
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lit_worker as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', required=True)
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_legacy', type=int, default=6)
    ap.add_argument('--n_scene', type=int, default=5)
    ap.add_argument('--lit', default='/home/koyama/code/lit')
    ap.add_argument('--seq_info_dir', default='/home/koyama/code/ST3D/data/waymo/waymo_processed_data')
    ap.add_argument('--calib', default='/home/koyama/data/waymo_top_calib/top_calib.pkl')
    a = ap.parse_args()
    smoke = Path(a.smoke)
    a.root, a.scene_tmp = str(smoke), str(smoke / 'scene')
    res = {}
    for n in a.scenes.split(','):
        paths = W.lit_setup(a, [n])
        from lit.containers.scene import Scene
        sc = Scene.load(smoke / 'scene' / f'{n}.pkl')
        poses = np.stack([np.asarray(f.frame_pose, np.float32) for f in sc.frames])
        L2V = np.asarray(sc.frames[0].lidar_to_vehicle_poses[0], np.float32)
        (smoke / 'poses').mkdir(exist_ok=True)
        np.savez(smoke / 'poses' / f'{n}.npz', frame_poses=poses, L2V=L2V)
        r = {}
        # 3. scene rebuild
        import lit01_waymo_from_processed as l01
        infos = pickle.load(open(Path(a.seq_info_dir) / n / f'{n}.pkl', 'rb'))
        calib = pickle.load(open(a.calib, 'rb'))
        sc2, L2V2 = l01.build_scene(n, infos, a.seq_info_dir, calib, max_frames=a.n_scene)
        ok = bool(np.array_equal(L2V, L2V2))
        for k in range(a.n_scene):
            f1, f2 = sc.frames[k], sc2.frames[k]
            ok &= bool(np.array_equal(f1.local_points, f2.local_points) and np.array_equal(f1.local_bboxes, f2.local_bboxes)
                       and np.array_equal(f1.frame_pose, f2.frame_pose) and list(f1.object_ids) == list(f2.object_ids))
        r['scene_rebuild_identical'] = ok
        del sc, sc2
        s = {'name': n, 'sampled': list(range(len(poses)))}
        stored = smoke / 'sim_nuscenes' / n

        def cmp(frames):
            bad, nz = [], 0
            for k, f in frames.items():
                z = np.load(stored / f'{k:04d}.npz', allow_pickle=True)
                same = (np.array_equal(z['local_points'], f.local_points) and z['local_points'].dtype == f.local_points.dtype
                        and np.array_equal(z['local_bboxes'], f.local_bboxes) and np.array_equal(z['frame_pose'], f.frame_pose)
                        and int(z['frame_index']) == k)
                nz += len(f.local_points)
                if not same:
                    bad.append(k)
            return bad, nz
        t = time.time(); fr = W.sim_sequence(a, paths, s, 'fast'); tf = time.time() - t
        bad, nz = cmp(fr)
        r.update(fast_frames=len(fr), fast_points_compared=nz, fast_mismatching_frames=bad, fast_s=tf, fast_s_per_frame=tf / len(fr))
        t = time.time(); fl = W.sim_sequence(a, paths, s, 'legacy', frames=list(range(a.n_legacy))); tl = time.time() - t
        badl, _ = cmp(fl)
        r.update(legacy_frames=len(fl), legacy_mismatching_frames=badl, legacy_s_per_frame=tl / len(fl))
        res[n] = r
        print(n, r, flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    allok = all(r['scene_rebuild_identical'] and not r['fast_mismatching_frames'] and not r['legacy_mismatching_frames']
                for r in res.values())
    print('IDENTICAL' if allok else 'MISMATCH')
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
