"""Work-queue worker for the full LiT translation (experiments_md 20261015_02 §7). Runs INSIDE the LiT container, `python -I`.

    lit_port/lit_exec.sh [--nv] python -I lit_port/lit_worker.py --role gpu|sim --root <dir> --plan <plan.json> [...]

Tasks are sequences of the plan. A worker CLAIMS a task with an atomic `mkdir <root>/claims/<stage>/<seq>` on NFS, so any
number of workers in any number of Slurm jobs share the queue, and a finished task is skipped by a restart (markers in
<root>/done). Stages:
  gpu : extract the LiT Scene from OUR processed Waymo data -> NKSR background (lit_02) -> DeepSDF foreground (lit_03) ->
        done/<seq>.gpu. The Scene pickle (~600 MB) lives in --scene_tmp only while the sequence is processed.
  sim : ray cast the SAMPLED frames of the plan (the ones the detector reads) with the nuScenes HDL-32E model, ray-drop OFF,
        scaling OFF -> sim_nuscenes/<seq>/NNNN.npz (NNNN = sample_idx), then done/<seq>.sim. `--engine fast` casts the static
        background from one cached Embree scene and the moving foreground from a per-frame scene, nearest hit of the two
        (proved identical to LiT's per-frame fused mesh: lit_port/check_fast_sim.py); `--engine legacy` is LiT's engine.
A sim worker waits for sequences whose gpu stage is done, and exits when no task is left or no GPU worker has beaten
(heartbeat) for --hb_timeout seconds. A task that raises is retried once, then marked failed/<seq>.<stage> and the worker goes
on; the process exits non-zero if anything failed. No fallback anywhere.
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import time
import traceback
import types
from pathlib import Path


def log(*a):
    print(time.strftime('%m-%d %H:%M:%S'), *a, flush=True)


class Queue:
    def __init__(self, root, owner):
        self.root, self.owner = Path(root), owner
        for d in ('claims/gpu', 'claims/sim', 'done', 'failed', 'hb'):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def claim(self, stage, seq):
        try:
            os.mkdir(self.root / 'claims' / stage / seq)
        except FileExistsError:
            return False
        (self.root / 'claims' / stage / seq / 'owner').write_text(self.owner)
        return True

    def done(self, stage, seq):
        return (self.root / 'done' / f'{seq}.{stage}').exists()

    def mark_done(self, stage, seq, info):
        tmp = self.root / 'done' / f'.{seq}.{stage}.tmp'
        tmp.write_text(json.dumps(info))
        os.replace(tmp, self.root / 'done' / f'{seq}.{stage}')

    def mark_failed(self, stage, seq, msg):
        (self.root / 'failed' / f'{seq}.{stage}').write_text(msg)

    def beat(self):
        (self.root / 'hb' / self.owner).write_text(str(time.time()))

    def gpu_alive(self, timeout):
        now = time.time()
        for f in (self.root / 'hb').glob('gpu*'):
            if now - f.stat().st_mtime < timeout:
                return True
        return False


def lit_setup(a, names):
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here / 'shims'), a.lit, str(here)]     # the shim pcdet must win over LiT's own fork
    _g = types.ModuleType('lit.raycast_engine_gpu')           # OptiX engine not installed: CPU Embree only
    class _NoGPU:                                              # noqa: E301
        def __init__(self, *a, **k):
            raise RuntimeError('the OptiX ray caster is not installed')
    _g.RaycastEngineGPU = _NoGPU
    sys.modules['lit.raycast_engine_gpu'] = _g
    from lit.path_utils import LitPaths
    root = Path(a.root)
    scene_dir = Path(a.scene_tmp)
    return LitPaths(data_version='v0', data_domain='waymo', scene_names=names, lit_data_root=root,
                    scene_dir=scene_dir, fg_dir=root / 'fg', bg_dir=root / 'bg', sim_waymo_dir=root / 'sim_waymo',
                    sim_nuscenes_dir=root / 'sim_nuscenes', sim_kitti_dir=root / 'sim_kitti')


def run_gpu(a, plan, q):
    names = [s['name'] for s in plan['sequences']]
    paths = lit_setup(a, names)
    import numpy as np  # noqa: E402
    import importlib
    l01 = importlib.import_module('lit01_waymo_from_processed')
    m2 = importlib.import_module('lit_tools.lit_02_recon_bg')
    m3 = importlib.import_module('lit_tools.lit_03_recon_fg')
    sys.excepthook = sys.__excepthook__
    fgr = m3.FGReconstructor(use_deepsdf=True)
    calib = pickle.load(open(a.calib, 'rb'))
    for d in (paths.fg_dir, paths.bg_dir, paths.scene_dir, Path(a.root) / 'poses'):
        d.mkdir(parents=True, exist_ok=True)
    nfail, ndone = 0, 0
    for s in plan['sequences']:
        if a.max_seqs and ndone >= a.max_seqs:
            break
        n = s['name']
        q.beat()
        if q.done('gpu', n) or not q.claim('gpu', n):
            continue
        for attempt in (1, 2):
            t0 = time.time()
            try:
                infos = pickle.load(open(Path(a.seq_info_dir) / n / f'{n}.pkl', 'rb'))
                scene, L2V = l01.build_scene(n, infos, a.data, calib)
                assert len(scene) == s['n_frames'], f'{n}: scene has {len(scene)} frames, plan {s["n_frames"]}'
                poses = np.stack([np.asarray(f.frame_pose, np.float32) for f in scene.frames])
                scene.save(paths.scene_dir / f'{n}.pkl')
                np.savez(Path(a.root) / 'poses' / f'{n}.npz', frame_poses=poses, L2V=L2V)
                del scene, infos
                t1 = time.time()
                m2.recon_bg(scene_name=n, lit_paths=paths, skip_existing=False)
                t2 = time.time()
                m3.recon_fg(n, fgr=fgr, lit_paths=paths, skip_existing=False, verbose=False)
                t3 = time.time()
                (paths.scene_dir / f'{n}.pkl').unlink()
                q.mark_done('gpu', n, {'extract_s': t1 - t0, 'bg_s': t2 - t1, 'fg_s': t3 - t2, 'owner': q.owner})
                log(f'[gpu] {n} extract {t1 - t0:.0f}s bg {t2 - t1:.0f}s fg {t3 - t2:.0f}s')
                ndone += 1
                break
            except Exception:
                log(f'[gpu] {n} attempt {attempt} FAILED\n{traceback.format_exc()}')
                import torch
                torch.cuda.empty_cache()
                if attempt == 2:
                    q.mark_failed('gpu', n, traceback.format_exc())
                    nfail += 1
    log(f'gpu worker done: {ndone} sequences, {nfail} failed')
    return nfail


def cast_fast(bg_scene, fg_mesh, lidar, o3d, np):
    """Same arithmetic as lit.raycast_engine_cpu.RaycastEngineCPU.lidar_intersect_mesh, on two scenes; nearest hit wins."""
    rays = lidar.get_rays().astype(np.float32)
    t = o3d.core.Tensor(rays)
    depth = None
    if bg_scene is not None:
        depth = bg_scene.cast_rays(t)['t_hit'].numpy()
    if len(fg_mesh.vertices) > 0:
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(fg_mesh))
        d2 = sc.cast_rays(t)['t_hit'].numpy()
        depth = d2 if depth is None else np.minimum(depth, d2)
    if depth is None:
        return np.zeros((0, 3), np.float32)
    masks = depth != np.inf
    rays_o, rays_d = rays[:, :3], rays[:, 3:]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)
    points = (rays_o + rays_d * depth[:, None])[masks]
    center = lidar.pose[:3, 3]
    return points[np.linalg.norm(points - center, axis=1) < lidar.intrinsics.max_range]


def sim_sequence(a, paths, s, engine, frames=None):
    """Returns {frame_index: SimFrame}; `frames` overrides the plan's sampled list (checks)."""
    import numpy as np
    import open3d as o3d
    import camtools as ct
    from lit.containers.fg_scene import FGScene
    from lit.containers.sim_frame import SimFrame
    from lit.lidar import Lidar, NuScenesLidarIntrinsics
    n = s['name']
    z = np.load(Path(a.root) / 'poses' / f'{n}.npz')
    poses, L2V = z['frame_poses'], z['L2V']
    fg_scene = FGScene.load(paths.fg_dir / f'{n}.pkl')
    bg_mesh = o3d.io.read_triangle_mesh(str(paths.bg_dir / f'{n}.ply'))
    intr = NuScenesLidarIntrinsics()
    bg_scene = None
    if engine == 'fast' and len(bg_mesh.vertices) > 0:
        bg_scene = o3d.t.geometry.RaycastingScene()
        bg_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(bg_mesh))
    leg = None
    if engine == 'legacy':
        from lit.raycast_engine_cpu import RaycastEngineCPU
        leg = RaycastEngineCPU()
    out = {}
    for k in (frames if frames is not None else s['sampled']):
        fg_mesh = fg_scene.get_frame_mesh(frame_index=k, src_to_dst_scales=None)
        lidar = Lidar(intrinsics=intr, pose=poses[k] @ L2V)
        if engine == 'legacy':
            fused = fg_mesh + bg_mesh
            if len(fused.vertices) == 0:
                fused = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.0001)
            pts = leg.lidar_intersect_mesh(lidar=lidar, mesh=fused)
        else:
            if bg_scene is None and len(fg_mesh.vertices) == 0:       # LiT's dummy mesh, never hit in practice
                dummy = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.0001)
                from lit.raycast_engine_cpu import RaycastEngineCPU
                pts = RaycastEngineCPU().lidar_intersect_mesh(lidar=lidar, mesh=dummy)
            else:
                pts = cast_fast(bg_scene, fg_mesh, lidar, o3d, np)
        local = ct.transform.transform_points(pts, ct.convert.pose_to_T(poses[k]))
        f32 = lambda x: x.astype(np.float32) if isinstance(x, np.ndarray) and x.dtype == np.float64 else x  # noqa: E731
        out[k] = SimFrame(frame_index=k, frame_pose=poses[k], local_points=f32(local),
                          local_bboxes=f32(fg_scene.get_frame_local_bboxes(frame_index=k, src_to_dst_scales=None)),
                          incident_angles=np.zeros(len(pts), np.float32))
    return out


def run_sim(a, plan, q):
    names = [s['name'] for s in plan['sequences']]
    paths = lit_setup(a, names)
    ndone, nfail = 0, 0
    t_start = time.time()
    from lit.containers.sim_scene import SimScene
    while True:
        if a.max_seqs and ndone >= a.max_seqs:
            break
        todo = [s for s in plan['sequences'] if not q.done('sim', s['name']) and
                not (q.root / 'claims' / 'sim' / s['name']).exists()]
        if not todo:
            break
        ready = [s for s in todo if q.done('gpu', s['name'])]
        picked = None
        for s in ready:
            if q.claim('sim', s['name']):
                picked = s
                break
        if picked is None:
            if not q.gpu_alive(a.hb_timeout) and time.time() - t_start > a.hb_timeout and not ready:
                log('no ready task and no live gpu worker: exit')
                nfail += 1
                break
            time.sleep(20)
            continue
        n = picked['name']
        for attempt in (1, 2):
            t0 = time.time()
            try:
                frames = sim_sequence(a, paths, picked, a.engine)
                sc = SimScene()
                for k in sorted(frames):
                    sc.append_frame(frames[k])
                d = paths.sim_nuscenes_dir / n
                if d.exists():
                    shutil.rmtree(d)
                sc.save_sim_frames(d)
                q.mark_done('sim', n, {'frames': len(frames), 's': time.time() - t0, 'engine': a.engine, 'owner': q.owner})
                log(f'[sim] {n} {len(frames)} frames {time.time() - t0:.0f}s')
                ndone += 1
                break
            except Exception:
                log(f'[sim] {n} attempt {attempt} FAILED\n{traceback.format_exc()}')
                if attempt == 2:
                    q.mark_failed('sim', n, traceback.format_exc())
                    nfail += 1
    log(f'sim worker done: {ndone} sequences, {nfail} failed')
    return nfail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', required=True, choices=['gpu', 'sim'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--lit', default='/home/koyama/code/lit')
    ap.add_argument('--seq_info_dir', default='/home/koyama/code/ST3D/data/waymo/waymo_processed_data')
    ap.add_argument('--data', default='/home/koyama/code/ST3D/data/waymo/waymo_processed_data')
    ap.add_argument('--calib', default='/home/koyama/data/waymo_top_calib/top_calib.pkl')
    ap.add_argument('--scene_tmp', default=None)
    ap.add_argument('--engine', default='fast', choices=['fast', 'legacy'])
    ap.add_argument('--tag', default='0')
    ap.add_argument('--max_seqs', type=int, default=0)
    ap.add_argument('--hb_timeout', type=int, default=1200)
    a = ap.parse_args()
    a.scene_tmp = a.scene_tmp or str(Path(a.root) / 'scene_tmp')
    plan = json.load(open(a.plan))
    owner = f'{a.role}.{os.environ.get("SLURM_JOB_ID", "nojob")}.{a.tag}'
    q = Queue(a.root, owner)
    log('worker', owner, 'sequences', len(plan['sequences']))
    n = run_gpu(a, plan, q) if a.role == 'gpu' else run_sim(a, plan, q)
    sys.exit(1 if n else 0)


if __name__ == '__main__':
    main()
