"""Where the per-frame time of the cached-background ray caster goes (experiments_md 20261015_02 §7). Container, python -I."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lit_worker as W
import argparse, numpy as np
a = argparse.Namespace(lit='/home/koyama/code/lit', root='/home/koyama/data/lit/smoke', scene_tmp='/home/koyama/data/lit/smoke/scene')
n = sys.argv[1]
paths = W.lit_setup(a, [n])
import open3d as o3d, camtools as ct
from lit.containers.fg_scene import FGScene
from lit.lidar import Lidar, NuScenesLidarIntrinsics
z = np.load(Path(a.root) / 'poses' / f'{n}.npz'); poses, L2V = z['frame_poses'], z['L2V']
t = time.time(); fg = FGScene.load(paths.fg_dir / f'{n}.pkl'); print('load fg', time.time() - t)
t = time.time(); bg = o3d.io.read_triangle_mesh(str(paths.bg_dir / f'{n}.ply')); print('read bg', time.time() - t, len(bg.triangles), 'tris')
t = time.time(); sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(bg)); print('build bg scene', time.time() - t)
intr = NuScenesLidarIntrinsics()
T = dict(mesh=0, rays=0, bgcast=0, fgbuild=0, fgcast=0, post=0)
for k in range(40, 60):
    t = time.time(); m = fg.get_frame_mesh(frame_index=k, src_to_dst_scales=None); T['mesh'] += time.time() - t
    t = time.time(); L = Lidar(intrinsics=intr, pose=poses[k] @ L2V); r = L.get_rays().astype(np.float32); T['rays'] += time.time() - t
    ts = o3d.core.Tensor(r)
    t = time.time(); d = sc.cast_rays(ts)['t_hit'].numpy(); T['bgcast'] += time.time() - t
    t = time.time(); s2 = o3d.t.geometry.RaycastingScene(); s2.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m)); T['fgbuild'] += time.time() - t
    t = time.time(); d2 = s2.cast_rays(ts)['t_hit'].numpy(); T['fgcast'] += time.time() - t
print({k: round(v / 20, 3) for k, v in T.items()}, 'tris fg', len(m.triangles))
