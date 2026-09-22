"""Accumulation beyond the 9 stored sweeps, by chaining keyframes.

Each nuScenes keyframe carries 9 sweeps covering the 0.45 s before it, and keyframes are 0.5 s
apart - so the sweeps of consecutive keyframes very nearly tile the scene. Chaining keyframes with
S(j) = ref_from_car @ car_from_global, then composing each sweep's own transform, gives arbitrary
depth within a scene. The composition reproduces nuScenes' own stored sweep transform to 1e-13.

No motion compensation: moving objects smear over the accumulation window, which is the thing this
sweep is meant to expose.
"""
import json
import pickle
import sys
import numpy as np
from pathlib import Path

B = Path(__file__).resolve().parents[2] / 'data'
NS = B / 'nuscenes/v1.0-trainval'
_meta = NS / 'v1.0-trainval'
_logs = {l['token']: l for l in json.load(open(_meta / 'log.json'))}
_scene = {s['token']: s for s in json.load(open(_meta / 'scene.json'))}
_tok2scene = {s['token']: s['scene_token'] for s in json.load(open(_meta / 'sample.json'))}
INFOS = pickle.load(open(NS / 'nuscenes_infos_10sweeps_train.pkl', 'rb'))
SCENE = [_tok2scene[i['token']] for i in INFOS]


def _ld(path):
    return np.fromfile(str(NS / path), dtype=np.float32).reshape([-1, 5])[:, :3]


def _rm(p, c):
    return p[~((np.abs(p[:, 0]) < c) & (np.abs(p[:, 1]) < c))]


def _S(info):
    return np.asarray(info['ref_from_car']) @ np.asarray(info['car_from_global'])


def accumulate(anchor_idx, n_frames):
    """n_frames LiDAR frames ending at anchor_idx, in the anchor keyframe's coordinates."""
    anchor = INFOS[anchor_idx]
    Sa = _S(anchor)
    out, used, k = [], 0, anchor_idx
    while used < n_frames and k >= 0 and SCENE[k] == SCENE[anchor_idx]:
        info = INFOS[k]
        T = Sa @ np.linalg.inv(_S(info))                 # that keyframe -> anchor frame
        R, t = T[:3, :3], T[:3, 3]
        p = _rm(_ld(info['lidar_path']), 1.5)
        out.append((R @ p.T).T + t if k != anchor_idx else p)
        used += 1
        for s in info['sweeps']:
            if used >= n_frames:
                break
            q = _rm(_ld(s['lidar_path']), 1.0)
            tm = s['transform_matrix']
            if tm is not None:
                q = (tm[:3, :3] @ q.T).T + tm[:3, 3]      # sweep -> its own keyframe
            out.append((R @ q.T).T + t)
            used += 1
        k -= 1
    return np.concatenate(out), used


if __name__ == '__main__':
    import time
    idx = 3000
    for n in (10, 50, 100):
        t0 = time.time()
        pts, used = accumulate(idx, n)
        print('N=%3d requested, %3d frames used, %8d points, %.1f s'
              % (n, used, len(pts), time.time() - t0))
