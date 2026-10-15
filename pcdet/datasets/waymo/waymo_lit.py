"""Translated Waymo frames from LiT (experiments_md 20261015_02): the opt-in `LIT_TRANSLATED` training hook.

`lit_port/lit_run_scenes.py` writes, per Waymo sequence, one `NNNN.npz` per frame (`SimScene.save_sim_frames`) whose
`local_points` are the nuScenes-style scan in the Waymo VEHICLE frame (the frame of the processed `NNNN.npy`; SHIFT_COOR is
applied afterwards by the caller, exactly as for the stored cloud). `NNNN` is the frame's `sample_idx`.

Conventions the loader depends on, each checked in tests/test_waymo_lit.py:
- (N, 3) float32 is padded to the stored (N, 5) layout [x, y, z, intensity, elongation] with zeros. The Waymo source
  configs encode `used_feature_list: [x, y, z]`, so the padded columns never reach the model, but a (N, 3) array would
  break the 5-column `src_feature_list`.
- A missing file, a wrong frame index, a non-finite value or a wrong shape is an ERROR, never a fallback to the stored
  cloud: a silent fallback would train part of the row on real Waymo scans.
- An empty translated cloud (a frame with no hit) is legal and returns shape (0, 5).
"""
from pathlib import Path

import numpy as np


def load_translated(root, sequence, sample_idx):
    path = Path(root) / sequence / ('%04d.npz' % sample_idx)
    if not path.exists():
        raise FileNotFoundError('LIT_TRANSLATED: no translated frame %s' % path)
    with np.load(path) as z:
        pts = np.asarray(z['local_points'])
        idx = int(np.asarray(z['frame_index']).reshape(-1)[0])
    if idx != sample_idx:
        raise ValueError('LIT_TRANSLATED: %s holds frame_index %d, expected %d' % (path, idx, sample_idx))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('LIT_TRANSLATED: %s local_points has shape %s, expected (N, 3)' % (path, pts.shape))
    if not np.isfinite(pts).all():
        raise ValueError('LIT_TRANSLATED: %s has non-finite points' % path)
    out = np.zeros((len(pts), 5), np.float32)
    out[:, :3] = pts
    return out
