"""A sequence's lidar poses at any time (experiments_md 20261121_02 §5.1 item 3).

pcocc's own copy of the semantics of the semseg session's `pcseg.render.PoseTrack` (branch s/pcseg-s3, ST3D 7c1b327),
kept until that class merges into v20261121_lidar_seg; then pcocc imports it read-only and a test checks the two agree.
- At a frame's pose time: exactly pcseg.pandaset_seg.read_frame's normative ego frame (tests/test_occ_pandaset.py).
- Between poses: position linear, rotation slerp.
- Beyond either end: the end segment's constant velocity and body-frame angular velocity (a Pandar64 scan spans the
  100 ms BEFORE its pose time, so frame 0 starts before the first pose).
"""
import json
import os

import numpy as np

# PandaSet ego axes (x right, y forward, z up) -> normative (x forward, y left, z up); the swap of
# pcseg.pandaset_seg._ego_from_world.
SWAP = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class Poses:
    def __init__(self, root, seq):
        from scipy.spatial.transform import Rotation
        with open(os.path.join(root, seq, 'lidar', 'poses.json')) as f:
            poses = json.load(f)
        with open(os.path.join(root, seq, 'lidar', 'timestamps.json')) as f:
            self.times = np.asarray(json.load(f), dtype=np.float64)
        assert len(poses) == len(self.times)
        assert np.all(np.diff(self.times) > 0), 'pose times must increase'
        self.rot = Rotation.from_quat([[p['heading'][k] for k in 'xyzw'] for p in poses])
        self.pos = np.array([[p['position'][k] for k in 'xyz'] for p in poses], dtype=np.float64)

    def at(self, times):
        """(Rotation, position [N, 3]) of the lidar pose at each time."""
        from scipy.spatial.transform import Rotation, Slerp
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        tc = np.clip(t, self.times[0], self.times[-1])
        rot = Slerp(self.times, self.rot)(tc)
        pos = np.stack([np.interp(tc, self.times, self.pos[:, k]) for k in range(3)], axis=1)
        quat = None
        for m, (i, j), e in ((t < self.times[0], (0, 1), 0), (t > self.times[-1], (-2, -1), -1)):
            if not m.any():
                continue
            s = (t[m] - self.times[e]) / (self.times[j] - self.times[i])
            pos[m] = self.pos[e] + s[:, None] * (self.pos[j] - self.pos[i])
            w = (self.rot[i].inv() * self.rot[j]).as_rotvec()
            quat = rot.as_quat() if quat is None else quat
            quat[m] = (self.rot[e] * Rotation.from_rotvec(s[:, None] * w)).as_quat()
        if quat is not None:
            rot = Rotation.from_quat(quat)
        return rot, pos

    def ego_from_world_static(self, xyz_world, frame):
        """World -> normative ego frame of `frame` at its pose time (read_frame's frame)."""
        q = self.rot[frame].inv().apply(np.asarray(xyz_world, dtype=np.float64) - self.pos[frame])
        return q @ SWAP.T

    def world_from_ego_at(self, xyz_ego, times):
        """Normative ego coordinates at each time (one per point, or one for all) -> world."""
        xyz_ego = np.atleast_2d(np.asarray(xyz_ego, dtype=np.float64))
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        n = max(len(xyz_ego), len(times))
        rot, pos = self.at(np.broadcast_to(times, (n,)))
        return rot.apply(np.broadcast_to(xyz_ego, (n, 3)) @ SWAP) + pos
