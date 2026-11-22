"""A sequence's lidar poses at any time (experiments_md 20261121_02 §5.1 item 3).

A read-only adapter over the semseg session's `pcseg.render.PoseTrack` (merged into v20261121_lidar_seg at ST3D
8d979e7), which replaced pcocc's own copy of the same semantics (ST3D c233e3c). tests/test_occ_pandaset.py keeps that
copy as a reference and checks the two agree exactly.
- At a frame's pose time: exactly pcseg.pandaset_seg.read_frame's normative ego frame (tests/test_occ_pandaset.py).
- Between poses: position linear, rotation slerp.
- Beyond either end: the end segment's constant velocity and body-frame angular velocity (a Pandar64 scan spans the
  100 ms BEFORE its pose time, so frame 0 starts before the first pose).
"""
import numpy as np

from pcseg.render import SWAP, PoseTrack  # noqa: F401  (SWAP re-exported for pcocc's tools)


class Poses(PoseTrack):
    def __init__(self, root, seq):
        super().__init__(root, seq)
        assert len(self.rot) == len(self.times) == len(self.pos)
        assert np.all(np.diff(self.times) > 0), 'pose times must increase'

    def at(self, times):
        """(Rotation, position [N, 3]) of the lidar pose at each time."""
        return self._at(times)

    def ego_from_world_static(self, xyz_world, frame):
        """World -> normative ego frame of `frame` at its pose time (read_frame's frame)."""
        return self.ego_static(xyz_world, frame)

    def world_from_ego_at(self, xyz_ego, times):
        """Normative ego coordinates at each time (one per point, or one for all) -> world."""
        xyz_ego = np.atleast_2d(np.asarray(xyz_ego, dtype=np.float64))
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        n = max(len(xyz_ego), len(times))
        return PoseTrack.world_from_ego_at(self, np.broadcast_to(xyz_ego, (n, 3)), np.broadcast_to(times, (n,)))
