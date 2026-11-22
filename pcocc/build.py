"""The three occupancy GTs of a PandaSet frame: G64 (Pandar64 points), GGT (PandarGT points), GU (both)
(experiments_md 20261121_02 §3-§5.1).

For frame t, in its normative ego frame (read_frame's), on `grid`:
- OCCUPANCY from the window's points of the GT's device(s), in range (2.5-70 m from their sensor at capture time), not
  stale, not Reflection; transient classes (Smoke, Exhaust, Spray or rain) and UNBOXED movable-class points (in no
  mover and no stationary cuboid: objects no cuboid can place) only from frame t itself. Static points are
  moved world -> frame t; mover points (movers.assign_movers) are placed at their uuid's frame-t cuboid and dropped when
  it has none. Labels by majority vote (vote.py).
- VISIBILITY from the measured rays of the same device(s) (raycast.py): from the sensor at the point's capture time to
  the point, both in world geometry, in range, not stale, not Reflection, transient included. GU's traversal is the
  union of the two devices'.
- unknown = neither occupied nor traversed. Free voxels keep EMPTY.
"""
import time

import numpy as np

from . import EMPTY
from .grid import cone_voxels
from .movers import RAW_MOVABLE, Cuboids, assign_movers, in_stationary_boxes, place, placement_rows, time_cuboids
from .poses import Poses
from .raycast import traverse
from .scans import PANDAR64, PANDARGT, SequenceScans
from .vote import vote

GTS = {'g64': (PANDAR64,), 'ggt': (PANDARGT,), 'gu': (PANDAR64, PANDARGT)}
RANGE_CUT = (2.5, 70.0)                    # jbehley/voxelizer semantic_kitti.cfg: min range 2.5, max range 70
# Mover shell (m), chosen 2026-11-22 by the rule declared in experiments_md 20261121_02 §6.2 before the sweep ran
# (jobs 34959 / 34960, val 097 + 011, frames 10/30/50/70): GU vehicle traces 10,704 / 6,092 / 5,195 / 5,121 / 5,048 at
# 0 / 0.1 / 0.2 / 0.3 / 0.5 m; 0.2 is the smallest within 10% of 0.5; stationary vehicle voxels unchanged at every s.
MOVER_SHELL = 0.2


class SequenceGT:
    def __init__(self, root, seq, grid, class_map, origins, stale, range_cut=RANGE_CUT, movers=True, frames=None,
                 log=print, interpolate=True, shell=MOVER_SHELL, unboxed_transient=True):
        t0 = time.time()
        self.seq, self.grid, self.num_classes = seq, grid, class_map.num_classes
        self.scans = s = SequenceScans(root, seq, class_map, stale=stale, frames=frames)
        self.poses = Poses(root, seq)
        # ray origin per point: its device's origin in the ego frame at the point's capture time, in world
        self.origin = np.empty_like(s.xyz)
        for d in (PANDAR64, PANDARGT):
            m = s.dev == d
            if m.any():
                self.origin[m] = self.poses.world_from_ego_at(origins[d][None], s.t[m])
        rng = np.linalg.norm(s.xyz - self.origin, axis=1)
        lo, hi = range_cut
        self.in_range = (rng >= lo) & (rng <= hi)
        self.range_cut = (lo, hi)
        self.base = self.in_range & ~s.stale & ~s.reflection          # points that may occupy or cast a ray
        self.stats = {'points': len(s), 'stale_points': int(s.stale.sum()),
                      'out_of_range_points': int((~self.in_range).sum()), 'reflection_points': int(s.reflection.sum()),
                      'transient_points': int(s.transient.sum())}
        self.movers_on = movers
        if movers:
            self.cub = Cuboids(root, seq, self.poses.times, frames=frames)
            self.stats.update(time_cuboids(self.cub, s, usable=~s.stale))
            self.mover, self.box, st = assign_movers(self.cub, s, usable=self.base, interpolate=interpolate,
                                                     shell=shell)
            self.stats['mover_shell_m'] = shell
            self.stats.update(st)
            self.stats['sibling_unknown'] = self.cub.sibling_unknown
            # movable-class points no cuboid places (neither a mover nor inside a stationary box + shell): treated as
            # transient, i.e. occupying only from frame t itself (20261121_02 §6.2, declared before the full build)
            cand = self.base & np.isin(s.raw, RAW_MOVABLE) & (self.mover < 0)
            self.unboxed = cand & ~in_stationary_boxes(self.cub, s, cand, shell)
            self.stats['unboxed_movable_points'] = int(self.unboxed.sum())
        else:
            self.cub = None
            self.mover = np.full(len(s), -1, dtype=np.int64)
            self.box = None
            self.unboxed = np.zeros(len(s), dtype=bool)
        self.unboxed_transient = unboxed_transient
        self.frame_only = s.transient | (self.unboxed if unboxed_transient else False)
        self.stats['unboxed_transient'] = unboxed_transient
        self.stats['load_s'] = round(time.time() - t0, 2)
        log('%s: %d points loaded in %.1f s' % (seq, len(s), time.time() - t0))

    # --- one frame -------------------------------------------------------------------------------------------------

    def _occupancy_points(self, t, devices, window, mode):
        """(xyz in frame t's ego frame [M, 3], cls [M]) of the points that occupy, and diagnostics."""
        s = self.scans
        m = self.base & np.isin(s.dev, devices) & window & (~self.frame_only | (s.frame == t))
        static = m & (self.mover < 0)
        xyz = [s.xyz[static]]
        cls = [s.cls[static]]
        info = {'static_points': int(static.sum())}
        if self.movers_on:
            mv = np.flatnonzero(m & (self.mover >= 0))
            rows, fb = placement_rows(self.cub, t, mode)
            w, placed = place(self.cub, rows, self.mover[mv], self.box[mv])
            xyz.append(w[placed])
            cls.append(s.cls[mv][placed])
            info.update(mover_points=int(len(mv)), mover_points_placed=int(placed.sum()), p64_fallbacks=int(fb))
        xyz = np.concatenate(xyz)
        return self.poses.ego_from_world_static(xyz, t), np.concatenate(cls), info

    def _traversal(self, t, device, window):
        s = self.scans
        m = self.base & (s.dev == device) & window
        o = self.poses.ego_from_world_static(self.origin[m], t)
        e = self.poses.ego_from_world_static(s.xyz[m], t)
        return traverse(o, e, self.grid)

    def frame(self, t, window_frames=None, visibility=True, gts=('g64', 'ggt', 'gu')):
        """{gt: (label uint8 grid.shape, unknown bool grid.shape)} and per-GT diagnostics."""
        t0 = time.time()
        s = self.scans
        window = np.ones(len(s), dtype=bool) if window_frames is None else np.isin(s.frame, list(window_frames))
        trav = {}
        stats = {}
        if visibility:
            for d in sorted({d for g in gts for d in GTS[g]}):
                trav[d], st = self._traversal(t, d, window)
                stats['rays_%d' % d] = st
        out = {}
        cone = cone_voxels(self.grid)[:, :, None]
        for g in gts:
            devices = GTS[g]
            mode = 'p64' if g == 'gu' else 'own'
            xyz, cls, info = self._occupancy_points(t, devices, window, mode)
            ijk, ins = self.grid.index(xyz)
            label, occ, vst = vote(self.grid.flat(ijk[ins]), cls[ins], self.num_classes, self.grid.size)
            label = label.reshape(self.grid.shape)
            occ = occ.reshape(self.grid.shape)
            if visibility:
                tr = np.zeros(self.grid.shape, dtype=bool)
                for d in devices:
                    tr |= trav[d].astype(bool)
                unknown = ~occ & ~tr
            else:
                unknown = ~occ
            assert np.all(label[unknown] == EMPTY)
            info.update(vst)
            info.update(points_in_grid=int(ins.sum()), free=int((~occ & ~unknown).sum()), unknown=int(unknown.sum()),
                        occupied_cone=int((occ & cone).sum()), free_cone=int((~occ & ~unknown & cone).sum()),
                        unknown_cone=int((unknown & np.broadcast_to(cone, unknown.shape)).sum()))
            out[g] = (label, unknown)
            stats[g] = info
        stats['seconds'] = round(time.time() - t0, 2)
        return out, stats
