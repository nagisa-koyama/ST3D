"""The occupancy grid (experiments_md 20261121_02 §3 and §3.1).

SemanticKITTI's volume (jbehley/voxelizer: 0.2 m, x [0, 51.2), y [-25.6, 25.6), z [-2.0, 4.4) about a Velodyne 1.73 m
above the ground) kept at the same place RELATIVE TO THE GROUND: [ground - 0.27, ground + 6.13). In PandaSet's normative
ego frame (pcseg.pandaset_seg.read_frame: pose origin, x forward, y left, z up, no SHIFT_COOR) the ground is at
-SHIFT_COOR z = -0.30 m, so z in [-0.57, 5.83). x and y are about the ego origin, where P1's cone is centred.
"""
import numpy as np

from . import EMPTY, IGNORE_VOXEL


class Grid:
    def __init__(self, lo, voxel, shape):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.voxel = float(voxel)
        self.shape = tuple(int(s) for s in shape)
        self.hi = self.lo + self.voxel * np.asarray(self.shape, dtype=np.float64)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def index(self, xyz):
        """Integer voxel index [N, 3] of each point, and the mask of points inside the grid."""
        ijk = np.floor((np.asarray(xyz, dtype=np.float64) - self.lo) / self.voxel).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.asarray(self.shape)), axis=1)
        return ijk, inside

    def flat(self, ijk):
        return np.ravel_multi_index(tuple(np.asarray(ijk).T), self.shape)

    def centres(self):
        """Voxel centres [X, Y, Z, 3] in metres."""
        axes = [self.lo[k] + self.voxel * (np.arange(self.shape[k]) + 0.5) for k in range(3)]
        return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)

    def to_dict(self):
        return {'lo': self.lo.tolist(), 'voxel': self.voxel, 'shape': list(self.shape)}


GROUND_Z = -0.30                                  # -SHIFT_COOR z of tools/cfgs/pcseg/spunet-pandaset64.yaml
SEMKITTI_BELOW_GROUND = 2.0 - 1.73                # 0.27 m: voxelizer floor -2.0 about a Velodyne 1.73 m up
PANDASET_GRID = Grid(lo=(0.0, -25.6, GROUND_Z - SEMKITTI_BELOW_GROUND), voxel=0.2, shape=(256, 256, 32))


def encode(cls):
    """pcseg class index (0..16) -> stored label byte. Raises on anything else (the -1 sentinel included)."""
    cls = np.asarray(cls)
    if cls.size and (cls.min() < 0 or cls.max() > 253):
        raise ValueError('class index outside 0..253: min %d, max %d' % (cls.min(), cls.max()))
    return (cls + 1).astype(np.uint8)


def scoring_labels(label, unknown):
    """The label byte with UNKNOWN folded into IGNORE_VOXEL: what a metric or a loss may read."""
    label = np.asarray(label, dtype=np.uint8)
    unknown = np.asarray(unknown, dtype=bool)
    assert label.shape == unknown.shape
    out = label.copy()
    out[unknown] = IGNORE_VOXEL
    return out


def cone_voxels(grid, fov_deg=60.0, heading_deg=0.0):
    """[X, Y] mask of voxel columns whose centre lies in P1's cone (pcseg.engine.cone_mask, about the ego xy origin)."""
    from pcseg.engine import cone_mask
    c = grid.centres()[:, :, 0, :2].reshape(-1, 2)
    return cone_mask(c, fov_deg, heading_deg).reshape(grid.shape[:2])


__all__ = ['Grid', 'PANDASET_GRID', 'GROUND_Z', 'encode', 'scoring_labels', 'cone_voxels', 'EMPTY', 'IGNORE_VOXEL']
