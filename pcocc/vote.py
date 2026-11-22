"""Per-voxel majority vote (experiments_md 20261121_02 §3 "voxel label").

Each labelled point votes for its class. Ignore points (pcseg.IGNORE = -1) occupy their voxel but never vote. The
winner is the class with the most votes; a tie goes to the LOWEST class index (the voxelizer keeps the first strict
maximum over ascending label ids). A voxel holding only ignore points is IGNORE_VOXEL.
"""
import numpy as np

from pcseg import IGNORE

from . import IGNORE_VOXEL


def vote(flat_index, cls, num_classes, size):
    """flat_index [N] voxel ids in [0, size); cls [N] class index in 0..num_classes-1 or IGNORE.

    Returns (label [size] uint8: EMPTY where no point, class + 1, or IGNORE_VOXEL; occupied [size] bool;
             counts: dict of diagnostics)."""
    flat_index = np.asarray(flat_index, dtype=np.int64)
    cls = np.asarray(cls, dtype=np.int64)
    assert flat_index.shape == cls.shape
    if flat_index.size:
        if flat_index.min() < 0 or flat_index.max() >= size:
            raise ValueError('voxel index outside [0, %d)' % size)
        bad = (cls != IGNORE) & ((cls < 0) | (cls >= num_classes))
        if bad.any():
            raise ValueError('class ids outside 0..%d and not IGNORE: %s' % (num_classes - 1,
                                                                             np.unique(cls[bad]).tolist()))
    label = np.zeros(size, dtype=np.uint8)
    occupied = np.zeros(size, dtype=bool)
    occupied[flat_index] = True
    label[flat_index] = IGNORE_VOXEL
    voting = cls != IGNORE
    if voting.any():
        # one key per (voxel, class); np.unique sorts keys, so within a voxel classes come in ascending order
        key = flat_index[voting] * num_classes + cls[voting]
        uk, cnt = np.unique(key, return_counts=True)
        vox, c = uk // num_classes, uk % num_classes
        # sort by voxel, then count DESCENDING, then class ascending; the first row per voxel is the winner
        order = np.lexsort((c, -cnt, vox))
        vox, c = vox[order], c[order]
        first = np.ones(len(vox), dtype=bool)
        first[1:] = vox[1:] != vox[:-1]
        label[vox[first]] = (c[first] + 1).astype(np.uint8)
    return label, occupied, {'points': int(flat_index.size), 'voting_points': int(voting.sum()),
                             'occupied': int(occupied.sum()), 'ignore_only': int((label == IGNORE_VOXEL).sum())}
