"""LiT's `box_to_corners_3d` (pcdet/utils/box_utils.py of b1516de), numpy only. bbox (7,) or (8,): x, y, z, dx, dy, dz,
heading[, cls]; returns the (8, 3) corners in LiT's order."""
import numpy as np

_TEMPLATE = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
                      [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]], np.float32) / 2


def boxes_to_corners_3d(boxes3d):
    b = np.asarray(boxes3d, np.float32)
    corners = b[:, None, 3:6] * _TEMPLATE[None]                       # (N, 8, 3)
    c, s = np.cos(b[:, 6]), np.sin(b[:, 6])
    rot = np.stack([c, s, np.zeros_like(c), -s, c, np.zeros_like(c), np.zeros_like(c), np.zeros_like(c),
                    np.ones_like(c)], 1).reshape(-1, 3, 3).astype(np.float32)   # points @ rot, as rotate_points_along_z
    corners = np.matmul(corners, rot)
    return corners + b[:, None, 0:3]


def box_to_corners_3d(bbox):
    return boxes_to_corners_3d(np.expand_dims(np.asarray(bbox), axis=0))[0]
