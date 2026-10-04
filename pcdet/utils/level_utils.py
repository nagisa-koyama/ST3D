"""LEVEL_COOR: undo a LiDAR's roll/pitch relative to the vehicle, and optionally re-tilt.

nuScenes and Lyft keep points in the SENSOR frame, and their roof sensors are not mounted level:
nuScenes LIDAR_TOP is pitched +1.4 deg (Singapore) / +2.65 deg (Boston) and Lyft's -1.3 deg, so the
ground the detector sees slopes along the driving direction by a dataset-specific amount. Measured
per frame with tools/analysis/ground_tilt.py the slopes are +1.35 / +2.29 / -1.12 deg, matching the
extrinsic. SHIFT_COOR corrects the height only.

The rotation is applied about the SENSOR origin, before SHIFT_COOR:

    R = R_roll @ R_pitch @ R_level

R_level is the smallest rotation that takes the vehicle's up axis (expressed in the sensor frame,
i.e. the third column of ref_from_car's rotation) onto +z, so the sensor's own heading convention
is kept and only roll/pitch are removed. R_pitch / R_roll then tilt the levelled cloud about the
vehicle's lateral / forward axis, so that the ground RISES by PITCH_DEG towards the front and by
ROLL_DEG towards the left - which lets a target be presented with the source's attitude.

Boxes stay upright: centres and (vx, vy) are rotated, the heading is re-read from the rotated
heading vector's xy part. At the angles involved (< 3 deg) the heading changes by well under 0.1 deg,
and rotating by R then R.T returns it to within a second-order ~1e-3 rad (2 mm at a car's end).
"""
import numpy as np


def _rodrigues(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12 or abs(angle) < 1e-15:
        return np.eye(3)
    k = axis / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def level_rotation(sensor_from_vehicle=None, pitch_deg=0.0, roll_deg=0.0, from_extrinsic=True):
    """3x3 rotation for LEVEL_COOR.

    Args:
        sensor_from_vehicle: 3x3 rotation taking VEHICLE-frame vectors into the sensor frame (the
            rotation block of nuScenes/Lyft `ref_from_car`). Its columns are the vehicle's forward,
            left and up axes in sensor coordinates. None means the sensor frame is taken as
            x-forward, y-left, z-up (KITTI velodyne).
        pitch_deg: after levelling, tilt so the ground rises by this angle towards the front.
        roll_deg: after levelling, tilt so the ground rises by this angle towards the left.
        from_extrinsic: remove the extrinsic roll/pitch first (False: only re-tilt).
    """
    if sensor_from_vehicle is None:
        fwd, left, up = np.eye(3)
    else:
        S = np.asarray(sensor_from_vehicle, dtype=np.float64)[:3, :3]
        fwd, left, up = S[:, 0], S[:, 1], S[:, 2]
    R = np.eye(3)
    if from_extrinsic and sensor_from_vehicle is not None:
        z = np.array([0.0, 0.0, 1.0])
        u = up / np.linalg.norm(up)
        R = _rodrigues(np.cross(u, z), np.arccos(np.clip(u @ z, -1.0, 1.0)))
    f, l = R @ fwd, R @ left
    f[2] = 0.0
    l[2] = 0.0
    # a positive rotation about the LEFT axis sends forward points DOWN, hence the minus sign;
    # a positive rotation about the FORWARD axis sends left points UP.
    R = _rodrigues(l, -np.radians(pitch_deg)) @ R
    R = _rodrigues(f, np.radians(roll_deg)) @ R
    return R


def rotate_points(points, R):
    out = points.copy()
    out[:, 0:3] = points[:, 0:3] @ R.T.astype(points.dtype)
    return out


def rotate_boxes(boxes, R):
    """Rotate (N, 7+) boxes [x, y, z, dx, dy, dz, heading, (vx, vy, ...)] by R, keeping them upright."""
    if boxes is None or len(boxes) == 0:
        return boxes
    out = boxes.copy()
    R = R.astype(np.float64)
    out[:, 0:3] = (boxes[:, 0:3].astype(np.float64) @ R.T).astype(boxes.dtype)
    h = np.stack([np.cos(boxes[:, 6]), np.sin(boxes[:, 6]), np.zeros(len(boxes))], axis=1) @ R.T
    out[:, 6] = np.arctan2(h[:, 1], h[:, 0]).astype(boxes.dtype)
    if boxes.shape[1] >= 9:
        v = np.stack([boxes[:, 7], boxes[:, 8], np.zeros(len(boxes))], axis=1).astype(np.float64) @ R.T
        out[:, 7:9] = v[:, :2].astype(boxes.dtype)
    return out
