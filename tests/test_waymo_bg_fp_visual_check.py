"""tools/analysis/waymo_bg_fp_visual_check.py on hand-built toy data (CPU; experiments_md 20261017_13 §6).

Pins the world-frame transform (the same physical box seen from two poses lands on one world box), the temporal label
check (window bounds, static vs moving hits, other classes), the local ground (lowest band points, footprint excluded,
NaN sentinel), the enlarged-box count and the selection of confident background FPs.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import waymo_bg_fp_visual_check as V  # noqa: E402
import waymo_ranking_loss_fp_taxonomy as T  # noqa: E402


def pose(yaw, tx, ty, tz=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, tx], [s, c, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]], np.float64)


def to_vehicle(world_box, p):
    """Inverse of V.to_world, for building toy frames."""
    b = np.array(world_box, np.float64)
    c = np.linalg.inv(p) @ np.r_[b[:3], 1.0]
    return np.r_[c[:3], b[3:6], b[6] - math.atan2(p[1, 0], p[0, 0])]


W = np.array([1110.0, 520.0, 1.0, 4.5, 2.0, 1.6, 0.4])   # a parked car in the world frame
PA, PB = pose(0.0, 1100.0, 500.0), pose(math.radians(30), 1080.0, 515.0, 0.2)


def test_world_transform_round_trip():
    for p in (PA, PB):
        assert np.allclose(V.to_world(to_vehicle(W, p), p)[0], W, atol=1e-9)
    # seen from two poses, the same physical box overlaps itself completely in the world frame
    a, b = V.to_world(to_vehicle(W, PA), PA), V.to_world(to_vehicle(W, PB), PB)
    a[:, :3] -= W[:3]
    b[:, :3] -= W[:3]
    assert T.bev_iou(a.astype(np.float32), b.astype(np.float32))[0, 0] == pytest.approx(1.0, abs=1e-4)


def rec(p, names, world_boxes, ids):
    return dict(pose=p, names=np.array(names), ids=np.array(ids),
                boxes=np.array([to_vehicle(b, p) for b in world_boxes], np.float32).reshape(-1, 7))


def test_temporal_check_static_moving_other_and_window():
    moved = W + np.array([0.3, 0.2, 0, 0, 0, 0, 0])            # the parked car, 0.36 m label jitter
    far = W + np.array([30.0, 0, 0, 0, 0, 0, 0])
    seq = {
        50: rec(PA, [], [], []),                               # the FP's own frame: nothing labelled there
        48: rec(PB, ['Vehicle'], [W], ['parked']),
        51: rec(PA, ['Vehicle', 'Pedestrian'], [moved, W], ['parked', 'ped']),
        61: rec(PA, ['Vehicle'], [W], ['outside']),            # offset 11: outside the +-10 window
        45: rec(PA, ['Vehicle'], [W], ['mover']),              # a vehicle driving through the place
        44: rec(PA, ['Vehicle'], [far], ['mover']),
    }
    win = V.window_frames(seq, 50)
    assert sorted(o for o, _ in win) == [-6, -5, -2, 0, 1]
    fp = to_vehicle(W, PA)
    r = V.temporal_check(fp, PA, win)
    assert r['veh_hit'] and r['other_hit'] and r['veh_static']
    assert r['veh_nearest'] == 1
    assert r['veh_min_spread'] == pytest.approx(math.hypot(0.3, 0.2), abs=1e-6)
    # only the mover hits: a Vehicle hit, but not static
    only_mover = [(o, x) for o, x in win if o in (-6, -5, 0)]
    r = V.temporal_check(fp, PA, only_mover)
    assert r['veh_hit'] and not r['veh_static'] and not r['other_hit'] and r['veh_nearest'] == 5
    # nothing in the window
    r = V.temporal_check(fp, PA, [(0, seq[50])])
    assert not r['veh_hit'] and r['veh_nearest'] == -1 and not r['other_hit'] and np.isnan(r['veh_min_spread'])


def test_local_ground_and_counts():
    box = np.array([20.0, 5.0, 0.6, 4.0, 2.0, 1.5, 0.0])
    xs, ys = np.meshgrid(np.arange(17.0, 23.01, 0.25), np.arange(2.0, 8.01, 0.25))
    ground = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, -0.3)]
    wall = np.c_[np.full(40, 23.5), np.linspace(2, 8, 40), np.linspace(0, 2, 40)]   # in the band, high
    inside = np.c_[np.full(30, 20.0), np.linspace(4.5, 5.5, 30), np.full(30, -2.0)]  # in the footprint, very low
    pts = np.r_[ground, wall, inside]
    assert V.local_ground(pts, box) == pytest.approx(-0.3)       # the footprint's low points are ignored
    g = V.geometry(pts, None, box)
    assert g['bottom_above_ground'] == pytest.approx(0.6 - 0.75 + 0.3)
    assert V.local_ground(pts[:4], box) != V.local_ground(pts[:4], box)  # NaN: fewer than 5 band points
    # 1.5x box: a point 1.2 half-lengths out is inside the enlarged box only
    p = np.array([[20.0 + 2.4, 5.0, 0.6], [20.0, 5.0, 0.6]])
    assert int(T.points_in_boxes_count(p, box[None])[0]) == 1
    assert int(T.points_in_boxes_count(p, V.enlarged(box)[None])[0]) == 2


def test_selection_uses_the_pass_codes():
    det = dict(score=np.array([0.9, 0.8, 0.7, 0.95, 0.85, 0.6, 0.99]),
               gt7=np.array([0, 1, 2, -1, -1, -1, -1]),
               code=np.array([T.TP, T.TP, T.TP, T.BG_PTS, T.BG_EMPTY, T.BG_PTS, T.LOC]))
    idx, med = V.select_confident_background(det)
    assert med == pytest.approx(0.8)
    assert idx.tolist() == [3, 4]          # 0.6 is below the median; the localisation FP is not background
