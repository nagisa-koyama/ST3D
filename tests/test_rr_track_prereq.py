"""tools/analysis/rr_track_prereq.py (CPU; experiments_md 20261018_03 §9): world / vehicle round trip, the association
(constant velocity, gates, gaps, greedy one-to-one), interior-gap interpolation and the tail, BEV IoU, Waymo pixel
assignment of synthetic points against the stored-order rows."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import rr_track_prereq as T  # noqa: E402


def _pose(x, y, yaw):
    P = np.eye(4); c, s = np.cos(yaw), np.sin(yaw)
    P[:2, :2] = [[c, -s], [s, c]]; P[0, 3], P[1, 3] = x, y
    return P


def test_world_vehicle_round_trip():
    P = _pose(100.0, -30.0, 0.7)
    b = np.array([[20.0, 3.0, 1.0, 4.5, 1.9, 1.6, 0.4]])
    w = T.to_world(b, P)
    assert np.isclose(w[0, 6], 1.1) and np.allclose(T.to_vehicle(w, P), b)
    assert np.allclose(T.head_diff_mod_pi([0.1, 0.1], [0.1 + np.pi, 0.1 + np.pi / 2]), [0.0, np.pi / 2])


def test_associate_two_cars_with_a_gap_and_a_crossing():
    # car A moves +2 m/step along x; car B is parked 3 m away; A is missing at step 2 (gap of one)
    frames = []
    for st in range(6):
        boxes = [[st * 2.0, 0.0, 0.8, 4.5, 1.9, 1.6, 0.0], [4.0, 3.0, 0.8, 4.5, 1.9, 1.6, 1.5]]
        ex = [[st, 0], [st, 1]]
        if st == 2:
            boxes, ex = boxes[1:], ex[1:]
        frames.append((st, np.array(boxes), np.array(ex, float)))
    tracks = T.associate(frames)
    assert len(tracks) == 2
    a = [t for t in tracks if t['extra'][0, 1] == 0][0]; b = [t for t in tracks if t['extra'][0, 1] == 1][0]
    assert a['steps'].tolist() == [0, 1, 3, 4, 5] and b['steps'].tolist() == list(range(6))
    gaps = T.interior_gaps(a)
    assert len(gaps) == 1 and gaps[0][0] == 2 and np.isclose(gaps[0][1][0], 4.0)
    tl = T.tail(a, 3, 7)
    assert [g for g, _ in tl] == [6, 7] and np.isclose(tl[1][1][0], 14.0)
    e = T.extrapolate_from(a, 1, 2)
    assert np.isclose(e[0], 6.0) and T.extrapolate_from(a, 0, 1) is None
    # a heading flip by pi is the same car; a 60 deg turn is not
    f2 = [(0, np.array([[0.0, 0.0, 0.8, 4.5, 1.9, 1.6, 0.0]]), np.zeros((1, 1))),
          (1, np.array([[0.5, 0.0, 0.8, 4.5, 1.9, 1.6, np.pi], [0.0, 1.0, 0.8, 4.5, 1.9, 1.6, np.radians(60)]]),
           np.zeros((2, 1)))]
    t2 = T.associate(f2)
    assert len(t2) == 2 and t2[0]['steps'].tolist() == [0, 1] and np.isclose(t2[0]['wboxes'][1, 6], np.pi)


def test_bev_iou_and_waymo_pixels():
    a = np.array([0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]); b = a.copy(); b[0] = 2.0
    assert np.isclose(T.bev_iou(a, b), 4 / 12) and np.isclose(T.bev_iou(a, a), 1.0)
    from pcdet.datasets.waymo.waymo_rings import top_columns_and_elevations
    E = np.eye(4); E[:3, 3] = [1.43, 0.0, 2.184]; yaw = np.radians(148.0)
    E[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    inc_desc = np.radians(np.linspace(2.3, -17.6, 64))
    k, az = 30, np.radians(20.0); r = 50.0
    ps = np.array([r * np.cos(inc_desc[k]) * np.cos(az), r * np.cos(inc_desc[k]) * np.sin(az), r * np.sin(inc_desc[k])])
    pv = (E[:3, :3] @ ps + E[:3, 3])[None]
    pix = T.pixels_waymo(pv, E, inc_desc)[0]
    colf, _ = top_columns_and_elevations(pv, E)
    assert pix // T.W_TOP == k and pix % T.W_TOP == int(np.floor(colf[0]))
