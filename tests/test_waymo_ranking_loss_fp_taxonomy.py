"""tools/analysis/waymo_ranking_loss_fp_taxonomy.py on a hand-built toy frame (CPU; experiments_md 20261017_13).

Pins the greedy matching, every FP class of the taxonomy, the AP split (AP_R40, perfect ranking, continuous version),
the removal gains (FPs ranked below every TP cost nothing), the point count inside boxes against pcdet's own op, and
the evaluated-cloud loader (no-label-zone points and the z range dropped).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import waymo_ranking_loss_fp_taxonomy as T  # noqa: E402


def car(x, y, l=4.0, w=2.0, h=1.5, z=0.0, yaw=0.0):
    return [x, y, z, l, w, h, yaw]


# Vehicles V0-V3 on the x axis; V3 is missed by everything that could match it.
VEH = np.array([car(10, 0), car(20, 0), car(30, 0), car(40, 0)], np.float32)
# Pedestrian P0 (a car box on it has BEV IoU 0.08 < 0.1, but P0's centre is inside), Sign S0 (nothing on it),
# a large Cyclist-class box C0 whose centre lies OUTSIDE d10 but with BEV IoU 0.185 >= 0.1.
OTHER = np.array([car(50, 5, 0.8, 0.8, 1.8), car(60, -5, 0.5, 0.5, 1.0), car(72.2, 0, 3.0, 1.5, 1.5)], np.float32)
DETS = np.array([
    car(10, 0),      # d0  0.90  TP on V0
    car(10.3, 0),    # d1  0.80  IoU 0.86 to V0, taken by d0           -> duplicate
    car(22, 0),      # d2  0.70  IoU 0.33 to V1, V1 taken LATER by d3  -> localisation
    car(20, 0),      # d3  0.60  TP on V1
    car(31.5, 0),    # d4  0.50  IoU 0.45 to V2, taken by d5 (0.95)    -> duplicate
    car(30, 0),      # d5  0.95  TP on V2
    car(50, 5),      # d6  0.40  P0's centre inside                    -> other-class
    car(0, 30),      # d7  0.30  five points inside                    -> background, with points
    car(0, -30),     # d8  0.20  points only just outside              -> background, empty
    car(43.7, 0),    # d9  0.10  IoU 0.039 to V3 (< 0.1)               -> background, empty
    car(70, 0),      # d10 0.05  IoU 0.185 to C0, C0's centre outside  -> other-class
], np.float32)
SCORES = np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.95, 0.40, 0.30, 0.20, 0.10, 0.05])
EXPECTED = [T.TP, T.DUP, T.LOC, T.TP, T.DUP, T.TP, T.OTHER, T.BG_PTS, T.BG_EMPTY, T.BG_EMPTY, T.OTHER]
POINTS = np.array([[0.5, 30.2, 0.1], [-1.5, 29.5, -0.5], [1.9, 30.9, 0.7], [0.0, 30.0, 0.0], [-1.0, 30.5, 0.3],
                   [0.0, -28.8, 0.0],   # |local y| 1.2 > 1
                   [0.0, -30.0, 2.0],   # above the top (0.75)
                   [2.1, -30.0, 0.0]])  # |local x| 2.1 > 2


def toy():
    from scipy.spatial import cKDTree
    return T.frame_detections(DETS, SCORES, VEH, OTHER, (POINTS, cKDTree(POINTS[:, :2])))


def test_taxonomy_codes():
    out = toy()
    assert list(out['codes']) == EXPECTED
    assert list(out['npts_box'][[7, 8, 9]]) == [5, 0, 0]
    assert list(out['dt_gt7']) == [0, -1, -1, 1, -1, 2, -1, -1, -1, -1, -1]
    assert list(out['gt_dt7']) == [0, 3, 5, -1]
    # IoU 0.5: d4 (0.45) and d2 (0.33) still fail; the same TPs
    assert list(out['gt_dt5']) == [0, 3, 5, -1]
    assert out['gt_best_iou'][3] == pytest.approx(0.6 / 15.4, abs=1e-4)


def test_ap_split_by_hand():
    out = toy()
    s = T.ap_split(SCORES, out['dt_gt7'] >= 0, len(VEH))
    # ranks: T T F F T F ...; recall 0.25 0.5 | 0.75 at precision 3/5; R40: 20 points at 1.0, 10 at 0.6
    assert s['ap'] == pytest.approx(65.0)
    assert s['pr'] == pytest.approx(75.0)      # 2.5 x floor(40 x 0.75)
    assert s['rl'] == pytest.approx(10.0)
    assert s['ap_c'] == pytest.approx(65.0)    # (1 + 1 + 0.6) / 4
    assert s['pr_c'] == pytest.approx(75.0)
    assert s['recall'] == pytest.approx(0.75)


def test_removal_gains():
    out = toy()
    tp, codes = out['dt_gt7'] >= 0, out['codes']

    def ap_without(drop):
        keep = ~np.isin(codes, drop)
        return T.ap_stats(SCORES[keep], tp[keep], len(VEH))[0]

    assert ap_without([T.DUP]) == pytest.approx(68.75)          # rank 4 precision 3/4
    assert ap_without([T.LOC]) == pytest.approx(68.75)
    assert ap_without([T.DUP, T.LOC]) == pytest.approx(75.0)    # = PR: the remaining FPs rank below every TP
    assert ap_without([T.BG_PTS, T.BG_EMPTY]) == pytest.approx(65.0)  # the user's point: below every TP, no cost
    assert ap_without([T.OTHER]) == pytest.approx(65.0)


def test_edge_cases():
    e = np.zeros((0, 7), np.float32)
    out = T.frame_detections(e, np.zeros(0), VEH, OTHER, None)
    assert len(out['codes']) == 0 and list(out['gt_dt7']) == [-1] * 4
    out = T.frame_detections(DETS, SCORES, e, e, None)          # no GT at all: every detection is background
    assert set(out['codes']) == {T.BG}
    assert np.isnan(T.ap_stats(SCORES, np.zeros(len(SCORES), bool), 0)[0])
    assert T.ap_stats(SCORES, np.zeros(len(SCORES), bool), 3)[:2] == (0.0, 0.0)


def test_centres_in_rotated_box():
    box = np.array([car(0, 0, yaw=np.pi / 2)], np.float32)      # long axis along y
    inside = T.centres_in_boxes_bev(np.array([[0.0, 1.8], [1.8, 0.0]]), box)
    assert inside.tolist() == [[True, False]]


def test_points_in_boxes_matches_pcdet():
    from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
    rng = np.random.default_rng(0)
    pts = rng.uniform([-20, -20, -2], [20, 20, 3], (20000, 3))
    boxes = np.stack([rng.uniform(-18, 18, 30), rng.uniform(-18, 18, 30), rng.uniform(-1, 1, 30),
                      rng.uniform(1, 6, 30), rng.uniform(1, 3, 30), rng.uniform(1, 2, 30),
                      rng.uniform(-np.pi, np.pi, 30)], 1)
    ref = np.asarray(points_in_boxes_cpu(pts.astype(np.float32), boxes.astype(np.float32))).sum(1)
    got = T.points_in_boxes_count(pts, boxes)
    assert ref.sum() > 100
    assert got.tolist() == ref.tolist()


def test_load_points_uses_the_loader(tmp_path, monkeypatch):
    seq = tmp_path / 'seq'
    seq.mkdir()
    pf = np.array([[1, 1, 0, 0.5, 0, -1],     # kept
                   [2, 2, 0, 0.5, 0, 1],      # in a no-label zone: dropped by get_lidar
                   [3, 3, 5, 0.5, 0, -1],     # z above 4: dropped by the crop
                   [80, 0, 0, 0.5, 0, -1]],   # x beyond 75.2: dropped by the crop
                  np.float32)
    np.save(seq / '0007.npy', pf)
    monkeypatch.setattr(T, 'DATA_PATH', tmp_path)
    xyz, tree = T.load_points('seq', 7)
    assert xyz.tolist() == [[1.0, 1.0, 0.0]]
    assert tree.n == 1
