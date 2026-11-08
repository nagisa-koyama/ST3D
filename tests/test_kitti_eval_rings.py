"""tools/analysis/kitti_eval_cpu.py --rings: Car AP per ring of box-centre range through the official evaluator
(experiments_md 20261003_05 §5 item 38). Synthetic KITTI-style frames with known answers:
  - a GT outside the ring is ignored, not a miss;
  - a detection outside the ring is dropped, not a false positive;
  - a detection inside the ring that matches a GT just outside is neither TP nor FP (the border caveat);
  - the ring [0, inf) equals the plain evaluation;
  - a frame with no GT in the ring is harmless; a ring with no valid GT gives n/a, not a number.
The official R40 curve adds at most one recall point per detection, so each case has at least 40 GT per ring (N frames);
with fewer, a perfect detector reads below 100.
"""
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))
kitti_eval_cpu = pytest.importorskip('kitti_eval_cpu')

L, W, H = 4.0, 1.8, 1.5
N = 50


def box(x, y):
    """One Car at LiDAR (x, y): its LiDAR box and the camera-frame fields the evaluator reads (cam x = -y, z = x)."""
    return {'lidar': [x, y, -1.0, L, W, H, 0.0], 'location': [-y, 1.0, x], 'dimensions': [L, H, W],
            'rotation_y': -np.pi / 2, 'bbox': [100.0, 100.0, 200.0, 160.0]}


def frame_gt(cars, dontcare=0):
    n = len(cars)
    names = ['Car'] * n + ['DontCare'] * dontcare
    pad = lambda rows, fill: np.array(list(rows) + [fill] * dontcare, dtype=np.float32)
    return {
        'name': np.array(names), 'truncated': pad([0.0] * n, -1.0), 'occluded': pad([0] * n, -1),
        'alpha': pad([0.0] * n, -10.0), 'bbox': pad([c['bbox'] for c in cars], [0, 0, 1, 1]).reshape(-1, 4),
        'dimensions': pad([c['dimensions'] for c in cars], [-1, -1, -1]).reshape(-1, 3),
        'location': pad([c['location'] for c in cars], [-1000, -1000, -1000]).reshape(-1, 3),
        'rotation_y': pad([c['rotation_y'] for c in cars], -10.0),
        'gt_boxes_lidar': np.array([c['lidar'] for c in cars], dtype=np.float32).reshape(-1, 7),
    }


def frame_dt(cars, scores, fid):
    n = len(cars)
    return {
        'frame_id': fid, 'name': np.array(['Car'] * n), 'truncated': np.zeros(n), 'occluded': np.zeros(n),
        'alpha': np.zeros(n), 'bbox': np.array([c['bbox'] for c in cars], dtype=np.float32).reshape(-1, 4),
        'dimensions': np.array([c['dimensions'] for c in cars], dtype=np.float32).reshape(-1, 3),
        'location': np.array([c['location'] for c in cars], dtype=np.float32).reshape(-1, 3),
        'rotation_y': np.array([c['rotation_y'] for c in cars], dtype=np.float32),
        'score': np.array(scores, dtype=np.float32),
        'boxes_lidar': np.array([c['lidar'] for c in cars], dtype=np.float32).reshape(-1, 7),
        'pred_labels': np.ones(n, dtype=np.int64),
    }


def write(tmp_path, frames):
    """frames: list of (gt cars, det cars, det scores). Returns (infos path, result path)."""
    infos, results = [], []
    for i, (gt, dt, sc) in enumerate(frames):
        fid = '%06d' % i
        infos.append({'point_cloud': {'lidar_idx': fid}, 'annos': frame_gt(gt, dontcare=1)})
        results.append(frame_dt(dt, sc, fid))
    ip, rp = tmp_path / 'infos.pkl', tmp_path / 'result.pkl'
    pickle.dump(infos, open(ip, 'wb')); pickle.dump(results, open(rp, 'wb'))
    return ip, rp


@pytest.fixture(scope='module')
def official():
    return kitti_eval_cpu.load_official_eval()


def rings_of(rp, ip, rings, official):
    return kitti_eval_cpu.evaluate_rings(rp, rings, infos_pkl=ip, official=official)


def test_out_of_ring_gt_is_ignored_not_missed(tmp_path, official):
    near, far = box(8, 0), box(35, 2)
    ip, rp = write(tmp_path, [([near, far], [near], [0.9])] * N)
    pooled, inner, outer = rings_of(rp, ip, [(0, math.inf), (0, 20), (20, 70.4)], official)
    assert pooled['n_gt'] == 2 * N and inner['n_gt'] == N and outer['n_gt'] == N
    assert pooled['bev'] == pytest.approx(50.0, abs=0.01)   # half the cars missed
    assert inner['bev'] == pytest.approx(100.0, abs=0.01)   # the far car is ignored here, not a miss
    assert outer['bev'] == pytest.approx(0.0, abs=0.01)


def test_out_of_ring_detection_is_dropped_not_a_false_positive(tmp_path, official):
    near, ghost = box(8, 0), box(40, -3)
    ip, rp = write(tmp_path, [([near], [ghost, near], [0.95, 0.5])] * N)
    pooled, inner = rings_of(rp, ip, [(0, math.inf), (0, 20)], official)
    assert pooled['bev'] < 75.0                              # the top-scored ghost is an FP in every frame
    assert inner['bev'] == pytest.approx(100.0, abs=0.01)


def test_detection_matching_a_gt_just_outside_is_neither_tp_nor_fp(tmp_path, official):
    near, edge_gt, edge_dt = box(8, 0), box(20.3, 0), box(19.9, 0)   # BEV IoU about 0.82, centres either side of 20 m
    ip, rp = write(tmp_path, [([near, edge_gt], [edge_dt, near], [0.95, 0.5])] * N)
    (inner,) = rings_of(rp, ip, [(0, 20)], official)
    assert inner['n_gt'] == N and inner['bev'] == pytest.approx(100.0, abs=0.01)


def test_full_ring_equals_the_plain_evaluation(tmp_path, official):
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(8):
        gt = [box(float(rng.uniform(5, 60)), float(rng.uniform(-10, 10))) for _ in range(3)]
        dt = [box(c['lidar'][0] + float(rng.normal(0, 0.3)), c['lidar'][1]) for c in gt[:2]] + [box(50, 15)]
        frames.append((gt, dt, list(rng.uniform(0.1, 1.0, 3))))
    ip, rp = write(tmp_path, frames)
    (full,) = rings_of(rp, ip, [(0, math.inf)], official)
    _, ap = kitti_eval_cpu.evaluate(rp, infos_pkl=ip, official=official)
    assert full['bev'] == pytest.approx(ap['Car_bev/moderate_R40'], abs=1e-9)
    assert full['3d'] == pytest.approx(ap['Car_3d/moderate_R40'], abs=1e-9)


def test_frames_without_gt_in_the_ring_and_an_empty_ring(tmp_path, official):
    near, far = box(8, 0), box(45, 0)
    frames = [([near], [near], [0.9])] * N + [([far], [far], [0.8])] * N
    ip, rp = write(tmp_path, frames)
    inner, empty = rings_of(rp, ip, [(0, 20), (60, 70.4)], official)
    assert inner['n_gt'] == N and inner['bev'] == pytest.approx(100.0, abs=0.01)
    assert empty['n_gt'] == 0 and empty['bev'] is None and empty['bev05'] is None


def test_ring_view_keeps_dontcare_and_relabels_only_cars(tmp_path):
    gt = frame_gt([box(8, 0), box(35, 0)], dontcare=2)
    gt['name'][1] = 'Truck'
    dt = frame_dt([box(8, 0), box(35, 0)], [0.9, 0.8], '000000')
    g, d = kitti_eval_cpu.ring_view(gt, dt, 20, 70.4)
    assert list(g['name']) == ['Van', 'Truck', 'DontCare', 'DontCare']
    assert len(d['name']) == 1 and d['boxes_lidar'][0, 0] == 35 and len(d['score']) == 1 and d['frame_id'] == '000000'
    assert list(gt['name']) == ['Car', 'Truck', 'DontCare', 'DontCare']    # the input is not modified
