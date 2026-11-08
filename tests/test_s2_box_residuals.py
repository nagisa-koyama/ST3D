"""tools/analysis/s2_box_residuals.py: the one-to-one matching, the residual definitions, the two scaling
conventions and the parser of the evaluator's printed IoU 0.5 block (experiments_md 20261108_06)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import s2_box_residuals as sbr  # noqa: E402


def test_matching_takes_the_highest_iou_first_and_is_one_to_one():
    # pred 0 overlaps GT 0 at 0.6 and GT 1 at 0.9; pred 1 overlaps GT 1 at 0.8 only.
    # Highest-IoU-first gives (0, 1) then nothing for pred 1 on GT 1 (taken) -> pred 1 unmatched;
    # GT 0 then has only pred 0 (taken). A per-prediction greedy would give (0, 0) and (1, 1) instead.
    iou = np.array([[0.6, 0.9], [0.0, 0.8]])
    assert sbr.match_one_to_one(iou, 0.5) == [(0, 1)]
    # a pair below the threshold is never kept
    assert sbr.match_one_to_one(np.array([[0.49]]), 0.5) == []
    assert sbr.match_one_to_one(np.zeros((0, 3)), 0.5) == []
    # each GT and each prediction used once
    iou = np.array([[0.7, 0.6], [0.65, 0.55]])
    pairs = sbr.match_one_to_one(iou, 0.5)
    assert pairs == [(0, 0), (1, 1)]


def test_residuals_are_pred_minus_gt_in_the_lidar_frame():
    gt = np.array([[10.0, 0.0, -0.9, 4.0, 1.6, 1.5, 0.0]])    # bottom at -1.65
    pred = np.array([[10.0, 0.0, -0.7, 4.4, 1.76, 1.8, 0.0]])  # bottom at -1.60
    r = sbr.pair_residuals(pred, gt)
    assert np.isclose(r['l_ratio'][0], 1.1) and np.isclose(r['w_ratio'][0], 1.1)
    assert np.isclose(r['h_ratio'][0], 1.2)
    assert np.isclose(r['dz_centre'][0], 0.2)
    assert np.isclose(r['dz_bottom'][0], 0.05)


def _det(h=1.5):
    return {
        'name': np.array(['Car']),
        'dimensions': np.array([[4.0, h, 1.6]], np.float32),     # camera l, h, w
        'location': np.array([[1.0, 1.7, 20.0]], np.float32),     # bottom centre, y down
        'rotation_y': np.array([0.3], np.float32),
        'boxes_lidar': np.array([[20.0, -1.0, -1.7 + h / 2, 4.0, 1.6, h, 0.0]], np.float32),
        'bbox': np.array([[0.0, 0.0, 10.0, 10.0]], np.float32),
        'score': np.array([0.9], np.float32),
    }


def test_scaling_about_the_centre_keeps_the_centre_and_raises_the_bottom():
    s = 0.8
    d = sbr.scale_det(_det(), s, 'centre', reproject=False)
    assert np.allclose(d['dimensions'], [[3.2, 1.2, 1.28]])
    # camera y points down: the bottom rises by (h - h') / 2 = 0.15
    assert np.isclose(d['location'][0, 1], 1.7 - 0.15)
    assert np.isclose(d['boxes_lidar'][0, 2], -1.7 + 0.75)       # LiDAR centre unchanged
    assert np.allclose(d['boxes_lidar'][0, 3:6], [3.2, 1.28, 1.2])
    assert np.allclose(d['bbox'], _det()['bbox'])                 # not re-projected when asked not to


def test_scaling_with_the_bottom_fixed_drops_the_centre():
    s = 0.8
    d = sbr.scale_det(_det(), s, 'bottom', reproject=False)
    assert np.isclose(d['location'][0, 1], 1.7)                   # bottom unchanged
    assert np.isclose(d['boxes_lidar'][0, 2], -1.7 + 0.6)         # centre = bottom + h' / 2
    bottom = d['boxes_lidar'][0, 2] - d['boxes_lidar'][0, 5] / 2
    assert np.isclose(bottom, -1.7)


def test_parse_r40_reads_moderate_at_both_thresholds():
    text = '\n'.join([
        'Car AP@0.70, 0.70, 0.70:', 'bbox AP:1, 2, 3', 'bev  AP:1, 2, 3', '3d   AP:1, 2, 3', 'aos  AP:1, 2, 3',
        'Car AP_R40@0.70, 0.70, 0.70:', 'bbox AP:93.5, 79.4, 78.7', 'bev  AP:88.8860, 76.6611, 74.7640',
        '3d   AP:64.2715, 48.2549, 46.2822', 'aos  AP:93.39, 79.16, 78.34',
        'Car AP@0.70, 0.50, 0.50:', 'bbox AP:1, 2, 3', 'bev  AP:1, 2, 3', '3d   AP:1, 2, 3', 'aos  AP:1, 2, 3',
        'Car AP_R40@0.70, 0.50, 0.50:', 'bbox AP:93.5, 79.4, 78.7', 'bev  AP:96.4714, 91.9375, 92.0699',
        '3d   AP:95.1750, 86.9652, 87.1274', 'aos  AP:93.39, 79.16, 78.34',
    ])
    ap = sbr.parse_r40(text)
    assert ap['0.7']['bev'][1] == 76.6611 and ap['0.7']['3d'][1] == 48.2549
    assert ap['0.5']['bev'][1] == 91.9375 and ap['0.5']['3d'][1] == 86.9652


def _lidar_det(frame='000001'):
    """A detection whose camera fields are built from its LiDAR box with the frame's real calibration."""
    import pytest
    if not (sbr.CALIB_DIR / f'{frame}.txt').exists():
        pytest.skip('KITTI calibration not on this machine')
    from pcdet.utils import box_utils
    lid = np.array([[15.0, 2.0, -0.9, 4.2, 1.8, 1.6, 0.4]], np.float32)
    cam = box_utils.boxes3d_lidar_to_kitti_camera(lid, sbr._calib(frame))
    det = {'name': np.array(['Car']), 'boxes_lidar': lid, 'location': cam[:, 0:3], 'dimensions': cam[:, 3:6],
           'rotation_y': cam[:, 6], 'score': np.array([0.9], np.float32)}
    info = {'point_cloud': {'lidar_idx': frame}, 'image': {'image_shape': np.array([375, 1242])}}
    det['bbox'] = sbr.project_bbox(det, info)
    return det, info


def test_lidar_frame_arm_is_the_identity_at_unit_scale_and_moves_the_box_as_asked():
    det, info = _lidar_det()
    same = sbr.scale_det_lidar(det, (1.0, 1.0, 1.0), 0.0, info)
    for k in ('boxes_lidar', 'location', 'dimensions', 'rotation_y', 'bbox'):
        assert np.allclose(same[k], det[k], atol=1e-4), k
    moved = sbr.scale_det_lidar(det, (0.9, 0.8, 0.5), -0.1, info)
    assert np.allclose(moved['boxes_lidar'][0, 3:6], [4.2 * 0.9, 1.8 * 0.8, 1.6 * 0.5], atol=1e-5)
    assert np.isclose(moved['boxes_lidar'][0, 2], -1.0, atol=1e-6)
    # camera dims are (l, h, w); the camera bottom (y down) of the new box: old centre + 0.1 down, + h' / 2
    assert np.allclose(moved['dimensions'][0], [4.2 * 0.9, 0.8, 1.8 * 0.8], atol=1e-4)
    old_centre_y = det['location'][0, 1] - det['dimensions'][0, 1] / 2
    assert abs(moved['location'][0, 1] - (old_centre_y + 0.1 + 0.4)) < 5e-3  # Tr_velo_to_cam is nearly axis-aligned
