"""tools/analysis/kitti_box_error_census.py: the along / across split of the centre error, the yaw wrap, the size and
vertical signs, the IoU 0.7 failure shares and the substitution arms. CPU only, synthetic camera-frame boxes
(x, y_bottom, z, l, h, w, ry); no KITTI file is read (experiments_md 20261024_04 item 4).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import kitti_box_error_census as c  # noqa: E402
import kitti_target_box_errors as kte  # noqa: E402

GT = np.array([2.0, 1.6, 20.0, 4.0, 1.5, 1.8, 0.4])  # l 4.0 along the length axis, w 1.8


def moved(b, along=0.0, across=0.0):
    """b shifted along / across ITS OWN length axis, built independently from bev_poly's corner rotation."""
    corners = np.array([[1.0, 0.0], [0.0, 1.0]])  # unit length / width axes in box coordinates
    ry = b[6]
    rot = np.array([[np.cos(ry), np.sin(ry)], [-np.sin(ry), np.cos(ry)]]).T
    ax, wx = corners @ rot                       # the same rotation bev_poly applies to its corners
    q = b.copy()
    q[[0, 2]] += along * ax + across * wx
    return q


def test_along_shift_is_along_the_length_axis():
    p = moved(GT, along=0.5)
    e = c.pair_errors(p, GT)
    assert e['along'] == pytest.approx(0.5) and e['across'] == pytest.approx(0.0, abs=1e-12)
    l, w = GT[3], GT[5]
    assert kte.ious(p, GT)[0] == pytest.approx((l - 0.5) / (l + 0.5), rel=1e-6)  # not (w - .5) / (w + .5)


def test_across_shift_is_across_the_length_axis():
    p = moved(GT, across=-0.3)
    e = c.pair_errors(p, GT)
    assert e['across'] == pytest.approx(-0.3) and e['along'] == pytest.approx(0.0, abs=1e-12)
    w = GT[5]
    assert kte.ious(p, GT)[0] == pytest.approx((w - 0.3) / (w + 0.3), rel=1e-6)
    assert e['centre_bev'] == pytest.approx(0.3)


@pytest.mark.parametrize('dyaw, yaw, yaw_pi', [(0.1, 0.1, 0.1), (np.pi + 0.1, -np.pi + 0.1, 0.1),
                                               (-np.pi + 0.05, -np.pi + 0.05, 0.05), (2 * np.pi - 0.2, -0.2, -0.2)])
def test_yaw_wraps(dyaw, yaw, yaw_pi):
    p = GT.copy(); p[6] += dyaw
    e = c.pair_errors(p, GT)
    assert e['yaw'] == pytest.approx(yaw) and e['yaw_pi'] == pytest.approx(yaw_pi)


def test_a_flipped_box_has_full_iou_and_counts_as_flipped():
    p = GT.copy(); p[6] += np.pi
    assert kte.ious(p, GT)[0] == pytest.approx(1.0, abs=1e-9)
    out = c.census([(p, GT)])
    assert out['share_heading_flipped'] == 1.0 and out['abs_yaw_pi_deg']['median'] == pytest.approx(0.0, abs=1e-9)


def test_vertical_and_size_signs():
    p = GT.copy(); p[1] -= 0.2; p[4] += 0.1; p[3] += 0.3; p[5] -= 0.1   # bottom 0.2 m higher (y down)
    e = c.pair_errors(p, GT)
    assert e['bottom_up'] == pytest.approx(0.2)
    assert e['centre_up'] == pytest.approx(0.2 + 0.05)
    assert (e['dl'], e['dh'], e['dw']) == pytest.approx((0.3, 0.1, -0.1))


def test_substitution_arms_put_back_exactly_one_component():
    p = moved(GT, along=0.6); p[6] += 0.3; p[3] *= 0.85; p[1] -= 0.3
    for w in c.SUBST:
        q = c.substitute(p, GT, w)
        changed = np.flatnonzero(~np.isclose(q, p))
        expect = {'centre_xz': [0, 2], 'yaw': [6], 'lw': [3], 'vertical': [1], 'lhw': [3], 'all': [0, 1, 2, 3, 6]}[w]
        assert list(changed) == expect, w
    assert kte.ious(c.substitute(p, GT, 'all'), GT) == pytest.approx((1.0, 1.0))
    with pytest.raises(ValueError):
        c.substitute(p, GT, 'nothing')


def test_census_shares_and_the_component_that_fails():
    good = moved(GT, along=0.05)                    # passes 0.7 in BEV and 3D
    along = moved(GT, along=1.0)                    # BEV 3/5 = 0.6 < 0.7; fixed by centre_xz
    up = GT.copy(); up[1] -= 0.5                    # BEV 1.0, 3D fails; fixed by vertical, not by centre_xz
    out = c.census([(good, GT), (along, GT), (up, GT)])
    assert out['n_matched'] == 3
    assert out['share_fail_bev_0.7'] == pytest.approx(1 / 3)
    assert out['share_fail_3d_0.7'] == pytest.approx(2 / 3)
    assert out['subst']['centre_xz']['share_fail_bev_0.7'] == 0.0
    assert out['subst']['centre_xz']['share_fail_3d_0.7'] == pytest.approx(1 / 3)
    assert out['subst']['vertical']['share_fail_bev_0.7'] == pytest.approx(1 / 3)
    assert out['subst']['all']['share_fail_3d_0.7'] == 0.0
    assert out['errors']['along']['median'] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        c.census([])


def _anno(boxes, h2d=40.0):
    boxes = np.atleast_2d(boxes)
    n = len(boxes)
    return {'name': np.array(['Car'] * n), 'bbox': np.tile([0.0, 0.0, 50.0, h2d], (n, 1)),
            'occluded': np.zeros(n), 'truncated': np.zeros(n), 'location': boxes[:, :3], 'dimensions': boxes[:, 3:6],
            'rotation_y': boxes[:, 6]}


def test_match_through_kte_respects_the_score_floor_and_the_gate(tmp_path):
    g2 = GT.copy(); g2[0] += 8.0
    gt_by_id = {'000001': _anno(np.stack([GT, g2]))}
    preds = np.stack([moved(GT, along=0.3), moved(g2, along=2.5), moved(GT, across=0.1)])
    det = {'frame_id': '000001', 'name': np.array(['Car'] * 3), 'score': np.array([0.9, 0.8, 0.2]),
           'location': preds[:, :3], 'dimensions': preds[:, 3:6], 'rotation_y': preds[:, 6]}
    path = tmp_path / 'result.pkl'
    pickle.dump([det], open(path, 'wb'))
    saved = kte.SCORE_MIN, kte.MATCH_BEV_IOU
    try:
        kte.SCORE_MIN, kte.MATCH_BEV_IOU = 0.3, 0.5
        pairs = kte.match(str(path), gt_by_id)      # the 2.5 m shift has BEV IoU 0.23: below the gate
        assert len(pairs) == 1 and np.allclose(pairs[0][1], GT)
        kte.SCORE_MIN, kte.MATCH_BEV_IOU = 0.1, 0.2
        pairs = kte.match(str(path), gt_by_id)      # both first two match; the 0.2-score box finds GT taken
        assert len(pairs) == 2
    finally:
        kte.SCORE_MIN, kte.MATCH_BEV_IOU = saved
    assert c.gt_counts(str(path), gt_by_id) == (1, 2)
