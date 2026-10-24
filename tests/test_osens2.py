"""OSENS-2 (experiments_md 20261025_03; design 20261003_05 §5, 2026-10-25 01:34 JST): the eval-only perturbations of
the target oracles' evaluation clouds. ANALYSIS.

Pinned here, on synthetic clouds and (when the data is on disk) on real KITTI / nuScenes val frames through the real
loaders and twin configs:
  - pitch (EVAL_OSENS2.PITCH_DEG): a rigid rotation about the lateral axis through SENSOR_ORIGIN, positive = points
    ahead rise (x-forward and y-forward frames), only x, y, z of the POINTS move; GT boxes and prediction frames are the
    oracle's;
  - height (SHIFT_COOR twin): points and GT move by exactly dz, predictions come back to the infos' frame exactly as
    SHIFT_COOR un-shifts them;
  - background / foreground thinning (REGION_THIN) with GT boxes enlarged by BOX_MARGIN (enlarge_box3d convention): the
    other set is untouched, the thinned set keeps round(f * n) points, seeded by the frame id; sentinels (no box, no
    point, nuScenes' 9-column boxes);
  - ground removal (GROUND_REMOVE_DIST): only outside points near the fitted plane go; in-box points stay; a failed
    fit removes nothing; the estimator is ground_tilt.py's, verbatim;
  - nuScenes EVAL_KEEP_EGO_POINTS: evaluation only, and the extra points are exactly the |x|, |y| < 1.5 m square;
  - the hook runs at evaluation only, and the twins differ from their oracle in DATA_CONFIG_TAR only.
"""
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(TOOLS))
import _init_path  # noqa: F401,E402

from pcdet.datasets import osens2_utils as O  # noqa: E402


@pytest.fixture(autouse=True)
def _in_tools(monkeypatch):
    monkeypatch.chdir(TOOLS)      # configs and the loaders' relative data paths resolve from tools/


# ------------------------------------------------------------------------------------------------------------- pitch
@pytest.mark.parametrize('fwd', ['x', 'y'])
@pytest.mark.parametrize('deg', [-2.0, -1.0, 1.0, 2.0])
def test_pitch_raises_points_ahead_for_positive_angle(fwd, deg):
    origin = np.array([1.43, 0.0, 2.184])
    f = 0 if fwd == 'x' else 1
    p = np.zeros((1, 5), np.float32)
    p[0, :3] = origin
    p[0, f] += 30.0
    p[0, 3:] = (0.7, 0.2)
    q = O.pitch_points(p, deg, origin, fwd)
    assert np.isclose(q[0, 2] - origin[2], 30.0 * np.sin(np.radians(deg)), atol=1e-4)
    assert np.isclose(q[0, f] - origin[f], 30.0 * np.cos(np.radians(deg)), atol=1e-4)
    assert np.array_equal(q[0, 3:], p[0, 3:])            # features untouched
    lateral = 1 - f
    assert q[0, lateral] == p[0, lateral]               # the rotation axis coordinate does not move


def test_pitch_is_rigid_about_the_origin_and_zero_is_identity():
    rng = np.random.default_rng(0)
    p = rng.uniform(-50, 50, (2000, 4)).astype(np.float32)
    origin = np.array([0.0, 0.0, 1.75])
    q = O.pitch_points(p, 2.0, origin, 'y')
    d0 = np.linalg.norm(p[:, :3] - origin, axis=1)
    d1 = np.linalg.norm(q[:, :3] - origin, axis=1)
    assert np.allclose(d0, d1, atol=2e-3)
    assert np.allclose(O.pitch_points(p, 0.0, origin, 'x'), p)
    back = O.pitch_points(q, -2.0, origin, 'y')
    assert np.allclose(back, p, atol=2e-3)


def test_pitch_makes_a_flat_ground_rise_ahead_by_the_angle():
    rng = np.random.default_rng(1)
    g = np.c_[rng.uniform(5, 40, 3000), rng.uniform(-20, 20, 3000), np.zeros(3000)].astype(np.float32)
    q = O.pitch_points(g, 1.5, np.array([0.0, 0.0, 1.7]), 'x')
    a = np.linalg.lstsq(np.c_[q[:, 0], q[:, 1], np.ones(len(q))], q[:, 2], rcond=None)[0]
    assert np.isclose(np.degrees(np.arctan(a[0])), 1.5, atol=1e-3) and abs(a[1]) < 1e-6


def test_pitch_leaves_gt_boxes_alone():
    pts = np.random.default_rng(2).uniform(-30, 30, (500, 4)).astype(np.float32)
    boxes = np.array([[10, 0, 0.8, 4, 2, 1.6, 0.3]], np.float32)
    dd = {'points': pts.copy(), 'gt_boxes': boxes.copy(), 'frame_id': '000001'}
    cfg = {'PITCH_DEG': 2.0, 'SENSOR_ORIGIN': [0.0, 0.0, 1.7], 'FORWARD_AXIS': 'x'}
    out = O.apply_in_prepare_data(dd, cfg)
    assert np.array_equal(out['gt_boxes'], boxes) and not np.array_equal(out['points'], pts)


# ------------------------------------------------------------------------------------------------- region thinning
def _scene(seed=3, n=20000):
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.uniform(-40, 40, (n, 2)), rng.uniform(-0.2, 3.0, n), rng.uniform(0, 1, n)].astype(np.float32)
    boxes = np.array([[10, 5, 0.8, 4.5, 2.0, 1.6, 0.4], [-12, -3, 0.9, 0.8, 0.8, 1.8, -1.0],
                      [25, -10, 1.5, 12, 3, 3.2, 2.0]], np.float32)
    return pts, boxes


def _inside(pts, boxes, margin):
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    from pcdet.utils import box_utils
    import torch
    big = box_utils.enlarge_box3d(torch.from_numpy(boxes[:, :7]), extra_width=(margin, margin, margin))
    return roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(pts[:, :3]), big).numpy().any(0)


def test_in_any_box_matches_pcdet_with_enlarge_box3d():
    pts, boxes = _scene()
    got = O.in_any_box(pts, O.enlarge_boxes(boxes, 0.25))
    assert np.array_equal(got, _inside(pts, boxes, 0.25)) and 0 < got.sum() < len(pts)


def test_in_any_box_chunks_many_boxes(monkeypatch):
    pts, boxes = _scene()
    many = np.repeat(boxes, 50, axis=0)
    many[:, 0] += np.repeat(np.arange(50) * 0.01, 3)[np.argsort(np.tile(np.arange(3), 50), kind='stable')][: len(many)]
    monkeypatch.setattr(O, 'BOX_CHUNK', 7)
    assert np.array_equal(O.in_any_box(pts, many), _inside(pts, many, 0.0))


def test_enlargement_is_full_extent_not_per_side():
    # a car box 4 m long at the origin: a point 0.10 m past its front face is inside the box grown by 0.25 m
    # (0.125 m per side), one 0.20 m past it is not
    box = np.array([[0, 0, 1, 4, 2, 1.5, 0]], np.float32)
    p = np.array([[2.10, 0, 1, 0], [2.20, 0, 1, 0]], np.float32)
    assert O.in_any_box(p, O.enlarge_boxes(box, 0.25)).tolist() == [True, False]


@pytest.mark.parametrize('side', ['background', 'foreground'])
@pytest.mark.parametrize('keep', [0.5, 0.25])
def test_region_thin_touches_one_side_only(side, keep):
    pts, boxes = _scene()
    cfg = {'REGION_THIN': {'SIDE': side, 'KEEP': keep}, 'BOX_MARGIN': 0.25}
    out, st = O.perturb(pts, boxes, '000123', cfg)
    inside = _inside(pts, boxes, 0.25)
    out_in = _inside(out, boxes, 0.25)
    if side == 'background':
        assert out_in.sum() == inside.sum()                                   # every in-box point kept
        assert (~out_in).sum() == int(round((~inside).sum() * keep))
        assert set(map(tuple, out[out_in])) == set(map(tuple, pts[inside]))
    else:
        assert (~out_in).sum() == (~inside).sum()                             # every outside point kept
        assert out_in.sum() == int(round(inside.sum() * keep))
        assert set(map(tuple, out[~out_in])) == set(map(tuple, pts[~inside]))
    assert st['n_out'] == len(out)
    # rows are a subset of the input, in input order
    idx = np.flatnonzero((pts[:, None, :] == out[None, :50, :]).all(-1).any(1))
    assert np.all(np.diff(idx) > 0)


def test_region_thin_is_seeded_by_the_frame_id():
    pts, boxes = _scene()
    cfg = {'REGION_THIN': {'SIDE': 'background', 'KEEP': 0.5}, 'BOX_MARGIN': 0.25}
    a, _ = O.perturb(pts, boxes, 'seg_a_100', cfg)
    b, _ = O.perturb(pts, boxes, 'seg_a_100', cfg)
    c, _ = O.perturb(pts, boxes, 'seg_a_101', cfg)
    assert np.array_equal(a, b) and len(a) == len(c) and not np.array_equal(a, c)


def test_region_thin_leaves_boxes_alone_and_takes_nuscenes_nine_columns():
    pts, boxes = _scene()
    b9 = np.c_[boxes, np.zeros((len(boxes), 2), np.float32)]
    dd = {'points': pts.copy(), 'gt_boxes': b9.copy(), 'frame_id': 'n008-2018-08-01'}
    out = O.apply_in_prepare_data(dd, {'REGION_THIN': {'SIDE': 'foreground', 'KEEP': 0.25}, 'BOX_MARGIN': 0.25})
    assert np.array_equal(out['gt_boxes'], b9) and len(out['points']) < len(pts)


@pytest.mark.parametrize('side,expect', [('background', 0.5), ('foreground', 1.0)])
def test_region_thin_with_no_box(side, expect):
    pts, _ = _scene()
    for boxes in (np.zeros((0, 7), np.float32), None):
        out, _ = O.perturb(pts, boxes, 'x', {'REGION_THIN': {'SIDE': side, 'KEEP': 0.5}, 'BOX_MARGIN': 0.25})
        assert len(out) == int(round(len(pts) * expect))


@pytest.mark.parametrize('cfg', [{'REGION_THIN': {'SIDE': 'background', 'KEEP': 0.5}, 'BOX_MARGIN': 0.25},
                                 {'GROUND_REMOVE_DIST': 0.2, 'BOX_MARGIN': 0.25},
                                 {'PITCH_DEG': 1.0, 'SENSOR_ORIGIN': [0, 0, 1.7], 'FORWARD_AXIS': 'x'}])
def test_empty_cloud(cfg):
    _, boxes = _scene()
    out, _ = O.perturb(np.zeros((0, 4), np.float32), boxes, 'x', cfg)
    assert out.shape == (0, 4)


def test_exactly_one_arm():
    pts, boxes = _scene()
    with pytest.raises(AssertionError, match='exactly one'):
        O.perturb(pts, boxes, 'x', {'PITCH_DEG': 1.0, 'GROUND_REMOVE_DIST': 0.2, 'BOX_MARGIN': 0.25})
    with pytest.raises(AssertionError, match='exactly one'):
        O.perturb(pts, boxes, 'x', {'BOX_MARGIN': 0.25})


# ----------------------------------------------------------------------------------------------------------- ground
def _ground_scene(seed=4):
    rng = np.random.default_rng(seed)
    n = 30000
    xy = rng.uniform(-40, 40, (n, 2))
    a, b, c = np.tan(np.radians(1.2)), np.tan(np.radians(-0.6)), 0.05
    z = a * xy[:, 0] + b * xy[:, 1] + c + rng.normal(0, 0.02, n)
    ground = np.c_[xy, z, np.zeros(n)]
    walls = np.c_[rng.uniform(-40, 40, (5000, 2)), rng.uniform(0.5, 4.0, 5000), np.zeros(5000)]
    walls[:, 2] += a * walls[:, 0] + b * walls[:, 1] + c
    box = np.array([[12.0, 4.0, a * 12 + b * 4 + c + 0.8, 4.5, 2.0, 1.6, 0.2]], np.float32)
    under = np.c_[rng.uniform(10.5, 13.5, 300), rng.uniform(3.5, 4.5, 300), np.zeros(300), np.zeros(300)]
    under[:, 2] = a * under[:, 0] + b * under[:, 1] + c + 0.05                    # ground points inside the box
    pts = np.r_[ground, walls, under].astype(np.float32)
    return pts, box, (a, b, c)


def test_ground_removal_drops_only_outside_points_near_the_plane():
    pts, box, (a, b, c) = _ground_scene()
    out, st = O.perturb(pts, box, 'f', {'GROUND_REMOVE_DIST': 0.2, 'BOX_MARGIN': 0.25})
    assert st['fit_failed'] == 0
    assert np.allclose(st['plane'], [a, b, c], atol=0.01)
    inside = _inside(pts, box, 0.25)
    res = pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)
    expect_drop = (~inside) & (np.abs(res) < 0.2)
    assert abs(st['n_dropped'] - expect_drop.sum()) <= 0.002 * len(pts)       # the fitted plane, not the true one
    out_inside = _inside(out, box, 0.25)
    assert out_inside.sum() == inside.sum()                                   # in-box ground points stay
    res_out = out[:, 2] - (a * out[:, 0] + b * out[:, 1] + c)
    assert np.mean(np.abs(res_out[~out_inside]) < 0.15) < 0.001               # no ground left outside
    assert (res_out[~out_inside] > 0.5).sum() == (res[~inside] > 0.5).sum()    # everything well above stays


def test_ground_fit_failure_removes_nothing():
    pts = np.array([[5.0, 0, 0, 0], [6.0, 0, 0, 0]], np.float32)
    out, st = O.perturb(pts, np.zeros((0, 7), np.float32), 'f', {'GROUND_REMOVE_DIST': 0.2, 'BOX_MARGIN': 0.25})
    assert st['fit_failed'] == 1 and np.array_equal(out, pts)


def test_ground_estimator_is_ground_tilt_verbatim():
    sys.path.insert(0, str(TOOLS / 'analysis'))
    import ground_tilt
    pts, _, _ = _ground_scene(5)
    g0, g1 = ground_tilt.lowest_per_cell(pts[:, :3].astype(np.float64)), O.lowest_per_cell(pts[:, :3].astype(np.float64))
    assert np.array_equal(g0, g1)
    f0, f1 = ground_tilt.fit_plane(g0), O.fit_plane(g1)
    assert np.array_equal(f0[0], f1[0]) and f0[1:] == f1[1:]


# ------------------------------------------------------------------------------------------- real loaders and twins
def _cfg(path):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        return cfg_from_yaml_file(path, EasyDict())
    finally:
        os.chdir(cwd)


def _dataset(cfg, training=False, strip=True):
    from pcdet.datasets import build_dataloader
    log = logging.getLogger('osens2'); log.setLevel(logging.WARNING)
    d = copy.deepcopy(cfg.DATA_CONFIG_TAR)
    if strip:
        d.DATA_PROCESSOR = []           # no crop, shuffle or voxels: compare the clouds point for point
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        ds, _, _ = build_dataloader(dataset_cfg=d, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                    logger=log, training=training, model_ontology=cfg.get('ONTOLOGY', None))
    finally:
        os.chdir(cwd)
    return ds


KITTI = 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml'
NUSC = 'cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml'
TWIN = 'cfgs/da-ieee-access-analysis/osens2/{}-osens2-{}.yaml'
needs_kitti = pytest.mark.skipif(not (ROOT / 'data/kitti/training/velodyne').is_dir(), reason='KITTI data not on disk')
needs_nusc = pytest.mark.skipif(not (ROOT / 'data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl').exists(),
                                reason='nuScenes infos not on disk')


def test_twins_resolve_and_differ_in_the_target_block_only():
    import subprocess
    r = subprocess.run([sys.executable, 'analysis/make_osens2_configs.py', '--check'], cwd=TOOLS, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert r.stdout.count(': OK') == 46


def _kitti_pred_dicts(ds, boxes, frame):
    import torch
    batch = {'calib': [frame['calib']], 'image_shape': [frame['image_shape']], 'frame_id': [frame['frame_id']]}
    pred = [{'pred_boxes': torch.from_numpy(boxes[:, :7].copy()), 'pred_scores': torch.ones(len(boxes)),
             'pred_labels': torch.ones(len(boxes), dtype=torch.long)}]
    return ds.generate_prediction_dicts(batch, pred, ['Car', 'Pedestrian', 'Cyclist'])[0]


@needs_kitti
@pytest.mark.parametrize('tag,dz', [('dzp030', 0.30), ('dzm045', -0.45)])
def test_kitti_height_twin_moves_points_and_gt_and_unshifts_predictions(tag, dz):
    base, twin = _cfg(KITTI), _cfg(TWIN.format('centerpoint-sourceonly-kitti2kitti-noros-zshift', tag))
    db, dt = _dataset(base), _dataset(twin)
    fb, ft = db[3], dt[3]
    assert np.allclose(ft['points'][:, 2] - fb['points'][:, 2], dz, atol=1e-5)
    assert np.array_equal(ft['points'][:, :2], fb['points'][:, :2])
    assert np.allclose(ft['gt_boxes'][:, 2] - fb['gt_boxes'][:, 2], dz, atol=1e-5)
    assert np.array_equal(ft['gt_boxes'][:, [0, 1, 3, 4, 5, 6, 7]], fb['gt_boxes'][:, [0, 1, 3, 4, 5, 6, 7]])
    # a perfect prediction in each model frame comes back to the SAME boxes in the infos' frame
    pb, pt = _kitti_pred_dicts(db, fb['gt_boxes'], fb), _kitti_pred_dicts(dt, ft['gt_boxes'], ft)
    assert np.allclose(pb['boxes_lidar'], pt['boxes_lidar'], atol=1e-5)
    assert np.allclose(pb['location'], pt['location'], atol=1e-5)


@needs_kitti
def test_kitti_pitch_twin_rotates_points_only():
    base = _cfg(KITTI)
    twin = _cfg(TWIN.format('centerpoint-sourceonly-kitti2kitti-noros-zshift', 'pitchp2'))
    db, dt = _dataset(base), _dataset(twin)
    fb, ft = db[3], dt[3]
    assert np.array_equal(ft['gt_boxes'], fb['gt_boxes'])
    assert np.allclose(ft['points'], O.pitch_points(fb['points'], 2.0, [0.0, 0.0, 1.7], 'x'), atol=1e-5)
    pb, pt = _kitti_pred_dicts(db, fb['gt_boxes'], fb), _kitti_pred_dicts(dt, fb['gt_boxes'], ft)
    assert np.array_equal(pb['boxes_lidar'], pt['boxes_lidar'])


@needs_kitti
@pytest.mark.parametrize('tag', ['bgkeep025', 'fgkeep050', 'groundrm020'])
def test_kitti_c_twins_keep_gt_and_cut_points(tag):
    base = _cfg(KITTI)
    twin = _cfg(TWIN.format('centerpoint-sourceonly-kitti2kitti-noros-zshift', tag))
    db, dt = _dataset(base), _dataset(twin)
    fb, ft = db[3], dt[3]
    assert np.array_equal(ft['gt_boxes'], fb['gt_boxes'])
    assert 0 < len(ft['points']) < len(fb['points'])
    # the twin's cloud is the hook applied to the base cloud with ALL the loader's boxes (Van, Misc ... included)
    info = db.kitti_infos[3]
    from pcdet.utils import box_utils, common_utils
    annos = common_utils.drop_info_with_name(info['annos'], name='DontCare')
    cam = np.concatenate([annos['location'], annos['dimensions'], annos['rotation_y'][:, None]], 1).astype(np.float32)
    all_boxes = box_utils.boxes3d_kitti_camera_to_lidar(cam, db.get_calib(info['point_cloud']['lidar_idx']))
    all_boxes[:, 2] += 1.7
    expect, _ = O.perturb(fb['points'], all_boxes, fb['frame_id'], twin.DATA_CONFIG_TAR.EVAL_OSENS2)
    assert np.array_equal(ft['points'], expect)


@needs_kitti
def test_hook_is_ignored_in_training():
    twin = _cfg(TWIN.format('centerpoint-sourceonly-kitti2kitti-noros-zshift', 'bgkeep025'))
    d = _dataset(twin, training=True)
    called = []
    import pcdet.datasets.osens2_utils as mod
    orig = mod.apply_in_prepare_data
    mod.apply_in_prepare_data = lambda *a, **k: called.append(1) or orig(*a, **k)
    try:
        d[0]
    finally:
        mod.apply_in_prepare_data = orig
    assert not called


@needs_nusc
def test_nuscenes_keep_ego_points_adds_exactly_the_ego_square():
    base = _cfg(NUSC)
    twin = _cfg(TWIN.format('centerpoint-sourceonly-nuscenes', 'keepego'))
    db, dt = _dataset(base), _dataset(twin)
    pb = db.get_lidar_with_sweeps(5, max_sweeps=1)
    pt = dt.get_lidar_with_sweeps(5, max_sweeps=1)
    ego = (np.abs(pt[:, 0]) < 1.5) & (np.abs(pt[:, 1]) < 1.5)
    assert ego.sum() > 1000 and np.array_equal(pt[~ego], pb)
    # training ignores the key: the (train-split) frame still has its ego square removed
    ptr = _dataset(twin, training=True).get_lidar_with_sweeps(5, max_sweeps=1)
    assert not ((np.abs(ptr[:, 0]) < 1.5) & (np.abs(ptr[:, 1]) < 1.5)).any()


@needs_nusc
def test_nuscenes_pitch_twin_uses_the_y_forward_axis_and_keeps_gt():
    base = _cfg(NUSC)
    twin = _cfg(TWIN.format('centerpoint-sourceonly-nuscenes', 'pitchm1'))
    assert twin.DATA_CONFIG_TAR.EVAL_OSENS2.FORWARD_AXIS == 'y'
    assert list(twin.DATA_CONFIG_TAR.EVAL_OSENS2.SENSOR_ORIGIN) == [0.0, 0.0, 1.75]
    db, dt = _dataset(base), _dataset(twin)
    fb, ft = db[5], dt[5]
    assert np.array_equal(ft['gt_boxes'], fb['gt_boxes'])
    assert np.allclose(ft['points'], O.pitch_points(fb['points'], -1.0, [0.0, 0.0, 1.75], 'y'), atol=1e-5)
    # nuScenes LIDAR_TOP: the vehicle's forward axis is the sensor's +y (ref_from_car maps vehicle x to sensor y)
    R = np.asarray(db.infos[5]['ref_from_car'])[:3, :3]
    assert abs((R @ np.array([1.0, 0, 0]))[1]) > 0.99
