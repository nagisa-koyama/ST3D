"""Whole-track Car aggregation (pcdet/datasets/nuscenes/track_aggregation.py; experiments_md 20261010_04 §5.1 items 1-4,
10): sentinel cases of the bug-prevention checklist plus the ICP, culling and ray-drop contracts."""
import numpy as np
import pytest

from pcdet.datasets import rerender_utils as RR
from pcdet.datasets.nuscenes import track_aggregation as TA

CAR_DIMS = np.array([4.5, 1.8, 1.5])


def car_surface(n_side=400, seed=0):
    """Box-local points on a car-like shell: the -x face, the +y side and the top, inside a 4.5 x 1.8 x 1.5 box."""
    r = np.random.default_rng(seed)
    hx, hy, hz = CAR_DIMS / 2 - 0.05
    back = np.column_stack([np.full(n_side, -hx), r.uniform(-hy, hy, n_side), r.uniform(-hz, hz, n_side)])
    side = np.column_stack([r.uniform(-hx, hx, n_side), np.full(n_side, hy), r.uniform(-hz, hz, n_side)])
    top = np.column_stack([r.uniform(-hx, hx, n_side), r.uniform(-hy, hy, n_side), np.full(n_side, hz)])
    return np.concatenate([back, side, top])


def test_box_local_round_trip():
    box = np.array([12.0, -3.0, -0.9, 4.5, 1.8, 1.5, 0.7])
    p = np.random.default_rng(1).normal(size=(50, 3)) * 3 + box[:3]
    assert np.allclose(TA.from_box_local(TA.to_box_local(p, box), box), p)
    # x of the local frame runs along the heading
    ahead = box[:3] + np.array([np.cos(0.7), np.sin(0.7), 0.0])
    assert np.allclose(TA.to_box_local(ahead[None], box)[0], [1, 0, 0])


def test_keyframe_tracks_cars_with_tracks_only_and_each_point_once():
    boxes = np.array([[10, 0, 0, 4, 2, 2, 0], [10.5, 0, 0, 4, 2, 2, 0], [20, 0, 0, 4, 2, 2, 0], [30, 0, 0, 4, 2, 2, 0],
                      [40, 0, 0, 4, 2, 2, 0]], float)
    names = np.array(['car', 'car', 'truck', 'car', 'ignore'])
    tids = ['b', 'a', 'c', None, 'e']
    pts = np.array([[10.2, 0, 0, 7, 3], [20, 0, 0, 7, 3], [30, 0, 0, 7, 3], [40, 0, 0, 7, 3], [50, 0, 0, 7, 3]], float)
    tr = TA.keyframe_tracks(pts, boxes, names, tids)
    # the truck, the car without an instance token and the non-car name are never tracks
    assert set(tr) <= {'a', 'b'}
    # the point inside both overlapping car boxes is claimed once, by the first tid in sorted order
    assert set(tr) == {'a'} and len(tr['a'][0]) == 1
    assert np.allclose(tr['a'][1], [[7, 3]])


@pytest.mark.parametrize('pts,boxes', [(np.zeros((0, 5)), np.array([[0, 0, 0, 4, 2, 2, 0.0]])),
                                       (np.ones((3, 5)), np.zeros((0, 7))), (None, None)])
def test_keyframe_tracks_empty_inputs(pts, boxes):
    assert TA.keyframe_tracks(pts, boxes, ['car'], ['a']) == {}


def test_icp_recovers_a_small_rigid_error():
    src = car_surface()
    yaw, t = np.radians(2.0), np.array([0.20, -0.10, 0.05])
    tgt = TA.apply_4dof(src[::7], yaw, t) + np.random.default_rng(2).normal(scale=0.01, size=(len(src[::7]), 3))
    y, tt, ok, n = TA.icp_4dof(src, tgt)
    assert ok and n >= 5
    assert abs(np.degrees(y - yaw)) < 0.3 and np.allclose(tt, t, atol=0.03)


def test_icp_falls_back_with_too_few_points():
    src = car_surface()
    y, t, ok, n = TA.icp_4dof(src, src[:4] + 0.1)
    assert not ok and y == 0.0 and np.all(t == 0)


def test_icp_falls_back_outside_its_bounds():
    src = car_surface()
    tgt = src[::5] + np.array([0.8, 0.0, 0.0])          # a 0.8 m error is beyond |t_xy| <= 0.5 m
    y, t, ok, _ = TA.icp_4dof(src, tgt)
    assert not ok and np.all(t == 0)
    y, t, ok, _ = TA.icp_4dof(src, TA.apply_4dof(src[::5], np.radians(8.0), np.zeros(3)))  # yaw beyond 5 deg
    assert not ok


def test_icp_without_overlap_is_identity():
    y, t, ok, _ = TA.icp_4dof(car_surface(), car_surface()[:50] + np.array([0, 0, 5.0]))   # nothing within 1 m
    assert not ok and np.all(t == 0)


def test_backface_culling_keeps_the_faces_the_sensor_sees():
    box = np.array([10.0, 0.0, -1.0, 4.5, 1.8, 1.5, 0.0])
    sensor = np.array([0.0, 0.0, 0.434])                   # virtual Waymo TOP in the nuScenes lidar frame
    s_loc = TA.to_box_local(sensor[None], box)[0]
    vis = TA.visible_faces(s_loc, box[3:6])
    assert vis.tolist() == [False, True, False, False, True]    # near end (-x) and the top; not the far end or sides
    hx, hy, hz = box[3:6] / 2 - 0.02
    loc = np.array([[-hx, 0, 0], [hx, 0, 0], [0, hy, 0], [0, -hy, 0], [0, 0, hz], [-hx + 0.01, 0.3, -hz + 0.05]])
    assert TA.visible_mask(loc, box[3:6], s_loc).tolist() == [True, False, False, False, True, True]
    assert TA.visible_mask(np.zeros((0, 3)), box[3:6], s_loc).shape == (0,)


def _scene():
    """Three keyframes. Track A at every keyframe; B only in keyframes 1-2 (absent at anchor 0); C only at anchor 0;
    D at anchor 0 and keyframe 2 but with no points there (an empty track)."""
    surf = car_surface(100)
    box = np.array([15.0, 2.0, -1.0, 4.5, 1.8, 1.5, 0.3])
    frames = []
    for j in range(3):
        b = box.copy(); b[0] += j * 3.0
        tids, boxes, pts = ['A'], [b], {'A': (surf + 0.0, np.full((len(surf), 2), j, np.uint8))}
        if j >= 1:
            tids.append('B'); boxes.append(b + [0, 10, 0, 0, 0, 0, 0]); pts['B'] = (surf.copy(), np.zeros((len(surf), 2)))
        if j == 0:
            tids.append('C'); boxes.append(b + [0, -10, 0, 0, 0, 0, 0]); pts['C'] = (surf.copy(), np.zeros((len(surf), 2)))
        if j in (0, 2):
            tids.append('D'); boxes.append(b + [20, 0, 0, 0, 0, 0, 0])
        frames.append({'token': f't{j}', 't': float(j) / 2, 'tids': tids, 'boxes': np.array(boxes), 'points': pts})
    return {'frames': frames, 'icp': {}}


def test_aggregate_uses_other_keyframes_of_tracks_present_at_the_anchor():
    sc = _scene()
    pts, ids = TA.aggregate_anchor(sc, 0, np.array([0, 0, 0.434]), visibility=False, return_tids=True)
    # only A contributes: B is absent at the anchor, C has no other keyframe, D has no points anywhere else
    assert set(ids.tolist()) == {'A'}
    assert len(pts) == 2 * 300 and pts.shape[1] == 5
    # the points land on the ANCHOR's box of A, from keyframes 1 and 2 (attribute column = keyframe index)
    loc = TA.to_box_local(pts, sc['frames'][0]['boxes'][0])
    assert TA.inside_local(loc, CAR_DIMS + 1e-6).all()
    assert set(np.unique(pts[:, 3]).tolist()) == {1.0, 2.0}


def test_aggregate_single_keyframe_scene_is_empty():
    sc = _scene(); sc['frames'] = sc['frames'][:1]
    assert TA.aggregate_anchor(sc, 0, np.array([0, 0, 0.434])).shape == (0, 5)


def test_aggregate_applies_icp_and_culling():
    sc = _scene()
    sc['icp'] = {(0, 'A'): np.array([[1, 0.0, 0.3, 0.0, 0.0]])}      # shift keyframe 1's A by +0.3 m in local x
    pts = TA.aggregate_anchor(sc, 0, np.array([0, 0, 0.434]), visibility=False)
    loc = TA.to_box_local(pts, sc['frames'][0]['boxes'][0])
    from1, from2 = loc[pts[:, 3] == 1], loc[pts[:, 3] == 2]
    assert np.isclose(from1[:, 0].min() - from2[:, 0].min(), 0.3)
    # from the origin this car shows its -x end, its +y side and its top, i.e. every face of the test shell: nothing goes
    assert len(TA.aggregate_anchor(sc, 0, np.array([0, 0, 0.434]), icp=False, visibility=True)) == len(pts)
    # a sensor straight behind the car (local -x) sees the -x end and the top, not the +y side: the side is culled
    box = sc['frames'][0]['boxes'][0]
    behind = TA.from_box_local(np.array([[-20.0, 0.0, 1.434]]), box)[0]
    culled = TA.aggregate_anchor(sc, 0, behind, icp=False, visibility=True)
    assert len(culled) == 2 * 200
    assert np.allclose(TA.to_box_local(culled, box)[:, 1].max() < 0.85, True)


def test_raydrop_p0_keeps_all_p1_drops_every_cell_and_keeps_out_of_fov():
    th = RR.lattice_spec('waymo_top')['thetas']
    surf = TA.from_box_local(car_surface(200), np.array([25.0, 0, -1.0, 4.5, 1.8, 1.5, 0.0]))
    far_up = np.array([[5.0, 0.0, 30.0]])                              # far above Waymo TOP's vertical FOV
    xyz = np.concatenate([surf, far_up])
    edges, split = np.array([0.0, 20.0, 40.0, 75.0]), np.radians(-3.0)
    rng = np.random.default_rng(0)
    assert TA.raydrop_keep(xyz, np.zeros((3, 2)), edges, split, th, 2650, 0.434, rng).all()
    k1 = TA.raydrop_keep(xyz, np.ones((3, 2)), edges, split, th, 2650, 0.434, rng)
    assert not k1[:-1].any() and k1[-1]
    assert TA.raydrop_keep(np.zeros((0, 3)), np.ones((3, 2)), edges, split, th, 2650, 0.434, rng).shape == (0,)


def test_raydrop_drops_whole_cells():
    th = RR.lattice_spec('waymo_top')['thetas']
    p = np.array([[30.0, 0.0, 0.434]])
    xyz = np.concatenate([p, p * 1.001, p * 1.002])                     # three points in one cell
    edges, split = np.array([0.0, 20.0, 40.0, 75.0]), np.radians(-3.0)
    for seed in range(20):
        k = TA.raydrop_keep(xyz, np.full((3, 2), 0.5), edges, split, th, 2650, 0.434, np.random.default_rng(seed))
        assert k.all() or not k.any()


def test_build_scene_and_cache_round_trip(tmp_path):
    surf = car_surface(100)
    box0 = np.array([15.0, 2.0, -1.0, 4.5, 1.8, 1.5, 0.3]); box1 = box0 + [3, 0, 0, 0, 0, 0, 0.02]
    clouds = {'t0': np.column_stack([TA.from_box_local(surf[::10], box0), np.full((len(surf[::10]), 2), 9.0)]),
              't1': np.column_stack([TA.from_box_local(TA.apply_4dof(surf, 0.0, [0.1, 0, 0]), box1),
                                     np.full((len(surf), 2), 4.0)])}
    kfs = [{'token': 't0', 't': 0.0, 'boxes': box0[None], 'names': np.array(['car']), 'tids': ['A']},
           {'token': 't1', 't': 0.5, 'boxes': np.stack([box1, box1 + [0, 9, 0, 0, 0, 0, 0]]),
            'names': np.array(['car', 'truck']), 'tids': ['A', 'T']}]
    sc = TA.build_scene(kfs, clouds.get)
    assert [f['tids'] for f in sc['frames']] == [['A'], ['A']]
    # keyframe 1's A points sit 0.1 m off in its box: ICP (anchor 0, track A, keyframe 1) corrects about -0.1 m
    row = sc['icp'][(0, 'A')]
    assert row.shape == (1, 5) and row[0, 0] == 1 and np.isclose(row[0, 2], -0.1, atol=0.03)
    TA.save_scene(tmp_path / 's.pkl', sc)
    back = TA.load_scene(tmp_path / 's.pkl')
    assert back['frames'][1]['points']['A'][0].dtype == np.float16
    assert back['frames'][1]['points']['A'][1].dtype == np.uint8
    a1 = TA.aggregate_anchor(back, 0, np.array([0, 0, 0.434]), visibility=False)
    a2 = TA.aggregate_anchor(sc, 0, np.array([0, 0, 0.434]), visibility=False)
    assert a1.shape == a2.shape and np.abs(a1[:, :3] - a2[:, :3]).max() < 5e-3


def _two_cars_xyz():
    """Two aggregated cars side by side at 30 m (several lattice rows and many cells each), plus one point far above
    Waymo TOP's vertical field of view."""
    a = TA.from_box_local(car_surface(300, seed=3), np.array([30.0, 3.0, -1.0, 4.5, 1.8, 1.5, 0.0]))
    b = TA.from_box_local(car_surface(300, seed=4), np.array([30.0, -3.0, -1.0, 4.5, 1.8, 1.5, 0.0]))
    xyz = np.concatenate([a, b, [[5.0, 0.0, 30.0]]])
    ids = np.array(['a'] * len(a) + ['b'] * len(b) + ['a'], dtype=object)
    return xyz, ids


def test_rowrun_drop_q0_keeps_all_q1_drops_all_in_fov():
    th = RR.lattice_spec('waymo_top')['thetas']
    xyz, ids = _two_cars_xyz()
    rng = np.random.default_rng(0)
    assert TA.rowrun_keep(xyz, ids, 0.0, th, 2650, 0.434, rng).all()
    k = TA.rowrun_keep(xyz, ids, 1.0, th, 2650, 0.434, rng)
    assert not k[:-1].any() and k[-1]
    assert TA.rowrun_keep(np.zeros((0, 3)), np.zeros(0, dtype=object), 0.5, th, 2650, 0.434, rng).shape == (0,)


def test_rowrun_drop_removes_whole_object_rows_independently_per_object():
    from pcdet.datasets.processor.point_sampler import lattice_pixels
    th = RR.lattice_spec('waymo_top')['thetas']
    xyz, ids = _two_cars_xyz()
    _, row, _ = lattice_pixels(xyz, th, 2650, 0.434)
    seen_split = False
    for seed in range(10):
        k = TA.rowrun_keep(xyz, ids, 0.5, th, 2650, 0.434, np.random.default_rng(seed))
        for obj in ('a', 'b'):
            for r in np.unique(row[(ids == obj) & (row >= 0)]):
                g = k[(ids == obj) & (row == r)]
                assert g.all() or not g.any()                # an (object, row) group shares one fate
        # the same row of the two cars is decided independently at least once
        common = np.intersect1d(row[(ids == 'a') & (row >= 0)], row[(ids == 'b') & (row >= 0)])
        for r in common:
            if k[(ids == 'a') & (row == r)].all() != k[(ids == 'b') & (row == r)].all():
                seen_split = True
    assert seen_split


def test_rowrun_drop_rate_matches_q():
    th = RR.lattice_spec('waymo_top')['thetas']
    xyz, ids = _two_cars_xyz()
    from pcdet.datasets.processor.point_sampler import lattice_pixels
    _, row, _ = lattice_pixels(xyz, th, 2650, 0.434)
    groups = {(o, r) for o, r in zip(ids[row >= 0], row[row >= 0])}
    dropped = 0; total = 0
    for seed in range(400):
        k = TA.rowrun_keep(xyz, ids, 0.3, th, 2650, 0.434, np.random.default_rng(seed))
        for o, r in groups:
            dropped += not k[(ids == o) & (row == r)].any(); total += 1
    assert abs(dropped / total - 0.3) < 0.03


def test_range_matching_keeps_only_views_at_least_f_times_as_far():
    sc = _scene()             # track A at x = 15, 18, 21 m (+ y 2 m) in keyframes 0, 1, 2
    s = np.array([0, 0, 0.434])
    r = [np.hypot(*f['boxes'][0][:2]) for f in sc['frames']]
    full = TA.aggregate_anchor(sc, 2, s, visibility=False)
    assert np.array_equal(TA.aggregate_anchor(sc, 2, s, visibility=False, min_range_ratio=0.0), full)
    for f, expect in ((0.70, {0.0, 1.0}), (0.80, {1.0}), (0.90, set())):
        assert (r[0] / r[2] >= f) == (0.0 in expect) and (r[1] / r[2] >= f) == (1.0 in expect)
        got = TA.aggregate_anchor(sc, 2, s, visibility=False, min_range_ratio=f, tracks={'A'})
        assert set(np.unique(got[:, 3]).tolist()) == expect          # attribute column = source keyframe of A
    # farther views are always kept: from anchor 0 (the nearest), f = 1.0 still keeps keyframes 1 and 2
    got = TA.aggregate_anchor(sc, 0, s, visibility=False, min_range_ratio=1.0, tracks={'A'})
    assert set(np.unique(got[:, 3]).tolist()) == {1.0, 2.0}
