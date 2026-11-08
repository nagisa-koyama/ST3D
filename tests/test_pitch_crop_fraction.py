"""tools/analysis/pitch_crop_fraction.py (experiments_md 20261108_06, T05 §5 item 42d record 6): the share of an oracle's
points that the range crop removes once the evaluation cloud is pitched by OSENS-2's arm A.

Pinned on synthetic points against an independent rotation written here: the sign (d > 0 raises the points ahead), the
centre (SENSOR_ORIGIN, not the frame origin), the forward axis (y for nuScenes), the crop semantics (x / y closed by
mask_points_by_range and half-open by the voxel grid; z by the voxel grid only, which agrees with pcdet's
VoxelGeneratorWrapper), the counting, the twin configs read, and, when KITTI is on disk, the twin's own loader end to end.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'analysis'))
import _init_path  # noqa: F401,E402
import pitch_crop_fraction as P  # noqa: E402

PCR = np.array([-75.2, -75.2, -2.0, 75.2, 75.2, 4.0], np.float32)
VS = [0.1, 0.1, 0.15]
GRID = np.round((PCR[3:] - PCR[:3]) / np.array(VS)).astype(np.int64)


@pytest.fixture(autouse=True)
def _in_tools(monkeypatch):
    monkeypatch.chdir(TOOLS)


def ref_z(p, deg, origin, fwd):
    """Independent reference: z after a rotation about the lateral axis through `origin`, d > 0 raising points ahead."""
    f = 0 if fwd == 'x' else 1
    th = np.radians(deg)
    return origin[2] + np.sin(th) * (p[f] - origin[f]) + np.cos(th) * (p[2] - origin[2])


def cfg(deg, origin, fwd):
    return {'PITCH_DEG': deg, 'SENSOR_ORIGIN': list(origin), 'FORWARD_AXIS': fwd}


def counts(points, c):
    d = {'points': np.asarray(points, np.float32), 'gt_boxes': None, 'frame_id': 'synthetic'}
    return P.frame_counts(d['points'], P.pitched(d, c), PCR, VS, GRID)


@pytest.mark.parametrize('deg, high_leaves, ground_leaves', [(2.0, 1, 0), (1.0, 0, 0), (-1.0, 0, 0), (-2.0, 0, 1)])
def test_sign_kitti_frame(deg, high_leaves, ground_leaves):
    origin = (0.0, 0.0, 1.7)
    high = [50.0, 0.0, 3.0]     # +2 deg: z = 4.74 > 4; -2 deg: 1.25
    ground = [60.0, 0.0, 0.0]   # -2 deg: z = -2.09 < -2; +2 deg: 2.09
    assert (ref_z(high, deg, origin, 'x') >= 4.0) == bool(high_leaves)
    assert (ref_z(ground, deg, origin, 'x') < -2.0) == bool(ground_leaves)
    c = counts([high, ground], cfg(deg, origin, 'x'))
    assert c['in0'] == 2
    assert c['leave_z_high'] == high_leaves and c['leave_z_low'] == ground_leaves
    assert c['leave_z'] == c['leave_any'] == high_leaves + ground_leaves
    assert c['leave_xy'] == 0 and c['enter'] == 0


def test_centre_is_the_sensor_origin_not_the_frame_origin():
    origin, deg, x = (1.43, 0.0, 2.184), 2.0, 60.0           # Waymo TOP
    th = np.radians(deg)
    # z chosen so that the pitch about the SENSOR puts the point at 3.98 m (inside); about (0, 0, 0) it would be 4.03
    z = origin[2] + (3.98 - origin[2] - np.sin(th) * (x - origin[0])) / np.cos(th)
    p = [x, 0.0, z]
    assert abs(ref_z(p, deg, origin, 'x') - 3.98) < 1e-9
    assert ref_z(p, deg, (0.0, 0.0, 0.0), 'x') >= 4.0          # the wrong centre would make it leave
    assert counts([p], cfg(deg, origin, 'x'))['leave_any'] == 0
    assert counts([p], cfg(deg, (0.0, 0.0, 0.0), 'x'))['leave_any'] == 1


def test_forward_axis_y_for_nuscenes():
    origin = (0.0, 0.0, 1.75)
    p = [0.0, 50.0, 3.0]        # ahead along y in LIDAR_TOP
    assert ref_z(p, 2.0, origin, 'y') >= 4.0
    assert counts([p], cfg(2.0, origin, 'y'))['leave_z_high'] == 1
    assert counts([p], cfg(2.0, origin, 'x'))['leave_any'] == 0   # a wrong axis would keep it


def ref_xz(p, deg, origin):
    """Independent reference (x forward): (x, z) after the pitch about `origin`."""
    th = np.radians(deg)
    dx, dz = p[0] - origin[0], p[2] - origin[2]
    return origin[0] + np.cos(th) * dx - np.sin(th) * dz, origin[2] + np.sin(th) * dx + np.cos(th) * dz


def ref_inside(x, y, z):
    return (-75.2 <= x < 75.2) and (-75.2 <= y < 75.2) and (-2.0 <= z < 4.0)


@pytest.mark.parametrize('deg', [-2.0, -1.0, 1.0, 2.0])
def test_xy_leave_and_enter_match_the_reference(deg):
    origin = (0.0, 0.0, 1.7)
    # near the x edge, low and high: the pitch moves x by cos * dx - sin * dz, so some cross the edge either way
    pts = [[x, 0.0, z] for x in (75.10, 75.15, 75.19, 75.21, 75.25, 75.30) for z in (-1.9, -1.0, 0.0, 1.0, 2.5, 3.9)]
    c = counts(pts, cfg(deg, origin, 'x'))
    exp = dict(in0=0, leave_xy=0, leave_z=0, leave_any=0, enter=0)
    for p in pts:
        x1, z1 = ref_xz(p, deg, origin)
        before, after = ref_inside(*p), ref_inside(x1, 0.0, z1)
        exp['in0'] += before
        exp['leave_any'] += before and not after
        exp['leave_xy'] += before and not (-75.2 <= x1 < 75.2)
        exp['leave_z'] += before and not (-2.0 <= z1 < 4.0)
        exp['enter'] += (not before) and after
    assert {k: c[k] for k in exp} == exp
    assert exp['leave_xy'] + exp['enter'] > 0      # the edge is actually exercised at every angle


def test_mask_points_by_range_ignores_z_and_the_grid_does_not():
    pts = np.array([[10.0, 0.0, 5.0], [10.0, 0.0, -2.5], [10.0, 0.0, 1.0]], np.float32)
    in_xy, in_z, low, high = P.in_range_masks(pts, PCR, VS, GRID)
    assert in_xy.all()
    assert in_z.tolist() == [False, False, True]
    assert low.tolist() == [False, True, False] and high.tolist() == [True, False, False]


def test_crop_agrees_with_the_voxel_generator():
    from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(-80, 80, 3000), rng.uniform(-80, 80, 3000), rng.uniform(-3, 5, 3000)], 1)
    keep = np.ones(len(pts), bool)
    for j, lo, hi in ((0, -75.2, 75.2), (1, -75.2, 75.2), (2, -2.0, 4.0)):   # keep clear of float32 boundary rounding
        keep &= (np.abs(pts[:, j] - lo) > 1e-3) & (np.abs(pts[:, j] - hi) > 1e-3)
    pts = pts[keep].astype(np.float32)
    gen = VoxelGeneratorWrapper(vsize_xyz=VS, coors_range_xyz=PCR, num_point_features=3, max_num_points_per_voxel=20,
                                max_num_voxels=20000)
    _, _, num = gen.generate(pts)
    in_xy, in_z, _, _ = P.in_range_masks(pts, PCR, VS, GRID)
    assert int(np.asarray(num).sum()) == int((in_xy & in_z).sum())
    assert 0 < int((in_xy & in_z).sum()) < len(pts)


def test_summarise():
    rows = [dict(n=10, in0=8, out0_xy=1, out0_z=1, out0_any=2, in1=7, leave_z=2, leave_z_low=1, leave_z_high=1,
                 leave_xy=0, leave_any=2, enter=1),
            dict(n=20, in0=20, out0_xy=0, out0_z=0, out0_any=0, in1=20, leave_z=0, leave_z_low=0, leave_z_high=0,
                 leave_xy=0, leave_any=0, enter=0)]
    s = P.summarise(rows)
    assert s['leave_any']['mean'] == pytest.approx((2 / 8 + 0) / 2)
    assert s['leave_any']['pooled'] == pytest.approx(2 / 28)
    assert s['net']['pooled'] == pytest.approx(-1 / 28)
    assert s['out0_any']['pooled'] == pytest.approx(2 / 30)


def test_frame_indices_even_spread():
    idx = P.frame_indices(7998, 300)
    assert len(idx) == 300 and idx[0] == 0 and idx[-1] == 7997
    assert np.all(np.diff(idx) >= 26) and np.all(np.diff(idx) <= 28)


def test_loader_input_restores_the_method():
    class D:
        def prepare_data(self, data_dict):
            return {'processed': True}

        def __getitem__(self, i):
            return self.prepare_data(data_dict={'points': np.zeros((1, 3)), 'i': i})
    d = D()
    raw = P.loader_input(d, 3)
    assert raw['i'] == 3 and 'processed' not in raw
    assert d[0] == {'processed': True}


@pytest.mark.parametrize('oracle, origin, fwd', [('waymo', [1.43, 0.0, 2.184], 'x'), ('nuscenes', [0.0, 0.0, 1.75], 'y'),
                                                 ('kitti', [0.0, 0.0, 1.7], 'x')])
def test_twin_configs_read(oracle, origin, fwd):
    twins = P.twin_osens2_cfgs(oracle)
    assert [t[1] for t in twins] == [-2.0, -1.0, 1.0, 2.0]
    for _, _, block in twins:
        assert list(map(float, block['SENSOR_ORIGIN'])) == pytest.approx(origin) and block['FORWARD_AXIS'] == fwd


@pytest.mark.skipif(not (TOOLS.parent / 'data/kitti/kitti_infos_val.pkl').exists(), reason='KITTI infos not on disk')
def test_kitti_twin_loader_end_to_end():
    base = P.cfg_of(P.base_config_path('kitti'))
    ds = P.dataset_of(base)
    twins = [t for t in P.twin_osens2_cfgs('kitti') if t[0] in ('pitchm2', 'pitchp2')]
    res = P.check_twin_end_to_end('kitti', twins, ds, [100])
    assert all(r['equal'] for r in res) and all(r['n_twin'] > 0 for r in res)
