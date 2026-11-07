"""REFLECTION_DROP, the reflection rows R11 (threshold) / R11r (count-matched random twin) of the factor-swap oracles
(pcdet/datasets/reflection_drop.py and its hooks; experiments_md 20261003_05 §5 item 35g). CPU only.

Pins: the classification is M1's (tools/analysis/reflection_raydrop_m1.py, --ground plane) bit for bit; the threshold
drops below T, ties at T with probability Q, nothing above; ties of a constant intensity (KITTI's zeros beyond 40 m)
are dropped at random with the pooled share; E <= 0 / null cells and unclassified returns (ego, beyond 75 m, below the
ground band, above 3 m) are never dropped; draws are fixed per frame and differ between modes; the Waymo arm touches TOP
returns only and recounts exactly the boxes it empties; Waymo's get_lidar is unchanged without the TOP mask; Lyft
refuses the arm; malformed configs fail.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets import reflection_drop as R  # noqa: E402
from pcdet.datasets import loader_factor_swap as lfs  # noqa: E402
from pcdet.datasets.kitti import kitti_factor_swap as kfs  # noqa: E402
from pcdet.datasets.waymo import waymo_factor_swap as wfs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, '..', 'tools', 'analysis', 'reflection_raydrop_m1.py')
spec = importlib.util.spec_from_file_location('reflection_raydrop_m1_for_r11', TOOL)
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)

H = -1.8      # ground z in the sensor frame


def frame(seed=0, n_ground=20000, n_above=6000, tilt=0.02):
    """A sensor-frame cloud: a tilted road plane (z = H + tilt * x), objects 0.3-3 m above it, plus returns the arm must
    never touch (ego < 3 m, beyond 75 m, 3-6 m above ground, 0.5-1 m below it). Intensity 0-255 integers."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n_ground); rr = rng.uniform(3.5, 74.0, n_ground)
    g = np.c_[rr * np.cos(a), rr * np.sin(a), np.zeros(n_ground)]
    g[:, 2] = H + tilt * g[:, 0] + rng.normal(0, 0.02, n_ground)
    a = rng.uniform(0, 2 * np.pi, n_above); rr = rng.uniform(4.0, 70.0, n_above)
    o = np.c_[rr * np.cos(a), rr * np.sin(a), np.zeros(n_above)]
    o[:, 2] = H + tilt * o[:, 0] + rng.uniform(0.5, 2.8, n_above)
    ego = np.c_[rng.uniform(-1.5, 1.5, (200, 2)), rng.uniform(-1, 0, 200)]
    far = np.c_[np.full(200, 80.0), rng.uniform(-5, 5, 200), np.full(200, H + tilt * 80.0)]
    high = np.c_[rng.uniform(10, 40, 300), rng.uniform(-10, 10, 300), np.zeros(300)]
    high[:, 2] = H + tilt * high[:, 0] + rng.uniform(3.2, 6.0, 300)
    low = np.c_[rng.uniform(10, 40, 300), rng.uniform(-10, 10, 300), np.zeros(300)]
    low[:, 2] = H + tilt * low[:, 0] - rng.uniform(0.5, 1.0, 300)
    xyz = np.concatenate([g, o, ego, far, high, low])
    inten = rng.integers(0, 256, len(xyz)).astype(np.float64)
    tag = np.concatenate([np.zeros(n_ground + n_above), np.ones(200 + 200 + 300 + 300)]).astype(bool)   # untouched
    return xyz, inten, tag


def table(v_ground, v_above):
    f = (lambda v: list(v) if isinstance(v, (list, tuple)) else [v] * R.N_BINS)
    return {'ground': f(v_ground), 'above': f(v_above)}


def cfg(mode='threshold', e=0.1, t=50.0, q=0.5, **kw):
    c = {'MODE': mode, 'E': table(e, e)}
    if mode == 'threshold':
        c.update(T=table(t, t), Q=table(q, q))
    c.update(kw)
    return EasyDict(c)


# ------------------------------------------------------------------------------------------- classification = M1
def test_bins_and_classes_are_m1s():
    r = np.array([-1.0, 0.0, 2.9, 3.0, 9.99, 10.0, 59.9, 60.0, 74.99, 75.0, 120.0])
    np.testing.assert_array_equal(R.range_bin(r), m1.range_bin(r))
    dz = np.array([-0.31, -0.3, 0.0, 0.3, 0.30001, 2.9, 3.0, 3.01])
    np.testing.assert_array_equal(R.height_class(dz), m1.height_class(dz))
    assert R.RANGE_EDGES.tolist() == m1.RANGE_EDGES.tolist() and R.EGO_M == m1.EGO_M
    assert (R.GROUND_DZ, R.ABOVE_HI, R.PLANE_SEED) == (m1.GROUND_DZ, m1.ABOVE_HI, m1.SEED)


@pytest.mark.parametrize('seed,tilt', [(0, 0.02), (1, -0.03), (2, 0.0)])
def test_ground_plane_is_m1s_bit_for_bit(seed, tilt):
    xyz, _, _ = frame(seed, tilt=tilt)
    d = dict(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])
    dz_m1, info = m1.ground_height(d, 'plane')
    dz, plane = R.ground_dz(xyz)
    assert plane == info['plane']
    np.testing.assert_array_equal(dz, dz_m1)


def test_ground_falls_back_to_m1s_mode_without_a_plane():
    xyz = np.c_[np.linspace(5, 20, 20), np.zeros(20), np.full(20, H)]      # too few cells for a plane
    d = dict(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])
    dz_m1, info = m1.ground_height(d, 'plane')
    dz, plane = R.ground_dz(xyz)
    assert plane is None and info['plane'] is None
    np.testing.assert_array_equal(dz, dz_m1)


def test_classify_cells_and_untouched():
    xyz, _, tag = frame(3)
    cell, plane = R.classify(xyz)
    assert plane is not None and abs(plane[0] - 0.02) < 0.005
    assert np.all(cell[tag] == -1)
    assert np.all(cell[~tag] >= 0) and cell.max() < R.N_BINS * 2
    r = np.linalg.norm(xyz, axis=1)
    np.testing.assert_array_equal(cell[~tag] // 2, m1.range_bin(r[~tag]))


# ------------------------------------------------------------------------------------------- the arms
def test_threshold_drops_below_keeps_above_ties_with_q():
    n = 200000
    cell = np.full(n, 3)                                   # 10-20 m, above-ground
    inten = np.repeat([10.0, 50.0, 90.0], [n // 4, n // 2, n - n // 4 - n // 2])
    c = cfg(e=0.5, t=50.0, q=0.5)
    d = R.drop_mask(cell, inten, c, np.random.default_rng(0))
    assert d[inten == 10].all() and not d[inten == 90].any()
    share = d[inten == 50].mean()
    assert abs(share - 0.5) < 0.01
    assert abs(d.mean() - 0.5) < 0.01                      # pooled share = E


def test_ties_at_a_constant_intensity_drop_the_pooled_share_at_random():
    """KITTI beyond 40 m: median intensity 0, so T = 0 and Q = E / share(0): a random drop among the zeros."""
    n = 100000
    rng = np.random.default_rng(1)
    inten = np.where(rng.random(n) < 0.6, 0.0, rng.uniform(0.01, 0.5, n))
    e = 0.12
    q = e / (inten == 0).mean()
    d = R.drop_mask(np.full(n, 8), inten, cfg(e=e, t=0.0, q=q), np.random.default_rng(2))
    assert abs(d.mean() - e) < 0.005 and not d[inten > 0].any()


def test_random_mode_ignores_intensity():
    n = 200000
    inten = np.random.default_rng(3).uniform(0, 255, n)
    d = R.drop_mask(np.full(n, 5), inten, cfg('random', e=0.2), np.random.default_rng(4))
    assert abs(d.mean() - 0.2) < 0.005
    assert abs(d[inten < 128].mean() - d[inten >= 128].mean()) < 0.01


@pytest.mark.parametrize('mode', ['threshold', 'random'])
def test_zero_or_null_e_and_unclassified_never_drop(mode):
    xyz, inten, tag = frame(4)
    c = cfg(mode, e=0.9, t=255.0, q=1.0)
    keep, cell = R.keep_mask(xyz, inten, c, 'f0')
    assert keep[tag].all() and (~keep[~tag]).mean() > 0.8
    e = [0.0, None, 0.0, None, 0.0, None, 0.0]
    c = EasyDict({'MODE': mode, 'E': {'ground': e, 'above': e}})
    if mode == 'threshold':
        c.update(T=table(255.0, 255.0), Q=table(1.0, 1.0))
    assert R.keep_mask(xyz, inten, c, 'f0')[0].all()


def test_per_cell_values_reach_their_cells():
    xyz, inten, tag = frame(5)
    e_g = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]               # ground 10-20 m only
    c = EasyDict({'MODE': 'random', 'E': {'ground': e_g, 'above': [0.0] * 7}})
    keep, cell = R.keep_mask(xyz, inten, c, 'f1')
    assert keep[cell != 2].all()
    assert abs((~keep[cell == 2]).mean() - 0.3) < 0.03


def test_draws_fixed_per_frame_and_distinct_per_mode_and_frame():
    xyz, inten, _ = frame(6)
    c = cfg('random', e=0.3)
    k1, _ = R.keep_mask(xyz, inten, c, '000123')
    k2, _ = R.keep_mask(xyz, inten, c, '000123')
    k3, _ = R.keep_mask(xyz, inten, c, '000124')
    np.testing.assert_array_equal(k1, k2)
    assert not np.array_equal(k1, k3)
    u_thr = R.frame_rng('000123', 'threshold').random(10)
    u_rnd = R.frame_rng('000123', 'random').random(10)
    assert not np.allclose(u_thr, u_rnd)


@pytest.mark.parametrize('bad', [
    dict(MODE='darkest'),
    dict(E=table(1.0, 0.1)),
    dict(E=table(-0.1, 0.1)),
    dict(T=None),
    dict(Q=table(1.5, 0.5)),
    dict(EXTRA=1),
    dict(E={'ground': [0.1] * 6, 'above': [0.1] * 7}),
])
def test_malformed_configs_fail(bad):
    c = cfg()
    for k, v in bad.items():
        if v is None:
            c.pop(k)
        else:
            c[k] = v
    with pytest.raises((AssertionError, KeyError)):
        R.check_cfg(c)


# ------------------------------------------------------------------------------------------- KITTI hook
def test_kitti_point_arm():
    xyz, inten, tag = frame(7)
    pts = np.c_[xyz, inten / 255.0].astype(np.float32)
    swap = EasyDict({'REFLECTION_DROP': cfg('random', e=0.0)})
    assert kfs.arm_of(swap) == 'REFLECTION_DROP'
    np.testing.assert_array_equal(kfs.apply_points(pts, swap, '000001'), pts)
    swap = EasyDict({'REFLECTION_DROP': cfg('threshold', e=0.2, t=0.3, q=0.0)})
    out = kfs.apply_points(pts, swap, '000001')
    kept = {tuple(p) for p in out.tolist()}
    dropped = [p for p, t in zip(pts.tolist(), tag) if tuple(p) not in kept]
    assert len(dropped) > 0 and all(p[3] < 0.3 for p in dropped)
    assert len(out) + len(dropped) == len(pts)


def test_lyft_refuses_the_arm_pandaset_takes_it():
    swap = EasyDict({'REFLECTION_DROP': cfg()})
    assert lfs.arm_of(swap, 'pandaset') == 'REFLECTION_DROP'
    with pytest.raises(AssertionError):
        lfs.arm_of(swap, 'lyft')


# ------------------------------------------------------------------------------------------- Waymo hook
def _waymo_dataset(tmp, lidar_indices=None):
    from pcdet.datasets.waymo.waymo_dataset import WaymoDataset
    ds = WaymoDataset.__new__(WaymoDataset)
    ds.data_path = Path(tmp)
    ds.dataset_cfg = EasyDict({'LIDAR_INDICES': lidar_indices} if lidar_indices is not None else {})
    return ds


@pytest.mark.parametrize('lidar_indices', [None, [0], [0, 2], [1, 2]])
def test_waymo_get_lidar_unchanged_and_top_mask(lidar_indices):
    rng = np.random.default_rng(8)
    counts = [500, 100, 80, 60, 40]
    raw = np.c_[rng.normal(0, 10, (sum(counts), 3)), rng.uniform(0, 2, sum(counts)), rng.uniform(0, 1, sum(counts)),
                np.where(rng.random(sum(counts)) < 0.9, -1.0, 1.0)].astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'seq'))
        np.save(os.path.join(tmp, 'seq', '0003.npy'), raw)
        ds = _waymo_dataset(tmp, lidar_indices)
        plain = ds.get_lidar('seq', 3, counts)
        pts, is_top = ds.get_lidar('seq', 3, counts, return_top_mask=True)
    np.testing.assert_array_equal(pts, plain)
    lid = np.repeat(np.arange(5), counts)
    keep = np.isin(lid, lidar_indices if lidar_indices is not None else range(5)) & (raw[:, 5] == -1)
    np.testing.assert_array_equal(is_top, lid[keep] == 0)


def test_waymo_arm_touches_top_only_and_recounts_emptied_boxes():
    xyz, inten, _ = frame(9)
    origin = np.array([1.43, 0.0, 2.184])
    top = np.c_[xyz + origin, np.tanh(inten / 100.0), np.zeros(len(xyz))].astype(np.float32)
    side = np.c_[np.random.default_rng(10).uniform(-5, 5, (300, 3)), np.ones((300, 2))].astype(np.float32)
    points = np.concatenate([top, side])
    is_top = np.r_[np.ones(len(top), bool), np.zeros(len(side), bool)]
    ext = np.eye(4); ext[:3, 3] = origin
    swap = EasyDict({'REFLECTION_DROP': cfg('random', e=0.5, TOP_CALIB='x.pkl')})
    assert wfs.arm_of(swap) == 'REFLECTION_DROP'
    keep = wfs.reflection_keep(points, is_top, ext, swap, 'seg_0001')
    assert keep[~is_top].all() and (~keep[is_top]).sum() > 1000
    # the same mask as the shared module on TOP points about the TOP position
    k_top, _ = R.keep_mask(np.asarray(top[:, :3], np.float64) - origin, top[:, 3], swap.REFLECTION_DROP, 'seg_0001')
    np.testing.assert_array_equal(keep[is_top], k_top)
    # finish: applies the mask, drops exactly the boxes the arm emptied, keeps the stored count otherwise
    p_drop = points[~keep][0, :3]
    p_kept = points[keep & is_top][0, :3]
    boxes = np.array([[*p_drop, 0.01, 0.01, 0.01, 0.0],       # holds one dropped point only -> emptied
                      [*p_kept, 0.01, 0.01, 0.01, 0.0],       # holds a kept point
                      [500.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float32)   # empty already (stored 3)
    d = {'points': points.copy(), 'gt_boxes': boxes, 'gt_names': np.array(['Vehicle'] * 3),
         'num_points_in_gt': np.array([7, 9, 3]), 'reflection_keep': keep}
    out = wfs.finish(d, swap, ['Car', 'Pedestrian', 'Cyclist'], None, 'seg_0001')
    assert 'reflection_keep' not in out
    np.testing.assert_array_equal(out['points'], points[keep])
    np.testing.assert_array_equal(out['num_points_in_gt'], [0, 9, 3])


def test_waymo_check_needs_top_calib():
    swap = EasyDict({'REFLECTION_DROP': cfg('random', e=0.1)})
    dcfg = EasyDict({'TRAIN_FACTOR_SWAP': swap})
    with pytest.raises(AssertionError):
        wfs.check_dataset_cfg(dcfg, ['Car', 'Pedestrian', 'Cyclist'])
    swap.REFLECTION_DROP.TOP_CALIB = 'x.pkl'
    wfs.check_dataset_cfg(dcfg, ['Car', 'Pedestrian', 'Cyclist'])
    dcfg.LIDAR_INDICES = [1, 2]
    with pytest.raises(AssertionError):
        wfs.check_dataset_cfg(dcfg, ['Car', 'Pedestrian', 'Cyclist'])


def test_waymo_emptied_counts_equal_the_full_recount():
    """finish() counts only the boxes holding a dropped point; recount_emptied must come out as with two full counts."""
    rng = np.random.default_rng(11)
    pts = np.c_[rng.uniform(-30, 30, (20000, 2)), rng.uniform(-1, 2, 20000), rng.uniform(0, 1, (20000, 2))]
    pts = pts.astype(np.float32)
    boxes = np.c_[rng.uniform(-25, 25, (60, 2)), rng.uniform(0, 1, 60), rng.uniform(0.2, 4, (60, 3)),
                  rng.uniform(-3, 3, 60)].astype(np.float32)
    boxes[:3, 3:6] = 0.0                                       # degenerate (count 0 in points_per_box)
    boxes[3] = [100, 100, 0, 1, 1, 1, 0]                        # empty
    for p in (0.02, 0.5, 0.98, 1.0):
        keep = rng.random(len(pts)) >= p
        if p == 1.0:
            keep[:] = False
        stored = rng.integers(0, 50, len(boxes))
        full = wfs.recount_emptied(stored, wfs.points_per_box(pts, boxes), wfs.points_per_box(pts[keep], boxes))
        fast = wfs.recount_emptied(stored, *wfs.emptied_counts(pts, keep, boxes))
        np.testing.assert_array_equal(fast, full)
    np.testing.assert_array_equal(wfs.recount_emptied(stored, *wfs.emptied_counts(pts, np.ones(len(pts), bool), boxes)),
                                  stored)


def test_dump_box_match_float64_raw_against_float32_collated():
    """The N->W dump smoke (2026-11-07) compared Waymo's float64 boxes with float32 copies exactly and matched none of
    the boxes the arm emptied; either side may be the float64 one."""
    sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'analysis'))
    from reflection_r11_dump_utils import gone_mask
    raw = np.array([[10.123456789, -3.3, 0.71, 4.52, 1.91, 1.63, 0.1],
                    [20.0, 5.0, 0.8, 4.0, 2.0, 1.5, 0.0]], dtype=np.float64)
    assert not np.array_equal(raw[0, :6], raw[0, :6].astype(np.float32))       # what the exact comparison tripped on
    final32 = np.c_[raw.astype(np.float32), np.ones(2, np.float32)]
    final64 = np.c_[raw, np.ones(2)]
    np.testing.assert_array_equal(gone_mask(final32, raw[:1]), [True, False])
    np.testing.assert_array_equal(gone_mask(final64, raw[:1].astype(np.float32)), [True, False])
    assert not gone_mask(final32, np.zeros((0, 7))).any() and len(gone_mask(np.zeros((0, 8)), raw)) == 0
