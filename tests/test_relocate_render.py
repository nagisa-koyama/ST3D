"""pcdet/datasets/augmentor/relocate_render.py (CPU; experiments_md 20261018_02, bug-prevention checklist).

Pins: the TRAIN-segment Waymo TOP lattice; the object-only render against the analysis implementation
(tools/analysis/source_coverage.py `render`, 20261018_01) on identical inputs; the in-box test against roiaware's
points_in_boxes_cpu; the relocation geometry (exact r', azimuth, bottom height, heading relative to the line of sight,
the visible face, velocity); snapped points on their pixel rays, one point per pixel; the free-space rules (GT of any
label, earlier copies, range, scene points above the clearance); the label sentinels (negative pseudo-label classes,
-99 IGNORE_CLASS_LABEL, other classes, masked and ignore-mask boxes: never donors, never moved, always obstacles); an
empty donor, a one-point donor, a donor whose copy loses every point to occlusion, a frame with no donors; shadow and
occlusion on a two-object toy; feature columns (time lag 0) and x, y, z-only clouds; the DataAugmentor hook (absent
by default, picklable).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

from pcdet.datasets.augmentor import relocate_render as R  # noqa: E402
from pcdet.datasets import rerender_utils as RU  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

TH = np.radians(np.concatenate([np.linspace(-17.6, -3.0, 40, endpoint=False), np.linspace(-3.0, 2.4, 24)]))
NC, H = 2650, 2.184
LAT = dict(thetas=TH, n_cols=NC, height=H)
P = dict(R.DEFAULTS)


class FixedRng:
    """permutation = identity; uniform returns queued values (asserting they lie in [lo, hi])."""

    def __init__(self, values):
        self.v = list(values)

    def permutation(self, n):
        return np.arange(n)

    def uniform(self, lo, hi):
        x = self.v.pop(0)
        assert lo - 1e-9 <= x <= hi + 1e-9, (lo, x, hi)
        return x


def car_surface(box, n_side=24, n_up=12, jitter=0.0, seed=0):
    """Points on the 4 vertical faces and the roof of a box (dense, like a near accumulated car)."""
    rng = np.random.default_rng(seed)
    x, y, z, dx, dy, dz, h = box[:7]
    pts = []
    for u in np.linspace(-0.5, 0.5, n_side):
        for v in np.linspace(-0.5, 0.5, n_up):
            for (a, b) in ((u * dx, -dy / 2), (u * dx, dy / 2), (-dx / 2, u * dy), (dx / 2, u * dy)):
                pts.append([a * 0.999, b * 0.999, v * dz * 0.999])
    for u in np.linspace(-0.5, 0.5, n_side):
        for w in np.linspace(-0.5, 0.5, 8):
            pts.append([u * dx * 0.999, w * dy * 0.999, dz / 2 * 0.999])
    p = np.array(pts) + (rng.normal(0, jitter, (len(pts), 3)) if jitter else 0)
    c, s = np.cos(h), np.sin(h)
    out = np.stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1], z + p[:, 2]], 1)
    return out


def frame(boxes, names, pts, extra_cols=2, **kw):
    pts = np.asarray(pts, np.float64)
    cols = [pts]
    if extra_cols:
        cols.append(np.full((len(pts), 1), 0.5))                    # intensity
        if extra_cols > 1:
            cols.append(np.full((len(pts), 1), 0.35))               # time lag (sweeps)
    d = dict(points=np.concatenate(cols, 1).astype(np.float32), gt_boxes=np.asarray(boxes, np.float32),
             gt_names=np.asarray(names), gt_boxes_mask=np.ones(len(boxes), bool))
    d.update(kw)
    return d


DONOR = np.array([10.0, 0.0, 0.75, 4.0, 2.0, 1.5, 0.3])


# ---------------------------------------------------------------------------------------------------- lattice
def test_train_lattice_is_the_train_median():
    lat = R.waymo_top_train_lattice()
    assert lat['n_segments'] == 798 and len(lat['thetas']) == 64 and lat['n_cols'] == 2650
    assert np.all(np.diff(lat['thetas']) > 0) and lat['height'] == 2.184 and lat['extrinsic_z'] == (2.184,)
    d = pickle.load(open(R.WAYMO_TOP_CALIB, 'rb'))
    tr = [ln.strip().replace('.tfrecord', '') for ln in open(R.WAYMO_TRAIN_LIST) if ln.strip()]
    va = set(d) - set(tr)
    assert len(va) == 202
    ref = np.median([np.sort(np.asarray(d[k]['inclinations'], np.float64)) for k in tr], 0)
    np.testing.assert_array_equal(lat['thetas'], ref)
    allm = RU.lattice_spec('waymo_top')['thetas']
    assert np.degrees(np.abs(lat['thetas'] - allm)).max() < 0.03          # 0.025 deg measured, 20261018_02 §1.2
    assert R.lattice_from_config(EasyDict(R.DEFAULTS))['n_segments'] == 798


# ----------------------------------------------------------------------------------- render vs the analysis
def test_render_equals_source_coverage_render():
    import source_coverage as S
    rng = np.random.default_rng(3)
    for r_new in (8.0, 40.0, 63.3):
        b, q = R.relocate(DONOR, car_surface(DONOR, jitter=0.01), r_new, rng.uniform(0, 2 * np.pi))
        a = R.render_object(q, TH, NC, H)
        s = S.render(q, TH, NC, H)
        assert len(a[0]) > 0
        for x, y in zip(a, s):
            np.testing.assert_array_equal(x, y)
    for x, y in zip(R.render_object(np.zeros((0, 3)), TH, NC, H), S.render(np.zeros((0, 3)), TH, NC, H)):
        assert x.shape == y.shape


def test_render_snaps_onto_pixel_rays_one_point_per_pixel_nearest_kept():
    b, q = R.relocate(DONOR, car_surface(DONOR, jitter=0.02), 45.0, 1.0)
    xyz, rows, cols, kept = R.render_object(q, TH, NC, H)
    pix = rows * NC + cols
    assert len(np.unique(pix)) == len(pix) > 10
    d = xyz - [0, 0, H]
    rng = np.linalg.norm(d, axis=1)
    np.testing.assert_allclose(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])), TH[rows], atol=1e-9)
    np.testing.assert_allclose(np.arctan2(d[:, 1], d[:, 0]), (cols + 0.5) / NC * 2 * np.pi - np.pi, atol=1e-9)
    qr = np.linalg.norm(q - [0, 0, H], axis=1)
    np.testing.assert_allclose(rng, qr[kept], atol=1e-9)
    row, col, rr, val = RU.to_lattice(q, H, TH, NC)
    for k, p in zip(kept, pix):                                  # the kept point is the nearest of its pixel
        same = val & (row * NC + col == p)
        assert rr[k] == rr[same].min()


# --------------------------------------------------------------------------------------------- in-box test
def test_points_in_box_matches_roiaware():
    import torch
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    rng = np.random.default_rng(1)
    boxes = np.array([[10, 3, 0.8, 4.2, 1.9, 1.6, 0.7], [-20, 15, 1.0, 5.0, 2.2, 2.0, -2.5], [40, -40, 0.9, 1, 1, 1, 3.0]])
    for b in boxes:
        p = b[:3] + rng.uniform(-3.5, 3.5, (20000, 3))
        ref = roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(p.astype(np.float32)),
                                                        torch.from_numpy(b[None].astype(np.float32))).numpy()[0] > 0
        np.testing.assert_array_equal(R.points_in_box(p, b), ref)


# ------------------------------------------------------------------------------------------ relocation geometry
@pytest.mark.parametrize('r_new,phi_new', [(10.0, 0.0), (33.3, 2.0), (75.0, -2.9), (50.0, 5.5)])
def test_relocate_geometry(r_new, phi_new):
    box = np.array([10.0, 4.0, 0.62, 4.5, 1.9, 1.5, 0.4, 3.0, -1.0])
    pts = car_surface(box)
    nb, q = R.relocate(box, pts, r_new, phi_new)
    assert np.hypot(nb[0], nb[1]) == pytest.approx(r_new, abs=1e-12)
    assert np.arctan2(nb[1], nb[0]) == pytest.approx(np.arctan2(np.sin(phi_new), np.cos(phi_new)), abs=1e-12)
    assert nb[2] == box[2] and np.all(nb[3:6] == box[3:6])                         # bottom height and size kept
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    rel0 = wrap(box[6] - np.arctan2(box[1], box[0])); rel1 = wrap(nb[6] - np.arctan2(nb[1], nb[0]))
    assert rel1 == pytest.approx(rel0, abs=1e-12)                                  # heading vs line of sight
    np.testing.assert_allclose(q[:, 2], pts[:, 2], atol=0)                         # z untouched
    loc = lambda b, p: np.stack([np.cos(-b[6]) * (p[:, 0] - b[0]) - np.sin(-b[6]) * (p[:, 1] - b[1]),
                                 np.sin(-b[6]) * (p[:, 0] - b[0]) + np.cos(-b[6]) * (p[:, 1] - b[1])], 1)
    np.testing.assert_allclose(loc(nb, q), loc(box, pts), atol=1e-9)               # rigid, in the box frame
    ray = lambda b: wrap(np.arctan2(b[1], b[0]) - b[6])                            # line of sight in the box frame
    assert ray(nb) == pytest.approx(ray(box), abs=1e-12)                           # so the same face is seen
    assert np.hypot(nb[7], nb[8]) == pytest.approx(np.hypot(3.0, -1.0))
    assert wrap(np.arctan2(nb[8], nb[7]) - np.arctan2(-1.0, 3.0)) == pytest.approx(wrap(nb[6] - box[6]), abs=1e-12)
    assert R.points_in_box(q, nb).all() == R.points_in_box(pts, box).all()


def test_relocate_refuses_r0():
    with pytest.raises(ValueError):
        R.relocate(np.array([0, 0, 0.7, 4, 2, 1.5, 0.0]), np.zeros((1, 3)), 20.0, 0.0)


def test_render_object_at_carries_features_and_zeroes_time():
    pts = np.concatenate([car_surface(DONOR), np.full((len(car_surface(DONOR)), 1), 0.7),
                          np.full((len(car_surface(DONOR)), 1), 0.4)], 1).astype(np.float32)
    out = R.render_object_at(DONOR, pts, 50.0, 1.0, LAT, time_column=4)
    assert out['points'].shape[1] == 5 and len(out['points']) > 0
    assert np.all(out['points'][:, 3] == np.float32(0.7)) and np.all(out['points'][:, 4] == 0)
    out3 = R.render_object_at(DONOR, pts[:, :3], 50.0, 1.0, LAT, time_column=4)  # x, y, z only (features None)
    np.testing.assert_allclose(out3['points'], out['points'][:, :3])


# --------------------------------------------------------------------------------------------- scene level
def base_scene(extra_boxes=(), extra_names=(), ground=True, **kw):
    boxes = [DONOR] + list(extra_boxes)
    names = ['Car'] + list(extra_names)
    pts = [car_surface(DONOR)]
    if ground:
        g = np.stack(np.meshgrid(np.arange(-74, 74, 1.0), np.arange(-74, 74, 1.0)), -1).reshape(-1, 2)
        pts.append(np.concatenate([g, np.zeros((len(g), 1))], 1))
    return frame(boxes, names, np.concatenate(pts), **kw)


def run(d, values, **over):
    p = dict(P); p.update(over)
    return R.relocate_render_scene(d, p, LAT, rng=FixedRng(values), keep_points=True)


def test_copy_is_added_with_labels_and_originals_kept():
    d = base_scene(gt_classes=np.array([1], np.int32), gt_scores=np.array([0.9], np.float32))
    b0, n0 = d['gt_boxes'].copy(), d['gt_names'].copy()
    out, rec = run(d, [40.0, 2.0])
    assert rec['n_eligible'] == 1 and rec['n_placed'] == 1
    assert len(out['gt_boxes']) == 2 and list(out['gt_names']) == ['Car', 'Car']
    np.testing.assert_array_equal(out['gt_boxes'][0], b0[0])
    assert out['gt_boxes_mask'].tolist() == [True, True] and out['gt_classes'].tolist() == [1, 1]
    assert out['gt_scores'][1] == np.float32(0.9)
    c = rec['copies'][0]
    assert c['r_new'] == pytest.approx(40.0, abs=1e-9) and out['gt_boxes'][1, 2] == np.float32(DONOR[2])
    p = out['points'][c['point_index']]
    assert len(p) == c['n_final'] > 0 and np.all(p[:, 4] == 0) and np.all(p[:, 3] == np.float32(0.5))
    row, col, rr, val = RU.to_lattice(p[:, :3], H, TH, NC)
    assert val.all() and len(np.unique(row * NC + col)) == len(p)                # one copy point per pixel
    assert n0.tolist() == ['Car']


def test_no_donor_frames_are_unchanged():
    for d in (base_scene(extra_boxes=[], ground=True),):
        d['gt_names'] = np.array(['Truck'])                                         # not the donor class
        before = {k: np.copy(v) for k, v in d.items()}
        out, rec = run(d, [])
        assert rec['n_candidates'] == 0 and rec['n_placed'] == 0
        for k in before:
            np.testing.assert_array_equal(out[k], before[k])
    d = frame(np.zeros((0, 7)), np.zeros(0, str), car_surface(DONOR))
    out, rec = run(d, [])
    assert rec['n_placed'] == 0 and len(out['gt_boxes']) == 0
    d = frame([DONOR], ['Car'], np.zeros((0, 3)))
    out, rec = run(d, [])
    assert rec['n_placed'] == 0


def test_empty_and_one_point_donors_are_not_donors():
    empty = np.array([-30.0, 10.0, 0.75, 4.0, 2.0, 1.5, 0.0])
    one = np.array([20.0, -20.0, 0.75, 4.0, 2.0, 1.5, 0.0])
    pts = np.concatenate([car_surface(DONOR), one[None, :3]])
    d = frame([empty, one], ['Car', 'Car'], pts)
    out, rec = run(d, [])
    assert rec['n_candidates'] == 2 and rec['n_eligible'] == 0 and rec['n_placed'] == 0
    out, rec = run(frame([one], ['Car'], one[None, :3]), [40.0, 0.5], MIN_DONOR_POINTS=1)   # the one-point path
    assert rec['n_placed'] == 1 and rec['copies'][0]['n_final'] == 1


def test_donor_beyond_rmax_is_skipped():
    far = np.array([75.5, 0.0, 0.75, 4.0, 2.0, 1.5, 0.0])
    d = frame([far], ['Car'], car_surface(far))
    out, rec = run(d, [])
    assert rec['n_candidates'] == 0


@pytest.mark.parametrize('kind', ['neg_class', 'ignore99', 'masked', 'ignore_mask', 'other_class'])
def test_sentinels_never_donors_never_moved_always_obstacles(kind):
    sb = np.array([0.0, 40.0, 0.75, 4.0, 2.0, 1.5, 0.0])                             # where the copy is forced to go
    pts = np.concatenate([car_surface(DONOR), car_surface(sb)])
    kw = {}
    names = ['Car', 'Car']
    if kind == 'neg_class':
        kw = dict(gt_classes=np.array([1, -1], np.int32), gt_scores=np.array([1, 0.2], np.float32))
    elif kind == 'ignore99':
        kw = dict(gt_classes=np.array([1, common_utils.IGNORE_CLASS_LABEL], np.int32),
                  gt_scores=np.ones(2, np.float32))
    elif kind == 'ignore_mask':
        kw = dict(gt_boxes_ignore_mask=np.array([False, True]))
    elif kind == 'other_class':
        names = ['Car', 'Truck']
    d = frame([DONOR, sb], names, pts, **kw)
    if kind == 'masked':
        d['gt_boxes_mask'] = np.array([True, False])
    if kind == 'ignore_mask':
        d['gt_boxes_mask'] = np.array([True, False])
    sb_before = d['gt_boxes'][1].copy()
    out, rec = run(d, [40.0, np.pi / 2] * 20)                                      # every try onto the sentinel
    assert rec['n_eligible'] == 1                                                   # the sentinel is no donor
    assert rec['n_placed'] == 0 and rec['reject']['overlap'] == 20 and rec['n_no_place'] == 1
    np.testing.assert_array_equal(out['gt_boxes'][1], sb_before)
    out, rec = run(d, [40.0, 3 * np.pi / 2])                                        # free elsewhere
    assert rec['n_placed'] == 1 and rec['copies'][0]['donor'] == 0
    np.testing.assert_array_equal(out['gt_boxes'][1], sb_before)
    if kind in ('neg_class', 'ignore99'):
        assert out['gt_classes'].tolist() == [1, int(d['gt_classes'][1]), 1]
    if kind == 'ignore_mask':
        assert out['gt_boxes_ignore_mask'].tolist() == [False, True, False]
    assert not R.points_in_box(out['points'][:, :3], sb_before).sum() < R.points_in_box(pts, sb_before).sum()


def test_free_space_rules():
    d = base_scene()
    _, rec = run(d, [74.0, 0.0] + [40.0, 2.0])                                      # grown box leaves the range
    assert rec['reject']['range'] == 1 and rec['n_placed'] == 1
    pole = lambda n: np.stack([np.zeros(n), np.full(n, 40.0), np.linspace(0.5, 3.0, n)], 1)
    for n, placed in ((5, 1), (6, 0)):                                              # > 5 points above 0.3 m
        dd = frame([DONOR], ['Car'], np.concatenate([car_surface(DONOR), pole(n) + [1.2, 0, 0]]))
        _, rec = run(dd, [40.0, np.pi / 2], MAX_TRIES=1)
        assert rec['n_placed'] == placed and rec['reject']['scene'] == 1 - placed
    low = np.stack([np.zeros(50), np.full(50, 40.0), np.full(50, 0.29)], 1)        # below the clearance: free
    _, rec = run(frame([DONOR], ['Car'], np.concatenate([car_surface(DONOR), low])), [40.0, np.pi / 2], MAX_TRIES=1)
    assert rec['n_placed'] == 1


def test_copies_never_collide_and_stay_in_range():
    rng = np.random.default_rng(5)
    boxes, names, pts = [], [], []
    for k in range(14):
        a = 2 * np.pi * k / 14
        b = np.array([12 * np.cos(a), 12 * np.sin(a), 0.75, 4.2, 1.9, 1.5, a + 0.3])
        boxes.append(b); names.append('Car'); pts.append(car_surface(b))
    boxes.append(np.array([30.0, 5.0, 1.0, 10, 2.5, 3, 0.2])); names.append('Truck')
    d = frame(boxes, names, np.concatenate(pts))
    out, rec = R.relocate_render_scene(d, P, LAT, rng=rng)
    assert 0 < rec['n_placed'] <= 8
    gb = out['gt_boxes'].astype(np.float64)
    nb = len(boxes)
    for i in range(nb, len(gb)):
        g = R.bev_corners(gb[i:i + 1], P['BEV_MARGIN'])[0]
        others = np.concatenate([gb[:nb], gb[nb:i]])
        assert not R.rects_overlap(g, R.bev_corners(others)).any()
        assert np.all(np.abs(g) <= 75.2)
        assert np.hypot(gb[i, 0], gb[i, 1]) >= np.hypot(*boxes[rec['copies'][i - nb]['donor']][:2]) - 1e-4


def test_rects_overlap_sat():
    a = R.bev_corners(np.array([[0, 0, 0, 4, 2, 1, 0.0]]))[0]
    bs = R.bev_corners(np.array([[3.9, 0, 0, 4, 2, 1, 0.0], [4.01, 0, 0, 4, 2, 1, 0.0], [3.2, 2.2, 0, 4, 2, 1, 0.8],
                                 [0, 0, 0, 0.5, 0.5, 1, 1.0]]))
    assert R.rects_overlap(a, bs).tolist() == [True, False, True, True]


def two_object_toy():
    """Donor at 10 m (azimuth 0); the copy is forced to (0, 30) (r' 30, azimuth pi / 2). A wall behind it at y = 40
    and a pole in front at y = 20."""
    wall = np.stack(np.meshgrid(np.linspace(-4, 4, 161), np.linspace(0.0, 3.0, 61)), -1).reshape(-1, 2)
    wall = np.stack([wall[:, 0], np.full(len(wall), 40.0), wall[:, 1]], 1)
    pole = np.stack(np.meshgrid(np.linspace(-0.15, 0.15, 7), np.linspace(0.0, 3.0, 61)), -1).reshape(-1, 2)
    pole = np.stack([pole[:, 0], np.full(len(pole), 20.0), pole[:, 1]], 1)
    under = np.stack([np.linspace(-0.5, 0.5, 9), np.full(9, 30.0), np.full(9, 0.05)], 1)   # ground inside the box
    pts = np.concatenate([car_surface(DONOR), wall, pole, under])
    return frame([DONOR], ['Car'], pts), wall, pole, under


def test_shadow_and_occlusion_on_a_two_object_toy():
    d, wall, pole, under = two_object_toy()
    pts0 = d['points'].copy()
    out, rec = run(d, [30.0, np.pi / 2])
    assert rec['n_placed'] == 1
    c = rec['copies'][0]
    o = out['points'][:, :3].astype(np.float64)
    def present(p):
        p32 = p.astype(np.float32)
        s = set(map(tuple, o.astype(np.float32)))
        return np.array([tuple(x) in s for x in p32])
    # (1) scene points inside the copy's box removed
    assert not present(under).any() and c['n_inbox_removed'] == len(under)
    # the pole is nearer than the copy: kept, and copy points in its pixels dropped
    assert present(pole).all() and c['n_occluded'] > 0
    f32 = lambda a: a.astype(np.float32).astype(np.float64)                        # the scene is float32
    prow, pcol, prng, pval = RU.to_lattice(f32(pole), H, TH, NC)
    ppix = set((prow * NC + pcol)[pval])
    cp = o[c['point_index']]
    crow, ccol, crng, cval = RU.to_lattice(cp, H, TH, NC)
    cpix = set(crow * NC + ccol)
    assert not (cpix & ppix)
    # (4) wall points in the copy's pixels (all of them, occluded or not) removed; wall points elsewhere kept
    _, arow, acol, _, _ = R._render(c['moved'], TH, NC, H)
    allpix = set(arow * NC + acol)
    wrow, wcol, wrng, wval = RU.to_lattice(f32(wall), H, TH, NC)
    wpix = np.where(wval, wrow * NC + wcol, -1)
    shadowed = np.array([p in allpix for p in wpix])
    assert shadowed.any() and (~shadowed).any()
    np.testing.assert_array_equal(present(wall), ~shadowed)
    assert c['n_shadow_removed'] == int(shadowed.sum())
    # nothing else of the scene changed
    others = car_surface(DONOR)
    assert present(others).all()
    assert len(o) == len(pts0) - len(under) - shadowed.sum() + c['n_final']


def test_copy_fully_occluded_is_not_added_and_scene_is_untouched():
    screen = np.stack(np.meshgrid(np.linspace(-6, 6, 601), np.linspace(0.0, 6.0, 301)), -1).reshape(-1, 2)
    screen = np.stack([screen[:, 0], np.full(len(screen), 20.0), screen[:, 1]], 1)
    d = frame([DONOR], ['Car'], np.concatenate([car_surface(DONOR), screen]))
    before = {k: np.copy(v) for k, v in d.items()}
    out, rec = run(d, [30.0, np.pi / 2])
    assert rec['n_zero'] == 1 and rec['n_placed'] == 0
    for k in before:
        np.testing.assert_array_equal(out[k], before[k])


def test_max_copies_and_xyz_only_cloud():
    boxes = [np.array([8 * np.cos(a), 8 * np.sin(a), 0.75, 4.0, 1.8, 1.5, a]) for a in np.linspace(0, 2 * np.pi, 12,
                                                                                                 endpoint=False)]
    d = frame(boxes, ['Car'] * 12, np.concatenate([car_surface(b) for b in boxes]), extra_cols=0)
    out, rec = R.relocate_render_scene(d, P, LAT, rng=np.random.default_rng(2))
    assert rec['n_placed'] <= 8 and out['points'].shape[1] == 3
    assert len(out['gt_boxes']) == 12 + rec['n_placed']


def test_data_augmentor_hook_absent_by_default_and_picklable():
    from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
    flip = EasyDict(NAME='random_world_flip', ALONG_AXIS_LIST=['x'])
    plain = DataAugmentor(None, EasyDict(AUG_CONFIG_LIST=[flip], DISABLE_AUG_LIST=[]), ['Car'])
    assert len(plain.data_augmentor_queue) == 1
    rr = EasyDict(dict(R.DEFAULTS, NAME='relocate_render'))
    aug = DataAugmentor(None, EasyDict(AUG_CONFIG_LIST=[rr, flip], DISABLE_AUG_LIST=[]), ['Car'])
    aug2 = pickle.loads(pickle.dumps(aug))
    np.random.seed(0)
    d = base_scene(); n_pts = len(d['points'])
    out = aug2.forward(d)
    assert len(out['gt_boxes']) >= 1 and 'gt_boxes_mask' not in out
    assert len(out['points']) != n_pts or len(out['gt_boxes']) == 1


# ------------------------------------------------------------------------------------------------- the row's config
def test_rr_config_differs_from_27471_only_by_rr_and_the_waymo_target(monkeypatch):
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
    monkeypatch.chdir(ROOT / 'tools')
    load = lambda n: cfg_from_yaml_file('cfgs/da-ieee-access/' + n, EasyDict())
    rrc = load('centerpoint-gblobs-accum-legaldepth-rr-nuscenes2waymo.yaml')
    base = load('centerpoint-gblobs-accum-legaldepth-nuscenes2kitti.yaml')          # 27471
    waymo = load('centerpoint-gblobs-sourceonly-nuscenes2waymo.yaml')                # 27596's target block
    assert rrc.MODEL == base.MODEL and rrc.OPTIMIZATION == base.OPTIMIZATION and rrc.CLASS_NAMES == base.CLASS_NAMES
    assert rrc.DATA_CONFIG_TAR == waymo.DATA_CONFIG_TAR and rrc.DATA_CONFIG_TAR.DATASET == 'WaymoDataset'
    for key in ('NUSCENES_N008', 'NUSCENES_N015'):
        a, b = rrc.DATA_CONFIGS[key], base.DATA_CONFIGS[key]
        la, lb = a.DATA_AUGMENTOR.AUG_CONFIG_LIST, b.DATA_AUGMENTOR.AUG_CONFIG_LIST
        assert la[0].NAME == 'relocate_render' and la[1:] == lb
        strip = lambda c: {k: v for k, v in c.items() if k not in ('DATA_AUGMENTOR', '_BASE_CONFIG_')}
        assert strip(a) == strip(b)
        assert a.DATA_AUGMENTOR.DISABLE_AUG_LIST == b.DATA_AUGMENTOR.DISABLE_AUG_LIST
        assert list(la[0].POINT_CLOUD_RANGE) == list(a.POINT_CLOUD_RANGE)
        assert a.POINT_FEATURE_ENCODING.src_feature_list.index('timestamp') == la[0].TIME_COLUMN
        assert {k: la[0][k] for k in ('MIN_DONOR_POINTS', 'MAX_COPIES', 'MAX_TRIES', 'R_MAX', 'BEV_MARGIN',
                                      'MAX_SCENE_POINTS', 'CLEARANCE', 'OCCLUSION_M', 'N_COLS', 'MOUNT_HEIGHT')} == \
            dict(MIN_DONOR_POINTS=200, MAX_COPIES=8, MAX_TRIES=20, R_MAX=75.2, BEV_MARGIN=0.5, MAX_SCENE_POINTS=5,
                 CLEARANCE=0.3, OCCLUSION_M=0.5, N_COLS=2650, MOUNT_HEIGHT=2.184)       # T05 §5, fixed
        aug = DataAugmentor(None, a.DATA_AUGMENTOR, rrc.CLASS_NAMES)
        names = [getattr(f, 'func', f).__name__ for f in aug.data_augmentor_queue]
        assert names == ['relocate_render', 'random_object_scaling', 'random_world_flip', 'random_world_rotation',
                         'random_world_scaling']
        assert aug.data_augmentor_queue[0].keywords['lattice']['n_segments'] == 798
    assert 'relocate_render' not in str(rrc.DATA_CONFIG_TAR)


# ================================================================================================= RR v2 (§6)
def test_v2_box_frame_round_trip():
    rng = np.random.default_rng(7)
    for _ in range(5):
        b = np.r_[rng.uniform(-60, 60, 2), rng.uniform(-1, 2), rng.uniform(1, 6, 3), rng.uniform(-7, 7)]
        p = rng.normal(0, 3, (500, 3)) + b[:3]
        np.testing.assert_allclose(R.from_box_frame(R.to_box_frame(p, b), b), p, atol=1e-9)
        q = rng.normal(0, 2, (300, 3))
        np.testing.assert_allclose(R.to_box_frame(R.from_box_frame(q, b), b), q, atol=1e-9)
    b = np.array([10.0, 0, 0.8, 4, 2, 1.5, np.pi / 2])                  # +x of the box is +y of the frame
    np.testing.assert_allclose(R.to_box_frame(np.array([[10.0, 1.0, 0.8]]), b), [[1.0, 0, 0]], atol=1e-12)


def test_v2_aggregation_of_a_moving_instance_across_two_keyframes():
    local = R.to_box_frame(car_surface(np.array([0, 0, 0, 4.2, 1.8, 1.5, 0.0]), n_side=30, n_up=10), np.zeros(7))
    ka = np.array([12.0, 3.0, 0.75, 4.2, 1.8, 1.5, 0.3])                # keyframe A
    kb = np.array([31.0, -6.0, 0.95, 4.2, 1.8, 1.5, 1.4])               # keyframe B: moved 20 m, turned 1.1 rad, 0.2 m up
    wa, wb = R.from_box_frame(local[::2], ka), R.from_box_frame(local[1::2], kb)   # each keyframe sees half the surface
    acc = R.VoxelAccumulator(0.02, 4)
    for w, b in ((wa, ka), (wb, kb)):
        assert R.points_in_box(w, b).all()
        acc.add(np.c_[R.to_box_frame(w, b), np.full(len(w), 0.5)])
    ref = R.VoxelAccumulator(0.02, 4); ref.add(np.c_[local, np.full(len(local), 0.5)])
    np.testing.assert_allclose(np.sort(acc.points(), 0), np.sort(ref.points(), 0), atol=1e-9)
    assert acc.n_raw == len(local) and len(acc.points()) == len(ref.points())
    acc.add(np.c_[R.to_box_frame(wa, ka), np.full(len(wa), 0.5)])     # the same points again: no new voxel
    assert len(acc.points()) == len(ref.points())
    one = R.VoxelAccumulator(0.02, 4); one.add(np.array([[0.001, 0.001, 0.001, 1.0], [0.011, 0.015, 0.019, 3.0]]))
    np.testing.assert_allclose(one.points(), [[0.006, 0.008, 0.010, 2.0]])          # centroid of one voxel


def test_v2_lowest_per_cell_and_plane_equal_ground_tilt():
    import ground_tilt as GT
    rng = np.random.default_rng(11)
    g = rng.uniform(-45, 45, (60000, 2))
    z = 0.04 * g[:, 0] - 0.02 * g[:, 1] + rng.normal(0, 0.03, len(g))
    z[::7] += rng.uniform(0.5, 3.0, len(z[::7]))                        # clutter above the ground
    z[::97] = z[1::97][:len(z[::97])]                                   # ties
    pts = np.c_[g, z]
    np.testing.assert_array_equal(R.lowest_per_cell(pts), GT.lowest_per_cell(pts))
    ours = R.fit_plane(R.lowest_per_cell(pts), np.random.default_rng(0))
    ref = GT.fit_plane(GT.lowest_per_cell(pts), rng=np.random.default_rng(0))[0]
    np.testing.assert_allclose(ours, ref)


def test_v2_ground_plane_on_a_tilted_toy():
    rng = np.random.default_rng(12)
    g = rng.uniform(-50, 50, (80000, 2))
    slope = np.tan(np.radians(2.3))
    z = slope * g[:, 1] - 0.05 + rng.normal(0, 0.02, len(g))
    wall = np.c_[rng.uniform(-30, 30, 20000), np.full(20000, 15.0), rng.uniform(0, 6, 20000)]
    wall[:, 2] += slope * 15.0
    pl = R.ground_plane(np.concatenate([np.c_[g, z], wall]))
    assert abs(np.degrees(np.arctan(pl[1])) - 2.3) < 0.05 and abs(pl[0]) < 1e-3 and abs(pl[2] + 0.05) < 0.02
    assert R.ground_plane(np.c_[rng.uniform(-2, 2, (100, 2)), np.zeros(100)]) is None   # too few cells
