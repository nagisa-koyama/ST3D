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


class RngV2:
    """uniform(lo, hi) from a queue (logged); uniform(lo, hi, size) -> `arr` (default zeros); randint -> `ints`
    (default 0). With the defaults the sparsifier keeps everything (p = 1 - 0 = 1, u = 0 < 1)."""

    def __init__(self, values, ints=(), arr=0.0):
        self.v, self.ints, self.arr, self.calls = list(values), list(ints), arr, []

    def uniform(self, lo, hi, size=None):
        if size is not None:
            return np.full(size, self.arr)
        self.calls.append((lo, hi))
        x = self.v.pop(0) if self.v else lo
        assert lo - 1e-9 <= x <= hi + 1e-9, (lo, x, hi)
        return x

    def randint(self, n):
        return self.ints.pop(0) if self.ints else 0


def make_db(tmp_path, instances):
    """instances: list of (local points (k, 3 or 4), size (3,), alphas)."""
    pts, cnt, size, oa, oc = [], [], [], [], []
    for loc, s, al in instances:
        loc = np.asarray(loc, np.float64)
        if loc.shape[1] == 3:
            loc = np.c_[loc, np.full(len(loc), 0.25)]
        pts.append(loc.astype(np.float32)); cnt.append(len(loc)); size.append(s); oa += list(al); oc.append(len(al))
    cnt = np.array(cnt); oc = np.array(oc)
    pp, ip = tmp_path / 'db_points.npy', tmp_path / 'db_index.npz'
    np.save(pp, np.concatenate(pts))
    np.savez(ip, offset=np.r_[0, np.cumsum(cnt)[:-1]], count=cnt, size=np.array(size, np.float64),
             obs_offset=np.r_[0, np.cumsum(oc)[:-1]], obs_count=oc, obs_alpha=np.array(oa, np.float64))
    return R.DonorDB(pp, ip)


SIZE = np.array([4.2, 1.8, 1.5])
CAR_LOCAL = R.to_box_frame(car_surface(np.r_[0, 0, 0, SIZE, 0.0], n_side=40, n_up=16), np.zeros(7))


def ground(slope=0.0, extent=46.0, step=0.4):
    g = np.stack(np.meshgrid(np.arange(-extent, extent, step), np.arange(-extent, extent, step)), -1).reshape(-1, 2)
    far = np.stack(np.meshgrid(np.arange(-74, 74, 1.0), np.arange(-74, 74, 1.0)), -1).reshape(-1, 2)
    far = far[np.abs(far).max(1) > extent]
    g = np.concatenate([g, far])
    return np.c_[g, slope * g[:, 1] - 0.03]


def scene_v2(extra=(), boxes=(), names=(), slope=0.0, **kw):
    pts = np.concatenate([ground(slope)] + list(extra))
    b = np.asarray(boxes, np.float64).reshape(-1, 7)
    return frame(b, list(names), pts, **kw)


P2 = dict(R.DEFAULTS); P2.update(R.DEFAULTS_V2); P2['N_COPIES'] = 1


def run2(d, db, rng, **over):
    p = dict(P2); p.update(over)
    return R.relocate_render_scene_v2(d, p, LAT, db, rng=rng, keep_points=True)


def test_v2_copy_on_the_plane_with_the_observed_view_one_point_per_pixel(tmp_path):
    db = make_db(tmp_path, [(CAR_LOCAL, SIZE, [0.7, -2.0])])
    d = scene_v2(slope=np.tan(np.radians(2.0)))
    out, rec = run2(d, db, RngV2([30.0, 1.0], ints=[0, 1]))
    assert rec['n_placed'] == 1, rec['reject']
    c = rec['copies'][0]; b = c['box']; pl = rec['plane']
    assert b[2] - b[5] / 2 == pytest.approx(R.plane_z(pl, b[0], b[1]), abs=1e-9)          # bottom on the plane
    assert abs(np.degrees(np.arctan(pl[1])) - 2.0) < 0.05
    assert c['alpha'] == -2.0 and b[6] == pytest.approx(1.0 - 2.0)                         # heading = phi + alpha
    assert R.relative_view_angle(b) == pytest.approx(-2.0, abs=1e-12)
    assert c['r_new'] == pytest.approx(30.0) and c['rc_loader'] >= 0.8 and c['vis_share_loader'] < 0.1
    p = out['points'][c['point_index']]
    row, col, rr, val = RU.to_lattice(p[:, :3], H, TH, NC)
    assert val.all() and len(np.unique(row * NC + col)) == len(p)
    assert np.all(p[:, 3] == np.float32(0.25)) and np.all(p[:, 4] == 0)
    assert not R.points_in_box(out['points'][:-len(p), :3], b).any()                       # in-box scene removed
    assert out['gt_names'].tolist() == ['Car'] and out['gt_boxes_mask'].tolist() == [True]
    assert c['p_keep'] == 1.0 and c['n_kept'] == c['n_visible']


def test_v2_visibility_rule_on_a_wall_toy(tmp_path):
    db = make_db(tmp_path, [(CAR_LOCAL, SIZE, [0.0])])
    wall = lambda zlo, zhi: np.stack(np.meshgrid(np.linspace(-6, 6, 241), np.linspace(zlo, zhi, 40)), -1).reshape(-1, 2)
    hi = wall(0.5, 3.0); hi = np.c_[hi[:, 0], np.full(len(hi), 25.0), hi[:, 1]]           # > 0.3 m above the plane
    lo = wall(-0.03, 0.2); lo = np.c_[lo[:, 0], np.full(len(lo), 25.0), lo[:, 1]]         # within 0.3 m of the plane
    out, rec = run2(scene_v2([hi]), db, RngV2([40.0, np.pi / 2] * 3), MAX_TRIES=3)
    assert rec['n_placed'] == 0 and rec['reject']['visibility'] == 3
    out, rec = run2(scene_v2([lo]), db, RngV2([40.0, np.pi / 2]), MAX_TRIES=1)
    assert rec['n_placed'] == 1 and rec['copies'][0]['vis_share_loader'] == 0.0
    part = hi[np.abs(hi[:, 0]) < 0.06]                                                   # a pole: few columns only
    side = make_db(tmp_path / 'side', [(CAR_LOCAL, SIZE, [np.pi / 2])]) if (tmp_path / 'side').mkdir() is None else None
    out, rec = run2(scene_v2([part]), side, RngV2([40.0, np.pi / 2]), MAX_TRIES=1)       # broadside: ~44 columns
    assert rec['n_placed'] == 1, rec['reject']
    assert 0 < rec['copies'][0]['vis_share_loader'] < 0.1


def test_v2_row_coverage_replacement(tmp_path):
    roof = CAR_LOCAL[CAR_LOCAL[:, 2] > SIZE[2] / 2 * 0.99]                                  # only the roof: RC low
    db = make_db(tmp_path, [(roof, SIZE, [0.0])])
    rng = RngV2([10.0, np.pi / 2, 15.0, 20.0, 26.0])
    out, rec = run2(scene_v2(), db, rng, MAX_TRIES=4)
    assert rec['n_placed'] == 0 and rec['reject']['rc'] == 4 and rec['n_no_place'] == 1
    assert rng.calls == [(5.0, 75.2), (0.0, 2 * np.pi), (10.0, 75.2), (15.0, 75.2), (20.0, 75.2)]


def test_v2_sparsifier_bounds_and_zero_point_drop(tmp_path):
    db = make_db(tmp_path, [(CAR_LOCAL, SIZE, [0.0])])
    d = scene_v2(); before = {k: np.copy(v) for k, v in d.items()}
    out, rec = run2(d, db, RngV2([30.0, 1.0, 1.0 - 1e-9], arr=0.5))                     # p = 1e-9: nothing kept
    assert rec['n_zero'] == 1 and rec['n_placed'] == 0
    for k in before:
        np.testing.assert_array_equal(out[k], before[k])
    out, rec = run2(scene_v2(), db, RngV2([30.0, 1.0, 0.5], arr=0.4999))                 # p = 0.5, u = 0.4999 < p
    assert rec['copies'][0]['p_keep'] == 0.5 and rec['copies'][0]['n_kept'] == rec['copies'][0]['n_visible']
    out, rec = run2(scene_v2(), db, RngV2([30.0, 1.0, 0.5], arr=0.5))                    # u = p is dropped: keep is u < p
    assert rec['n_zero'] == 1
    ps = []
    for s in range(30):                                                                  # the real RNG: p in (0, 1]
        np.random.seed(s)
        _, r2 = run2(scene_v2(), db, np.random, N_COPIES=2)
        ps += [c['p_keep'] for c in r2['copies']]
        assert all(c['n_kept'] <= c['n_visible'] for c in r2['copies'])
    assert len(ps) > 20 and min(ps) > 0 and max(ps) <= 1


@pytest.mark.parametrize('kind', ['neg_class', 'ignore99', 'masked', 'ignore_mask', 'other_class'])
def test_v2_sentinels_never_moved_always_obstacles(tmp_path, kind):
    db = make_db(tmp_path, [(CAR_LOCAL, SIZE, [0.0])])
    sb = np.array([0.0, 40.0, 0.72, 4.0, 2.0, 1.5, 0.0])
    kw, names = {}, ['Truck' if kind == 'other_class' else 'Car']
    if kind == 'neg_class':
        kw = dict(gt_classes=np.array([-1], np.int32), gt_scores=np.array([0.2], np.float32))
    elif kind == 'ignore99':
        kw = dict(gt_classes=np.array([common_utils.IGNORE_CLASS_LABEL], np.int32), gt_scores=np.ones(1, np.float32))
    elif kind == 'ignore_mask':
        kw = dict(gt_boxes_ignore_mask=np.array([True]))
    d = scene_v2([car_surface(sb)], [sb], names, **kw)
    if kind in ('masked', 'ignore_mask'):
        d['gt_boxes_mask'] = np.array([False])
    b0 = d['gt_boxes'].copy()
    out, rec = run2(d, db, RngV2([40.0, np.pi / 2] * 5), MAX_TRIES=5)                    # every try onto the sentinel
    assert rec['n_placed'] == 0 and rec['reject']['overlap'] == 5
    np.testing.assert_array_equal(out['gt_boxes'], b0)
    out, rec = run2(d, db, RngV2([40.0, 3 * np.pi / 2]))                                 # free elsewhere
    assert rec['n_placed'] == 1
    np.testing.assert_array_equal(out['gt_boxes'][0], b0[0])
    if 'gt_classes' in kw:
        assert out['gt_classes'].tolist() == [int(kw['gt_classes'][0]), 1]
    if kind == 'ignore_mask':
        assert out['gt_boxes_ignore_mask'].tolist() == [True, False]


def test_v2_no_plane_no_points_no_db(tmp_path):
    db = make_db(tmp_path, [(CAR_LOCAL, SIZE, [0.0])])
    d = frame(np.zeros((0, 7)), [], np.c_[np.random.default_rng(0).uniform(-2, 2, (50, 2)), np.zeros(50)])
    out, rec = run2(d, db, RngV2([30.0, 1.0]))
    assert rec['plane'] is None and rec['n_placed'] == 0
    out, rec = run2(frame(np.zeros((0, 7)), [], np.zeros((0, 3))), db, RngV2([]))
    assert rec['n_placed'] == 0


def test_v2_silhouette_and_row_coverage_equal_the_analysis():
    import source_coverage as S
    from lattice_sampler_precheck import silhouette as S_sil
    rng = np.random.default_rng(21)
    inc_desc = np.sort(TH)[::-1]
    for r_new, phi in ((12.0, 0.4), (35.0, 2.2), (61.0, -1.0)):
        b = np.array([r_new * np.cos(phi), r_new * np.sin(phi), 0.75, 4.4, 1.9, 1.5, phi + 0.6])
        sp, srow, tin = R.silhouette(b, TH, NC, H)
        sp2, srow2, tin2 = S_sil(b, inc_desc, hs=H, n_cols=NC)
        a = sorted(zip(len(TH) - 1 - srow, sp % NC, np.round(tin, 9)))
        c = sorted(zip(srow2, sp2 % NC, np.round(tin2, 9)))
        assert a == c and len(a) > 0
        q = R.from_box_frame(R.to_box_frame(car_surface(b, n_side=30, n_up=12), b) * 0.98, b)
        xyz, rows, cols, kept = R.render_object(q, TH, NC, H)
        wall = np.c_[rng.uniform(-3, 3, 400), rng.uniform(-3, 3, 400), rng.uniform(0, 3, 400)] * [1, 1, 1]
        wall[:, :2] += b[:2] * 0.7
        scene = np.concatenate([wall, np.c_[rng.uniform(-70, 70, (3000, 2)), np.zeros(3000)]])
        allp = np.concatenate([xyz, scene])
        ref = S.lattice_coverage(allp, b[None], [np.arange(len(xyz))], TH, NC, H)[0]
        _, _, crng, _ = RU.to_lattice(xyz, H, TH, NC)
        srw, scl, srn, svl = RU.to_lattice(scene, H, TH, NC)
        rc, nvr = R.row_coverage(b, rows * NC + cols, crng, np.where(svl, srw * NC + scl, -1), srn, TH, NC, H)
        assert rc == pytest.approx(ref[0], abs=1e-12) and nvr == ref[2]


def test_v2_db_is_never_pickled_with_its_points(tmp_path):
    big = np.random.default_rng(3).normal(0, 1, (200000, 3))
    db = make_db(tmp_path, [(big, SIZE, [0.0]), (CAR_LOCAL, SIZE, [0.1, 0.2])])
    assert db.points(0).shape == (200000, 4) and db._pts is not None
    s = pickle.dumps(db)
    assert len(s) < 20000
    db2 = pickle.loads(s)
    assert db2._pts is None and db2.n == 2
    np.testing.assert_array_equal(db2.points(1), db.points(1)); np.testing.assert_array_equal(db2.alphas(1), [0.1, 0.2])


def test_rr2_config_differs_from_27471_only_by_rr_v2_and_the_waymo_target(monkeypatch):
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
    monkeypatch.chdir(ROOT / 'tools')
    load = lambda n: cfg_from_yaml_file('cfgs/da-ieee-access/' + n, EasyDict())
    rrc = load('centerpoint-gblobs-accum-legaldepth-rr2-nuscenes2waymo.yaml')
    base = load('centerpoint-gblobs-accum-legaldepth-nuscenes2kitti.yaml')          # 27471
    waymo = load('centerpoint-gblobs-sourceonly-nuscenes2waymo.yaml')                # 27596's target block
    v1 = load('centerpoint-gblobs-accum-legaldepth-rr-nuscenes2waymo.yaml')
    assert 'VERSION' not in v1.DATA_CONFIGS.NUSCENES_N008.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]   # v1 unchanged
    assert rrc.MODEL == base.MODEL and rrc.OPTIMIZATION == base.OPTIMIZATION and rrc.CLASS_NAMES == base.CLASS_NAMES
    assert rrc.DATA_CONFIG_TAR == waymo.DATA_CONFIG_TAR
    for key in ('NUSCENES_N008', 'NUSCENES_N015'):
        a, b = rrc.DATA_CONFIGS[key], base.DATA_CONFIGS[key]
        la, lb = a.DATA_AUGMENTOR.AUG_CONFIG_LIST, b.DATA_AUGMENTOR.AUG_CONFIG_LIST
        assert la[0].NAME == 'relocate_render' and la[0].VERSION == 2 and la[1:] == lb
        strip = lambda c: {k: v for k, v in c.items() if k not in ('DATA_AUGMENTOR', '_BASE_CONFIG_')}
        assert strip(a) == strip(b)
        e = la[0]
        assert {k: e[k] for k in ('N_COPIES', 'MAX_TRIES', 'R_MIN', 'R_MAX', 'BEV_MARGIN', 'MAX_SCENE_POINTS', 'CLEARANCE',
                                  'VIS_MAX_SHARE', 'VIS_HEIGHT', 'RC_MIN', 'RC_OCCL_M', 'OCCLUSION_M', 'N_COLS',
                                  'MOUNT_HEIGHT', 'TIME_COLUMN')} == \
            dict(N_COPIES=8, MAX_TRIES=20, R_MIN=5.0, R_MAX=75.2, BEV_MARGIN=0.5, MAX_SCENE_POINTS=5, CLEARANCE=0.3,
                 VIS_MAX_SHARE=0.1, VIS_HEIGHT=0.3, RC_MIN=0.8, RC_OCCL_M=0.5, OCCLUSION_M=0.5, N_COLS=2650,
                 MOUNT_HEIGHT=2.184, TIME_COLUMN=4)                                   # T05 §5 v2, fixed
        assert e.DB_POINTS.startswith('/home/koyama/data/') and key in e.DB_POINTS and key in e.DB_INDEX
        if not (Path(e.DB_POINTS).exists() and Path(e.DB_INDEX).exists()):
            continue                                                                  # database not built yet
        aug = DataAugmentor(None, a.DATA_AUGMENTOR, rrc.CLASS_NAMES)
        q0 = aug.data_augmentor_queue[0]
        assert q0.func.__name__ == 'relocate_render' and q0.keywords['db'].n > 0 and q0.keywords['db']._pts is None
        assert len(pickle.dumps(aug)) < 5e7
    assert 'relocate_render' not in str(rrc.DATA_CONFIG_TAR)


# ================================================================================================= RR v3 (§7)
P3 = dict(P2); P3['VERSION'] = 3


def step_ground(z_hi=0.8, y_step=30.0):
    g = ground()
    g[:, 2] = np.where(g[:, 1] >= y_step, z_hi, -0.03)
    return g


def test_v3_ground_cells_are_second_lowest_of_cells_with_3_points():
    pts = np.array([[0.2, 0.2, 0.5], [0.4, 0.1, 0.1], [0.9, 0.9, 0.3], [0.5, 0.5, 0.1],     # cell (0, 0): 0.1, 0.1, ...
                    [1.5, 0.5, 2.0], [1.6, 0.4, 1.0],                                       # cell (1, 0): 2 points only
                    [-0.5, -0.5, 0.7], [-0.4, -0.6, 0.2], [-0.3, -0.2, 0.4]])               # cell (-1, -1)
    c = R.GroundCells(pts)
    assert c.vals[c.LIM, c.LIM] == 0.1 and np.isnan(c.vals[c.LIM + 1, c.LIM]) and c.vals[c.LIM - 1, c.LIM - 1] == 0.4
    v = c.local(0.0, 0.0, 1.0)                                       # centres (0.5, 0.5) and (-0.5, -0.5) at 0.71 m
    assert sorted(v.tolist()) == [0.1, 0.4]
    assert len(c.local(0.0, 0.0, 0.5)) == 0


def test_v3_local_height_on_a_step_toy():
    cells = R.GroundCells(step_ground())
    plane = np.array([0.0, 0.0, -0.03])
    zb, fb = R.local_bottom(cells, 0.0, 40.0, plane)                  # all local cells on the upper step
    assert not fb and zb == pytest.approx(0.8, abs=1e-9)
    zb, fb = R.local_bottom(cells, 0.0, 20.0, plane)                  # all on the lower level
    assert not fb and zb == pytest.approx(-0.03, abs=1e-9)
    v = cells.local(0.0, 30.2, 5.0)                                   # straddling the step: both levels present
    zb, fb = R.local_bottom(cells, 0.0, 30.2, plane)
    assert not fb and zb == pytest.approx(np.percentile(v, 20)) and (v > 0.5).any() and (v < 0).any()


def test_v3_fallback_below_three_cells():
    rng = np.random.default_rng(4)
    dense = np.c_[rng.uniform(-30, 30, (200000, 2)), np.zeros(200000)]
    sparse = np.array([[40.2, 0.3, 0.5], [40.4, 0.5, 0.6], [40.6, 0.7, 0.7],               # one cell with 3 points
                       [42.5, 1.5, 0.9], [42.6, 1.6, 1.0]])                                  # one cell with 2 points
    cells = R.GroundCells(np.concatenate([dense, sparse]))
    plane = np.array([0.01, 0.0, -0.02])
    zb, fb = R.local_bottom(cells, 41.0, 0.5, plane)
    assert fb and zb == pytest.approx(R.plane_z(plane, 41.0, 0.5))
    zb, fb = R.local_bottom(cells, 5.0, 5.0, plane)
    assert not fb and abs(zb) < 1e-9


def test_v3_scene_bottom_on_the_local_step_and_fallback_flag(tmp_path):
    db = make_db(tmp_path, [(CAR_LOCAL, SIZE, [0.0])])
    # a 0.25 m step: below the visibility rule's 0.3 m above the plane, so the step face does not hide the copy (a 0.8 m
    # step does, and the copy is then placed elsewhere)
    d = frame(np.zeros((0, 7)), [], step_ground(0.25))
    out, rec = run2(d, db, RngV2([40.0, np.pi / 2]), VERSION=3, MAX_TRIES=1)
    assert rec['version'] == 3 and rec['n_placed'] == 1, rec['reject']
    c = rec['copies'][0]
    assert c['ground_fallback'] is False and c['bottom'] == pytest.approx(0.25, abs=1e-6)
    assert c['r_new'] == pytest.approx(40.0)
    out2, rec2 = run2(frame(np.zeros((0, 7)), [], step_ground(0.25)), db, RngV2([40.0, np.pi / 2]), MAX_TRIES=1)  # v2
    assert rec2['version'] == 2 and rec2['copies'][0]['ground_fallback'] is None
    assert abs(rec2['copies'][0]['bottom'] - R.plane_z(rec2['plane'], 0.0, 40.0)) < 1e-9
    far = ground(); far = far[np.hypot(far[:, 0], far[:, 1]) < 45]                                   # nothing near 60 m
    out3, rec3 = run2(frame(np.zeros((0, 7)), [], far), db, RngV2([60.0, np.pi / 2]), VERSION=3, MAX_TRIES=1)
    assert rec3['n_placed'] == 1 and rec3['copies'][0]['ground_fallback'] is True
    assert rec3['copies'][0]['bottom'] == pytest.approx(R.plane_z(rec3['plane'], 0.0, 60.0), abs=1e-9)


def test_v3_footprint_distance_and_ground_reference_toy():
    b = np.array([10.0, 5.0, 1.0, 4.0, 2.0, 1.5, 0.7])
    pts = R.from_box_frame(np.array([[3.0, 0, 0], [0, 2.0, 0], [3.0, 2.0, 0], [0.5, 0.2, 0]]), b)
    np.testing.assert_allclose(R.footprint_distance(pts[:, :2], b), [1.0, 1.0, np.hypot(1.0, 1.0), 0.0], atol=1e-9)
    rng = np.random.default_rng(9)
    ring = R.from_box_frame(np.c_[rng.uniform(-4.5, 4.5, (4000, 2)), np.zeros(4000)], b)
    d = R.footprint_distance(ring[:, :2], b)
    ring = ring[(d > 0.5) & (d <= 3.0)]
    ring[:, 2] = 0.42 + rng.normal(0, 0.01, len(ring))
    clutter = ring[:200].copy(); clutter[:, 2] += 2.0                                   # a hedge above the ground
    inside = R.from_box_frame(np.c_[rng.uniform(-1.8, 1.8, 500), rng.uniform(-0.9, 0.9, 500), rng.uniform(-0.7, 0.7, 500)], b)
    ref, n = R.ground_reference(np.concatenate([ring, clutter, inside]), b)
    assert n == len(ring) + len(clutter) and abs(ref - 0.42) < 0.03
    ref, n = R.ground_reference(ring[:9], b)
    assert np.isnan(ref) and n == 9


def test_v3_hook_counts_copies_and_fallbacks(tmp_path):
    from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
    db = make_db(tmp_path, [(CAR_LOCAL, SIZE, [0.0])])
    cfg = EasyDict(dict(P3, NAME='relocate_render', DB_POINTS=db.points_path, DB_INDEX=db.index_path))
    aug = DataAugmentor(None, EasyDict(AUG_CONFIG_LIST=[cfg], DISABLE_AUG_LIST=[]), ['Car'])
    aug = pickle.loads(pickle.dumps(aug))
    np.random.seed(1)
    out = aug.forward(frame(np.zeros((0, 7)), [], step_ground()))
    st = aug._rr_stats
    assert st[0] == 1 and st[1] == len(out['gt_boxes']) and 0 <= st[2] <= st[1]


def test_rr3_config_is_rr2_plus_the_v3_keys(monkeypatch):
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(ROOT / 'tools')
    load = lambda n: cfg_from_yaml_file('cfgs/da-ieee-access/' + n, EasyDict())
    r3 = load('centerpoint-gblobs-accum-legaldepth-rr3-nuscenes2waymo.yaml')
    r2 = load('centerpoint-gblobs-accum-legaldepth-rr2-nuscenes2waymo.yaml')
    v1 = load('centerpoint-gblobs-accum-legaldepth-rr-nuscenes2waymo.yaml')
    for key in ('NUSCENES_N008', 'NUSCENES_N015'):
        e3 = r3.DATA_CONFIGS[key].DATA_AUGMENTOR.AUG_CONFIG_LIST
        e2 = r2.DATA_CONFIGS[key].DATA_AUGMENTOR.AUG_CONFIG_LIST
        assert e3[1:] == e2[1:] and e2[0].VERSION == 2 and e3[0].VERSION == 3
        assert {k: v for k, v in e3[0].items() if k != 'VERSION' and not k.startswith('GROUND_')} == \
            {k: v for k, v in e2[0].items() if k != 'VERSION'}
        assert {k: e3[0][k] for k in e3[0] if k.startswith('GROUND_')} == dict(
            GROUND_CELL=1.0, GROUND_CELL_MIN_PTS=3, GROUND_RADIUS=5.0, GROUND_MIN_CELLS=3, GROUND_PCT=20.0)
        assert not any(k.startswith('GROUND_') for k in e2[0]) and 'VERSION' not in v1.DATA_CONFIGS[key].DATA_AUGMENTOR.AUG_CONFIG_LIST[0]
        strip = lambda c: {k: v for k, v in c.items() if k != 'DATA_AUGMENTOR'}
        assert strip(r3.DATA_CONFIGS[key]) == strip(r2.DATA_CONFIGS[key])
    for k in ('MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'DATA_CONFIG_TAR'):
        assert r3[k] == r2[k]


# ================================================================================================= RR v4 (§8)
def ring_points(box, z, n, d_lo=0.6, d_hi=2.9, seed=0):
    """n points at BEV distance in [d_lo, d_hi] from the box's rectangle, at heights z (scalar or array)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        q = R.from_box_frame(np.c_[rng.uniform(-6, 6, (1, 2)), np.zeros(1)], box)
        d = R.footprint_distance(q[:, :2], box)[0]
        if d_lo <= d <= d_hi:
            out.append(q[0])
    out = np.array(out); out[:, 2] = z
    return out


def test_v4_nearby_height_with_one_low_outlier_among_12():
    b = np.array([20.0, 5.0, 0.0, 4.2, 1.8, 1.5, 0.4])
    z = np.r_[-5.0, np.full(11, 0.3)]
    pts = np.concatenate([ring_points(b, z, 12),
                          R.from_box_frame(np.array([[0.0, 0, -7.0], [1.0, 0.2, -7.0]]), b),          # inside: ignored
                          R.from_box_frame(np.array([[2.1 + 3.5, 0, -7.0], [0, 0.9 + 3.2, -7.0]]), b),  # > 3 m: ignored
                          R.from_box_frame(np.array([[2.1 + 0.3, 0, -7.0]]), b)])                     # < 0.5 m: ignored
    grid = R._BevGrid(pts)
    plane = np.array([0.0, 0.0, -0.4])
    zb, fb = R.nearby_bottom(pts, grid, b, plane)
    assert not fb and zb == pytest.approx(np.percentile(z, 10)) and zb == pytest.approx(0.3)
    assert np.percentile(z, 5) < -2                                                   # v3's reference quantile would not be
    zb, fb = R.nearby_bottom(pts[:9], R._BevGrid(pts[:9]), b, plane)                 # 9 nearby points: the plane
    assert fb and zb == pytest.approx(-0.4)
    zb, fb = R.nearby_bottom(pts[:10], R._BevGrid(pts[:10]), b, plane)               # exactly 10: used
    assert not fb and zb == pytest.approx(np.percentile(pts[:10, 2], 10))


def test_v4_z_range_rule_replaces_the_attempt(tmp_path):
    db = make_db(tmp_path, [(CAR_LOCAL, SIZE, [0.0])])
    g = ground()
    on = (np.abs(g[:, 0]) < 8) & (np.abs(g[:, 1] - 40) < 8)
    g[on, 2] = 3.5                                                                    # a platform 3.5 m up around (0, 40)
    pit = (np.abs(g[:, 0] - 40) < 8) & (np.abs(g[:, 1]) < 8)
    g[pit, 2] = -3.0                                                                  # a pit around (40, 0)
    d = frame(np.zeros((0, 7)), [], g)
    rng = RngV2([40.0, np.pi / 2, 40.0, 0.0, 25.0, np.pi])
    out, rec = run2(d, db, rng, VERSION=4, MAX_TRIES=3)
    assert rec['version'] == 4 and rec['reject']['zrange'] == 2 and rec['n_placed'] == 1, rec['reject']
    c = rec['copies'][0]
    assert c['r_new'] == pytest.approx(25.0) and c['ground_fallback'] is False
    assert abs(c['bottom'] - (-0.03)) < 1e-6 and c['bottom'] >= -2 and c['bottom'] + SIZE[2] <= 4
    out, rec = run2(frame(np.zeros((0, 7)), [], g), db, RngV2([40.0, np.pi / 2]), VERSION=3, MAX_TRIES=1)
    assert rec['reject']['zrange'] == 0                                               # v3 has no such rule


def test_v4_copy_bottom_matches_the_ground_cell_reference_on_flat_ground(tmp_path):
    db = make_db(tmp_path, [(CAR_LOCAL, SIZE, [0.0])])
    g = ground()
    d = frame(np.zeros((0, 7)), [], g)
    out, rec = run2(d, db, RngV2([30.0, 1.0]), VERSION=4, MAX_TRIES=1)
    c = rec['copies'][0]
    ref, fb = R.local_bottom(R.GroundCells(g.astype(np.float32)), c['box'][0], c['box'][1], rec['plane'])
    assert not fb and abs(c['bottom'] - ref) < 1e-6 and c['ground_fallback'] is False
    far = g[np.hypot(g[:, 0], g[:, 1]) < 45]
    out, rec = run2(frame(np.zeros((0, 7)), [], far), db, RngV2([60.0, np.pi / 2]), VERSION=4, MAX_TRIES=1)
    assert rec['copies'][0]['ground_fallback'] is True
    assert rec['copies'][0]['bottom'] == pytest.approx(R.plane_z(rec['plane'], 0.0, 60.0), abs=1e-9)


def test_rr4_config_is_rr3_plus_the_v4_keys_and_v1_to_v3_unchanged(monkeypatch):
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(ROOT / 'tools')
    load = lambda n: cfg_from_yaml_file('cfgs/da-ieee-access/' + n, EasyDict())
    r4 = load('centerpoint-gblobs-accum-legaldepth-rr4-nuscenes2waymo.yaml')
    r3 = load('centerpoint-gblobs-accum-legaldepth-rr3-nuscenes2waymo.yaml')
    r2 = load('centerpoint-gblobs-accum-legaldepth-rr2-nuscenes2waymo.yaml')
    v1 = load('centerpoint-gblobs-accum-legaldepth-rr-nuscenes2waymo.yaml')
    for key in ('NUSCENES_N008', 'NUSCENES_N015'):
        e4, e3 = (c.DATA_CONFIGS[key].DATA_AUGMENTOR.AUG_CONFIG_LIST for c in (r4, r3))
        assert e4[1:] == e3[1:] and e4[0].VERSION == 4 and e3[0].VERSION == 3
        assert r2.DATA_CONFIGS[key].DATA_AUGMENTOR.AUG_CONFIG_LIST[0].VERSION == 2
        assert {k: v for k, v in e4[0].items() if k != "VERSION" and not k.startswith("NEAR_")} == \
            {k: v for k, v in e3[0].items() if k != 'VERSION'}
        assert {k: e4[0][k] for k in e4[0] if k.startswith('NEAR_')} == dict(NEAR_D_LO=0.5, NEAR_D_HI=3.0,
                                                                             NEAR_MIN_POINTS=10, NEAR_PCT=10.0)
        for c in (r3, r2, v1):
            assert not any(k.startswith('NEAR_') for k in c.DATA_CONFIGS[key].DATA_AUGMENTOR.AUG_CONFIG_LIST[0])
        strip = lambda c: {k: v for k, v in c.items() if k != 'DATA_AUGMENTOR'}
        assert strip(r4.DATA_CONFIGS[key]) == strip(r3.DATA_CONFIGS[key])
    for k in ('MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'DATA_CONFIG_TAR'):
        assert r4[k] == r3[k]


def test_v4_hook_counts(tmp_path):
    from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
    db = make_db(tmp_path, [(CAR_LOCAL, SIZE, [0.0])])
    cfg = EasyDict(dict(P2, VERSION=4, NAME='relocate_render', DB_POINTS=db.points_path, DB_INDEX=db.index_path))
    aug = pickle.loads(pickle.dumps(DataAugmentor(None, EasyDict(AUG_CONFIG_LIST=[cfg], DISABLE_AUG_LIST=[]), ['Car'])))
    np.random.seed(3)
    out = aug.forward(frame(np.zeros((0, 7)), [], ground()))
    assert aug._rr_stats[0] == 1 and aug._rr_stats[1] == len(out['gt_boxes'])


# ================================================================================================= RR v5 (§9)
def test_v5_normals_on_a_plane_toy_and_the_fallback():
    import rr_build_db as BD
    rng = np.random.default_rng(0)
    g = np.stack(np.meshgrid(np.arange(0, 2, 0.02), np.arange(0, 1, 0.02)), -1).reshape(-1, 2)
    tilt = np.c_[g, 0.3 * g[:, 0] + rng.normal(0, 1e-4, len(g))]                       # a plane z = 0.3 x
    far = np.array([[10.0, 10.0, 10.0], [10.05, 10.0, 10.0], [10.0, 10.05, 10.0]])       # 2 other points only
    n, deg = BD.instance_normals(np.concatenate([tilt, far]))
    ref = np.array([-0.3, 0.0, 1.0]) / np.hypot(0.3, 1.0)
    ok = deg >= 5
    assert ok[:len(tilt)].all() and not ok[len(tilt):].any()
    np.testing.assert_allclose(np.abs(n[:len(tilt)] @ ref), 1.0, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(n[:len(tilt)], axis=1), 1.0, atol=1e-5)
    assert np.all(n[len(tilt):] == 0)
    n0, d0 = BD.instance_normals(np.zeros((0, 3)))
    assert n0.shape == (0, 3)


def _copy_render(r=40.0, phi=np.pi / 2, alpha=np.pi / 2):
    b, q = R.place_instance(CAR_LOCAL, SIZE, r, phi, alpha, np.array([0.0, 0.0, 0.0]))
    xyz, rows, cols, kept, rr = R._render(q, TH, NC, H)
    return b, xyz, rows, cols, kept


def test_v5_occlusion_envelope_toy():
    b, xyz, rows, cols, kept = _copy_render()
    sil = R.silhouette(b, TH, NC, H)
    plane = np.array([0.0, 0.0, 0.0])
    ground = np.c_[np.random.default_rng(0).uniform(-50, 50, (2000, 2)), np.zeros(2000)]
    def env_for(extra):
        occ = R.Occluders(np.concatenate([ground, extra]), plane, H, NC)
        return R.envelope_visibility(xyz, rows, cols, sil, TH, NC, occ)
    # nothing in front: all visible
    e0 = env_for(np.zeros((0, 3)))
    assert e0['rend_vis'].all() and e0['vis_frac'] == 1.0 and e0['rc'] >= 0.8
    # a car in front at 25 m (0.3-1.5 m above the plane): the copy's lower rows hidden, its upper rows visible
    car = np.stack(np.meshgrid(np.linspace(-4, 4, 161), np.linspace(0.35, 1.5, 24)), -1).reshape(-1, 2)
    car = np.c_[car[:, 0], np.full(len(car), 25.0), car[:, 1]]
    e1 = env_for(car)
    el_top = np.arctan2(1.5 - H, 25.0)
    assert (~e1['rend_vis']).any() and e1['rend_vis'].any()
    np.testing.assert_array_equal(e1['rend_vis'], TH[rows] > el_top)
    # a wall up to 3.0 m at 25 m: everything hidden (its top is above the copy's top as seen from 2.184 m)
    wall = np.stack(np.meshgrid(np.linspace(-6, 6, 241), np.linspace(0.35, 3.0, 54)), -1).reshape(-1, 2)
    wall = np.c_[wall[:, 0], np.full(len(wall), 25.0), wall[:, 1]]
    e2 = env_for(wall)
    assert not e2['rend_vis'].any() and e2['vis_frac'] == 0.0
    # a canopy 3.2-4 m above the plane in front: ignored (occluders are (0.3, 3.0] m above the plane)
    can = wall.copy(); can[:, 2] = can[:, 2] / 3.0 * 0.8 + 3.2
    e3 = env_for(can)
    assert e3['rend_vis'].all()
    # an occluder BEHIND the copy never hides it
    e4 = env_for(np.c_[wall[:, 0], np.full(len(wall), 60.0), wall[:, 2]])
    assert e4['rend_vis'].all()


def test_v5_return_model_p_bounds_and_monotonic_in_i_rel():
    par = dict(c0=0.3, c_band=np.array([-0.5, 0.0, -1.0, 0.4]), beta=0.8, gamma=-1.2, delta=0.6, log_sigma=np.log(0.7))
    n = 200
    i_rel = np.geomspace(0.05, 20, n); h = np.full(n, 0.4); r = np.full(n, 55.0); cos = np.full(n, 0.5)
    lg = R.return_logit((i_rel, h, r, cos), 'physics', par)
    p = 1 / (1 + np.exp(-lg))
    assert np.all((p > 0) & (p < 1)) and np.all(np.diff(p) > 0)
    lg2 = R.return_logit((i_rel, np.full(n, 0.6), r, cos), 'physics', par)
    np.testing.assert_allclose(lg2 - lg, -1.0)                                        # band 2 vs the body band
    mlp = dict(W1=np.ones((4, 32)) * 0.01, b1=np.zeros(32), W2=np.eye(32), b2=np.zeros(32), W3=np.ones((32, 1)), b3=np.zeros(1),
               log_sigma=0.0)
    pm = 1 / (1 + np.exp(-R.return_logit((i_rel, h, r, cos), 'mlp', mlp)))
    assert np.all((pm > 0) & (pm < 1))
    f = R.return_features(np.c_[CAR_LOCAL, np.full(len(CAR_LOCAL), 10.0)], np.arange(5), _copy_render()[1][:5],
                          _copy_render()[0], H, None)
    assert np.all(f[0] == 1.0) and np.all(f[3] == 1.0) and np.all((f[1] >= 0) & (f[1] <= 1))


def test_v5_fit_is_deterministic_on_a_toy():
    import torch
    import rr_v5_fit as FT
    rng = np.random.default_rng(1)
    paths = []
    import tempfile
    d = Path(tempfile.mkdtemp())
    for key, n in (('NUSCENES_N008', 30), ('NUSCENES_N015', 20)):
        C, sc, sr, srel, sf, F = [], [], [], [], [], []
        fo = ro = 0
        for c in range(n):
            r = rng.uniform(15, 75); C.append([0, c, r, 1.0, c])
            npx = 40; rows = np.repeat(np.arange(8), 5)
            sc.append(np.full(npx, c)); sr.append(ro + rows); srel.append(np.repeat(np.linspace(0.05, 0.95, 8), 5))
            has = rng.random(npx) < 0.8
            fi = np.where(has, fo + np.cumsum(has) - 1, -1); sf.append(fi)
            F.append(np.c_[rng.uniform(0.1, 5, has.sum()), rng.uniform(0, 1, has.sum()), np.full(has.sum(), r),
                           rng.uniform(0.1, 1, has.sum())])
            fo += int(has.sum()); ro += 8
        p = d / f'copies_{key}.npz'
        np.savez(p, C=np.array(C, float), s_copy=np.concatenate(sc), s_row=np.concatenate(sr), s_relh=np.concatenate(srel),
                 s_feat=np.concatenate(sf), F=np.concatenate(F), n_all=n, frames=np.arange(n), dropped=np.zeros(0))
        paths.append(p)
    cd = FT.CopyData(paths)
    tgt = (np.full((5, 4), 0.5), np.full((5, 4), 0.1), np.tile([0.3, 0.5, 0.6], (5, 1)))
    out = []
    for _ in range(2):
        m, hist = FT.fit_model(cd, tgt, 'a', steps=30, log_every=10)
        out.append((m.export(), hist))
    for k in out[0][0]:
        np.testing.assert_array_equal(np.asarray(out[0][0][k]), np.asarray(out[1][0][k]))
    assert out[0][1] == out[1][1] and out[0][1][-1][1] < out[0][1][0][1]
    mb, hb = FT.fit_model(cd, tgt, 'b', steps=10, log_every=5)
    mb2, hb2 = FT.fit_model(cd, tgt, 'b', steps=10, log_every=5)
    np.testing.assert_array_equal(mb.export()['W1'], mb2.export()['W1'])
    # the numpy loader logit equals the torch one
    par = out[0][0]
    F = cd.F.numpy()
    ma = FT.Model('a'); ma.theta.data = torch.tensor([par['c0'], par['c_band'][0], par['c_band'][2], par['c_band'][3],
                                                     par['beta'], par['gamma'], par['delta']], dtype=torch.float64)
    np.testing.assert_allclose(R.return_logit((F[:, 0], F[:, 1], F[:, 2], F[:, 3]), 'physics', par),
                               ma.logit(cd.F).detach().numpy(), atol=1e-9)
    np.testing.assert_allclose(_torch_logit_a(par, F), ma.logit(cd.F).detach().numpy(), atol=1e-9)
    np.testing.assert_allclose(R.return_logit((F[:, 0], F[:, 1], F[:, 2], F[:, 3]), 'mlp', mb.export()),
                               mb.logit(cd.F).detach().numpy(), atol=1e-9)


def _torch_logit_a(par, F):
    band = np.clip(np.floor(F[:, 1] * 4).astype(int), 0, 3)
    return (par['c0'] + par['c_band'][band] + par['beta'] * np.log(F[:, 0]) + par['gamma'] * np.log(F[:, 2] / 50.0)
            + par['delta'] * np.log(F[:, 3]))


def make_db_v5(tmp_path, instances):
    db = make_db(tmp_path, instances)
    import rr_build_db as BD
    P = np.load(db.points_path)
    N = np.zeros((len(P), 3), np.float32)
    for k in range(db.n):
        N[db.offset[k]:db.offset[k] + db.count[k]] = BD.instance_normals(P[db.offset[k]:db.offset[k] + db.count[k], :3])[0]
    npth = tmp_path / 'db_normals.npy'; np.save(npth, N)
    return R.DonorDB(db.points_path, db.index_path, npth)


def test_v5_scene_occlusion_return_model_and_sentinels(tmp_path):
    db = make_db_v5(tmp_path, [(CAR_LOCAL, SIZE, [np.pi / 2])])
    par = dict(c0=0.0, c_band=np.zeros(4), beta=0.0, gamma=0.0, delta=0.0, log_sigma=np.log(1e-6))
    pp = tmp_path / 'fit_a.npz'; np.savez(pp, **{k: np.asarray(v, float) for k, v in par.items()})
    sb = np.array([0.0, 40.0, 0.72, 4.0, 2.0, 1.5, 0.0])
    d = scene_v2([car_surface(sb)], [sb], ['Car'], gt_classes=np.array([-1], np.int32), gt_scores=np.ones(1, np.float32))
    b0 = d['gt_boxes'].copy()
    base = dict(VERSION=5, RETURN_MODEL='physics', RETURN_PARAMS=str(pp), MAX_TRIES=3)
    out, rec = run2(d, db, RngV2([40.0, np.pi / 2] * 3), **base)                       # onto the sentinel: overlap
    assert rec['n_placed'] == 0 and rec['reject']['overlap'] == 3
    np.testing.assert_array_equal(out['gt_boxes'], b0)
    out, rec = run2(d, db, RngV2([40.0, 3 * np.pi / 2], arr=0.3), **base)            # p = 0.5 everywhere, u = 0.3 < p
    assert rec['n_placed'] == 1 and rec['version'] == 5, rec['reject']
    c = rec['copies'][0]
    assert c['n_kept'] == c['n_visible'] and abs(c['p_keep'] - 0.5) < 1e-6 and c['vis_frac'] == 1.0
    assert out['gt_classes'].tolist() == [-1, 1]
    np.testing.assert_array_equal(out['gt_boxes'][0], b0[0])
    out, rec = run2(scene_v2(), db, RngV2([40.0, 1.0], arr=0.6), **base)              # u = 0.6 >= p: 0 points, dropped
    assert rec['n_zero'] == 1 and rec['n_placed'] == 0
    wall = np.stack(np.meshgrid(np.linspace(-6, 6, 241), np.linspace(0.35, 3.0, 54)), -1).reshape(-1, 2)
    wall = np.c_[wall[:, 0], np.full(len(wall), 25.0), wall[:, 1]]
    out, rec = run2(scene_v2([wall]), db, RngV2([40.0, np.pi / 2] * 3), **base)      # fully hidden: novis, re-placed
    assert rec['reject']['novis'] >= 1 and rec['reject']['visibility'] == 0
    out, rec = run2(scene_v2(), db, RngV2([40.0, 1.0]), **dict(base, RETURN_OFF=True))
    c = rec['copies'][0]
    assert c['n_kept'] == c['n_visible'] and c['vis_feats'].shape == (c['n_visible'], 4) and (c['vis_feats'][:, 3] <= 1).all()
    assert len(c['sil_vis_pix']) == len(c['sil_vis_relh']) and c['n_sil_vis'] == c['n_sil']


def test_rr5_configs_are_rr4_plus_the_v5_keys_and_v1_to_v4_unchanged(monkeypatch):
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(ROOT / 'tools')
    load = lambda n: cfg_from_yaml_file('cfgs/da-ieee-access/' + n, EasyDict())
    r4 = load('centerpoint-gblobs-accum-legaldepth-rr4-nuscenes2waymo.yaml')
    for arm, kind in (('a', 'physics'), ('b', 'mlp')):
        r5 = load(f'centerpoint-gblobs-accum-legaldepth-rr5{arm}-nuscenes2waymo.yaml')
        for key in ('NUSCENES_N008', 'NUSCENES_N015'):
            e5, e4 = (c.DATA_CONFIGS[key].DATA_AUGMENTOR.AUG_CONFIG_LIST for c in (r5, r4))
            assert e5[1:] == e4[1:] and e5[0].VERSION == 5 and e4[0].VERSION == 4 and e5[0].RETURN_MODEL == kind
            new = {'VERSION', 'RETURN_MODEL', 'RETURN_PARAMS', 'OCC_H_LO', 'OCC_H_HI', 'OCC_MARGIN', 'OCC_NEIGHBOUR_COLS',
                   'I_REL_CLIP', 'R_REF', 'COS_MIN', 'DB_NORMALS'}
            assert {k: v for k, v in e5[0].items() if k not in new} == {k: v for k, v in e4[0].items() if k != 'VERSION'}
            assert (e5[0].OCC_H_LO, e5[0].OCC_H_HI, e5[0].OCC_MARGIN, e5[0].OCC_NEIGHBOUR_COLS, list(e5[0].I_REL_CLIP),
                    e5[0].R_REF) == (0.3, 3.0, 0.5, 1, [0.05, 20.0], 50.0)
            assert key in e5[0].DB_NORMALS and e5[0].RETURN_PARAMS.endswith(f'fit_{arm}.npz')
            assert not any(k in e4[0] for k in new - {'VERSION'})
        for k in ('MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'DATA_CONFIG_TAR'):
            assert r5[k] == r4[k]


# ================================================================================================= RR v6 (§10)
def test_v6_nearby_radius_10m_outside_the_grown_footprint():
    b = np.array([30.0, 0.0, 0.0, 4.2, 1.8, 1.5, 0.0])
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring8 = np.c_[30 + 8.0 * np.cos(ang), 8.0 * np.sin(ang), np.full(40, 0.2)]       # within 10 m: used
    ring11 = np.c_[30 + 11.0 * np.cos(ang), 11.0 * np.sin(ang), np.full(40, -3.0)]   # beyond 10 m: ignored
    inner = np.c_[np.full(20, 30 + 2.1 + 0.3), np.linspace(-0.5, 0.5, 20), np.full(20, -3.0)]   # within 0.5 m of it: ignored
    pts = np.concatenate([ring8, ring11, inner])
    plane = np.array([0.0, 0.0, -1.0])
    zb, fb = R.nearby_bottom(pts, R._BevGrid(pts), b, plane, 0.5, 3.0, 10, 10.0, radius=10.0)
    assert not fb and zb == pytest.approx(0.2)
    zb3, fb3 = R.nearby_bottom(pts, R._BevGrid(pts), b, plane, 0.5, 3.0, 10, 10.0)        # v4 / v5 rule: none within 3 m
    assert fb3 and zb3 == pytest.approx(-1.0)
    zb9, fb9 = R.nearby_bottom(ring8[:9], R._BevGrid(ring8[:9]), b, plane, 0.5, 3.0, 10, 10.0, radius=10.0)
    assert fb9


def test_v6_plane_over_75m_equals_ground_tilt_with_that_range():
    import ground_tilt as GT
    rng = np.random.default_rng(13)
    g = rng.uniform(-80, 80, (120000, 2))
    z = 0.03 * g[:, 1] + rng.normal(0, 0.03, len(g))
    z[::5] += rng.uniform(0.5, 4, len(z[::5]))
    pts = np.c_[g, z]
    np.testing.assert_array_equal(R.lowest_per_cell(pts, 75.2), GT.lowest_per_cell(pts, rmax=75.2))
    np.testing.assert_allclose(R.ground_plane(pts, 75.2),
                               GT.fit_plane(GT.lowest_per_cell(pts, rmax=75.2), rng=np.random.default_rng(0))[0])
    np.testing.assert_array_equal(R.lowest_per_cell(pts), GT.lowest_per_cell(pts))            # the 40 m default unchanged


def test_v6_patch_shapes_flips_and_rank_transform():
    r_ = R.intensity_rank(np.array([5.0, 1.0, 1.0, 9.0]))
    np.testing.assert_allclose(r_, [0.75, 0.375, 0.375, 1.0])
    v = np.random.default_rng(0).uniform(0.01, 5, 200)                 # tanh saturates to 1.0 (ties) above ~19
    np.testing.assert_array_equal(R.intensity_rank(np.tanh(v)), R.intensity_rank(v))           # monotone-invariant
    np.testing.assert_array_equal(R.intensity_rank(np.log(v / np.median(v))), R.intensity_rank(v))
    sr = np.repeat(np.arange(10, 15), 8); sc = np.tile(np.arange(2646, 2654) % 2650, 5)      # wraps at the seam
    vis = np.ones(len(sr), bool); vis[:8] = False
    rr_, rc_ = np.array([12, 14]), np.array([2649, 1])
    out = R.build_patch(sr, sc, vis, rr_, rc_, np.array([0.3, -0.2]), np.array([0.5, 1.0]), np.array([0.2, 0.9]),
                        np.log(60 / 50), 2650, fixed=True)
    X, (ri, rj), (si, sj) = out
    assert X.shape == (6, 48, 96) and X[4].sum() == 32 and X[3].sum() == 8 and np.allclose(X[5], np.log(1.2))
    assert (ri.tolist(), rj.tolist()) == ([2, 4], [3, 5])
    assert X[0, 2, 3] == np.float32(0.3) and X[1, 4, 5] == 1.0 and X[2, 2, 3] == np.float32(0.2)
    Xf, (rif, rjf), _ = R.build_patch(sr, sc, vis, rr_, rc_, np.zeros(2), np.ones(2), np.ones(2), 0.0, 2650,
                                      flip_rows=True, flip_cols=True, fixed=True)
    assert (rif.tolist(), rjf.tolist()) == ([2, 0], [4, 2])
    big = R.build_patch(np.repeat(np.arange(60), 100), np.tile(np.arange(100), 60), np.ones(6000, bool), rr_, rc_,
                        np.zeros(2), np.ones(2), np.ones(2), 0.0, 2650, fixed=True)
    assert big is None
    Xv, _, _ = R.build_patch(np.repeat(np.arange(61), 101), np.tile(np.arange(101), 61), np.ones(6161, bool), rr_, rc_,
                             np.zeros(2), np.ones(2), np.ones(2), 0.0, 2650, fixed=False)
    assert Xv.shape == (6, 64, 104)


def test_v6_label_construction_on_a_toy_track_frame():
    import rr_v6_unet as UN
    sil = np.array([10 * 2650 + 5, 10 * 2650 + 6, 11 * 2650 + 5, 11 * 2650 + 6, 12 * 2650 + 5])
    vis = np.array([True, True, True, False, True])
    real = np.array([10 * 2650 + 6, 11 * 2650 + 6, 99])                                 # one hidden pixel has a return
    si, sj = np.array([0, 0, 1, 1, 2]), np.array([0, 1, 0, 1, 0])
    Y, lab = UN.patch_labels(sil, vis, real, si, sj)
    assert Y.shape == (48, 96) and Y.sum() == 1 and Y[0, 1] == 1 and Y[1, 1] == 0 and lab.tolist() == [0, 1, 0, 1, 0]
    occ = (np.array([5, 6]), np.array([20.0, 20.0]), np.radians(np.array([1.0, -3.0])))
    e = UN.envelope_e(np.array([5, 6, 7, 9]), np.array([40.0, 40.0, 40.0, 40.0]), occ)
    np.testing.assert_allclose(e[:3], np.radians([1.0, 1.0, -3.0])); assert e[3] == -np.inf
    e2 = UN.envelope_e(np.array([5]), np.array([20.3]), occ)                           # within the 0.5 m margin: no
    assert e2[0] == -np.inf


def test_v6_cnn_keep_prob_bounds_no_weight_pickling_and_sentinels(tmp_path):
    import torch
    from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
    torch.manual_seed(0)
    net = R._unet_class()()
    wp = tmp_path / 'unet.pt'; torch.save(net.state_dict(), wp)
    db = make_db_v5(tmp_path, [(np.c_[CAR_LOCAL, np.random.default_rng(0).uniform(1, 50, len(CAR_LOCAL))], SIZE, [np.pi / 2])])
    sb = np.array([0.0, 40.0, 0.72, 4.0, 2.0, 1.5, 0.0])
    d = scene_v2([car_surface(sb)], [sb], ['Car'], gt_classes=np.array([common_utils.IGNORE_CLASS_LABEL], np.int32),
                 gt_scores=np.ones(1, np.float32))
    b0 = d['gt_boxes'].copy()
    base = dict(VERSION=6, RETURN_MODEL='cnn', CNN_WEIGHTS=str(wp), MAX_TRIES=3, PLANE_RMAX=75.2, NEAR_RADIUS=10.0)
    out, rec = run2(d, db, RngV2([40.0, np.pi / 2] * 3), **base)
    assert rec['n_placed'] == 0 and rec['reject']['overlap'] == 3
    np.testing.assert_array_equal(out['gt_boxes'], b0)
    out, rec = run2(d, db, RngV2([40.0, 3 * np.pi / 2], arr=0.0), **base)                 # u = 0 < p: every visible pixel kept
    assert rec['n_placed'] == 1 and rec['version'] == 6, rec['reject']
    c = rec['copies'][0]
    assert 0 < c['p_keep'] < 1 and 0 < c['n_kept'] <= c['n_visible']     # rendered pixels outside the patch get p = 0
    np.testing.assert_array_equal(out['gt_boxes'][0], b0[0])
    b, xyz, rows, cols, kept = _copy_render(30.0, 1.0, 0.0)
    sil = R.silhouette(b, TH, NC, H)
    env = R.envelope_visibility(xyz, rows, cols, sil, TH, NC, R.Occluders(np.zeros((0, 3)), np.zeros(3), H, NC))
    loc = np.c_[CAR_LOCAL, np.linspace(1, 9, len(CAR_LOCAL))]
    p = R.cnn_keep_prob(loc, kept, xyz, rows, cols, b, sil, env, TH, NC, H, str(wp))
    sp_, sr_, _ = sil
    inbox = (rows >= sr_.min()) & (rows <= sr_.max()) & np.isin(cols, np.unique(sp_ % NC))
    assert p.shape == (len(kept),) and np.all((p[inbox] > 0) & (p[inbox] < 1)) and np.all(p[~inbox] == 0)
    cfg = EasyDict(dict(P2, VERSION=6, RETURN_MODEL='cnn', CNN_WEIGHTS=str(wp), NAME='relocate_render',
                        DB_POINTS=db.points_path, DB_INDEX=db.index_path, DB_NORMALS=db.normals_path))
    aug = DataAugmentor(None, EasyDict(AUG_CONFIG_LIST=[cfg], DISABLE_AUG_LIST=[]), ['Car'])
    s = pickle.dumps(aug)
    assert len(s) < 20000 and b'state_dict' not in s and wp.stat().st_size > 100000   # weights never pickled
    assert R._load_cnn.cache_info().currsize >= 1                                      # loaded lazily in this process


def test_rr6_configs_are_rr5a_plus_the_v6_keys_and_v1_to_v5_unchanged(monkeypatch):
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(ROOT / 'tools')
    load = lambda n: cfg_from_yaml_file('cfgs/da-ieee-access/' + n, EasyDict())
    r5 = load('centerpoint-gblobs-accum-legaldepth-rr5a-nuscenes2waymo.yaml')
    for arm in ('a', 'b'):
        r6 = load(f'centerpoint-gblobs-accum-legaldepth-rr6{arm}-nuscenes2waymo.yaml')
        for key in ('NUSCENES_N008', 'NUSCENES_N015'):
            e6, e5 = (c.DATA_CONFIGS[key].DATA_AUGMENTOR.AUG_CONFIG_LIST for c in (r6, r5))
            assert e6[1:] == e5[1:] and e6[0].VERSION == 6 and e5[0].VERSION == 5
            assert (e6[0].PLANE_RMAX, e6[0].NEAR_RADIUS, e6[0].NEAR_D_LO, e6[0].NEAR_MIN_POINTS, e6[0].NEAR_PCT) == \
                (75.2, 10.0, 0.5, 10, 10.0)
            assert 'PLANE_RMAX' not in e5[0] and 'NEAR_RADIUS' not in e5[0] and e5[0].NEAR_D_HI == 3.0
            if arm == 'a':
                assert e6[0].RETURN_MODEL == 'physics' and e6[0].RETURN_PARAMS.endswith('v6/fit_a.npz')
            else:
                assert e6[0].RETURN_MODEL == 'cnn' and e6[0].CNN_WEIGHTS.endswith('v6/unet_b.pt') and 'RETURN_PARAMS' not in e6[0]
            same = {'VERSION', 'PLANE_RMAX', 'NEAR_RADIUS', 'RETURN_MODEL', 'RETURN_PARAMS', 'CNN_WEIGHTS'}
            assert {k: v for k, v in e6[0].items() if k not in same} == {k: v for k, v in e5[0].items() if k not in same}
        for k in ('MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'DATA_CONFIG_TAR'):
            assert r6[k] == r5[k]


# ================================================================================================= RR v7 (§12)
def _v7_cloud(seed=0, hole=(35.0, -35.0, 12.0)):
    """A tilted noisy ground over +-60 m with a square hole (no points within `hole[2]` of (hole[0], hole[1])) and a wall."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(-60, 60, (90000, 2))
    g = g[np.abs(g - np.array(hole[:2])).max(1) > hole[2]]
    z = 0.01 * g[:, 0] - 0.02 * g[:, 1] + rng.normal(0, 0.02, len(g))
    wy, wz = np.meshgrid(np.linspace(-8, 8, 120), np.linspace(0.2, 3.0, 20))
    wall = np.c_[np.full(wy.size, -20.0), wy.ravel(), wz.ravel()]
    return np.concatenate([np.c_[g, z], wall])


def _v7_box(x, y, bottom, yaw=0.3):
    return np.array([x, y, bottom + 0.75, 4.5, 1.9, 1.5, yaw])


def test_v7_e5a_and_its_fallback_equal_the_analysis():
    import rr_ground_explore as GE
    P = _v7_cloud()
    pl = R.ground_plane(P, 40.0)
    grid = R._BevGrid(P)
    rng = np.random.default_rng(3)
    cars = np.stack([_v7_box(x, y, R.plane_z(pl, x, y) + rng.normal(0, 0.2)) for x, y in rng.uniform(-55, 55, (12, 2))])
    kinds = []
    for t in range(40):
        x, y = rng.uniform(-55, 55, 2) if t < 36 else (35.0 + (t - 36), -35.0)        # the last ones sit in the hole
        b = _v7_box(x, y, 0.0, rng.uniform(-np.pi, np.pi))
        oth = cars[rng.random(len(cars)) < 0.5] if t % 3 else np.zeros((0, 7))
        z, kind = R.e5a_bottom(P, grid, b, pl, oth)
        est = GE.estimate_car(P, b, oth, pl, pl)
        if not est['E5a'][1]:
            assert kind == 0 and np.isclose(z, est['E5a'][0]), t
        else:
            zc, fb = est['E2-cell-p5']
            assert kind == (2 if fb else 1) and np.isclose(z, zc), t
        kinds.append(kind)
    assert {0, 1, 2} <= set(kinds)                                     # every path exercised


def test_v7_e5a_radius_boundary_and_the_median_residual():
    pl = np.array([0.01, 0.0, -0.1])
    b = _v7_box(10.0, 0.0, 0.0)
    near = [_v7_box(10.0, 30.0, R.plane_z(pl, 10.0, 30.0) + 0.3), _v7_box(40.0, 0.0, R.plane_z(pl, 40.0, 0.0) + 0.1),
            _v7_box(10.0, -12.0, R.plane_z(pl, 10.0, -12.0) - 0.2)]
    z, kind = R.e5a_bottom(np.zeros((0, 3)), None, b, pl, np.stack(near))           # 30.0 m away is included
    assert kind == 0 and np.isclose(z, R.plane_z(pl, 10.0, 0.0) + 0.1)
    far = _v7_box(10.0, 30.5, 5.0)
    z, kind = R.e5a_bottom(np.zeros((0, 3)), None, b, pl, far[None])               # 30.5 m: none -> no cloud -> plane
    assert kind == 2 and np.isclose(z, R.plane_z(pl, 10.0, 0.0))
    few = np.array([[15.0, 0.0, 0.0]] * 3 + [[17.5, 0.5, 0.1]] * 3)                 # 2 cells: < 3 -> the plane
    z, kind = R.e5a_bottom(few, R._BevGrid(few), b, pl, np.zeros((0, 7)))
    assert kind == 2
    cells = np.array([[x + 0.5, 0.5, zz] for x in (15.0, 16.0, 17.0, 18.0) for zz in (0.3, 0.1, 0.2)])
    z, kind = R.e5a_bottom(cells, R._BevGrid(cells), b, pl, np.zeros((0, 7)))      # 4 cells, second-lowest 0.2 each
    assert kind == 1 and np.isclose(z, 0.2)


def test_v7_gt_car_mask_sentinels():
    d = dict(gt_boxes=np.zeros((7, 7), np.float32), gt_names=np.array(['Car', 'Car', 'Car', 'Car', 'Truck', 'Car', 'car']),
             gt_boxes_mask=np.array([1, 0, 1, 1, 1, 1, 1], bool), gt_boxes_ignore_mask=np.array([0, 0, 1, 0, 0, 0, 0], bool),
             gt_classes=np.array([1, 1, 1, -1, 1, common_utils.IGNORE_CLASS_LABEL, 1], np.int32))
    assert R.gt_car_mask(d).tolist() == [True, False, False, False, False, False, False]
    assert R.gt_car_mask(dict(gt_boxes=np.zeros((0, 7)), gt_names=np.array([]))).tolist() == []


def test_v7_scene_bottom_from_frame_gt_cars_not_copies_nor_sentinels(tmp_path):
    db = make_db(tmp_path, [(CAR_LOCAL, SIZE, [np.pi / 2])])
    gt = _v7_box(0.0, -52.0, -0.03 + 0.25, 0.0)
    bad = _v7_box(12.0, -40.0, -0.03 + 1.5, 0.0)                                    # a sentinel: never an E5a car
    d = scene_v2([], [gt, bad], ['Car', 'Car'], gt_classes=np.array([1, -1], np.int32), gt_scores=np.ones(2, np.float32))
    base = dict(VERSION=7, RETURN_OFF=True, RETURN_MODEL='physics', MAX_TRIES=3, PLANE_RMAX=75.2, N_COPIES=2)
    out, rec = run2(d, db, RngV2([40.0, 3 * np.pi / 2, 40.0, 3 * np.pi / 2 + 0.6]), **base)
    assert rec['version'] == 7 and rec['n_placed'] == 2, rec['reject']
    pl40 = rec['plane40']
    np.testing.assert_allclose(pl40, R.ground_plane(d['points'][:, :3], 40.0))
    for c in rec['copies']:
        bx = c['box']
        want = R.plane_z(pl40, bx[0], bx[1]) + (gt[2] - 0.75 - R.plane_z(pl40, gt[0], gt[1]))
        if np.hypot(bx[0] - gt[0], bx[1] - gt[1]) <= 30.0:
            assert c['ground_kind'] == 0 and np.isclose(c['bottom'], want, atol=1e-5)
    assert out['gt_classes'].tolist() == [1, -1, 1, 1]
    out, rec = run2(scene_v2(), db, RngV2([40.0, 1.0]), **dict(base, N_COPIES=1))       # no GT car: the cell fallback
    c = rec['copies'][0]
    assert c['ground_kind'] == 1 and c['ground_fallback'] and abs(c['bottom'] + 0.03) < 0.02


def test_v7_d1v5_rows_on_a_toy():
    import rr_gate as RG
    pts = ground(0.0)
    cars = [_v7_box(20.0, 0.0, -0.03), _v7_box(30.0, 5.0, -0.03), _v7_box(-35.0, 20.0, -0.03), _v7_box(-30.0, 0.0, -0.03)]
    pts = np.concatenate([pts] + [car_surface(b) for b in cars])
    B = frame(np.stack(cars), ['Car', 'Car', 'Car', 'Pedestrian'], pts)
    cfg = EasyDict(dict(R.DEFAULTS_V7, DONOR_CLASS='Car', NEAR_D_LO=0.5, GROUND_CELL=1.0, GROUND_CELL_MIN_PTS=3))
    pl40 = R.ground_plane(B['points'][:, :3], 40.0)
    rows, dropped = RG.d1v5_rows(B['points'], B, cfg, pl40, 0, 7)
    assert dropped == 0 and len(rows) == 3                                          # the pedestrian is not a car
    rw = np.array(rows)
    c = {k: j for j, k in enumerate(RG.D1V5_COLS)}
    assert rw[:, c['kind_loo']].tolist() == [0, 0, 1]                    # the third car has no other car within 30 m
    assert np.all(np.abs(rw[:, c['est_loo']] - rw[:, c['gt_bottom']]) < 0.05)
    assert np.all(np.abs(rw[:, c['est_frame']] - rw[:, c['gt_bottom']]) < 0.05)
    n, med, sgn, g05, g1, ks = RG.d1v5_summary(rw, 0.0, 75.2)
    assert n == 3 and med < 0.05 and g1 == 0.0 and np.isclose(sum(ks), 1.0)


def test_v7_g2_skips_copies_with_undefined_ppp():
    import rr_gate as RG
    assert RG.g2_cell_median([1.0, np.nan, 1.0, 1.05], skip_undefined=True) == (1.0, 1)
    med, n = RG.g2_cell_median([1.0, np.nan, 1.0], skip_undefined=False)
    assert np.isnan(med) and n == 0                                                 # v6's reading: a nan cell fails
    assert RG.g2_cell_median([], skip_undefined=True)[1] == 0


def test_v7_cnn_logit_calibration_bounds_and_no_weight_pickling(tmp_path):
    import torch
    from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
    torch.manual_seed(1)
    net = R._unet_class()()
    wp = tmp_path / 'unet_b2.pt'; torch.save(net.state_dict(), wp)
    par = dict(a=0.4, b_band=np.array([-0.5, 0.0, -1.0, -2.0]), c=-1.5, log_sigma=np.log(0.7))
    cp_ = tmp_path / 'cal_b2.npz'; np.savez(cp_, **{k: np.asarray(v, float) for k, v in par.items()})
    b, xyz, rows, cols, kept = _copy_render(30.0, 1.0, 0.0)
    sil = R.silhouette(b, TH, NC, H)
    env = R.envelope_visibility(xyz, rows, cols, sil, TH, NC, R.Occluders(np.zeros((0, 3)), np.zeros(3), H, NC))
    loc = np.c_[CAR_LOCAL, np.linspace(1, 9, len(CAR_LOCAL))]
    u = R.cnn_logit(loc, kept, xyz, rows, cols, b, sil, env, TH, NC, H, str(wp))
    pv6 = R.cnn_keep_prob(loc, kept, xyz, rows, cols, b, sil, env, TH, NC, H, str(wp))
    f = np.isfinite(u)
    assert f.any() and (~f).any() and np.all(pv6[~f] == 0)
    np.testing.assert_allclose(pv6[f], 1 / (1 + np.exp(-u[f])), rtol=1e-5)          # v6's p is the sigmoid of the logit
    hrel = np.clip((loc[kept, 2] + b[5] / 2) / b[5], 0, 1)
    q = R.cal_params(str(cp_))
    p = R.calibrated_prob(u, hrel, float(np.hypot(b[0], b[1])), q, eps=0.8)
    want = 1 / (1 + np.exp(-(u + 0.4 + par['b_band'][R.band_of(hrel)] - 1.5 * np.log(np.hypot(b[0], b[1]) / 50) + 0.7 * 0.8)))
    assert np.all((p[f] > 0) & (p[f] < 1)) and np.all(p[~f] == 0)
    np.testing.assert_allclose(p[f], want[f], rtol=1e-9)
    big = R.calibrated_prob(np.array([200.0, -200.0, -np.inf]), np.array([0.3, 0.3, 0.3]), 50.0, q)
    assert 0 < big[1] < big[0] < 1 and big[2] == 0
    db = make_db_v5(tmp_path, [(np.c_[CAR_LOCAL, np.random.default_rng(0).uniform(1, 50, len(CAR_LOCAL))], SIZE, [np.pi / 2])])
    base = dict(VERSION=7, RETURN_MODEL='cnn_cal', CNN_WEIGHTS=str(wp), RETURN_PARAMS=str(cp_), MAX_TRIES=3, PLANE_RMAX=75.2)
    out, rec = run2(scene_v2(), db, RngV2([40.0, 3 * np.pi / 2], arr=0.0), **base)
    c = rec['copies'][0]
    assert 0 < c['p_keep'] < 1 and 0 < c['n_kept'] <= c['n_visible'] and len(c['vis_ulogit']) == c['n_visible']
    out, rec = run2(scene_v2(), db, RngV2([40.0, 3 * np.pi / 2]), **dict(base, RETURN_OFF=True))
    c = rec['copies'][0]
    assert c['n_kept'] == c['n_visible'] and len(c['vis_ulogit']) == c['n_visible']   # the fitting copies record U(x)
    cfg = EasyDict(dict(P2, **base, NAME='relocate_render', DB_POINTS=db.points_path, DB_INDEX=db.index_path,
                        DB_NORMALS=db.normals_path))
    aug = DataAugmentor(None, EasyDict(AUG_CONFIG_LIST=[cfg], DISABLE_AUG_LIST=[]), ['Car'])
    s = pickle.dumps(aug)
    assert len(s) < 20000 and b'state_dict' not in s and R._load_cnn.cache_info().currsize >= 1


def test_v7_defaults_are_new_keys_only_so_v1_to_v6_are_unchanged():
    earlier = set(R.DEFAULTS) | set(R.DEFAULTS_V2) | set(R.DEFAULTS_V3) | set(R.DEFAULTS_V4) | set(R.DEFAULTS_V5) | \
        set(R.DEFAULTS_V6)
    assert not set(R.DEFAULTS_V7) & earlier
    p6 = R._params_v2(dict(VERSION=6, PLANE_RMAX=75.2, NEAR_RADIUS=10.0))
    assert p6['VERSION'] == 6 and p6['NEAR_RADIUS'] == 10.0 and p6['E5A_RADIUS'] == 30.0     # present, unread below v7


def test_v7_fit_arm_c_logit_and_the_auc_helpers():
    import torch
    import rr_v5_fit as FT
    m = FT.Model('c', 0)
    F = torch.tensor([[1.0, 0.3, 40.0, 1.0, 0.7, 40.0], [1.0, 0.9, 40.0, 1.0, -np.inf, 60.0]], dtype=torch.float64)
    lg = m.logit(F)
    assert float(lg[0]) == 0.7 and float(lg[1]) == -1e4                               # zero calibration = U; outside -> p 0
    with torch.no_grad():
        m.theta[:] = torch.tensor([0.5, -1.0, -2.0, -3.0, 2.0], dtype=torch.float64)
    lg = m.logit(F)
    assert np.isclose(float(lg[0]), 0.7 + 0.5 + 0.0 + 2.0 * np.log(40 / 50))          # band 1 (0.3) is the reference
    e = m.export()
    assert set(e) == {'a', 'b_band', 'c', 'log_sigma'} and e['b_band'][1] == 0.0 and e['b_band'][3] == -3.0
    X = np.zeros((2, 6, 2, 2)); X[:, 2] = 0.8; X[:, 5] = np.log(60 / 50)
    U = np.array([[[0.0, 1.0], [2.0, 3.0]], [[-1.0, 0.0], [1.0, 2.0]]])
    L = FT.cal_logits(U, X, dict(a=1.0, b_band=np.array([0, 0, 0, -0.5]), c=2.0))
    np.testing.assert_allclose(L, U + 1.0 - 0.5 + 2.0 * np.log(1.2))
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.5, 0.6]); y = np.array([0, 1, 0, 1, 1, 1.0]); pid = np.array([0, 0, 1, 1, 2, 2])
    pooled, within, nw = FT.auc_pooled_within(s, y, pid)
    assert nw == 2 and within == 1.0 and 0 < pooled <= 1


def test_rr7_configs_are_rr6a_plus_the_v7_keys_and_v1_to_v6_unchanged(monkeypatch):
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(ROOT / 'tools')
    load = lambda n: cfg_from_yaml_file('cfgs/da-ieee-access/' + n, EasyDict())
    r6 = load('centerpoint-gblobs-accum-legaldepth-rr6a-nuscenes2waymo.yaml')
    for arm in ('a', 'b'):
        r7 = load(f'centerpoint-gblobs-accum-legaldepth-rr7{arm}-nuscenes2waymo.yaml')
        for key in ('NUSCENES_N008', 'NUSCENES_N015'):
            e7, e6 = (c.DATA_CONFIGS[key].DATA_AUGMENTOR.AUG_CONFIG_LIST for c in (r7, r6))
            assert e7[1:] == e6[1:] and e7[0].VERSION == 7 and e6[0].VERSION == 6
            assert (e7[0].E5A_PLANE_RMAX, e7[0].E5A_RADIUS, e7[0].E5A_FB_RADIUS, e7[0].E5A_FB_PCT, e7[0].E5A_FB_MIN_CELLS,
                    e7[0].NEAR_D_LO, e7[0].PLANE_RMAX, e7[0].GROUND_CELL, e7[0].GROUND_CELL_MIN_PTS) == \
                (40.0, 30.0, 10.0, 5.0, 3, 0.5, 75.2, 1.0, 3)
            assert not any(k in e7[0] for k in ('NEAR_D_HI', 'NEAR_RADIUS', 'NEAR_MIN_POINTS', 'NEAR_PCT'))
            if arm == 'a':
                assert e7[0].RETURN_MODEL == 'physics' and e7[0].RETURN_PARAMS.endswith('v6/fit_a.npz') and 'CNN_WEIGHTS' not in e7[0]
            else:
                assert e7[0].RETURN_MODEL == 'cnn_cal' and e7[0].CNN_WEIGHTS.endswith('v6/unet_b2.pt') and \
                    e7[0].RETURN_PARAMS.endswith('v7/cal_b2.npz')
            drop = {'VERSION', 'RETURN_MODEL', 'RETURN_PARAMS', 'CNN_WEIGHTS', 'NEAR_D_HI', 'NEAR_RADIUS', 'NEAR_MIN_POINTS',
                    'NEAR_PCT'} | set(R.DEFAULTS_V7)
            assert {k: v for k, v in e7[0].items() if k not in drop} == {k: v for k, v in e6[0].items() if k not in drop}
        for k in ('MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'DATA_CONFIG_TAR'):
            assert r7[k] == r6[k]
    r7a, r7a2 = (load(f'centerpoint-gblobs-accum-legaldepth-rr7{x}-nuscenes2waymo.yaml') for x in ('a', 'a2'))
    for key in ('NUSCENES_N008', 'NUSCENES_N015'):                    # §12.2: the refit config differs only by its fit
        e, e2 = (c.DATA_CONFIGS[key].DATA_AUGMENTOR.AUG_CONFIG_LIST for c in (r7a, r7a2))
        assert e2[1:] == e[1:] and e2[0].RETURN_PARAMS.endswith('v7/fit_a.npz')
        assert {k: v for k, v in e2[0].items() if k != 'RETURN_PARAMS'} == {k: v for k, v in e[0].items() if k != 'RETURN_PARAMS'}
    assert all(r7a2[k] == r7a[k] for k in ('MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'DATA_CONFIG_TAR'))
