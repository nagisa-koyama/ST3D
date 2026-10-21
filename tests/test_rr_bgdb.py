"""tools/analysis/rr_build_bgdb.py (CPU; experiments_md 20261018_02 §13.1): the background region sampler on a toy with a flat
ground, an annotated car, a bush (clutter above the clearance) and an empty road."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import rr_build_bgdb as BG  # noqa: E402
from pcdet.datasets.augmentor import relocate_render as R  # noqa: E402


def toy(seed=0):
    rng = np.random.default_rng(seed)
    g = rng.uniform(-40, 40, (120000, 2)); ground = np.c_[g, rng.normal(0.0, 0.01, len(g))]
    bush = np.c_[rng.uniform(14, 18, 4000), rng.uniform(-2, 2, 4000), rng.uniform(0.0, 1.6, 4000)]   # clutter at (16, 0)
    car_box = np.array([0.0, 16.0, 0.75, 4.5, 1.9, 1.5, 0.0])                                       # annotated, at (0, 16)
    car = np.c_[rng.uniform(-2.2, 2.2, 3000), rng.uniform(14.1, 17.9, 3000), rng.uniform(0.0, 1.5, 3000)]
    P = np.concatenate([ground, bush, car])
    return P, car_box


def test_regions_avoid_grown_boxes_hold_200_points_above_the_clearance_and_drop_the_ground():
    P, car_box = toy()
    plane = R.ground_plane(P, 40.0)
    cells = R.GroundCells(P, 1.0, 3)
    sizes = np.array([[4.0, 1.8, 1.5], [4.6, 2.0, 1.6]])
    got = []
    for seed in range(40):
        regs, tries = BG.sample_regions(P, car_box[None], sizes, np.random.default_rng(seed), plane, cells)
        assert tries <= BG.MAX_TRIES and len(regs) <= BG.MAX_ACCEPT
        for k, (box, keep) in enumerate(regs):
            assert 5.0 <= np.hypot(box[0], box[1]) <= 30.0
            assert tuple(np.round(box[3:6], 3)) in {tuple(s) for s in np.round(sizes, 3)}
            assert not R.rects_overlap(R.bev_corners(box[None])[0], R.bev_corners(car_box[None], 0.5)).any()
            for other, _ in regs[:k]:
                assert not R.rects_overlap(R.bev_corners(box[None])[0], R.bev_corners(other[None])).any()
            zb = box[2] - box[5] / 2
            assert len(keep) >= BG.MIN_PTS and np.all(P[keep, 2] > zb + BG.CLEAR)
            assert np.all(R.points_in_box(P[keep], box))
            got.append(box)
    assert got, 'the bush must yield regions'
    # every accepted region holds bush points: the empty road never reaches 200 points above 0.3 m
    for box in got:
        c = np.hypot(box[0] - 16.0, box[1]) < 5.0
        assert c


def test_region_bottom_is_the_local_ground_and_no_plane_frames_are_handled():
    P, car_box = toy()
    plane = R.ground_plane(P, 40.0)
    cells = R.GroundCells(P, 1.0, 3)
    regs, _ = BG.sample_regions(P, np.zeros((0, 7)), np.array([[4.0, 1.8, 1.5]]), np.random.default_rng(3), plane, cells)
    for box, keep in regs:
        zb = box[2] - box[5] / 2
        assert np.isclose(zb, R.local_bottom(cells, box[0], box[1], plane, 5.0, 3, 20.0)[0])
