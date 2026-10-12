"""EVAL_TOP_THIN.REGION (ANALYSIS, evaluation only; experiments_md 20261010_04 §5.0a): the TOP block thinned only inside,
or only outside, given boxes. Sentinels: no boxes, boxes holding no point, a point on a box edge, ignore labels never
used as boxes, count matching per region. Uses one real processed Waymo frame; skipped if absent."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from pcdet.datasets.waymo.waymo_rings import eval_top_thin_points, top_beam_ids
from pcdet.datasets.waymo.waymo_region_thin import (enlarge_boxes, points_in_any_box, region_boxes,
                                                    region_top_thin_points)

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'waymo'
SEQ = 'segment-10017090168044687777_6380_000_6400_000_with_camera_labels'
CALIB = Path('/home/koyama/data/waymo_top_calib/top_calib.pkl')
FRAME = ROOT / 'waymo_processed_data' / SEQ / '0000.npy'
needs_frame = pytest.mark.skipif(not (FRAME.exists() and CALIB.exists()), reason='Waymo frame or TOP calibration absent')


@pytest.fixture(scope='module')
def frame():
    raw = np.load(FRAME)
    infos = pickle.load(open(ROOT / 'waymo_processed_data' / SEQ / (SEQ + '.pkl'), 'rb'))
    counts = infos[0]['num_points_of_each_lidar']
    calib = pickle.load(open(CALIB, 'rb'))[SEQ]
    nlz_free = raw.copy()
    nlz_free[:, 5] = -1           # every point survives the no-label-zone filter, so counts are exact
    return nlz_free, counts, calib


def _box_around_top_points(raw, counts, rng_m=(12.0, 18.0)):
    """A 5 x 3 x 3 m box centred on a TOP point at 12-18 m: it holds TOP points from several beams."""
    top = raw[:int(counts[0])]
    r = np.hypot(top[:, 0], top[:, 1])
    c = top[np.nonzero((r > rng_m[0]) & (r < rng_m[1]))[0][0], :3]
    return np.array([[c[0], c[1], c[2], 5.0, 3.0, 3.0, 0.3]])


# --- geometry and box-file sentinels (no data needed) ---

def test_point_on_a_box_edge_is_inside_and_just_beyond_is_not():
    box = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.0, 0.0]])
    pts = np.array([[2.0, 0.0, 0.0], [2.0 + 1e-6, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 1.0, 0.5 + 1e-6]])
    assert points_in_any_box(pts, box).tolist() == [True, False, True, False]


def test_margin_grows_every_side_and_keeps_centre_and_heading():
    box = np.array([[1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.7]])
    big = enlarge_boxes(box, 0.5, 0.15)
    assert np.allclose(big[0, :3], box[0, :3]) and big[0, 6] == box[0, 6]
    assert np.allclose(big[0, 3:6], [5.0, 3.0, 1.8])


def test_empty_and_missing_entries_give_no_boxes():
    assert region_boxes(None).shape == (0, 7)
    assert region_boxes({'boxes': np.zeros((0, 7)), 'labels': np.zeros(0)}).shape == (0, 7)


def test_ignore_and_background_labels_never_define_a_region():
    boxes = np.tile([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], (4, 1))
    entry = {'boxes': boxes, 'labels': np.array([-1, -99, 0, 1])}
    assert len(region_boxes(entry)) == 1
    assert len(region_boxes({'boxes': boxes[:3], 'labels': np.array([-1, -99, 0])})) == 0


def test_extra_box_columns_are_dropped():
    entry = {'boxes': np.zeros((2, 9)), 'labels': np.array([1, 1])}
    assert region_boxes(entry).shape == (2, 7)


# --- the thinning, on a real frame ---

@needs_frame
def test_no_boxes_inside_is_untouched_and_outside_is_the_global_cut(frame):
    raw, counts, calib = frame
    for entry in (None, {'boxes': np.zeros((0, 7)), 'labels': np.zeros(0)}):
        assert len(region_top_thin_points(raw, counts, calib, 'rows', entry, 'inside')) == len(raw)
        out = region_top_thin_points(raw, counts, calib, 'rows', entry, 'outside')
        ref = eval_top_thin_points(raw, counts, calib, 'rows')
        assert np.array_equal(out, ref)


@needs_frame
def test_boxes_holding_no_point_change_nothing_inside(frame):
    raw, counts, calib = frame
    far = {'boxes': np.array([[1000.0, 1000.0, 0.0, 4.0, 2.0, 1.5, 0.0]]), 'labels': np.array([1])}
    assert len(region_top_thin_points(raw, counts, calib, 'rows', far, 'inside')) == len(raw)
    assert len(region_top_thin_points(raw, counts, calib, 'random', far, 'inside', rng=np.random.default_rng(1))) == len(raw)


@needs_frame
def test_ignore_labelled_boxes_do_not_thin(frame):
    raw, counts, calib = frame
    box = _box_around_top_points(raw, counts)
    entry = {'boxes': np.repeat(box, 2, axis=0), 'labels': np.array([-1, -99])}
    assert len(region_top_thin_points(raw, counts, calib, 'rows', entry, 'inside')) == len(raw)


@needs_frame
def test_rows_inside_drops_odd_beams_only_inside(frame):
    raw, counts, calib = frame
    n_top = int(counts[0])
    box = _box_around_top_points(raw, counts)
    entry = {'boxes': box, 'labels': np.array([1])}
    beam = top_beam_ids(raw[:n_top, :3], calib['extrinsic'], calib['inclinations'])[0]
    inside = points_in_any_box(raw[:n_top, :3], enlarge_boxes(box, 0.5, 0.15))
    assert inside.sum() > 50 and len(np.unique(beam[inside])) >= 4
    out = region_top_thin_points(raw, counts, calib, 'rows', entry, 'inside')
    assert len(out) == len(raw) - int((inside & (beam % 2 == 1)).sum())
    out = region_top_thin_points(raw, counts, calib, 'rows', entry, 'outside')
    assert len(out) == len(raw) - int((~inside & (beam % 2 == 1)).sum())


@needs_frame
@pytest.mark.parametrize('side', ['inside', 'outside'])
def test_random_is_count_matched_to_rows_in_its_region_and_spares_the_other(frame, side):
    raw, counts, calib = frame
    n_top = int(counts[0])
    box = _box_around_top_points(raw, counts)
    entry = {'boxes': box, 'labels': np.array([1])}
    rows = region_top_thin_points(raw, counts, calib, 'rows', entry, side)
    rnd = region_top_thin_points(raw, counts, calib, 'random', entry, side, rng=np.random.default_rng(7))
    assert len(rnd) == len(rows)
    # side lidars are all kept and stored after TOP (no NLZ filtering here), so the output's TOP block is its head
    n_side = len(raw) - n_top
    big = enlarge_boxes(box, 0.5, 0.15)
    before = points_in_any_box(raw[:n_top, :3], big)
    for out in (rows, rnd):
        assert np.array_equal(out[len(out) - n_side:, :3], raw[n_top:, :3])
        after = points_in_any_box(out[:len(out) - n_side, :3], big)
        if side == 'inside':      # the outside keeps every TOP point; the inside is thinned
            assert (~after).sum() == (~before).sum() and after.sum() < before.sum()
        else:
            assert after.sum() == before.sum() and (~after).sum() < (~before).sum()


@needs_frame
def test_random_is_deterministic_per_seed(frame):
    raw, counts, calib = frame
    entry = {'boxes': _box_around_top_points(raw, counts), 'labels': np.array([1])}
    a = region_top_thin_points(raw, counts, calib, 'random', entry, 'inside', rng=np.random.default_rng(3))
    b = region_top_thin_points(raw, counts, calib, 'random', entry, 'inside', rng=np.random.default_rng(3))
    assert np.array_equal(a, b)


@needs_frame
def test_bad_side_or_mode_is_refused(frame):
    raw, counts, calib = frame
    with pytest.raises(ValueError):
        region_top_thin_points(raw, counts, calib, 'rows', None, 'both')
    with pytest.raises(ValueError):
        region_top_thin_points(raw, counts, calib, 'cols', None, 'inside')
