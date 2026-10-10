"""EVAL_TOP_THIN (ANALYSIS, evaluation only; experiments_md 20261009_02): the TOP block of an evaluation frame thinned
by whole scan lines, or by a count-matched random draw. Uses one real processed Waymo frame; skipped if absent."""
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pcdet.datasets.waymo.waymo_rings import eval_top_thin_points, top_beam_ids

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'waymo'
SEQ = 'segment-10017090168044687777_6380_000_6400_000_with_camera_labels'
CALIB = Path('/home/koyama/data/waymo_top_calib/top_calib.pkl')
FRAME = ROOT / 'waymo_processed_data' / SEQ / '0000.npy'
pytestmark = pytest.mark.skipif(not (FRAME.exists() and CALIB.exists()), reason='Waymo frame or TOP calibration absent')


@pytest.fixture(scope='module')
def frame():
    raw = np.load(FRAME)
    infos = pickle.load(open(ROOT / 'waymo_processed_data' / SEQ / (SEQ + '.pkl'), 'rb'))
    counts = infos[0]['num_points_of_each_lidar']
    calib = pickle.load(open(CALIB, 'rb'))[SEQ]
    nlz_free = raw.copy()
    nlz_free[:, 5] = -1           # every point survives the no-label-zone filter, so counts are exact
    return raw, nlz_free, counts, calib


def test_rows_keep_only_even_beams_and_about_half_of_top(frame):
    _, raw, counts, calib = frame
    out = eval_top_thin_points(raw, counts, calib, 'rows', stride=2)
    n_top = counts[0]
    beam = top_beam_ids(raw[:n_top, :3], calib['extrinsic'], calib['inclinations'])[0]
    kept_top = len(out) - (len(raw) - n_top)
    assert kept_top == int((beam % 2 == 0).sum())
    assert 0.4 < kept_top / n_top < 0.6
    # side lidars identical, in order, after the TOP block
    assert np.array_equal(out[kept_top:, :3], raw[n_top:, :3])


def test_random_matches_the_row_count_and_spares_every_row(frame):
    _, raw, counts, calib = frame
    rows = eval_top_thin_points(raw, counts, calib, 'rows', stride=2)
    rnd = eval_top_thin_points(raw, counts, calib, 'random', stride=2, rng=np.random.default_rng(1))
    assert len(rnd) == len(rows)
    n_top = counts[0]
    beam = top_beam_ids(raw[:n_top, :3], calib['extrinsic'], calib['inclinations'])[0]
    kept = np.isin(raw[:n_top, :3].view([('', raw.dtype)] * 3).ravel(),
                   rnd[:len(rnd) - (len(raw) - n_top), :3].astype(raw.dtype).view([('', raw.dtype)] * 3).ravel())
    assert len(np.unique(beam[kept])) == len(np.unique(beam))      # no scan line lost


def test_random_is_deterministic_per_seed(frame):
    _, raw, counts, calib = frame
    a = eval_top_thin_points(raw, counts, calib, 'random', rng=np.random.default_rng(7))
    b = eval_top_thin_points(raw, counts, calib, 'random', rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_nlz_filter_and_intensity_follow_get_lidar(frame):
    raw, _, counts, calib = frame
    out = eval_top_thin_points(raw, counts, calib, 'rows')
    assert out.shape[1] == 5 and np.all(np.abs(out[:, 3]) <= 1.0)
    assert len(out) <= int((raw[:, 5] == -1).sum())


def test_unknown_mode_is_refused(frame):
    _, raw, counts, calib = frame
    with pytest.raises(ValueError):
        eval_top_thin_points(raw, counts, calib, 'beams')


def test_cols_keep_even_columns_and_random_cols_matches_their_count(frame):
    _, raw, counts, calib = frame
    n_top = counts[0]
    _, col, _, _, _ = top_beam_ids(raw[:n_top, :3], calib['extrinsic'], calib['inclinations'])
    cols = eval_top_thin_points(raw, counts, calib, 'cols', stride=2)
    rcol = eval_top_thin_points(raw, counts, calib, 'random_cols', stride=2, rng=np.random.default_rng(3))
    kept_top = len(cols) - (len(raw) - n_top)
    assert kept_top == int((np.floor(col).astype(np.int64) % 2 == 0).sum()) and 0.4 < kept_top / n_top < 0.6
    assert len(rcol) == len(cols)


def test_ring_thin_mask_on_synthetic_rings():
    from pcdet.utils.beam_downsample_utils import ring_thin_mask
    n_ring, n_az = 32, 100
    ring = np.repeat(np.arange(n_ring), n_az)
    az = np.tile(np.linspace(-np.pi, np.pi, n_az, endpoint=False), n_ring)
    pts = np.stack([10 * np.cos(az), 10 * np.sin(az), np.zeros_like(az)], 1)
    rows = ring_thin_mask(pts, ring, 'rows'); cols = ring_thin_mask(pts, ring, 'cols')
    assert set(np.unique(ring[rows])) == set(range(0, 32, 2)) and rows.sum() == 16 * n_az
    assert len(np.unique(ring[cols])) == 32 and cols.sum() == 32 * n_az // 2
    r = ring_thin_mask(pts, ring, 'random', rng=np.random.default_rng(0))
    rc = ring_thin_mask(pts, ring, 'random_cols', rng=np.random.default_rng(0))
    assert r.sum() == rows.sum() and rc.sum() == cols.sum()
    assert len(np.unique(ring[r])) == 32                       # the random control keeps every ring
    with pytest.raises(ValueError):
        ring_thin_mask(pts, ring, 'beams')


def test_azbin_keeps_one_point_per_line_per_bin(frame):
    _, raw, counts, calib = frame
    n_top = counts[0]
    beam, col, _, _, _ = top_beam_ids(raw[:n_top, :3], calib['extrinsic'], calib['inclinations'])
    out = eval_top_thin_points(raw, counts, calib, 'azbin', az_res_deg=0.332)
    kept = len(out) - (len(raw) - n_top)
    key = beam.astype(np.int64) * 100000 + np.floor(col * (360.0 / 2650) / 0.332).astype(np.int64)
    assert kept == len(np.unique(key))
    assert 0.3 < kept / n_top < 0.5                      # 0.136 -> 0.332 deg keeps ~1 / 2.44 of each line
    rnd = eval_top_thin_points(raw, counts, calib, 'random_azbin', az_res_deg=0.332, rng=np.random.default_rng(5))
    assert len(rnd) == len(out)
