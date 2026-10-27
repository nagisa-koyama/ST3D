"""tools/analysis/nw_strat.py (NW-STRAT, experiments_md 20261027_03; design 20261003_05 §5 item 23, FIXED 2026-10-27
21:15 JST). ANALYSIS. CPU only.

Pinned here:
  - M1's subset convention (20261007_03 §3): a detection matched to a don't-care GT is dropped, an unmatched one stays a
    false positive, the GT count excludes the don't-care set;
  - the fixed categories at their boundaries (50 m, 50 points, 6 m, u 0.5 / 0.9; NaN u is in no u category);
  - M2's cells, eligibility (>= 100 GT per compared group) and both aggregate forms, including a zero rate;
  - the readings at each fixed bar;
  - the official evaluator's don't-care ('Van') and 50 m cap equal removing the GT and the detections on them;
  - S1: the three pitch arms (source slope minus Waymo's), the twins differ from 27704's config only in
    DATA_CONFIG_TAR.EVAL_OSENS2, and a ground rising 0.11 deg ahead, pitched by an arm, rises by the source's slope as the
    hook's own estimator fits it (the sign check).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / 'tools'
for _p in (str(TOOLS), str(TOOLS / 'analysis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nw_strat as N  # noqa: E402
import waymo_ranking_loss_fp_taxonomy as T  # noqa: E402


def car(x, y, l=4.0, w=2.0, h=1.5, z=0.0, yaw=0.0):
    return [x, y, z, l, w, h, yaw]


# ------------------------------------------------------------------------------------------------------------- M1
def test_dc_ap_drops_matched_dontcare_keeps_unmatched_fp():
    score = np.array([0.9, 0.8, 0.7, 0.6])
    gt_idx = np.array([0, 1, -1, 2])          # d1 is matched to the don't-care GT 1; d2 is unmatched
    dc = np.array([False, True, False])
    got = N.dc_ap(score, gt_idx, dc)
    want = T.ap_stats(np.array([0.9, 0.7, 0.6]), np.array([True, False, True]), 2)[0]
    assert got == pytest.approx(want)
    assert N.dc_ap(score, gt_idx, np.zeros(3, bool)) == pytest.approx(
        T.ap_stats(score, np.array([True, True, False, True]), 3)[0])
    # all GT don't-care: no GT left -> NaN (the cell does not exist)
    assert np.isnan(N.dc_ap(score, gt_idx, np.ones(3, bool)))


def test_categories_at_their_boundaries():
    gt = dict(rng=np.array([49.99, 50.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
              npts=np.array([500, 500, 49, 50, 500, 500, 500]),
              length=np.array([4.0, 4.0, 4.0, 4.0, 6.0, 5.99, 4.0]))
    u = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.49, np.nan])
    c = N.categories(gt, u)
    assert c['far >= 50 m'].tolist() == [False, True, False, False, False, False, False]
    assert c['sparse < 50 points'].tolist() == [False, False, True, False, False, False, False]
    assert c['long >= 6 m'].tolist() == [False, False, False, False, True, False, False]
    assert c['occluded u < 0.5'].tolist() == [False, False, False, False, False, True, False]
    assert c['joint'].tolist() == [False, True, True, False, True, True, False]
    u2 = np.array([0.5, 0.89, 0.9, np.nan, 0.0, 1.0, 0.2])
    c2 = N.categories(gt, u2)
    assert c2['occluded u < 0.5'].tolist() == [False, False, False, False, True, False, True]
    assert c2['occluded u < 0.9 (beside)'].tolist() == [True, True, False, False, True, False, True]


# ------------------------------------------------------------------------------------------------------------- M2
def _cell_gt(n_occl, n_vis, miss_occl, miss_vis, rng=10.0, npts=300):
    n = n_occl + n_vis
    gt = dict(rng=np.full(n, rng), npts=np.full(n, npts), length=np.full(n, 4.0))
    u = np.r_[np.full(n_occl, 0.3), np.full(n_vis, 0.95)]
    oracle = np.ones(n, bool)
    model = np.ones(n, bool)
    model[:miss_occl] = False
    model[n_occl:n_occl + miss_vis] = False
    return gt, u, oracle, model


def _cat(*parts):
    gt = {k: np.concatenate([p[0][k] for p in parts]) for k in ('rng', 'npts', 'length')}
    return gt, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]), \
        np.concatenate([p[3] for p in parts])


def test_m2_cells_eligibility_and_aggregates():
    a = _cell_gt(100, 100, 20, 10, rng=10.0, npts=300)      # 0-30 m, >= 200: ratio 2.0
    b = _cell_gt(99, 300, 50, 10, rng=40.0, npts=300)       # 30-50 m, >= 200: only 99 occluded -> not eligible
    c = _cell_gt(200, 100, 30, 10, rng=60.0, npts=20)       # 50-75 m, 10-49: ratio 1.5
    gt, u, o, m = _cat(a, b, c)
    cells = N.m2_cells(gt, u, o, m)
    assert len(cells) == 15
    by = {(x['range'], x['points']): x for x in cells}
    ca, cb, cc = by[('0-30', '>=200')], by[('30-50', '>=200')], by[('50-75', '10-49')]
    assert ca['eligible'] and ca['ratio'] == pytest.approx(2.0) and ca['n_occl'] == 100 and ca['miss_occl'] == 20
    assert not cb['eligible'] and np.isnan(cb['ratio'])
    assert cc['eligible'] and cc['ratio'] == pytest.approx(1.5)
    agg = N.m2_aggregate(cells)
    assert agg['n_eligible'] == 2
    # (a) weighted mean of ratios, weights 200 and 300; (b) ratio of the weighted rates
    assert agg['w_groups']['mean_of_ratios'] == pytest.approx((200 * 2.0 + 300 * 1.5) / 500)
    assert agg['w_groups']['ratio_of_rates'] == pytest.approx((200 * 0.2 + 300 * 0.15) / (200 * 0.1 + 300 * 0.1))
    # the oracle-only miss: a GT the oracle also misses is not counted
    o2 = o.copy(); o2[:20] = False
    assert N.m2_cells(gt, u, o2, m)[4]['miss_occl'] == 0


def test_m2_zero_visible_rate_is_reported_not_divided():
    a = _cell_gt(100, 100, 5, 0)
    gt, u, o, m = _cat(a)
    cells = N.m2_cells(gt, u, o, m)
    c = [x for x in cells if x['eligible']][0]
    assert np.isinf(c['ratio'])
    agg = N.m2_aggregate(cells)
    assert agg['w_groups']['n_cells_ratio_undefined'] == 1 and np.isnan(agg['w_groups']['mean_of_ratios'])
    assert np.isnan(agg['w_groups']['ratio_of_rates'])            # no visible miss anywhere: undefined, not infinite


# -------------------------------------------------------------------------------------------------------- readings
def test_readings_at_the_bars():
    r = N.readings(10.0, 2.0, [1.5] * 4, 8.0)
    assert r['factor'] and not r['proxy'] and not r['dominant'] and r['range_carries_2']
    r = N.readings(10.0, 1.99, [1.5] * 4, 8.01)
    assert not r['factor'] and not r['range_carries_2']
    r = N.readings(10.0, 5.0, [1.2] * 4, 9.0)
    assert r['proxy'] and r['dominant'] and not r['factor']
    r = N.readings(10.0, 3.0, [1.6, 1.4, 1.6, 1.6], 9.0)       # the forms disagree about the 1.5 bar
    assert r['ratio_forms_split'] and not r['factor'] and not r['proxy']
    r = N.readings(10.0, 3.0, [1.3] * 4, 9.0)                   # between the bars: neither
    assert not r['factor'] and not r['proxy'] and not r['ratio_forms_split']


# -------------------------------------------------------------------------------------------------------- official
def _frames():
    f0 = dict(gv=np.array([car(10, 0), car(20, 5), car(55, 0)], np.float32), go=np.zeros((0, 7), np.float32))
    f1 = dict(gv=np.array([car(15, 3), car(30, -10)], np.float32), go=np.zeros((0, 7), np.float32))
    d0 = (np.array([car(10, 0), car(20.2, 5), car(30, -5), car(55, 0)], np.float32), np.array([0.9, 0.8, 0.5, 0.85]))
    d1 = (np.array([car(15, 3), car(31, -10)], np.float32), np.array([0.7, 0.6]))
    return [f0, f1], [d0, d1]


def test_official_dontcare_and_cap_equal_removal():
    frames, dets = _frames()
    dc = np.array([False, True, False, False, False])          # f0's second GT
    got = N.official_ap(frames, dets, dc=dc)
    rm_frames = [dict(gv=frames[0]['gv'][[0, 2]], go=frames[0]['go']), frames[1]]
    rm_dets = [(dets[0][0][[0, 2, 3]], dets[0][1][[0, 2, 3]]), dets[1]]
    want = N.official_ap(rm_frames, rm_dets)
    assert got == pytest.approx(want)
    assert got != pytest.approx(N.official_ap(frames, dets))  # the don't-care changes the AP here
    cap = N.official_ap(frames, dets, cap=True)
    cap_frames = [dict(gv=frames[0]['gv'][:2], go=frames[0]['go']), frames[1]]
    cap_dets = [(dets[0][0][:3], dets[0][1][:3]), dets[1]]
    assert cap == pytest.approx(N.official_ap(cap_frames, cap_dets))


# -------------------------------------------------------------------------------------------------------------- S1
def test_pitch_arms_are_source_slope_minus_waymo():
    arms = N.pitch_arms()
    assert arms == {'p218': 2.18, 'p124': 1.24, 'p176': 1.76}
    mean = (15695 * 2.29 + 12435 * 1.35) / 28130
    assert round(mean, 2) == 1.87


@pytest.mark.parametrize('tag', ['p218', 'p124', 'p176'])
def test_pitch_twins_differ_from_27704_only_in_eval_osens2(tag, monkeypatch):
    monkeypatch.chdir(TOOLS)
    from analysis.make_sensitivity_configs import resolved
    base = resolved(Path(N.BASE_CFG))
    twin = resolved(N.twin_path(tag))
    e = twin['DATA_CONFIG_TAR'].pop('EVAL_OSENS2')
    assert twin.pop('_BASE_CONFIG_') == N.BASE_CFG                      # the twin's own base: 27704's config
    base.pop('_BASE_CONFIG_')
    assert twin == base
    assert 'EVAL_OSENS2' not in base['DATA_CONFIG_TAR']
    assert e == {'PITCH_DEG': N.pitch_arms()[tag], 'SENSOR_ORIGIN': [1.43, 0.0, 2.184], 'FORWARD_AXIS': 'x'}
    assert base['DATA_CONFIG_TAR']['SHIFT_COOR'] == [0.0, 0.0, 0.0]        # the sensor origin needs no shift


@pytest.mark.parametrize('tag', ['p218', 'p124', 'p176'])
def test_pitch_sign_maps_waymo_ground_onto_the_source_slope(tag):
    from pcdet.datasets import osens2_utils as O
    deg = N.pitch_arms()[tag]
    rng = np.random.default_rng(0)
    xy = rng.uniform(-42, 42, size=(120000, 2))
    z = np.tan(np.radians(N.SLOPE['waymo'])) * xy[:, 0] + rng.normal(0, 0.02, len(xy))
    pts = np.c_[xy, z].astype(np.float32)
    assert N._forward_slope_deg(pts) == pytest.approx(N.SLOPE['waymo'], abs=0.02)
    out = O.pitch_points(pts, deg, [1.43, 0.0, 2.184], 'x')
    assert N._forward_slope_deg(out) == pytest.approx(N.SLOPE['waymo'] + deg, abs=0.02)
    src = {'p218': N.SLOPE['boston'], 'p124': N.SLOPE['singapore'], 'p176': 1.8745}[tag]
    assert N._forward_slope_deg(out) == pytest.approx(src, abs=0.03)
