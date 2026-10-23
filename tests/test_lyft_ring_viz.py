"""tools/analysis/lyft_ring_viz.py (CPU; experiments_md 20261022_05 §4.4 and §6).

Pins what the 2026-10-23 alignment of the two Lyft platforms added, without touching the gate replay:
- `lattice` builds RING_PATTERN's lines with lr.ring_pattern_mask's own arithmetic, so its nearest rings ARE the rings
  ring_pattern_mask keeps (random levels and phases);
- `pattern_lines` names why the recovered and the joined rings keep different rings: a fragment outside the joined
  range adds lines (widened), a fragment nearest a line takes it (fragment);
- `what_differs` writes the 40-beam table's cells as they stand in the report (one wording excepted, see below);
- `f4_wrong_frames` / `draws_lowpurity`: without --align_platforms the selection is the one on record (all 40-beam,
  six 64-beam frames, p10 frame on 64-beam only); with it, every wrong-count frame and the p10 frame on both.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import lyft_ring_viz as V  # noqa: E402
from pcdet.datasets.lyft import lyft_rings as lr  # noqa: E402


def _scan(levels, n_az=720):
    """One point per ring per 0.5 deg of azimuth at 20 m, ring IDs in the given order."""
    az = np.radians(np.arange(n_az) * 360.0 / n_az - 180.0 + 0.25)
    xyz, ring = [], []
    for k, e in enumerate(levels):
        el = np.radians(e)
        xyz.append(np.c_[20 * np.cos(el) * np.cos(az), 20 * np.cos(el) * np.sin(az), 20 * np.sin(el) * np.ones(n_az)])
        ring.append(np.full(n_az, k))
    return np.concatenate(xyz), np.concatenate(ring)


@pytest.mark.parametrize('seed', range(6))
def test_lattice_nearest_rings_are_the_rings_ring_pattern_mask_keeps(seed):
    rng = np.random.default_rng(seed)
    lev = np.sort(rng.uniform(-25.0, 15.0, size=rng.integers(20, 65)))[::-1]
    pv, ph = float(rng.random()), float(rng.random())
    xyz, ring = _scan(lev)
    _, kept = lr.ring_pattern_mask(xyz, ring, lev, V.SPACING, V.AZ_RES, pv, ph)
    j, t, near = V.lattice(lev, pv)
    assert sorted(set(near.tolist())) == kept.tolist()
    assert np.allclose(t, (j + pv) * V.SPACING) and t.min() >= lev.min() and t.max() <= lev.max()


def test_pattern_lines_widened_by_a_low_fragment_and_a_line_taken_by_a_fragment():
    # joined rings 0, -1, ..., -10 deg; recovered = the same plus a fragment at -2.66 (joins ring 2) and one at -14
    # (joins ring 10). Lattice phase 0: lines at -1.33 * j.
    lev_j = -np.arange(11.0)
    lev = np.r_[0.0, -1.0, -2.0, -2.66, -np.arange(3.0, 11.0), -14.0]
    cm = np.r_[0, 1, 2, 2, np.arange(3, 11), 10]
    lines, kept_r, kept_j = V.pattern_lines(lev, lev_j, cm, [3, 12], 0.0)
    kind = {ln['line']: ln for ln in lines}
    assert kind[-2]['kind'] == 'fragment' and kind[-2]['recovered_ring'] == 3 and kind[-2]['joined_ring'] == 3
    assert [kind[j]['kind'] for j in (-8, -9, -10)] == ['widened'] * 3
    assert kind[-10]['recovered_is_fragment'] and kind[-10]['recovered_ring'] == 12
    assert not kind[-9]['recovered_is_fragment'] and kind[-9]['recovered_joined_id'] == 10
    assert all(kind[j]['kind'] == 'same' for j in (0, -1, -3, -4, -5, -6, -7))
    for levels, kept in ((lev, kept_r), (lev_j, kept_j)):
        xyz, ring = _scan(levels)
        assert lr.ring_pattern_mask(xyz, ring, levels, V.SPACING, V.AZ_RES, 0.0, 0.5)[1].tolist() == kept


def _diag(kept_full_r, kept_frag_r, kept_j, frags, same=False):
    """kept_full_r: joined IDs of full rings kept with the recovered rings; kept_frag_r: (fragment ring, joined ID,
    points kept); kept_j: joined IDs kept after joining; frags: (fragment ring, points, joined ID)."""
    kr = [{'ring': 100 + i, 'joined_id': i, 'fragment': False, 'points_kept': 1000} for i in kept_full_r]
    kr += [{'ring': r, 'joined_id': i, 'fragment': True, 'points_kept': k} for r, i, k in kept_frag_r]
    return {'kept_same': same, 'kept_recovered': kr,
            'kept_after_join': [{'joined_id': i, 'points_kept': 1000} for i in kept_j],
            'fragments': [{'ring': r, 'points': n, 'azimuth_deg': -0.04, 'joins_joined_id': i} for r, n, i in frags]}


BASE = [1, 2, 3, 5, 8, 12, 16, 20, 24, 28, 30, 31, 33, 34]


@pytest.mark.parametrize('d, text', [
    (_diag(BASE, [], BASE, [(31, 2, 30)], same=True), 'same rings'),
    # host-a011_lidar1_1232732480401735046
    (_diag(BASE + [39], [(41, 39, 3)], BASE, [(41, 5, 39)]), 'recovered also keeps ring 39 and its fragment (3 pt)'),
    # host-a004_lidar1_1233444839801385486
    (_diag(BASE, [(39, 38, 1)], BASE + [36], [(39, 1, 38)]),
     'ring 36 kept only after joining; recovered keeps a 1-point fragment of 38'),
    # host-a011_lidar1_1232732474001183166 (the 40-beam example)
    (_diag(BASE + [39], [(39, 38, 1), (41, 39, 3)], BASE + [36], [(39, 1, 38), (41, 4, 39)]),
     'recovered also keeps ring 39; ring 36 kept only after joining; recovered keeps fragments of 38 (1 pt) and 39 (3 pt)'),
    # host-a004_lidar1_1233014860701341366
    (_diag(BASE + [39], [(39, 38, 1), (40, 39, 1)], BASE + [37, 38], [(39, 1, 38), (40, 1, 39), (42, 4, 39)]),
     'recovered also keeps ring 39; rings 37, 38 kept only after joining; recovered keeps fragments of 38 and 39 '
     '(1 pt each)'),
    # host-a004_lidar1_1233947127000911366
    (_diag(BASE + [39], [(38, 36, 1), (40, 37, 1), (44, 39, 2)], BASE + [35, 37],
           [(30, 2, 29), (38, 1, 36), (40, 1, 37), (42, 1, 38), (44, 4, 39)]),
     'recovered also keeps ring 39; rings 35, 37 kept only after joining; recovered keeps fragments of 36, 37 (1 pt each) '
     'and 39 (2 pt)'),
    # host-a011_lidar1_1235933786202038686
    (_diag(BASE + [39], [], BASE, [(39, 1, 38), (41, 4, 39)]), 'recovered also keeps ring 39'),
    # 64-beam shapes: a several-point fragment kept alone; two kept fragments of one ring
    (_diag(BASE, [(0, 0, 3)], BASE, [(0, 4, 0)]), 'recovered keeps a fragment of 0 (3 pt)'),
    (_diag(BASE, [(22, 20, 2), (23, 20, 2)], BASE, [(22, 3, 20), (23, 2, 20)]),
     'recovered keeps fragments of 20 (2 fragments, 2 and 2 pt)'),
])
def test_what_differs_writes_the_40_beam_cells(d, text):
    assert V.what_differs(d) == text


def test_what_differs_one_known_wording_difference():
    # host-a004_lidar1_1233422550501267926: the report says "a fragment of 39 (10 pt)"; the generator says "its fragment"
    d = _diag(BASE + [39], [(41, 39, 10)], BASE, [(38, 1, 37), (41, 17, 39)])
    assert V.what_differs(d) == 'recovered also keeps ring 39 and its fragment (10 pt)'


def test_cells():
    t = {'n_extra': 0, 'n_missing': 0, 'n_displaced': 0}
    assert V.template_cell(t) == 'none'
    assert V.template_cell(dict(t, n_extra=1)) == '1 extra cut'
    assert V.template_cell(dict(t, n_extra=2)) == '2 extra cuts'
    assert V.template_cell({'n_extra': 3, 'n_missing': 1, 'n_displaced': 2}) == '3 extra, 1 missing and 2 displaced cuts'
    d = _diag([], [], [], [(3, 1, 32), (5, 17, 39)])
    d['fragments'][1]['azimuth_deg'] = -4.46
    assert V.fragments_cell(d) == '1 @ 0.0 -> 32, 17 @ -4.5 -> 39'
    assert V.fragments_cell(_diag([], [], [], [])) == 'none'


def _recs(plat, counts):
    return [{'file': 'f%03d.bin' % i, 'run2': {'rings': c}} for i, c in enumerate(counts)]


def test_f4_selection_unchanged_by_default_and_all_frames_when_aligned():
    wrong64 = [int(c) if c != 64 else 65 for c in np.random.default_rng(0).integers(63, 105, size=64)]
    recs64 = _recs('64', [64] * 136 + wrong64)
    recs40 = _recs('40', [40] * 184 + [41] * 6 + [42] * 3 + [43] * 3 + [44] * 2 + [45, 46])
    key64 = sorted([r for r in recs64 if r['run2']['rings'] != 64], key=lambda r: (abs(r['run2']['rings'] - 64), r['file']))
    six = [key64[int(round(x * 63))] for x in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    assert V.f4_wrong_frames(recs64, '64') == six
    assert V.f4_wrong_frames(recs64, '64', align=True) == key64 and len(key64) == 64
    assert len(V.f4_wrong_frames(recs40, '40')) == 16 == len(V.f4_wrong_frames(recs40, '40', align=True))
    assert V.f4_wrong_frames(recs40, '40') == V.f4_wrong_frames(recs40, '40', align=True)
    assert V.draws_lowpurity('64') and not V.draws_lowpurity('40')
    assert V.draws_lowpurity('40', align=True) and V.draws_lowpurity('64', align=True)
