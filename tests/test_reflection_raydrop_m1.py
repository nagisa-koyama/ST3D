"""tools/analysis/reflection_raydrop_m1.py (experiments_md 20261003_05 §5 item 35b / 35c) on synthetic rings.

Pins the hole definition (runs of 1-4 missing slots between returns agreeing within max(0.5 m, 0.02 r); run length 5
not counted), eligible returns, duplicates (< 0.5 Δ0, later one dropped), the ego exclusion (< 3 m), no link across
rings, the quartile / S arithmetic (hole runs weighted by their length), e_T and the 35c verdict, and the frame draw.
"""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, '..', 'tools', 'analysis', 'reflection_raydrop_m1.py')
spec = importlib.util.spec_from_file_location('reflection_raydrop_m1', TOOL)
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)

D0 = 0.2


def ring_of(n=100, r=20.0, dz=1.0, inten=10.0, ring=0, drop=(), rr=None):
    """One ring of n firing slots, D0 apart; `drop` = missing slot indices; `rr` overrides ranges per slot."""
    idx = np.array([i for i in range(n) if i not in set(drop)])
    rng_ = np.full(n, r) if rr is None else np.asarray(rr, dtype=float)
    return dict(ring=np.full(len(idx), ring), az=idx * D0, r=rng_[idx], dz=np.full(len(idx), dz),
                inten=np.full(len(idx), inten))


def slots_of(*rings, d0=D0):
    cat = {k: np.concatenate([g[k] for g in rings]) for k in ('ring', 'az', 'r', 'dz', 'inten')}
    return m1.frame_slots(cat['ring'], cat['az'], cat['r'], cat['dz'], cat['inten'], d0)


def counts(s):
    return int(s['count'][s['hole']].sum()), int((~s['hole']).sum())


def test_single_hole_and_eligible_returns():
    s, n_dup, n0 = slots_of(ring_of(drop=[50]))
    holes, elig = counts(s)
    assert holes == 1 and n_dup == 0 and n0 == 99
    # returns 1..98 except the dropped 50 and its two neighbours (one neighbour two slots away)
    assert elig == 95
    assert s['count'][s['hole']].tolist() == [1]


def test_runs_of_one_to_four_count_and_five_does_not():
    s, _, _ = slots_of(ring_of(drop=[10, 11, 12, 13] + [40, 41, 42, 43, 44]))
    assert sorted(s['count'][s['hole']].tolist()) == [4]


def test_tolerance_edge():
    rr = np.full(100, 20.0); rr[51:] = 20.4                    # tol max(0.5, 0.4) = 0.5: agrees
    s, _, _ = slots_of(ring_of(drop=[50], rr=rr))
    assert counts(s)[0] == 1
    rr[51:] = 20.6                                              # 0.6 > 0.5: no hole
    s, _, _ = slots_of(ring_of(drop=[50], rr=rr))
    assert counts(s)[0] == 0
    rr = np.full(100, 40.0); rr[51:] = 40.75                    # tol 0.02 * 40.375 = 0.8075: agrees
    s, _, _ = slots_of(ring_of(drop=[50], rr=rr))
    assert counts(s)[0] == 1
    rr[51:] = 40.9                                              # 0.9 > 0.809: no hole
    s, _, _ = slots_of(ring_of(drop=[50], rr=rr))
    assert counts(s)[0] == 0


def test_duplicates_dropped_and_counted():
    base = ring_of(drop=[50])
    dup = {k: np.r_[v, v[20]] for k, v in base.items()}
    dup['az'][-1] = base['az'][20] + 0.3 * D0                   # < 0.5 D0 after slot 20: a duplicate
    s0, _, _ = slots_of(base)
    s1, n_dup, n0 = slots_of(dup)
    assert n_dup == 1 and n0 == 100
    assert counts(s1) == counts(s0)


def test_input_order_does_not_matter():
    g = ring_of(drop=[30, 70, 71])
    perm = np.random.default_rng(0).permutation(len(g['r']))
    s0, _, _ = slots_of(g)
    s1, _, _ = slots_of({k: v[perm] for k, v in g.items()})
    assert counts(s0) == counts(s1)


def test_ego_returns_take_part_in_no_slot():
    s, _, _ = slots_of(ring_of(drop=[50], r=2.5))              # a surface at 2.5 m: neither holes nor eligible
    assert counts(s) == (0, 0)
    s, _, _ = slots_of(ring_of(drop=[50], r=3.0))              # at 3.0 m it counts
    assert counts(s) == (1, 95)
    rr = np.full(100, 20.0); rr[50] = 2.0                      # a near return inside a far surface is a return,
    s, _, _ = slots_of(ring_of(rr=rr))                         # not a hole, and its neighbours are not eligible
    holes, elig = counts(s)
    assert holes == 0 and elig == 98 - 3


def test_no_link_across_rings():
    a = ring_of(n=50, ring=0)
    b = ring_of(n=50, ring=1)
    b['az'] = b['az'] + 0.4                                     # would read as a one-slot gap if linked
    s, _, _ = slots_of(a, b)
    holes, elig = counts(s)
    assert holes == 0 and elig == 2 * 48


def test_quartiles_and_selectivity():
    rel = np.arange(1, 9, dtype=float)
    hole = np.array([True] + [False] * 7)
    cnt = np.ones(8, dtype=np.int64)
    edges, qs, qh, qr, s = m1.quartile_rates(rel, hole, cnt)
    assert qs == [2, 2, 2, 2] and qh == [1, 0, 0, 0]
    assert qr[0] == pytest.approx(0.5) and s == np.inf
    hole[-1] = True
    _, _, _, qr, s = m1.quartile_rates(rel, hole, cnt)
    assert s == pytest.approx(1.0)
    # a run of three missing slots weighs three in the edges and in the rates
    cnt = np.array([3, 1, 1, 1, 1, 1, 1, 1]); hole = np.array([True] + [False] * 7)
    edges, qs, qh, qr, s = m1.quartile_rates(rel, hole, cnt)
    assert sum(qs) == 10 and qh[0] == 3
    # one quartile empty (constant intensity): rates there are nan, S is nan
    _, qs, _, qr, s = m1.quartile_rates(np.ones(8), np.array([True] + [False] * 7), np.ones(8, np.int64))
    assert qs[:3] == [0, 0, 0] and np.isnan(qr[0]) and np.isnan(s)


def test_excess_drop_and_verdict():
    assert m1.excess_drop(0.10, 0.05) == pytest.approx(0.05 / 0.95)
    good = dict(slots=5000, h=0.20, S=3.0)
    tgt = dict(slots=5000, h=0.10, S=1.0)                       # e = 0.10 / 0.90 = 0.111
    v, q, e = m1.verdict({(2, 1): good}, {(2, 1): tgt})
    assert v == 'room' and q == [(2, 1)] and e[(2, 1)] == pytest.approx(0.1 / 0.9)
    v, q, _ = m1.verdict({(2, 1): dict(good, S=1.9)}, {(2, 1): tgt})
    assert v.startswith('no room')
    v, q, _ = m1.verdict({(0, 1): good}, {(0, 1): tgt})           # 0-10 m is outside the room bins
    assert v.startswith('no room')
    v, q, _ = m1.verdict({(2, 1): good}, {(2, 1): dict(tgt, slots=1999)})
    assert v.startswith('no room')
    v, q, _ = m1.verdict({(2, 1): dict(good, S=np.inf)}, {(2, 1): tgt})
    assert v == 'room'
    v, q, _ = m1.verdict({(2, 1): dict(good, h=0.05), (3, 0): dict(good, h=0.01)},
                         {(2, 1): tgt, (3, 0): dict(tgt, h=0.02)})
    assert v.startswith('not makeable')
    v, q, _ = m1.verdict({(2, 1): dict(good, h=0.12)}, {(2, 1): tgt})   # e = 0.022 < 0.05, >= 0
    assert v.startswith('no room')


def test_ring_steps_median_and_ground_mode():
    g = ring_of(drop=[5, 6, 50])
    st = m1.ring_steps(g['ring'], g['az'])
    assert np.median(st) == pytest.approx(D0)
    z = np.r_[np.full(50, -1.73), np.full(10, 0.4)]
    rh = np.r_[np.full(50, 10.0), np.full(10, 12.0)]
    assert m1.frame_ground(z, rh) == pytest.approx(-1.725)       # centre of the [-1.75, -1.70) bin
    assert np.isnan(m1.frame_ground(z, np.full(60, 40.0)))


def test_bins_and_classes():
    assert m1.range_bin(np.array([0.0, 9.99, 10.0, 74.9, 75.0, -1.0])).tolist() == [0, 0, 1, 6, -1, -1]
    assert m1.height_class(np.array([0.0, 0.3, -0.3, 0.31, 3.0, 3.01, -0.31])).tolist() == [0, 0, 0, 1, 1, -1, -1]


def test_spread_draw():
    lists = [[(s, f) for f in range(80)] for s in range(61)]
    out, how = m1.spread_draw(np.random.default_rng(1), lists)
    assert len(out) == 200 and len(set(out)) == 200 and how.startswith('spread')
    per = np.bincount([s for s, _ in out], minlength=61)
    assert per.min() == 3 and per.max() == 4 and (per == 4).sum() == 200 - 183
    out, how = m1.spread_draw(np.random.default_rng(1), lists, mode='one_each')
    assert len(out) == 61 and len({s for s, _ in out}) == 61
    lists = [[(s, f) for f in range(5)] for s in range(250)]
    out, how = m1.spread_draw(np.random.default_rng(1), lists)
    assert len(out) == 200 and len({s for s, _ in out}) == 200 and how == 'one per scene'


def kitti_like_ring(n=140, half=0.48, after=1.03, drop=()):
    """Firing grid with a half step after every 6th slot (KITTI's HDL-64E pattern): the half step is `half` d0 and the
    step after it `after` d0 (the measured jitter is about +-0.03)."""
    pos, x = [], 0.0
    for i in range(n):
        pos.append(x)
        x += (half if i % 7 == 6 else (after if i % 7 == 0 and i > 0 else 1.0)) * D0
    idx = [i for i in range(n) if i not in set(drop)]
    return dict(ring=np.zeros(len(idx), np.int64), az=np.array(pos)[idx], r=np.full(len(idx), 20.0),
                dz=np.ones(len(idx)), inten=np.full(len(idx), 10.0))


def test_half_step_grid_needs_the_quarter_duplicate_rule():
    g = kitti_like_ring()
    # 0.5 d0 rule: every 0.48 half step drops a real return, and 0.48 + 1.03 = 1.51 d0 leaves a fake hole
    s, n_dup, _ = slots_of(g)
    assert n_dup == 19 and counts(s)[0] == 19
    # 0.25 d0 rule: the half step is a link, no duplicate, no hole, and its returns are eligible
    s, n_dup, _ = m1.frame_slots(g['ring'], g['az'], g['r'], g['dz'], g['inten'], D0, dup_frac=0.25)
    assert n_dup == 0 and counts(s) == (0, 138)
    # one missing slot next to a half step is a ~1.5 d0 gap: one hole
    g = kitti_like_ring(half=0.5, after=1.0, drop=[7])
    s, _, _ = m1.frame_slots(g['ring'], g['az'], g['r'], g['dz'], g['inten'], D0, dup_frac=0.25)
    assert s['count'][s['hole']].tolist() == [1]
