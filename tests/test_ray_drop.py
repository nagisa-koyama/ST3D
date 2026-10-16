"""CPU tests of the label-free ray-drop (tools/lit_port/ray_drop.py, experiments_md 20261017_01)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools' / 'lit_port'))
import ray_drop as rd  # noqa: E402


def pts(r):
    r = np.asarray(r, np.float64)
    return np.stack([r, np.zeros_like(r), np.zeros_like(r)], 1).astype(np.float32)


def test_bins_edges_and_outside():
    b = rd.bin_of(np.array([2.99, 3.0, 4.99, 5.0, 69.99, 70.0, 100.0]))
    assert b.tolist() == [-1, 0, 0, 1, rd.NBIN - 1, -1, -1]


def test_outside_range_always_kept_and_p0_drops_inside():
    p = np.zeros(rd.NBIN)
    m = rd.keep_mask(pts([1, 2.9, 10, 69, 71]), np.zeros(3), p, 'seq', 3)
    assert m.tolist() == [True, True, False, False, True]


def test_p1_keeps_all_and_deterministic():
    p = np.full(rd.NBIN, 0.5)
    x = pts(np.linspace(3, 69, 5000))
    a = rd.keep_mask(x, np.zeros(3), p, 'segA', 7)
    b = rd.keep_mask(x, np.zeros(3), p, 'segA', 7)
    c = rd.keep_mask(x, np.zeros(3), p, 'segA', 8)
    assert (a == b).all() and (a != c).any()
    assert abs(a.mean() - 0.5) < 0.03
    assert rd.keep_mask(x, np.zeros(3), np.ones(rd.NBIN), 'segA', 7).all()


def test_origin_shift_and_empty_and_bad_p():
    p = np.zeros(rd.NBIN)
    o = np.array([100.0, 0, 0])
    assert rd.keep_mask(pts([105]), o, p, 's', 0).tolist() == [True] * 0 + [False]   # 5 m from the origin: dropped
    assert rd.keep_mask(np.zeros((0, 3), np.float32), np.zeros(3), p, 's', 0).shape == (0,)
    with pytest.raises(AssertionError):
        rd.keep_mask(pts([10]), np.zeros(3), np.full(rd.NBIN, 1.5), 's', 0)
    with pytest.raises(AssertionError):
        rd.keep_mask(pts([10]), np.zeros(3), np.ones(3), 's', 0)
