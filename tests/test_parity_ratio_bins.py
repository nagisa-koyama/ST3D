"""tools/analysis/parity_ratio_bins.py: per-5-m-bin points per frame and source / target ratios (T05 §5 item 42d.7).

Synthetic clouds with points at known radii pin which row each lands in, so an off-by-one bin (a shifted row label,
dropping the 0-5 m row, or keeping the 70-75 m one) fails, and the parity depth reads the 5-70 m band only.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))
import parity_ratio_bins as prb  # noqa: E402


def cloud(radii, z=0.0):
    """Points on the +x axis at the given ranges (x-y radius = x), height z, intensity 0."""
    r = np.asarray(radii, float)
    return np.column_stack([r, np.zeros_like(r), np.full_like(r, z), np.zeros_like(r)])


def test_points_land_in_their_5m_row():
    # one point per bin centre 2.5, 7.5, ..., 72.5; plus one below the z cut that must not count
    pts = np.concatenate([cloud(np.arange(15) * 5 + 2.5), cloud([12.5], z=-2.5)])
    h = prb.mean_hist([pts])
    rows = prb.ratio_rows(h, {9: 2 * h})
    assert len(rows) == 14
    assert [r['lo'] for r in rows] == [5.0 * b for b in range(14)]
    assert [r['hi'] for r in rows] == [5.0 * b + 5 for b in range(14)]
    assert all(r['target'] == 1.0 for r in rows)          # the z-cut point at 12.5 m is not in row 2
    assert all(r['ratio_9'] == 2.0 for r in rows)


def test_far_row_is_65_70_and_70_75_is_dropped():
    tgt = prb.mean_hist([cloud([67.5, 72.5, 72.6])])
    src = prb.mean_hist([cloud([67.5, 67.6, 67.7])])
    rows = prb.ratio_rows(tgt, {10: src})
    assert (rows[13]['lo'], rows[13]['hi']) == (65.0, 70.0)
    assert rows[13]['target'] == 1.0 and rows[13]['src_10'] == 3.0 and rows[13]['ratio_10'] == 3.0
    assert rows[12]['target'] == 0.0 and np.isnan(rows[12]['ratio_10'])


def test_mean_is_per_frame():
    h = prb.mean_hist([cloud([7.5, 7.6]), cloud([7.5, 7.6, 7.7, 7.8])])
    assert h[1] == 3.0 and h.sum() == 3.0


def test_parity_depth_reads_5_to_70_only():
    tgt = np.ones(15)
    shallow = np.ones(15); shallow[0] = 0.0; shallow[14] = 0.0   # short only outside the band
    deep = np.ones(15); deep[13] = 0.99                            # short in 65-70 m
    assert prb.smallest_parity_depth(tgt, {3: deep, 9: shallow}) == 9
    assert prb.smallest_parity_depth(tgt, {3: deep}) is None
