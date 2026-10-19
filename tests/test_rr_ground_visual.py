"""tools/analysis/rr_ground_visual.py (CPU; experiments_md 20261018_03 §13): per-bin statistics and the quantile picks."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import rr_ground_visual as G  # noqa: E402


def test_bin_stats():
    r = np.r_[np.full(10, 2.0), np.full(3, 7.0), np.full(20, 12.0)]
    e = np.r_[np.arange(10, dtype=float), [1.0, 2.0, 3.0], np.linspace(-1, 1, 20)]
    s = G.bin_stats(r, e, edges=np.array([0.0, 5.0, 10.0, 15.0]), min_n=5)
    assert s[:, 1].tolist() == [10, 3, 20]
    assert np.isclose(s[0, 2], 4.5) and np.isclose(s[0, 3], np.percentile(np.arange(10), 10))
    assert np.isnan(s[1, 2]) and np.isclose(s[2, 2], 0.0)


def test_pick_near_quantiles():
    v = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 2.0])
    k, q = G.pick_near_quantiles(v)
    assert np.allclose(q, np.percentile(v, (25, 50, 75)))
    assert len(set(k.tolist())) == 3 and k[1] == 2          # the median 2.0: the earlier of the two 2.0s
