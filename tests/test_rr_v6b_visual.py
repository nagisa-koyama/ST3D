"""tools/analysis/rr_v6b_visual.py (CPU; experiments_md 20261018_03 §10): the pools' mean p by h_rel band counts only the
COVERED valid pixels of a patch (channel 1 > 0 and channel 4 > 0), banded by channel 2."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import rr_v6b_visual as R  # noqa: E402


def test_patch_band_means_covered_valid_only():
    X = np.zeros((6, 4, 4), np.float32); P = np.zeros((4, 4), np.float32)
    X[4, :, :3] = 1.0                       # valid: the first three columns
    X[1, :, 1:] = 0.5                       # covered: the last three columns
    X[2] = np.array([0.1, 0.3, 0.6, 1.0])[:, None]      # h_rel by row: bands 0, 1, 2, 3
    P[:] = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    s, n = R.patch_band_means(X, P)
    assert n.tolist() == [2, 2, 2, 2]       # columns 1-2 of each row
    assert np.allclose(s, [(1 + 2) / 16, (5 + 6) / 16, (9 + 10) / 16, (13 + 14) / 16])
    assert R.bin_of(55.0)[0] == 3 and R.bin_of(27.0)[0] == -1
