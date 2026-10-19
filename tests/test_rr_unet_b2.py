"""tools/analysis/rr_unet_b2.py (CPU; experiments_md 20261018_03 §12): the validation carve-out, and the augmentation
(intensity before the within-instance rank, depth noise and the 3 x 3 mean over covered pixels only)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import rr_unet_b2 as B  # noqa: E402


class StubRng:
    """random() returns the queued values in order; normal() returns `mu` (a constant) per draw."""

    def __init__(self, randoms, mu=0.0):
        self.r = list(randoms); self.mu = mu

    def random(self):
        return self.r.pop(0)

    def normal(self, loc, scale, n):
        return np.full(n, self.mu)


def _patch():
    X = np.zeros((6, 6, 8), np.float32)
    cov = np.array([[1, 1, 0], [1, 2, 1], [2, 1, 2], [4, 6, 3]], np.int64)
    agg = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 6.0])
    from pcdet.datasets.augmentor import relocate_render as rr
    X[1, cov[:, 0], cov[:, 1]] = rr.intensity_rank(agg)[cov[:, 2]]
    X[0, cov[:, 0], cov[:, 1]] = [1.0, 2.0, 3.0, 10.0]
    return X, cov, agg


def test_val_segments():
    segs = [f's{i:03d}' for i in range(638)]
    v = B.val_segments(segs)
    assert len(v) == 64 and v == B.val_segments(list(reversed(segs))) and v <= set(segs)


def test_augment_none_identity_and_rank_before_transform():
    X, cov, agg = _patch()
    assert np.array_equal(B.augment(X, cov, agg, StubRng([0.9, 0.9])), X)            # neither branch
    # intensity branch with eps = 0: the recomputed ranks equal relocate_render.intensity_rank
    assert np.allclose(B.augment(X, cov, agg, StubRng([0.1, 0.9], mu=0.0))[1], X[1])
    # eps = log(10) on covered points 0..3: values 50, 10, 30, 20 among [.., 4, 6] -> ranks among all six points
    out = B.augment(X, cov, agg, StubRng([0.1, 0.9], mu=np.log(10.0)))
    v = np.array([50.0, 10.0, 30.0, 20.0, 4.0, 6.0]); exp = B.rank_within(v)[:4]
    assert np.allclose(out[1, cov[:, 0], cov[:, 1]], exp) and np.array_equal(out[0], X[0])


def test_augment_depth_noise_and_covered_mean():
    X, cov, agg = _patch()
    out = B.augment(X, cov, agg, StubRng([0.9, 0.1, 0.9], mu=0.5))                    # noise only
    assert np.allclose(out[0, cov[:, 0], cov[:, 1]], [1.5, 2.5, 3.5, 10.5]) and np.array_equal(out[1], X[1])
    out = B.augment(X, cov, agg, StubRng([0.9, 0.1, 0.1], mu=0.0))                    # noise 0, then the mean filter
    # (1,1),(1,2),(2,1) are mutual neighbours: each averages the three; (4,6) has no covered neighbour
    assert np.allclose(out[0, cov[:, 0], cov[:, 1]], [2.0, 2.0, 2.0, 10.0])
    assert np.count_nonzero(out[0]) == 4                                               # uncovered pixels untouched
