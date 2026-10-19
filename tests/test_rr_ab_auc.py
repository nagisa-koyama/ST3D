"""tools/analysis/rr_ab_auc.py (CPU; experiments_md 20261018_03 §11): arm (a)'s logit equals relocate_render.return_logit
('physics'); the Gauss-Hermite marginal equals a Monte-Carlo average; the within-patch AUC averages per-group AUCs over
groups holding both classes."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import rr_ab_auc as A  # noqa: E402


def test_logit_a_equals_return_logit():
    from pcdet.datasets.augmentor import relocate_render as rr
    rng = np.random.default_rng(0)
    i_rel = rng.uniform(0.05, 20, 200); h = rng.uniform(0, 1, 200); r = rng.uniform(40, 80, 200); c = rng.uniform(0.01, 1, 200)
    assert np.allclose(A.logit_a(i_rel, h, r, c), rr.return_logit((i_rel, h, r, c), 'physics', A.PAR_A))
    l0 = A.logit_a(i_rel, h, r, c, beta=0.0, delta=0.0)
    assert np.allclose(l0, A.PAR_A['c0'] + np.asarray(A.PAR_A['c_band'])[np.minimum((h * 4).astype(int), 3)]
                       + A.PAR_A['gamma'] * np.log(r / 50.0))


def test_marginal_p_gauss_hermite():
    z = np.random.default_rng(1).normal(size=400000)
    for L in (-3.0, 0.0, 2.0):
        mc = np.mean(1 / (1 + np.exp(-(L + np.exp(1.1252) * z))))
        assert abs(A.marginal_p(L, 1.1252) - mc) < 3e-3


def test_within_auc():
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.5, 0.4, 0.3])
    y = np.array([0, 1, 1, 0, 1, 1, 1])
    g = np.array([0, 0, 1, 1, 2, 2, 2])           # group 2 has one class only
    m, n = A.within_auc(s, y, g)
    assert n == 2 and np.isclose(m, 0.5)           # group 0: 1.0, group 1: 0.0
