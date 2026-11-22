"""Item 53 D1 box-error components and readings (tools/analysis/item53_iassd_box_errors.py). CPU only, synthetic boxes.

KITTI camera frame, (x, y_bottom, z, l, h, w, ry), y pointing DOWN: a prediction whose bottom is 0.3 m higher has y 0.3 smaller.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'analysis'))
pytest.importorskip('shapely')
import item53_iassd_box_errors as m  # noqa: E402

G = np.array([2.0, 1.6, 20.0, 3.9, 1.53, 1.6, 0.3])


def test_a_perfect_prediction_has_zero_components_and_passes_every_counterfactual():
    c = m.components(G.copy(), G)
    assert all(abs(v) < 1e-12 for v in c.values())
    assert all(v > 0.999 for v in m.counterfactual_ious(G.copy(), G).values())


def test_signs_and_which_counterfactual_repairs_a_vertical_offset():
    p = G.copy(); p[1] -= 0.3                                   # bottom 0.3 m higher (y down)
    c = m.components(p, G)
    assert c['bottom_up'] == pytest.approx(0.3) and c['centre_up'] == pytest.approx(0.3)
    cf = m.counterfactual_ious(p, G)
    assert cf['as_is'] < 0.7 and cf['bottom'] > 0.999 and cf['vertical'] > 0.999 and cf['height'] < 0.7
    p = G.copy(); p[4] *= np.exp(0.2)                          # taller box, same bottom
    c = m.components(p, G)
    assert c['log_h'] == pytest.approx(0.2) and c['bottom_up'] == pytest.approx(0.0)
    assert m.counterfactual_ious(p, G)['height'] > 0.999


def test_heading_error_wraps_and_flags_a_flip():
    p = G.copy(); p[6] = G[6] + 2 * np.pi - 0.1
    assert m.components(p, G)['dyaw'] == pytest.approx(-0.1) and m.components(p, G)['flip'] == 0.0
    p[6] = G[6] + np.pi - 0.05
    assert m.components(p, G)['flip'] == 1.0


def test_loss_shares_and_reading_one():
    none = {'as_is': 0.6, 'height': 0.7, 'bottom': 0.6, 'vertical': 0.7, 'lw': 0.6, 'lhw': 0.7, 'centre': 0.62,
            'heading': 0.6}
    tgt = {'as_is': 0.1, 'height': 0.15, 'bottom': 0.45, 'vertical': 0.55, 'lw': 0.12, 'lhw': 0.2, 'centre': 0.11,
           'heading': 0.1}
    loss, s = m.loss_shares(none, tgt)
    assert loss == pytest.approx(0.5) and s['vertical'] == pytest.approx(0.9) and s['bottom'] == pytest.approx(0.7)
    r = {'S': s}
    assert m.reading1([r, r]) == 'vertical carries the collapse'
    other = {'S': dict(s, vertical=0.3)}
    assert m.reading1([r, other]) == 'no single component'      # seeds must agree
    assert m.loss_shares(tgt, none)[1]['vertical'] is None        # no loss: no share


def test_directions_read_a_shift_a_spread_and_a_level():
    rng = np.random.RandomState(0)
    base = {n: rng.normal(0, 0.05, 400) for n in m.SHIFT}
    base.update(centre=np.abs(rng.normal(0, 0.2, 400)), abs_dyaw=np.abs(rng.normal(0, 0.03, 400)), flip=np.zeros(400))
    x = {n: v.copy() for n, v in base.items()}
    x['bottom_up'] = x['bottom_up'] - 0.2                         # shifted
    x['log_l'] = x['log_l'] * 3                                   # spreads, no shift
    d = m.directions(base, x)
    assert d['bottom_up']['reading'] == 'shifted' and d['log_l']['reading'] == 'spreads'
    assert d['log_w']['reading'] == 'level' and d['centre']['reading'] == 'level' and d['heading']['reading'] == 'level'
    x['flip'] = np.r_[np.ones(40), np.zeros(360)]                 # +10 points of flips
    assert m.directions(base, x)['heading']['reading'] == 'turns'


def test_pairing_keeps_only_the_gt_both_arms_matched():
    a = {('f', 0): (G.copy(), G), ('f', 1): (G.copy(), G)}
    q = G.copy(); q[1] -= 0.3
    b = {('f', 1): (q, G), ('g', 0): (G.copy(), G)}
    r = m.paired(a, b)
    assert r['n'] == 1 and r['pass_a']['as_is'] == 1.0 and r['pass_b']['as_is'] == 0.0
    assert r['S']['bottom'] == pytest.approx(1.0)
