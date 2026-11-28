"""Item 60g' box terms (tools/analysis/item60g2_box_terms.py; experiments_md 20261003_05 §5 item 60g', design fixed
2026-11-28 12:27 JST). CPU only, synthetic KITTI-style frames; run from the repo root.

KITTI camera frame, (x, y_bottom, z, l, h, w, ry), y pointing DOWN: a prediction whose bottom is 0.3 m higher has y 0.3
smaller.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools')); sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))
pytest.importorskip('shapely')
import item60g2_box_terms as m  # noqa: E402

G = np.array([2.0, 1.6, 12.0, 3.9, 1.53, 1.6, 0.3])


def _annos():
    # rows: valid Car | Car 25.0 px tall (moderate needs > 25) | Car 25.5 px | Car occluded 2 | Car truncated 0.31 |
    # Van | Pedestrian | DontCare (last, no LiDAR box)
    names = ['Car', 'Car', 'Car', 'Car', 'Car', 'Van', 'Pedestrian', 'DontCare']
    tops = [100.0] * 8
    hts = [40.0, 25.0, 25.5, 40.0, 40.0, 40.0, 40.0, 40.0]
    n = len(names)
    lidar = np.array([[3.0 + 4 * i, -1.0 - i, -1.0, 3.9, 1.6, 1.53, 0.0] for i in range(n - 1)])
    return {'name': np.array(names), 'bbox': np.array([[0.0, t, 50.0, t + h] for t, h in zip(tops, hts)]),
            'occluded': np.array([0, 0, 1, 2, 0, 0, 0, -1]), 'truncated': np.array([0.0, 0.0, 0.3, 0.0, 0.31, 0, 0, -1]),
            'location': np.array([[-l[1], 1.6, l[0]] for l in lidar] + [[-1000.0, -1000.0, -1000.0]]),
            'dimensions': np.tile([3.9, 1.53, 1.6], (n, 1)), 'rotation_y': np.zeros(n), 'gt_boxes_lidar': lidar}


@pytest.fixture(scope='module')
def official():
    kitti_eval_cpu = pytest.importorskip('kitti_eval_cpu')
    return kitti_eval_cpu.load_official_eval()


def test_gt_set_is_the_official_moderate_car_set(official):
    a = _annos()
    idx, boxes, rng = m.moderate_gt(a, official.clean_data)
    assert idx.tolist() == [0, 2]                       # 25.0 px, occlusion 2, truncation 0.31, Van, Ped, DontCare out
    ign = np.asarray(official.clean_data(a, {'name': np.zeros(0, dtype=str)}, 0, 1)[1])
    assert idx.tolist() == np.flatnonzero(ign == 0).tolist()


def test_range_comes_from_the_gts_own_lidar_row(official):
    a = _annos()
    idx, boxes, rng = m.moderate_gt(a, official.clean_data)
    np.testing.assert_allclose(rng, [np.hypot(3.0, -1.0), np.hypot(11.0, -3.0)])   # rows 0 and 2, not 0 and 1
    np.testing.assert_allclose(boxes[1, :3], a['location'][2])


def _frame(preds, scores):
    P = np.asarray(preds, dtype=np.float64).reshape(-1, 7)
    return {'frame_id': 'f', 'name': np.array(['Car'] * len(P)), 'score': np.asarray(scores, dtype=np.float64),
            'location': P[:, :3], 'dimensions': P[:, 3:6], 'rotation_y': P[:, 6]}


def test_match_keys_by_gt_row_and_carries_the_range():
    gts = {'f': (np.array([5]), G[None], np.array([12.5]))}
    p = G.copy(); p[0] += 0.2
    got = m.match([_frame([p, G + np.array([30.0, 0, 0, 0, 0, 0, 0])], [0.9, 0.8])], gts)
    assert list(got) == [('f', 5)] and got[('f', 5)][2] == 12.5
    assert m.match([_frame([p], [0.2])], gts) == {}     # below the score floor
    q = G.copy(); q[0] += 3.0                           # BEV IoU < 0.5
    assert m.match([_frame([q], [0.9])], gts) == {}


def test_bins_and_counts_against_the_rings():
    gts = {'a': (np.arange(4), np.tile(G, (4, 1)), np.array([5.0, 19.99, 20.0, 40.0]))}
    assert m.gt_counts(gts) == {'near': 2, 'far': 1, 'all': 4}
    pairs = {('a', i): (G, G, r) for i, r in enumerate([5.0, 19.99, 20.0, 40.0])}
    assert sorted(m.in_bin(pairs, 'far')) == [('a', 2)] and len(m.in_bin(pairs, 'all')) == 4
    ring = lambda lo, hi, n: {'lo': lo, 'hi': hi, 'n_gt': n}            # noqa: E731
    row = {'pooled': {'n_gt': 9}, 'rings': [ring(0.0, 10.0, 1), ring(10.0, 20.0, 2), ring(20.0, 30.0, 3),
                                            ring(30.0, 40.0, 2), ring(40.0, 70.4, 1)]}
    assert m.ring_counts([row, row]) == {'near': 3, 'far': 5, 'all': 9}
    bad = dict(row, pooled={'n_gt': 8})
    with pytest.raises(AssertionError):
        m.ring_counts([row, bad])


def _arms(change, n=40):
    """`ada` near-exact (small jitter, so every term has a spread), `glob` the same boxes with `change` applied."""
    rng = np.random.default_rng(0)
    ada, glob = {}, {}
    for i in range(n):
        g = G.copy(); g[0] += 0.01 * i
        p = g.copy()
        p[:3] += rng.uniform(-0.02, 0.02, 3)
        p[3:6] *= np.exp(rng.uniform(-0.01, 0.01, 3))
        p[6] += rng.uniform(-0.01, 0.01)
        ada[('f', i)] = (p.copy(), g)
        glob[('f', i)] = (change(p.copy()), g)
    return ada, glob


def _bottom(p):
    p[1] -= 0.3
    return p


def _height(p):
    p[4] *= np.exp(0.5)
    return p


def _heading(p):
    p[6] += 0.5
    return p


def test_a_pure_bottom_offset_is_carried_by_bottom():
    r = m.paired(*_arms(_bottom))
    assert r['loss'] == pytest.approx(1.0) and r['S']['bottom'] == pytest.approx(1.0)
    assert r['S']['height'] == pytest.approx(0.0) and r['S']['vertical'] == pytest.approx(1.0)
    assert m.reading1([r, r], 'near') == 'bottom carries the near 3D loss'
    d = m.reading2([r, r])
    assert d['bottom_up'] == 'shifted' and d['centre_up'] == 'shifted' and d['log_h'] == 'level'


def test_seeds_disagreeing_read_no_single_term_or_the_combination():
    b, h, y = (m.paired(*_arms(f)) for f in (_bottom, _height, _heading))
    assert h['S']['height'] == pytest.approx(1.0) and y['S']['heading'] == pytest.approx(1.0)
    assert m.reading1([b, h], 'far') == 'the vertical terms together carry it'
    assert m.reading1([b, y], 'near') == m.NO_SINGLE
    assert m.reading2([b, h])['bottom_up'] == 'seeds disagree (shifted / level)'


def test_a_small_loss_is_not_carried_and_missing_is_not_read():
    same = m.paired(*_arms(lambda p: p))
    assert same['loss'] == 0.0
    assert m.reading1([m.paired(*_arms(_bottom)), same], 'near') == m.NOT_CARRIED
    assert m.reading1([None, same], 'near') == 'not read'
    assert m.paired({('f', 0): (G, G)}, {('f', 1): (G, G)}) is None
