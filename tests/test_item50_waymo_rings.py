"""Item 50's ring view and row table (experiments_md 20261003_05 §5 item 50, design fixed 2026-11-21 21:58 JST). CPU only."""
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS / 'analysis'))
import item50_waymo_rings as m  # noqa: E402


def frame():
    gt = {'name': np.array(['Car', 'Car', 'Pedestrian', 'Car']), 'score': np.zeros(4)}
    gxy = np.array([[5.0, 0.0], [0.0, 25.0], [12.0, 0.0], [60.0, 0.0]])
    dt = {'name': np.array(['Car', 'Car', 'Car']), 'score': np.array([0.9, 0.8, 0.7]),
          'bbox': np.zeros((3, 4)), 'frame_id': 'f'}
    dxy = np.array([[5.0, 0.1], [0.0, 25.5], [45.0, 0.0]])
    return dt, gt, dxy, gxy


def test_ring_view_ignores_car_gt_outside_and_drops_detections_outside():
    dt, gt, dxy, gxy = frame()
    d, g = m.ring_filter(dt, gt, dxy, gxy, 20.0, 30.0)
    assert list(g['name']) == ['Van', 'Car', 'Pedestrian', 'Van']  # only Car becomes Van; other classes untouched
    assert list(d['score']) == [0.8] and d['bbox'].shape == (1, 4) and d['frame_id'] == 'f'
    d, g = m.ring_filter(dt, gt, dxy, gxy, 40.0, float('inf'))
    assert list(g['name']) == ['Van', 'Van', 'Pedestrian', 'Car'] and list(d['score']) == [0.7]
    assert list(gt['name']) == ['Car', 'Car', 'Pedestrian', 'Car']  # the inputs are not modified


def test_whole_range_ring_is_the_identity():
    dt, gt, dxy, gxy = frame()
    d, g = m.ring_filter(dt, gt, dxy, gxy, 0.0, float('inf'))
    assert list(g['name']) == list(gt['name']) and list(d['score']) == list(dt['score'])


def test_rings_cover_every_range_once():
    edges = [lo for lo, _ in m.RINGS] + [m.RINGS[-1][1]]
    assert edges[0] == 0.0 and edges[-1] == float('inf')
    assert all(a[1] == b[0] for a, b in zip(m.RINGS, m.RINGS[1:]))


def test_rows_share_the_teacher_and_use_the_rerun_for_pl2_src():
    rs = m.rows()
    assert len(rs) == 10
    assert sum(r['row'] == 'T' for r in rs) == 2  # T 27471 once, both stats
    pl2 = [r for r in rs if (r['row'], r['stats'], r['seed']) == ('PL', 'src', 667)][0]
    assert pl2['dir'].endswith('/i46_P2nw_src_r/eval/epoch_20/i46_P2nw_src_r')
    assert {r['job'] for r in rs} == {27471, 33102, 33264, 27704, 28563}
