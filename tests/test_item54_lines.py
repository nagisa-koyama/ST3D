"""Item 54's line counts, detection assignment, cell view and readings (experiments_md 20261003_05 §5 item 54, design fixed
2026-11-26 22:53 JST). CPU only; the last test reads one KITTI val frame when the data are mounted."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item54_lines as m  # noqa: E402


def axis_inside(points, boxes):
    """Heading-0 containment, (m, n) like points_in_boxes_cpu."""
    p, b = points[None, :, :3], boxes[:, None, :]
    return np.all(np.abs(p - b[..., :3]) <= b[..., 3:6] / 2, axis=-1).astype(np.int32)


def test_lines_count_distinct_rings_above_the_bottom_cut():
    box = np.array([[10, 0, 0.75, 4, 2, 1.5, 0]], np.float32)            # bottom at z = 0
    pts = np.array([[10, 0, 0.1, 0], [10.5, 0, 0.2, 0],                  # ring 7: ground hits under 0.3 m, not counted
                    [10, 0.5, 0.5, 0], [9.5, 0, 0.5, 0],                  # ring 3 twice
                    [10, -0.5, 1.2, 0],                                   # ring 1
                    [30, 0, 1.0, 0]], np.float32)                         # ring 5, outside the box
    ring = np.array([7, 7, 3, 3, 1, 5])
    assert m.box_lines(pts, ring, box, inside_fn=axis_inside).tolist() == [2]
    assert m.box_lines(pts, ring, box, min_height=0.0, inside_fn=axis_inside).tolist() == [3]
    assert m.box_lines(pts[:0], ring[:0], box, inside_fn=axis_inside).tolist() == [0]
    assert m.box_lines(pts, ring, np.zeros((0, 7)), inside_fn=axis_inside).tolist() == []


def test_a_detection_takes_the_cell_of_the_gt_it_overlaps_most():
    own_l, own_r = np.array([5, 1, 0]), np.array([12.0, 35.0, 50.0])
    iou = np.array([[0.2, 0.6], [0.0, 0.0], [0.05, 0.0]])
    lines, rng = m.assign(own_l, own_r, iou, gt_lines=np.array([2, 7]), gt_range=np.array([31.0, 9.0]))
    assert lines.tolist() == [7, 1, 2] and rng.tolist() == [9.0, 35.0, 31.0]
    lines, rng = m.assign(own_l, own_r, np.zeros((3, 0)), np.zeros(0, int), np.zeros(0))
    assert lines.tolist() == [5, 1, 0] and rng.tolist() == [12.0, 35.0, 50.0]


def test_cell_view_ignores_gt_outside_and_drops_detections_outside():
    gt = {'name': np.array(['Car', 'Car', 'Pedestrian', 'DontCare']), 'gt_boxes_lidar': np.zeros((3, 7))}
    dt = {'name': np.array(['Car', 'Car', 'Car']), 'score': np.array([0.9, 0.8, 0.7]), 'boxes_lidar': np.zeros((3, 7))}
    g, d = m.cell_view(gt, dt, gt_lines=[2, 5, -1], gt_range=[35.0, 12.0, 8.0], dt_lines=[2, 5, 1], dt_range=[35.0, 12.0, 55.0],
                       lines_b=(1, 3), range_b=(30.0, 40.0))
    assert g['name'].tolist() == ['Car', 'Van', 'Pedestrian', 'DontCare']
    assert d['score'].tolist() == [0.9] and len(d['boxes_lidar']) == 1
    assert gt['name'].tolist()[1] == 'Car'                                   # copies, inputs untouched


def test_cells_cover_pooled_bins_classes_rings_and_their_crossing():
    names = [c[0] for c in m.cells()]
    assert names[0] == 'pooled' and len(names) == 1 + 7 + 2 + 5 + 10
    assert 'lines >=10' in names and 'lines 4-5' in names and 'ring 40-70.4 few' in names


def _table(q_fn, p=40.0, n=500):
    """Rows (O, P) per cell and seed where O = q_fn(lines class, ring index) * P."""
    rows = []
    for row in ('O', 'P'):
        for seed in (666, 667):
            cells = []
            for name, lb, rb in m.cells():
                cls = 'few' if lb == (1, 3) else 'many' if lb == (3, m.INF) else None
                ring = m.RINGS.index(rb) if rb in m.RINGS else None
                q = q_fn(cls, ring) if cls else 0.5
                cells.append({'cell': name, 'n_gt': n, 'bev': p if row == 'P' else q * p})
            rows.append({'row': row, 'seed': seed, 'stats': 'none', 'reproduced': True, 'cells': cells})
    return m.table(rows, 'none')


def _read(t):
    r1, _ = m.reading1(t)
    r2, _ = m.reading2(t)
    return r1, r2, m.combined(r1, r2, m.reading3(t))


def test_readings_lines_not_range():
    t = _table(lambda cls, ring: 0.9 if cls == 'many' else 0.4)
    r1, r2, c = _read(t)
    assert (r1, r2, c) == ('present', 'present', 'line count, not range')


def test_readings_lines_and_range():
    t = _table(lambda cls, ring: (0.9 if cls == 'many' else 0.5) - 0.06 * (ring or 0))
    assert _read(t)[2] == 'lines and range'


def test_readings_not_lines():
    t = _table(lambda cls, ring: 0.9 - 0.15 * (ring or 0))
    r1, r2, c = _read(t)
    assert r1 == 'absent' and r2 == 'absent' and c == 'not lines'


def test_cells_with_few_gt_or_weak_p_are_not_read():
    assert m.q_of(_table(lambda c, r: 0.5, n=99), 'class few') is None
    assert m.q_of(_table(lambda c, r: 0.5, p=9.9), 'class few') is None
    assert m.q_of(_table(lambda c, r: 0.5), 'class few') == [0.5, 0.5]


def test_s_prime_matches_the_eval_loader_on_a_real_frame():
    pytest.importorskip('torch')
    import kitti_eval_cpu
    if not kitti_eval_cpu.INFOS.exists():
        pytest.skip('KITTI infos not mounted')
    import _init_path  # noqa: F401
    from pcdet.datasets.kitti import kitti_factor_swap
    info = pickle.load(open(kitti_eval_cpu.INFOS, 'rb'))[0]
    fid = info['point_cloud']['lidar_idx']
    pts = np.fromfile(kitti_eval_cpu.VELODYNE / ('%s.bin' % fid), dtype=np.float32).reshape(-1, 4)
    swap = {'RING_THIN': {'MODE': 'pattern', 'SPACING_DEG': 1.33, 'AZ_RES_DEG': 0.332}}
    ring, keep = m._sprime(pts, fid, swap)
    assert np.array_equal(pts[keep], kitti_factor_swap.apply_points(pts, swap, fid))
    assert 0.05 < keep.mean() < 0.4 and ring.max() >= 60
    boxes = np.asarray(info['annos']['gt_boxes_lidar'], np.float32)
    full, thin = m.box_lines(pts, ring, boxes), m.box_lines(pts[keep], ring[keep], boxes)
    assert np.all(thin <= full) and np.all(full <= 64)
