"""Temporal pseudo-label propagation (tools/analysis/pseudo_label_propagation*.py, experiments_md/20261017_02 §9).

Sentinel values (bug_prevention_checklist rule 1): a negative label (pseudo-label ignore band) and -99 are never seeds
and never become positives; the original rows stay bit-identical; empty frames and one-frame sequences work. Geometry:
exact static propagation under known poses, no static cluster for a moving car, the CV tracker, rotated BEV IoU, the
ground-relative count. Consumers: DatasetTemplate.fill_pseudo_labels on a merged entry; the label-free Waymo loader.
"""
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'tools' / 'analysis'
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))
import _init_path  # noqa: F401,E402  (the live pcdet, not the copy baked into the image)
import pseudo_label_propagation as P  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pose(x, y, yaw, z=0.0):
    M = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    M[:2, :2] = [[c, -s], [s, c]]
    M[:3, 3] = [x, y, z]
    return M


def local_box(gbox, M):
    return P.transform_boxes(np.asarray(gbox, float)[None], np.linalg.inv(M))[0]


def row(box, label, score):
    return np.concatenate([np.asarray(box, float)[:7], [label, score]])


CAR_G = np.array([60.0, 5.0, 0.8, 4.5, 1.9, 1.6, 0.3])  # a parked car in the global frame


def ego_poses(n=5):
    # ego drives along x at 10 m/s, turning slightly
    return [pose(10.0 * i, 0.5 * i, 0.02 * i) for i in range(n)]


def static_sequence(detected=(3, 4), extra=None):
    Ms = ego_poses()
    frames = []
    for i, M in enumerate(Ms):
        rows = [row(local_box(CAR_G, M), 1, 0.9)] if i in detected else []
        if extra is not None:
            rows += extra(i, M)
        frames.append(P.Frame(f'seq_{i:03d}', float(i), M, np.array(rows).reshape(-1, 9)))
    return frames, Ms


# ------------------------------------------------------------------ geometry

def test_bev_iou_known_values():
    a = np.array([[0, 0, 0, 2, 2, 1, 0]], float)
    assert P.bev_iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert P.bev_iou_matrix(a, a + [[10, 0, 0, 0, 0, 0, 0]])[0, 0] == 0.0
    assert P.bev_iou_matrix(a, a + [[1, 0, 0, 0, 0, 0, 0]])[0, 0] == pytest.approx(1 / 3)
    assert P.bev_iou_matrix(a, a + [[0, 0, 0, 0, 0, 0, np.pi / 2]])[0, 0] == pytest.approx(1.0)
    inter = 4 * 2 * (np.sqrt(2) - 1)  # square of side 2 against itself rotated by 45 deg
    assert P.bev_iou_matrix(a, a + [[0, 0, 0, 0, 0, 0, np.pi / 4]])[0, 0] == pytest.approx(inter / (8 - inter))
    assert P.bev_iou_matrix(np.zeros((0, 7)), a).shape == (0, 1)


def test_static_propagation_is_exact():
    frames, Ms = static_sequence(detected=(3, 4))
    clusters, _ = P.static_clusters(frames, s_conf=0.5, d_assoc=0.5)
    assert len(clusters) == 1 and clusters[0]['static'] and clusters[0]['frames'] == [3, 4]
    cands = P.propagate_static(frames, clusters, H=3)
    assert sorted(cands) == [0, 1, 2]
    for g in (0, 1, 2):
        (c,) = cands[g]
        truth = local_box(CAR_G, Ms[g])
        assert np.allclose(c['box'][:6], truth[:6], atol=1e-9)
        assert abs(P.wrap_angle(c['box'][6] - truth[6])) < 1e-9
        assert c['delta'] == pytest.approx(3 - g)
    assert sorted(P.propagate_static(frames, clusters, H=2)) == [1, 2]


def test_negative_labels_and_minus99_never_seed():
    def extra(i, M):
        b = local_box(CAR_G + [0, 8, 0, 0, 0, 0, 0], M)
        return [row(b, -1, 0.99), row(b + [0, 3, 0, 0, 0, 0, 0], -99, 0.99),
                row(b + [0, 6, 0, 0, 0, 0, 0], 2, 0.99), row(b + [0, 9, 0, 0, 0, 0, 0], -2, 0.99)]
    frames, _ = static_sequence(detected=(), extra=extra)
    for f in frames:
        assert len(P.seeds_of(f, 0.1)) == 0
    clusters, _ = P.static_clusters(frames, 0.1, 1.0)
    assert clusters == []
    assert P.propagate_static(frames, clusters, 5) == {}
    assert P.cv_tracks(frames, 0.1, set(), 20.0, 2.0) == []


def test_moving_car_forms_no_static_cluster_and_cv_track_recovers_it():
    Ms = ego_poses(5)
    def car_at(t):
        return np.array([20.0 + 8.0 * t, -3.0, 0.8, 4.5, 1.9, 1.6, 0.0])
    frames = []
    for i, M in enumerate(Ms):
        rows = [] if i == 2 else [row(local_box(car_at(i), M), 1, 0.9)]
        frames.append(P.Frame(f's_{i:03d}', float(i), M, np.array(rows).reshape(-1, 9)))
    clusters, _ = P.static_clusters(frames, 0.5, 1.0)
    assert not any(c['static'] for c in clusters)
    assert P.propagate_static(frames, clusters, 5) == {}
    tracks = P.cv_tracks(frames, 0.5, set(), d_init=10.0, d_cv=1.0)
    assert len(tracks) == 1 and len(tracks[0]['obs']) == 4
    cands = P.propagate_tracks(frames, tracks, H=1)
    (c,) = cands[2]
    truth = local_box(car_at(2), Ms[2])
    assert np.allclose(c['box'][:3], truth[:3], atol=1e-6)
    # extrapolation by constant velocity
    frames[0] = P.Frame('s_000', 0.0, Ms[0], np.zeros((0, 9)))
    frames[2] = P.Frame('s_002', 2.0, Ms[2], np.array([row(local_box(car_at(2), Ms[2]), 1, 0.9)]))
    tracks = P.cv_tracks(frames, 0.5, set(), d_init=10.0, d_cv=1.0)
    (c,) = P.propagate_tracks(frames, tracks, H=1)[0]
    assert np.allclose(c['box'][:3], local_box(car_at(0), Ms[0])[:3], atol=1e-6)


def test_cv_tracker_needs_three_observations():
    Ms = ego_poses(2)
    frames = [P.Frame(f'x_{i:03d}', float(i), M, np.array([row(local_box([5 + 5 * i, 0, 0.8, 4, 2, 1.5, 0], M), 1, .9)]))
              for i, M in enumerate(Ms)]
    assert P.cv_tracks(frames, 0.5, set(), 10.0, 1.0) == []


def test_empty_frames_and_one_frame_sequence():
    empty = [P.Frame(f'e_{i:03d}', float(i), np.eye(4), np.zeros((0, 9))) for i in range(3)]
    cl, mem = P.static_clusters(empty, 0.5, 1.0)
    assert cl == [] and mem == {}
    assert P.propagate_static(empty, cl, 5) == {} and P.cv_tracks(empty, 0.5, set(), 10, 1) == []
    assert P.dedup([], np.zeros((0, 7))) == ([], 0, 0)
    one = [P.Frame('o_000', 0.0, np.eye(4), np.array([row([10, 0, 0.8, 4, 2, 1.5, 0], 1, 0.9)]))]
    cl, _ = P.static_clusters(one, 0.5, 1.0)
    assert len(cl) == 1 and not cl[0]['static']
    assert P.propagate_static(one, cl, 5) == {} and P.cv_tracks(one, 0.5, set(), 10, 1) == []
    assert len(P.count_nonground(np.zeros((0, 3)), np.array([[0, 0, 0, 4, 2, 1.5, 0]]))) == 1


def test_dedup_against_any_direct_row_and_among_candidates():
    box = np.array([20.0, 0, 0.8, 4.5, 1.9, 1.6, 0])
    cands = [{'box': box, 'score': 0.9, 'delta': 2.0, 'src': 0},
             {'box': box + [0.3, 0, 0, 0, 0, 0, 0], 'score': 0.95, 'delta': 1.0, 'src': 1},
             {'box': box + [0, 10, 0, 0, 0, 0, 0], 'score': 0.9, 'delta': 1.0, 'src': 2},
             {'box': box + [0, 20, 0, 0, 0, 0, 0], 'score': 0.9, 'delta': 4.0, 'src': 3}]
    existing = np.array([row(box + [0, 10.2, 0, 0, 0, 0, 0], -2, 0.25)])  # a Pedestrian ignore-band row
    added, de, dp = P.dedup(cands, existing[:, :7], H=3)
    assert de == 1 and dp == 1
    assert [a['src'] for a in added] == [1]  # the smaller Delta wins; src 3 is beyond H


def test_count_nonground_excludes_ground_and_survives_a_z_error():
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(-8, 8, 0.1), np.arange(-5, 5, 0.1))
    ground = np.stack([gx.ravel(), gy.ravel(), rng.normal(0, 0.02, gx.size)], 1)
    body = np.stack([rng.uniform(-2, 2, 400), rng.uniform(-0.9, 0.9, 400), rng.uniform(0.5, 1.5, 400)], 1)
    pts = np.concatenate([ground, body])
    box = np.array([[0, 0, 0.8, 4.5, 1.9, 1.6, 0]])
    assert P.count_nonground(pts, box)[0] == 400
    assert P.count_nonground(pts, box + [0, 0, 1.0, 0, 0, 0, 0])[0] == 400  # propagated z off by 1 m
    assert P.count_nonground(ground, box)[0] == 0


def test_k_table_and_classify():
    counts = np.array([0, 3, 5, 7, 9] * 10 + [100] * 5)
    ranges = np.array([5.0] * 50 + [60.0] * 5)
    k0 = P.k_table(counts, ranges, 0.0)
    assert k0[0] == 1.0  # min 0 -> max(1, 0)
    assert k0[5] == 1.0  # 5 boxes < 20 -> pooled minimum
    k3 = P.k_table(counts, ranges, 0.3)
    assert k3[0] == 3.0
    cls = P.classify(np.array([0, 2, 3, 50]), np.array([5.0, 5, 5, 60]), k3)
    assert list(cls) == ['empty', 'ign', 'pos', 'pos']


def test_merge_entry_keeps_original_rows_and_signs():
    gt = np.array([row([1, 2, 0, 4, 2, 1.5, 0], 1, 0.5), row([5, 2, 0, .6, .6, 1.7, 0], -2, 0.25)], dtype=np.float32)
    entry = {'gt_boxes': gt, 'cls_scores': gt[:, 8].copy(), 'iou_scores': gt[:, 8].copy(),
             'memory_counter': np.zeros(2), 'teacher_classes': np.array(['Car', 'Ignored'])}
    add = np.array([[30, 0, 0.8, 4, 2, 1.5, 0], [40, 0, 0.8, 4, 2, 1.5, 0]])
    out = P.merge_entry(entry, add, [1, -1], [0.9, 0.8])
    assert np.array_equal(out['gt_boxes'][:2], gt) and out['gt_boxes'].dtype == gt.dtype
    assert list(out['gt_boxes'][:, 7]) == [1, -2, 1, -1]
    for k in ('cls_scores', 'iou_scores', 'memory_counter', 'teacher_classes'):
        assert len(out[k]) == 4
    assert list(out['teacher_classes']) == ['Car', 'Ignored', 'Car', 'Ignored']
    assert np.array_equal(entry['gt_boxes'], gt) and len(entry['gt_boxes']) == 2  # input untouched
    with pytest.raises(AssertionError):
        P.merge_entry(entry, add[:1], [2], [0.9])


def test_fill_pseudo_labels_reads_a_merged_entry():
    from pcdet.datasets.dataset import DatasetTemplate
    gt = np.array([row([1, 2, 0, 4, 2, 1.5, 0], 1, 0.5), row([5, 2, 0, .6, .6, 1.7, 0], -2, 0.25)], dtype=np.float32)
    entry = {'gt_boxes': gt, 'cls_scores': gt[:, 8], 'iou_scores': gt[:, 8], 'memory_counter': np.zeros(2),
             'teacher_classes': np.array(['Car', 'Ignored'])}
    merged = P.merge_entry(entry, np.array([[30, 0, 0.8, 4, 2, 1.5, 0], [40, 0, 0.8, 4, 2, 1.5, 0]]), [1, -1], [.9, .8])

    class Fake:
        class_names = ['Car', 'Pedestrian', 'Cyclist']
        pseudo_labels = {'f': merged}
    d = {'frame_id': 'f', 'num_points_in_gt': np.ones(4)}
    DatasetTemplate.fill_pseudo_labels(Fake(), d)
    assert list(d['gt_names']) == ['Car', 'Pedestrian', 'Car', 'Car']
    assert list(d['gt_classes']) == [1, -2, 1, -1]
    assert list(d['pos_ps_bbox']) == [2, 0, 0] and list(d['ign_ps_bbox']) == [1, 1, 0]
    assert 'num_points_in_gt' not in d  # recounted downstream (MIN_POINTS_OF_GT)


def test_label_free_waymo_loader_strips_annos(tmp_path, monkeypatch):
    W = _load('pseudo_label_propagation_waymo')
    seq = 'segment-x_with_camera_labels'
    (tmp_path / seq).mkdir()
    Ms = ego_poses(3)
    infos = [{'frame_id': f'{seq}_{10 * i:03d}', 'pose': M, 'point_cloud': {'lidar_sequence': seq, 'sample_idx': 10 * i},
              'annos': {'name': np.array(['Vehicle']), 'gt_boxes_lidar': np.zeros((1, 7))}} for i, M in enumerate(Ms)]
    pickle.dump(infos, open(tmp_path / seq / f'{seq}.pkl', 'wb'))
    monkeypatch.setattr(W, 'WAYMO', tmp_path)
    lf = W.load_label_free_infos(seq)
    assert all('annos' not in v for v in lf.values()) and len(lf) == 3
    # end to end on the label-free path: a car confident in the last two frames reaches frame 0
    ps = {}
    for i, M in enumerate(Ms):
        rows = [row([-30, 0, 0, .6, .6, 1.7, 0], -2, 0.25)]
        if i >= 1:
            rows.append(row(local_box(CAR_G, M), 1, 0.9))
        ps[f'{seq}_{10 * i:03d}'] = {'gt_boxes': np.array(rows, dtype=np.float32)}
    calib = {'s_conf': 0.5, 'd_assoc': 0.5, 'd_init': 10.0, 'd_cv': 1.0, 'S': {'q': 0.0, 'H': 2}, 'D': {'q': 0.0, 'H': 1}}
    recs, cl, _ = W.process_sequence(seq, list(ps), ps, calib, with_points=False)
    assert len(cl) == 1 and len(recs) == 3
    assert len(recs[0]['S_box']) == 1 and len(recs[1]['S_box']) == 0
    assert np.allclose(recs[0]['S_box'][0, :3], local_box(CAR_G, Ms[0])[:3], atol=1e-6)
    assert len(recs[0]['D_box']) == 0


def test_waymo_get_lidar_filters_nlz_and_range(tmp_path):
    pf = np.array([[1, 2, 0.5, 0.3, 0.1, -1], [3, 4, 0.5, 0.3, 0.1, 1], [80, 0, 0.5, 0.3, 0.1, -1],
                   [5, 5, 5.0, 0.3, 0.1, -1]], dtype=np.float32)
    np.save(tmp_path / 'f.npy', pf)
    pts = P.waymo_get_lidar(tmp_path / 'f.npy')
    assert pts.shape == (1, 5) and np.allclose(pts[0, :3], [1, 2, 0.5]) and pts[0, 3] == pytest.approx(np.tanh(0.3))
