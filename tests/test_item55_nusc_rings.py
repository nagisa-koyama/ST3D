"""Item 55's capture of the nuScenes kitti conversion, ring view and readings (experiments_md 20261003_05 §5 item 55, design fixed
2026-11-26 22:53 JST). CPU only, no files."""
import sys
import types
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item55_nusc_rings as m  # noqa: E402


def kitti_anno(names, xy, **extra):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    loc = np.zeros((len(xy), 3)); loc[:, 0] = -xy[:, 1]; loc[:, 2] = xy[:, 0]   # (-y_lidar, -z_lidar, x_lidar)
    return dict(name=np.array(names), location=loc, **extra)


def test_range_comes_from_the_converted_location():
    a = kitti_anno(['Car', 'Car'], [[3, 4], [-30, 40]])
    assert m.xy_range(a).tolist() == [5.0, 50.0]


def test_ring_view_relabels_gt_and_drops_detections_outside():
    gt = kitti_anno(['Car', 'Car', 'Truck'], [[5, 0], [35, 0], [36, 0]])
    dt = kitti_anno(['Car', 'Car'], [[34, 0], [12, 0]], score=np.array([0.9, 0.4]))
    g, d = m.ring_view(gt, dt, 30.0, 40.0)
    assert g['name'].tolist() == ['Van', 'Car', 'Truck']
    assert d['score'].tolist() == [0.9] and len(d['location']) == 1
    g, d = m.ring_view(gt, dt, 40.0, float('inf'))
    assert g['name'].tolist() == ['Van', 'Van', 'Truck'] and len(d['name']) == 0


def test_capture_takes_what_kitti_eval_hands_the_evaluator_and_restores_it(monkeypatch):
    name = 'fake_kitti_eval_for_item55'
    mod = types.ModuleType(name)
    original = lambda **kw: ('real', {})  # noqa: E731
    mod.get_official_eval_result = original
    monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(m, '_EVAL_MODULE', name)

    class FakeDataset:
        infos = [{'gt_names': np.array(['car'])}]

        def kitti_eval(self, det, gt, class_names):
            for a in det + gt:
                a['converted'] = True
            return sys.modules[name].get_official_eval_result(gt_annos=gt, dt_annos=det, current_classes=class_names)

    det = [{'name': np.array(['Car'])}]
    gt, dt = m.capture(FakeDataset(), det, ['Car'])
    assert gt[0]['converted'] and dt[0]['converted'] and 'converted' not in det[0]
    assert mod.get_official_eval_result is original


def test_readings_name_the_gain():
    def row(part, r, seed, rings):
        return {'part': part, 'row': r, 'seed': seed, 'stats': 'none', 'reproduced': True, 'rings': [{'bev': x} for x in rings]}
    rs = []
    for seed in (666, 667):
        rs += [row('BW', 'P', seed, [80, 70, 50, 30, 15]), row('BW', 'C1', seed, [79, 69, 45, 20, 8]),
               row('BW', 'L', seed, [72, 63, 45, 27, 13.5])]
    assert m.reading(rs, 'BW', 'C1', 'none')[1] == 'gain far'
    assert m.reading(rs, 'BW', 'L', 'none')[1] == 'over the whole range'
    rs[1]['reproduced'] = False
    assert m.reading(rs, 'BW', 'C1', 'none')[1] == 'not read'
