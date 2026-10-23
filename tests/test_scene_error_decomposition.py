"""CPU tests of tools/analysis/kitti_frame_decomposition.py (ENV E1 / E2, experiments_md 20261024_02).

On synthetic boxes in the KITTI format the nuScenes evaluation builds (fabricated 2D boxes, all GT boxes kept):
1. BEV-IoU categorisation: a TP, a duplicate, a mislocalised detection, a background detection and a missed GT land
   in the categories the design defines (background < 0.1 <= localisation, duplicates included; missed = no
   detection at IoU >= 0.1).
2. AP with nothing removed equals the official evaluator's AP exactly, over all frames and over a multiset of frames.
3. AP with background FPs removed / missed GT removed equals the official evaluator on annos with those boxes deleted.
4. Removing localisation FPs (which here score high) raises AP; the step counts equal compute_statistics_jit.
5. The dataset's own NuScenesDataset.kitti_eval builds the annos the capture shim hands over (class map, every GT).
6. The fixed reading of E1 is applied literally.
Run from the repo root inside the container: python3 -m pytest tests/test_scene_error_decomposition.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import kitti_frame_decomposition as kfd  # noqa: E402


@pytest.fixture(scope='module')
def official():
    return kfd.load_official_eval()


def lidar_to_kitti(boxes, names, scores=None):
    """The nuScenes kitti_eval transform (transform_to_kitti_format), for synthetic lidar boxes."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 7).copy()
    n = len(boxes)
    anno = {'name': np.array(names, dtype='<U16'), 'bbox': np.zeros((n, 4)), 'truncated': np.zeros(n),
            'occluded': np.zeros(n)}
    anno['bbox'][:, 2:4] = 50
    b = boxes.copy()
    b[:, 2] -= b[:, 5] / 2
    anno['location'] = np.stack([-b[:, 1], -b[:, 2], b[:, 0]], 1) if n else np.zeros((0, 3))
    anno['dimensions'] = b[:, 3:6][:, [0, 2, 1]] if n else np.zeros((0, 3))
    anno['rotation_y'] = -b[:, 6] - np.pi / 2.0
    anno['alpha'] = -np.arctan2(-b[:, 1], b[:, 0]) + anno['rotation_y']
    if scores is not None:
        anno['score'] = np.asarray(scores, np.float32)
    return anno


def car(x, y, l=4.5, w=1.9, yaw=0.0):
    return [x, y, 0.0, l, w, 1.6, yaw]


def make_frames():
    """Three frames. Frame 0 holds one example of every category."""
    gts, dts = [], []
    # frame 0: GT A (TP + duplicate), GT B (mislocalised det only), GT C (missed), a pedestrian GT;
    #          detections: TP of A (0.9), duplicate of A (0.8, shifted 0.25 m), mislocalised B (0.85, shifted 2 m),
    #          background far away (0.95), a pedestrian detection (other class)
    gts.append(kfd_anno([car(10, 0), car(20, 5), car(40, -8), [5, 5, 0, 0.8, 0.8, 1.8, 0]],
                        ['Car', 'Car', 'Car', 'Pedestrian']))
    dts.append(kfd_anno([car(10, 0), car(10.25, 0), car(22.0, 5), car(60, 30), [5, 5, 0, 0.8, 0.8, 1.8, 0]],
                        ['Car', 'Car', 'Car', 'Car', 'Pedestrian'], [0.9, 0.8, 0.85, 0.95, 0.7]))
    # frame 1: two GT, two TPs at low scores, one background at a mid score
    gts.append(kfd_anno([car(15, 2), car(30, -3)], ['Car', 'Car']))
    dts.append(kfd_anno([car(15.1, 2), car(30, -3.05), car(-20, -20)], ['Car', 'Car', 'Car'], [0.4, 0.3, 0.5]))
    # frame 2: no GT car, one background detection
    gts.append(kfd_anno([[0, 8, 0, 0.8, 0.8, 1.8, 0]], ['Pedestrian']))
    dts.append(kfd_anno([car(25, 25)], ['Car'], [0.6]))
    return gts, dts


def kfd_anno(boxes, names, scores=None):
    return lidar_to_kitti(boxes, names, scores)


def test_categories(official):
    gts, dts = make_frames()
    fr = kfd.CarBevFrames(gts, dts, official)
    # GT records (cars only): frame 0 boxes 0, 1, 2; frame 1 boxes 0, 1
    assert list(fr.gt['frame']) == [0, 0, 0, 1, 1]
    missed = fr.gt['best_iou'] < kfd.CATEGORY_IOU
    assert list(missed) == [False, False, True, False, False]
    assert list(fr.n_missed) == [1, 0, 0]
    assert list(fr.gt['matched07']) == [True, False, False, True, True]
    # detection records (Car only, anno order): frame 0 TP, duplicate, mislocalised, background; frame 1; frame 2
    cat = fr.dt['category']
    assert list(cat[:4]) == [0, 2, 2, 1]
    assert fr.dt['max_iou'][1] > kfd.MIN_OVERLAP           # the duplicate overlaps its GT above 0.7
    assert kfd.CATEGORY_IOU <= fr.dt['max_iou'][2] < kfd.MIN_OVERLAP  # the mislocalised one between 0.1 and 0.7
    assert fr.dt['max_iou'][3] == 0.0
    assert list(cat[4:]) == [0, 0, 1, 1]
    assert fr.n_state_checks > 0


def test_ap_equals_official(official):
    gts, dts = make_frames()
    fr = kfd.CarBevFrames(gts, dts, official)
    ref = kfd.official_car_bev_ap(gts, dts, official)
    assert ref > 0
    assert fr.ap() == pytest.approx(ref, abs=1e-9)
    # a multiset of frames (a bootstrap resample) equals the official evaluator on the repeated annos
    pick = [0, 0, 1, 2, 1]
    ref2 = kfd.official_car_bev_ap([gts[i] for i in pick], [dts[i] for i in pick], official)
    assert fr.ap(pick) == pytest.approx(ref2, abs=1e-9)


def test_removals_equal_official_deletions(official):
    gts, dts = make_frames()
    fr = kfd.CarBevFrames(gts, dts, official)
    g, d = kfd.annos_without(gts, dts, fr, 'bkg')
    assert sum(len(a['name']) for a in d) == sum(len(a['name']) for a in dts) - 3
    assert fr.ap(drop='bkg') == pytest.approx(kfd.official_car_bev_ap(g, d, official), abs=1e-9)
    g, d = kfd.annos_without(gts, dts, fr, 'miss')
    assert sum(len(a['name']) for a in g) == sum(len(a['name']) for a in gts) - 1
    assert fr.ap(drop_missed=True) == pytest.approx(kfd.official_car_bev_ap(g, d, official), abs=1e-9)
    # the background and localisation FPs score high here, so removing them raises AP; removing the missed GT
    # cannot lower it (with 3 TPs the R40 sampling fills 3 slots whatever the GT count)
    base = fr.ap()
    assert fr.ap(drop='bkg') > base and fr.ap(drop='loc') > base and fr.ap(drop_missed=True) >= base


def random_frames(n_frames=150, seed=0):
    """Many frames: jittered TPs, mislocalised and duplicate detections, background, missed GT, other classes."""
    rng = np.random.default_rng(seed)
    gts, dts = [], []
    for _ in range(n_frames):
        n = rng.integers(0, 9)
        g = [car(rng.uniform(-60, 60), rng.uniform(-60, 60), yaw=rng.uniform(-np.pi, np.pi)) for _ in range(n)]
        names = ['Car'] * n
        d, dn, sc = [], [], []
        for b in g:
            r = rng.random()
            if r < 0.15:
                continue  # missed (unless a background box lands on it)
            jit = 0.1 if r < 0.75 else 1.5  # good box or mislocalised
            d.append([b[0] + rng.normal(0, jit), b[1] + rng.normal(0, jit), 0, b[3], b[4], b[5],
                      b[6] + rng.normal(0, 0.05)])
            dn.append('Car'); sc.append(rng.uniform(0.2, 1.0))
            if rng.random() < 0.2:  # duplicate
                d.append([b[0] + 0.2, b[1], 0, b[3], b[4], b[5], b[6]]); dn.append('Car'); sc.append(rng.uniform(0.05, 0.8))
        for _ in range(rng.integers(0, 6)):  # background
            d.append(car(rng.uniform(-70, 70), rng.uniform(-70, 70), yaw=rng.uniform(-np.pi, np.pi)))
            dn.append('Car'); sc.append(rng.uniform(0.05, 0.9))
        if rng.random() < 0.5:  # another class on both sides
            g.append([rng.uniform(-30, 30), rng.uniform(-30, 30), 0, 0.8, 0.8, 1.8, 0]); names.append('Pedestrian')
            d.append(g[-1]); dn.append('Pedestrian'); sc.append(0.5)
        gts.append(kfd_anno(g if g else np.zeros((0, 7)), names))
        dts.append(kfd_anno(d if d else np.zeros((0, 7)), dn, sc))
    return gts, dts


def test_random_frames_equal_official(official):
    gts, dts = random_frames()
    fr = kfd.CarBevFrames(gts, dts, official)
    assert fr.n_missed.sum() > 10 and (fr.dt['category'] == 1).sum() > 50 and (fr.dt['category'] == 2).sum() > 20
    assert fr.ap() == pytest.approx(kfd.official_car_bev_ap(gts, dts, official), abs=1e-9)
    for what, kw in (('bkg', dict(drop='bkg')), ('miss', dict(drop_missed=True))):
        g, d = kfd.annos_without(gts, dts, fr, what)
        assert fr.ap(**kw) == pytest.approx(kfd.official_car_bev_ap(g, d, official), abs=1e-9), what
    pick = np.random.default_rng(1).integers(0, len(gts), len(gts))
    ref = kfd.official_car_bev_ap([gts[i] for i in pick], [dts[i] for i in pick], official)
    assert fr.ap(pick) == pytest.approx(ref, abs=1e-9)
    assert fr.ap(drop='loc') > fr.ap()


def test_step_counts_match_official(official):
    gts, dts = make_frames()
    fr = kfd.CarBevFrames(gts, dts, official, verify=True)  # asserts at every distinct score internally
    # total TP / FP at the lowest threshold of frame 0: 1 TP, 3 FP (duplicate, mislocalised, background)
    sl = slice(fr.ev_offsets[0], fr.ev_offsets[1])
    assert fr.ev['tp'][sl].sum() == 1 and fr.ev['fp'][sl].sum() == 3
    assert fr.ev['fpb'][sl].sum() == 1 and fr.ev['fpl'][sl].sum() == 2


def test_capture_uses_the_dataset_kitti_eval(official):
    """NuScenesDataset.kitti_eval, run through the capture shim, hands over the annos this module consumes."""
    from easydict import EasyDict
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    ds = object.__new__(NuScenesDataset)
    ds.dataset_cfg = EasyDict({})
    ds.draw_conf_calib_curve = False
    ds.run_conf_calib = False
    infos = [{'token': 't0', 'gt_boxes': np.array([car(10, 0), car(20, 5)], np.float64),
              'gt_names': np.array(['car', 'truck']), 'num_lidar_pts': np.array([12, 3])}]
    dets = [{'name': np.array(['Car']), 'score': np.array([0.9], np.float32),
             'boxes_lidar': np.array([car(10, 0)], np.float32), 'pred_labels': np.array([1]),
             'frame_id': 't0', 'metadata': {'token': 't0'}}]
    captured = {}

    def capture(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
        captured.update(gt=gt_annos, dt=dt_annos)
        return '', {}

    import types
    shim = types.ModuleType(kfd._EVAL_MODULE)
    shim.get_official_eval_result = capture
    import copy
    with kfd._module_swapped(shim):
        ds.kitti_eval(copy.deepcopy(dets), copy.deepcopy(infos), ['Car', 'Pedestrian', 'Cyclist'])
    g, d = captured['gt'], captured['dt']
    assert list(g[0]['name']) == ['Car', 'Truck'] and len(g[0]['num_lidar_pts']) == 2
    fr = kfd.CarBevFrames(g, d, official)
    assert list(fr.gt['matched07']) == [True]  # the truck is not a GT car
    assert fr.ap() == pytest.approx(kfd.official_car_bev_ap(g, d, official), abs=1e-9)
    # the shim is gone afterwards
    import importlib
    pkg = importlib.import_module(kfd._PACKAGE)
    assert getattr(pkg, 'eval', None) is not shim and sys.modules.get(kfd._EVAL_MODULE) is not shim


def test_fixed_reading():
    from scene_error_decomposition import e1_reading
    assert e1_reading({'bkg': 5.0, 'miss': 1.0, 'loc': 2.4}) == 'background-led'
    assert e1_reading({'bkg': 0.5, 'miss': 4.0, 'loc': -1.0}) == 'foreground-led'
    assert e1_reading({'bkg': 0.5, 'miss': 1.0, 'loc': 3.0}) == 'localisation-led'
    assert e1_reading({'bkg': 3.0, 'miss': 2.5, 'loc': 0.0}) == 'mixed'
    assert e1_reading({'bkg': 1.9, 'miss': 1.0, 'loc': 0.0}) == 'unresolved'
    assert e1_reading({'bkg': 3.0, 'miss': 1.8, 'loc': 0.0}).startswith('not covered')
