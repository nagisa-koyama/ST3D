"""tools/analysis/nuscenes_native_eval.py: the class renaming, the label sentinels it must refuse, the 500-box cap,
the summary / NDS-det arithmetic, the log parser, and the LiDAR -> global box conversion it reuses from
pcdet.datasets.nuscenes.nuscenes_utils. CPU only; no nuScenes tables are read (experiments_md 20261017_12).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

CLASSES = ['Car', 'Pedestrian', 'Cyclist']


def det(labels, names=None, scores=None, boxes=None, token='tok0'):
    labels = np.asarray(labels)
    n = len(labels)
    if names is None:
        names = np.array(CLASSES)[labels - 1]
    if boxes is None:
        boxes = np.tile(np.array([10.0, 2.0, -0.5, 4.5, 1.9, 1.6, 0.3], np.float32), (n, 1))
    return {'name': np.asarray(names), 'score': np.asarray(scores if scores is not None else np.full(n, 0.5),
                                                           np.float32),
            'boxes_lidar': np.asarray(boxes, np.float32), 'pred_labels': labels, 'frame_id': 'f0',
            'metadata': {'token': token}}


def test_name_map_for_the_kitti_vocabulary():
    from nuscenes_native_eval import model_to_nusc_names
    assert model_to_nusc_names(CLASSES) == {'Car': 'car', 'Pedestrian': 'pedestrian', 'Cyclist': 'bicycle'}


@pytest.mark.parametrize('names', [['nuscenes:car'], ['Bus'], ['Misc'], ['Car', 'Van']])
def test_name_map_refuses_prefixed_unknown_out_of_vocabulary_and_merged(names):
    from nuscenes_native_eval import model_to_nusc_names
    with pytest.raises(ValueError):
        model_to_nusc_names(names)  # prefixed; no mapping; 'debris' is not a detection name; two -> 'car'


def test_rename_keeps_boxes_and_casts_for_json():
    from nuscenes_native_eval import check_and_rename, model_to_nusc_names
    d = det([1, 2, 3])
    out, counts = check_and_rename([d], CLASSES, model_to_nusc_names(CLASSES))
    assert list(out[0]['name']) == ['car', 'pedestrian', 'bicycle']
    assert out[0]['score'].dtype == np.float64 and out[0]['boxes_lidar'].dtype == np.float64
    np.testing.assert_allclose(out[0]['boxes_lidar'], d['boxes_lidar'])
    assert list(d['name']) == ['Car', 'Pedestrian', 'Cyclist']  # input untouched
    assert counts == {'samples': 1, 'boxes': 3, 'samples_capped_at_500': 0}


@pytest.mark.parametrize('bad', [0, -1, -3, -99])
def test_refuses_sentinel_labels_even_when_the_wrapped_name_looks_valid(bad):
    """A 0 / negative label wrapped when the result was written: class_names[-2] == 'Pedestrian' etc."""
    from nuscenes_native_eval import check_and_rename, model_to_nusc_names
    wrapped = np.array(CLASSES)[np.array([bad]) - 1] if bad - 1 >= -len(CLASSES) else np.array(['Car'])
    with pytest.raises(ValueError):
        check_and_rename([det([bad], names=wrapped)], CLASSES, model_to_nusc_names(CLASSES))


def test_refuses_label_above_range_and_name_label_disagreement():
    from nuscenes_native_eval import check_and_rename, model_to_nusc_names
    m = model_to_nusc_names(CLASSES)
    with pytest.raises(ValueError):
        check_and_rename([det([4], names=['Car'])], CLASSES, m)
    with pytest.raises(ValueError):
        check_and_rename([det([1], names=['Pedestrian'])], CLASSES, m)
    with pytest.raises(ValueError):
        check_and_rename([det([1.5], names=['Car'])], CLASSES, m)


@pytest.mark.parametrize('mutate', ['eight_columns', 'nan_box', 'zero_size', 'nan_score', 'no_token'])
def test_refuses_malformed_boxes(mutate):
    from nuscenes_native_eval import check_and_rename, model_to_nusc_names
    d = det([1, 1])
    if mutate == 'eight_columns':
        d['boxes_lidar'] = np.zeros((2, 8), np.float32) + 1
    elif mutate == 'nan_box':
        d['boxes_lidar'][0, 0] = np.nan
    elif mutate == 'zero_size':
        d['boxes_lidar'][1, 4] = 0.0
    elif mutate == 'nan_score':
        d['score'][0] = np.nan
    else:
        d['metadata'] = {}
    with pytest.raises(ValueError):
        check_and_rename([d], CLASSES, model_to_nusc_names(CLASSES))


def test_cap_keeps_the_500_highest_scores_and_empty_samples_pass():
    from nuscenes_native_eval import check_and_rename, model_to_nusc_names
    rng = np.random.default_rng(0)
    scores = rng.random(600)
    big = det(np.ones(600, np.int64), scores=scores)
    empty = det(np.zeros(0, np.int64), names=np.zeros(0), token='tok1')
    empty['boxes_lidar'] = np.zeros((0, 7))
    out, counts = check_and_rename([big, empty], CLASSES, model_to_nusc_names(CLASSES))
    assert len(out[0]['score']) == 500 and len(out[1]['score']) == 0
    np.testing.assert_allclose(np.sort(out[0]['score']), np.sort(scores.astype(np.float32))[-500:])
    assert counts == {'samples': 2, 'boxes': 500, 'samples_capped_at_500': 1}


def _metrics(aps, tps):
    names = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
             'traffic_cone', 'barrier']
    m = {'mean_dist_aps': {}, 'label_aps': {}, 'label_tp_errors': {}, 'mean_ap': 0.123, 'nd_score': 0.2,
         'tp_errors': {'trans_err': 0.9}}
    for c in names:
        a = aps.get(c, 0.0)
        m['mean_dist_aps'][c] = a
        m['label_aps'][c] = {'0.5': a, '1.0': a, '2.0': a, '4.0': a}
        e = tps.get(c, (1.0, 1.0, 1.0))
        m['label_tp_errors'][c] = {'trans_err': e[0], 'scale_err': e[1], 'orient_err': e[2], 'vel_err': 1.0,
                                   'attr_err': 1.0}
    return m


def test_summary_is_over_the_model_classes_only_and_nds_det_clips_the_mean():
    from nuscenes_native_eval import summarise
    m = _metrics({'car': 0.5, 'pedestrian': 0.2, 'truck': 0.9},
                 {'car': (0.3, 0.2, 0.4), 'pedestrian': (0.5, 0.3, 1.2), 'truck': (0.0, 0.0, 0.0)})
    s = summarise(m, ['car', 'pedestrian', 'bicycle'])
    assert s['per_class']['car']['AP'] == pytest.approx(50.0)
    assert s['mAP_k'] == pytest.approx(100 * 0.7 / 3)  # truck's 0.9 is not in it
    assert s['mATE_k'] == pytest.approx(0.6) and s['mASE_k'] == pytest.approx(0.5)
    assert s['mAOE_k'] == pytest.approx((0.4 + 1.2 + 1.0) / 3)
    expect = (5 * 0.7 / 3 + (1 - 0.6) + (1 - 0.5) + (1 - (0.4 + 1.2 + 1.0) / 3)) / 8
    assert s['NDS_det'] == pytest.approx(100 * expect)
    worst = summarise(_metrics({'car': 0.4}, {'car': (2.0, 3.0, 4.0)}), ['car'])
    assert worst['NDS_det'] == pytest.approx(100 * 5 * 0.4 / 8)  # each mean error > 1 contributes 0
    assert worst['devkit_10class_not_reported']['mAP'] == pytest.approx(12.3)


def test_log_parser_picks_the_requested_epoch():
    from nuscenes_native_eval import parse_logged_car_bev

    def block(ep, v):
        return (f"[t eval_utils.py 46 INFO]  *************** EPOCH {ep} of nuscenes EVALUATION start.****\r"
                f"eval:  50%|#####| 3/6\reval: 100%|##########| 6/6\n"
                f"[t eval_utils.py 189 INFO]  Car AP@0.70, 0.70, 0.70:\nbbox AP:1, 1, 1\nbev  AP:9, 9, 9\n"
                f"Car AP_R40@0.70, 0.70, 0.70:\nbbox AP:{v}, {v}, {v}\nbev  AP:{v}, {v}, {v}\n3d   AP:1, 1, 1\n"
                f"[t eval_utils.py 195 INFO]  Result is save to /storage/wandb/run-x/files/eval/eval_with_train/"
                f"epoch_{ep}/val\n")
    text = block(11, 1.0) + block(115, 33.3943) + block(29, 2.0)
    vals, res = parse_logged_car_bev(text, 115)
    assert vals == [33.3943] * 3 and res.endswith('epoch_115/val')
    assert parse_logged_car_bev(text, 11)[0] == [1.0] * 3
    with pytest.raises(ValueError):
        parse_logged_car_bev(text, 30)


def test_numpy_float_alias_is_the_builtin_and_changes_nothing():
    from nuscenes_native_eval import numpy_float_alias
    assert numpy_float_alias() is float
    x = np.cumsum(np.array([1, 0, 1, 1]))
    assert x.astype(np.float).dtype == np.float64 and np.array_equal(x.astype(np.float), x.astype(np.float64))


class FakeNusc:
    """Just the records lidar_nusc_box_to_global reads."""

    def __init__(self, cs_q, cs_t, pose_q, pose_t):
        self.recs = {
            ('sample', 'tok0'): {'data': {'LIDAR_TOP': 'sd0'}},
            ('sample_data', 'sd0'): {'calibrated_sensor_token': 'cs0', 'ego_pose_token': 'ep0'},
            ('calibrated_sensor', 'cs0'): {'rotation': list(cs_q.elements), 'translation': list(cs_t),
                                           'sensor_token': 's0'},
            ('sensor', 's0'): {},
            ('ego_pose', 'ep0'): {'rotation': list(pose_q.elements), 'translation': list(pose_t)},
        }

    def get(self, table, token):
        return self.recs[(table, token)]

    def get_sample_data_path(self, token):
        return '/nonexistent'


def test_lidar_to_global_conversion_against_an_independent_transform():
    """Pins the size order (wlh = dy, dx, dz), the yaw and the two rigid transforms, and that the record dumps."""
    from nuscenes_native_eval import check_and_rename, model_to_nusc_names
    from pcdet.datasets.nuscenes import nuscenes_utils
    cs_q, cs_t = Quaternion(axis=[0, 0, 1], radians=-1.5), np.array([0.94, 0.0, 1.84])
    pose_q, pose_t = Quaternion(axis=[0, 0, 1], radians=2.1), np.array([400.0, 1100.0, 0.0])
    box = np.array([[12.0, -3.0, -0.8, 4.6, 1.95, 1.7, 0.4]], np.float32)
    d = det([1], boxes=box, scores=[0.75])
    renamed, _ = check_and_rename([d], CLASSES, model_to_nusc_names(CLASSES))
    annos = nuscenes_utils.transform_det_annos_to_nusc_annos(renamed, FakeNusc(cs_q, cs_t, pose_q, pose_t))
    rec = annos['results']['tok0'][0]
    json.dumps(annos)  # float32 scores would raise here
    c = box[0, :3].astype(np.float64)
    expect_c = pose_q.rotation_matrix @ (cs_q.rotation_matrix @ c + cs_t) + pose_t
    np.testing.assert_allclose(rec['translation'], expect_c, atol=1e-4)
    np.testing.assert_allclose(rec['size'], [1.95, 4.6, 1.7], atol=1e-5)  # w, l, h
    yaw = Quaternion(rec['rotation']).yaw_pitch_roll[0]
    assert np.isclose(np.angle(np.exp(1j * (yaw - (0.4 - 1.5 + 2.1)))), 0.0, atol=1e-5)
    assert rec['detection_name'] == 'car' and rec['velocity'] == [0.0, 0.0]
    assert rec['attribute_name'] == 'vehicle.parked'  # zero velocity -> the class prior
    assert rec['detection_score'] == pytest.approx(0.75)
