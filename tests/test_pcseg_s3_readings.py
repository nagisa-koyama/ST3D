"""tools/analysis/pcseg_s3_readings.py (experiments_md episodes/20261116_01 §8.10): the declared 16-class cone mIoU, the
shard sum, the in-run lookup, and the seed statistics behind the term readings.
"""
import importlib.util
import json
import os

import numpy as np
import pytest

from pcseg.losses_metrics import ConfusionMatrix

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ['road', 'car', 'barrier', 'bus']


def _mod():
    spec = importlib.util.spec_from_file_location(
        'pcseg_s3_readings', os.path.join(REPO, 'tools', 'analysis', 'pcseg_s3_readings.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _mat():
    # rows ground truth, columns prediction; barrier has no ground truth but is predicted on 3 points
    return np.array([[50, 2, 3, 0],
                     [4, 30, 0, 1],
                     [0, 0, 0, 0],
                     [0, 5, 0, 10]], dtype=np.int64)


def test_cone16_drops_barrier_and_matches_the_in_run_require_gt_mean():
    m = _mod()
    mat = _mat()
    cm = ConfusionMatrix(len(NAMES))
    cm.mat = mat
    iou = cm.iou()
    want = np.mean([iou[0], iou[1], iou[3]])
    assert m.cone16(mat, NAMES) == pytest.approx(want, abs=1e-15)
    assert m.cone16(mat, NAMES) == pytest.approx(cm.miou(require_gt=True), abs=1e-15)   # barrier has no ground truth
    # an all-class mean would count barrier's IoU (0 here) and come out lower
    assert m.cone16(mat, NAMES) > float(np.mean(np.nan_to_num(iou)))


def test_cone16_raises_when_a_declared_class_lacks_ground_truth_or_the_shape_is_wrong():
    m = _mod()
    mat = _mat()
    mat[3] = 0                                                        # bus loses its ground truth
    with pytest.raises(ValueError, match='bus'):
        m.cone16(mat, NAMES)
    with pytest.raises(ValueError):
        m.cone16(_mat()[:3, :3], NAMES)
    with pytest.raises(ValueError):
        m.cone16(_mat(), ['road', 'car', 'truck', 'bus'])            # no barrier to drop


def test_per_class_covers_the_declared_classes_and_averages_to_cone16():
    m = _mod()
    pc = m.per_class(_mat(), NAMES)
    assert sorted(pc) == ['bus', 'car', 'road']
    assert np.mean(list(pc.values())) == pytest.approx(m.cone16(_mat(), NAMES), abs=1e-15)


def _shard(d, i, n, mat, ckpt='/x/run-1-abc/files/ckpt/checkpoint_epoch_391.pth'):
    os.makedirs(d, exist_ok=True)
    rec = {'cfg_file': 'cfgs/pcseg/spunet-pandasetgt.yaml', 'ckpt': ckpt, 'epoch': 390, 'shard': i, 'num_shards': n,
           'frames_in_shard': 10, 'commit': 'c0ffee', 'plain': {'matrices': {'cone': mat.tolist()}}}
    json.dump(rec, open(os.path.join(d, 'shard_%03d_of_%03d.json' % (i, n)), 'w'))


def test_load_test_seg_sums_shards_and_refuses_an_incomplete_or_mixed_set(tmp_path):
    m = _mod()
    d = str(tmp_path / 'two')
    _shard(d, 0, 2, _mat())
    _shard(d, 1, 2, 2 * _mat())
    t = m.load_test_seg(d)
    assert (t['cone'] == 3 * _mat()).all() and t['frames'] == 20
    assert m.load_test_seg(str(tmp_path / 'absent')) is None
    d = str(tmp_path / 'half')
    _shard(d, 0, 2, _mat())
    with pytest.raises(ValueError):
        m.load_test_seg(d)
    d = str(tmp_path / 'mixed')
    _shard(d, 0, 2, _mat())
    _shard(d, 1, 2, _mat(), ckpt='/x/run-2-def/files/ckpt/checkpoint_epoch_391.pth')
    with pytest.raises(ValueError):
        m.load_test_seg(d)


def test_load_inrun_finds_the_eval_beside_the_checkpoint(tmp_path):
    m = _mod()
    files = tmp_path / 'run-20261122_223805-tcygv7mi' / 'files'
    (files / 'ckpt').mkdir(parents=True)
    json.dump({'matrices': {'cone': _mat().tolist()}}, open(files / 'eval_epoch_391.json', 'w'))
    r = m.load_inrun(str(files / 'ckpt' / 'checkpoint_epoch_391.pth'))
    assert r['run'] == 'tcygv7mi' and (r['cone'] == _mat()).all()
    assert m.load_inrun(str(files / 'ckpt' / 'checkpoint_epoch_390.pth')) is None


def test_summary_readability_needs_two_points_and_one_sign_over_all_seeds():
    m = _mod()
    s = m.summary([3.0, 2.5, 2.0])
    assert s['complete'] and s['readable'] and s['mean'] == pytest.approx(2.5)
    assert s['sd'] == pytest.approx(float(np.std([3.0, 2.5, 2.0], ddof=1)))
    assert not m.summary([5.0, 2.0, -0.5])['readable']               # one seed has the other sign
    assert not m.summary([1.9, 2.0, 2.0])['readable']                 # |mean| < 2
    assert m.summary([-3.0, -2.0, -2.5])['readable']                  # negative terms are readable too
    assert not m.summary([2.0, None, 3.0])['complete']                # a missing seed is never filled in


def test_diff_is_in_points_and_propagates_missing():
    m = _mod()
    assert m.diff(0.5248, 0.2301) == pytest.approx(29.47)
    assert m.diff(None, 0.2) is None and m.diff(0.2, None) is None


def test_layout_matches_the_declared_output_paths():
    m = _mod()
    assert m.cross_dir(666, 'gt', '64').endswith('test_seg/p1_cross_391/model_gt_on_pandaset64')
    assert m.cross_dir(667, '64', 'gt').endswith('test_seg/p1_cross_391_s667/model_64_on_pandasetgt')
    assert m.arm_dir('b0', 668, 'spunet-p1-b1-lines-reliance').endswith('s3_b0_s668/spunet-p1-b1-lines-reliance')
    assert m.reliance_dir('gt', 666, 'spunet-p1-a1-density-reliance').endswith(
        's3_reliance/oracle_gt_s666/spunet-p1-a1-density-reliance')


def test_corrections_compare_each_intensity_arm_with_the_source_only_model_on_the_other_val():
    """§8.11: a4 trains on Pandar64 with PandarGT's response, so it is scored on PandarGT val against the Pandar64 oracle
    there, and its share is of the Pandar64 -> PandarGT gap; a2 the other way round."""
    m = _mod()
    assert m.CORRECTIONS == {'intensity_a4_on_gt': ('a4', 'gt', '64', 'gap_64_to_gt'),
                             'intensity_a2_on_64': ('a2', '64', 'gt', 'gap_gt_to_64')}
    assert m.arm_dir('a4', 667, m.VAL['gt']).endswith('s3_a4_s667/spunet-pandasetgt')
    assert m.arm_dir('a2', 668, m.VAL['64']).endswith('s3_a2_s668/spunet-pandaset64')
    assert m.cross_dir(667, '64', 'gt').endswith('p1_cross_391_s667/model_64_on_pandasetgt')


def test_v1_layout_reads_b0_as_its_identity_arm():
    """§8.11: term_v1 = b0 - v1 on PandarGT val, the reliance on spunet-p1-v1-viewpoint-reliance."""
    m = _mod()
    assert m.arm_dir('v1', 666, m.VAL['gt']).endswith('s3_v1_s666/spunet-pandasetgt')
    assert m.arm_dir('b0', 667, 'spunet-p1-v1-viewpoint-reliance').endswith('s3_b0_s667/spunet-p1-v1-viewpoint-reliance')
    src = open(os.path.join(REPO, 'tools', 'analysis', 'pcseg_s3_readings.py')).read()
    assert "'term_v1': ([diff(a[('b0', s, 'gt')], a[('v1', s, 'gt')])" in src
    assert "'v1_correction': summary([diff(a[('v1', s, '64')], a[('b0', s, '64')])" in src
