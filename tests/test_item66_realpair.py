"""Item 66, the real-pair 2 x 2 (tools/analysis/item66_realpair.py; experiments_md 20261003_05 §5 item 66, design fixed
2026-11-27 20:55 JST). CPU only; run from the repo root."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item66_realpair as m  # noqa: E402

# nuScenes' LIDAR_TOP: x = -ego_y (right), y = ego_x (forward), 0.9 m ahead of the ego origin, 1.86 m up
REF_FROM_CAR = np.array([[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, -0.9], [0.0, 0.0, 1.0, -1.86], [0.0, 0.0, 0.0, 1.0]])
SECTOR = (np.deg2rad(41.0), np.deg2rad(40.0))                     # (right, left), about KITTI camera 2's


def test_ego_azimuth_turns_nuscenes_lidar_into_forward_left():
    c = np.array([[0.0, 10.0, 0.0], [-10.0, -0.9, 0.0], [10.0, -0.9, 0.0], [0.0, -10.0, 0.0]])   # ego (10.9, 0), (0, 10), ...
    az = np.rad2deg(m.ego_azimuth(c, REF_FROM_CAR))
    np.testing.assert_allclose(az, [0.0, 90.0, -90.0, 180.0], atol=1e-6)


def _info(boxes, names, npts):
    return {'gt_boxes': np.asarray(boxes, dtype=np.float64), 'gt_names': np.asarray(names),
            'num_lidar_pts': np.asarray(npts), 'ref_from_car': REF_FROM_CAR}


def test_frame_boxes_keeps_what_the_teacher_trained_with_inside_the_sector():
    from pcdet.utils.ontology_mapping import get_ontology_mapping
    nm = get_ontology_mapping('nuscenes', 'kitti')
    box = lambda x, y: [x, y, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]          # noqa: E731
    info = _info([box(0, 20), box(0, 25), box(5, 20), box(0, 30), box(0, 35), box(0, 40), box(-30, 0.9), box(0, -20),
                  box(-14, 20), box(-20, 20)],
                 ['car', 'car', 'pedestrian', 'truck', 'motorcycle', 'bicycle', 'car', 'car', 'car', 'car'],
                 [5, 0, 3, 9, 9, 2, 5, 5, 5, 5])
    b = m.frame_boxes(info, 1, nm, SECTOR)
    # kept: car ahead (5 pts), pedestrian ahead-right, bicycle -> Cyclist, car 14 m left at 20.9 m ahead (33.8 deg);
    # dropped: car with 0 points, truck and motorcycle (no model class), car at 86.6 deg left, car behind, car at 43.7 deg left
    assert b[:, 7].tolist() == [1, 2, 3, 1]
    np.testing.assert_allclose(b[:, :2], [[0, 20], [5, 20], [0, 40], [-14, 20]])
    assert m.frame_boxes(_info(np.zeros((0, 9)), [], []), 1, nm, SECTOR).shape == (0, 8)


def test_the_sector_is_asymmetric_about_the_principal_point():
    import item63_range_cuts as m63
    # W 1242, c_x 609.6, f_x 721.5: right atan(632.4 / 721.5) = 41.2 deg, left atan(609.6 / 721.5) = 40.2 deg
    r, l = np.arctan((1242 - 609.6) / 721.5), np.arctan(609.6 / 721.5)
    assert np.rad2deg(r) == pytest.approx(41.24, abs=0.01) and np.rad2deg(l) == pytest.approx(40.20, abs=0.01)
    assert m63.EDGES == (0.0, 10.0, 20.0, 30.0, 40.0)


@pytest.mark.skipif(not Path('/home/koyama/data/kitti/kitti_infos_train.pkl').exists(), reason='no KITTI')
def test_kitti_sector_from_real_calibrations():
    import pickle
    infos = pickle.load(open('/home/koyama/data/kitti/kitti_infos_train.pkl', 'rb'))[:50]
    r, l, cal = m.kitti_sector(infos)
    assert 38 < np.rad2deg(r) < 44 and 38 < np.rad2deg(l) < 44 and cal['W'] > 1200


@pytest.mark.parametrize('arm', m.ARMS)
def test_leaf_text_holds_only_its_arms_keys(arm):
    cuts = {'edges': [0.0, 10.0, 20.0, 30.0, 40.0],
            'rule_a': {'Car': {'neg': 0.123}, 'Pedestrian': {'neg': 0.2}, 'Cyclist': {'neg': 0.25}},
            'count_balance': {c: {'cuts': [0.1, 0.2, 0.3, 0.4, 0.5]} for c in m.CLASSES}}
    y = yaml.safe_load(m.leaf_text(arm, cuts))
    w, c = arm.split('_')
    if c == 'A':
        assert y['_BASE_CONFIG_'] == m.BASE
        assert y['SELF_TRAIN'] == {'SCORE_THRESH': [0.123, 0.2, 0.25], 'NEG_THRESH': [0.123, 0.2, 0.25]}
        assert (y.get('OPTIMIZATION') == {'TRAIN_ONLY_BN': True}) == (w == 'bn')
        assert set(y) == ({'_BASE_CONFIG_', 'SELF_TRAIN', 'OPTIMIZATION'} if w == 'bn' else {'_BASE_CONFIG_', 'SELF_TRAIN'})
    else:
        assert y['_BASE_CONFIG_'] == m.leaf_path(w + '_A') and set(y) == {'_BASE_CONFIG_', 'SELF_TRAIN'}
        assert y['SELF_TRAIN'] == {'THRESH_BY_RANGE': {'EDGES': cuts['edges'], 'CUTS': [[0.1, 0.2, 0.3, 0.4, 0.5]] * 3}}


def test_base_config_is_t_plus_59b_recipe():
    t, b59, b = m.load_cfg(m.T_CFG), m.load_cfg(m.B59), m.load_cfg(m.BASE)
    P = m._plain
    assert P(b.MODEL) == P(t.MODEL) and P(b.DATA_CONFIGS) == P(t.DATA_CONFIGS)
    teacher = P(b.SELF_TRAIN.MODEL_TEACHER)
    assert teacher == dict(P(t.MODEL), CLASS_NAMES=list(t.CLASS_NAMES), ONTOLOGY=t.ONTOLOGY)
    st, st59 = P(b.SELF_TRAIN), P(b59.SELF_TRAIN)
    st.pop('MODEL_TEACHER'); st59.pop('MODEL_TEACHER')
    assert st == st59                                              # 59b's recipe, unchanged
    assert not b.SELF_TRAIN.SRC.USE_DATA and b.SELF_TRAIN.TAR.USE_DATA and b.SELF_TRAIN.FROZEN_TEACHER_SINGLE_PASS
    assert b.SELF_TRAIN.TEACHER_ADABN == {'FRAMES': 1000, 'MIX': 1.0, 'BATCH_SIZE': 4}
    assert not b.SELF_TRAIN.PROG_AUG.ENABLED and not b.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED
    assert 'SCORE_THRESH' not in b.SELF_TRAIN and 'THRESH_BY_RANGE' not in b.SELF_TRAIN
    tar, ttar = P(b.DATA_CONFIG_TAR), P(t.DATA_CONFIG_TAR)
    assert tar.pop('USE_PSEUDO_LABEL') is True and tar.pop('DATA_AUGMENTOR') == P(b59.DATA_CONFIG_TAR.DATA_AUGMENTOR)
    ttar.pop('DATA_AUGMENTOR')
    tar.pop('_BASE_CONFIG_', None); ttar.pop('_BASE_CONFIG_', None)
    assert tar == ttar                                             # T's KITTI target: train split, FOV filter, SHIFT_COOR
    assert b.DATA_CONFIG_TAR.DATA_SPLIT.train == 'train' and b.DATA_CONFIG_TAR.TEST.BOX_FILTER.FOV_FILTER
    o, o59 = P(b.OPTIMIZATION), P(b59.OPTIMIZATION)
    assert o['NUM_EPOCHS'] == 20 and o['OPTIMIZER'] == o59['OPTIMIZER'] == 'adam_onecycle'
    for k in ('LR', 'MOMS', 'PCT_START', 'DIV_FACTOR', 'WEIGHT_DECAY', 'GRAD_NORM_CLIP', 'BATCH_SIZE_PER_GPU'):
        assert o[k] == o59[k], k                                   # OneCycle's own settings equal 59b's


@pytest.mark.skipif(not (TOOLS / m.leaf_path('bn_C')).exists(), reason='leaves not written yet')
@pytest.mark.parametrize('arm', m.ARMS)
def test_written_leaves_resolve(arm):
    c = m.load_cfg(m.leaf_path(arm))
    w, k = arm.split('_')
    assert bool(c.OPTIMIZATION.get('TRAIN_ONLY_BN', False)) == (w == 'bn')
    assert ('THRESH_BY_RANGE' in c.SELF_TRAIN) == (k == 'C')
    assert c.SELF_TRAIN.SCORE_THRESH == c.SELF_TRAIN.NEG_THRESH    # no ignore band, so no negative label is written
    assert not c.SELF_TRAIN.SRC.USE_DATA and c.OPTIMIZATION.NUM_EPOCHS == 20


def test_words():
    assert m.word([2.0, 3.0], 'cut') == m.WORDS_CUT[0] and m.word([-2.0, -2.5], 'cut') == m.WORDS_CUT[1]
    assert m.word([0.9, -0.9], 'cut') == m.WORDS_CUT[2] and m.word([2.5, 1.5], 'cut') == m.WORDS_CUT[3]
    assert m.word([1.9, -1.9], 'bn') == m.WORDS_BN[0] and m.word([-2.0, -3.0], 'bn') == m.WORDS_BN[1]
    assert m.word([2.0, 2.1], 'bn') == m.WORDS_BN[2] and m.word([2.5, 0.5], 'bn') == m.WORDS_BN[3]


def test_readings_and_jobs():
    rows = [dict(row=m.ROWS[a], seed=s, pooled={'bev': v + (0.5 if s == 667 else 0.0)})
            for a, v in (('bn_A', 75.0), ('all_A', 74.0), ('bn_C', 78.0), ('all_C', 76.0)) for s in m.SEEDS]
    r = {x['name']: x for x in m.readings(rows)}
    assert r['H_cut, BN only: bn_C - bn_A']['diffs'] == [3.0, 3.0] and r['H_cut, BN only: bn_C - bn_A']['word'] == m.WORDS_CUT[0]
    assert r['H_bn, rule A: bn_A - all_A']['word'] == m.WORDS_BN[0]
    assert r['H_bn, count balance: bn_C - all_C']['word'] == m.WORDS_BN[2]
    assert m.parse_jobs(['bn_A666=1', 'all_C667=2']) == {('bn_A', 666): 1, ('all_C', 667): 2}
    assert m.T_CKPT.endswith('o3l96a3p/files/ckpt/checkpoint_epoch_20.pth') and m.EPOCHS == 20
