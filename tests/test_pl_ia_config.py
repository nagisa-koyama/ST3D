"""S2 IA-SSD pseudo-labelling row PL-IA (experiments_md 20261025_01; design FIXED 10-25 00:42 JST, experiments_md 20261003_05
§5, commit d8dd0d8): config integrity, the cuts and budget it carries, and the sentinel behaviour of its target path
(memory/repo/bug_prevention_checklist.md rules 1-4).

Pinned:
  * the resolved config is the PB A1 crop twin plus ONLY the listed keys: DATA_CONFIG_TAR.USE_PSEUDO_LABEL, the target
    budget's TRAIN key and drop_empty_gt_boxes after it, and a SELF_TRAIN block;
  * that SELF_TRAIN is 27566's (centerpoint-accum-legaldepth-st3d-nuscenes2kitti.yaml, unchanged since 27566 froze
    a893705) but MODEL_TEACHER (= this MODEL + 27566's CLASS_NAMES / ONTOLOGY keys) and the two cut lines;
  * the cut lines are the cut job's output (rule B, rule A, step 6) and the TRAIN budget is the census' (rule PB on the
    target's training view: the larger P99 of PROG_AUG epoch 0 and its last intensity, next multiple of 4,096);
  * ignore-band labels -1..-3 go through fill_pseudo_labels -> prepare_data (augmentation) -> budget -> recount kept, never
    become IA-SSD positives; a positive the budget empties is dropped; -99 cannot enter through fill_pseudo_labels
    (IndexError, loud) and is kept by the budget / recount; IA-SSD-shaped prediction dicts through save_pseudo_label_batch
    give exactly apply_pseudo_label_cuts.py's labels at the row's cuts;
  * a batch whose gt_boxes has ZERO rows: collates to (B, 0, 8), the upstream assignment raises on it, and the guard in
    IASSD_Head.assign_targets / get_loss returns a graph-connected zero for that batch only (a batch with a row is not
    touched).
"""
import copy
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(TOOLS))
import _init_path  # noqa: F401,E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

C = 'cfgs/da-ieee-access-pointrcnn/'
ROW = C + 'iassd-st3d-pb-nuscenes2kitti-fovtar.yaml'
BASE = C + 'iassd-sourceonly-recount-pb-nuscenes2kitti-fovtar.yaml'
CP27566 = 'cfgs/da-ieee-access/centerpoint-accum-legaldepth-st3d-nuscenes2kitti.yaml'
LOGS = TOOLS / 'logs' / 'pl_ia_20261025'

# The cut job's output (logs/pl_ia_20261025/cuts/final_cuts.json) and the census' budget. Filled from those files.
PINNED_SCORE = [0.1, 0.148, 0.22]     # rule B, then rule A step 6 (Pedestrian raised from 0.10)
PINNED_NEG = [0.035, 0.148, 0.015]     # rule A
PINNED_TRAIN_BUDGET = 20480   # census job 30490: larger P99 20,470.0 (PROG_AUG epoch 0) -> next multiple of 4,096
EVAL_BUDGET = 20480            # the crop twin's eval budget (rule PB, 20261024_05 §2)
CUT_KEYS = ('SCORE_THRESH', 'NEG_THRESH')


def load(rel):
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        cfg = EasyDict()
        cfg_from_yaml_file(rel, cfg)
    finally:
        os.chdir(cwd)
    return cfg


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items() if k != '_BASE_CONFIG_'}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def _budget(procs):
    hits = [p for p in procs if p['NAME'] == 'sample_points']
    assert len(hits) == 1
    return hits[0]


def _cuts_pinned():
    if PINNED_SCORE is None:
        pytest.fail('the cut lines are not pinned yet (the cut job has not run)')
    return PINNED_SCORE, PINNED_NEG


# ---------------------------------------------------------------------------------------------------- config integrity
def test_resolved_config_is_the_crop_twin_plus_the_listed_keys():
    row, base = plain(load(ROW)), plain(load(BASE))
    assert 'SELF_TRAIN' in row and 'SELF_TRAIN' not in base
    row.pop('SELF_TRAIN')
    tar, btar = row['DATA_CONFIG_TAR'], base['DATA_CONFIG_TAR']
    assert tar.pop('USE_PSEUDO_LABEL') is True and 'USE_PSEUDO_LABEL' not in btar
    names = [p['NAME'] for p in tar['DATA_PROCESSOR']]
    assert names == ['mask_points_and_boxes_outside_range', 'sample_points', 'drop_empty_gt_boxes', 'shuffle_points']
    assert tar['DATA_PROCESSOR'][2].get('MIN_POINTS', 1) == 1
    b, bb = _budget(tar['DATA_PROCESSOR']), _budget(btar['DATA_PROCESSOR'])
    assert b['NUM_POINTS']['test'] == bb['NUM_POINTS']['test'] == EVAL_BUDGET
    assert b['NUM_POINTS']['train'] % 4096 == 0
    tar['DATA_PROCESSOR'] = btar['DATA_PROCESSOR']        # put the base's chain back: nothing else may differ
    assert tar['FOV_POINTS_ONLY'] and tar['ALLOW_FOV_POINTS_ONLY'] and tar['DATASET'] == 'KittiDataset'
    assert row == base


def test_target_train_budget_is_the_census_value():
    row = load(ROW)
    train = _budget(row.DATA_CONFIG_TAR.DATA_PROCESSOR)['NUM_POINTS']['train']
    if PINNED_TRAIN_BUDGET is None:
        pytest.fail('the target TRAIN budget is not pinned yet (the census has not run)')
    assert train == PINNED_TRAIN_BUDGET
    js = [LOGS / 'census' / ('pl_ia_target_train_%s.json' % s) for s in ('epoch0', 'last')]
    if not all(p.exists() for p in js):
        pytest.skip('census JSONs not in this checkout (tools/logs is not tracked)')
    from analysis.iassd_pb_budget_census import pb_budget
    p99 = [json.load(open(p))['summary']['n_in']['p99'] for p in js]
    assert train == pb_budget(max(p99))


def test_self_train_is_27566s_but_the_teacher_and_the_cut_lines():
    st, ref = plain(load(ROW).SELF_TRAIN), plain(load(CP27566).SELF_TRAIN)
    for k in CUT_KEYS + ('MODEL_TEACHER',):
        st.pop(k), ref.pop(k)
    assert st == ref
    assert st['FROZEN_TEACHER_SINGLE_PASS'] is True and st['UPDATE_PSEUDO_LABEL'] == [0]
    assert st['EPOCH_FOLLOWS'] == 'source'
    assert st['PROG_AUG'] == {'ENABLED': True, 'UPDATE_AUG': [5, 10, 15], 'SCALE': 1.1}
    assert 'DSNORM' not in st and 'TEACHER_ADABN' not in st and 'TEACHER_IS_STUDENT' not in st


def test_model_teacher_is_the_model():
    row, ref = plain(load(ROW)), plain(load(CP27566))
    mt = dict(row['SELF_TRAIN']['MODEL_TEACHER'])
    for k in ('CLASS_NAMES', 'ONTOLOGY'):              # the bookkeeping keys 27566's teacher block carries
        assert mt.pop(k) == ref['SELF_TRAIN']['MODEL_TEACHER'][k]
    assert mt == row['MODEL']
    assert mt['NAME'] == 'IASSD' and mt['POST_PROCESSING']['SCORE_THRESH'] == 0.0001   # the generation floor


def test_cut_lines_are_the_cut_jobs_output():
    score, neg = _cuts_pinned()
    st = load(ROW).SELF_TRAIN
    assert list(st.SCORE_THRESH) == score and list(st.NEG_THRESH) == neg
    assert all(s >= n for s, n in zip(score, neg)), 'rule A step 6: SCORE_THRESH >= NEG_THRESH'
    f = LOGS / 'cuts' / 'final_cuts.json'
    if not f.exists():
        pytest.skip('cut-job output not in this checkout (tools/logs is not tracked)')
    out = json.load(open(f))
    assert [round(v, 3) for v in out['SCORE_THRESH']] == score and [round(v, 3) for v in out['NEG_THRESH']] == neg


def test_the_rule_b_generation_config_is_the_teacher_on_nuscenes_val_with_pb_ns_eval_block():
    gen = plain(load('analysis/pseudo_label_threshold/cfgs/teacher_pl_ia_qmjuvxoe_on_nuscenes_val_1sweep.yaml'))
    n = plain(load(C + 'iassd-sourceonly-recount-pb-nuscenes2nuscenes.yaml'))
    row = plain(load(ROW))
    g, t = dict(gen['DATA_CONFIG_TAR']), dict(n['DATA_CONFIG_TAR'])
    assert g['DATA_SPLIT'] == {'train': 'val', 'test': 'val'}
    assert g['INFO_PATH']['train'] == g['INFO_PATH']['test'] == t['INFO_PATH']['test']
    for k in ('DATA_SPLIT', 'INFO_PATH', 'USE_PSEUDO_LABEL'):
        g.pop(k, None), t.pop(k, None)
    assert g == t and g['MAX_SWEEPS'] == 1
    assert gen['SELF_TRAIN']['MODEL_TEACHER'] == row['SELF_TRAIN']['MODEL_TEACHER']
    assert gen['MODEL'] == row['MODEL'] and gen['CLASS_NAMES'] == row['CLASS_NAMES']


# ------------------------------------------------------------------------------- sentinels through the target path
def _box(x, label, dx=4.0, dy=2.0, dz=1.5):
    return [x, 0.0, 0.5, dx, dy, dz, 0.0, label]


def _target_frame(n_background, rng):
    """Pseudo-labels at x = 5 (Car, 400 points), 12 (Car, ONE point: what a budget can empty), 19 (Pedestrian, 60),
    26 (Cyclist, 60) and the ignore band at 33 / 40 / 47 (-1 / -2 / -3, ONE point each), plus background."""
    bg = np.c_[rng.uniform(1, 70, n_background), rng.uniform(-35, 35, n_background), rng.uniform(-1.5, 3, n_background)]
    bg = bg[np.abs(bg[:, 1]) > 4]
    dense = np.c_[rng.uniform(3.5, 6.5, 400), rng.uniform(-0.8, 0.8, 400), rng.uniform(0, 1, 400)]
    ped = np.c_[rng.uniform(18.8, 19.2, 60), rng.uniform(-0.2, 0.2, 60), rng.uniform(0, 1, 60)]
    cyc = np.c_[rng.uniform(25.4, 26.6, 60), rng.uniform(-0.2, 0.2, 60), rng.uniform(0, 1, 60)]
    singles = np.array([[12.0, 0.0, 0.5], [33.0, 0.0, 0.5], [40.0, 0.0, 0.5], [47.0, 0.0, 0.5]])
    pts = np.concatenate([bg, dense, ped, cyc, singles]).astype(np.float32)
    boxes = np.array([_box(5, 1), _box(12, 1), _box(19, 2, 0.8, 0.6, 1.7), _box(26, 3, 1.7, 0.6, 1.3),
                      _box(33, -1), _box(40, -2, 0.8, 0.6, 1.7), _box(47, -3, 1.7, 0.6, 1.3)], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.15, 0.2, 0.25], dtype=np.float32)
    return pts, np.c_[boxes, scores]


@pytest.fixture(scope='module')
def kitti_target():
    """The row's own self-training target dataset (KittiDataset, training mode, augmentation on), built as train.py."""
    import logging
    from pcdet.datasets import build_dataloader
    cfg = load(ROW)
    if not (ROOT / 'data' / 'kitti' / 'kitti_infos_train.pkl').exists():
        pytest.skip('KITTI infos not available here')
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1,
                                    dist=False, workers=0, logger=logging.getLogger('t'), training=True,
                                    model_ontology=cfg.get('ONTOLOGY', None))
    finally:
        os.chdir(cwd)
    return cfg, ds


def test_ignore_band_through_fill_prepare_budget_recount(kitti_target):
    cfg, ds = kitti_target
    budget = _budget(cfg.DATA_CONFIG_TAR.DATA_PROCESSOR)['NUM_POINTS']['train']
    rng = np.random.RandomState(0)
    pts, ps = _target_frame(int(2.2 * budget), rng)
    assert len(pts) > budget
    ds.set_pseudo_labels({'synthetic': {'gt_boxes': ps}})
    dropped = 0
    for seed in range(16):
        np.random.seed(seed)
        d = {'points': np.c_[pts, np.zeros(len(pts), np.float32)], 'frame_id': 'synthetic'}
        ds.fill_pseudo_labels(d)
        assert list(d['pos_ps_bbox']) == [2, 1, 1] and list(d['ign_ps_bbox']) == [1, 1, 1]   # what the run logs
        assert list(d['gt_names']) == ['Car', 'Car', 'Pedestrian', 'Cyclist', 'Car', 'Pedestrian', 'Cyclist']
        out = ds.prepare_data(d)
        assert len(out['points']) == budget
        lab = out['gt_boxes'][:, 7]
        assert sorted(lab[lab < 0].tolist()) == [-3, -2, -1], 'every ignore-band box survives the budget and recount'
        from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
        pos = out['gt_boxes'][lab > 0]
        held = roiaware_pool3d_utils.points_in_boxes_cpu(
            np.ascontiguousarray(out['points'][:, :3]), np.ascontiguousarray(pos[:, :7])).sum(1)
        assert (held >= 1).all(), 'no positive left on empty space after the recount'
        assert sorted(pos[:, 7].tolist()) in ([1, 1, 2, 3], [1, 2, 3])
        dropped += len(pos) == 3
    assert 0 < dropped < 16, 'the budget empties the one-point Car in some draws and not in others'


def test_minus_99_cannot_enter_through_fill_pseudo_labels_and_is_kept_by_budget_and_recount(kitti_target):
    cfg, ds = kitti_target
    ds.set_pseudo_labels({'s': {'gt_boxes': np.array([_box(5, common_utils.IGNORE_CLASS_LABEL) + [0.5]], np.float32)}})
    with pytest.raises(IndexError):          # np.array(class_names)[98]: loud, never a silent permuted class
        ds.fill_pseudo_labels({'points': np.zeros((10, 4), np.float32), 'frame_id': 's'})
    budget = _budget(cfg.DATA_CONFIG_TAR.DATA_PROCESSOR)['NUM_POINTS']['train']
    b = cfg.DATA_CONFIG_TAR
    proc = DataProcessor(b.DATA_PROCESSOR, point_cloud_range=np.array(b.POINT_CLOUD_RANGE, np.float32), training=True,
                         num_point_features=3)
    pts, ps = _target_frame(int(0.9 * budget), np.random.RandomState(1))
    boxes = np.r_[ps[:, :8], [_box(54, common_utils.IGNORE_CLASS_LABEL)]].astype(np.float32)   # -99, empty
    np.random.seed(0)
    out = proc.forward({'points': pts.copy(), 'gt_boxes': boxes.copy()})
    assert len(out['points']) == budget
    assert common_utils.IGNORE_CLASS_LABEL in out['gt_boxes'][:, 7].tolist()
    assert sorted(out['gt_boxes'][out['gt_boxes'][:, 7] < 0, 7].tolist()) == sorted([-3, -2, -1, common_utils.IGNORE_CLASS_LABEL])


def test_eval_mode_leaves_the_pseudo_label_boxes_and_samples_to_the_eval_budget():
    b = load(ROW).DATA_CONFIG_TAR
    proc = DataProcessor(b.DATA_PROCESSOR, point_cloud_range=np.array(b.POINT_CLOUD_RANGE, np.float32), training=False,
                         num_point_features=3)
    pts, ps = _target_frame(int(1.5 * EVAL_BUDGET), np.random.RandomState(2))
    np.random.seed(0)
    out = proc.forward({'points': pts.copy(), 'gt_boxes': ps[:, :8].copy()})
    assert len(out['points']) == EVAL_BUDGET and out['gt_boxes'].shape == (7, 8)


def test_recounted_target_ignore_boxes_never_become_iassd_positives(kitti_target, monkeypatch):
    from pcdet.models.dense_heads import IASSD_head as head_mod
    from pcdet.utils import box_coder_utils, box_utils
    monkeypatch.setattr(head_mod.roiaware_pool3d_utils, 'points_in_boxes_gpu', _cpu_points_in_boxes)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    cfg, ds = kitti_target
    bc = cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG
    head = types.SimpleNamespace(num_class=3, box_coder=box_coder_utils.PointResidual_BinOri_Coder(
        angle_bin_num=bc.angle_bin_num, use_mean_size=bc.use_mean_size, mean_size=copy.deepcopy(bc.mean_size)))
    budget = _budget(cfg.DATA_CONFIG_TAR.DATA_PROCESSOR)['NUM_POINTS']['train']
    rng = np.random.RandomState(3)
    pts, ps = _target_frame(int(0.8 * budget), rng)
    extra = np.concatenate([np.c_[rng.uniform(x - 0.3, x + 0.3, 40), rng.uniform(-0.2, 0.2, 40), rng.uniform(0, 1, 40)]
                            for x in (33.0, 40.0, 47.0)]).astype(np.float32)       # points inside the ignore boxes
    ds.set_pseudo_labels({'s': {'gt_boxes': ps}})
    np.random.seed(0)
    d = {'points': np.c_[np.r_[pts, extra], np.zeros(len(pts) + len(extra), np.float32)], 'frame_id': 's'}
    ds.fill_pseudo_labels(d)
    out = ds.prepare_data(d)
    batch = ds.collate_batch([out])
    points = torch.from_numpy(batch['points'][:, :4]).float()
    gt = torch.from_numpy(batch['gt_boxes']).float()
    assert (gt[0, :, 7] < 0).sum() == 3
    for ext_w, kw in (([0.2, 0.2, 0.2], dict()), ([1.0, 1.0, 1.0], dict(use_ex_gt_assign=True, fg_pc_ignore=False))):
        ext = box_utils.enlarge_box3d(gt.view(-1, 8), extra_width=ext_w).view(1, -1, 8)
        res = head_mod.IASSD_Head.assign_stack_targets_IASSD(head, points=points, gt_boxes=gt, extend_gt_boxes=ext,
                                                             set_ignore_flag=True, ret_box_labels=True, **kw)
        lab = res['point_cls_labels']
        ign = gt[0][gt[0, :, 7] < 0]
        in_ign = (roiaware_pool3d_cpu(points[:, 1:4], ign[:, :7]) > 0).any(0)
        assert int(in_ign.sum()) > 0 and (lab[in_ign] == -1).all()
        assert (lab > 0).any() and res['gt_box_of_fg_points'].shape[0] == int((lab > 0).sum())
        assert (res['gt_box_of_fg_points'][:, 7] > 0).all()


def roiaware_pool3d_cpu(points, boxes):
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    return roiaware_pool3d_utils.points_in_boxes_cpu(points.float(), boxes.float())


def _cpu_points_in_boxes(points, boxes):
    """CPU stand-in for points_in_boxes_gpu: (1, N, 3), (1, T, 7) -> (1, N) first containing box, or -1."""
    idx = torch.full((points.shape[1],), -1, dtype=torch.int)
    if boxes.shape[1] == 0:
        return idx[None]
    inside = roiaware_pool3d_cpu(points[0], boxes[0]) > 0
    hit = inside.any(dim=0)
    idx[hit] = torch.argmax(inside.int(), dim=0)[hit].int()
    return idx[None]


# ------------------------------------------------------------- the labelling pass: save_pseudo_label_batch at the cuts
@pytest.fixture
def row_cfg_installed():
    """pcdet.config.cfg as train.py leaves it for this row (save_pseudo_label_batch reads the global)."""
    from pcdet.config import cfg as gcfg
    from pcdet.utils import self_training_utils
    row = load(ROW)
    saved = {k: gcfg.get(k, None) for k in ('SELF_TRAIN', 'CLASS_NAMES', 'ONTOLOGY')}
    st = copy.deepcopy(row.SELF_TRAIN)
    score, neg = _cuts_pinned()
    assert list(st.SCORE_THRESH) == score and list(st.NEG_THRESH) == neg
    for k, v in (('SELF_TRAIN', st), ('CLASS_NAMES', row.CLASS_NAMES), ('ONTOLOGY', row.ONTOLOGY)):
        gcfg[k] = v
    self_training_utils.NEW_PSEUDO_LABELS.clear()
    try:
        yield row
    finally:
        self_training_utils.NEW_PSEUDO_LABELS.clear()
        for k, v in saved.items():
            if v is not None:
                gcfg[k] = v
                continue
            dict.pop(gcfg, k, None)
            try:
                delattr(gcfg, k)
            except AttributeError:
                pass


def test_iassd_predictions_through_save_pseudo_label_batch_equal_apply_pseudo_label_cuts(row_cfg_installed):
    from analysis.pseudo_label_threshold.apply_pseudo_label_cuts import apply_cuts
    from pcdet.utils import self_training_utils
    score, neg = _cuts_pinned()
    rng = np.random.RandomState(5)
    pred_dicts, ids, raw = [], [], []
    for f in range(6):                      # IA-SSD post_processing's record: pred_boxes / pred_scores / pred_labels
        m = [40, 0, 1, 25, 300, 7][f]
        boxes = np.c_[rng.uniform(0, 70, (m, 2)), rng.uniform(-1, 1, (m, 1)), rng.uniform(0.5, 5, (m, 3)),
                      rng.uniform(-3, 3, (m, 1))].astype(np.float32)
        s = rng.choice(np.r_[rng.uniform(1e-4, 1, 400), np.array(score), np.array(neg)], m).astype(np.float32)
        lab = rng.randint(1, 4, m)
        pred_dicts.append({'pred_boxes': torch.from_numpy(boxes), 'pred_scores': torch.from_numpy(s),
                           'pred_labels': torch.from_numpy(lab).long()})
        ids.append('%06d' % f)
        raw.append(np.c_[boxes, lab, s])
    pos, ign = self_training_utils.save_pseudo_label_batch({'frame_id': ids}, pred_dicts=pred_dicts, need_update=False,
                                                           student_ontology='kitti')
    n_pos, n_ign = np.zeros(3), np.zeros(3)
    for fid, r in zip(ids, raw):
        got = self_training_utils.NEW_PSEUDO_LABELS[fid]['gt_boxes']
        want = apply_cuts(r, score, neg)
        assert got.shape == want.shape and np.allclose(got, want), fid
        for c in range(3):
            n_pos[c] += (want[:, 7] == c + 1).sum()
            n_ign[c] += (want[:, 7] == -(c + 1)).sum()
    assert np.allclose([pos.meters[c].sum for c in range(3)], n_pos)
    assert np.allclose([ign.meters[c].sum for c in range(3)], n_ign)
    assert n_ign.sum() > 0 and n_pos.sum() > 0, 'the draw exercises both the ignore band and positives'


# --------------------------------------------------------------------------------------------- the zero-row batch
def _zero_row_batch(b=2):
    from pcdet.datasets.dataset import DatasetTemplate
    return DatasetTemplate.collate_batch([{'points': np.random.rand(5, 3).astype(np.float32),
                                           'gt_boxes': np.zeros((0, 8), np.float32)} for _ in range(b)])


def test_a_batch_without_any_box_row_collates_to_zero_rows_and_the_upstream_assignment_raises(monkeypatch):
    from pcdet.models.dense_heads import IASSD_head as head_mod
    from pcdet.utils import box_coder_utils, box_utils
    monkeypatch.setattr(head_mod.roiaware_pool3d_utils, 'points_in_boxes_gpu', _cpu_points_in_boxes)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    batch = _zero_row_batch()
    assert batch['gt_boxes'].shape == (2, 0, 8)
    gt = torch.from_numpy(batch['gt_boxes'])
    pts = torch.from_numpy(np.c_[np.repeat([0, 1], 50), np.random.rand(100, 3) * 10].astype(np.float32))
    head = types.SimpleNamespace(num_class=3, box_coder=box_coder_utils.PointResidual_BinOri_Coder(
        angle_bin_num=12, use_mean_size=True, mean_size=[[4.63, 1.96, 1.74], [0.73, 0.67, 1.77], [1.70, 0.61, 1.30]]))
    ext = box_utils.enlarge_box3d(gt.view(-1, 8), extra_width=[0.2, 0.2, 0.2]).view(2, -1, 8)
    with pytest.raises(IndexError):
        head_mod.IASSD_Head.assign_stack_targets_IASSD(head, points=pts, gt_boxes=gt, extend_gt_boxes=ext,
                                                       set_ignore_flag=True, ret_box_labels=True)


def _real_head(monkeypatch):
    from pcdet.models.dense_heads import IASSD_head as head_mod
    monkeypatch.setattr(head_mod.roiaware_pool3d_utils, 'points_in_boxes_gpu', _cpu_points_in_boxes)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    torch.manual_seed(0)
    return head_mod.IASSD_Head(num_class=3, input_channels=512, model_cfg=load(ROW).MODEL.POINT_HEAD,
                               predict_boxes_when_training=False)


def _head_inputs(gt, b=2, n=64):
    rng = torch.Generator().manual_seed(1)
    bidx = torch.arange(b).repeat_interleave(n).float()[:, None]
    xyz = torch.rand(b * n, 3, generator=rng) * 20
    cen = torch.cat([bidx, xyz], 1)
    coords = [torch.cat([bidx.view(b, n, 1), (torch.rand(b * n, 3, generator=rng) * 20).view(b, n, 3)], -1)
              for _ in range(3)]
    return {'batch_size': b, 'gt_boxes': gt, 'centers': cen, 'centers_origin': cen.clone(),
            'encoder_coords': coords, 'sa_ins_preds': [[], torch.zeros(b, n, 4), torch.zeros(b, n, 4)]}


def test_zero_row_guard_returns_a_graph_connected_zero_for_that_batch_only(monkeypatch):
    head = _real_head(monkeypatch)
    gt = torch.from_numpy(_zero_row_batch()['gt_boxes'])
    t = head.assign_targets(_head_inputs(gt))
    assert t == {'zero_gt_rows': True}
    feats = torch.randn(128, 512, requires_grad=True)
    sa1 = torch.randn(2, 64, 4, requires_grad=True)
    head.forward_ret_dict = dict(t, center_cls_preds=head.cls_center_layers(feats),
                                 center_box_preds=head.box_center_layers(feats),
                                 ctr_offsets=torch.randn(128, 4, requires_grad=True), sa_ins_preds=[[], sa1, sa1 * 2],
                                 box_iou3d_preds=None)
    loss, tb = head.get_loss()
    assert torch.isfinite(loss) and float(loss) == 0.0 and tb == {'zero_gt_rows_batch': 1.0}
    loss.backward()
    grads = [p.grad for p in list(head.cls_center_layers.parameters()) + list(head.box_center_layers.parameters())]
    assert all(g is not None for g in grads) and all(float(g.abs().sum()) == 0.0 for g in grads)
    assert feats.grad is not None and sa1.grad is not None    # every upstream parameter stays in the graph


@pytest.mark.parametrize('rows', ['padding only', 'one box'])
def test_a_batch_with_a_row_is_not_touched_by_the_guard(monkeypatch, rows):
    head = _real_head(monkeypatch)
    gt = torch.zeros(2, 1, 8)
    if rows == 'one box':
        gt[0, 0] = torch.tensor(_box(5, 1))
    t = head.assign_targets(_head_inputs(gt))
    assert 'zero_gt_rows' not in t
    assert {'center_cls_labels', 'center_origin_cls_labels', 'sa_ins_labels'} <= set(t)
