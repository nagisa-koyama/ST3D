"""IA-SSD gap rows of experiments_md 20261024_03 (T05 §5 "IA-SSD designs FIXED 13:43 JST"): config integrity and the
sentinel behaviour of their processor chains (memory/repo/bug_prevention_checklist.md rules 1-2).

- S2R (item 2): A1R / A2R are A1 (iassd-sourceonly-nuscenes2kitti) / A2 (iassd-accum-legaldepth-nuscenes2kitti) plus
  `drop_empty_gt_boxes` right after the 16,384-point budget; nothing else may differ, crop twins included.
- I2 (item 3): the Lyft 40-beam IA-SSD rows P / C1-analog carry CenterPoint rows 30312 / 30313's data blocks with IA-SSD's
  processor (budget, then the recount), and 27571's MODEL / OPTIMIZATION / mean_size / target block.
- The processor chains, run on synthetic frames: a positive box the budget empties is dropped, ignore labels (the
  pseudo-label band -1..-3 and IGNORE_CLASS_LABEL -99) are kept whatever they hold and never become IA-SSD positives, a
  frame under the budget is padded to it, and evaluation is untouched.
"""
import copy
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import _init_path  # noqa: F401,E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets.point_calibration import point_budget_after_correction  # noqa: E402
from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

C = 'cfgs/da-ieee-access-pointrcnn/'
CP = 'cfgs/da-ieee-access/'
I2_P, I2_C1 = C + 'iassd-ringpattern-lyft40-lyft2nuscenes.yaml', C + 'iassd-global-recount-lyft40-lyft2nuscenes.yaml'
LYFT_MEANS = [[4.75, 1.92, 1.72], [0.80, 0.77, 1.76], [1.76, 0.61, 1.42]]
BUDGET = 16384


def load(rel):
    import os
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


def names(procs):
    return [p.NAME for p in procs]


def without_recount(procs):
    return [p for p in plain(procs) if p['NAME'] != 'drop_empty_gt_boxes']


# ---------------------------------------------------------------------------------------------- S2R config integrity
@pytest.mark.parametrize('child,parent,blocks', [
    ('iassd-sourceonly-recount-nuscenes2kitti', 'iassd-sourceonly-nuscenes2kitti', ['DATA_CONFIG']),
    ('iassd-sourceonly-recount-nuscenes2kitti-fovtar', 'iassd-sourceonly-nuscenes2kitti-fovtar', ['DATA_CONFIG']),
    ('iassd-accum-legaldepth-recount-nuscenes2kitti', 'iassd-accum-legaldepth-nuscenes2kitti',
     ['DATA_CONFIGS.NUSCENES_N008', 'DATA_CONFIGS.NUSCENES_N015']),
    ('iassd-accum-legaldepth-recount-nuscenes2kitti-fovtar', 'iassd-accum-legaldepth-nuscenes2kitti-fovtar',
     ['DATA_CONFIGS.NUSCENES_N008', 'DATA_CONFIGS.NUSCENES_N015']),
])
def test_s2r_differs_from_its_parent_by_the_recount_only(child, parent, blocks):
    a, b = plain(load(C + child + '.yaml')), plain(load(C + parent + '.yaml'))
    for blk in blocks:
        node_a, node_b = a, b
        for k in blk.split('.'):
            node_a, node_b = node_a[k], node_b[k]
        pa = node_a['DATA_PROCESSOR']
        assert [p['NAME'] for p in pa] == ['mask_points_and_boxes_outside_range', 'sample_points',
                                           'drop_empty_gt_boxes', 'shuffle_points'], blk
        assert pa[2].get('MIN_POINTS', 1) == 1
        assert 'drop_empty_gt_boxes' not in [p['NAME'] for p in node_b['DATA_PROCESSOR']]
        node_a['DATA_PROCESSOR'] = without_recount(pa)
    assert a == b


def test_s2r_crop_twins_crop_the_target_only():
    for c in ('iassd-sourceonly-recount-nuscenes2kitti-fovtar', 'iassd-accum-legaldepth-recount-nuscenes2kitti-fovtar'):
        cfg = load(C + c + '.yaml')
        assert cfg.DATA_CONFIG_TAR.FOV_POINTS_ONLY and cfg.DATA_CONFIG_TAR.ALLOW_FOV_POINTS_ONLY
        assert 'drop_empty_gt_boxes' not in names(cfg.DATA_CONFIG_TAR.DATA_PROCESSOR)


# ----------------------------------------------------------------------------------------------- I2 config integrity
@pytest.mark.parametrize('rel', [I2_P, I2_C1])
def test_i2_model_optimiser_target_and_mean_size_are_27571s(rel):
    cfg, ref = load(rel), load(C + 'iassd-sourceonly-lyft2nuscenes.yaml')
    assert 'DATA_CONFIG' not in cfg, 'train.py would take a pooled DATA_CONFIG over DATA_CONFIGS'
    assert list(cfg.DATA_CONFIGS.keys()) == ['LYFT_40BEAM'], 'the 64-beam platform is not used'
    for k in ('MODEL', 'OPTIMIZATION', 'DATA_CONFIG_TAR', 'CLASS_NAMES', 'ONTOLOGY'):
        assert plain(cfg[k]) == plain(ref[k]), k
    assert plain(cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG.mean_size) == LYFT_MEANS
    assert cfg.OPTIMIZATION.NUM_EPOCHS == 30
    assert names(cfg.DATA_CONFIG_TAR.DATA_PROCESSOR) == ['mask_points_and_boxes_outside_range', 'sample_points',
                                                         'shuffle_points']


@pytest.mark.parametrize('rel,cp,proc', [
    (I2_P, CP + 'centerpoint-ringpattern-lyft40-lyft2nuscenes.yaml',
     ['mask_points_and_boxes_outside_range', 'sample_points', 'drop_empty_gt_boxes', 'shuffle_points']),
    (I2_C1, CP + 'centerpoint-global-recount-lyft40-lyft2nuscenes.yaml',
     ['mask_points_and_boxes_outside_range', 'sample_points_hist_based', 'sample_points', 'drop_empty_gt_boxes',
      'shuffle_points']),
])
def test_i2_data_block_is_the_centerpoint_rows_with_iassds_processor(rel, cp, proc):
    blk, ref = load(rel).DATA_CONFIGS.LYFT_40BEAM, load(cp).DATA_CONFIGS.LYFT_40BEAM
    assert names(blk.DATA_PROCESSOR) == proc
    budget = [p for p in blk.DATA_PROCESSOR if p.NAME == 'sample_points'][0]
    assert budget.NUM_POINTS.train == budget.NUM_POINTS.test == BUDGET
    a, b = plain(blk), plain(ref)
    a.pop('DATA_PROCESSOR'), b.pop('DATA_PROCESSOR')
    if 'sample_points_hist_based' in proc:
        assert a.pop('HIST_DIST_BEFORE_POINT_BUDGET') is True
        assert point_budget_after_correction(blk.DATA_PROCESSOR)
    else:
        assert not point_budget_after_correction(blk.DATA_PROCESSOR)
    assert a == b


def test_i2_rows_differ_in_the_selection_rule_only():
    p, c = plain(load(I2_P)), plain(load(I2_C1))
    bp, bc = p['DATA_CONFIGS']['LYFT_40BEAM'], c['DATA_CONFIGS']['LYFT_40BEAM']
    assert 'RING_PATTERN' in bp and 'RING_PATTERN' not in bc
    bp.pop('RING_PATTERN')
    for k in [k for k in bc if k.startswith('HIST_DIST_')]:
        bc.pop(k)
    bc['DATA_PROCESSOR'] = [x for x in bc['DATA_PROCESSOR'] if x['NAME'] != 'sample_points_hist_based']
    assert p == c


# ------------------------------------------------------------------------------- processor chains against sentinels
def _box(x, label):
    return [x, 0.0, 0.5, 4.0, 2.0, 1.5, 0.0, label]


def _frame(n_background, rng):
    """Boxes at x = 5 (positive, 400 points), 15 (positive, 1 point), 25 / 35 / 45 (ignore -1, -3 and -99, empty or
    near-empty), plus background. The 1-point positive is the one a budget can empty."""
    bg = np.c_[rng.uniform(-70, 70, n_background), rng.uniform(-70, 70, n_background), rng.uniform(-1.5, 3, n_background)]
    bg = bg[(np.abs(bg[:, 1]) > 3)]                      # keep background out of every box
    dense = np.c_[rng.uniform(3.5, 6.5, 400), rng.uniform(-0.8, 0.8, 400), rng.uniform(0, 1, 400)]
    one = np.array([[15.0, 0.0, 0.5]])
    ign = np.array([[35.0, 0.0, 0.5]])                   # one point in the -3 box
    pts = np.concatenate([bg, dense, one, ign]).astype(np.float32)
    boxes = np.array([_box(5, 1), _box(15, 1), _box(25, -1), _box(35, -3), _box(45, common_utils.IGNORE_CLASS_LABEL)],
                     dtype=np.float32)
    return pts, boxes


def _chain(rel, training):
    blk = load(rel).DATA_CONFIGS.LYFT_40BEAM if 'lyft40' in rel else load(rel).DATA_CONFIG
    procs = [p for p in blk.DATA_PROCESSOR if p.NAME != 'sample_points_hist_based']  # needs installed calibration
    return DataProcessor(procs, point_cloud_range=np.array(blk.POINT_CLOUD_RANGE, dtype=np.float32),
                         training=training, num_point_features=3)


@pytest.mark.parametrize('rel', [I2_P, I2_C1, C + 'iassd-sourceonly-recount-nuscenes2kitti.yaml'])
def test_budget_then_recount_drops_the_emptied_positive_and_keeps_every_ignore_label(rel):
    rng = np.random.RandomState(0)
    pts, boxes = _frame(60000, rng)
    proc = _chain(rel, training=True)
    dropped = kept_ignore = 0
    for seed in range(20):
        np.random.seed(seed)
        out = proc.forward({'points': pts.copy(), 'gt_boxes': boxes.copy(), 'gt_names': np.array(['Car'] * 5)})
        assert len(out['points']) == BUDGET
        lab = out['gt_boxes'][:, 7]
        assert sorted(lab[lab < 0].tolist()) == sorted([-1, -3, common_utils.IGNORE_CLASS_LABEL])  # never dropped
        assert 5.0 in out['gt_boxes'][:, 0].tolist()                                             # the dense positive
        one_kept = 15.0 in out['gt_boxes'][:, 0].tolist()
        one_has_point = bool((np.abs(out['points'][:, 0] - 15.0) < 1e-4).any())
        assert one_kept == one_has_point, 'a positive must be kept iff it still holds a point'
        dropped += not one_kept
        kept_ignore += 1
    assert 0 < dropped < 20, 'the budget should empty the 1-point positive in some draws and not in others'


@pytest.mark.parametrize('rel', [I2_P, I2_C1])
def test_frame_under_the_budget_is_padded_and_keeps_its_labels(rel):
    rng = np.random.RandomState(1)
    pts, boxes = _frame(15000, rng)                      # ~14,000 real points: P's regime (15,991 < 16,384)
    assert len(pts) < BUDGET
    np.random.seed(0)
    out = _chain(rel, training=True).forward({'points': pts.copy(), 'gt_boxes': boxes.copy()})
    assert len(out['points']) == BUDGET
    assert len(np.unique(out['points'], axis=0)) == len(pts)    # every real point kept, the rest are copies
    assert out['gt_boxes'].shape == boxes.shape                 # every box still holds its points


@pytest.mark.parametrize('rel', [I2_P, I2_C1, C + 'iassd-accum-legaldepth-recount-nuscenes2kitti.yaml'])
def test_evaluation_is_untouched_by_the_recount(rel):
    cfg = load(rel)
    blk = cfg.DATA_CONFIGS[list(cfg.DATA_CONFIGS.keys())[0]]
    procs = [p for p in blk.DATA_PROCESSOR if p.NAME != 'sample_points_hist_based']
    proc = DataProcessor(procs, point_cloud_range=np.array(blk.POINT_CLOUD_RANGE, dtype=np.float32), training=False,
                         num_point_features=3)
    pts, boxes = _frame(60000, np.random.RandomState(2))
    out = proc.forward({'points': pts.copy(), 'gt_boxes': boxes.copy()})
    assert out['gt_boxes'].shape == boxes.shape


@pytest.mark.parametrize('rel', [I2_P, I2_C1])
def test_recounted_ignore_boxes_never_become_iassd_positives(rel, monkeypatch):
    pytest.importorskip('pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils')
    from pcdet.models.dense_heads import IASSD_head as head_mod
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    from pcdet.utils import box_coder_utils, box_utils

    def cpu_points_in_boxes(points, boxes):
        inside = roiaware_pool3d_utils.points_in_boxes_cpu(points[0].float(), boxes[0].float()) > 0
        idx = torch.full((points.shape[1],), -1, dtype=torch.int)
        hit = inside.any(dim=0)
        idx[hit] = torch.argmax(inside.int(), dim=0)[hit].int()
        return idx[None]

    monkeypatch.setattr(head_mod.roiaware_pool3d_utils, 'points_in_boxes_gpu', cpu_points_in_boxes)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    cfg = load(rel)
    bc = cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG
    head = types.SimpleNamespace(num_class=3, box_coder=box_coder_utils.PointResidual_BinOri_Coder(
        angle_bin_num=bc.angle_bin_num, use_mean_size=bc.use_mean_size, mean_size=copy.deepcopy(bc.mean_size)))
    rng = np.random.RandomState(3)
    pts, boxes = _frame(60000, rng)
    # put points into every ignore box so the head sees them
    extra = np.concatenate([np.c_[rng.uniform(x - 1.5, x + 1.5, 50), rng.uniform(-0.8, 0.8, 50), rng.uniform(0, 1, 50)]
                            for x in (25.0, 35.0, 45.0)]).astype(np.float32)
    np.random.seed(0)
    out = _chain(rel, training=True).forward({'points': np.concatenate([pts, extra]), 'gt_boxes': boxes.copy()})
    points = torch.from_numpy(np.c_[np.zeros(len(out['points'])), out['points'][:, :3]].astype(np.float32))
    gt = torch.from_numpy(np.r_[out['gt_boxes'], np.zeros((1, 8), np.float32)][None].astype(np.float32))
    ext = box_utils.enlarge_box3d(gt.view(-1, 8), extra_width=[0.2, 0.2, 0.2]).view(1, -1, 8)
    res = head_mod.IASSD_Head.assign_stack_targets_IASSD(head, points=points, gt_boxes=gt, extend_gt_boxes=ext,
                                                         set_ignore_flag=True, ret_box_labels=True)
    lab = res['point_cls_labels']
    in_ignore = (points[:, 1] > 23) & (points[:, 1] < 47) & (points[:, 2].abs() < 1)
    assert int(in_ignore.sum()) > 0
    assert (lab[in_ignore] <= 0).all()
    assert res['gt_box_of_fg_points'].shape[0] == int((lab > 0).sum())
    assert (res['gt_box_of_fg_points'][:, 7] > 0).all()
