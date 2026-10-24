"""Rule PB IA-SSD configs (experiments_md 20261024_05; T05 §5 "Rule PB, FIXED 15:53 JST", clarified 16:06 in experiments_md
cbfbcf2): config integrity and the sentinel behaviour of every new block's processor chain
(memory/repo/bug_prevention_checklist.md rules 1-2).

- Each PB config differs from its parent row's config ONLY in each data block's budget (NUM_POINTS, both mode keys = the
  block's PB value) and, for N and K, `drop_empty_gt_boxes` right after the budget. Every PB training block recounts.
- The budgets are pinned to the census (job 30383 / 30384 / 30385): the smallest multiple of 4,096 >= the P99 of the
  points entering the budget step on 1,000 strided TRAIN frames.
- One PB run serves both S2 A1 and the nuScenes oracle N (coordinator 16:06): their resolved DATA_CONFIG, MODEL,
  OPTIMIZATION, CLASS_NAMES and ONTOLOGY are identical; only DATA_CONFIG_TAR differs.
- Processor chains on synthetic frames, per new block: a positive box the budget empties is dropped and one it keeps is
  kept; ignore labels (-1, -3, IGNORE_CLASS_LABEL -99) are kept whatever they hold and never become IA-SSD positives; a
  frame under the budget is padded to it with every real point and box kept; evaluation is untouched.
"""
import copy
import os
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
from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

C = 'cfgs/da-ieee-access-pointrcnn/'
BUDGET_STEPS = ('sample_points', 'sample_points_by_voxels')

# PB config -> (parent config, {block: PB budget}, blocks that gain the recount). Budgets from the census JSONs.
PINNED = {
    # PB config: (parent config, {block: PB budget}, blocks that gain the recount)
    'iassd-sourceonly-pb-waymo2nuscenes': ('iassd-sourceonly-waymo2nuscenes',
                                           {'DATA_CONFIG': 180224, 'DATA_CONFIG_TAR': 28672}, []),
    'iassd-sourceonly-recount-pb-nuscenes2nuscenes': ('iassd-sourceonly-nuscenes2nuscenes',
                                                      {'DATA_CONFIG': 28672, 'DATA_CONFIG_TAR': 28672}, ['DATA_CONFIG']),
    'iassd-sourceonly-recount-pb-nuscenes2kitti': ('iassd-sourceonly-recount-nuscenes2kitti',
                                                   {'DATA_CONFIG': 28672, 'DATA_CONFIG_TAR': 131072}, []),
    'iassd-sourceonly-recount-pb-nuscenes2kitti-fovtar': ('iassd-sourceonly-recount-nuscenes2kitti-fovtar',
                                                          {'DATA_CONFIG': 28672, 'DATA_CONFIG_TAR': 20480}, []),
    'iassd-sourceonly-recount-pb-kitti2kitti-fovcrop-noros-zshift': ('iassd-sourceonly-kitti2kitti-fovcrop-noros-zshift',
                                                                     {'DATA_CONFIG': 20480, 'DATA_CONFIG_TAR': 20480},
                                                                     ['DATA_CONFIG']),
    'iassd-sourceonly-pb-kitti2kitti-fovcrop-noros-zshift': ('iassd-sourceonly-kitti2kitti-fovcrop-noros-zshift',
                                                             {'DATA_CONFIG': 20480, 'DATA_CONFIG_TAR': 20480}, []),
    'iassd-ringpattern-pb-lyft40-lyft2nuscenes': ('iassd-ringpattern-lyft40-lyft2nuscenes',
                                                  {'LYFT_40BEAM': 20480, 'DATA_CONFIG_TAR': 28672}, []),
    'iassd-global-recount-pb-lyft40-lyft2nuscenes': ('iassd-global-recount-lyft40-lyft2nuscenes',
                                                     {'LYFT_40BEAM': 36864, 'DATA_CONFIG_TAR': 28672}, []),
    'iassd-ringpattern-pb-waymo2nuscenes': ('iassd-ringpattern-waymo2nuscenes',                # P99 < 16,384: unchanged
                                            {'DATA_CONFIG': 16384, 'DATA_CONFIG_TAR': 28672}, []),
    'iassd-global-toponly-recount-pb-waymo2nuscenes': ('iassd-global-toponly-recount-waymo2nuscenes',
                                                       {'DATA_CONFIG': 57344, 'DATA_CONFIG_TAR': 28672}, []),
    'iassd-accum-legaldepth-recount-pb-nuscenes2kitti': ('iassd-accum-legaldepth-recount-nuscenes2kitti',
                                                         {'NUSCENES_N008': 413696, 'NUSCENES_N015': 286720,
                                                          'DATA_CONFIG_TAR': 131072}, []),
    'iassd-accum-legaldepth-recount-pb-nuscenes2kitti-fovtar': ('iassd-accum-legaldepth-recount-nuscenes2kitti-fovtar',
                                                                {'NUSCENES_N008': 413696, 'NUSCENES_N015': 286720,
                                                                 'DATA_CONFIG_TAR': 20480}, []),
}
LAUNCHED_NEVER = ['iassd-sourceonly-pb-kitti2kitti-fovcrop-noros-zshift']   # K budget-only candidate (coordinator 16:06)


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


def node(cfg, block):
    return cfg[block] if block in ('DATA_CONFIG', 'DATA_CONFIG_TAR') else cfg['DATA_CONFIGS'][block]


def budget_entry(procs):
    hits = [p for p in procs if p['NAME'] in BUDGET_STEPS]
    assert len(hits) == 1, [p['NAME'] for p in procs]
    return hits[0]


# ---------------------------------------------------------------------------------------------------- config integrity
@pytest.mark.parametrize('pb', sorted(PINNED))
def test_pb_config_differs_from_its_parent_only_in_the_budget_and_the_declared_recount(pb):
    parent, budgets, recount = PINNED[pb]
    a, b = plain(load(C + pb + '.yaml')), plain(load(C + parent + '.yaml'))
    blocks = ['DATA_CONFIG_TAR'] + (['DATA_CONFIG'] if 'DATA_CONFIG' in a else list(a.get('DATA_CONFIGS', {})))
    assert sorted(budgets) == sorted(blocks), 'every data block gets a PB value'
    for blk in blocks:
        pa, pp = node(a, blk)['DATA_PROCESSOR'], node(b, blk)['DATA_PROCESSOR']
        ba = budget_entry(pa)
        assert ba['NUM_POINTS'] == {'train': budgets[blk], 'test': budgets[blk]}, blk
        assert budgets[blk] % 4096 == 0
        assert budget_entry(pp)['NUM_POINTS'] != ba['NUM_POINTS'] or budgets[blk] == 16384
        names = [p['NAME'] for p in pa]
        if blk != 'DATA_CONFIG_TAR' and pb not in LAUNCHED_NEVER:   # every PB training block recounts (rule PB item 4)
            assert names[names.index(ba['NAME']) + 1] == 'drop_empty_gt_boxes', (blk, names)
            assert pa[names.index(ba['NAME']) + 1].get('MIN_POINTS', 1) == 1
        else:                          # the eval block never recounts; the budget-only K candidate does not either
            assert 'drop_empty_gt_boxes' not in names
        if blk in recount:
            assert 'drop_empty_gt_boxes' not in [p['NAME'] for p in pp]
            pa = [p for p in pa if p['NAME'] != 'drop_empty_gt_boxes']
        budget_entry(pa)['NUM_POINTS'] = budget_entry(pp)['NUM_POINTS']
        node(a, blk)['DATA_PROCESSOR'] = pa
    assert a == b


def test_one_pb_run_serves_s2_a1_and_the_nuscenes_oracle():
    a1 = plain(load(C + 'iassd-sourceonly-recount-pb-nuscenes2kitti.yaml'))
    n = plain(load(C + 'iassd-sourceonly-recount-pb-nuscenes2nuscenes.yaml'))
    for k in ('DATA_CONFIG', 'MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'ONTOLOGY'):
        assert a1[k] == n[k], k
    assert sorted(a1) == sorted(n)
    assert a1['DATA_CONFIG_TAR'] != n['DATA_CONFIG_TAR']
    assert n['DATA_CONFIG_TAR']['DATASET'] == 'NuScenesDataset' and a1['DATA_CONFIG_TAR']['DATASET'] == 'KittiDataset'


def test_crop_twins_crop_the_target_only():
    for twin, base in (('iassd-sourceonly-recount-pb-nuscenes2kitti-fovtar', 'iassd-sourceonly-recount-pb-nuscenes2kitti'),
                       ('iassd-accum-legaldepth-recount-pb-nuscenes2kitti-fovtar',
                        'iassd-accum-legaldepth-recount-pb-nuscenes2kitti')):
        t, b = plain(load(C + twin + '.yaml')), plain(load(C + base + '.yaml'))
        assert t['DATA_CONFIG_TAR']['FOV_POINTS_ONLY'] and t['DATA_CONFIG_TAR']['ALLOW_FOV_POINTS_ONLY']
        assert not b['DATA_CONFIG_TAR']['FOV_POINTS_ONLY']
        t.pop('DATA_CONFIG_TAR'), b.pop('DATA_CONFIG_TAR')
        assert t == b, twin


def test_the_budget_only_k_candidate_is_never_launched():
    script = (TOOLS / 'logs' / 'iassd_pb_20261024' / 'launch_pb.sh')
    if not script.exists():
        pytest.skip('launch script is not in this checkout (tools/logs is not tracked)')
    text = script.read_text()
    for stem in LAUNCHED_NEVER:
        assert stem + '.yaml' not in text


# ------------------------------------------------------------------------------- processor chains against sentinels
def _box(x, label):
    return [x, 0.0, 0.5, 4.0, 2.0, 1.5, 0.0, label]


def _frame(n_background, rng):
    """Boxes at x = 5 (positive, 400 points), 15 (positive, 1 point), 25 / 35 / 45 (ignore -1, -3 and -99, empty or
    near-empty), plus background outside every box. The 1-point positive is the one a budget can empty."""
    bg = np.c_[rng.uniform(-70, 70, n_background), rng.uniform(-70, 70, n_background), rng.uniform(-1.5, 3, n_background)]
    bg = bg[(np.abs(bg[:, 1]) > 3)]
    dense = np.c_[rng.uniform(3.5, 6.5, 400), rng.uniform(-0.8, 0.8, 400), rng.uniform(0, 1, 400)]
    one = np.array([[15.0, 0.0, 0.5]])
    ign = np.array([[35.0, 0.0, 0.5]])
    pts = np.concatenate([bg, dense, one, ign]).astype(np.float32)
    boxes = np.array([_box(5, 1), _box(15, 1), _box(25, -1), _box(35, -3), _box(45, common_utils.IGNORE_CLASS_LABEL)],
                     dtype=np.float32)
    return pts, boxes


def _chain(cfg, blk, training):
    b = node(cfg, blk)
    procs = [p for p in b.DATA_PROCESSOR if p.NAME != 'sample_points_hist_based']   # needs an installed calibration
    return DataProcessor(procs, point_cloud_range=np.array(b.POINT_CLOUD_RANGE, dtype=np.float32), training=training,
                         num_point_features=3)


TRAIN_BLOCKS = sorted({(pb, blk, PINNED[pb][1][blk]) for pb in PINNED for blk in PINNED[pb][1] if blk != 'DATA_CONFIG_TAR'
                       and pb not in LAUNCHED_NEVER})
EVAL_BLOCKS = sorted({(pb, PINNED[pb][1]['DATA_CONFIG_TAR']) for pb in PINNED})


@pytest.mark.parametrize('pb,blk,budget', TRAIN_BLOCKS)
def test_pb_budget_then_recount_drops_only_an_emptied_positive_and_keeps_every_ignore_label(pb, blk, budget):
    rng = np.random.RandomState(0)
    pts, boxes = _frame(int(2.2 * budget), rng)          # about twice the budget: the 1-point box is kept about half the time
    assert len(pts) > budget
    proc = _chain(load(C + pb + '.yaml'), blk, training=True)
    dropped = 0
    for seed in range(16):
        np.random.seed(seed)
        out = proc.forward({'points': pts.copy(), 'gt_boxes': boxes.copy(), 'gt_names': np.array(['Car'] * 5)})
        assert len(out['points']) == budget
        lab = out['gt_boxes'][:, 7]
        assert sorted(lab[lab < 0].tolist()) == sorted([-1, -3, common_utils.IGNORE_CLASS_LABEL])
        assert 5.0 in out['gt_boxes'][:, 0].tolist()
        one_kept = 15.0 in out['gt_boxes'][:, 0].tolist()
        one_has_point = bool((np.abs(out['points'][:, :3] - np.array([15.0, 0.0, 0.5])) < 1e-4).all(axis=1).any())
        assert one_kept == one_has_point, 'a positive is kept iff it still holds a point'
        dropped += not one_kept
    assert 0 < dropped < 16, 'the budget empties the 1-point positive in some draws and not in others'


@pytest.mark.parametrize('pb,blk,budget', TRAIN_BLOCKS)
def test_frame_under_the_pb_budget_is_padded_with_every_real_point_and_box_kept(pb, blk, budget):
    rng = np.random.RandomState(1)
    pts, boxes = _frame(int(0.7 * budget), rng)
    assert len(pts) < budget
    np.random.seed(0)
    out = _chain(load(C + pb + '.yaml'), blk, training=True).forward({'points': pts.copy(), 'gt_boxes': boxes.copy()})
    assert len(out['points']) == budget
    assert len(np.unique(out['points'], axis=0)) == len(np.unique(pts, axis=0))
    assert out['gt_boxes'].shape == boxes.shape


@pytest.mark.parametrize('pb,budget', EVAL_BLOCKS)
def test_eval_block_samples_to_its_pb_budget_and_leaves_boxes_untouched(pb, budget):
    pts, boxes = _frame(int(1.5 * budget), np.random.RandomState(2))
    np.random.seed(0)
    out = _chain(load(C + pb + '.yaml'), 'DATA_CONFIG_TAR', training=False).forward(
        {'points': pts.copy(), 'gt_boxes': boxes.copy()})
    assert len(out['points']) == budget
    assert out['gt_boxes'].shape == boxes.shape


@pytest.mark.parametrize('pb,blk,budget', TRAIN_BLOCKS)
def test_recounted_ignore_boxes_never_become_iassd_positives(pb, blk, budget, monkeypatch):
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
    cfg = load(C + pb + '.yaml')
    bc = cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG
    head = types.SimpleNamespace(num_class=3, box_coder=box_coder_utils.PointResidual_BinOri_Coder(
        angle_bin_num=bc.angle_bin_num, use_mean_size=bc.use_mean_size, mean_size=copy.deepcopy(bc.mean_size)))
    rng = np.random.RandomState(3)
    pts, boxes = _frame(int(0.9 * budget), rng)
    extra = np.concatenate([np.c_[rng.uniform(x - 1.5, x + 1.5, 50), rng.uniform(-0.8, 0.8, 50), rng.uniform(0, 1, 50)]
                            for x in (25.0, 35.0, 45.0)]).astype(np.float32)
    np.random.seed(0)
    out = _chain(cfg, blk, training=True).forward({'points': np.concatenate([pts, extra]), 'gt_boxes': boxes.copy()})
    points = torch.from_numpy(np.c_[np.zeros(len(out['points'])), out['points'][:, :3]].astype(np.float32))
    gt = torch.from_numpy(np.r_[out['gt_boxes'], np.zeros((1, 8), np.float32)][None].astype(np.float32))
    ext = box_utils.enlarge_box3d(gt.view(-1, 8), extra_width=[0.2, 0.2, 0.2]).view(1, -1, 8)
    res = head_mod.IASSD_Head.assign_stack_targets_IASSD(head, points=points, gt_boxes=gt, extend_gt_boxes=ext,
                                                         set_ignore_flag=True, ret_box_labels=True)
    lab = res['point_cls_labels']
    in_ignore = (points[:, 1] > 23) & (points[:, 1] < 47) & (points[:, 2].abs() < 1)
    assert int(in_ignore.sum()) > 0
    assert (lab[in_ignore] <= 0).all()
    assert (lab > 0).any(), 'the dense positive is assigned'
    assert res['gt_box_of_fg_points'].shape[0] == int((lab > 0).sum())
    assert (res['gt_box_of_fg_points'][:, 7] > 0).all()
