"""Rule PB row (d), PandaSet -> nuScenes IA-SSD P / C1-analog (experiments_md 20261024_06; T05 §5 rule PB item 11, fixed
10-24 18:42 JST, experiments_md 932dc88): config integrity and the sentinel behaviour of every new block's processor chain
(memory/repo/bug_prevention_checklist.md rules 1-2). The same checks as tests/test_iassd_pb_configs.py (rows (a)-(c)),
on this row's four files; that file is not edited.

- The 16,384 twins take the CenterPoint parents' data blocks (28034; 28041 / 28280) unchanged but for the processor list
  (CenterPoint's voxel processor -> IA-SSD's budget, recount, shuffle) and, on the C1-analog, HIST_DIST_BEFORE_POINT_BUDGET
  (train.py requires it once a budget follows the thinning). Their target block is the nuScenes block of every
  nuScenes-target IA-SSD row, and MODEL / OPTIMIZATION / CLASS_NAMES / ONTOLOGY are iassd-sourceonly-pandaset.yaml's.
- Each PB config differs from its twin ONLY in each data block's budget (NUM_POINTS, both mode keys), pinned to the census
  (P99 of 1,000 strided TRAIN frames entering the budget step, the smallest multiple of 4,096 at or above it).
- The pair differs in the selection rule only.
- Processor chains on synthetic frames: a positive box the budget empties is dropped and one it keeps is kept; ignore
  labels (-1, -3, IGNORE_CLASS_LABEL -99) are kept whatever they hold and never become IA-SSD positives; a frame under the
  budget is padded to it with every real point and box kept; evaluation is untouched.
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
CP = 'cfgs/da-ieee-access/'
BUDGET_STEPS = ('sample_points', 'sample_points_by_voxels')

# 16,384 twin -> (CenterPoint parent, keys the twin adds to the parent's DATA_CONFIG, processor names per block)
TWINS = {
    'iassd-ringpattern-pandaset2nuscenes': (
        'centerpoint-ringpattern-pandaset2nuscenes', [],
        ['mask_points_and_boxes_outside_range', 'sample_points', 'drop_empty_gt_boxes', 'shuffle_points']),
    'iassd-global-fovcut-recount-pandaset2nuscenes': (
        'centerpoint-global-fovcut-recount-pandaset2nuscenes', ['HIST_DIST_BEFORE_POINT_BUDGET'],
        ['mask_points_and_boxes_outside_range', 'sample_points_hist_based', 'sample_points', 'drop_empty_gt_boxes',
         'shuffle_points']),
}
TAR_PROCS = ['mask_points_and_boxes_outside_range', 'sample_points', 'shuffle_points']
# PB config -> (its 16,384 twin, {block: PB budget}). Census: logs/iassd_pb_20261024/pandaset/census (jobs 30419 /
# 30420); the nuScenes eval block: 20261024_05 §2 (28,672).
PINNED = {
    'iassd-ringpattern-pb-pandaset2nuscenes': ('iassd-ringpattern-pandaset2nuscenes',            # P99 19,343
                                               {'DATA_CONFIG': 20480, 'DATA_CONFIG_TAR': 28672}),
    'iassd-global-fovcut-recount-pb-pandaset2nuscenes': ('iassd-global-fovcut-recount-pandaset2nuscenes',  # P99 40,048
                                                         {'DATA_CONFIG': 40960, 'DATA_CONFIG_TAR': 28672}),
}
VOXEL_FALLBACK = set()   # (PB config, block) sent to the cost fallback by the timing smoke (rule PB item 5): none so far
NUSCENES_PB_TARGET = 'iassd-ringpattern-pb-waymo2nuscenes'   # a PB row whose nuScenes eval block is 28,672 (13a7988)
LAUNCH = TOOLS / 'logs' / 'iassd_pb_20261024' / 'launch_pb_pandaset.sh'


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


def budget_entry(procs):
    hits = [p for p in procs if p['NAME'] in BUDGET_STEPS]
    assert len(hits) == 1, [p['NAME'] for p in procs]
    return hits[0]


def without_processor(block):
    return {k: v for k, v in block.items() if k != 'DATA_PROCESSOR'}


# ---------------------------------------------------------------------------------------------------- config integrity
@pytest.mark.parametrize('twin', sorted(TWINS))
def test_twin_takes_the_centerpoint_parents_data_blocks_but_the_processor(twin):
    parent, added, procs = TWINS[twin]
    t, p = plain(load(C + twin + '.yaml')), plain(load(CP + parent + '.yaml'))
    td = without_processor(t['DATA_CONFIG'])
    for k in added:
        assert td.pop(k) is True, k
    assert td == without_processor(p['DATA_CONFIG'])
    assert without_processor(t['DATA_CONFIG_TAR']) == without_processor(p['DATA_CONFIG_TAR'])
    assert t['CLASS_NAMES'] == p['CLASS_NAMES'] and t['ONTOLOGY'] == p['ONTOLOGY']
    assert [q['NAME'] for q in t['DATA_CONFIG']['DATA_PROCESSOR']] == procs
    assert [q['NAME'] for q in t['DATA_CONFIG_TAR']['DATA_PROCESSOR']] == TAR_PROCS
    for blk in ('DATA_CONFIG', 'DATA_CONFIG_TAR'):
        assert budget_entry(t[blk]['DATA_PROCESSOR'])['NUM_POINTS'] == {'train': 16384, 'test': 16384}
    names = procs
    assert names[names.index('sample_points') + 1] == 'drop_empty_gt_boxes'   # the recount right after the budget
    assert t['DATA_CONFIG']['DATA_PROCESSOR'][names.index('drop_empty_gt_boxes')]['MIN_POINTS'] == 1
    # the parents' own processor parameters carried over (range mask, thinning floor)
    pp = {q['NAME']: q for q in p['DATA_CONFIG']['DATA_PROCESSOR']}
    tp = {q['NAME']: q for q in t['DATA_CONFIG']['DATA_PROCESSOR']}
    assert tp['mask_points_and_boxes_outside_range'] == pp['mask_points_and_boxes_outside_range']
    if 'sample_points_hist_based' in pp:
        assert tp['sample_points_hist_based'] == pp['sample_points_hist_based']


@pytest.mark.parametrize('twin', sorted(TWINS))
def test_twin_model_block_is_iassd_sourceonly_pandaset(twin):
    t, ref = plain(load(C + twin + '.yaml')), plain(load(C + 'iassd-sourceonly-pandaset.yaml'))
    for k in ('MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'ONTOLOGY'):
        assert t[k] == ref[k], k
    assert t['MODEL']['POINT_HEAD']['TARGET_CONFIG']['BOX_CODER_CONFIG']['mean_size'] == [
        [4.51, 1.92, 1.69], [0.85, 0.82, 1.80], [2.14, 0.71, 1.42]]   # PandaSet SOURCE train means (job 28124)
    assert t['OPTIMIZATION']['NUM_EPOCHS'] == 115 and t['OPTIMIZATION']['LR'] == 0.01   # item 11
    assert t['DATA_CONFIG']['DATASET'] == 'PandasetDataset' and t['DATA_CONFIG']['LIDAR_DEVICE'] == 0
    assert t['DATA_CONFIG']['SHIFT_COOR'] == [0.0, 0.0, 0.3]
    assert t['DATA_CONFIG_TAR']['DATASET'] == 'NuScenesDataset'


def test_twin_target_block_is_the_nuscenes_block_of_the_other_iassd_rows():
    ref = plain(load(C + 'iassd-sourceonly-waymo2nuscenes.yaml'))['DATA_CONFIG_TAR']   # the census stem of 28,672
    for twin in TWINS:
        assert plain(load(C + twin + '.yaml'))['DATA_CONFIG_TAR'] == ref, twin


@pytest.mark.parametrize('pb', sorted(PINNED))
def test_pb_config_differs_from_its_twin_only_in_the_budget(pb):
    twin, budgets = PINNED[pb]
    a, b = plain(load(C + pb + '.yaml')), plain(load(C + twin + '.yaml'))
    for blk in ('DATA_CONFIG', 'DATA_CONFIG_TAR'):
        pa, pp = a[blk]['DATA_PROCESSOR'], b[blk]['DATA_PROCESSOR']
        assert budgets[blk] is not None and budgets[blk] % 4096 == 0, blk
        names = [p['NAME'] for p in pa]
        if (pb, blk) in VOXEL_FALLBACK:
            raise AssertionError('no fallback block is pinned yet; extend this test with the (a)-(c) fallback check')
        assert 'sample_points_by_voxels' not in names
        assert budget_entry(pa)['NUM_POINTS'] == {'train': budgets[blk], 'test': budgets[blk]}, blk
        if blk == 'DATA_CONFIG':
            assert names[names.index('sample_points') + 1] == 'drop_empty_gt_boxes', names
        else:
            assert 'drop_empty_gt_boxes' not in names
        budget_entry(pa)['NUM_POINTS'] = budget_entry(pp)['NUM_POINTS']
    assert a == b


def test_pb_target_block_is_the_measured_nuscenes_pb_block():
    ref = plain(load(C + NUSCENES_PB_TARGET + '.yaml'))['DATA_CONFIG_TAR']
    for pb in PINNED:
        assert plain(load(C + pb + '.yaml'))['DATA_CONFIG_TAR'] == ref, pb


def test_the_pair_differs_in_the_selection_rule_only():
    p = plain(load(C + 'iassd-ringpattern-pb-pandaset2nuscenes.yaml'))
    c1 = plain(load(C + 'iassd-global-fovcut-recount-pb-pandaset2nuscenes.yaml'))
    pd, cd = dict(p['DATA_CONFIG']), dict(c1['DATA_CONFIG'])
    ring = pd.pop('RING_PATTERN')
    cut = cd.pop('RING_FOV_CUT')
    assert cut == {k: ring[k] for k in ('LABEL_CACHE', 'SPACING_DEG', 'TARGET_FOV_DEG')}   # same channels' FOV cut
    for k in ('HIST_DIST_ON_THE_FLY', 'HIST_DIST_FRAMES', 'HIST_DIST_BINS', 'HIST_DIST_MAX_DIST',
              'HIST_DIST_BEFORE_POINT_BUDGET'):
        assert k in cd and k not in pd, k
        cd.pop(k)
    pp, cp = pd.pop('DATA_PROCESSOR'), cd.pop('DATA_PROCESSOR')
    cp = [q for q in cp if q['NAME'] != 'sample_points_hist_based']
    assert [q['NAME'] for q in cp] == [q['NAME'] for q in pp]
    for a, b in zip(pp, cp):   # every step but the budget value (each block's own PB value) is the same
        if a['NAME'] != 'sample_points':
            assert a == b, a['NAME']
    assert pd == cd
    for k in set(p) - {'DATA_CONFIG'}:
        assert p[k] == c1[k], k


def test_the_16384_twins_are_never_launched():
    if not LAUNCH.exists():
        pytest.skip('launch script is not in this checkout (tools/logs is not tracked)')
    text = LAUNCH.read_text()
    for twin in TWINS:
        assert twin + '.yaml' not in text, twin
    for pb in PINNED:
        assert pb + '.yaml' in text, pb


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
    b = cfg[blk]
    procs = [p for p in b.DATA_PROCESSOR if p.NAME != 'sample_points_hist_based']   # needs an installed calibration
    return DataProcessor(procs, point_cloud_range=np.array(b.POINT_CLOUD_RANGE, dtype=np.float32), training=training,
                         num_point_features=3)


TRAIN_BLOCKS = sorted((pb, 'DATA_CONFIG', PINNED[pb][1]['DATA_CONFIG']) for pb in PINNED)
EVAL_BLOCKS = sorted((pb, PINNED[pb][1]['DATA_CONFIG_TAR']) for pb in PINNED)


def test_hist_based_thinning_is_a_no_op_without_a_calibration_and_keeps_points_by_their_bin_rate():
    """The C1-analog's thinning is left out of the chains below. Without an installed calibration it returns the frame
    untouched (train.py installs one before the loaders start); with one it keeps each point independently with its bin's
    rate (a Bernoulli draw per point), so the CenterPoint parent's shuffle before it changed nothing but the RNG stream."""
    cfg = load(C + 'iassd-global-fovcut-recount-pb-pandaset2nuscenes.yaml')
    b = cfg.DATA_CONFIG
    proc = DataProcessor(b.DATA_PROCESSOR, point_cloud_range=np.array(b.POINT_CLOUD_RANGE, dtype=np.float32),
                         training=True, num_point_features=3)
    pts, boxes = _frame(4000, np.random.RandomState(5))
    step = [p for p in proc.data_processor_queue if p.func.__name__ == 'sample_points_hist_based'][0]
    out = step({'points': pts.copy(), 'gt_boxes': boxes.copy()})
    assert np.array_equal(out['points'], pts)
    src = np.full(50, 2.0)
    src[:10] = 4.0                                                           # rate 0.25 inside 15 m, 0.5 beyond
    proc.set_hist_dist(src, np.full(50, 1.0), max_dist=75.0)
    dist = np.linalg.norm(pts[:, :2], axis=1)
    for perm in (np.arange(len(pts)), np.random.RandomState(6).permutation(len(pts))):
        np.random.seed(0)
        kept = step({'points': pts[perm].copy()})['points']
        kd = np.linalg.norm(kept[:, :2], axis=1)
        for lo, hi, rate in ((0, 15, 0.25), (15, 75, 0.5)):
            n = int(((dist >= lo) & (dist < hi)).sum())
            k = int(((kd >= lo) & (kd < hi)).sum())
            assert abs(k - rate * n) < 5 * np.sqrt(n * rate * (1 - rate)) + 1, (lo, hi, k, n)


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
