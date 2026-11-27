"""KITTI's loader under the two frozen-file keys (PL-MATCH, PL-GT): the served dict carries no `num_points_in_gt`, and two
frames with different box counts collate (experiments_md 20261003_05 §5 item 61c, defect found 2026-11-27 by 61c's
identity check, 37283 / 37317).

Both keys hand prepare_data a per-box count for its zero-point filter (DatasetTemplate.fill_pseudo_labels). Waymo, Lyft and
PandaSet pop it after prepare_data; KITTI did not, so the count (pre-filter length) reached collate_batch, whose fallback
np.stack fails on frames with different box counts: a KITTI-target PL-MATCH / PL-GT row would have died at its first
batch, and the served dict differed from the plain pseudo-label path's by that key. CPU only; reads /home/koyama/data/kitti
(skipped if absent). Run from the repo root."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item61c_moved as m  # noqa: E402

pytestmark = pytest.mark.skipif(not Path('/home/koyama/data/kitti/kitti_infos_test.pkl').exists(), reason='no KITTI')


def _labels(ds, shift, counts, recount):
    """Frame id -> one frozen entry per frame: boxes centred on real points (so none is emptied), `counts[k]` boxes."""
    from pcdet.utils import self_training_utils as stu
    rng = np.random.default_rng(0)
    out = {}
    for idx, n in enumerate(counts):
        fid = ds.kitti_infos[idx]['point_cloud']['lidar_idx']
        pts = ds.get_lidar(fid)
        pts = pts[(pts[:, 0] > 5) & (pts[:, 0] < 40) & (np.abs(pts[:, 1]) < 10)]
        b = np.zeros((n, 9), dtype=np.float32)
        b[:, 0:3] = pts[rng.choice(len(pts), n, replace=False), :3] + np.asarray(shift)
        b[:, 3:6] = [4.0, 4.0, 3.0]
        b[:, 6] = 0.0
        b[:, 7] = 1
        b[:, 8] = 0.9
        out[fid] = {'gt_boxes': b, 'cls_scores': b[:, 8].copy(), 'iou_scores': b[:, 8].copy(),
                    'memory_counter': np.zeros(n), 'teacher_classes': np.array(['Car'] * n),
                    'num_points_in_gt': np.full(n, stu.PL_MATCH_RECOUNT if recount else 50, dtype=np.int64)}
    return out


@pytest.mark.parametrize('which', ['pl_match', 'pl_gt'])
def test_served_dict_has_no_count_and_frames_with_different_box_counts_collate(which):
    import _init_path  # noqa: F401
    from pcdet.utils import common_utils
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        cfg = m.load_cfg(m.ROW_CFG % 666)                    # 61c's row: KITTI testing clouds, PL-MATCH key set
        tar = cfg.DATA_CONFIG_TAR
        if which == 'pl_gt':
            del tar[common_utils.ANALYSIS_TARGET_PL_MATCH_KEY]
            tar[common_utils.ANALYSIS_TARGET_GT_KEY] = '/nonexistent/only-the-key-matters.pkl'
        ds = m.train_dataset(cfg)
        ds.data_augmentor.data_augmentor_queue = []           # identity, so the boxes can be compared exactly
        labels = _labels(ds, tar.SHIFT_COOR, (2, 5), recount=(which == 'pl_match'))
        ds.set_pseudo_labels(labels)
        served = [ds[0], ds[1]]
        for d, n in zip(served, (2, 5)):
            assert 'num_points_in_gt' not in d, sorted(d)
            assert len(d['gt_boxes']) == n                    # every box kept: the count still reached the filter
        batch = ds.collate_batch(served)
        assert batch['batch_size'] == 2 and batch['gt_boxes'].shape[:2] == (2, 5)
        assert 'num_points_in_gt' not in batch
    finally:
        os.chdir(cwd)


def test_the_count_still_drives_the_zero_point_filter_under_pl_gt():
    """The pop comes AFTER prepare_data: a PL-GT box whose stored count is 0 is still dropped by the zero-point filter."""
    import _init_path  # noqa: F401
    from pcdet.utils import common_utils
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        cfg = m.load_cfg(m.ROW_CFG % 666)
        tar = cfg.DATA_CONFIG_TAR
        del tar[common_utils.ANALYSIS_TARGET_PL_MATCH_KEY]
        tar[common_utils.ANALYSIS_TARGET_GT_KEY] = '/nonexistent/only-the-key-matters.pkl'
        ds = m.train_dataset(cfg)
        ds.data_augmentor.data_augmentor_queue = []
        labels = _labels(ds, tar.SHIFT_COOR, (3,), recount=False)
        fid = next(iter(labels))
        labels[fid]['num_points_in_gt'][1] = 0                # the stored count says empty, though points lie inside
        ds.set_pseudo_labels(labels)
        d = ds[0]
        assert len(d['gt_boxes']) == 2 and 'num_points_in_gt' not in d
    finally:
        os.chdir(cwd)
