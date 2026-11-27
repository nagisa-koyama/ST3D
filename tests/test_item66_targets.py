"""Item 66's training target on real KITTI TRAIN frames (experiments_md 20261003_05 §5 item 66, design fixed 2026-11-27 20:55
JST; bug_prevention_checklist rule 4: check TARGETS before a new data path reaches a GPU). The item-66 base config's target
loader (T's KITTI target, 59b's augmentation) must serve the injected pseudo-labels, nothing else, on the KITTI train cloud,
and CenterHead's heatmap must hold one peak per positive. CPU only; reads /home/koyama/data/kitti (skipped if absent)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item66_realpair as m  # noqa: E402

pytestmark = pytest.mark.skipif(not Path('/home/koyama/data/kitti/kitti_infos_train.pkl').exists(), reason='no KITTI')


@pytest.fixture(scope='module')
def target():
    import _init_path  # noqa: F401
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    cfg = m.load_cfg(m.BASE)
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        ds, _, _ = build_dataloader(cfg.DATA_CONFIG_TAR, cfg.CLASS_NAMES, 1, dist=False, workers=0, training=True,
                                    logger=common_utils.create_logger(), model_ontology=cfg.get('ONTOLOGY', None))
        yield ds, cfg
    finally:
        os.chdir(cwd)


def test_frames_are_kitti_train(target):
    ds, _ = target
    assert ds.split == 'train' and ds.root_split_path.name == 'training' and len(ds.kitti_infos) == 3712


def _inject(ds, cfg, idxs):
    rng = np.random.default_rng(0)
    labels = {}
    for idx in idxs:
        fid = ds.kitti_infos[idx]['point_cloud']['lidar_idx']
        pts = ds.get_lidar(fid)
        pts = pts[(pts[:, 0] > 5) & (pts[:, 0] < 40) & (np.abs(pts[:, 1]) < 10)]
        b = np.zeros((3, 9), dtype=np.float32)
        b[:, 0:3] = pts[rng.choice(len(pts), 3, replace=False), :3] + np.asarray(cfg.DATA_CONFIG_TAR.SHIFT_COOR)
        b[:, 3:6] = [4.0, 4.0, 3.0]
        b[:, 7] = [1, 2, 3]                                   # SCORE = NEG in item 66: positives only, no ignore label
        b[:, 8] = [0.9, 0.8, 0.5]
        labels[fid] = {'gt_boxes': b}
    return labels


def test_boxes_are_the_injected_pseudo_labels_and_points_are_the_train_cloud(target):
    ds, cfg = target
    queue = ds.data_augmentor.data_augmentor_queue
    ds.data_augmentor.data_augmentor_queue = []               # identity, so the boxes can be compared exactly
    try:
        labels = _inject(ds, cfg, (0, 1, 2))
        ds.pseudo_labels = labels
        for idx in (0, 1, 2):
            fid = ds.kitti_infos[idx]['point_cloud']['lidar_idx']
            raw = ds.get_lidar(fid)
            assert str(ds.root_split_path / 'velodyne' / ('%s.bin' % fid)).find('/training/') > 0
            d = ds[idx]
            assert d['frame_id'] == fid and 'num_points_in_gt' not in d
            got, want = d['gt_boxes'], labels[fid]['gt_boxes']
            assert got.shape[0] == 3                           # the KITTI annotations never enter
            assert np.allclose(got[:, :7], want[:, :7], atol=1e-5)
            assert list(got[:, 7]) == [1, 2, 3]
            assert 0 < d['voxel_num_points'].sum() <= raw.shape[0]
    finally:
        ds.data_augmentor.data_augmentor_queue = queue
        ds.pseudo_labels = None


def test_heatmap_peaks_equal_positives(target):
    import plmatch_labels as pm
    ds, cfg = target
    queue = ds.data_augmentor.data_augmentor_queue
    ds.data_augmentor.data_augmentor_queue = []
    try:
        ds.pseudo_labels = _inject(ds, cfg, (0, 1, 2))
        head, side, _, _ = pm.make_head(cfg)
        for idx in (0, 1, 2):
            d = ds[idx]
            t = pm.targets(head, side, d['gt_boxes'])
            lab = np.asarray(d['gt_boxes'])[:, 7].astype(int)
            for c in range(3):
                assert int(t['heatmaps'][0][0][c].eq(1).sum()) == int((lab == c + 1).sum())
    finally:
        ds.data_augmentor.data_augmentor_queue = queue
        ds.pseudo_labels = None
