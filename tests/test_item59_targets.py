"""Item 59b's training target on real KITTI testing frames (experiments_md 20261003_05 §5 item 59, design fixed 2026-11-27
08:15 JST; bug_prevention_checklist rule 4: check TARGETS before a new data path reaches a GPU). CPU only; reads
/home/koyama/data/kitti (skipped if absent). Run from the repo root."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item59_bn_split as m  # noqa: E402

pytestmark = pytest.mark.skipif(not Path('/home/koyama/data/kitti/kitti_infos_test.pkl').exists(), reason='no KITTI')


@pytest.fixture(scope='module')
def target():
    import _init_path  # noqa: F401
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    cwd = os.getcwd()
    os.chdir(TOOLS)                  # DATA_PATH ../data/kitti is relative to tools/, at build time and at every read
    try:
        cfg = cfg_from_yaml_file(m.B_BASE, EasyDict())
        ds, _, _ = build_dataloader(cfg.DATA_CONFIG_TAR, cfg.CLASS_NAMES, 1, dist=False, workers=0, training=True,
                                    logger=common_utils.create_logger(), model_ontology=cfg.get('ONTOLOGY', None))
        yield ds, cfg
    finally:
        os.chdir(cwd)


def test_frames_are_kitti_testing_and_carry_no_labels(target):
    ds, _ = target
    assert ds.split == 'test' and ds.root_split_path.name == 'testing' and len(ds.kitti_infos) == 7518
    assert not any('annos' in i for i in ds.kitti_infos)


def test_boxes_are_the_injected_pseudo_labels_and_points_are_the_testing_cloud(target):
    ds, cfg = target
    queue = ds.data_augmentor.data_augmentor_queue
    ds.data_augmentor.data_augmentor_queue = []            # identity, so the boxes can be compared exactly
    try:
        rng = np.random.default_rng(0)
        labels = {}
        for idx in (0, 1, 2):
            fid = ds.kitti_infos[idx]['point_cloud']['lidar_idx']
            pts = ds.get_lidar(fid)
            pts = pts[(pts[:, 0] > 5) & (pts[:, 0] < 40) & (np.abs(pts[:, 1]) < 10)]
            b = np.zeros((3, 9), dtype=np.float32)
            # centred on real points (the zero-point GT filter drops an empty box, as it does any pseudo-label), in the
            # loader frame after SHIFT_COOR
            b[:, 0:3] = pts[rng.choice(len(pts), 3, replace=False), :3] + np.asarray(cfg.DATA_CONFIG_TAR.SHIFT_COOR)
            b[:, 3:6] = [4.0, 4.0, 3.0]
            b[:, 7] = [1, 2, -1]                                              # Car, Pedestrian, an ignored Car
            b[:, 8] = [0.9, 0.8, 0.5]
            labels[fid] = {'gt_boxes': b}
        ds.pseudo_labels = labels
        for idx in (0, 1, 2):
            fid = ds.kitti_infos[idx]['point_cloud']['lidar_idx']
            raw = ds.get_lidar(fid)
            assert str(ds.root_split_path / 'velodyne' / ('%s.bin' % fid)).find('/testing/') > 0
            d = ds[idx]
            assert d['frame_id'] == fid
            got, want = d['gt_boxes'], labels[fid]['gt_boxes']
            assert got.shape[0] == 3                                         # nothing added, nothing taken from elsewhere
            assert np.allclose(got[:, :7], want[:, :7], atol=1e-5)
            assert list(got[:, 7]) == [1, 2, -1]                             # class column: ignore sign kept for CenterHead
            n_in = d['voxel_num_points'].sum()
            assert 0 < n_in <= raw.shape[0]
    finally:
        ds.data_augmentor.data_augmentor_queue = queue
        ds.pseudo_labels = None
