"""The KITTI camera-FOV crop stays LOCKED unless a config opts in explicitly (kitti_dataset.py).

Commit 1d20716 (2024-12-23) put an unconditional assert(0) on FOV_POINTS_ONLY so the 360-degree cross-dataset configs
could not switch it on by accident. Since 2026-10-07 a config may crop on purpose by also setting
ALLOW_FOV_POINTS_ONLY: True (first user: the published IA-SSD KITTI recipe, experiments_md/20261006_01). Pinned here:
the lock still holds without the key, and with it every returned point lies in the camera FOV. Needs the KITTI data.
"""
import copy
import logging
import os

import numpy as np
import pytest

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
CFG = 'cfgs/da-ieee-access-pointrcnn/iassd-published-kitti.yaml'


@pytest.fixture
def kitti_cfg():
    if not os.path.isdir(os.path.join(TOOLS, '../data/kitti/training/velodyne')):
        pytest.skip('KITTI data not available')
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        cfg = cfg_from_yaml_file(CFG, EasyDict())
        yield cfg
    finally:
        os.chdir(cwd)


def _dataset(cfg, dcfg):
    from pcdet.datasets import build_dataloader
    log = logging.getLogger('fov_lock'); log.setLevel(logging.WARNING)
    ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=log, training=False, model_ontology=cfg.get('ONTOLOGY', None))
    return ds


def test_crop_without_opt_in_is_still_locked(kitti_cfg):
    d = copy.deepcopy(kitti_cfg.DATA_CONFIG)
    d.pop('ALLOW_FOV_POINTS_ONLY')
    ds = _dataset(kitti_cfg, d)
    with pytest.raises(AssertionError, match='locked'):
        ds[0]


def test_opt_in_returns_only_points_in_the_camera_fov(kitti_cfg):
    d = copy.deepcopy(kitti_cfg.DATA_CONFIG)
    d.DATA_PROCESSOR = [p for p in d.DATA_PROCESSOR if p.NAME == 'mask_points_and_boxes_outside_range']  # no sampling
    d.SHIFT_COOR = None
    ds = _dataset(kitti_cfg, d)
    info = ds.kitti_infos[0]
    idx = info['point_cloud']['lidar_idx']
    raw = ds.get_lidar(idx)
    calib = ds.get_calib(idx)
    fov = ds.get_fov_flag(calib.lidar_to_rect(raw[:, :3]), info['image']['image_shape'], calib)
    pts = ds[0]['points']
    assert 0 < len(pts) <= fov.sum() < len(raw)
    in_fov = ds.get_fov_flag(calib.lidar_to_rect(pts[:, :3]), info['image']['image_shape'], calib)
    assert in_fov.all()
