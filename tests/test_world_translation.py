"""random_world_translation, the SECOND-IoU loss unpack, and the two oracle rows built on them
(experiments_md/20260930_07 section 6).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets.augmentor.data_augmentor import DataAugmentor  # noqa: E402

TOOLS = ROOT / 'tools'
NOROS = 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros.yaml'
ZSHIFT = 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml'
SECONDIOU = 'cfgs/da-ieee-access/secondiou-oracle-kitti-st3drecipe.yaml'


@pytest.fixture
def in_tools_dir(monkeypatch):
    monkeypatch.chdir(TOOLS)


def _load(path):
    cfg = EasyDict()
    cfg_from_yaml_file(path, cfg)
    return cfg


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-20, 20, (500, 4)).astype(np.float32)
    boxes = np.concatenate([rng.uniform(-20, 20, (6, 3)), rng.uniform(1, 4, (6, 3)),
                            rng.uniform(-np.pi, np.pi, (6, 1))], 1).astype(np.float32)
    return {'points': points, 'gt_boxes': boxes}


def _translate(data, std):
    return DataAugmentor.random_world_translation(None, data_dict=data, config={'NOISE_TRANSLATE_STD': std})


def test_one_offset_moves_points_and_boxes_together():
    np.random.seed(3)
    before = _scene()
    after = _translate({k: v.copy() for k, v in before.items()}, [0.5, 0.5, 0.5])
    dp = after['points'][:, :3] - before['points'][:, :3]
    db = after['gt_boxes'][:, :3] - before['gt_boxes'][:, :3]
    assert np.allclose(dp, dp[0], atol=1e-5), 'every point must move by the same vector'
    assert np.allclose(db, dp[0], atol=1e-5), 'boxes must move with their points'
    assert np.abs(dp[0]).max() > 0
    assert np.array_equal(after['points'][:, 3], before['points'][:, 3]), 'features untouched'
    assert np.array_equal(after['gt_boxes'][:, 3:], before['gt_boxes'][:, 3:]), 'size and heading untouched'


def test_z_only_std_moves_height_only():
    np.random.seed(4)
    before = _scene()
    after = _translate({k: v.copy() for k, v in before.items()}, [0.0, 0.0, 0.3])
    d = after['points'][0, :3] - before['points'][0, :3]
    assert d[0] == 0 and d[1] == 0 and d[2] != 0


def test_offsets_follow_the_std():
    np.random.seed(5)
    zs = [(_translate(_scene(), [0.0, 0.0, 0.3])['points'][0, 2] - _scene()['points'][0, 2]) for _ in range(2000)]
    assert np.std(zs) == pytest.approx(0.3, rel=0.1)
    assert abs(np.mean(zs)) < 0.03


def test_std_must_have_three_axes():
    with pytest.raises(AssertionError):
        _translate(_scene(), [0.3])


def test_zshift_row_differs_from_its_parent_only_by_the_translation(in_tools_dir):
    parent, child = _load(NOROS), _load(ZSHIFT)
    p, c = parent.DATA_CONFIG.DATA_AUGMENTOR, child.DATA_CONFIG.DATA_AUGMENTOR
    assert c.DISABLE_AUG_LIST == p.DISABLE_AUG_LIST
    assert [dict(x) for x in c.AUG_CONFIG_LIST[:-1]] == [dict(x) for x in p.AUG_CONFIG_LIST]
    assert c.AUG_CONFIG_LIST[-1].NAME == 'random_world_translation'
    assert list(c.AUG_CONFIG_LIST[-1].NOISE_TRANSLATE_STD) == [0.0, 0.0, 0.3]
    for key in ('NUM_EPOCHS', 'BATCH_SIZE_PER_GPU', 'NUM_EPOCHS_TO_EVAL', 'LR'):
        assert child.OPTIMIZATION[key] == parent.OPTIMIZATION[key], key
    assert child.MODEL == parent.MODEL
    assert child.DATA_CONFIG_TAR == parent.DATA_CONFIG_TAR, 'evaluation must be unchanged'


def test_secondiou_oracle_is_upstream_plus_the_forced_changes(in_tools_dir):
    cfg = _load(SECONDIOU)
    upstream = _load('cfgs/kitti_models/secondiou_orcale.yaml')
    assert cfg.MODEL == upstream.MODEL and cfg.MODEL.NAME == 'SECONDNetIoU'
    assert list(cfg.CLASS_NAMES) == ['Car']
    d = cfg.DATA_CONFIG
    assert d.FOV_POINTS_ONLY is False, 'True hits the assert(0) canary in kitti_dataset.py'
    assert d.ONTOLOGY == 'kitti' and cfg.get('ONTOLOGY') is None, 'dataset lookup key, no class remap'
    assert d.TEST.BOX_FILTER['FOV_FILTER'] is True
    assert list(d.POINT_CLOUD_RANGE) == list(upstream.DATA_CONFIG.POINT_CLOUD_RANGE) == [0, -75.2, -4, 75.2, 75.2, 2]
    assert d.DATA_AUGMENTOR == upstream.DATA_CONFIG.DATA_AUGMENTOR
    assert cfg.OPTIMIZATION.NUM_EPOCHS == 80 and cfg.OPTIMIZATION.NUM_EPOCHS_TO_EVAL == 1


def test_secondiou_unpacks_the_three_value_anchor_head_loss():
    """AnchorHeadTemplate.get_loss returns (loss, tb_dict, domain_loss); unpacking two raised on
    the first iteration of every SECOND-IoU config."""
    from pcdet.models.detectors.second_net_iou import SECONDNetIoU
    head = SimpleNamespace(get_loss=lambda weights=None: (1.0, {'rpn_loss': 1.0}, None))
    roi = SimpleNamespace(get_loss=lambda tb: (0.5, dict(tb, rcnn_loss=0.5)))
    loss, tb, _ = SECONDNetIoU.get_training_loss(SimpleNamespace(dense_head=head, roi_head=roi))
    assert loss == 1.5 and tb == {'rpn_loss': 1.0, 'rcnn_loss': 0.5}
