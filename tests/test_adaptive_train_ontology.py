"""adaptive_train.py must hand every loader a model_ontology, or a source can lose all its labels.

It used to call build_dataloader without one. With the da-ieee-access UADA3D configs
(ONTOLOGY 'kitti', CLASS_NAMES ['Car', 'Pedestrian', 'Cyclist']) that left Lyft's names lowercase,
and keep_arrays_by_name() in prepare_data then dropped every box: the Lyft UADA3D row trained with
ZERO source ground truth, and its loss fell to the discriminator's alone (0.036 on job 25928, while
the KITTI row - already capitalised - sat near 10). Same defect as train.py's cc5d69a.
experiments_md/20260926_05 section 6a.

The sibling test for train.py re-implements the selection rule. That cannot catch a call site that
never passes the value, which is exactly what this bug was - so the end-to-end test here drives
adaptive_train.py's own loader construction on the real config and counts boxes.
"""
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parent.parent / 'tools'
sys.path.insert(0, str(TOOLS))

import adaptive_train  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

UADA3D_CONFIGS = ['cfgs/da-ieee-access/centerpoint-uada3d-lyft2nuscenes.yaml',
                  'cfgs/da-ieee-access/centerpoint-uada3d-kitti2nuscenes.yaml']


def _load(path):
    cfg = EasyDict()
    prev = os.getcwd()
    os.chdir(TOOLS)
    try:
        cfg_from_yaml_file(path, cfg)
    finally:
        os.chdir(prev)
    return cfg


def test_ontology_rule_matches_train_py():
    assert adaptive_train.model_ontologies(EasyDict({'ONTOLOGY': 'kitti'})) == ('kitti', 'kitti')
    assert adaptive_train.model_ontologies(
        EasyDict({'ONTOLOGY': 'kitti', 'EVAL_ONTOLOGY': 'nuscenes'})) == ('kitti', 'nuscenes')
    # Configs that declare no ontology keep exactly their old behaviour.
    assert adaptive_train.model_ontologies(EasyDict()) == (None, None)


@pytest.mark.parametrize('path', UADA3D_CONFIGS)
def test_uada3d_configs_declare_an_ontology(path):
    """The fix only helps configs that say what vocabulary their CLASS_NAMES are in."""
    assert adaptive_train.model_ontologies(_load(path))[0] == 'kitti'


def _source_infos(path):
    """The source's training infos file, resolved the way LyftDataset resolves it."""
    data = _load(path).DATA_CONFIG
    return TOOLS / data.DATA_PATH / data.get('VERSION', '') / data.INFO_PATH['train'][0]


@pytest.mark.skipif(not _source_infos(UADA3D_CONFIGS[0]).exists(),
                    reason='needs the Lyft infos on disk')
def test_lyft_source_keeps_its_boxes(monkeypatch):
    """End to end through adaptive_train.build_domain_dataloaders on the real Lyft config."""
    monkeypatch.chdir(TOOLS)
    cfg = _load(UADA3D_CONFIGS[0])
    args = Namespace(batch_size=12, workers=0, merge_all_iters_to_one_epoch=False, epochs=1)
    logger = common_utils.create_logger(None, rank=0)
    (source_set, _, _), (target_set, _, _) = adaptive_train.build_domain_dataloaders(
        cfg, args, False, logger)

    boxes = [len(source_set[i]['gt_boxes']) for i in range(8)]
    assert sum(boxes) > 0, (
        'Lyft source frames carry no boxes (%s): its names were not mapped to CLASS_NAMES' % boxes)
    assert source_set.map_ontology_dataset_to_model is not None
    assert target_set.map_ontology_dataset_to_model is not None

    test_set, _, _ = adaptive_train.build_eval_dataloader(cfg, args, False, logger)
    assert test_set.map_ontology_dataset_to_model is not None
