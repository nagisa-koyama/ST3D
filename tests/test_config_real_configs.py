"""Oracle test: the repo's own multi-level configs must resolve to complete dataset blocks.

The `tmp_path`-based tests in test_config_merge.py pin the merge *semantics*. This file pins
the thing that actually broke in production: a `da-MIRU2025`-family config reaches its dataset
through `DATA_CONFIGS.<NAME>._BASE_CONFIG_` -> a dataset yaml with its own `_BASE_CONFIG_` ->
a preprocessing yaml. Between `de9f9d7` (2026-08-23) and 2026-09-21 that second hop was never
opened, so `DATA_PROCESSOR` and `POINT_CLOUD_RANGE` silently vanished and the density
correction (`sample_points_hist_based`) was not applied - while the config still "loaded fine".

See experiments_md/20260921_01_base_config_recursion_regression_fix.md.
"""
import os
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

TOOLS_DIR = Path(__file__).resolve().parent.parent / 'tools'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.config import cfg_from_yaml_file  # noqa: E402


@pytest.fixture
def in_tools_dir():
    """`_BASE_CONFIG_` paths are relative to tools/, which is where train.py is always run."""
    prev = os.getcwd()
    os.chdir(TOOLS_DIR)
    try:
        yield
    finally:
        os.chdir(prev)


def _dataset_blocks(cfg):
    for name in ('DATA_CONFIG', 'DATA_CONFIG_TAR'):
        if name in cfg:
            yield name, cfg[name]
    for name, blk in cfg.get('DATA_CONFIGS', {}).items():
        yield 'DATA_CONFIGS.%s' % name, blk


# (config, expect_density_correction) - one representative per affected family.
MULTI_LEVEL_CONFIGS = [
    ('cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2nuscenes_car_ped_default.yaml', False),
    ('cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2nuscenes_car_ped_point_calibrated.yaml', True),
    ('cfgs/da-MIRU2025/second_old_anchor_basebev_multi_nuscenes2kitti_car_ped_point_label_calibrated.yaml', True),
    ('cfgs/da-post-MIRU2025/second_old_anchor_st3d_basebev_multi_lyft2nuscenes_dann_source_target_car_ped_point_label_calibrated.yaml', True),
] + [
    # The IEEE Access source-only family: DATA_CONFIG._BASE_CONFIG_ -> da_<source>_dataset.yaml,
    # which has its own _BASE_CONFIG_ -> preprocess_sourceonly.yaml. Same two-hop shape as
    # da-MIRU2025, so the same regression would empty it the same way. No density correction.
    ('cfgs/da-ieee-access/centerpoint-sourceonly-%s.yaml' % src, False)
    for src in ('kitti', 'lyft', 'nuscenes', 'pandaset', 'waymo')
]

IEEE_ACCESS_SOURCEONLY = [c for c, _ in MULTI_LEVEL_CONFIGS if 'da-ieee-access' in c]


@pytest.mark.parametrize('cfg_file,expect_hist', MULTI_LEVEL_CONFIGS)
def test_multi_level_config_resolves_complete_dataset_blocks(cfg_file, expect_hist, in_tools_dir):
    assert Path(cfg_file).exists(), 'config moved or renamed: %s' % cfg_file

    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)

    blocks = list(_dataset_blocks(cfg))
    assert blocks, 'no dataset blocks found in %s' % cfg_file

    hist_seen = False
    for label, blk in blocks:
        # These live in the SECOND hop of the base chain - the one the regression dropped.
        assert 'DATA_PROCESSOR' in blk, '%s.DATA_PROCESSOR missing in %s' % (label, cfg_file)
        assert 'POINT_CLOUD_RANGE' in blk, '%s.POINT_CLOUD_RANGE missing in %s' % (label, cfg_file)
        names = [step.get('NAME') for step in blk['DATA_PROCESSOR']]
        assert 'transform_points_to_voxels' in names, \
            '%s has no voxelization step in %s' % (label, cfg_file)
        if 'sample_points_hist_based' in names:
            hist_seen = True

    assert hist_seen == expect_hist, (
        'density correction presence mismatch for %s: expected %s, got %s'
        % (cfg_file, expect_hist, hist_seen))


@pytest.mark.parametrize('cfg_file', IEEE_ACCESS_SOURCEONLY)
def test_ieee_access_sourceonly_keeps_random_object_scaling_enabled(cfg_file, in_tools_dir):
    """ROS must survive the base chain - the trap that silently disabled it elsewhere.

    Every `cfgs/dataset_configs/da_*_dataset.yaml` lists `random_object_scaling` in its own
    `DISABLE_AUG_LIST`, and a child overriding only `AUG_CONFIG_LIST` inherits that list intact.
    That is why the per-class `SCALE_UNIFORM_NOISE` in `kitti2nuscenes_models/`
    `centerpoint-sourceonly.yaml` never runs. This family sidesteps it by owning its own
    `DISABLE_AUG_LIST`; the assertion pins that, because the failure is silent - the config loads,
    trains, and simply never scales an object.

    Also pins the UDA-legality property the family is built on: the intervals are per CLASS but
    IDENTICAL across every source, so no source's augmentation encodes target statistics.
    """
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)

    expected = {'Car': [0.85, 1.20], 'Pedestrian': [0.80, 1.25]}

    for label, blk in _dataset_blocks(cfg):
        augmentor = blk.get('DATA_AUGMENTOR')
        assert augmentor is not None, '%s.DATA_AUGMENTOR missing in %s' % (label, cfg_file)

        # data_augmentor.py reads this with direct attribute access - absent raises AttributeError.
        disabled = augmentor['DISABLE_AUG_LIST']
        assert 'random_object_scaling' not in disabled, (
            '%s disables random_object_scaling in %s - the per-class SCALE_UNIFORM_NOISE below it '
            'would be dead config' % (label, cfg_file))

        by_name = {step.get('NAME'): step for step in augmentor['AUG_CONFIG_LIST']}
        assert 'random_object_scaling' in by_name, \
            '%s has no random_object_scaling step in %s' % (label, cfg_file)
        assert dict(by_name['random_object_scaling']['SCALE_UNIFORM_NOISE']) == expected, (
            '%s in %s must use the shared target-free per-class intervals %s'
            % (label, cfg_file, expected))

        # Statistical Normalization needs the target's labelled sizes; it must stay off.
        assert 'normalize_object_size' in disabled, \
            '%s must keep normalize_object_size disabled in %s' % (label, cfg_file)


def test_ieee_access_sourceonly_family_is_internally_consistent(in_tools_dir):
    """All five configs must share one model/recipe, differing only in source and SHIFT_COOR.

    That is the whole premise of the family: the five APs are a comparable column, so any drift in
    head vocabulary, ontology or score threshold between them makes the comparison meaningless.
    """
    expected_shift = {
        'KittiDataset': [0.0, 0.0, 1.70],
        'LyftDataset': [0.0, 0.0, 1.87],
        'NuScenesDataset': [0.0, 0.0, 1.75],
        'PandasetDataset': [0.0, 0.0, 0.30],
        'WaymoDataset': [0.0, 0.0, 0.0],
    }
    # NUM_EPOCHS is intentionally NOT uniform - see the block comment in any of the five configs.
    # The sources differ by 21x in size, so what is held constant is the optimization BUDGET in
    # sample presentations (epochs x frames), anchored on nuScenes at 20 epochs = 562,600. Since
    # BATCH_SIZE_PER_GPU is shared, that also equalises ITERATIONS at ~94k, which is the property
    # that matters downstream: these checkpoints seed the self-training / DANN runs, so unequal step
    # counts here would confound those comparisons before adaptation starts.
    #
    # Asserted as a budget rather than as five magic numbers, so the property under test is the one
    # that matters - a hand-edit that stays self-consistent but drifts off the budget still fails.
    train_frames = {
        'KittiDataset': 3712,
        'PandasetDataset': 4880,
        'LyftDataset': 18900,
        'NuScenesDataset': 28130,
        'WaymoDataset': 79041,
    }
    budget = 20 * train_frames['NuScenesDataset']
    expected_epochs = {k: round(budget / n) for k, n in train_frames.items()}
    assert expected_epochs['NuScenesDataset'] == 20, 'anchor must be nuScenes at 20 epochs'
    seen = {}
    for cfg_file in IEEE_ACCESS_SOURCEONLY:
        cfg = EasyDict()
        cfg_from_yaml_file(cfg_file, cfg)

        assert cfg.CLASS_NAMES == ['Car', 'Pedestrian', 'Cyclist'], cfg_file
        assert cfg.ONTOLOGY == 'kitti', cfg_file
        assert cfg.MODEL.NAME == 'CenterPoint', cfg_file
        # Both live post-processing blocks, kept in step on purpose.
        assert cfg.MODEL.POST_PROCESSING.SCORE_THRESH == 0.0001, cfg_file
        assert cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH == 0.0001, cfg_file

        # Every config evaluates on the same target, which is what makes the column comparable.
        assert cfg.DATA_CONFIG_TAR.DATASET == 'NuScenesDataset', cfg_file

        src = cfg.DATA_CONFIG.DATASET
        assert [float(v) for v in cfg.DATA_CONFIG.SHIFT_COOR] == expected_shift[src], cfg_file
        assert cfg.OPTIMIZATION.NUM_EPOCHS == expected_epochs[src], (
            '%s: NUM_EPOCHS %s does not match the per-source budget %s'
            % (cfg_file, cfg.OPTIMIZATION.NUM_EPOCHS, expected_epochs[src]))
        # Rounding to whole epochs is the only slack allowed; Waymo is the worst case at -1.66%.
        presentations = cfg.OPTIMIZATION.NUM_EPOCHS * train_frames[src]
        assert abs(presentations - budget) / budget < 0.02, (
            '%s: %d presentations is %.1f%% off the %d budget'
            % (cfg_file, presentations, 100.0 * (presentations - budget) / budget, budget))
        seen[src] = cfg_file

    assert set(seen) == set(expected_shift), 'family is incomplete: %s' % sorted(seen)
