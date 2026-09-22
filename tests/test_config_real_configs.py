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
] + [
    # The Lyft->Lyft oracle has the same two-hop shape and needs the same resolution check, but it
    # is NOT part of the X->nuScenes column, so it is deliberately excluded from
    # IEEE_ACCESS_SOURCEONLY below - its target is Lyft, which those assertions forbid.
    ('cfgs/da-ieee-access/centerpoint-oracle-lyft2lyft.yaml', False),
]

IEEE_ACCESS_SOURCEONLY = [c for c, _ in MULTI_LEVEL_CONFIGS if 'da-ieee-access/centerpoint-sourceonly-' in c]


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
    # This family is deployed on two GPUs (scripts/run_sourceonly_2gpu.sh, --nproc_per_node=2).
    # BATCH_SIZE_PER_GPU is a PER-GPU value, so the global batch is this times GPUS_PER_RUN.
    GPUS_PER_RUN = 2

    # Measured per source rather than defaulted - the two 8s are the sources whose loaders stall.
    expected_workers = {
        'KittiDataset': 4,
        'PandasetDataset': 8,
        'LyftDataset': 4,
        'NuScenesDataset': 4,
        'WaymoDataset': 8,
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
        # NUM_WORKERS is measured per source, not defaulted. The value tracks whether that
        # source's loader actually stalls the GPU: Waymo (8.98% data-wait) and PandaSet (46.54%)
        # need 8, and the other three measure SLOWER at 8 because they have no stall to remove.
        # See experiments_md/20260922_06 section 1b, and the comment at each config's own key.
        # Evaluation costs roughly one training epoch per checkpoint (20260922_06 section 4d), and
        # the repo default of 100 would add ~30.8 h across this family. Uniform across sources so
        # the five runs stay comparable in what they spend on eval as well as on training.
        assert cfg.OPTIMIZATION.NUM_EPOCHS_TO_EVAL == 1, (
            '%s: NUM_EPOCHS_TO_EVAL is %s, expected 1'
            % (cfg_file, cfg.OPTIMIZATION.get('NUM_EPOCHS_TO_EVAL', None)))
        # Single GPU: BATCH_SIZE_PER_GPU is also the global batch, so the shared budget gives the
        # same optimizer-step count for every source. Pinned because running this family under a
        # launcher would silently double the global batch and halve the steps.
        # BATCH_SIZE_PER_GPU is PER GPU and this family is deployed on two, so 3 per GPU is a
        # GLOBAL batch of 6 - the number the family is actually built on, because it fixes the
        # optimizer-step count at 93,766 for every source. Asserted as the global batch rather
        # than as the raw key, so the invariant under test is the one that matters.
        assert cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU == 3, cfg_file
        global_batch = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU * GPUS_PER_RUN
        assert global_batch == 6, cfg_file
        assert budget // global_batch == 93766, cfg_file
        assert cfg.OPTIMIZATION.NUM_WORKERS == expected_workers[src], (
            '%s: NUM_WORKERS %s does not match the measured recommendation %s'
            % (cfg_file, cfg.OPTIMIZATION.get('NUM_WORKERS', None), expected_workers[src]))
        seen[src] = cfg_file

    assert set(seen) == set(expected_shift), 'family is incomplete: %s' % sorted(seen)


@pytest.mark.parametrize('entry_point', ['train.py', 'adaptive_train.py', 'test.py'])
def test_workers_argparse_default_is_none(entry_point):
    """`--workers` must default to None so OPTIMIZATION.NUM_WORKERS is reachable.

    A non-None argparse default silently wins over the config and nothing reports it. That exact
    bug lived in train.py's `--batch_size` (default=16) for months, overriding every config's
    BATCH_SIZE_PER_GPU; test.py still carries it. Parsed from source with `ast` rather than by
    importing, so this runs without a GPU and without pcdet's heavy import chain.
    """
    import ast

    path = Path(__file__).resolve().parent.parent / 'tools' / entry_point
    tree = ast.parse(path.read_text())
    defaults = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--workers'):
            continue
        for kw in node.keywords:
            if kw.arg == 'default':
                defaults.append(kw.value)

    assert len(defaults) == 1, '%s: expected exactly one --workers argument' % entry_point
    assert isinstance(defaults[0], ast.Constant) and defaults[0].value is None, (
        '%s: --workers default must be None so the config can supply it' % entry_point)


@pytest.mark.parametrize('entry_point', ['train.py', 'adaptive_train.py'])
def test_num_epochs_to_eval_argparse_default_is_none(entry_point):
    """Same reachability guard as `--workers`: a non-None default hides the config value."""
    import ast

    path = Path(__file__).resolve().parent.parent / 'tools' / entry_point
    tree = ast.parse(path.read_text())
    defaults = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == '--num_epochs_to_eval'):
            continue
        for kw in node.keywords:
            if kw.arg == 'default':
                defaults.append(kw.value)

    assert len(defaults) == 1, '%s: expected exactly one --num_epochs_to_eval argument' % entry_point
    assert isinstance(defaults[0], ast.Constant) and defaults[0].value is None, (
        '%s: --num_epochs_to_eval default must be None so the config can supply it' % entry_point)


@pytest.mark.parametrize('num_epochs_to_eval,expected_checkpoints', [(0, 1), (1, 2), (3, 4)])
def test_num_epochs_to_eval_is_off_by_one(num_epochs_to_eval, expected_checkpoints):
    """`n` evaluates n+1 checkpoints, not n. Pinned so the configs' arithmetic stays honest.

    `train.py` computes `start_epoch = max(NUM_EPOCHS - NUM_EPOCHS_TO_EVAL, 0)` and
    `get_no_evaluated_ckpt` keeps checkpoints with `epoch_id >= start_epoch`, which is inclusive at
    both ends. So `NUM_EPOCHS_TO_EVAL: 1` on a 152-epoch run evaluates epochs 151 AND 152. The
    da-ieee-access configs say so at the key; this test is what keeps that comment true.
    """
    num_epochs = 152
    start_epoch = max(num_epochs - num_epochs_to_eval, 0)
    evaluated = [e for e in range(1, num_epochs + 1) if e >= start_epoch]
    assert len(evaluated) == expected_checkpoints


def test_two_gpu_launch_script_passes_an_explicit_global_batch():
    """The 2-GPU launch must pass `--batch_size 6`, or it silently halves the optimizer steps.

    `--batch_size` is the TOTAL across ranks and train.py divides it by the GPU count, so 6 gives
    3 per rank and a global batch of 6 - the single-GPU recipe. Omitting it instead makes
    args.batch_size = BATCH_SIZE_PER_GPU = 6 PER rank: global batch 12, 46,883 steps, and LR 0.003
    out of calibration. Nothing errors. This test is the guard.
    """
    script = (Path(__file__).resolve().parent.parent
              / 'tools' / 'scripts' / 'run_sourceonly_2gpu.sh').read_text()
    # Comments explain these flags at length, so test what the shell actually runs.
    code = '\n'.join(l for l in script.splitlines() if not l.lstrip().startswith('#'))
    assert '--nproc_per_node=2' in code, 'the configs\' BATCH_SIZE_PER_GPU 3 assumes exactly 2 GPUs'
    # The job must run against a frozen copy of the repo, not the live checkout. Under DDP,
    # spawned DataLoader workers re-import every module from disk, so a commit landing mid-run
    # reaches them - which is how job 25743 lost its evaluation after 8 h of clean training.
    assert 'rsync' in code and '/local_cache/' in code, 'long DDP runs must snapshot the code'
    assert '--bind "$SNAP":/home/koyama/code/ST3D' in code, 'the snapshot must shadow the repo path'
    assert '--bind "$SNAP":/root/ST3D' in code, 'PandaSet infos need /root/ST3D on the snapshot too'
    # Redundant with BATCH_SIZE_PER_GPU 3 by design: passing 6 is divided by the GPU count back to
    # 3 per rank, so the flag and the config agree instead of one covering for the other.
    assert '--batch_size 6' in code, 'the 2-GPU launch must pin the global batch to 6'
    # --workers must NOT be passed: it is per-rank and comes from each config's NUM_WORKERS,
    # which differs by source (8 for Waymo and PandaSet, 4 otherwise).
    assert '--workers' not in code, '--workers would override the per-source NUM_WORKERS'


def test_lyft_oracle_differs_from_the_lyft_source_row_only_in_its_target(in_tools_dir):
    """The Lyft->Lyft oracle must be the Lyft source-only row with the eval target swapped.

    Its whole purpose is to bound `X -> Lyft` rows, and that reading only holds if the recipe is
    otherwise identical - same model, schedule, augmentation, SHIFT_COOR, budget and thresholds -
    so that the difference between the two numbers is the domain gap and not a config drift. This
    compares the RESOLVED configs rather than the file text, since the two carry different headers.
    """
    base, oracle = EasyDict(), EasyDict()
    cfg_from_yaml_file('cfgs/da-ieee-access/centerpoint-sourceonly-lyft.yaml', base)
    cfg_from_yaml_file('cfgs/da-ieee-access/centerpoint-oracle-lyft2lyft.yaml', oracle)

    assert base.DATA_CONFIG_TAR.DATASET == 'NuScenesDataset'
    assert oracle.DATA_CONFIG_TAR.DATASET == 'LyftDataset', 'the oracle must evaluate on Lyft'
    # Source side untouched: same dataset, same platform blend, same sensor-height constant.
    assert oracle.DATA_CONFIG.DATASET == base.DATA_CONFIG.DATASET == 'LyftDataset'
    assert oracle.DATA_CONFIG.SHIFT_COOR == base.DATA_CONFIG.SHIFT_COOR
    # Recipe identical, so the gap is the domain gap.
    assert oracle.CLASS_NAMES == base.CLASS_NAMES
    assert oracle.ONTOLOGY == base.ONTOLOGY == 'kitti'
    assert oracle.MODEL == base.MODEL, 'model/thresholds must not drift between the pair'
    assert oracle.OPTIMIZATION == base.OPTIMIZATION, 'schedule and budget must not drift'
