"""Every KITTI-target config in the da-ieee-access families scores its predictions through the camera-FOV filter.

Defect: `tools/cfgs/da-ieee-access/da_kitti_dataset.yaml` omitted `TEST.BOX_FILTER.FOV_FILTER` until
ST3D 0a5d4b4 (2026-09-26). KITTI is annotated only inside the camera FOV (0.0% of GT behind the
vehicle) while this family's detectors predict over 360 degrees, so without the filter every rear
prediction is an unmatched detection: a 360-trained source scored ~4x low and the difficulty ordering
inverted (job 25780, Car BEV R40: 13.62 / 17.46 / 20.13 off, 84.06 / 71.11 / 69.42 on).
`da-MIRU2025` and `dataset_configs/da_kitti_dataset.yaml` always had it; only this family did not.
Reports: experiments_md/20260926_02, 20261003_13 §3.M; memory/repo/index.md "Pipeline and evaluation".

Also pinned on the same configs: `FOV_POINTS_ONLY` is declared and False (kitti_dataset.py reads the key by
attribute, so a missing key raises at the first eval batch, and True trips the 1d20716 lock unless the config
opts in with ALLOW_FOV_POINTS_ONLY; see tests/test_kitti_fov_lock.py).

The configs are resolved from tools/ exactly as train.py resolves them (`_BASE_CONFIG_` chains expanded),
because the key lives in the shared dataset yaml and the defect was invisible in any child's own text.
Another session's in-progress directory (da-ieee-access-analysis/factorswap-n2w, Waymo target) is skipped.
"""
import os
import re
from pathlib import Path

import pytest
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
CFGS = TOOLS / 'cfgs'
SKIP_DIRS = ('da-ieee-access-analysis/factorswap-n2w',)


def _candidate_configs():
    """Model configs under tools/cfgs/da-ieee-access* whose text mentions KITTI at all (cheap pre-filter;
    the real test is on the RESOLVED target dataset)."""
    out = []
    for path in sorted(CFGS.glob('da-ieee-access*/**/*.yaml')):
        rel = path.relative_to(CFGS).as_posix()
        if any(rel.startswith(d) for d in SKIP_DIRS):
            continue
        text = path.read_text(encoding='utf-8')
        if not re.search(r'^CLASS_NAMES:', text, re.M):
            continue  # a dataset / preprocessing yaml, not a row
        if not re.search(r'kitti', text, re.I):
            continue
        out.append(path)
    return out


def _resolved(path):
    from pcdet.config import cfg_from_yaml_file
    prev = os.getcwd()
    os.chdir(TOOLS)
    try:
        return cfg_from_yaml_file(str(path.relative_to(TOOLS)), EasyDict())
    finally:
        os.chdir(prev)


def _eval_dataset_cfg(cfg):
    """The block the evaluation loader is built from: DATA_CONFIG_TAR when present, else DATA_CONFIG."""
    return cfg.get('DATA_CONFIG_TAR', None) or cfg.get('DATA_CONFIG', None)


def _fov_filter_on(dataset_cfg):
    return bool(((dataset_cfg.get('TEST', None) or {}).get('BOX_FILTER', None) or {}).get('FOV_FILTER', False))


CANDIDATES = _candidate_configs()


def test_the_scan_finds_the_known_kitti_target_rows():
    names = {p.name for p in CANDIDATES}
    assert 'centerpoint-accum-global-nuscenes2kitti.yaml' in names, 'job 25780\'s row must be in the scan'
    assert 'centerpoint-sourceonly-kitti2kitti.yaml' in names


@pytest.mark.parametrize('path', CANDIDATES, ids=lambda p: p.relative_to(CFGS).as_posix())
def test_kitti_target_scores_through_the_camera_fov_filter(path):
    cfg = _resolved(path)
    ds = _eval_dataset_cfg(cfg)
    if ds is None or ds.get('DATASET', None) != 'KittiDataset':
        pytest.skip('evaluation target is not KITTI')
    rel = path.relative_to(CFGS).as_posix()
    fov_points_only = ds.get('FOV_POINTS_ONLY', None)
    if fov_points_only is True:
        # The published KITTI recipe (iassd-published-kitti, the fovcrop oracle): the INPUT is already the camera
        # FOV, so no rear prediction exists to filter; what must hold is the explicit opt-in of the 1d20716 lock.
        assert ds.get('ALLOW_FOV_POINTS_ONLY', None) is True, (
            '%s: FOV_POINTS_ONLY True without ALLOW_FOV_POINTS_ONLY trips the lock (tests/test_kitti_fov_lock.py)' % rel)
        return
    assert fov_points_only is False, (
        '%s: KITTI target must declare FOV_POINTS_ONLY: False (kitti_dataset.py reads it by attribute)' % rel)
    assert _fov_filter_on(ds), '%s: 360-degree KITTI target without TEST.BOX_FILTER.FOV_FILTER: True (job 25780 regime)' % rel


def test_the_family_base_itself_carries_the_filter():
    """The fix landed in the shared dataset yaml; a child cannot be expected to repeat it."""
    import yaml
    base = yaml.safe_load((CFGS / 'da-ieee-access' / 'da_kitti_dataset.yaml').read_text(encoding='utf-8'))
    assert base['TEST']['BOX_FILTER']['FOV_FILTER'] is True


def test_the_check_flags_the_pre_fix_shape():
    """Reference: a resolved KITTI block as it stood before 0a5d4b4 (no TEST key at all) must be flagged, and a
    TEST block without BOX_FILTER or with FOV_FILTER False as well."""
    assert not _fov_filter_on(EasyDict(DATASET='KittiDataset', FOV_POINTS_ONLY=False))
    assert not _fov_filter_on(EasyDict(DATASET='KittiDataset', TEST=EasyDict()))
    assert not _fov_filter_on(EasyDict(DATASET='KittiDataset', TEST=EasyDict(BOX_FILTER={'FOV_FILTER': False})))
    assert _fov_filter_on(EasyDict(DATASET='KittiDataset', TEST=EasyDict(BOX_FILTER={'FOV_FILTER': True})))
