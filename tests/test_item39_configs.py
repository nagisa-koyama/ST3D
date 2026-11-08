"""Item-39 configs (experiments_md 20261108_03 §2 B2 / C): the control's eval-only twins carry exactly the training cut they
name and differ from the control's evaluation config only in DATA_CONFIG_TAR.EVAL_FACTOR_SWAP. CPU only, no dataset files."""
import sys
from pathlib import Path

import pytest
import yaml
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import _init_path  # noqa: E402,F401

I39 = TOOLS / 'cfgs/da-ieee-access-analysis/item39'
S2 = TOOLS / 'cfgs/da-ieee-access-analysis/factorswap-s2'
CONTROL = 'cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti.yaml'
TWINS = {
    'r3-density': ('centerpoint-factorswap-s2-r3-density.yaml', 'TRAIN'),
    'r3-halfway': ('centerpoint-factorswap-s2-eval-r3-halfway.yaml', 'EVAL'),
    'p-pattern': ('centerpoint-factorswap-s2-p-pattern.yaml', 'TRAIN'),
}


def without_base_keys(d):
    if isinstance(d, dict):
        return {k: without_base_keys(v) for k, v in d.items() if k != '_BASE_CONFIG_'}
    if isinstance(d, list):
        return [without_base_keys(v) for v in d]
    return d


@pytest.mark.parametrize('name', sorted(TWINS))
def test_twin_carries_the_named_cut_exactly(name):
    src, kind = TWINS[name]
    s = yaml.safe_load(open(S2 / src))
    want = s['DATA_CONFIG']['TRAIN_FACTOR_SWAP'] if kind == 'TRAIN' else s['DATA_CONFIG_TAR']['EVAL_FACTOR_SWAP']
    got = yaml.safe_load(open(I39 / ('centerpoint-control-eval-%s.yaml' % name)))
    assert got['_BASE_CONFIG_'] == CONTROL
    assert got['DATA_CONFIG_TAR']['EVAL_FACTOR_SWAP'] == want


@pytest.mark.parametrize('name', sorted(TWINS))
def test_twin_resolves_to_the_control_plus_the_key(monkeypatch, name):
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(TOOLS)
    base = cfg_from_yaml_file(CONTROL, EasyDict())
    twin = cfg_from_yaml_file(str(I39 / ('centerpoint-control-eval-%s.yaml' % name)), EasyDict())
    assert 'EVAL_FACTOR_SWAP' in twin.DATA_CONFIG_TAR and 'EVAL_FACTOR_SWAP' not in base.DATA_CONFIG_TAR
    twin.DATA_CONFIG_TAR.pop('EVAL_FACTOR_SWAP')
    assert without_base_keys(twin) == without_base_keys(base)
