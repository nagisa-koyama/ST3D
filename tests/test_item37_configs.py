"""Configs of experiments_md 20261003_05 §5 item 37 (count-matched structure terms). CPU only, configs only.

37a: the eval-only twin that scores the P->N count-matched control L (31539 / 31550) on Pandar64 spin val must be L's
resolved config with only DATA_CONFIG_TAR replaced, and that block must be the PandaSet oracle's (spin2spin). A twin that
kept nuScenes' DATA_CONFIG_TAR, or changed the model, would score a different detector or a different val.
"""
import sys
from pathlib import Path

from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import _init_path  # noqa: E402,F401

L_PANDASET = 'cfgs/da-ieee-access/centerpoint-global-fovcut-recount-latcount-pandaset2nuscenes.yaml'
TWIN_37A = 'cfgs/da-ieee-access-analysis/factorswap-pandaset/centerpoint-factorswap-pandaset-evalonly-l-on-spin.yaml'
SPIN = 'cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml'
P_TWIN = 'cfgs/da-ieee-access-analysis/factorswap-pandaset/centerpoint-factorswap-pandaset-evalonly-p-on-spin.yaml'


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items() if k not in ('_BASE_CONFIG_', 'ROOT_DIR', 'LOCAL_RANK')}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    return x


def resolved(path):
    from pcdet.config import cfg_from_yaml_file
    return plain(cfg_from_yaml_file(path, EasyDict()))


def test_37a_twin_is_l_with_the_spin_oracle_eval_block(monkeypatch):
    monkeypatch.chdir(TOOLS)
    twin, src, spin = resolved(TWIN_37A), resolved(L_PANDASET), resolved(SPIN)
    assert twin['DATA_CONFIG_TAR'] == spin['DATA_CONFIG_TAR']
    assert twin['DATA_CONFIG_TAR'] != src['DATA_CONFIG_TAR']
    for k in set(twin) | set(src):
        if k != 'DATA_CONFIG_TAR':
            assert twin.get(k) == src.get(k), k


def test_37a_twin_scores_the_same_model_as_the_p_twin(monkeypatch):
    monkeypatch.chdir(TOOLS)
    twin, p = resolved(TWIN_37A), resolved(P_TWIN)
    for k in ('MODEL', 'CLASS_NAMES', 'DATA_CONFIG_TAR'):
        assert twin[k] == p[k], k


# ---- 37b: the Lyft-40 count-matched control and the multi-source reader it needs

def test_source_data_config_reads_single_and_multi_source_forms():
    from pcdet.datasets.point_calibration import source_data_config
    single = EasyDict({'DATA_CONFIG': {'A': 1}})
    multi = EasyDict({'DATA_CONFIGS': {'LYFT_40BEAM': {'A': 2}}})
    assert source_data_config(single).A == 1
    assert source_data_config(multi).A == 2


def test_source_data_config_refuses_two_sources():
    import pytest
    from pcdet.datasets.point_calibration import source_data_config
    with pytest.raises(AssertionError):
        source_data_config(EasyDict({'DATA_CONFIGS': {'A': {}, 'B': {}}}))
    with pytest.raises(AssertionError):
        source_data_config(EasyDict({}))


def test_build_reference_source_reads_a_data_configs_row(monkeypatch):
    """Before item 37b build_reference_source read ref.DATA_CONFIG only, so a Lyft-40 (DATA_CONFIGS) lattice row as the
    reference raised AttributeError; it must hand the single DATA_CONFIGS block to build_dataloader."""
    import pcdet.datasets as D
    from pcdet.datasets import point_calibration as PC
    monkeypatch.chdir(TOOLS)
    seen = {}

    def fake(dataset_cfg, **kw):
        seen['cfg'] = dataset_cfg
        return 'dataset', None, None
    monkeypatch.setattr(D, 'build_dataloader', fake)
    assert PC.build_reference_source('cfgs/da-ieee-access/centerpoint-ringpattern-lyft40-lyft2nuscenes.yaml') == 'dataset'
    assert 'RING_PATTERN' in seen['cfg'] and seen['cfg'].get('LIDAR_CONFIG') in (40, '40')


L_LYFT40 = 'cfgs/da-ieee-access/centerpoint-global-recount-latcount-lyft40-lyft2nuscenes.yaml'
C1_LYFT40 = 'cfgs/da-ieee-access/centerpoint-global-recount-lyft40-lyft2nuscenes.yaml'
P_LYFT40 = 'cfgs/da-ieee-access/centerpoint-ringpattern-lyft40-lyft2nuscenes.yaml'
TWIN_37B = 'cfgs/da-ieee-access-analysis/factorswap-lyft40/centerpoint-factorswap-lyft40-evalonly-l-on-lyft40.yaml'
O_L40 = 'cfgs/da-ieee-access-analysis/factorswap-lyft40/centerpoint-factorswap-lyft40-oracle.yaml'


def test_37b_l_is_c1_matched_to_the_lattice_arm(monkeypatch):
    monkeypatch.chdir(TOOLS)
    l, c1 = resolved(L_LYFT40), resolved(C1_LYFT40)
    lb, cb = l['DATA_CONFIGS']['LYFT_40BEAM'], c1['DATA_CONFIGS']['LYFT_40BEAM']
    assert lb.pop('HIST_DIST_REFERENCE_CFG') == P_LYFT40 and lb.pop('HIST_DIST_IN_GRID') is True
    assert lb == cb
    for k in set(l) | set(c1):
        if k != 'DATA_CONFIGS':
            assert l.get(k) == c1.get(k), k


def test_37b_twin_is_l_with_the_lyft40_oracle_eval_block(monkeypatch):
    monkeypatch.chdir(TOOLS)
    twin, src, o = resolved(TWIN_37B), resolved(L_LYFT40), resolved(O_L40)
    assert twin['DATA_CONFIG_TAR'] == o['DATA_CONFIG_TAR'] != src['DATA_CONFIG_TAR']
    for k in set(twin) | set(src):
        if k != 'DATA_CONFIG_TAR':
            assert twin.get(k) == src.get(k), k
