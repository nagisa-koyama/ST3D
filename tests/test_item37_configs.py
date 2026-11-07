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
