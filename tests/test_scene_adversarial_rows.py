"""Config pins of the three SCENE adversarial rows (experiments_md 20261022_06 §10; T05 §5 designs fixed 2026-10-24 10:13 JST).

1. DANN-M = 30306's config + LOSS_CONFIG.DANN_GRL_COEFF alpha_M (69, measured by analysis/dann_grad_ratio.py, job 30335).
2. DANN-M + PCGrad = row 1 + SELF_TRAIN.USE_TORCHJD.
3. UADA3D at GRL 1.0, fallback form = the S->B source-only config + the UADA3D model block of 26390 verbatim, target
   UNSUPERVISED, OPTIMIZATION.GRL_LAMBDA 1.0.
"""
import sys
from pathlib import Path

import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcdet.config import cfg_from_yaml_file  # noqa: E402

D = 'cfgs/da-ieee-access/'


@pytest.fixture(autouse=True)
def in_tools(monkeypatch):
    monkeypatch.chdir(ROOT / 'tools')


def _load(p):
    c = EasyDict()
    cfg_from_yaml_file(D + p, c)
    return c


def _flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, pre + k + '.'))
        else:
            out[pre + k] = v
    return out


def _diff(a, b):
    a, b = _flat(a), _flat(b)
    return sorted(k for k in set(a) | set(b) if not k.endswith('_BASE_CONFIG_') and str(a.get(k)) != str(b.get(k)))


def test_dann_m_is_30306_plus_the_matched_grl_coefficient():
    base, row = _load('centerpoint-st3d-dann-nuscenes-singapore2boston.yaml'), _load('centerpoint-st3d-dannm-nuscenes-singapore2boston.yaml')
    assert _diff(base, row) == ['MODEL.DENSE_HEAD.LOSS_CONFIG.DANN_GRL_COEFF']
    assert row.MODEL.DENSE_HEAD.LOSS_CONFIG.DANN_GRL_COEFF == 69.0
    assert row.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.dann_weight == 0.1, 'dann_weight kept at the published 0.1'
    assert 'DANN_GRL_COEFF' not in row.SELF_TRAIN.MODEL_TEACHER.DENSE_HEAD.LOSS_CONFIG
    assert list(row.SELF_TRAIN.SCORE_THRESH) == [0.2, 0.33, 0.46] and list(row.SELF_TRAIN.NEG_THRESH) == [0.149, 0.15, 0.151]


def test_dann_m_pcgrad_is_row_1_plus_torchjd():
    r1, r2 = _load('centerpoint-st3d-dannm-nuscenes-singapore2boston.yaml'), _load('centerpoint-st3d-dannm-pcgrad-nuscenes-singapore2boston.yaml')
    assert _diff(r1, r2) == ['SELF_TRAIN.USE_TORCHJD']
    assert r2.SELF_TRAIN.USE_TORCHJD is True
    assert r2.SELF_TRAIN.SRC.BACKWARD_TOGETHER and r2.SELF_TRAIN.TAR.BACKWARD_TOGETHER, 'the torchjd branch needs both'


def test_uada3d_g1_is_the_s2b_recipe_plus_the_26390_model_block_at_grl_1():
    so, row = _load('centerpoint-sourceonly-nuscenes-singapore2boston.yaml'), _load('centerpoint-uada3d-g1-nuscenes-singapore2boston.yaml')
    ref = _load('centerpoint-uada3d-lyft2nuscenes.yaml')
    assert _flat(row.MODEL) == _flat(ref.MODEL), 'the UADA3D model block must be 26390\'s, verbatim'
    d = _diff(so, row)
    assert all(k.startswith('MODEL.') or k in ('DATA_CONFIG_TAR.UNSUPERVISED', 'OPTIMIZATION.GRL_LAMBDA') for k in d), d
    assert row.OPTIMIZATION.GRL_LAMBDA == 1.0
    assert row.DATA_CONFIG_TAR.UNSUPERVISED is True
    assert row.DATA_CONFIG.VEHICLE == so.DATA_CONFIG.VEHICLE and row.DATA_CONFIG_TAR.VEHICLE == so.DATA_CONFIG_TAR.VEHICLE
    assert row.OPTIMIZATION.NUM_EPOCHS == 20 and row.OPTIMIZATION.OPTIMIZER == 'adam_onecycle'
    assert 'SELF_TRAIN' not in row
