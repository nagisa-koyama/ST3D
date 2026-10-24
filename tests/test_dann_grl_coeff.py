"""LOSS_CONFIG.DANN_GRL_COEFF: the opt-in GRL coefficient of the in-head DANN (experiments_md 20261022_06 §10, DANN-M).

Absent (every config before 2026-10-24) -> the published constant 1.0, so 30306 and every earlier DANN row are unchanged.
Set -> alpha scales ONLY the reversed gradient into the features: the discriminator's gradient and the loss value do not
depend on it, and the feature gradient is exactly -alpha times the GRL-bypassed one.
"""
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcdet.models.model_utils import dann_utils  # noqa: E402


def test_absent_key_gives_the_published_constant():
    assert dann_utils.GRL_COEFF == 1.0
    assert dann_utils.grl_coefficient(None) == 1.0
    assert dann_utils.grl_coefficient(EasyDict({'LOSS_WEIGHTS': {'dann_weight': 0.1}})) == 1.0


def test_key_is_read_as_a_float():
    assert dann_utils.grl_coefficient(EasyDict({'DANN_GRL_COEFF': 2300})) == 2300.0
    assert isinstance(dann_utils.grl_coefficient(EasyDict({'DANN_GRL_COEFF': 2300})), float)


@pytest.mark.parametrize('bad', [0, -1.0, float('nan'), float('inf')])
def test_non_positive_or_non_finite_is_refused(bad):
    with pytest.raises(ValueError):
        dann_utils.grl_coefficient(EasyDict({'DANN_GRL_COEFF': bad}))


@pytest.mark.parametrize('alpha', [1.0, 37.0, 4.6e4])
def test_alpha_scales_only_the_reversed_feature_gradient(alpha):
    torch.manual_seed(0)
    disc = dann_utils.build_domain_discriminator(EasyDict({'LOSS_WEIGHTS': {'dann_weight': 0.1}}), in_channels=4).double()
    feats = torch.randn(2, 4, 6, 6, dtype=torch.float64, requires_grad=True)
    out = {}
    for name, fn in (('grl', lambda f: dann_utils.domain_predictions(disc, f, alpha)), ('plain', disc)):
        disc.zero_grad()
        loss, _ = dann_utils.domain_adversarial_loss(fn(feats), 1, 0.1)
        g_feat, = torch.autograd.grad(loss, feats, retain_graph=True)
        loss.backward()
        out[name] = (float(loss), g_feat, torch.cat([p.grad.reshape(-1).clone() for p in disc.parameters()]))
    (l_grl, g_grl, d_grl), (l_plain, g_plain, d_plain) = out['grl'], out['plain']
    assert l_grl == pytest.approx(l_plain, rel=1e-12), 'the loss value must not depend on alpha'
    assert g_plain.abs().max() > 0
    assert torch.allclose(g_grl, -alpha * g_plain, rtol=1e-10, atol=1e-18), 'feature gradient must be -alpha x plain'
    assert torch.allclose(d_grl, d_plain, rtol=1e-10, atol=1e-18), 'discriminator gradient must not depend on alpha'


def test_center_head_reads_the_key_and_passes_it_to_the_grl():
    src = (ROOT / 'pcdet/models/dense_heads/center_head.py').read_text()
    assert "self.dann_grl_coeff = dann_utils.grl_coefficient(self.model_cfg.get('LOSS_CONFIG', None))" in src
    assert 'self.domain_discriminator, spatial_features_2d, self.dann_grl_coeff' in src
