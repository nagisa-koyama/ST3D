"""The conditional discriminator must reverse a NON-ZERO gradient into the detector.

Upstream UADA3D (maxiuw/UADA3D b60b846) and this port until 2026-09-26 updated the GRL coefficient
only on the marginal discriminators. The conditional ones kept GradientReversal's initial
lambda_ = 0.0, so the backward through them was -0.0 * grad: the conditional branch trained itself
and sent exactly zero gradient to the features and to the box/class predictions it conditions on.
Every conditional-only UADA3D config was source-only training with a discriminator attached.
experiments_md/20260926_05 section 6b.

The discriminator here is built from the real da-ieee-access UADA3D config, so a config change that
drops the conditional branch or its GRL shows up here too.
"""
import copy
import os
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict
from torch.nn.functional import mse_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.models.discriminators.discriminator import Discriminator2  # noqa: E402
from pcdet.models.discriminators.gradient_reversal import GradientReversal  # noqa: E402

CONFIGS = ['cfgs/da-ieee-access/centerpoint-uada3d-lyft2nuscenes.yaml',
           'cfgs/da-ieee-access/centerpoint-uada3d-kitti2nuscenes.yaml']
COEFF = 0.1  # GrlStandard's constant, what train_one_epoch_adaptive puts in the batch


def _discriminator(path):
    cfg = EasyDict()
    prev = os.getcwd()
    os.chdir(ROOT / 'tools')
    try:
        cfg_from_yaml_file(path, cfg)
    finally:
        os.chdir(prev)
    # __init__ prepends to MLPS in place, so never hand it the loaded config itself.
    return Discriminator2(copy.deepcopy(cfg.MODEL.DISCRIMINATOR)).train(), cfg.MODEL.DISCRIMINATOR


def _batch(dcfg, domain, grl_coeff=COEFF, hw=10):  # three unpadded 3x3 convs eat 6 px
    cond = dcfg.CONDITIONAL_ADAPTATION
    cls_key, box_key, feat_key = cond.INPUT_DICT_KEYS
    torch.manual_seed(0)
    batch = {
        feat_key: torch.randn(2, cond.NUM_FEATURES, hw, hw, requires_grad=True),
        box_key: torch.randn(2, cond.BOX_REGRESSION_PARAMS, hw, hw, requires_grad=True),
        cls_key: torch.rand(2, cond.NUM_CLASSES, hw, hw).clamp(1e-4, 1 - 1e-4).requires_grad_(),
        'domain': domain,
    }
    if grl_coeff is not None:
        batch['grl_coeff'] = grl_coeff
    return batch, (cls_key, box_key, feat_key)


def _conditional_loss(disc):
    r = disc.forward_ret_dict
    return sum(mse_loss(p, t) for p, t in zip(r['cond_preds'], r['cond_refs']))


@pytest.mark.parametrize('path', CONFIGS)
def test_conditional_grls_receive_the_coefficient(path):
    disc, dcfg = _discriminator(path)
    assert all(isinstance(d[0], GradientReversal) for d in disc.cond_discriminators)
    batch, _ = _batch(dcfg, domain=1)
    disc(batch)
    assert [float(d[0].lambda_) for d in disc.cond_discriminators] == pytest.approx(
        [COEFF] * len(disc.cond_discriminators))


@pytest.mark.parametrize('domain', [0, 1])
@pytest.mark.parametrize('path', CONFIGS)
def test_gradient_into_the_detector_is_exactly_reversed_and_scaled(path, domain):
    """d(loss)/d(input) through the GRL must equal -lambda times the same gradient without it."""
    disc, dcfg = _discriminator(path)
    batch, keys = _batch(dcfg, domain)
    disc(batch)
    _conditional_loss(disc).backward()
    through_grl = [batch[k].grad.clone() for k in keys]

    # Same weights, same inputs, GRL bypassed.
    bypass, _ = _batch(dcfg, domain, grl_coeff=None)
    cls_key, box_key, feat_key = keys
    cls_feats = torch.cat([bypass[feat_key], bypass[box_key]], 1)
    loss = 0
    for i, d in enumerate(disc.cond_discriminators):
        pred = d[1:](bypass[cls_key][:, i:i + 1] * cls_feats)
        loss = loss + mse_loss(pred, torch.full_like(pred, float(domain)))
    loss.backward()

    for key, got in zip(keys, through_grl):
        plain = bypass[key].grad
        assert plain.abs().max() > 0, 'degenerate check: no gradient even without the GRL'
        assert torch.allclose(got, -COEFF * plain, rtol=1e-4, atol=1e-10), (
            '%s: conditional branch must send -%.2f x the plain gradient into the detector' % (
                key, COEFF))


def test_without_a_coefficient_the_branch_stays_inert():
    """Documents the dependency: the loop must put grl_coeff in the batch (it does, per iteration)."""
    disc, dcfg = _discriminator(CONFIGS[0])
    batch, keys = _batch(dcfg, domain=1, grl_coeff=None)
    disc(batch)
    _conditional_loss(disc).backward()
    assert all(float(batch[k].grad.abs().max()) == 0.0 for k in keys)
