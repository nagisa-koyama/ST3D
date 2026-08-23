"""
Regression test for pcdet/models/discriminators/discriminator.py's `Discriminator2.get_loss()`.

Covers a bug found 2026-08-23: the final `if self.regularization:` block (which adds the
conditional-consistency regularization term to the total discriminator loss) was indented as a
sibling of `if self.conditional:` instead of being nested inside it. Since `reg_loss` and the loop
variable `i` are only ever defined inside the `if self.conditional:` branch's for-loop, enabling
`CONSISTENCY_REGULARIZATION` without also enabling conditional adaptation raised a
NameError/UnboundLocalError instead of simply skipping the (inapplicable) regularization term.
No config in the repo currently sets `CONSISTENCY_REGULARIZATION: True`, so this bug was latent,
but it is reachable as soon as anyone enables that flag on its own.
"""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.models.discriminators.discriminator import Discriminator2  # noqa: E402


def _make_discriminator(marginal, conditional, regularization):
    disc = Discriminator2.__new__(Discriminator2)
    nn.Module.__init__(disc)
    disc.loss_cfg = EasyDict({
        'LOSS_FUNCTION': ['CrossEntropy'],
        'LOSS_CONDITIONAL': 'CrossEntropy',
    })
    disc.marginal = marginal
    disc.conditional = conditional
    disc.regularization = regularization
    return disc


def test_regularization_without_conditional_does_not_crash():
    # CONSISTENCY_REGULARIZATION=True with CONDITIONAL_ADAPTATION disabled: the regularization
    # term is inapplicable (no cond_preds exist) and must simply be skipped, not raise.
    disc = _make_discriminator(marginal=False, conditional=False, regularization=True)
    disc.forward_ret_dict = {}

    loss, tb_dict = disc.get_loss()

    assert loss == 0
    assert 'reg_loss0' not in tb_dict


def test_regularization_with_conditional_and_marginal_is_added_once():
    disc = _make_discriminator(marginal=True, conditional=True, regularization=True)

    domain_preds = [torch.zeros(2, 1)]
    domain_refs = [torch.zeros(2, 1)]
    cond_preds = [torch.zeros(2, 1)]
    cond_refs = [torch.zeros(2, 1)]
    disc.forward_ret_dict = {
        'domain_preds': domain_preds,
        'domain_refs': domain_refs,
        'cond_preds': cond_preds,
        'cond_refs': cond_refs,
    }

    loss, tb_dict = disc.get_loss()

    assert 'reg_loss0' in tb_dict
    assert torch.is_tensor(loss)
    # disc_loss0 (marginal) + cond_disc_loss0 (conditional) + reg_loss0 (regularization).
    # All inputs are zero logits/refs, so each BCEWithLogits component is -log(sigmoid(0)) =
    # log(2); the total loss is the sum of all three components.
    assert loss.item() == pytest.approx(3 * math.log(2), rel=1e-4)
