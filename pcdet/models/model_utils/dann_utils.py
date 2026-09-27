"""In-head DANN for CenterPoint, factored out of AnchorHeadMulti so it can be shared and unit-tested.

The mechanism is the one behind every MIRU2025 "Ours+DANN" run (SECOND + AnchorHeadMulti,
`anchor_head_multi.py`): a gradient-reversal layer with a CONSTANT coefficient of 1.0 on the BEV
feature map, a small conv discriminator (`pcdet/ops/dann/functions.py::DomainDiscriminator`), and
`BCEWithLogitsLoss` against the batch's domain label scaled by `LOSS_WEIGHTS['dann_weight']`.
Nothing here is new; it is a verbatim port so a CenterPoint DANN row stays comparable to the
published SECOND rows (experiments_md/20260927_06).

The training loop (`train_utils/train_st_utils.py`) stamps `batch_dict['domain_label']` (0 source,
1 target) and reads the loss back as the separate `dann_loss` that `model_fn_decorator` returns -
separate because PCGrad (`SELF_TRAIN.USE_TORCHJD`) projects the source, DANN and self-training
gradients against each other and needs them apart.

Two quirks kept deliberately, because changing them would make the CenterPoint row a different
method from the SECOND one:
  * the GRL coefficient is 1.0 from the first iteration (no warm-up ramp);
  * the loss is the BCE MEAN divided AGAIN by the batch size (AnchorHeadMulti does
    `loss.sum() / batch_size`), so `dann_weight` is effectively `dann_weight / batch_size`.
"""
import torch
import torch.nn as nn

from ...ops.dann.functions import DomainDiscriminator, ReverseLayerF

GRL_COEFF = 1.0


def wants_domain_discriminator(loss_cfg):
    """True when the head's LOSS_CONFIG asks for a DANN term. Gating construction on this keeps
    the state_dict of every config WITHOUT `dann_weight` byte-identical to what it was."""
    if loss_cfg is None:
        return False
    loss_weights = loss_cfg.get('LOSS_WEIGHTS', None)
    return loss_weights is not None and 'dann_weight' in loss_weights


def build_domain_discriminator(loss_cfg, in_channels):
    """The discriminator module, or None when the config does not ask for one."""
    if not wants_domain_discriminator(loss_cfg):
        return None
    return DomainDiscriminator(in_channels=in_channels)


def domain_predictions(discriminator, features):
    """Reverse the gradient into `features` and classify the domain per BEV cell."""
    reverse_feature = ReverseLayerF.apply(features, GRL_COEFF)
    return discriminator(reverse_feature)


def domain_adversarial_loss(domain_preds, domain_label, dann_weight):
    """AnchorHeadMulti.get_domain_adversarial_loss, device-agnostic.

    Returns (loss, tb_dict). `domain_label` is the batch's scalar label (0 or 1).
    """
    batch_size = int(domain_preds.shape[0])
    target = torch.full_like(domain_preds, float(domain_label))
    loss = nn.BCEWithLogitsLoss()(domain_preds, target)
    loss = loss.sum() / batch_size * dann_weight

    tb_dict = {
        'dann_loss': loss.item(),
        'domain_preds_num': torch.numel(domain_preds),
    }
    tb_dict['domain_preds_accuracy'] = ((domain_preds > 0) == bool(domain_label)).sum() / tb_dict['domain_preds_num']
    return loss, tb_dict
