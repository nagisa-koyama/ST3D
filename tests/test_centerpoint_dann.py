"""In-head DANN for CenterPoint (pcdet/models/model_utils/dann_utils.py), and the two rows that use it.

Until 2026-09-27 only SECONDNet + AnchorHeadMulti produced a `dann_loss`; a CenterPoint config with
`dann_weight` was silently inert and `SELF_TRAIN.USE_TORCHJD: True` hit PCGrad's three-loss assert.
experiments_md/20260927_06.

CenterHead itself cannot be built on the CPU-only master node (`__init__` calls `.cuda()`), so the
mechanism is tested through the helper module it delegates to, plus a source-level check that the
detector actually forwards the term, plus config pinning of the two new rows.
"""
import ast
import re
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.models.model_utils import dann_utils  # noqa: E402

TOOLS = ROOT / 'tools'
FG = 'cfgs/da-ieee-access/centerpoint-foreground-lyft2nuscenes.yaml'
GLOBAL = 'cfgs/da-ieee-access/centerpoint-st3d-global-lyft2nuscenes.yaml'
DANN = 'cfgs/da-ieee-access/centerpoint-st3d-global-dann-lyft2nuscenes.yaml'
PCGRAD = 'cfgs/da-ieee-access/centerpoint-st3d-global-dann-pcgrad-lyft2nuscenes.yaml'


@pytest.fixture
def in_tools_dir(monkeypatch):
    monkeypatch.chdir(TOOLS)


def _load(path):
    cfg = EasyDict()
    cfg_from_yaml_file(path, cfg)
    return cfg


def _flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, pre + k + '.'))
        else:
            out[pre + k] = v
    return out


# ---------------------------------------------------------------- the mechanism, on CPU

def test_discriminator_is_built_only_when_dann_weight_is_configured():
    without = EasyDict({'LOSS_WEIGHTS': {'cls_weight': 1.0, 'loc_weight': 2.0}})
    with_ = EasyDict({'LOSS_WEIGHTS': {'cls_weight': 1.0, 'loc_weight': 2.0, 'dann_weight': 0.1}})
    assert dann_utils.build_domain_discriminator(without, in_channels=8) is None
    assert dann_utils.build_domain_discriminator(None, in_channels=8) is None
    disc = dann_utils.build_domain_discriminator(with_, in_channels=8)
    assert disc is not None
    # every existing CenterPoint config keeps its parameter set: no module -> no state_dict keys
    assert len(list(disc.parameters())) > 0


def test_gradient_into_the_features_is_reversed():
    """The GRL is the whole point: the discriminator learns to tell domains apart while the features
    receive the NEGATED gradient. Compare against the same discriminator without the GRL."""
    torch.manual_seed(0)
    cfg = EasyDict({'LOSS_WEIGHTS': {'dann_weight': 1.0}})
    disc = dann_utils.build_domain_discriminator(cfg, in_channels=4)
    feats = torch.randn(2, 4, 6, 6, requires_grad=True)

    preds = dann_utils.domain_predictions(disc, feats)
    loss, _ = dann_utils.domain_adversarial_loss(preds, 1, 1.0)
    grad_reversed, = torch.autograd.grad(loss, feats)

    preds_plain = disc(feats)
    loss_plain, _ = dann_utils.domain_adversarial_loss(preds_plain, 1, 1.0)
    grad_plain, = torch.autograd.grad(loss_plain, feats)

    assert torch.allclose(grad_reversed, -grad_plain), 'GRL must negate the feature gradient'
    assert grad_plain.abs().sum() > 0, 'a zero gradient would make the check vacuous'
    assert dann_utils.GRL_COEFF == 1.0, 'MIRU2025 ran a constant 1.0; changing it changes the method'


def test_loss_matches_anchor_head_multi_arithmetic():
    """AnchorHeadMulti.get_domain_adversarial_loss: BCE-with-logits MEAN, then `.sum() / batch_size
    * dann_weight`. The second division is a quirk, kept so the CenterPoint row is the same method."""
    torch.manual_seed(1)
    preds = torch.randn(3, 1, 5, 5)
    for label in (0, 1):
        loss, tb = dann_utils.domain_adversarial_loss(preds, label, 0.1)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            preds, torch.full_like(preds, float(label))) / 3 * 0.1
        assert torch.isclose(loss, expected)
        assert tb['domain_preds_num'] == 75
        acc = ((preds > 0) == bool(label)).float().mean()
        assert torch.isclose(torch.as_tensor(tb['domain_preds_accuracy'], dtype=torch.float32), acc)


def test_loss_runs_on_cpu_tensors():
    # AnchorHeadMulti's version hard-codes `.cuda()`; the port must not, or it cannot be tested here.
    preds = torch.zeros(1, 1, 2, 2)
    loss, _ = dann_utils.domain_adversarial_loss(preds, 0, 0.1)
    assert loss.device.type == 'cpu'


# ---------------------------------------------------------------- the wiring, from source

def _source(rel):
    return (ROOT / rel).read_text()


def test_centerpoint_forwards_the_term_separately_from_loss():
    src = _source('pcdet/models/detectors/centerpoint.py')
    assert 'get_domain_adversarial_loss()' in src
    assert "ret_dict['dann_loss'] = dann_loss" in src, \
        'model_fn_decorator reads ret_dict["dann_loss"]; without it PCGrad has no third loss'
    # the term must NOT be folded into `loss`, or PCGrad would project a sum against its own part
    body = src[src.index('def get_training_loss'):]
    assert re.search(r'^\s*loss = loss_rpn\s*$', body, re.M), \
        'the detection loss returned as `loss` must stay loss_rpn alone'


def test_center_head_gates_construction_and_clears_stale_predictions():
    src = _source('pcdet/models/dense_heads/center_head.py')
    assert 'dann_utils.build_domain_discriminator(' in src
    assert 'def get_domain_adversarial_loss(self)' in src
    # a plain-training forward (no domain_label) must not serve the previous batch's predictions
    assert "self.forward_ret_dict.pop('domain_preds', None)" in src
    assert "'domain_label' in data_dict" in src


def test_train_st_utils_still_stamps_domain_label_and_collects_dann_loss():
    """The loop side of the contract, so a refactor there cannot silently orphan the head side."""
    src = _source('tools/train_utils/train_st_utils.py')
    assert "source_batch['domain_label'] = 0" in src
    assert "target_batch['domain_label'] = 1" in src
    assert "cfg.SELF_TRAIN.get('USE_TORCHJD', False)" in src
    assert 'torchjd.backward([loss_src_sum, dann_loss_sum, st_loss_sum]' in src


# ---------------------------------------------------------------- the rows

def test_dann_row_differs_from_st3d_global_only_in_dann_weight(in_tools_dir):
    a, b = _flat(_load(GLOBAL)), _flat(_load(DANN))
    diffs = sorted(k for k in set(a) | set(b)
                   if not k.endswith('_BASE_CONFIG_') and str(a.get(k)) != str(b.get(k)))
    assert diffs == ['MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.dann_weight'], diffs
    assert b['MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.dann_weight'] == 0.1, \
        'MIRU2025 "Ours+DANN" used 0.1; a different weight is a different row'
    # the merge must ADD the key, not replace the dict: the other weights survive
    for k in ('cls_weight', 'loc_weight', 'code_weights'):
        assert b['MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.' + k] == a['MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.' + k]
    # the teacher never sees a domain label; a discriminator there would be dead parameters
    assert 'dann_weight' not in _load(DANN).SELF_TRAIN.MODEL_TEACHER.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS
    assert not _load(DANN).SELF_TRAIN.get('USE_TORCHJD', False)


def test_pcgrad_row_differs_from_dann_row_only_in_use_torchjd(in_tools_dir):
    a, b = _flat(_load(DANN)), _flat(_load(PCGRAD))
    diffs = sorted(k for k in set(a) | set(b)
                   if not k.endswith('_BASE_CONFIG_') and str(a.get(k)) != str(b.get(k)))
    assert diffs == ['SELF_TRAIN.USE_TORCHJD'], diffs
    cfg = _load(PCGRAD)
    assert cfg.SELF_TRAIN.USE_TORCHJD is True
    # the torchjd branch is reached only with both BACKWARD_TOGETHER flags, and asserts three losses
    assert cfg.SELF_TRAIN.SRC.BACKWARD_TOGETHER and cfg.SELF_TRAIN.TAR.BACKWARD_TOGETHER
    assert 'dann_weight' in cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS, \
        'PCGrad asserts dann_loss_sum is not None: the row cannot run without the DANN term'


def test_both_rows_keep_the_foreground_rows_self_training_block(in_tools_dir):
    fg = _flat(_load(FG))
    for row in (DANN, PCGRAD):
        b = _flat(_load(row))
        for k in ('SELF_TRAIN.SCORE_THRESH', 'SELF_TRAIN.NEG_THRESH', 'SELF_TRAIN.PROG_AUG.UPDATE_AUG',
                  'SELF_TRAIN.MODEL_TEACHER.NAME', 'OPTIMIZATION.NUM_EPOCHS', 'OPTIMIZATION.BATCH_SIZE_PER_GPU'):
            assert str(b[k]) == str(fg[k]), (row, k)
        assert b['MODEL.NAME'] == 'CenterPoint', 'the in-head DANN lives in CenterHead, not DACenterPoint'
