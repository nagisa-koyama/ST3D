"""The three lists that define a multi-head detector must agree, and only one of them binds by name.

`AnchorHeadMulti` couples three independently-written config lists in three different ways:

  CLASS_NAMES (model, prefixed)   make_multihead -> self.class_names.index(name) + 1   BY NAME
  ANCHOR_GENERATOR_CONFIG         AnchorGenerator reads anchor_sizes / anchor_rotations /
                                  anchor_bottom_heights / align_center and NEVER class_name
                                                                                    BY POSITION
  RPN_HEAD_CFGS[*].HEAD_CLS_NAME  make_multihead ->
                                  num_anchors_per_location[class_names.index(head_cls)]
                                  where that list was produced in ANCHOR order and is indexed in
                                  concatenated HEAD order                    BY POSITION, CROSS-LIST

So `class_name` inside ANCHOR_GENERATOR_CONFIG is read by exactly ONE consumer -
`AxisAlignedTargetAssigner`, for anchor-set names and the IoU thresholds - and is invisible to the
generator that actually builds the anchors. If the anchor order and the concatenated head order
diverged, every head would receive the wrong `num_anchors_per_location` and nothing would raise.

Audited 2026-09-25: all 86 configs declaring both lists agree, so there is no live defect. Nothing
enforced it, which is the same failure shape as the ontology maps - a name that looks authoritative
and is not what does the binding. This test is the enforcement, pending the HeadSpec refactor that
would make the disagreement unrepresentable.
"""
from pathlib import Path

import pytest
import yaml

CFGS = Path(__file__).resolve().parent.parent / 'tools' / 'cfgs'


def _multihead_configs():
    """Configs declaring BOTH lists inline. A config inheriting them via _BASE_CONFIG_ is checked
    through whichever config does declare them, so nothing is missed by skipping it here."""
    out = []
    for path in sorted(CFGS.rglob('*.yaml')):
        try:
            cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        dh = (cfg.get('MODEL') or {}).get('DENSE_HEAD') or {}
        if dh.get('ANCHOR_GENERATOR_CONFIG') and dh.get('RPN_HEAD_CFGS'):
            out.append((path, cfg, dh))
    return out


MULTIHEAD = _multihead_configs()


def test_the_audit_found_configs_to_check():
    """Guard against the glob silently matching nothing and this file passing vacuously."""
    assert len(MULTIHEAD) >= 80, f'expected ~86 multi-head configs, found {len(MULTIHEAD)}'


@pytest.mark.parametrize('path,cfg,dh', MULTIHEAD, ids=lambda x: x.stem if isinstance(x, Path) else '')
def test_anchor_order_matches_concatenated_head_order(path, cfg, dh):
    """The cross-list positional coupling. A mismatch silently mis-sizes every head."""
    anchor_order = [a.get('class_name') for a in dh['ANCHOR_GENERATOR_CONFIG']]
    head_order = [n for h in dh['RPN_HEAD_CFGS'] for n in h.get('HEAD_CLS_NAME', [])]
    assert anchor_order == head_order, (
        f'{path.name}: ANCHOR_GENERATOR_CONFIG order {anchor_order} != concatenated '
        f'HEAD_CLS_NAME order {head_order}. num_anchors_per_location is built in anchor order and '
        f'indexed in head order, so every head would get the wrong anchor count, silently.')


@pytest.mark.parametrize('path,cfg,dh', MULTIHEAD, ids=lambda x: x.stem if isinstance(x, Path) else '')
def test_every_head_class_is_a_model_class(path, cfg, dh):
    """`make_multihead` does self.class_names.index(name); a name not in CLASS_NAMES is a
    ValueError at model build, minutes into a GPU allocation."""
    class_names = cfg.get('CLASS_NAMES')
    if class_names is None:      # inherited from a base config; checked where it is declared
        pytest.skip('CLASS_NAMES not declared inline')
    head_names = [n for h in dh['RPN_HEAD_CFGS'] for n in h.get('HEAD_CLS_NAME', [])]
    missing = [n for n in head_names if n not in class_names]
    assert not missing, (
        f'{path.name}: HEAD_CLS_NAME references {missing}, absent from CLASS_NAMES {class_names}')


@pytest.mark.parametrize('path,cfg,dh', MULTIHEAD, ids=lambda x: x.stem if isinstance(x, Path) else '')
def test_head_classes_are_unique_across_heads(path, cfg, dh):
    """make_multihead concatenates every HEAD_CLS_NAME and comments "It assumes unique classes".
    A duplicate makes `class_names.index()` return the first occurrence, so the second head would
    silently read the first head's anchor count."""
    head_names = [n for h in dh['RPN_HEAD_CFGS'] for n in h.get('HEAD_CLS_NAME', [])]
    dupes = {n for n in head_names if head_names.count(n) > 1}
    assert not dupes, f'{path.name}: {dupes} appear in more than one head'
