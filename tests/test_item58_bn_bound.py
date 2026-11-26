"""Item 58's procedure check and readings (experiments_md 20261003_05 §5 item 58, design fixed 2026-11-27 02:56 JST). CPU only."""
import sys
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item58_bn_bound as m  # noqa: E402


def test_norm_key_undoes_the_wrapper_only():
    assert m.norm_key('backbone_3d.conv_input.1.bn.weight') == 'backbone_3d.conv_input.1.weight'
    assert m.norm_key('backbone_3d.conv1.0.bn1.bn.running_var') == 'backbone_3d.conv1.0.bn1.running_var'
    assert m.norm_key('backbone_3d.conv1.0.bn1.weight') == 'backbone_3d.conv1.0.bn1.weight'
    assert m.norm_key('dense_head.bn.conv.weight') == 'dense_head.bn.conv.weight'


def pre_state():
    return {'conv.weight': torch.ones(2, 2), 'bn1.weight': torch.ones(2), 'bn1.bias': torch.zeros(2),
            'bn1.running_mean': torch.zeros(2), 'bn1.running_var': torch.ones(2), 'bn1.num_batches_tracked': torch.tensor(5)}


def test_compare_passes_when_only_normalization_changed_and_fails_otherwise():
    pre = pre_state()
    post = {'conv.weight': torch.ones(2, 2), 'bn1.bn.weight': torch.full((2,), 1.1), 'bn1.bn.bias': torch.zeros(2),
            'bn1.bn.running_mean': torch.full((2,), 0.3), 'bn1.bn.running_var': torch.ones(2),
            'bn1.bn.num_batches_tracked': torch.tensor(9), 'bn1.mod.a': torch.full((5, 2), 0.02), 'bn1.mod.b': torch.zeros(5, 2)}
    c = m.compare(pre, post)
    assert c['non_bn_differ'] == [] and c['missing'] == [] and c['bn_changed'] == 3 and abs(c['mod_max']['a'] - 0.02) < 1e-7
    post['conv.weight'] = torch.ones(2, 2) * 0.999
    assert m.compare(pre, post)['non_bn_differ'] == ['conv.weight']
    del post['bn1.bn.bias']
    assert m.compare(pre, post)['missing'] == ['bn1.bias']


def test_compare_reports_the_step_counter_and_does_not_compare_it():
    """37194 (2026-11-27) flagged Detector3DTemplate.global_step, which counts training steps, as a non-BN tensor that
    differs, on all four rows. It must be reported, not compared; a real weight that differs must still fail."""
    pre = dict(pre_state(), global_step=torch.tensor([94240]))
    post = dict(pre, global_step=torch.tensor([106620]), **{'bn1.weight': torch.full((2,), 1.1)})
    c = m.compare(pre, post)
    assert c['non_bn_differ'] == [] and c['missing'] == [] and c['counters'] == {'global_step': [94240, 106620]}
    post['conv.weight'] = torch.ones(2, 2) * 0.999
    assert m.compare(pre, post)['non_bn_differ'] == ['conv.weight']


def row(name, seed, rings):
    return dict(row=name, seed=seed, reproduced=True, rings=[{'bev': x} for x in rings])


L = [92.4, 95.8, 83.5, 65.7, 32.3]
P0 = [90.8, 91.7, 74.2, 55.5, 31.7]


def test_readings():
    def table(px):
        return [row(n, s, v) for s in (666, 667) for n, v in (('L_ada', L), ('P_RS1', P0), ('P_BN_G', px))]
    closes = [m.per_seed(table([91, 93, 81.0, 62.0, 32.0]), s, 'P_BN_G') for s in (666, 667)]
    assert m.reading(closes) == 'a normalization-level correction can close most of the far residual'
    assert m.reading([m.per_seed(table([91, 92, 74.6, 55.9, 31.7]), s, 'P_BN_G') for s in (666, 667)]) == 'it cannot'
    assert m.reading([m.per_seed(table([91, 92, 77.0, 58.5, 31.7]), s, 'P_BN_G') for s in (666, 667)]) == 'partly'
    check = {'P_BN_G s666': {'ok': False}, 'P_BN_G s667': {'ok': True}}
    assert m.reading([m.per_seed(table([91, 93, 81.0, 62.0, 32.0]), s, 'P_BN_G', check) for s in (666, 667)]) == 'not read'


def test_parse_jobs():
    assert m.parse_jobs(['g666=1', 'r667=4']) == {('g', 666): 1, ('r', 667): 4}
