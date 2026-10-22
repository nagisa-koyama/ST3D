"""Check PL-TGTONLY (experiments_md 20261022_08; design T05 §5, fixed in experiments_md 11f3ed5, T4 added in 7264a3e):
CPU checks before the first GPU step of the four target-only pseudo-labelling rows T1-T4.

1. Each config resolves to its with-source twin plus exactly four keys: SELF_TRAIN.SRC.USE_DATA False,
   SELF_TRAIN.EPOCH_FOLLOWS 'target', OPTIMIZATION.NUM_EPOCHS and SELF_TRAIN.PROG_AUG.UPDATE_AUG. The last two are
   recomputed here from the twin's own schedule: the epoch count whose step total is nearest the twin's, and each
   augmentation step moved to the target epoch whose start is nearest the twin's step.
2. The teacher policy stays the twin's: a separate frozen teacher, one pass, no TEACHER_IS_STUDENT, no DSNorm, no
   TEACHER_ADABN.
3. train_one_epoch_st under each row's real resolved config: the source readers are never read, there is one forward
   per iteration and it carries the target batch (domain_label 1), the optimiser steps every iteration, and the
   pseudo-label counters (pos / ign) are read from the target batch.
4. The plan train.py sizes and the loop's epoch length come from the same rule (target branch).
"""
import copy
import math
import os
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))
import _init_path  # noqa: F401,E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402

D = 'cfgs/da-ieee-access/'
# row config -> (twin config, twin iterations per epoch (source-following), target iterations per epoch).
# Iteration counts are the twins' logged `epoch length follows source: <src> iterations (source <src>, target <tgt>)`
# lines (27812 / 29334 / 27704 / 27592); the target counts are what the target-following rows will run.
ROWS = {
    'centerpoint-global-st3d-tgtonly-waymo2nuscenes.yaml': ('centerpoint-global-st3d-waymo2nuscenes.yaml', 13174, 4689),
    'centerpoint-ringpattern-st3d-tgtonly-waymo2nuscenes.yaml':
        ('centerpoint-ringpattern-st3d-waymo2nuscenes.yaml', 13174, 4689),
    'centerpoint-gblobs-accum-legaldepth-st3d-tgtonly-nuscenes2waymo.yaml':
        ('centerpoint-gblobs-accum-legaldepth-st3d-nuscenes2waymo.yaml', 4689, 2635),
    'centerpoint-st3d-cb-tgtonly-lyft2nuscenes.yaml': ('centerpoint-st3d-cb-lyft2nuscenes.yaml', 3150, 4689),
}
INTENDED = {'OPTIMIZATION.NUM_EPOCHS', 'SELF_TRAIN.EPOCH_FOLLOWS', 'SELF_TRAIN.SRC.USE_DATA',
            'SELF_TRAIN.PROG_AUG.UPDATE_AUG'}
NAMES = ['Car', 'Pedestrian', 'Cyclist']


def _load(name):
    old = os.getcwd()
    os.chdir(ROOT / 'tools')
    try:
        return cfg_from_yaml_file(D + name, EasyDict())
    finally:
        os.chdir(old)


def _flat(x, path=''):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(_flat(v, f'{path}.{k}' if path else str(k)))
        return out
    if isinstance(x, (list, tuple)):
        return {path: [(_flat(v) if isinstance(v, dict) else v) for v in x]}
    return {path: x}


def _half_up(x):
    return int(math.floor(x + 0.5))


@pytest.fixture(scope='module', params=sorted(ROWS))
def pair(request):
    row = request.param
    twin, src_it, tgt_it = ROWS[row]
    return row, _load(row), _load(twin), src_it, tgt_it


def test_only_the_four_intended_keys_differ_from_the_twin(pair):
    row, c, t, _, _ = pair
    a, b = _flat(c), _flat(t)
    # the top-level `_BASE_CONFIG_` is the pointer each file declares (the row names its twin), not a recipe key
    differ = {k for k in set(a) | set(b) if k != '_BASE_CONFIG_' and a.get(k, '<absent>') != b.get(k, '<absent>')}
    assert a['_BASE_CONFIG_'] == D + ROWS[row][0]
    assert differ == INTENDED, f'{row}: unexpected {sorted(differ - INTENDED)}, missing {sorted(INTENDED - differ)}'
    assert c.SELF_TRAIN.SRC.USE_DATA is False and t.SELF_TRAIN.SRC.USE_DATA is True
    assert c.SELF_TRAIN.EPOCH_FOLLOWS == 'target' and t.SELF_TRAIN.EPOCH_FOLLOWS == 'source'
    assert c.SELF_TRAIN.TAR.USE_DATA is True
    # the rest of SRC is the twin's (merged, not replaced)
    assert {k: v for k, v in c.SELF_TRAIN.SRC.items() if k != 'USE_DATA'} == \
        {k: v for k, v in t.SELF_TRAIN.SRC.items() if k != 'USE_DATA'}


def test_epochs_match_the_twin_in_steps_as_closely_as_an_integer_allows(pair):
    row, c, t, src_it, tgt_it = pair
    twin_steps = src_it * t.OPTIMIZATION.NUM_EPOCHS
    best = min(range(1, 200), key=lambda e: abs(e * tgt_it - twin_steps))
    assert c.OPTIMIZATION.NUM_EPOCHS == best, f'{row}: {c.OPTIMIZATION.NUM_EPOCHS} epochs, nearest is {best}'
    assert abs(best * tgt_it - twin_steps) / twin_steps < 0.02


def test_progressive_augmentation_steps_where_the_twin_stepped(pair):
    row, c, t, src_it, tgt_it = pair
    expected = [_half_up(u * src_it / tgt_it) for u in t.SELF_TRAIN.PROG_AUG.UPDATE_AUG]
    assert list(c.SELF_TRAIN.PROG_AUG.UPDATE_AUG) == expected, row
    assert c.SELF_TRAIN.PROG_AUG.ENABLED and c.SELF_TRAIN.PROG_AUG.SCALE == t.SELF_TRAIN.PROG_AUG.SCALE
    for u_new, u_twin in zip(c.SELF_TRAIN.PROG_AUG.UPDATE_AUG, t.SELF_TRAIN.PROG_AUG.UPDATE_AUG):
        assert abs(u_new * tgt_it - u_twin * src_it) <= tgt_it / 2
        assert u_new < c.OPTIMIZATION.NUM_EPOCHS


def test_the_teacher_policy_is_the_twins_frozen_single_pass(pair):
    row, c, _, _, _ = pair
    s = c.SELF_TRAIN
    assert s.FROZEN_TEACHER_SINGLE_PASS is True
    assert not s.get('TEACHER_IS_STUDENT', False)
    assert not s.get('DSNORM', False) and not s.get('TEACHER_ADABN', None)
    assert 'MODEL_TEACHER' in s and list(s.UPDATE_PSEUDO_LABEL) == [0]
    assert c.DATA_CONFIG_TAR.USE_PSEUDO_LABEL is True
    assert len(s.SCORE_THRESH) == len(s.NEG_THRESH) == 3


def test_the_plan_and_the_loop_count_one_target_pass_per_epoch(pair):
    from train_utils.train_st_utils import self_training_iters_per_epoch
    row, c, _, src_it, tgt_it = pair

    class _L:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    iters, follows = self_training_iters_per_epoch(c.SELF_TRAIN, [_L(src_it)], _L(tgt_it))
    assert (iters, follows) == (tgt_it, 'target')


def test_the_loop_trains_on_the_target_alone_and_never_reads_the_source(monkeypatch, pair):
    import train_utils.train_st_utils as tsu
    row, c, _, _, _ = pair
    monkeypatch.setattr(tsu, 'cfg', copy.deepcopy(c))
    monkeypatch.setattr(tsu.wandb, 'log', lambda *a, **k: None)

    class _NoSource:
        class dataloader:                      # noqa: N801
            class dataset:                     # noqa: N801
                def __len__(self):
                    return 10

        def read_data(self):
            raise AssertionError('the source was read with SRC.USE_DATA False')

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    steps = []
    orig_step = opt.step

    def _step(*a, **k):
        steps.append(1)
        return orig_step(*a, **k)

    opt.step = _step
    w0 = model.weight.detach().clone()
    n_it = 5
    batches = [{'x': torch.randn(3, 4), 'pos_ps_bbox': torch.tensor([[2., 1., 0.]] * 3),
                'ign_ps_bbox': torch.tensor([[1., 3., 0.]] * 3)} for _ in range(n_it)]
    seen = []

    def model_func(m, batch):
        seen.append(batch['domain_label'])
        return (m(batch['x']) ** 2).mean(), {}, {}, None

    class _DS:
        class_names = NAMES

    class _Loader:
        class dataset:                         # noqa: N801
            dataset = _DS()

        def __len__(self):
            return n_it

        def __iter__(self):
            return iter(batches)

    class _Sched:
        def step(self, it):
            pass

    class _Tbar:
        def set_postfix(self, *a, **k):
            pass

        def refresh(self):
            pass

    n = tsu.train_one_epoch_st(model, opt, [_NoSource()], _Loader(), model_func, _Sched(), 0,
                               EasyDict({'GRAD_NORM_CLIP': 10}), rank=0, tbar=_Tbar(), total_it_each_epoch=n_it,
                               dataloader_iter=iter(batches))
    assert n == n_it and len(steps) == n_it
    assert seen == [1] * n_it, 'one forward per iteration, and it is the target batch'
    assert not torch.equal(model.weight.detach(), w0)
