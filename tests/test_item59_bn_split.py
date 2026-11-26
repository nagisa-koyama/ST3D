"""Item 59 (experiments_md 20261003_05 §5 item 59, design fixed 2026-11-27 08:15 JST): hybrid checkpoints, cuts, generated
configs, readings. CPU only. Run from the repo root."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item59_bn_split as m  # noqa: E402


def state(seed):
    g = torch.Generator().manual_seed(seed)
    s = {}
    for p in ('backbone_3d.conv_input.1', 'backbone_2d.blocks.0.2', 'dense_head.shared_conv.1'):
        s[p + '.weight'] = torch.rand(4, generator=g)
        s[p + '.bias'] = torch.rand(4, generator=g)
        s[p + '.running_mean'] = torch.rand(4, generator=g)
        s[p + '.running_var'] = torch.rand(4, generator=g)
        s[p + '.num_batches_tracked'] = torch.tensor(seed)
    s['backbone_3d.conv_input.0.weight'] = torch.rand(3, 3, generator=g)
    s['dense_head.heads_list.0.hm.1.weight'] = torch.rand(2, 2, generator=g)
    s['global_step'] = torch.tensor([seed])
    return s


def test_hybrid_takes_gamma_beta_of_the_group_only():
    pk, g = state(1), state(2)
    for k in ('backbone_3d.conv_input.0.weight', 'dense_head.heads_list.0.hm.1.weight'):
        g[k] = pk[k].clone()                                    # G's non-BN tensors are P_K's (item 58's check)
    h = m.hybrid(pk, g, m.GROUPS['aff_3d'])
    n, prob = m.check_hybrid(h, pk, g, m.GROUPS['aff_3d'])
    assert n == 1 and prob == []
    assert torch.equal(h['backbone_3d.conv_input.1.weight'], g['backbone_3d.conv_input.1.weight'])
    assert torch.equal(h['backbone_2d.blocks.0.2.weight'], pk['backbone_2d.blocks.0.2.weight'])
    assert torch.equal(h['backbone_3d.conv_input.1.running_mean'], pk['backbone_3d.conv_input.1.running_mean'])
    h2 = m.hybrid(pk, g, m.GROUPS['aff_2d'])
    assert m.check_hybrid(h2, pk, g, m.GROUPS['aff_2d'])[0] == 2
    assert m.check_hybrid(m.hybrid(pk, g, m.GROUPS['aff_all']), pk, g, m.GROUPS['aff_all']) == (3, [])


def test_check_hybrid_catches_a_wrong_tensor():
    pk, g = state(1), state(2)
    h = m.hybrid(pk, g, m.GROUPS['aff_3d'])
    h['backbone_2d.blocks.0.2.bias'] = g['backbone_2d.blocks.0.2.bias']           # a gamma / beta outside the group
    h['backbone_3d.conv_input.0.weight'] = pk['backbone_3d.conv_input.0.weight'] * 1.001
    assert sorted(m.check_hybrid(h, pk, g, m.GROUPS['aff_3d'])[1]) == ['backbone_2d.blocks.0.2.bias',
                                                                       'backbone_3d.conv_input.0.weight']


def test_hybrid_path_stem_carries_only_the_epoch():
    p = m.hybrid_path('aff_3d', 666)
    assert ''.join(c for c in p.stem if c.isdigit()) == str(m.PK_EPOCH)


def test_rule_a_matches_plateau_tool_and_leaf_text():
    rng = np.random.default_rng(0)
    ps = {}
    for f in range(50):
        n = 40
        b = np.zeros((n, 9), dtype=np.float32)
        b[:, 0:2] = rng.uniform(-70, 70, (n, 2))
        b[:, 7] = rng.integers(1, 4, n)
        b[:, 8] = np.where(rng.random(n) < 0.8, rng.uniform(0.01, 0.05, n), rng.uniform(0.3, 0.9, n))
        ps['%06d' % f] = {'gt_boxes': b}
    out = m.rule_a(ps)
    assert set(out) == set(m.CLASSES) and all(0 < v['neg'] < 1 for v in out.values())
    cuts = {'666': {'rule_a': out}}
    t = m.leaf_text('bn', 666, cuts)
    v = ', '.join('%.3f' % out[c]['neg'] for c in m.CLASSES)
    assert 'TRAIN_ONLY_BN: True' in t and t.count('[%s]' % v) == 2 and m.B_BASE in t
    assert 'TRAIN_ONLY_BN' not in m.leaf_text('all', 666, cuts)


def test_parse_jobs():
    assert m.parse_jobs(['bn666=1', 'all667=4']) == {('bn', 666): 1, ('all', 667): 4}


def ring_row(name, seed, rings, pooled=70.0):
    return dict(row=name, seed=seed, reproduced=True, rings=[{'bev': x} for x in rings], pooled={'bev': pooled, '3d': 50.0},
                logged_bev=pooled)


def test_rs1_check():
    rows = [ring_row('P_RS1', 666, [1] * 5, 69.36), ring_row(m.ROWS_A['rs1'], 666, [1] * 5, 69.40)]
    assert m.rs1_check(rows, 666)
    rows[1]['pooled']['bev'] = 69.45
    assert not m.rs1_check(rows, 666)
    assert not m.rs1_check(rows, 667)


# ---------------------------------------------------------------- the generated 59b configs (need the repo's pcdet)
def resolve(path):
    import _init_path  # noqa: F401
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    import os
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        return cfg_from_yaml_file(path, EasyDict())
    finally:
        os.chdir(cwd)


def test_committed_base_config_is_what_the_tool_generates_with_the_repos_pcdet():
    """2026-11-27: the first base config, written by running the tool before it imported _init_path, came from the image's
    stale pcdet and lost G's z-shift augmentation (base overwrote child). The committed file must equal a fresh generation."""
    assert (TOOLS / m.B_BASE).read_text() == m.base_config_text()


def test_base_config_target_and_teacher():
    b = resolve(m.B_BASE)
    g = resolve('cfgs/da-ieee-access-analysis/item58/centerpoint-item58-p-bnonly-g.yaml')
    t = b.DATA_CONFIG_TAR
    assert t.DATA_SPLIT == {'train': 'test', 'test': 'val'} and t.INFO_PATH.train == ['kitti_infos_test.pkl']
    assert t.USE_PSEUDO_LABEL and t.TEST.BOX_FILTER.FOV_FILTER and t.INFO_PATH.test == ['kitti_infos_val.pkl']
    assert m._plain(t.DATA_AUGMENTOR) == m._plain(g.DATA_CONFIG.DATA_AUGMENTOR)          # G's augmentation, z shift included
    assert [a.NAME for a in t.DATA_AUGMENTOR.AUG_CONFIG_LIST][-1] == 'random_world_translation'
    st = b.SELF_TRAIN
    teacher = {k: v for k, v in m._plain(st.MODEL_TEACHER).items() if k not in ('CLASS_NAMES', 'ONTOLOGY')}
    assert teacher == m._plain(b.MODEL) and st.MODEL_TEACHER.CLASS_NAMES == list(b.CLASS_NAMES)
    assert st.FROZEN_TEACHER_SINGLE_PASS and not st.SRC.USE_DATA and st.EPOCH_FOLLOWS == 'target'
    assert not st.PROG_AUG.ENABLED and not st.MEMORY_ENSEMBLE.ENABLED and st.TEACHER_ADABN.MIX == 1.0
    assert 'SCORE_THRESH' not in st and b.OPTIMIZATION.NUM_EPOCHS == m.B_EPOCH
    assert b.OPTIMIZATION.LR == g.OPTIMIZATION.LR and b.OPTIMIZATION.OPTIMIZER == g.OPTIMIZATION.OPTIMIZER
    assert b.MODEL == resolve(m.EVAL_CFG).MODEL and not b.OPTIMIZATION.get('TRAIN_ONLY_BN', False)


def test_leaf_configs_once_written():
    import json
    leaves = [(a, s) for a in m.ROWS_B for s in (666, 667) if (TOOLS / m.leaf_path(a, s)).exists()]
    if not leaves:
        pytest.skip('leaf configs not generated yet (job P0)')
    cuts = json.load(open(m.OUT / 'cuts.json'))
    for arm, seed in leaves:
        c = resolve(m.leaf_path(arm, seed))
        v = [cuts[str(seed)]['rule_a'][k]['neg'] for k in m.CLASSES]
        assert list(c.SELF_TRAIN.SCORE_THRESH) == v and list(c.SELF_TRAIN.NEG_THRESH) == v
        assert bool(c.OPTIMIZATION.get('TRAIN_ONLY_BN', False)) == (arm == 'bn')
        assert (TOOLS / m.leaf_path(arm, seed)).read_text() == m.leaf_text(arm, seed, cuts)
