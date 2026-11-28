"""Item 70's arms, readings and gates (tools/analysis/item70_rs_adabn_real.py; experiments_md 20261003_05 §5 item 70, design
fixed 2026-11-28 20:10 JST). Run from the repo root."""
import re
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item70_rs_adabn_real as m  # noqa: E402

O_RINGS = [95.0, 95.0, 84.0, 70.0, 40.0]


def row(name, seed, rings, pooled):
    return dict(row=name, seed=seed, reproduced=True, logged_bev=pooled, pooled={'bev': pooled, '3d': pooled / 2},
                rings=[{'bev': v} for v in rings])


def rows_with(near_gain=0.0, far_gain=0.0, arm='RS5_2d', rs1=None):
    rs1 = rs1 or m.B1
    out = [row('O_ada', 666, O_RINGS, 80.0), row('O_ada', 667, O_RINGS, 80.0)]
    base = [75.0, 85.0, 76.0, 52.0, 20.0]                      # near gap 15, far gap 13 against the oracle
    for s in m.SEEDS:
        out.append(row('RS1', s, base, rs1[s]))
        for name in ('RS5', 'RS5_3d', 'RS5_2d'):
            ng, fg = (near_gain, far_gain) if name == arm else (0.0, 0.0)
            out.append(row(name, s, [base[0] + ng, base[1] + ng, base[2] + fg, base[3] + fg, base[4]], rs1[s] + ng / 2))
    return out


def test_near_words_and_h_near():
    r = m.readings(rows_with(near_gain=8.0))['readings']
    assert r['RS5_2d_near']['word'] == 'most'                  # delta 8 of 15, R 0.53
    assert r['RS5_2d_near']['hypothesis'].startswith("the BEV layers' range-pooled statistics carry part")
    s0 = r['RS5_2d_near']['seeds'][0]
    assert s0['gap_rs1'] == pytest.approx(15.0) and s0['delta'] == pytest.approx(8.0) and s0['R'] == pytest.approx(8 / 15)
    assert m.readings(rows_with(near_gain=4.0))['readings']['RS5_2d_near']['word'] == 'partly'
    assert m.readings(rows_with(near_gain=4.0))['readings']['RS5_2d_near']['hypothesis'].startswith("the BEV layers'")
    w = m.readings(rows_with(near_gain=0.5))['readings']['RS5_2d_near']
    assert w['word'] == 'cannot' and w['hypothesis'] == 'they do not'
    w = m.readings(rows_with(near_gain=-1.5))['readings']['RS5_2d_near']
    assert w['word'] == 'widens' and w['hypothesis'] == 'they do not'


def test_far_words_and_pooled():
    r = m.readings(rows_with(far_gain=7.0, arm='RS5'))['readings']
    assert r['RS5_far']['word'] == 'most' and r['RS5_far']['hypothesis'].startswith('per-range statistics close most')
    assert r['RS5_3d_far']['word'] == 'cannot' and r['RS5_3d_far']['hypothesis'] == 'not most (cannot)'
    assert r['RS5_near']['word'] == 'cannot' and r['RS5_near']['hypothesis'] is None
    x = r['RS5_pooled']['seeds'][0]
    assert x['gap_rs1'] == pytest.approx(80.0 - m.B1[666]) and x['delta'] == pytest.approx(0.0)


def test_gates():
    off = dict(m.B1)
    off[667] = m.B1[667] - 0.31
    assert m.readings(rows_with(near_gain=8.0, rs1=off))['readings'] == 'not read'
    near = dict(m.B1)
    near[667] = m.B1[667] - 0.29
    assert m.readings(rows_with(near_gain=8.0, rs1=near))['readings'] != 'not read'
    rows = rows_with(near_gain=8.0)
    rows[4]['reproduced'] = False
    out = m.readings(rows)
    assert out['readings'] == 'not read' and out['not_reproduced']
    rows = [r for r in rows_with(near_gain=8.0) if not (r['row'] == 'RS5_3d' and r['seed'] == 668)]
    r = m.readings(rows)['readings']
    assert r['RS5_3d_far']['word'] == 'not read' and r['RS5_2d_near']['word'] == 'most'


def test_names_and_paths():
    assert m.tag('item57_2d', 667) == 'i70_item57_2d_s667'
    assert str(m.result_dir('global', 668)).endswith(
        'output/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti/i70_global_s668/eval/epoch_20/i70_global_s668')
    assert [r[2] for r in m.runs(667)] == ['global', 'item57', 'item57_3d', 'item57_2d']
    assert {r[1] for r in m.runs(667)} == {32411} and m.JOBS41A == {666: 32373, 667: 32411, 668: 32434}
    assert m.EPOCH == 20 and m.TOL_RS1 == 0.3


def test_wandb_id_from_the_rs_adabn_log(tmp_path):
    (tmp_path / 'log_rsadabn_20261128-201500.txt').write_text('x\n2026 INFO  [rs-adabn] wandb run ab12cd34\ny\n')
    assert m.wandb_id(tmp_path) == 'ab12cd34'
    assert m.wandb_id(tmp_path / 'none') is None


CENTERPOINT_BN = (['backbone_3d.conv_input.1'] + ['backbone_3d.conv%d.%d.1' % (i, j) for i in range(1, 5) for j in range(5)] +
                  ['backbone_2d.blocks.0.%d' % i for i in range(2, 18, 3)] + ['backbone_2d.deblocks.0.1',
                   'dense_head.shared_conv.1'] + ['dense_head.heads_list.0.%s.0.1' % h for h in ('center', 'hm')])


def _partition(names):
    from pcdet.utils import adabn_range
    a = [n for n in names if re.search(adabn_range.ARM_LAYERS['item57_3d'], n)]
    b = [n for n in names if re.search(adabn_range.ARM_LAYERS['item57_2d'], n)]
    return a, b


def test_arm_layers_partition_and_item57b_unchanged():
    from pcdet.utils import adabn_range
    assert adabn_range.ARM_LAYERS['item57_3d'] == r'^backbone_3d\.'         # item 57b's regex, as it ran
    assert adabn_range.ARM_LAYERS['item57'] is None and adabn_range.ARM_LAYERS['global'] is None
    a, b = _partition(CENTERPOINT_BN)
    assert not set(a) & set(b) and sorted(a + b) == sorted(CENTERPOINT_BN)
    assert not re.search(adabn_range.ARM_LAYERS['item57_2d'], 'xbackbone_2d.blocks.0.1')


@pytest.mark.skipif(not Path('/home/koyama/data/wandb/run-20261109_001646-vviqjz00/files/ckpt/checkpoint_epoch_20.pth').exists(),
                    reason='no 41a checkpoint')
def test_arm_layers_partition_41a_checkpoint():
    import torch
    sd = torch.load('/home/koyama/data/wandb/run-20261109_001646-vviqjz00/files/ckpt/checkpoint_epoch_20.pth',
                    map_location='cpu', weights_only=False)['model_state']
    bn = sorted({k[:-len('.running_mean')] for k in sd if k.endswith('.running_mean')})
    a, b = _partition(bn)
    assert (len(bn), len(a), len(b)) == (34, 21, 13) and sorted(a + b) == bn


@pytest.mark.skipif(not (TOOLS / m.CFG).exists(), reason='no config')
def test_source_block_is_nuscenes_and_shares_the_range():
    import os
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    os.chdir(TOOLS)
    c = cfg_from_yaml_file(m.CFG, EasyDict())
    assert c.DATA_CONFIG.DATASET == 'NuScenesDataset' and c.DATA_CONFIG_TAR.DATASET == 'KittiDataset'
    assert list(c.DATA_CONFIG.POINT_CLOUD_RANGE) == list(c.DATA_CONFIG_TAR.POINT_CLOUD_RANGE)
    assert c.DATA_CONFIG.MAX_SWEEPS == 1
