"""Count-normalized sparse convolution (pcdet/models/model_utils/count_norm_conv.py; experiments_md 20261003_05 §5 item 64,
design fixed 2026-11-27 13:02 JST). The spconv tests need a GPU (no CPU path in this image) and skip without one; the config
and key-absent tests run anywhere. Run from the repo root."""
import os
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / 'tools'
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason='spconv needs a GPU here')


def sparse(idx, c=2, shape=(6, 6, 6)):
    import spconv.pytorch as spconv
    i = torch.tensor(idx, dtype=torch.int32).cuda()
    return spconv.SparseConvTensor(torch.randn(len(idx), c).cuda(), i, list(shape), 1)


@gpu
def test_counts_and_scale_on_a_hand_built_tensor():
    import spconv.pytorch as spconv
    from pcdet.models.model_utils.count_norm_conv import CountNormConv
    # (b, z, y, x): an isolated site, and two neighbours along x
    x = sparse([[0, 1, 1, 1], [0, 4, 4, 1], [0, 4, 4, 2]])
    conv = spconv.SubMConv3d(2, 3, 3, padding=1, bias=False, indice_key='t').cuda()
    w = CountNormConv(conv)
    c = w.count(x)
    assert c.features.flatten().tolist() == [1.0, 2.0, 2.0]
    plain = conv(x)
    y = w(x)
    assert torch.allclose(y.features, plain.features * torch.tensor([27.0, 13.5, 13.5]).cuda()[:, None], atol=1e-4)


@gpu
def test_scale_off_is_the_original_convolution_and_strided_sites_match():
    import spconv.pytorch as spconv
    from pcdet.models.model_utils.count_norm_conv import CountNormConv
    x = sparse([[0, 1, 1, 1], [0, 2, 2, 2], [0, 4, 4, 4], [0, 5, 1, 3]])
    conv = spconv.SparseConv3d(2, 3, 3, stride=2, padding=1, bias=False, indice_key='s').cuda()
    off = CountNormConv(conv, scale=False)
    assert torch.equal(off(x).features, conv(x).features)
    on = CountNormConv(conv)
    y = on(x)                                               # raises if the twin's output sites differ
    c = on.count(x)
    assert torch.equal(c.indices, y.indices) and (c.features >= 1).all()


def test_build_network_wraps_only_with_the_key(monkeypatch):
    import pcdet.models as models
    import spconv.pytorch as spconv

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone_3d = spconv.SparseSequential(spconv.SubMConv3d(4, 4, 3, padding=1, bias=False, indice_key='a'),
                                                       torch.nn.BatchNorm1d(4),
                                                       spconv.SparseConv3d(4, 4, 3, stride=2, padding=1, bias=False))
            self.head = torch.nn.Conv2d(4, 4, 1)

    monkeypatch.setattr(models, 'build_detector', lambda model_cfg, num_class, dataset: Toy())
    ds = EasyDict(point_cloud_range=[0, -40, -3, 70.4, 40, 1])
    plain = models.build_network(EasyDict(NAME='x'), 3, ds)
    assert not hasattr(plain, 'n_count_norm') and set(plain.state_dict()) == set(Toy().state_dict())
    wrapped = models.build_network(EasyDict(NAME='x', COUNT_NORM_CONV=True), 3, ds)
    assert wrapped.n_count_norm == 2
    keys = set(wrapped.state_dict())
    assert 'backbone_3d.0.conv.weight' in keys and 'backbone_3d.2.conv.weight' in keys
    assert not any('counter' in k for k in keys)                  # the counting twin holds no state and no parameter
    assert sum(p.numel() for p in wrapped.parameters()) == sum(p.numel() for p in Toy().parameters())


def test_item64_config_is_p_k_plus_the_key_only():
    from pcdet.config import cfg_from_yaml_file
    cwd = os.getcwd(); os.chdir(TOOLS)
    try:
        pk = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml', EasyDict())
        si = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/item64/centerpoint-item64-pk-si.yaml', EasyDict())
        sp = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/item64/centerpoint-item64-pk-si-on-sprime.yaml', EasyDict())
    finally:
        os.chdir(cwd)
    m = dict(si.MODEL); assert m.pop('COUNT_NORM_CONV') is True
    assert m == dict(pk.MODEL)
    for k in set(pk) | set(si):
        if k not in ('MODEL', '_BASE_CONFIG_'):
            assert si.get(k) == pk.get(k), k
    assert sp.DATA_CONFIG_TAR.EVAL_FACTOR_SWAP.RING_THIN == pk.DATA_CONFIG.TRAIN_FACTOR_SWAP.RING_THIN and sp.MODEL == si.MODEL


def test_item64_readings():
    import item64_si as m

    def row(name, seed, pooled, rings=(90, 90, 74, 56, 30)):
        return dict(row=name, seed=seed, reproduced=True, pooled={'bev': pooled, '3d': 50.0}, rings=[{'bev': v} for v in rings])
    rows = []
    for s, (none, src) in zip(m.SEEDS, ((35.0, 69.0), (30.0, 68.0))):
        rows += [row('P_none', s, 19.77 if s == 666 else 22.73), row('P_SI_none', s, none),
                 row('P_src', s, 67.8), row('P_SI_src', s, src)]
    assert m.h3(rows)[1] == 'partly'                                   # +15.2 and +7.3
    assert m.h1(rows)[1] == 'unclear'                                  # +1.2 and +0.2
    assert m.eval_dir('ada', 666).parts[-2:] == ('eval', 'ada_s666')
