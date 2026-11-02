"""IA-SSD: an x,y,z-only input (`features=None`) must pass through the sampling set-abstraction layer.

Defect: the ported `PointnetSAModuleMSG_WithSampling.forward` sliced `features.transpose(1, 2)[...]`
unconditionally. The da-ieee-access family feeds `used_feature_list: ['x', 'y', 'z']`, so
`IASSD_Backbone.break_up_pc` hands the first SA layer `features=None` (the sentinel in
memory/repo/bug_prevention_checklist.md, rule 1) and smoke 27499 died at iteration 0 with
`AttributeError: 'NoneType' object has no attribute 'transpose'` (2026-10-06). The published IA-SSD
config always had intensity there, which is why upstream never saw it.
Fix: the guard in `pcdet/ops/pointnet2/pointnet2_batch_iassd/pointnet2_modules.py` (`feature_tmp` is
None when there are no features; it is read only by F-FPS / FS sampling), committed with the port in
ST3D 1f35e70 (2026-10-06). Report: experiments_md/20261006_01 §3.8.

CPU only: the four CUDA kernels the layer calls (furthest point sampling, gather, ball query,
grouping) are replaced by torch equivalents through monkeypatch, so the layer's own Python - the
code that broke - runs for real on the master node. The pre-fix slicing expression (copied from the
port's source, UADA3D 5de3c16) is kept as a reference and shown to die on the same input.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools')); sys.path.insert(0, str(ROOT))
import _init_path  # noqa: F401,E402

pytest.importorskip('pcdet.ops.pointnet2.pointnet2_batch_iassd.pointnet2_utils')
from pcdet.models.backbones_3d.IASSD_backbone import IASSD_Backbone  # noqa: E402
from pcdet.ops.pointnet2.pointnet2_batch_iassd import pointnet2_modules as pm  # noqa: E402
from pcdet.ops.pointnet2.pointnet2_batch_iassd import pointnet2_utils as pu  # noqa: E402


# ---- CPU stand-ins for the CUDA kernels --------------------------------------------------------

def _fps(xyz, npoint):
    """Greedy farthest point sampling, (B, N, 3) -> (B, npoint) int32, starting at index 0 like the kernel."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.int32)
    for b in range(B):
        dist = torch.full((N,), float('inf'))
        cur = 0
        for i in range(npoint):
            idx[b, i] = cur
            dist = torch.minimum(dist, ((xyz[b] - xyz[b, cur]) ** 2).sum(-1))
            cur = int(torch.argmax(dist))
    return idx


def _gather(features, idx):
    """(B, C, N) gathered at (B, M) -> (B, C, M)."""
    return torch.gather(features, 2, idx.long().unsqueeze(1).expand(-1, features.shape[1], -1)).contiguous()


def _ball_query(radius, nsample, xyz, new_xyz):
    """(B, M, nsample) int32 indices of points within `radius` of each centre, padded with the first hit."""
    B, M, _ = new_xyz.shape
    out = torch.zeros(B, M, nsample, dtype=torch.int32)
    d = torch.cdist(new_xyz, xyz)
    for b in range(B):
        for m in range(M):
            near = torch.nonzero(d[b, m] <= radius).flatten()
            if near.numel() == 0:
                near = torch.zeros(1, dtype=torch.long)
            sel = near[:nsample]
            out[b, m, :sel.numel()] = sel.int()
            out[b, m, sel.numel():] = int(sel[0])
    return out


def _grouping(features, idx):
    """(B, C, N) grouped by (B, M, S) -> (B, C, M, S)."""
    B, C, _ = features.shape
    _, M, S = idx.shape
    flat = idx.long().reshape(B, 1, M * S).expand(-1, C, -1)
    return torch.gather(features, 2, flat).reshape(B, C, M, S)


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(pu, 'furthest_point_sample', _fps)
    monkeypatch.setattr(pu, 'gather_operation', _gather)
    monkeypatch.setattr(pu, 'ball_query', _ball_query)
    monkeypatch.setattr(pu, 'grouping_operation', _grouping)


def _layer(in_channels, sample_type='D-FPS', npoint=4, num_class=3):
    torch.manual_seed(0)
    layer = pm.PointnetSAModuleMSG_WithSampling(
        npoint_list=[npoint], sample_range_list=[-1], sample_type_list=[sample_type],
        radii=[1.5], nsamples=[8], mlps=[[in_channels, 8]], use_xyz=True, dilated_group=False,
        pool_method='max_pool', aggregation_mlp=[8], confidence_mlp=[8], num_class=num_class)
    return layer.eval()


def _cloud(B=2, N=16):
    g = torch.Generator().manual_seed(1)
    return torch.rand(B, N, 3, generator=g) * 4.0


def _pre_fix_slice(features, last_sample_end_index=0):
    """The sample_range == -1 branch as ported (UADA3D 5de3c16): no None guard."""
    return features.transpose(1, 2)[:, last_sample_end_index:, :].contiguous()


# ---- break_up_pc: where None comes from --------------------------------------------------------

def test_break_up_pc_gives_none_features_for_an_xyz_only_cloud():
    pc = torch.tensor([[0., 1., 2., 3.], [0., 4., 5., 6.], [1., 7., 8., 9.]])  # batch_idx, x, y, z
    batch_idx, xyz, features = IASSD_Backbone.break_up_pc(None, pc)
    assert features is None
    assert torch.equal(batch_idx, pc[:, 0]) and torch.equal(xyz, pc[:, 1:4])


def test_break_up_pc_keeps_extra_columns_as_features():
    pc = torch.tensor([[0., 1., 2., 3., 0.5], [0., 4., 5., 6., 0.7]])
    _, _, features = IASSD_Backbone.break_up_pc(None, pc)
    assert features is not None and features.shape == (2, 1)
    assert torch.equal(features[:, 0], pc[:, 4])


# ---- the layer ---------------------------------------------------------------------------------

def test_pre_fix_slicing_dies_on_none():
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'transpose'"):
        _pre_fix_slice(None)


def test_dfps_layer_runs_without_features(cpu_kernels):
    xyz = _cloud()
    layer = _layer(in_channels=0)
    new_xyz, new_features, cls_features = layer(xyz, None, None)
    assert new_xyz.shape == (2, 4, 3)
    assert new_features.shape == (2, 8, 4)
    assert cls_features.shape == (2, 4, 3)
    assert torch.isfinite(new_features).all() and torch.isfinite(cls_features).all()
    # the sampled centres are points of the input cloud
    for b in range(2):
        d = torch.cdist(new_xyz[b], xyz[b]).min(dim=1).values
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)


def test_no_downsampling_branch_runs_without_features(cpu_kernels):
    """N <= npoint takes the arange path; it must also tolerate features=None."""
    xyz = _cloud(N=4)
    new_xyz, new_features, _ = _layer(in_channels=0, npoint=8)(xyz, None, None)
    assert torch.equal(new_xyz, xyz) and new_features.shape == (2, 8, 4)


def test_ctr_aware_layer_runs_without_features(cpu_kernels):
    """The vote-centre path (ctr_xyz given) skips sampling and groups about the centres."""
    xyz = _cloud()
    ctr = xyz[:, :3, :].clone()
    new_xyz, new_features, cls_features = _layer(in_channels=0)(xyz, None, None, ctr_xyz=ctr)
    assert torch.equal(new_xyz, ctr)
    assert new_features.shape == (2, 8, 3) and cls_features.shape == (2, 3, 3)


def test_features_are_still_used_when_present(cpu_kernels):
    """The guard must not have turned the with-features path into the xyz-only one."""
    xyz = _cloud()
    layer = _layer(in_channels=1)
    feats_a = torch.zeros(2, 1, 16)
    feats_b = torch.ones(2, 1, 16)
    _, out_a, _ = layer(xyz, feats_a, None)
    _, out_b, _ = layer(xyz, feats_b, None)
    assert out_a.shape == (2, 8, 4)
    assert not torch.allclose(out_a, out_b), 'the point features did not reach the MLP'


def test_the_backbone_handles_none_the_same_way():
    """IASSD_Backbone.forward reshapes `features` with the same None guard (line 115); pin the expression."""
    src = Path(IASSD_Backbone.__module__.replace('.', '/') + '.py')
    text = (ROOT / src).read_text(encoding='utf-8')
    assert 'if features is not None else None' in text, 'the backbone lost its features=None guard'
