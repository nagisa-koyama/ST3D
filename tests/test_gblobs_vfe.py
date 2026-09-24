"""Tests for GBlobsVFE, the CVPR 2025 GBlobs voxel feature encoder.

GBlobs replaces MeanVFE's absolute xyz centroid with [mean - voxel centre, flatten(covariance)].
The properties worth pinning down are:

  * the declared output dim matches the tensor actually produced, for every flag combination and
    for C == 4 as well as C == 3. Upstream gets this wrong for C != 3 with RELATIVE_DISTANCE
    (declares C + C**2 while slicing the position block to 3), which would build the 3D backbone
    with the wrong input_channels.
  * zero padding does not bias the mean or the covariance. Voxels are padded to
    MAX_POINTS_PER_VOXEL, so every statistic has to be masked.
  * the covariance block is translation invariant. That is the whole claim of the method, so it
    gets an explicit test rather than being left implicit.
  * with both flags off, the position block is exactly MeanVFE - which is what makes the 4-cell
    ablation a single config key.

See experiments_md/20260922_04 for the plan, and the module docstring of gblobs_vfe.py for the two
deliberate deviations from upstream.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.models.backbones_3d.vfe.gblobs_vfe import GBlobsVFE  # noqa: E402
from pcdet.models.backbones_3d.vfe.mean_vfe import MeanVFE  # noqa: E402

VOXEL_SIZE = [0.1, 0.1, 0.2]
POINT_CLOUD_RANGE = [-75.2, -75.2, -2.0, 75.2, 75.2, 4.0]


def _vfe(num_point_features=3, **flags):
    return GBlobsVFE(
        model_cfg=EasyDict(dict(NAME='GBlobsVFE', **flags)),
        num_point_features=num_point_features,
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE,
    )


def _batch(num_points_per_voxel, num_point_features=3, max_points=8, seed=0):
    """Build a padded voxel batch whose padding rows are exactly zero, as spconv's produces."""
    rng = np.random.default_rng(seed)
    num_voxels = len(num_points_per_voxel)
    voxels = np.zeros((num_voxels, max_points, num_point_features), dtype=np.float32)
    for i, n in enumerate(num_points_per_voxel):
        voxels[i, :n] = rng.normal(scale=0.03, size=(n, num_point_features))
    coords = np.stack([
        np.zeros(num_voxels),                       # batch index
        rng.integers(0, 30, num_voxels),            # z
        rng.integers(0, 1500, num_voxels),          # y
        rng.integers(0, 1500, num_voxels),          # x
    ], axis=1).astype(np.int32)
    return {
        'voxels': torch.from_numpy(voxels),
        'voxel_num_points': torch.from_numpy(np.asarray(num_points_per_voxel, dtype=np.int32)),
        'voxel_coords': torch.from_numpy(coords),
    }


# ---------------------------------------------------------------- output dimension

@pytest.mark.parametrize('num_point_features', [3, 4])
@pytest.mark.parametrize('rel_d', [False, True])
@pytest.mark.parametrize('cov_only', [False, True])
def test_declared_output_dim_matches_tensor(num_point_features, rel_d, cov_only):
    """The bug this guards against: C=4 with RELATIVE_DISTANCE declares 20 but emits 19."""
    vfe = _vfe(num_point_features, RELATIVE_DISTANCE=rel_d, COVARIANCE_ONLY=cov_only)
    batch = _batch([5, 1, 8], num_point_features=num_point_features)
    out = vfe(batch)['voxel_features']
    assert out.shape[1] == vfe.get_output_feature_dim()


def test_output_dim_is_twelve_for_the_da_configs():
    """Every da_* config is xyz-only, so the number the 3D backbone is built with is 12."""
    assert _vfe(3, RELATIVE_DISTANCE=True).get_output_feature_dim() == 12


# ---------------------------------------------------------------- padding

def test_padding_does_not_bias_mean_or_covariance():
    """A half-full voxel must give the same statistics as the dense voxel of its real points."""
    n = 5
    batch = _batch([n], max_points=32)
    vfe = _vfe(3, RELATIVE_DISTANCE=False)
    out = vfe(dict(batch))['voxel_features'].numpy()[0]

    real = batch['voxels'].numpy()[0, :n]
    np.testing.assert_allclose(out[:3], real.mean(axis=0), rtol=0, atol=1e-6)
    np.testing.assert_allclose(out[3:].reshape(3, 3), np.cov(real, rowvar=False, ddof=1),
                               rtol=1e-5, atol=1e-7)


def test_single_point_voxel_has_zero_covariance_and_is_finite():
    """N=1 divides by N-1; it must clamp rather than produce inf/nan."""
    out = _vfe(3, RELATIVE_DISTANCE=False)(_batch([1]))['voxel_features']
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0, 3:], torch.zeros(9))


def test_empty_voxel_is_finite():
    """Defensive: a zero-count row should not produce nan even though spconv never emits one."""
    out = _vfe(3, RELATIVE_DISTANCE=False)(_batch([0, 4]))['voxel_features']
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------- the actual claim

def test_covariance_block_is_translation_invariant():
    """Shifting every point by a constant leaves the covariance untouched. This is the method."""
    batch = _batch([6, 3, 8])
    vfe = _vfe(3, RELATIVE_DISTANCE=False)
    before = vfe(dict(batch))['voxel_features'][:, 3:].clone()

    shifted = dict(batch)
    mask = (torch.arange(batch['voxels'].shape[1]).view(1, -1)
            < batch['voxel_num_points'].view(-1, 1)).unsqueeze(-1).float()
    shifted['voxels'] = batch['voxels'] + mask * torch.tensor([10.0, -4.0, 1.7])
    after = vfe(shifted)['voxel_features'][:, 3:]

    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-6)


def test_relative_distance_puts_the_mean_inside_half_a_voxel():
    """With RELATIVE_DISTANCE the position block is an offset from the voxel centre, so for a
    physically consistent batch it is bounded by half a voxel per axis - tens of metres of
    absolute coordinate collapse to centimetres. That bound IS the removal of absolute position.
    """
    rng = np.random.default_rng(3)
    voxel_size = np.asarray(VOXEL_SIZE)
    n, max_points = 6, 8
    coords_zyx = np.stack([rng.integers(0, 30, 4), rng.integers(0, 1500, 4),
                           rng.integers(0, 1500, 4)], axis=1)
    centres = (coords_zyx[:, ::-1] + 0.5) * voxel_size + np.asarray(POINT_CLOUD_RANGE[0:3])

    voxels = np.zeros((4, max_points, 3), dtype=np.float32)
    for i in range(4):
        voxels[i, :n] = centres[i] + rng.uniform(-0.5, 0.5, (n, 3)) * voxel_size
    batch = {
        'voxels': torch.from_numpy(voxels),
        'voxel_num_points': torch.from_numpy(np.full(4, n, dtype=np.int32)),
        'voxel_coords': torch.from_numpy(
            np.concatenate([np.zeros((4, 1)), coords_zyx], axis=1).astype(np.int32)),
    }

    absolute = _vfe(3, RELATIVE_DISTANCE=False)(dict(batch))['voxel_features'][:, 0:3]
    relative = _vfe(3, RELATIVE_DISTANCE=True)(dict(batch))['voxel_features'][:, 0:3]

    assert absolute.abs().max() > 1.0                      # the absolute coordinate is metres
    assert (relative.abs() <= torch.tensor(voxel_size).float() / 2 + 1e-6).all()


# ---------------------------------------------------------------- reduction to MeanVFE

def test_position_block_equals_mean_vfe_with_both_flags_off():
    """RELATIVE_DISTANCE=False, COVARIANCE_ONLY=False is MeanVFE plus a covariance block, which
    is what makes the ablation a single config key."""
    batch = _batch([5, 2, 8])
    gblobs = _vfe(3, RELATIVE_DISTANCE=False, COVARIANCE_ONLY=False)(dict(batch))['voxel_features']
    mean = MeanVFE(model_cfg=EasyDict({'NAME': 'MeanVFE'}), num_point_features=3)
    reference = mean(dict(batch))['voxel_features']
    torch.testing.assert_close(gblobs[:, :3], reference, rtol=1e-5, atol=1e-7)


def test_covariance_only_drops_the_position_block():
    batch = _batch([5, 2, 8])
    full = _vfe(3, RELATIVE_DISTANCE=False)(dict(batch))['voxel_features']
    cov_only = _vfe(3, COVARIANCE_ONLY=True)(dict(batch))['voxel_features']
    assert cov_only.shape[1] == 9
    torch.testing.assert_close(cov_only, full[:, 3:], rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------- the escape hatch

def test_covariance_scale_defaults_to_an_exact_no_op():
    batch = _batch([5, 2, 8])
    a = _vfe(3, RELATIVE_DISTANCE=True)(dict(batch))['voxel_features']
    b = _vfe(3, RELATIVE_DISTANCE=True, COVARIANCE_SCALE=1.0)(dict(batch))['voxel_features']
    assert torch.equal(a, b)


def test_covariance_scale_touches_only_the_covariance_block():
    batch = _batch([5, 2, 8])
    a = _vfe(3, RELATIVE_DISTANCE=True)(dict(batch))['voxel_features']
    b = _vfe(3, RELATIVE_DISTANCE=True, COVARIANCE_SCALE=1e4)(dict(batch))['voxel_features']
    torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(a[:, 3:] * 1e4, b[:, 3:], rtol=1e-5, atol=1e-7)
