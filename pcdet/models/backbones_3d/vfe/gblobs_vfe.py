"""GBlobs: per-voxel Gaussian blobs (mean + covariance) in place of the absolute centroid.

Port of the VFE from Malic et al., "GBlobs: Explicit Local Structure via Gaussian Blobs for
Improved Cross-Domain LiDAR-based 3D Object Detection" (CVPR 2025),
https://github.com/malicd/GBlobs.

The whole method is this one class. Where `MeanVFE` emits the absolute xyz centroid of each
voxel's points (3 numbers for our xyz-only DA configs), `GBlobsVFE` emits

    [ mean - voxel_centre , flatten(cov(points)) ]        3 + 9 = 12 numbers

i.e. the local neighbourhood described as a Gaussian blob. No learnable parameters are added.
The motivation is that a detector fed absolute coordinates can key on where an object is rather
than what it looks like, which does not transfer across datasets; the covariance is translation
invariant, so the first layer sees local shape instead.

Worth being precise about what this does NOT do: absolute position still reaches the network
through the sparse tensor's voxel indices and through the position-indexed BEV head. What changes
is the input *feature value*, not the architecture's access to position.

Two deviations from upstream, both deliberate:

1. Upstream's `get_output_feature_dim()` returns `C + C**2` unconditionally, but with
   `RELATIVE_DISTANCE: True` the position block is sliced to `[:, 0:3]`. For C == 3 (every `da_*`
   config here, see experiments_md/20260919_04) the two agree at 12; for C == 4 upstream declares
   20 while emitting 19, and the 3D backbone is then built with the wrong `input_channels`. Here
   the declared dim is computed from the same branch that builds the tensor, and
   `tests/test_gblobs_vfe.py` asserts they agree.
2. Upstream infers the zero-padding mask as `voxel_features.sum(2) != 0`, which also discards a
   genuine point whose features happen to sum to zero. `voxel_num_points` says exactly how many
   rows are real, so it is used instead.

`COVARIANCE_SCALE` is an escape hatch, not part of the method: the position block is in metres
(|.| < one voxel, ~0.05 m) while the covariance block is in metres squared (~1e-4), and
`VoxelResBackBone8x`'s `conv_input` applies no input normalisation. It defaults to 1.0, which is a
bit-exact no-op. See experiments_md/20260922_04 for the plan this implements.
"""
import torch

from .vfe_template import VFETemplate
from ....utils.common_utils import get_voxel_centers


class GBlobsVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        # Upstream defaults, kept so that an unflagged GBlobsVFE matches the reference
        # implementation. The configs here set RELATIVE_DISTANCE: True, as upstream's own do.
        self.cov_only = self.model_cfg.get('COVARIANCE_ONLY', False)
        self.rel_d = self.model_cfg.get('RELATIVE_DISTANCE', False)
        self.cov_scale = float(self.model_cfg.get('COVARIANCE_SCALE', 1.0))

    def get_num_position_features(self):
        """Width of the position block: 0 if dropped, 3 if made relative to the voxel centre
        (only xyz can be), otherwise the full mean over all point features."""
        if self.cov_only:
            return 0
        return 3 if self.rel_d else self.num_point_features

    def get_output_feature_dim(self):
        return self.get_num_position_features() + self.num_point_features ** 2

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: (num_voxels)
                voxel_coords: (num_voxels, 4) - [batch_idx, z, y, x]

        Returns:
            batch_dict:
                voxel_features: (num_voxels, get_output_feature_dim())
        """
        voxel_features = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']

        # (num_voxels, max_points, 1), 1.0 for a real point and 0.0 for zero padding.
        max_points = voxel_features.shape[1]
        mask = torch.arange(max_points, device=voxel_features.device).view(1, -1)
        mask = (mask < voxel_num_points.view(-1, 1)).unsqueeze(-1).type_as(voxel_features)

        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = (voxel_features * mask).sum(dim=1) / normalizer

        centered = mask * (voxel_features - points_mean.unsqueeze(1))
        cov = torch.einsum('nka,nkb->nab', centered, centered)
        # Unbiased, and defined as zero for a single-point voxel rather than dividing by zero.
        cov = cov / torch.clamp_min(normalizer - 1.0, min=1.0).unsqueeze(dim=-1)
        cov = cov.reshape(-1, self.num_point_features ** 2)
        if self.cov_scale != 1.0:
            cov = cov * self.cov_scale

        if self.cov_only:
            features = cov
        else:
            pos_features = points_mean
            if self.rel_d:
                voxel_centers = get_voxel_centers(
                    batch_dict['voxel_coords'][:, 1:], 1, self.voxel_size, self.point_cloud_range
                )
                pos_features = points_mean[:, 0:3] - voxel_centers
            features = torch.cat([pos_features, cov], dim=1)

        batch_dict['voxel_features'] = features.contiguous()
        return batch_dict
