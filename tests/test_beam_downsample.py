"""Beam downsampling and the BEV imitation loss - the LiDAR Distillation baseline.

The property that matters for the baseline to mean anything is that `beam_ratio` removes RINGS
rather than points: a method whose premise is "make a 64-beam source look like a 32-beam target"
is not implemented by discarding half the returns at random, because a random half keeps every
ring's coverage and only lowers density. So the tests below check ring structure explicitly, on a
synthetic cloud whose rings are known exactly, rather than only checking that the point count fell.
"""
import numpy as np
import pytest
import torch

from pcdet.utils import beam_downsample_utils as bd
from pcdet.utils import distill_utils


def make_rings(elevations, n_azimuth=180, radius=20.0):
    """A cloud with one perfect ring per elevation. Returns (points, ring index per point)."""
    pts, ring_id = [], []
    az = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    for i, el in enumerate(elevations):
        el_rad = np.radians(el)
        xy = radius * np.cos(el_rad)
        z = radius * np.sin(el_rad)
        pts.append(np.stack([xy * np.cos(az), xy * np.sin(az), np.full(n_azimuth, z),
                             np.zeros(n_azimuth)], axis=1))
        ring_id.append(np.full(n_azimuth, i))
    return np.concatenate(pts).astype(np.float32), np.concatenate(ring_id)


ELEVATIONS = np.linspace(-24.0, 2.0, 16)


class TestAngles:
    def test_elevation_and_azimuth_recovered(self):
        points, _ = make_rings([-10.0, 0.0, 5.0], n_azimuth=4)
        theta, phi, valid = bd.compute_angles(points)
        assert valid.all()
        assert np.allclose(sorted(set(np.round(theta, 6))), [-10.0, 0.0, 5.0], atol=1e-4)
        # azimuth covers [0, 360) and never returns 360 itself
        assert phi.min() >= 0.0 and phi.max() < 360.0

    def test_origin_points_are_marked_invalid_not_nan(self):
        """The reference divides by the horizontal range unguarded; a point at the sensor origin
        makes both angles NaN, and a NaN elevation silently joins whichever cluster it lands in."""
        points = np.zeros((3, 4), dtype=np.float32)
        points[1, 0] = 5.0
        theta, phi, valid = bd.compute_angles(points)
        assert list(valid) == [False, True, False]
        assert np.isfinite(theta).all() and np.isfinite(phi).all()


class TestRingLabelling:
    def test_synthetic_rings_are_recovered_exactly(self):
        points, ring_id = make_rings(ELEVATIONS)
        theta, _, valid = bd.compute_angles(points)
        centroids = bd.fit_beam_centroids(theta[valid], len(ELEVATIONS))
        label = bd.beam_label_from_centroids(theta, centroids, valid)
        # Centroids come back sorted, so the label IS the ring order.
        assert np.array_equal(label, ring_id)
        assert np.allclose(centroids, ELEVATIONS, atol=1e-3)

    def test_invalid_points_get_label_minus_one(self):
        points, _ = make_rings(ELEVATIONS, n_azimuth=8)
        points = np.concatenate([points, np.zeros((1, 4), dtype=np.float32)])
        theta, _, valid = bd.compute_angles(points)
        centroids = bd.fit_beam_centroids(theta[valid], len(ELEVATIONS))
        label = bd.beam_label_from_centroids(theta, centroids, valid)
        assert label[-1] == -1
        # and -1 is never a kept ring, so such points are dropped rather than joining ring 0
        mask = bd.generate_mask(np.zeros(len(points)), label, len(ELEVATIONS), beam_ratio=2)
        assert not mask[-1]

    def test_subsampled_fit_matches_full_fit(self):
        """Departure 2 in the module docstring: fitting on a subsample must give the same partition
        whenever rings are separated by more than the sampling noise."""
        points, ring_id = make_rings(ELEVATIONS, n_azimuth=400)
        theta, _, valid = bd.compute_angles(points)
        full = bd.fit_beam_centroids(theta[valid], len(ELEVATIONS), max_fit_points=10 ** 9)
        sub = bd.fit_beam_centroids(theta[valid], len(ELEVATIONS), max_fit_points=500)
        assert np.allclose(full, sub, atol=1e-3)
        assert np.array_equal(bd.beam_label_from_centroids(theta, sub, valid), ring_id)


class TestMasking:
    @pytest.mark.parametrize('beam_ratio,expected_rings', [(1, 16), (2, 8), (4, 4)])
    def test_beam_ratio_drops_whole_rings(self, beam_ratio, expected_rings):
        points, ring_id = make_rings(ELEVATIONS)
        _, phi, _ = bd.compute_angles(points)
        mask = bd.generate_mask(phi, ring_id, len(ELEVATIONS), beam_ratio=beam_ratio)
        kept = set(ring_id[mask])
        assert len(kept) == expected_rings
        assert kept == set(range(0, len(ELEVATIONS), beam_ratio))
        # every surviving ring is kept ENTIRELY when bin_ratio is 1
        for ring in kept:
            assert mask[ring_id == ring].all()

    def test_bin_ratio_thins_within_a_ring_by_azimuth(self):
        points, ring_id = make_rings(ELEVATIONS, n_azimuth=180)
        _, phi, _ = bd.compute_angles(points)
        mask = bd.generate_mask(phi, ring_id, len(ELEVATIONS), beam_ratio=1, bin_ratio=2)
        assert set(ring_id[mask]) == set(range(len(ELEVATIONS)))  # no ring removed
        for ring in range(len(ELEVATIONS)):
            in_ring = ring_id == ring
            assert mask[in_ring].sum() == 90
            # kept returns are evenly spaced in azimuth, which is what makes bin_ratio a
            # horizontal RESOLUTION reduction rather than an arbitrary thinning
            kept_phi = np.sort(phi[in_ring][mask[in_ring]])
            assert np.allclose(np.diff(kept_phi), np.diff(kept_phi)[0], atol=1e-3)

    def test_downsample_beams_is_identity_at_ratio_one(self):
        points, _ = make_rings(ELEVATIONS, n_azimuth=8)
        out, centroids = bd.downsample_beams(points, len(ELEVATIONS), 1, 1)
        assert out is points and centroids is None  # no fit paid for a no-op

    def test_cached_centroids_give_identical_output(self):
        points, _ = make_rings(ELEVATIONS, n_azimuth=64)
        first, centroids = bd.downsample_beams(points, len(ELEVATIONS), 2, 1)
        second, _ = bd.downsample_beams(points, len(ELEVATIONS), 2, 1, centroids=centroids)
        assert np.array_equal(first, second)

    def test_world_augmentation_preserves_the_ring_partition(self):
        """Why `_split_for_beam_distillation` may downsample AFTER augmentation, and why cached
        centroids stay valid: z rotation and uniform scaling leave every elevation unchanged."""
        points, ring_id = make_rings(ELEVATIONS, n_azimuth=64)
        rotated = points.copy()
        angle = 0.7
        c, s = np.cos(angle), np.sin(angle)
        rotated[:, 0] = points[:, 0] * c - points[:, 1] * s
        rotated[:, 1] = points[:, 0] * s + points[:, 1] * c
        rotated[:, :3] *= 1.05
        theta, _, valid = bd.compute_angles(rotated)
        centroids = bd.fit_beam_centroids(*(theta[valid], len(ELEVATIONS)))
        assert np.array_equal(bd.beam_label_from_centroids(theta, centroids, valid), ring_id)


class TestBevStride:
    def test_stride_is_measured_from_the_feature_map(self):
        features = torch.zeros(2, 64, 188, 188)
        assert distill_utils.bev_feature_stride(features, [1504, 1504, 40]) == 8

    def test_non_integer_stride_is_refused(self):
        """A backbone that does not divide the grid evenly makes every box's cell extent wrong by a
        sub-cell amount that grows with distance from the origin."""
        features = torch.zeros(1, 64, 100, 100)
        with pytest.raises(AssertionError, match='does not divide'):
            distill_utils.bev_feature_stride(features, [1504, 1504, 40])


class TestImitationLoss:
    PC_RANGE = [-75.2, -75.2, -2, 75.2, 75.2, 4]
    VOXEL = [0.1, 0.1, 0.15]
    GRID = [1504, 1504, 40]

    def _features(self, value=0.0):
        return torch.full((1, 8, 188, 188), value)

    def _boxes(self, n_valid=1, n_pad=0):
        boxes = []
        for i in range(n_valid):
            boxes.append([10.0 * (i + 1), 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 1.0])
        boxes += [[0.0] * 8] * n_pad  # collate pads gt_boxes with zeros; class id 0 marks padding
        return torch.tensor([boxes], dtype=torch.float32)

    def test_identical_features_give_zero_loss(self):
        student = {'spatial_features_2d': self._features(3.0), 'gt_boxes': self._boxes()}
        teacher = {'spatial_features_2d': self._features(3.0)}
        loss = distill_utils.bev_imitation_loss(
            student, teacher, 'gt', self.PC_RANGE, self.VOXEL, self.GRID)
        assert loss.item() == pytest.approx(0.0)

    def test_loss_is_positive_and_gradient_reaches_the_student_only(self):
        student_features = self._features(0.0).requires_grad_(True)
        teacher_features = self._features(1.0).requires_grad_(True)
        loss = distill_utils.bev_imitation_loss(
            {'spatial_features_2d': student_features, 'gt_boxes': self._boxes()},
            {'spatial_features_2d': teacher_features},
            'gt', self.PC_RANGE, self.VOXEL, self.GRID)
        assert loss.item() > 0
        loss.backward()
        assert student_features.grad is not None and student_features.grad.abs().sum() > 0
        # the teacher is detached inside the loss, so it can never be trained by it
        assert teacher_features.grad is None

    def test_padded_boxes_do_not_contribute(self):
        """Padding rows are all-zero, so an unguarded mask would put a footprint at the origin and
        normalise by a box count that counts them."""
        kwargs = dict(mode='gt', point_cloud_range=self.PC_RANGE, voxel_size=self.VOXEL,
                      grid_size=self.GRID, normalization='valid')   # 'reference' divides by the padded count on purpose
        teacher = {'spatial_features_2d': self._features(1.0)}
        one = distill_utils.bev_imitation_loss(
            {'spatial_features_2d': self._features(0.0), 'gt_boxes': self._boxes(1, 0)},
            teacher, **kwargs)
        padded = distill_utils.bev_imitation_loss(
            {'spatial_features_2d': self._features(0.0), 'gt_boxes': self._boxes(1, 9)},
            teacher, **kwargs)
        assert one.item() == pytest.approx(padded.item())

    def test_each_box_carries_equal_weight_regardless_of_size(self):
        """The per-box mask is normalised to sum 1, so a distant small object is worth as much as a
        near large one - otherwise the loss would be dominated by whichever boxes are biggest."""
        big = self._boxes(1)
        big[0, 0, 3:5] = torch.tensor([20.0, 20.0])
        kwargs = dict(mode='gt', point_cloud_range=self.PC_RANGE, voxel_size=self.VOXEL,
                      grid_size=self.GRID)
        teacher = {'spatial_features_2d': self._features(1.0)}
        small_loss = distill_utils.bev_imitation_loss(
            {'spatial_features_2d': self._features(0.0), 'gt_boxes': self._boxes(1)},
            teacher, **kwargs)
        big_loss = distill_utils.bev_imitation_loss(
            {'spatial_features_2d': self._features(0.0), 'gt_boxes': big}, teacher, **kwargs)
        assert small_loss.item() == pytest.approx(big_loss.item(), rel=1e-5)

    def test_no_valid_boxes_gives_a_zero_that_still_has_a_graph(self):
        """A frame with no labels must not produce a loss detached from the graph: under DDP that
        makes one rank's gradient buckets differ from another's."""
        student_features = self._features(0.0).requires_grad_(True)
        loss = distill_utils.bev_imitation_loss(
            {'spatial_features_2d': student_features, 'gt_boxes': self._boxes(0, 4)},
            {'spatial_features_2d': self._features(1.0)},
            'gt', self.PC_RANGE, self.VOXEL, self.GRID)
        assert loss.item() == pytest.approx(0.0)
        assert loss.requires_grad
        loss.backward()

    def test_all_mode_ignores_boxes(self):
        loss = distill_utils.bev_imitation_loss(
            {'spatial_features_2d': self._features(0.0)},
            {'spatial_features_2d': self._features(1.0)},
            'all', self.PC_RANGE, self.VOXEL, self.GRID)
        # 8 channels each differing by 1 -> per-cell L2 norm is sqrt(8)
        assert loss.item() == pytest.approx(np.sqrt(8.0), rel=1e-5)

    def test_roi_mode_is_refused_with_an_explanation(self):
        with pytest.raises(NotImplementedError, match='two-stage'):
            distill_utils.bev_imitation_loss(
                {'spatial_features_2d': self._features(0.0), 'gt_boxes': self._boxes()},
                {'spatial_features_2d': self._features(1.0)},
                'roi', self.PC_RANGE, self.VOXEL, self.GRID)

class TestPairedCollate:
    """`collate_batch` must fold a list of (student, teacher) samples into two batch dicts.

    Checked separately from the loss because this is where a pair silently becomes a single batch:
    the reference's branch keys off `type(batch_data[0]) == tuple`, and if it misses, every sample's
    student and teacher end up concatenated into ONE point cloud with no error anywhere.
    """

    def _sample(self, n_points, gt=2):
        return {
            'points': np.zeros((n_points, 4), dtype=np.float32),
            'gt_boxes': np.zeros((gt, 8), dtype=np.float32),
            'frame_id': 'f',
        }

    def test_pairs_collate_into_two_batches(self):
        from pcdet.datasets.dataset import DatasetTemplate
        batch = [(self._sample(10), self._sample(20)), (self._sample(11), self._sample(22))]
        out = DatasetTemplate.collate_batch(batch)
        assert isinstance(out, list) and len(out) == 2
        student, teacher = out
        assert student['batch_size'] == teacher['batch_size'] == 2
        # the two streams stay separate, and the teacher keeps MORE points (it is the high-beam one)
        assert student['points'].shape[0] == 21
        assert teacher['points'].shape[0] == 42

    def test_unpaired_batches_are_unchanged(self):
        """The branch must be inert for every other family in the repo."""
        from pcdet.datasets.dataset import DatasetTemplate
        out = DatasetTemplate.collate_batch([self._sample(10), self._sample(11)])
        assert isinstance(out, dict)
        assert out['points'].shape[0] == 21


class TestPairedSplit:
    """`_split_for_beam_distillation` on a stand-in object carrying only what the method reads."""

    class _FakeDataset:
        from pcdet.datasets.dataset import DatasetTemplate
        _split_for_beam_distillation = DatasetTemplate._split_for_beam_distillation

        def __init__(self, cfg):
            self.beam_distill_cfg = cfg
            self.beam_centroids = None
            self.processed = []

        class _Identity:
            def __init__(self, outer):
                self.outer = outer

            def forward(self, data_dict=None, **kwargs):
                d = data_dict if data_dict is not None else kwargs['data_dict']
                self.outer.processed.append(d)
                return d

        @property
        def point_feature_encoder(self):
            return self._Identity(self)

        @property
        def data_processor(self):
            return self._Identity(self)

    def _cfg(self, **kw):
        from easydict import EasyDict
        base = dict(NUM_BEAMS=len(ELEVATIONS), BEAM_RATIO=2, BIN_RATIO=1)
        base.update(kw)
        return EasyDict(base)

    def test_student_is_thinner_and_the_teacher_untouched(self):
        points, _ = make_rings(ELEVATIONS, n_azimuth=32)
        ds = self._FakeDataset(self._cfg())
        student, teacher = ds._split_for_beam_distillation(
            {'points': points, 'gt_boxes': np.zeros((3, 8), dtype=np.float32),
             'gt_names': np.array(['Car'] * 3)})
        assert len(teacher['points']) == len(points)
        assert len(student['points']) == len(points) // 2

    def test_streams_do_not_share_mutable_arrays(self):
        """The processor edits 'points' and 'gt_boxes' in place, so aliasing would make the first
        stream processed corrupt the second - and the corruption would look like a model bug."""
        points, _ = make_rings(ELEVATIONS, n_azimuth=32)
        boxes = np.zeros((3, 8), dtype=np.float32)
        ds = self._FakeDataset(self._cfg())
        data = {'points': points, 'gt_boxes': boxes, 'gt_names': np.array(['Car'] * 3)}
        student, teacher = ds._split_for_beam_distillation(data)
        for key in ('points', 'gt_boxes'):
            assert student[key] is not teacher[key]
            assert teacher[key] is not data[key]

    def test_identical_gt_boxes_reach_both_streams(self):
        """The imitation mask is built from the student's boxes and applied to both feature maps."""
        points, _ = make_rings(ELEVATIONS, n_azimuth=32)
        boxes = np.arange(24, dtype=np.float32).reshape(3, 8)
        ds = self._FakeDataset(self._cfg())
        student, teacher = ds._split_for_beam_distillation(
            {'points': points, 'gt_boxes': boxes, 'gt_names': np.array(['Car'] * 3)})
        assert np.array_equal(student['gt_boxes'], teacher['gt_boxes'])

    def test_a_no_op_ratio_is_refused(self):
        """Ratios of 1 would hand the student the teacher's own cloud, making the imitation loss
        identically zero - a run that looks like it is distilling and is not."""
        points, _ = make_rings(ELEVATIONS, n_azimuth=8)
        ds = self._FakeDataset(self._cfg(BEAM_RATIO=1, BIN_RATIO=1))
        with pytest.raises(AssertionError, match='identically zero'):
            ds._split_for_beam_distillation({'points': points})

    def test_centroids_are_fitted_once_and_cached(self):
        points, _ = make_rings(ELEVATIONS, n_azimuth=32)
        ds = self._FakeDataset(self._cfg())
        ds._split_for_beam_distillation({'points': points})
        assert ds.beam_centroids is not None
        first = ds.beam_centroids
        ds._split_for_beam_distillation({'points': points})
        assert ds.beam_centroids is first


class TestImitationLossExtra:
    PC_RANGE = TestImitationLoss.PC_RANGE
    VOXEL = TestImitationLoss.VOXEL
    GRID = TestImitationLoss.GRID

    def _features(self, value=0.0):
        return torch.full((1, 8, 188, 188), value)

    def _boxes(self):
        return torch.tensor([[[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 1.0]]], dtype=torch.float32)

    def test_mismatched_feature_maps_are_refused(self):
        """The commonest way to misconfigure this: student and teacher on different
        POINT_CLOUD_RANGE or VOXEL_SIZE, which makes a per-cell loss compare unrelated places."""
        with pytest.raises(AssertionError, match='does not match the teacher'):
            distill_utils.bev_imitation_loss(
                {'spatial_features_2d': torch.zeros(1, 8, 188, 188), 'gt_boxes': self._boxes()},
                {'spatial_features_2d': torch.zeros(1, 8, 94, 94)},
                'gt', self.PC_RANGE, self.VOXEL, self.GRID)


class TestReferenceNormalization:
    """'reference' must equal the released cal_mimic_loss (train_mimic_utils.py, mode 'gt') to float
    precision, including its padded-batch denominator; 'valid' differs from it by exactly B*R/n_valid."""
    PC_RANGE = [-75.2, -75.2, -2, 75.2, 75.2, 4]
    VOXEL = [0.1, 0.1, 0.15]
    GRID = [1504, 1504, 40]

    @staticmethod
    def _reference_cal_mimic_loss(teacher_features, student_features, rois, min_x, min_y, cell_x, cell_y):
        # literal CPU transcription of the reference, mode 'gt'
        batch_size, height, width = teacher_features.size(0), teacher_features.size(2), teacher_features.size(3)
        roi_size = rois.size(1)
        x1 = (rois[:, :, 0] - rois[:, :, 3] / 2 - min_x) / cell_x
        x2 = (rois[:, :, 0] + rois[:, :, 3] / 2 - min_x) / cell_x
        y1 = (rois[:, :, 1] - rois[:, :, 4] / 2 - min_y) / cell_y
        y2 = (rois[:, :, 1] + rois[:, :, 4] / 2 - min_y) / cell_y
        grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width), indexing='ij')
        grid_y = grid_y[None, None].repeat(batch_size, roi_size, 1, 1)
        grid_x = grid_x[None, None].repeat(batch_size, roi_size, 1, 1)
        mask_y = (grid_y >= y1[:, :, None, None]) * (grid_y <= y2[:, :, None, None])
        mask_x = (grid_x >= x1[:, :, None, None]) * (grid_x <= x2[:, :, None, None])
        mask = (mask_y * mask_x).float()
        mask[rois[:, :, -1] == 0] = 0
        weight = mask.sum(-1).sum(-1)
        weight[weight == 0] = 1
        mask = mask / weight[:, :, None, None]
        mimic_loss = torch.norm(teacher_features - student_features, p=2, dim=1)
        mask = mask.sum(1)
        mimic_loss = (mimic_loss * mask).sum() / batch_size / roi_size
        mimic_loss = (mimic_loss * mask).sum() / (rois[:, :, -1] > 0).sum()   # the reference's no-op line
        return mimic_loss

    def _batch(self):
        g = torch.Generator().manual_seed(0)
        student = torch.randn(2, 8, 188, 188, generator=g)
        teacher = torch.randn(2, 8, 188, 188, generator=g)
        # frame 0: three boxes, frame 1: one box + two padding rows -> R = 3, n_valid = 4
        boxes = torch.tensor([
            [[10.0, 5.0, 0, 4.0, 2.0, 1.5, 0.3, 1.0], [-20.0, 30.0, 0, 6.0, 2.5, 2.0, 0.0, 1.0], [40.0, -8.0, 0, 1.0, 1.0, 1.7, 0.0, 2.0]],
            [[-5.0, -5.0, 0, 4.5, 1.9, 1.6, 1.0, 1.0], [0.0] * 8, [0.0] * 8]], dtype=torch.float32)
        return student, teacher, boxes

    def test_reference_matches_the_released_function(self):
        student, teacher, boxes = self._batch()
        ours = distill_utils.bev_imitation_loss(
            {'spatial_features_2d': student, 'gt_boxes': boxes}, {'spatial_features_2d': teacher},
            'gt', self.PC_RANGE, self.VOXEL, self.GRID, normalization='reference')
        stride = distill_utils.bev_feature_stride(student, self.GRID)
        ref = self._reference_cal_mimic_loss(teacher, student, boxes, self.PC_RANGE[0], self.PC_RANGE[1],
                                             self.VOXEL[0] * stride, self.VOXEL[1] * stride)
        assert ours.item() == pytest.approx(ref.item(), rel=1e-5)

    def test_valid_differs_by_the_padding_factor(self):
        student, teacher, boxes = self._batch()
        kw = dict(mode='gt', point_cloud_range=self.PC_RANGE, voxel_size=self.VOXEL, grid_size=self.GRID)
        ref = distill_utils.bev_imitation_loss({'spatial_features_2d': student, 'gt_boxes': boxes}, {'spatial_features_2d': teacher}, normalization='reference', **kw)
        val = distill_utils.bev_imitation_loss({'spatial_features_2d': student, 'gt_boxes': boxes}, {'spatial_features_2d': teacher}, normalization='valid', **kw)
        assert val.item() == pytest.approx(ref.item() * (2 * 3) / 4, rel=1e-5)
