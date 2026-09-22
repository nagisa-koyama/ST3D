from functools import partial

import numpy as np
import math

from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features, hist_dist_src = None, hist_dist_tgt = None):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None
        self.hist_dist_src = hist_dist_src
        self.hist_dist_tgt = hist_dist_tgt
        # Foreground-aware calibration is opt-in; absent these, the correction uses the single
        # whole-cloud pair above and behaves exactly as before.
        self.hist_fg_src = self.hist_bg_src = None
        self.hist_fg_tgt = self.hist_bg_tgt = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def set_hist_dist(self, hist_dist_src, hist_dist_tgt):
        """Install measured histograms after construction (see datasets/point_calibration.py).

        Must be called before the dataloader is first iterated: workers fork a copy of the dataset
        and never see later mutations.
        """
        self.hist_dist_src = hist_dist_src
        self.hist_dist_tgt = hist_dist_tgt

    def set_foreground_hist(self, fg_src, bg_src, fg_tgt, bg_tgt):
        """Install the inside-box / outside-box histogram pairs (see point_calibration.py).

        Same forking constraint as `set_hist_dist`: must precede the first iteration of any loader
        over this dataset. Unlike `set_hist_dist` this one additionally needs the target's boxes to
        have existed when it was measured, which under self-training means after the first
        pseudo-label generation pass.
        """
        self.hist_fg_src, self.hist_bg_src = fg_src, bg_src
        self.hist_fg_tgt, self.hist_bg_tgt = fg_tgt, bg_tgt

    def mask_boxes_outside_length(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_boxes_outside_length, config=config)

        min_mask = data_dict['gt_boxes'][:, 3] >= config['LENGTH_RANGE'][0]
        max_mask = data_dict['gt_boxes'][:, 3] <= config['LENGTH_RANGE'][1]
        mask = min_mask & max_mask

        data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]

        return data_dict

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            # TODO(nagisa): revisit this if the performance is not good enough.
            # pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            # pts_near_flag = pts_depth < 40.0
            # far_idxs_choice = np.where(pts_near_flag == 0)[0]
            # near_idxs = np.where(pts_near_flag == 1)[0]
            # choice = []
            # if num_points > len(far_idxs_choice):
            #     near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            #     choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
            #         if len(far_idxs_choice) > 0 else near_idxs_choice
            # else:
            choice = np.arange(0, len(points), dtype=np.int32)
            choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def sample_points_hist_based(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points_hist_based, config=config)

        if self.hist_dist_src is None or self.hist_dist_tgt is None:
            return data_dict

        points = data_dict['points']
        points_dist = np.linalg.norm(points[:, 0:2], axis=1)
        # TODO: load MAX_DIST from config
        MAX_DIST = 75.0
        bin_num = len(self.hist_dist_src)
        indexes = np.floor(np.clip(points_dist, 0, MAX_DIST - 0.0001) / MAX_DIST * bin_num).astype(np.int32)

        if self.hist_fg_src is None:
            sample_rate = self.per_bin_sample_rate(config)[indexes]
        else:
            # Foreground-aware: correct the inside-box and outside-box channels separately.
            # A single per-bin rate cannot change a bin's foreground SHARE - it scales numerator
            # and denominator alike - so matching the global profile leaves source objects
            # under-sampled by exactly sigma_src/sigma_tgt. Two channels give the correction a
            # degree of freedom it structurally lacked.
            fg = self.points_in_any_box(points, data_dict.get('gt_boxes', None))
            sample_rate = np.where(fg,
                                   self.per_bin_sample_rate(config, 'fg')[indexes],
                                   self.per_bin_sample_rate(config, 'bg')[indexes])
        points_mask = np.random.rand(len(points)) < sample_rate
        data_dict['points'] = points[points_mask]
        return data_dict

    @staticmethod
    def points_in_any_box(points, boxes):
        """Boolean mask: is each point inside at least one of `boxes`?

        Degenerate boxes are dropped first. A zero- or negative-extent box reaching the C++
        geometry kernel is the same failure mode as the still-open `gt_sampling` segfault in
        `database_sampler.py`, and costs nothing to rule out here.
        """
        if boxes is None or len(boxes) == 0:
            return np.zeros(len(points), dtype=bool)
        boxes = np.asarray(boxes, dtype=np.float32)[:, :7]
        boxes = boxes[(boxes[:, 3:6] > 1e-3).all(axis=1)]
        if len(boxes) == 0:
            return np.zeros(len(points), dtype=bool)
        from ...ops.roiaware_pool3d import roiaware_pool3d_utils
        inside = roiaware_pool3d_utils.points_in_boxes_cpu(
            np.ascontiguousarray(points[:, 0:3], dtype=np.float32), boxes)
        return inside.any(axis=0) > 0

    def per_bin_sample_rate(self, config=None, channel='all'):
        """target/source density ratio per radial bin, with under-populated bins left alone.

        `channel` selects which pair of histograms to compare: 'all' is the whole cloud, 'fg' only
        the points inside GT boxes and 'bg' only those outside. The fg/bg pair is installed by
        `set_foreground_hist` and is absent unless foreground-aware calibration was requested.

        A bin the source barely reaches gives a ratio estimated from a handful of points. Worse,
        `hist_src == 0` makes the ratio inf or nan, and `rand() < nan` is False - so a zero source
        bin silently DROPS EVERY POINT that lands in it. That cannot happen when both histograms
        are measured on the same frames, but the shipped hist_dist_*.npy files were measured under
        different preprocessing, which is exactly when it can.

        Bins whose source count falls below MIN_HIST_BIN_FRACTION of the mean source bin are left
        uncorrected (rate 1) rather than corrected from noise. The fraction is scale-free, so it
        behaves the same for per-frame histograms and for the shipped raw counts.
        """
        pair = {'all': (self.hist_dist_src, self.hist_dist_tgt),
                'fg': (self.hist_fg_src, self.hist_fg_tgt),
                'bg': (self.hist_bg_src, self.hist_bg_tgt)}[channel]
        assert pair[0] is not None, 'no %s histogram installed' % channel
        src = np.asarray(pair[0], dtype=np.float64)
        tgt = np.asarray(pair[1], dtype=np.float64)
        frac = 0.01 if config is None else config.get('MIN_HIST_BIN_FRACTION', 0.01)
        floor = frac * src.mean()
        trusted = src > max(floor, 0.0)
        rate = np.ones_like(src)
        np.divide(tgt, src, out=rate, where=trusted)
        rate[~np.isfinite(rate)] = 1.0
        return rate


    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict

    def eval(self):
        self.training = False
        self.mode = 'test'

    def train(self):
        self.training = True
        self.mode = 'train'
