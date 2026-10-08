from functools import partial

import numpy as np
import math

from ...utils import beam_downsample_utils, box_utils, common_utils, ptsn_utils

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
        # 1-based class indices (column 7 of gt_boxes) whose boxes form the foreground channel;
        # None = every class, the original pooled behaviour.
        self.hist_fg_class_ids = None
        # Radial extent the histogram bins span. Together with the bin count it fixes the
        # resolution (75 m / 50 bins = 1.5 m). Held here rather than read from config at
        # correction time so that the value binning the points is by construction the one the
        # histogram was measured with - the two cannot drift apart.
        self.hist_max_dist = 75.0
        # DALI's PTSN input scale (see pcdet/utils/ptsn_utils.py). 1.0 is the identity and is the
        # only value that can be reached without an explicit set_ptsn_scale() call, so no config
        # that does not ask for PTSN can be affected by it.
        self.ptsn_scale = 1.0
        # Ring elevations for `downsample_beams`, fitted on the first frame this processor sees and
        # then reused. They are a sensor calibration, not a per-frame quantity, so re-fitting per
        # frame would only pay for KMeans repeatedly to get the same answer. Cached per processor
        # instance, which means per DataLoader worker - each worker pays the fit once.
        self.beam_centroids = None

        names = [c.NAME for c in processor_configs]
        assert not ('sample_points_learned' in names and 'sample_points_hist_based' in names), \
            'sample_points_learned replaces sample_points_hist_based; a config may not list both'
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def set_hist_dist(self, hist_dist_src, hist_dist_tgt, max_dist=None):
        """Install measured histograms after construction (see datasets/point_calibration.py).

        Must be called before the dataloader is first iterated: workers fork a copy of the dataset
        and never see later mutations.
        """
        self.hist_dist_src = hist_dist_src
        self.hist_dist_tgt = hist_dist_tgt
        if max_dist is not None:
            self.hist_max_dist = float(max_dist)

    def set_foreground_hist(self, fg_src, bg_src, fg_tgt, bg_tgt, class_ids=None):
        """Install the inside-box / outside-box histogram pairs (see point_calibration.py).

        Same forking constraint as `set_hist_dist`: must precede the first iteration of any loader
        over this dataset. Unlike `set_hist_dist` this one additionally needs the target's boxes to
        have existed when it was measured, which under self-training means after the first
        pseudo-label generation pass.
        """
        self.hist_fg_src, self.hist_bg_src = fg_src, bg_src
        self.hist_fg_tgt, self.hist_bg_tgt = fg_tgt, bg_tgt
        # Must be the SAME class set the histograms were measured with: a point in a box of an
        # excluded class was counted as background there, so it must be sampled as background here.
        self.hist_fg_class_ids = None if class_ids is None else sorted(int(c) for c in class_ids)

    def set_ptsn_scale(self, scale):
        """Install DALI's PTSN input scale (pcdet/utils/ptsn_utils.py).

        Same forking constraint as `set_hist_dist`: must precede the first iteration of any
        loader over this dataset, because a worker forks a copy and never sees a later mutation.
        In practice this is set once, from config, before the pseudo-label generation loader is
        first iterated, and never changed again - PTSN's scale is a constant for a run.
        """
        scale = float(scale)
        assert scale > 0, 'PTSN scale must be positive, got %r' % scale
        self.ptsn_scale = scale

    def scale_points_ptsn(self, data_dict):
        """Scale the cloud's geometry by `ptsn_scale`, for inference passes only.

        Deliberately NOT a configurable entry in DATA_PROCESSOR, and deliberately applied ahead
        of the queue rather than inside it, for two reasons:

          * It must run before `mask_points_and_boxes_outside_range` and before voxelisation, so
            that the range crop and the voxel grid are the ones the network actually sees. A
            consequence worth stating: at s > 1 the far field is cropped harder than it would be
            at s = 1, because POINT_CLOUD_RANGE is fixed while the scene grows.
          * The gate is `self.training`, not a config key. Under self-training one dataset object
            serves both a training loader and a generation loader; their workers fork in train and
            eval mode respectively, so this gate routes the scaling to the generation pass alone
            without any mid-run mutation. Scaling the training points too would be wrong: the
            pseudo-labels handed to the student have already been divided by s.

        GT boxes are left untouched. PTSN is an inference-time transform whose inverse is applied
        to predictions (`self_training_utils.save_pseudo_label_batch`), and no scored evaluation
        should ever run with a scale installed - which is why the only wiring is on the target
        dataset in the self-training loop, never on the eval dataset built for AP.
        """
        if self.training or self.ptsn_scale == 1.0:
            return data_dict
        data_dict['points'] = ptsn_utils.scale_points(data_dict['points'], self.ptsn_scale)
        return data_dict

    def mask_boxes_outside_length(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_boxes_outside_length, config=config)

        min_mask = data_dict['gt_boxes'][:, 3] >= config['LENGTH_RANGE'][0]
        max_mask = data_dict['gt_boxes'][:, 3] <= config['LENGTH_RANGE'][1]
        mask = min_mask & max_mask

        data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]

        return data_dict

    def mask_points_by_spec(self, data_dict=None, config=None):
        """Drop points outside a sensor SPEC window about SENSOR_ORIGIN: elevation in
        [ELEVATION_MIN_DEG, ELEVATION_MAX_DEG] and horizontal radius >= MIN_RADIUS_M.

        Analysis step (experiments_md 20261008_05): applies a target sensor's published vertical
        field of view or ego-removal radius to another sensor's cloud at EVALUATION time, to ask
        whether a model never exposed to that part of the sensor's view is hurt by it. Every key
        is from a published spec, so it is label-free. NOT gated on self.training: it runs in
        whichever mode the config lists it, and the configs that use it are eval-only twins.
        SENSOR_ORIGIN is the sensor position in the frame the processor sees, i.e. SHIFT_COOR
        already applied (nuScenes: [0, 0, 1.75]; Waymo TOP: [1.43, 0, 2.184] in the vehicle frame).
        Boxes are untouched - a cut GT box stays a GT box, as the scoring protocol requires.
        """
        if data_dict is None:
            return partial(self.mask_points_by_spec, config=config)
        points = data_dict['points']
        origin = np.asarray(config.SENSOR_ORIGIN, dtype=np.float64)
        d = points[:, :3].astype(np.float64) - origin
        r = np.hypot(d[:, 0], d[:, 1])
        el = np.degrees(np.arctan2(d[:, 2], np.maximum(r, 1e-6)))
        keep = np.ones(len(points), dtype=bool)
        if config.get('ELEVATION_MIN_DEG', None) is not None:
            keep &= el >= float(config.ELEVATION_MIN_DEG)
        if config.get('ELEVATION_MAX_DEG', None) is not None:
            keep &= el <= float(config.ELEVATION_MAX_DEG)
        if config.get('MIN_RADIUS_M', None) is not None:
            keep &= r >= float(config.MIN_RADIUS_M)
        data_dict['points'] = points[keep]
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

    def __getstate__(self):
        """Drop the spconv voxel generator when pickled.

        Under DDP the loader workers are SPAWNED, so the dataset (and this processor) is pickled
        into each worker. The generator wraps a C++ object (`Point2VoxelCPU`) that cannot be
        pickled; the original design created it lazily in the worker for exactly that reason. The
        on-the-fly density calibration broke the assumption: it walks source frames through
        `__getitem__` in the MAIN process before training, creating the generator there, and the
        first 2-GPU launch of a global-correction row (job 26326) died at `iter(train_loader)` with
        `TypeError: cannot pickle ... Point2VoxelCPU`. It is re-created lazily on first use, and its
        parameters are all in the bound config, so nothing is lost.
        """
        state = self.__dict__.copy()
        state['voxel_generator'] = None
        return state

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

        points = data_dict['points']
        # RATIO: keep a fixed FRACTION of the frame's points instead of a fixed count (a per-mode dict
        # like NUM_POINTS; absent or 1.0 = unchanged). Added 2026-10-06 for the GBlobs Fig. 4-style
        # test-time density control on S3 (thin the flash target to the spin source's in-cone
        # density, experiments_md/20261005_01): NUM_POINTS may then be -1.
        ratio = config.get('RATIO', None)
        if ratio is not None and ratio.get(self.mode, 1.0) < 1.0:
            num_points = int(round(len(points) * ratio[self.mode]))
        else:
            num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

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

    def downsample_beams(self, data_dict=None, config=None):
        """Drop whole laser rings, to make a high-beam source look like a low-beam target.

        The LiDAR Distillation baseline (Wei et al., ECCV 2022). Config keys:
            NUM_BEAMS:   rings the SOURCE sensor has (KITTI 64, Lyft 40 or 64, nuScenes 32).
            BEAM_RATIO:  keep every BEAM_RATIO-th ring. 64 -> 32 beams is 2; 64 -> 16 is 4.
            BIN_RATIO:   keep every BIN_RATIO-th return within a kept ring, i.e. the paper's `*`
                         variants, which also halve the HORIZONTAL resolution. Default 1.
            MAX_FIT_POINTS: elevations the ring fit uses (default 20000).

        Runs on the fly rather than reading a precomputed `modes/<beam>/` directory as the original
        does. Three reasons, in order of weight: the original selects the student's beam count by
        `ln -s data/waymo/modes/<tag> data/waymo/waymo_processed_data` from inside its training
        script, which mutates a data directory this cluster shares between two repos and every
        concurrent job; a cache would duplicate hundreds of GB across five datasets and two ratios
        each; and nothing in this repo's loaders reads points from a path a config can redirect
        per-run anyway. The cost is a KMeans fit once per worker (see `beam_centroids`) plus one
        searchsorted per frame - measure it with tools/analysis/profile_throughput.py before
        launching, since accumulation already showed how easily this family becomes loader-bound.

        Deliberately NOT gated on self.training: the point of the baseline is that the student sees
        low-beam data, and a student evaluated on its source would have to see it at eval too. What
        keeps the TARGET untouched is that the target has its own DATA_CONFIG, which does not carry
        this stage.
        """
        if data_dict is None:
            # Refused at BUILD time, before any frame is read. A processor stage runs after
            # SHIFT_COOR and after world augmentation, and ring recovery from there is no better
            # than chance (purity 0.09-0.13, keep-mask agreement 0.50, 2026-10-01): elevation must
            # be measured about the sensor, whose position this stage cannot know. Use the
            # dataset-level BEAM_DROP / BEAM_DISTILL blocks, which label rings before augmentation.
            raise NotImplementedError(
                'downsample_beams as a DATA_PROCESSOR stage measures ring elevation about the '
                'SHIFT_COOR-shifted, augmented origin and does not recover rings. Use BEAM_DROP or '
                'BEAM_DISTILL in the dataset config instead (see beam_downsample_utils.ring_labels).')

        beam_ratio = config.get('BEAM_RATIO', 1)
        bin_ratio = config.get('BIN_RATIO', 1)
        if beam_ratio == 1 and bin_ratio == 1:
            return data_dict

        points, self.beam_centroids = beam_downsample_utils.downsample_beams(
            data_dict['points'],
            num_beams=config.NUM_BEAMS,
            beam_ratio=beam_ratio,
            bin_ratio=bin_ratio,
            centroids=self.beam_centroids,
            max_fit_points=config.get('MAX_FIT_POINTS', 20000),
        )
        data_dict['points'] = points
        return data_dict

    def sample_points_learned(self, data_dict=None, config=None):
        """Keep each point with the learned sampler's probability (point_sampler.py).

        WEIGHTS is produced by tools/analysis/train_point_sampler.py for one source-target pair and
        loaded HERE, at construction, so the sampler is on the dataset before any DataLoader worker
        is forked or spawned (the stale-worker lesson, experiments_md/20260921_02). Training mode
        only, like the histogram rule; mutually exclusive with sample_points_hist_based.
        """
        if data_dict is None:
            from .point_sampler import LearnedPointSampler
            self.learned_sampler = LearnedPointSampler.load(config.WEIGHTS)
            return partial(self.sample_points_learned, config=config)
        if not self.training:
            return data_dict
        data_dict['points'] = self.learned_sampler.sample(data_dict['points'])
        return data_dict

    def ablate_intensity(self, data_dict=None, config=None):
        """DIAGNOSTIC: does a model trained with intensity use it? (experiments_md/20261003_03 s7)

        MODE `shuffle` permutes the intensity column among the frame's points - the frame's intensity
        distribution is unchanged, only which point carries which value is destroyed - so a drop in AP
        measures the per-point information the model reads, free of any distribution shift. MODE
        `constant` sets every point to VALUE (e.g. the dataset's median), which removes the information
        AND the spread. Deterministic per frame (seeded by the point count). Never used in training.
        """
        if data_dict is None:
            if config.MODE not in ('shuffle', 'constant'):
                raise ValueError('ablate_intensity MODE must be shuffle or constant, got %r' % config.MODE)
            return partial(self.ablate_intensity, config=config)
        points = data_dict['points']
        if len(points) == 0:
            return data_dict
        idx = config.get('INTENSITY_INDEX', 3)
        if config.MODE == 'shuffle':
            points[:, idx] = points[np.random.RandomState(len(points)).permutation(len(points)), idx]
        else:
            points[:, idx] = np.float32(config.VALUE)
        data_dict['points'] = points
        return data_dict

    def map_intensity_to_reference(self, data_dict=None, config=None, tables=None):
        """Test-time intensity calibration: map this dataset's intensity onto a reference sensor's.

        Per range ring, a point's intensity x becomes the reference value at the same percentile,
        F_ref^-1(F_this(x)) - so a model trained with intensity on the reference sensor sees target
        intensity in the units it learned. The tables are measured once from UNLABELLED point clouds of
        each dataset's train split (tools/analysis/intensity_testtime_tables.py), so this is label-free.
        Config keys: TABLES (npz path: edges, from_q, to_q per ring, from_step), INTENSITY_INDEX (column
        of intensity AFTER the point feature encoder, default 3). Integer-valued intensities (from_step
        > 0) are dequantised over one step with a per-frame deterministic draw before mapping, or the
        mapped values would form a comb (experiments_md/20261002_02 section 1).
        """
        if data_dict is None:
            t = np.load(config.TABLES)
            tables = {k: np.asarray(t[k], dtype=np.float64) for k in ('edges', 'from_q', 'to_q')}
            tables['from_step'] = float(t['from_step'])
            return partial(self.map_intensity_to_reference, config=config, tables=tables)

        points = data_dict['points']
        if len(points) == 0:
            return data_dict
        idx = config.get('INTENSITY_INDEX', 3)
        edges, from_q, to_q = tables['edges'], tables['from_q'], tables['to_q']
        levels = np.linspace(0.0, 1.0, from_q.shape[1])
        ring = np.clip(np.searchsorted(edges, np.linalg.norm(points[:, 0:2], axis=1), side='right') - 1,
                       0, len(edges) - 2)
        v = points[:, idx].astype(np.float64)
        if tables['from_step'] > 0:
            rng = np.random.RandomState(len(points))
            v = np.clip(v + rng.uniform(-tables['from_step'] / 2, tables['from_step'] / 2, len(v)), 0.0, 1.0)
        out = np.empty_like(v)
        for r in np.unique(ring):
            m = ring == r
            xs = np.maximum.accumulate(from_q[r]) + np.arange(len(levels)) * 1e-12
            out[m] = np.interp(np.interp(v[m], xs, levels), levels, to_q[r])
        points[:, idx] = out.astype(points.dtype)
        data_dict['points'] = points
        return data_dict

    def sample_points_by_range_rate(self, data_dict=None, config=None):
        """Keep each point with a FIXED per-planar-range-bin probability read from RATE_FILE (npz: `edges`, `rate`).

        Analysis tool (2026-10-07, experiments_md 20261007_03 test A'): thins one accumulation variant to another's
        radial point count so two evals differ in arrangement only. Applied in the modes listed in MODES (default
        ['test']). Rates above 1 are clipped (a keep probability cannot add points)."""
        if data_dict is None:
            d = np.load(config.RATE_FILE)
            self._range_rate = (d['edges'].astype(np.float64), np.clip(d['rate'].astype(np.float64), 0.0, 1.0))
            return partial(self.sample_points_by_range_rate, config=config)
        if self.mode not in config.get('MODES', ['test']):
            return data_dict
        edges, rate = self._range_rate
        pts = data_dict['points']
        r = np.hypot(pts[:, 0], pts[:, 1])
        k = np.clip(np.searchsorted(edges, r, side='right') - 1, 0, len(rate) - 1)
        keep = np.random.rand(len(pts)) < rate[k]
        data_dict['points'] = pts[keep]
        return data_dict

    def sample_points_hist_based(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points_hist_based, config=config)

        if self.hist_dist_src is None or self.hist_dist_tgt is None:
            return data_dict

        points = data_dict['points']
        points_dist = np.linalg.norm(points[:, 0:2], axis=1)
        max_dist = self.hist_max_dist
        bin_num = len(self.hist_dist_src)
        indexes = np.floor(np.clip(points_dist, 0, max_dist - 0.0001) / max_dist * bin_num).astype(np.int32)

        if self.hist_fg_src is None:
            sample_rate = self.per_bin_sample_rate(config)[indexes]
        else:
            # Foreground-aware: correct the inside-box and outside-box channels separately.
            # A single per-bin rate cannot change a bin's foreground SHARE - it scales numerator
            # and denominator alike - so matching the global profile leaves source objects
            # under-sampled by exactly sigma_src/sigma_tgt. Two channels give the correction a
            # degree of freedom it structurally lacked.
            fg = self.points_in_any_box(points, self.foreground_boxes(
                data_dict.get('gt_boxes', None), getattr(self, 'hist_fg_class_ids', None)))
            sample_rate = np.where(fg,
                                   self.per_bin_sample_rate(config, 'fg')[indexes],
                                   self.per_bin_sample_rate(config, 'bg')[indexes])
        points_mask = np.random.rand(len(points)) < sample_rate
        data_dict['points'] = points[points_mask]
        return data_dict

    @staticmethod
    def box_occupancy(points, boxes):
        """(point-in-any-box mask, points per box, the boxes those counts refer to).

        Returns the surviving boxes as well as the counts because degenerate boxes are dropped
        first, which renumbers them - a caller that histograms box positions must use THESE boxes
        or its per-box denominator will not match its per-box numerator. A zero- or
        negative-extent box reaching the C++ geometry kernel is the same failure mode as the
        still-open `gt_sampling` segfault in `database_sampler.py`, and costs nothing to rule out.
        """
        empty = (np.zeros(len(points), dtype=bool), np.zeros(0, dtype=np.int64),
                 np.zeros((0, 7), dtype=np.float32))
        if boxes is None or len(boxes) == 0:
            return empty
        boxes = np.asarray(boxes, dtype=np.float32)[:, :7]
        boxes = boxes[(boxes[:, 3:6] > 1e-3).all(axis=1)]
        if len(boxes) == 0:
            return empty
        from ...ops.roiaware_pool3d import roiaware_pool3d_utils
        inside = roiaware_pool3d_utils.points_in_boxes_cpu(
            np.ascontiguousarray(points[:, 0:3], dtype=np.float32), boxes)
        return inside.any(axis=0) > 0, inside.sum(axis=1), boxes

    @staticmethod
    def foreground_boxes(boxes, class_ids):
        """Keep only boxes whose class index (column 7, sign ignored) is in `class_ids`.

        `class_ids=None` keeps every box. The foreground-aware correction pools classes by default,
        and 20260926_06 section 2 measured what that costs: a third of nuScenes boxes are
        pedestrians at ~1/7 of a car's points, while Lyft's source is 93% cars, so pooling thins
        cars 20-25% harder than a Car-only comparison justifies. Restricting the channel to
        `HIST_DIST_FOREGROUND_CLASSES` leaves the other classes' points on the background rate.
        """
        if class_ids is None or boxes is None or len(boxes) == 0:
            return boxes
        boxes = np.asarray(boxes)
        if boxes.shape[1] <= 7:
            raise ValueError('HIST_DIST_FOREGROUND_CLASSES needs the class column of gt_boxes '
                             '(appended in prepare_data), but these boxes have %d columns'
                             % boxes.shape[1])
        return boxes[np.isin(np.abs(boxes[:, 7]).astype(np.int64), list(class_ids))]

    @staticmethod
    def points_in_any_box(points, boxes):
        """Boolean mask: is each point inside at least one of `boxes`?"""
        return DataProcessor.box_occupancy(points, boxes)[0]

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

        The TARGET side is guarded the same way, which matters most for the foreground channel.
        There the target histogram is built from pseudo-labels, so a bin the teacher happened to
        find nothing in gives tgt == 0 and a rate of exactly 0 - which would drop EVERY source
        foreground point in that bin, the precise opposite of what this correction is for. An
        under-populated target bin is missing evidence, not evidence of absence, so it is left
        uncorrected too.
        """
        pair = {'all': (self.hist_dist_src, self.hist_dist_tgt),
                'fg': (self.hist_fg_src, self.hist_fg_tgt),
                'bg': (self.hist_bg_src, self.hist_bg_tgt)}[channel]
        assert pair[0] is not None, 'no %s histogram installed' % channel
        src = np.asarray(pair[0], dtype=np.float64)
        tgt = np.asarray(pair[1], dtype=np.float64)
        frac = 0.01 if config is None else config.get('MIN_HIST_BIN_FRACTION', 0.01)
        if config is not None and config.get('UNIFORM_RATE', False):
            # Control for the radial SHAPE of the correction: one keep-probability for the whole
            # cloud, set so the expected total matches the target's points per frame. Same
            # histograms, same calibration pass, same frames - only the per-bin structure is gone.
            assert channel == 'all', 'UNIFORM_RATE is a control for the global correction only'
            return np.full_like(src, tgt.sum() / src.sum())
        trusted = (src > max(frac * src.mean(), 0.0)) & (tgt > max(frac * tgt.mean(), 0.0))
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

        data_dict = self.scale_points_ptsn(data_dict)

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict

    def eval(self):
        self.training = False
        self.mode = 'test'

    def train(self):
        self.training = True
        self.mode = 'train'
