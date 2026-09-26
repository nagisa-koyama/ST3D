from functools import partial
import numpy as np
from . import augmentor_utils, database_sampler
from ...utils import common_utils


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, dataset_class_names, logger=None, map_ontology_dataset_to_model=None, dataset_ontology=None):
        self.root_path = root_path
        self.class_names = dataset_class_names
        self.logger = logger
        self.augmentor_configs = augmentor_configs
        self.map_ontology_dataset_to_model = map_ontology_dataset_to_model
        self.dataset_ontology = dataset_ontology

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            dataset_class_names=self.class_names,
            logger=self.logger,
            map_ontology_dataset_to_model=self.map_ontology_dataset_to_model,
            dataset_ontology = self.dataset_ontology
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_object_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_rotation, config=config)

        gt_boxes, points = augmentor_utils.rotate_objects(
            data_dict['gt_boxes'],
            data_dict['points'],
            data_dict['gt_boxes_mask'],
            rotation_perturb=config['ROT_UNIFORM_NOISE'],
            prob=config['ROT_PROB'],
            num_try=50
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_object_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_scaling, config=config)
        points, gt_boxes = augmentor_utils.scale_pre_object(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            scale_perturb=config['SCALE_UNIFORM_NOISE'],
            gt_names=data_dict.get('gt_names', None)
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_sampling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_sampling, config=config)
        gt_boxes, points, gt_boxes_mask = augmentor_utils.global_sampling(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            sample_ratio_range=config['WORLD_SAMPLE_RATIO'],
            prob=config['PROB']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_boxes_mask'] = gt_boxes_mask
        data_dict['points'] = points
        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def normalize_object_size(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.normalize_object_size, config=config)
        points, gt_boxes = augmentor_utils.normalize_object_size(
            data_dict['gt_boxes'], data_dict['points'], data_dict['gt_boxes_mask'],
            config['SIZE_RES'], gt_names=data_dict.get('gt_names', None)
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict

    def re_prepare(self, augmentor_configs=None, intensity=None):
        self.data_augmentor_queue = []

        if augmentor_configs is None:
            augmentor_configs = self.augmentor_configs

        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            # scale data augmentation intensity
            if intensity is not None:
                cur_cfg = self.adjust_augment_intensity(cur_cfg, intensity)
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def adjust_augment_intensity(self, config, intensity):
        adjust_map = {
            'random_object_scaling': 'SCALE_UNIFORM_NOISE',
            'random_object_rotation': 'ROT_UNIFORM_NOISE',
            'random_world_rotation': 'WORLD_ROT_ANGLE',
            'random_world_scaling': 'WORLD_SCALE_RANGE',
        }

        def scale_interval(interval, flag):
            """Scale [lo, hi] about `flag`, EACH SIDE INDEPENDENTLY.

            The previous version required the interval to be symmetric about flag
            (`assert np.isclose(flag - lo, hi - flag)`) and rebuilt it from the upper half alone.
            Every per-class ROS interval derived from the data is deliberately ASYMMETRIC - Car is
            [0.85, 1.20], i.e. 0.15 below 1 and 0.20 above, because every source in this study is a
            larger-vehicle domain (experiments_md/20260922_03) - so that assert fires on the real
            configs and, when it does not, the rebuild would silently symmetrise the interval and
            discard the asymmetry the derivation exists to express.

            Scaling each side independently is identical for a symmetric interval, so no previously
            working config changes behaviour.
            """
            assert len(interval) == 2, interval
            lo, hi = interval
            return [flag - (flag - lo) * intensity, flag + (hi - flag) * intensity]

        def cal_new_intensity(config, flag):
            origin = config.get(adjust_map[config.NAME])
            # PER-CLASS intervals are a dict ({'Car': [...], 'Pedestrian': [...]}) since ROS and SN
            # became per-class (577a3f8). The old code indexed it as a list: with exactly two
            # classes `len(origin) == 2` passed by coincidence and the next line raised
            # `KeyError: 0` - eleven hours into job 25945, at the first UPDATE_AUG epoch.
            if isinstance(origin, dict):
                return {cls: scale_interval(vals, flag) for cls, vals in origin.items()}
            return scale_interval(origin, flag)

        if config.NAME not in adjust_map:
            return config
        
        # for data augmentations that init with 1
        if config.NAME in ["random_object_scaling", "random_world_scaling"]:
            new_intensity_list = cal_new_intensity(config, flag=1)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        elif config.NAME in ['random_object_rotation', 'random_world_rotation']:
            new_intensity_list = cal_new_intensity(config, flag=0)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        else:
            raise NotImplementedError
