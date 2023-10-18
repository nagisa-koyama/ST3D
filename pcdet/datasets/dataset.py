import torch
import copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch.utils.data as torch_data
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
from ..utils import common_utils, box_utils, self_training_utils
from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from ..utils.ontology_mapping import get_ontology_mapping

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None,
                 model_ontology=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = copy.deepcopy(class_names)
        self.dataset_ontology = dataset_cfg.get('ONTOLOGY', None)
        logger.info("Dataset ontology: %s", self.dataset_ontology)
        logger.info("Model ontology: %s", model_ontology)

        self.dataset_class_names = copy.deepcopy(class_names)
        self.map_ontology_dataset_to_model = None
        self.map_ontology_model_to_dataset = None
        if model_ontology is not None and self.dataset_ontology is not None and model_ontology != self.dataset_ontology:
            self.map_ontology_dataset_to_model = get_ontology_mapping(self.dataset_ontology, model_ontology)
            self.map_ontology_model_to_dataset = get_ontology_mapping(model_ontology, self.dataset_ontology)
            self.dataset_class_names = [self.map_ontology_model_to_dataset[label] for label in class_names]
        elif ":" in class_names[0]:
            # Multi-head setup. Handles only associated labels.
            self.dataset_class_names = copy.deepcopy([])
            for cls in class_names:
                assert(cls.count(":") == 1)
                ontology, label = cls.split(":")
                if ontology == self.dataset_ontology:
                    self.dataset_class_names.append(cls)
                assert(self.dataset_class_names[-1].count(":") == 1)

        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.dataset_class_names, logger=self.logger,
            map_ontology_dataset_to_model=self.map_ontology_dataset_to_model, dataset_ontology=self.dataset_ontology
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )
        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False



    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            # print("name in generate_single_sample_dict:", pred_dict['name'])
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos


    @staticmethod
    def __vis__(points, gt_boxes, ref_boxes=None, scores=None, labels=None, use_fakelidar=False):
        import visual_utils.visualize_utils as vis
        import mayavi.mlab as mlab
        gt_boxes = copy.deepcopy(gt_boxes)
        if use_fakelidar:
            gt_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(gt_boxes)

        if ref_boxes is not None:
            ref_boxes = copy.deepcopy(ref_boxes)
            if use_fakelidar:
                ref_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(ref_boxes)

        vis.draw_scenes(points, gt_boxes, ref_boxes=ref_boxes, ref_scores=scores, ref_labels=labels)
        # mlab.show(stop=True)

    @staticmethod
    def __vis_fake__(points, gt_boxes, ref_boxes=None, scores=None, use_fakelidar=True):
        import visual_utils.visualize_utils as vis
        import mayavi.mlab as mlab
        gt_boxes = copy.deepcopy(gt_boxes)
        if use_fakelidar:
            gt_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(gt_boxes)

        if ref_boxes is not None:
            ref_boxes = copy.deepcopy(ref_boxes)
            if use_fakelidar:
                ref_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(ref_boxes)

        vis.draw_scenes(points, gt_boxes, ref_boxes=ref_boxes, ref_scores=scores)
        # mlab.show(stop=True)

    @staticmethod
    def extract_fov_data(points, fov_degree, heading_angle):
        """
        Args:
            points: (N, 3 + C)
            fov_degree: [0~180]
            heading_angle: [0~360] in lidar coords, 0 is the x-axis, increase clockwise
        Returns:
        """
        half_fov_degree = fov_degree / 180 * np.pi / 2
        heading_angle = -heading_angle / 180 * np.pi
        points_new = common_utils.rotate_points_along_z(
            points.copy()[np.newaxis, :, :], np.array([heading_angle])
        )[0]
        angle = np.arctan2(points_new[:, 1], points_new[:, 0])
        fov_mask = ((np.abs(angle) < half_fov_degree) & (points_new[:, 0] > 0))
        points = points_new[fov_mask]
        return points

    @staticmethod
    def extract_fov_gt(gt_boxes, fov_degree, heading_angle):
        """
        Args:
            anno_dict:
            fov_degree: [0~180]
            heading_angle: [0~360] in lidar coords, 0 is the x-axis, increase clockwise
        Returns:
        """
        half_fov_degree = fov_degree / 180 * np.pi / 2
        heading_angle = -heading_angle / 180 * np.pi
        gt_boxes_lidar = copy.deepcopy(gt_boxes)
        gt_boxes_lidar = common_utils.rotate_points_along_z(
            gt_boxes_lidar[np.newaxis, :, :], np.array([heading_angle])
        )[0]
        gt_boxes_lidar[:, 6] += heading_angle
        gt_angle = np.arctan2(gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0])
        fov_gt_mask = ((np.abs(gt_angle) < half_fov_degree) & (gt_boxes_lidar[:, 0] > 0))
        return fov_gt_mask

    def fill_pseudo_labels(self, input_dict):
        gt_boxes = self_training_utils.load_ps_label(input_dict['frame_id'])
        gt_scores = gt_boxes[:, 8]
        gt_classes = gt_boxes[:, 7]
        gt_boxes = gt_boxes[:, :7]

        # only suitable for only one classes, generating gt_names for prepare data
        gt_names = np.array(self.class_names)[np.abs(gt_classes.astype(np.int32)) - 1]

        input_dict['gt_boxes'] = gt_boxes
        input_dict['gt_names'] = gt_names
        input_dict['gt_classes'] = gt_classes
        input_dict['gt_scores'] = gt_scores
        input_dict['pos_ps_bbox'] = np.zeros((len(self.class_names)), dtype=np.float32)
        input_dict['ign_ps_bbox'] = np.zeros((len(self.class_names)), dtype=np.float32)
        for i in range(len(self.class_names)):
            num_total_boxes = (np.abs(gt_classes) == (i+1)).sum()
            num_ps_bbox = (gt_classes == (i+1)).sum()
            input_dict['pos_ps_bbox'][i] = num_ps_bbox
            input_dict['ign_ps_bbox'][i] = num_total_boxes - num_ps_bbox

        input_dict.pop('num_points_in_gt', None)

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        # print("data_dict[gt_names] in the beginning of prepare_data", data_dict['gt_names'])

        # ontology remapping
        if self.map_ontology_dataset_to_model is not None:
            updated_gt_names = []
            for index in range(data_dict['gt_names'].size):
                # Note: previously updated name is trancated due to initially allocated smaller memory size.
                # Resolved by newly creating numpy.array instead of updating existing element.
                updated_gt_names.append(self.map_ontology_dataset_to_model[data_dict['gt_names'][index]])
            assert len(updated_gt_names) == len(data_dict['gt_names'])
            data_dict['gt_names'] = np.array(updated_gt_names)
            # print("data_dict[gt_names] in prepare_data after ontology remap", data_dict['gt_names'])

        # Multi-head handling
        if ":" in self.dataset_class_names[0]:
            updated_gt_names = []
            for index in range(data_dict['gt_names'].size):
                # Note: previously updated name is trancated due to initially allocated smaller memory size.
                # Resolved by newly creating numpy.array instead of updating existing element.
                if ":" in data_dict['gt_names'][index]:
                    updated_gt_names.append(data_dict['gt_names'][index])
                else:
                    updated_gt_names.append(self.dataset_ontology + ":" + data_dict['gt_names'][index])
                assert(updated_gt_names[-1].count(":") == 1)
            data_dict['gt_names'] = np.array(updated_gt_names)
            # print("data_dict[gt_names] in prepare_data after multi-head label update", data_dict['gt_names'])

        if self.training:
            # filter gt_boxes without points
            num_points_in_gt = data_dict.get('num_points_in_gt', None)
            if num_points_in_gt is None:
                num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(data_dict['points'][:, :3]),
                    torch.from_numpy(data_dict['gt_boxes'][:, :7])).numpy().sum(axis=1)
            mask = (num_points_in_gt >= self.dataset_cfg.get('MIN_POINTS_OF_GT', 1))
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]
            if 'gt_classes' in data_dict:
                data_dict['gt_classes'] = data_dict['gt_classes'][mask]
                data_dict['gt_scores'] = data_dict['gt_scores'][mask]

            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            # for pseudo label has ignore labels.
            if 'gt_classes' not in data_dict:
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            else:
                gt_classes = data_dict['gt_classes'][selected]
                data_dict['gt_scores'] = data_dict['gt_scores'][selected]
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        # if self.training and len(data_dict['gt_boxes']) == 0:
        #     new_index = np.random.randint(self.__len__())
        #     return self.__getitem__(new_index)
        # print("selected gt_names at prepare_data:", data_dict['gt_names'])
        # print("gt_classes at prepare_data:", gt_classes)
        data_dict.pop('gt_names', None)
        data_dict.pop('gt_classes', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_scores']:
                    max_gt = max([len(x) for x in val])
                    batch_scores = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_scores[k, :val[k].__len__()] = val[k]
                    ret[key] = batch_scores
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

    def eval(self):
        self.training = False
        self.data_processor.eval()

    def train(self):
        self.training = True
        self.data_processor.train()
