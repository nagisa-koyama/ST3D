"""
    Dataset from Pandaset (Hesai)
"""

import copy
import pickle
import os
try:
    import pandas as pd
    import pandaset as ps
except:
    pass 
import numpy as np

from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

import torch


def pose_dict_to_numpy(pose):
    """
        Conert pandaset pose dict to a numpy vector in order to pass it through the network
    """
    pose_np = [pose["position"]["x"],
               pose["position"]["y"],
               pose["position"]["z"],
               pose["heading"]["w"],
               pose["heading"]["x"],
               pose["heading"]["y"],
               pose["heading"]["z"]]

    return pose_np


def pose_numpy_to_dict(pose):
    """
        Conert pandaset pose dict to a numpy vector in order to pass it through the network
    """
    pose_dict = {'position':
                    {'x': pose[0],
                     'y': pose[1],
                     'z': pose[2]},
                 'heading':
                    {'w': pose[3],
                     'x': pose[4],
                     'y': pose[5],
                     'z': pose[6]}}

    return pose_dict


class PandasetDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, model_ontology=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger,
            model_ontology=model_ontology
        )
        if root_path is None:
            root_path = self.dataset_cfg.DATA_PATH
        self.dataset = ps.DataSet(os.path.join(root_path, 'dataset'))
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.pandaset_infos = []
        self.include_pandaset_infos(self.mode)


    def include_pandaset_infos(self, mode):
        if self.logger is not None:
            self.logger.info('Loading PandaSet dataset')
        pandaset_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = os.path.join(self.root_path, info_path)
            if not os.path.exists(info_path):
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                pandaset_infos.extend(infos)

        self.pandaset_infos.extend(pandaset_infos)

        if self.logger is not None:
            self.logger.info('Total samples for PandaSet dataset ({}): {}'.format(self.mode, len(pandaset_infos)))


    def set_split(self, split):
        self.sequences = self.dataset_cfg.SEQUENCES[split]
        self.split = split


    def __len__(self):
        return len(self.pandaset_infos)


    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate (x pointing forward, z pointing upwards) and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        info = self.pandaset_infos[index]
        seq_idx = info['sequence']

        pose = self._get_pose(info)
        points = self._get_lidar_points(info, pose)
        boxes, labels, zrot_world_to_ego = self._get_annotations(info, pose)
        pose_np = pose_dict_to_numpy(pose)

        input_dict = {'points': points,
                      'gt_boxes': boxes,
                      'gt_names': labels,
                      'sequence': int(seq_idx),
                      'frame_id': info['frame_idx'],
                      'zrot_world_to_ego': zrot_world_to_ego,
                      'pose': pose_dict_to_numpy(pose)
                     }
        # seq_idx is converted to int because strings can't be passed to
        # the gpu in pytorch
        # zrot_world_to_ego is propagated in order to be able to transform the
        # predicted yaws back to world coordinates

        # delete default label
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            input_dict['gt_boxes'] = None

        # load saved pseudo label for unlabel data
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


    def _get_pose(self, info):
        seq_idx = info['sequence']
        # get pose for world to ego frame transformation
        if self.dataset[seq_idx].lidar.poses is None:
            self.dataset[seq_idx].lidar._load_poses()

        pose = self.dataset[seq_idx].lidar.poses[info['frame_idx']]

        return pose


    def _get_lidar_points(self, info, pose):
        """
        Get lidar in the unified normative coordinate system for a given frame
        The intensity is normalized to fit [0-1] range (pandaset intensity is in [0-255] range)
        """
        # get lidar points
        lidar_frame = pd.read_pickle(info['lidar_path'])
        # get points for the required lidar(s) only
        device = self.dataset_cfg.get('LIDAR_DEVICE', 0)
        if device != -1:
            lidar_frame = lidar_frame[lidar_frame.d == device]
        world_points = lidar_frame.to_numpy()
        # There seems to be issues with the automatic deletion of pandas datasets sometimes
        del lidar_frame

        points_loc = world_points[:, :3]
        points_int = world_points[:, 3]

        # nromalize intensity
        points_int = points_int / 255

        ego_points = ps.geometry.lidar_points_to_ego(points_loc, pose)
        # Pandaset ego coordinates are:
        # - x pointing to the right
        # - y pointing to the front
        # - z pointing up
        # Normative coordinates are:
        # - x pointing foreward
        # - y pointings to the left
        # - z pointing to the top
        # So a transformation is required to the match the normative coordinates
        ego_points = ego_points[:, [1, 0, 2]] # switch x and y
        ego_points[:, 1] = - ego_points[:, 1] # revert y axis

        return np.append(ego_points, np.expand_dims(points_int, axis=1), axis=1).astype(np.float32)


    def _get_annotations(self,info, pose):
        """
        Get box informations in the unified normative coordinate system for a given frame
        """

        # get boxes
        cuboids = pd.read_pickle(info["cuboids_path"])
        device = self.dataset_cfg.get('LIDAR_DEVICE', 0)
        if device != -1:
            # keep cuboids that are seen by a given device
            cuboids = cuboids[cuboids["cuboids.sensor_id"] != 1 - device]

        xs = cuboids['position.x'].to_numpy()
        ys = cuboids['position.y'].to_numpy()
        zs = cuboids['position.z'].to_numpy()
        dxs = cuboids['dimensions.x'].to_numpy()
        dys = cuboids['dimensions.y'].to_numpy()
        dzs = cuboids['dimensions.z'].to_numpy()
        yaws = cuboids['yaw'].to_numpy()
        labels = cuboids['label'].to_numpy()

        del cuboids  # There seem to be issues with the automatic deletion of pandas datasets sometimes

        labels = np.array([self.dataset_cfg.TRAINING_CATEGORIES.get(lab, lab)
                           for lab in labels] )

        # Compute the center points coordinates in ego coordinates
        centers = np.vstack([xs, ys, zs]).T
        ego_centers = ps.geometry.lidar_points_to_ego(centers, pose)

        # Compute the yaw in ego coordinates
        # The following implementation supposes that the pitch of the car is
        # negligible compared to its yaw, in order to be able to express the
        # bbox coordinates in the ego coordinate system with an {axis aligned
        # box + yaw} only representation
        yaxis_points_from_pose = ps.geometry.lidar_points_to_ego(np.array([[0, 0, 0], [0, 1., 0]]), pose)
        yaxis_from_pose = yaxis_points_from_pose[1, :] - yaxis_points_from_pose[0, :]

        if yaxis_from_pose[-1] >= 10**-1:
            if self.logger is not None:
                self.logger.warning("The car's pitch is supposed to be negligible " +
                                    "sin(pitch) is >= 10**-1 ({})".format(yaxis_from_pose[-1]))

        # rotation angle in rads of the y axis around thz z axis
        zrot_world_to_ego = np.arctan2(-yaxis_from_pose[0], yaxis_from_pose[1])
        ego_yaws = yaws + zrot_world_to_ego

        # Pandaset ego coordinates are:
        # - x pointing to the right
        # - y pointing to the front
        # - z pointing up
        # Normative coordinates are:
        # - x pointing foreward
        # - y pointings to the left
        # - z pointing to the top
        # So a transformation is required to the match the normative coordinates
        ego_xs = ego_centers[:, 1]
        ego_ys = -ego_centers[:, 0]
        ego_zs = ego_centers[:, 2]
        ego_dxs = dys
        ego_dys = dxs  # stays >= 0
        ego_dzs = dzs
        ego_boxes = np.vstack([ego_xs, ego_ys, ego_zs, ego_dxs, ego_dys, ego_dzs, ego_yaws]).T
        return ego_boxes.astype(np.float32), labels, zrot_world_to_ego


    def get_infos(self):
        """
        Generate the dataset infos dict for each sample of the dataset.
        For each sample, this dict contains:
            - the sequence index
            - the frame index
            - the path to the lidar data
            - the path to the bounding box annotations
        """
        infos = []
        for seq in self.sequences:
            s = self.dataset[seq]
            s.load_lidar()
            if len(s.lidar.data) > 100:
                raise ValueError("The implementation for this dataset assumes that each sequence is " +
                                 "no longer than 100 frames. The current sequence has {}".format(len(s.lidar.data)))
            info = [{'sequence': seq,
                     'frame_idx': ii,
                     'lidar_path': os.path.join(self.root_path, 'dataset', seq, 'lidar', ("{:02d}.pkl.gz".format(ii))),
                     'cuboids_path': os.path.join(self.root_path, 'dataset', seq,
                                                  'annotations', 'cuboids', ("{:02d}.pkl.gz".format(ii)))
                    } for ii in range(len(s.lidar.data))]
            infos.extend(info)
            del self.dataset._sequences[seq]

        return infos


    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        database_save_path = os.path.join(self.root_path,
                'gt_database' if split == 'train' else 'gt_database_{}'.format(split))
        db_info_save_path = os.path.join(self.root_path,
                'pandaset_dbinfos_{}.pkl'.format(split))

        os.makedirs(database_save_path, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['frame_idx']
            pose = self._get_pose(info)
            points = self._get_lidar_points(info, pose)
            gt_boxes, names, _ = self._get_annotations(info, pose)

            num_obj = gt_boxes.shape[0]

            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                tmp_name = names[i].replace("/", "").replace(" ", "")
                filename = '%s_%s_%d.bin' % (sample_idx, tmp_name, i)
                filepath = os.path.join(database_save_path, filename)
                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'wb') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = os.path.relpath(filepath, self.root_path)  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': -1}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval
        from ..kitti import kitti_utils

        map_name_to_kitti = {
            'Car': 'Car',
            'Pedestrian': 'Pedestrian',
            'Pedestrian with Object': 'Pedestrian',
            'Pickup Truck': 'Truck',
            'Medium-sized Truck': 'Truck',
            'Semi-truck': 'Truck',
            'Motorized Scooter': 'Cyclist',
            'Bicycle': 'Cyclist',
        }
        self.logger.info("len(eval_det_annos): {}".format(len(eval_det_annos)))
        # self.logger.info("len(eval_det_annos[name]): {}".format(len(eval_det_annos['name'])))
        kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
        # self.logger.info("kitti eval_det_annos.shape: {}".format(eval_det_annos.shape))
        kitti_utils.transform_annotations_to_kitti_format(
            eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
            info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
        )
        kitti_class_names = [map_name_to_kitti[x] for x in class_names]

        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        return ap_result_str, ap_dict


    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = []
            for info in self.pandaset_infos:
                seq_idx = info['sequence']
                pose = self._get_pose(info)
                boxes, labels, zrot_world_to_ego = self._get_annotations(info, pose)
                gt_dict = {'gt_boxes': boxes,
                           'gt_names': labels,
                           'sequence': int(seq_idx),
                           'frame_id': info['frame_idx'],
                           'zrot_world_to_ego': zrot_world_to_ego,
                           'pose': pose_dict_to_numpy(pose)
                           }
                eval_gt_annos.append(gt_dict)
            return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        else:
            raise NotImplementedError


def create_pandaset_infos(dataset_cfg, class_names, data_path, save_path):
    """
    Create dataset_infos files in order not to have it in a preprocessed pickle
    file with the info for each sample
    See PandasetDataset.get_infos for further details.
    """
    dataset = PandasetDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    for split in ["train", "val", "test"]:
        print("---------------- Start to generate {} data infos ---------------".format(split))
        dataset.set_split(split)
        infos = dataset.get_infos()
        file_path = os.path.join(save_path, 'pandaset_infos_{}.pkl'.format(split))
        with open(file_path, 'wb') as f:
            pickle.dump(infos, f)
        print("Pandaset info {} file is saved to {}".format(split, file_path))

    print('------------Start create groundtruth database for data augmentation-----------')
    dataset = PandasetDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    dataset.set_split("train")
    dataset.create_groundtruth_database(
        os.path.join(save_path, 'pandaset_infos_train.pkl'),
        split="train"
    )
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_pandaset_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_pandaset_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'pandaset',
            save_path=ROOT_DIR / 'data' / 'pandaset'
        )




