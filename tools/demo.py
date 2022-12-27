
import _init_path
import os
import pickle
import torch
import argparse
import glob
import numpy as np
from pathlib import Path
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V
import mayavi.mlab as mlab


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.infos = []

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        waymo_infos = []

        num_skipped_infos = 0
        sequence_name = os.path.basename(os.path.dirname(self.sample_file_list[index]))
        info_path = os.path.join(os.path.dirname(self.sample_file_list[index]), ('%s.pkl' % sequence_name))
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            waymo_infos.extend(infos)
        self.infos.extend(waymo_infos[:])
        info = self.infos[index]
        input_dict = {
            'points': points,
            'frame_id': index,
            'gt_boxes': info['annos']['gt_boxes_lidar'],
            'gt_names': info['annos']['name'],
            'num_points_of_each_lidar': info['num_points_of_each_lidar']
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--out_path', type=str, default='/storage', help='specify the output directory')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            selected_lidar_head_index = None
            if selected_lidar_head_index:
                selected_lidar_begin = int(data_dict['num_points_of_each_lidar'][0, selected_lidar_head_index].item())
                selected_lidar_end = int(data_dict['num_points_of_each_lidar'][0, selected_lidar_head_index + 1].item())
            else:
                selected_lidar_begin = selected_lidar_end = -1
            V.draw_scenes(
                points=data_dict['points'][selected_lidar_begin:selected_lidar_end, 1:], gt_boxes=data_dict['gt_boxes'][0],
                ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.savefig(filename=os.path.join(args.out_path, 'scenes.png'))
            #mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()