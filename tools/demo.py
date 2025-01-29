import _init_path
import os
import torch
import time
import argparse
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from test import get_all_configs

import wandb
# try:
#     import open3d
#     from visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
# except:
import mayavi.mlab as mlab
OPEN3D_FLAG = False


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='specify the pretrained model')
    parser.add_argument('--dataset_index', type=int, default=0,
                        help='specify the index of DATA_CONFIGS')
    parser.add_argument('--sample_index', type=int, default=1,
                        help='specify the index of samples in the dataset')
    parser.add_argument('--out_dir', type=str,
                        default='/storage', help='specify the output directory')
    parser.add_argument('--out_filename', type=str,
                        default='scene.png', help='specify the output filename')
    parser.add_argument('--is_train', action='store_true', help='analyze train set, otherwise use test set')
    parser.add_argument('--run_name', type=str, default=None, help='run name for wandb')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    wandb.init(config=vars(cfg), project="st3d", name=args.run_name)

    # Dataset configs
    eval_configs = get_all_configs(cfg)
    eval_config_rep = list(eval_configs.values())[0]

    eval_datasets = list()
    for eval_config in eval_configs.values():
        eval_set, eval_loader, eval_sampler = build_dataloader(
            dataset_cfg=eval_config,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=False, workers=1,
            logger=logger, training=args.is_train,
            model_ontology=cfg.get('ONTOLOGY', None),
            no_shuffle=True
        )
        eval_dataset = dict(dataset_class=eval_set, loader=eval_loader, sampler=eval_sampler)
        eval_datasets.append(eval_dataset)
        logger.info(f'Total number of samples: \t{len(eval_loader)}')

    model = None
    if args.ckpt is not None:
        eval_dataset_rep = eval_datasets[0]
        model = build_network(model_cfg=cfg.MODEL, num_class=len(
            cfg.CLASS_NAMES), dataset=eval_dataset_rep['dataset_class'])
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        logger.info("Model loaded")

    for eval_dataset in eval_datasets:
        print("eval_dataset onotology:", eval_dataset['loader'].dataset.dataset_ontology)
        for idx, data_dict in enumerate(eval_dataset['loader']):
            start_time = time.time()
            if idx != args.sample_index:
                continue
            logger.info(f'Visualized sample index: \t{idx}')
            load_data_to_gpu(data_dict)
            load_data_to_gpu_duration = time.time()
            print("load_data_to_gpu_duration:", load_data_to_gpu_duration - start_time)

            annos = None
            if model is not None:
                with torch.no_grad():
                    pred_dicts, _ = model.forward(data_dict)
                forward_duration = time.time()
                print("forward_duration:", forward_duration - load_data_to_gpu_duration)
                annos = eval_dataset['loader'].dataset.generate_prediction_dicts(
                    data_dict, pred_dicts, cfg.CLASS_NAMES,
                )
                generate_prediction_dicts_duration = time.time()
                print("generate_prediction_dicts_duration:", generate_prediction_dicts_duration - forward_duration)
            else:
                generate_prediction_dicts_duration = load_data_to_gpu_duration
            mlab.options.offscreen = True
            first_elem_index = 0
            first_elem_mask = data_dict['points'][:, 0] == first_elem_index
            eval_dataset['loader'].dataset.__vis__(
                points=data_dict['points'][first_elem_mask, 1:], gt_boxes=data_dict['gt_boxes'][first_elem_index],
                ref_boxes=annos[first_elem_index]['boxes_lidar'] if annos else None,
                ref_scores=annos[first_elem_index]['score'] if annos else None,
                labels=annos[first_elem_index]['pred_labels'] if annos else None
            )
            vis_duration = time.time()
            print("__vis__duration:", vis_duration - generate_prediction_dicts_duration)
            filename = os.path.join(args.out_dir,
                                    f'{args.out_filename[:-4]}_{eval_dataset["loader"].dataset.dataset_ontology}_{idx}.png')
            if not OPEN3D_FLAG:
                mlab.savefig(filename=filename)
            else:
                img = vis.capture_screen_float_buffer(True)
                opencd.io.write_image(filename, img)

            wandb.log({f'{eval_dataset["loader"].dataset.dataset_ontology}/scene': wandb.Image(filename)})

            break
    wandb.finish()
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
