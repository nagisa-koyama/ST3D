import _init_path
import os
import torch
import time
import argparse
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
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
    parser.add_argument('--sample_index', type=int, default=0,
                        help='specify the index of samples in the dataset')
    parser.add_argument('--out_dir', type=str,
                        default='/storage', help='specify the output directory')
    parser.add_argument('--out_filename', type=str,
                        default='scene.png', help='specify the output filename')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    # Dataset priority
    if cfg.get('DATA_CONFIG_TAR', None):
        dataset_cfg = cfg.DATA_CONFIG_TAR
    elif cfg.get('DATA_CONFIGS', None):
        dataset_cfg = list((cfg.DATA_CONFIGS).values())[args.dataset_index]
    elif cfg.get('DATA_CONFIG', None):
        dataset_cfg = cfg.DATA_CONFIG
    else:
        assert False, "Eigher DATA_CONFIG, DATA_CONFIGS or DATA_CONFIG_TAR should be defined"

    dataset, loader, _ = build_dataloader(
        dataset_cfg=dataset_cfg,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=1,
        logger=logger,
        training=False,
        total_epochs=1,
        model_ontology=cfg.ONTOLOGY)

    logger.info(f'Total number of samples: \t{len(loader)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    logger.info("Model loaded")
    with torch.no_grad():
        for idx, data_dict in enumerate(loader):
            start_time = time.time()
            if idx != args.sample_index:
                continue
            logger.info(f'Visualized sample index: \t{idx}')
            load_data_to_gpu(data_dict)
            load_data_to_gpu_duration = time.time()
            print("load_data_to_gpu_duration:", load_data_to_gpu_duration - start_time)
            pred_dicts, _ = model.forward(data_dict)
            forward_duration = time.time()
            print("forward_duration:", forward_duration - load_data_to_gpu_duration)
            annos = dataset.generate_prediction_dicts(
                data_dict, pred_dicts, cfg.CLASS_NAMES,
            )
            generate_prediction_dicts_duration = time.time()
            print("generate_prediction_dicts_duration:", generate_prediction_dicts_duration - forward_duration)
            mlab.options.offscreen = True
            first_elem_index = 0
            first_elem_mask = data_dict['points'][:, 0] == first_elem_index
            dataset.__vis__(
                points=data_dict['points'][first_elem_mask, 1:], gt_boxes=data_dict['gt_boxes'][first_elem_index],
                ref_boxes=annos[first_elem_index]['boxes_lidar'],
                scores=annos[first_elem_index]['score'],
                labels=annos[first_elem_index]['pred_labels']
            )
            vis_duration = time.time()
            print("__vis__duration:", vis_duration - generate_prediction_dicts_duration)
            filename = os.path.join(args.out_dir, args.out_filename)
            if not OPEN3D_FLAG:
                mlab.savefig(filename=filename)
            else:
                img = vis.capture_screen_float_buffer(True)
                opencd.io.write_image(filename, img)
            break

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
