import _init_path
import os
import torch
import time
import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from test import get_eval_configs
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
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for visualization')
    parser.add_argument('--out_dir', type=str,
                        default='/storage', help='specify the output directory')
    parser.add_argument('--out_filename', type=str,
                        default='point_hist.png', help='specify the output filename')
    parser.add_argument('--run_name', type=str, default=None, help='run name for wandb')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Run Dataset Analysis-------------------------')

    wandb.init(config=vars(cfg), project="st3d", name=args.run_name)

    # Dataset configs
    eval_configs = get_eval_configs(cfg)
    eval_config_rep = list(eval_configs.values())[0]

    eval_datasets = list()
    for eval_config in eval_configs.values():
        eval_set, eval_loader, eval_sampler = build_dataloader(
            dataset_cfg=eval_config,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=False, workers=1,
            logger=logger, training=False,
            model_ontology=cfg.get('ONTOLOGY', None)
        )
        eval_dataset = dict(dataset_class=eval_set, loader=eval_loader, sampler=eval_sampler)
        eval_datasets.append(eval_dataset)
        logger.info(f'Total number of samples: \t{len(eval_loader)}')

    features = None
    labels = []
    feature_extraction_start = time.time()
    for eval_dataset in eval_datasets:
        print("dataset onotology:", eval_dataset['loader'].dataset.dataset_ontology)
        hist_x = None
        bins_x = None
        hist_y = None
        bins_y = None
        hist_z = None
        bins_z = None
        hist_intensity = None
        bins_intensity = None
        hist_x_car = None
        hist_y_car = None
        hist_z_car = None
        hist_length_car = None
        hist_width_car = None
        hist_height_car = None
        X_INDEX = 1
        Y_INDEX = 2
        Z_INDEX = 3
        INTENSITY_INDEX = 4
        CAR_X_INDEX = 0
        CAR_Y_INDEX = 1
        CAR_Z_INDEX = 2
        CAR_LENGTH_INDEX = 3
        CAR_WIDTH_INDEX = 4
        CAR_HEIGHT_INDEX = 5
        BINS = 100
        RANGE_XY = (-150, 150)
        RANGE_Z = (-10, 10)
        RANGE_INTENSITY = (-0.5, 2.5)
        RANGE_CAR_SIZE = (0, 10)
        progress_bar = tqdm.tqdm(total=len(eval_dataset['loader']), leave=True, desc='eval', dynamic_ncols=True)
        target_class_list = ["waymo:Vehicle", "pandaset:Car", "lyft:car", "nuscenes:car", "kitti:Car"]
        for idx, data_dict in enumerate(eval_dataset['loader']):
            # print('data_dict[gt_boxes]', data_dict['gt_boxes'])
            # print("data_dict['points'].shape:", data_dict['points'].shape)
            # print("data_dict['gt_boxes'].shape:", data_dict['gt_boxes'].shape)
            # print("data_dict['gt_names'].shape:", data_dict['gt_names'].shape)
            # print("data_dict['gt_names']:", data_dict['gt_names'])
            hist_x_curr, bins_x_curr = np.histogram(data_dict['points'][:, X_INDEX], bins=BINS, range=RANGE_XY)
            hist_y_curr, bins_y_curr = np.histogram(data_dict['points'][:, Y_INDEX], bins=BINS, range=RANGE_XY)
            hist_z_curr, bins_z_curr = np.histogram(data_dict['points'][:, Z_INDEX], bins=BINS, range=RANGE_Z)
            hist_intensity_curr, bins_intensity_curr = np.histogram(
                data_dict['points'][:, INTENSITY_INDEX], bins=BINS, range=RANGE_INTENSITY)

            mask_car = np.isin(data_dict['gt_names'], target_class_list)

            print("car gt_boxes:", data_dict['gt_boxes'][mask_car])
            hist_x_car_curr, bins_x_car_curr = np.histogram(
                data_dict['gt_boxes'][mask_car, CAR_X_INDEX], bins=BINS, range=RANGE_XY)
            hist_y_car_curr, bins_y_car_curr = np.histogram(
                data_dict['gt_boxes'][mask_car, CAR_Y_INDEX], bins=BINS, range=RANGE_XY)
            hist_z_car_curr, bins_z_car_curr = np.histogram(
                data_dict['gt_boxes'][mask_car, CAR_Z_INDEX], bins=BINS, range=RANGE_Z)
            hist_length_car_curr, bins_length_car_curr = np.histogram(
                data_dict['gt_boxes'][mask_car, CAR_LENGTH_INDEX], bins=BINS, range=RANGE_CAR_SIZE)
            hist_width_car_curr, bins_width_car_curr = np.histogram(
                data_dict['gt_boxes'][mask_car, CAR_WIDTH_INDEX], bins=BINS, range=RANGE_CAR_SIZE)
            hist_height_car_curr, bins_height_car_curr = np.histogram(
                data_dict['gt_boxes'][mask_car, CAR_HEIGHT_INDEX], bins=BINS, range=RANGE_CAR_SIZE)
            if hist_x is None:
                hist_x = hist_x_curr
                bins_x = bins_x_curr
                hist_y = hist_y_curr
                bins_y = bins_y_curr
                hist_z = hist_z_curr
                bins_z = bins_z_curr
                hist_intensity = hist_intensity_curr
                bins_intensity = bins_intensity_curr
                hist_x_car = hist_x_car_curr
                bins_x_car = bins_x_car_curr
                hist_y_car = hist_y_car_curr
                bins_y_car = bins_y_car_curr
                hist_z_car = hist_z_car_curr
                bins_z_car = bins_z_car_curr
                hist_length_car = hist_length_car_curr
                bins_length_car = bins_length_car_curr
                hist_width_car = hist_width_car_curr
                bins_width_car = bins_width_car_curr
                hist_height_car = hist_height_car_curr
                bins_height_car = bins_height_car_curr
            else:
                hist_x += hist_x_curr
                hist_y += hist_y_curr
                hist_z += hist_z_curr
                hist_intensity += hist_intensity_curr
                hist_x_car += hist_x_car_curr
                hist_y_car += hist_y_car_curr
                hist_z_car += hist_z_car_curr
                hist_length_car += hist_length_car_curr
                hist_width_car += hist_width_car_curr
                hist_height_car += hist_height_car_curr

            dataset_name = eval_dataset['loader'].dataset.dataset_ontology
            progress_bar.set_postfix_str(dataset_name)
            progress_bar.update()

            # if idx * args.batch_size >= 10000:
            #     print("Breaking after 10000 samples")
            #     break

        # initialize a matplotlib plot
        fig, (ax_x, ax_y, ax_z, ax_intensity) = plt.subplots(1, 4, figsize=(20, 5))
        ax_x.set_xlabel('point X [m]')
        ax_y.set_xlabel('point Y [m]')
        ax_z.set_xlabel('point Z [m]')
        ax_intensity.set_xlabel('point intensity')

        fig_car, ((ax_x_car, ax_y_car, ax_z_car), (ax_length_car, ax_width_car,
                  ax_height_car)) = plt.subplots(2, 3, figsize=(20, 10))
        ax_x_car.set_xlabel('car X [m]')
        ax_y_car.set_xlabel('car Y [m]')
        ax_z_car.set_xlabel('car Z [m]')
        ax_length_car.set_xlabel('car length [m]')
        ax_width_car.set_xlabel('car width [m]')
        ax_height_car.set_xlabel('car height [m]')

        # find peak of histogram
        peak_x = bins_x[np.argmax(hist_x)]
        peak_y = bins_y[np.argmax(hist_y)]
        peak_z = bins_z[np.argmax(hist_z)]
        peak_intensity = bins_intensity[np.argmax(hist_intensity)]
        peak_x_car = bins_x_car[np.argmax(hist_x_car)]
        peak_y_car = bins_y_car[np.argmax(hist_y_car)]
        peak_z_car = bins_z_car[np.argmax(hist_z_car)]
        peak_length_car = bins_length_car[np.argmax(hist_length_car)]
        peak_width_car = bins_width_car[np.argmax(hist_width_car)]
        peak_height_car = bins_height_car[np.argmax(hist_height_car)]

        # compute average of histgram
        average_x = np.average(bins_x[:-1], weights = hist_x)
        average_y = np.average(bins_y[:-1], weights = hist_y)
        average_z = np.average(bins_z[:-1], weights = hist_z)
        if np.sum(hist_intensity) == 0:
            average_intensity = 0
        else:
            average_intensity = np.average(bins_intensity[:-1], weights = hist_intensity)
        average_x_car = np.average(bins_x_car[:-1], weights = hist_x_car)
        average_y_car = np.average(bins_y_car[:-1], weights = hist_y_car)
        average_z_car = np.average(bins_z_car[:-1], weights = hist_z_car)
        average_length_car = np.average(bins_length_car[:-1], weights = hist_length_car)
        average_width_car = np.average(bins_width_car[:-1], weights = hist_width_car)
        average_height_car = np.average(bins_height_car[:-1], weights = hist_height_car)

        ax_x.title.set_text(f'Peak: {peak_x:.2f} m, Average: {average_x:.2f} m')
        ax_y.title.set_text(f'Peak: {peak_y:.2f} m, Average: {average_y:.2f} m')
        ax_z.title.set_text(f'Peak: {peak_z:.2f} m, Average: {average_z:.2f} m')
        ax_intensity.title.set_text(f'Peak: {peak_intensity:.2f}, Average: {average_intensity:.2f}')
        ax_x_car.title.set_text(f'Peak: {peak_x_car:.2f} m, Average: {average_x_car:.2f} m')
        ax_y_car.title.set_text(f'Peak: {peak_y_car:.2f} m, Average: {average_y_car:.2f} m')
        ax_z_car.title.set_text(f'Peak: {peak_z_car:.2f} m, Average: {average_z_car:.2f} m')
        ax_length_car.title.set_text(f'Peak: {peak_length_car:.2f} m, Average: {average_length_car:.2f} m')
        ax_width_car.title.set_text(f'Peak: {peak_width_car:.2f} m, Average: {average_width_car:.2f} m')
        ax_height_car.title.set_text(f'Peak: {peak_height_car:.2f} m, Average: {average_height_car:.2f} m')

        # finally, show the plot
        ax_x.bar(bins_x[:-1], hist_x / np.sum(hist_x), width=np.diff(bins_x), color='r', alpha=0.5)
        ax_y.bar(bins_y[:-1], hist_y / np.sum(hist_y), width=np.diff(bins_y), color='g', alpha=0.5)
        ax_z.bar(bins_z[:-1], hist_z / np.sum(hist_z), width=np.diff(bins_z), color='b', alpha=0.5)
        ax_intensity.bar(bins_intensity[:-1], hist_intensity / np.sum(hist_intensity),
                         width=np.diff(bins_intensity), color='y', alpha=0.5)

        ax_x_car.bar(bins_x_car[:-1], hist_x_car / np.sum(hist_x_car), width=np.diff(bins_x_car), color='r', alpha=0.5)
        ax_y_car.bar(bins_y_car[:-1], hist_y_car / np.sum(hist_y_car), width=np.diff(bins_y_car), color='g', alpha=0.5)
        ax_z_car.bar(bins_z_car[:-1], hist_z_car / np.sum(hist_z_car), width=np.diff(bins_z_car), color='b', alpha=0.5)
        ax_length_car.bar(bins_length_car[:-1], hist_length_car / np.sum(hist_length_car),
                          width=np.diff(bins_length_car), color='y', alpha=0.5)
        ax_width_car.bar(bins_width_car[:-1], hist_width_car / np.sum(hist_width_car),
                         width=np.diff(bins_width_car), color='y', alpha=0.5)
        ax_height_car.bar(bins_height_car[:-1], hist_height_car / np.sum(hist_height_car),
                          width=np.diff(bins_height_car), color='y', alpha=0.5)

        filename = os.path.join(args.out_dir, f'{dataset_name}_point_hist.png')
        fig.savefig(filename)
        wandb.save(filename)
        wandb.log({f'val/{dataset_name}/point histogram': wandb.Image(filename)})
        filename_car = os.path.join(args.out_dir, f'{dataset_name}_gt_car_hist.png')
        fig_car.savefig(filename_car)
        wandb.save(filename_car)
        wandb.log({f'val/{dataset_name}/GT car histogram': wandb.Image(filename_car)})

    logger.info('Dataset analysis done.')

    wandb.finish()


if __name__ == '__main__':
    main()
