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
from test import get_all_configs
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
    parser.add_argument('--is_train', action='store_true', help='analyze train set, otherwise use test set')
    parser.add_argument('--figure_suffix', type=str, default="", help='add suffix to figure name')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def init_point_plot():
    # initialize a matplotlib plot
    fig, ((ax_x, ax_y, ax_z, ax_intensity),
          (ax_dist, ax_num_points, ax_num_voxels, ax_num_points_in_voxel)) = plt.subplots(2, 4, figsize=(20, 10))
    YLABEL_PER_FRAME = "Num. of points per frame"
    YLABEL_STATS = "Probability"
    ax_x.set_xlabel('X [m]')
    ax_x.set_ylabel(YLABEL_STATS)
    ax_y.set_xlabel('Y [m]')
    ax_y.set_ylabel(YLABEL_STATS)
    ax_z.set_xlabel('Z [m]')
    ax_z.set_ylabel(YLABEL_STATS)
    ax_intensity.set_xlabel('Intensity')
    ax_intensity.set_ylabel(YLABEL_STATS)
    ax_dist.set_xlabel('Distance [m]')
    ax_dist.set_ylabel(YLABEL_PER_FRAME)
    ax_num_points.set_xlabel('Num. of points')
    ax_num_points.set_ylabel(YLABEL_STATS)
    ax_num_voxels.set_xlabel('Num. of voxels')
    ax_num_voxels.set_ylabel(YLABEL_STATS)
    ax_num_points_in_voxel.set_xlabel('Num. points in voxel')
    ax_num_points_in_voxel.set_ylabel(YLABEL_STATS)

    return fig, ((ax_x, ax_y, ax_z, ax_intensity), (ax_num_points, ax_num_voxels, ax_num_points_in_voxel, ax_dist))


def init_single_plot(xlabel="", ylabel=""):
    # initialize a matplotlib plot
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def init_gt_car_plot():
    # To make the figure size same as point.
    fig_car, ((ax_x_car, ax_y_car, ax_z_car, ax_dist_car), (ax_length_car, ax_width_car,
                                                            ax_height_car, _)) = plt.subplots(2, 4, figsize=(20, 10))
    YLABEL_PER_FRAME = "Num. of labels per frame"
    YLABEL_STATS = "Probability"
    ax_x_car.set_xlabel('X [m]')
    ax_x_car.set_ylabel(YLABEL_STATS)
    ax_y_car.set_xlabel('Y [m]')
    ax_y_car.set_ylabel(YLABEL_STATS)
    ax_z_car.set_xlabel('Z [m]')
    ax_z_car.set_ylabel(YLABEL_STATS)
    ax_dist_car.set_xlabel('Distance [m]')
    ax_dist_car.set_ylabel(YLABEL_PER_FRAME)
    ax_length_car.set_xlabel('Length [m]')
    ax_length_car.set_ylabel(YLABEL_STATS)
    ax_width_car.set_xlabel('Width [m]')
    ax_width_car.set_ylabel(YLABEL_STATS)
    ax_height_car.set_xlabel('Height [m]')
    ax_height_car.set_ylabel(YLABEL_STATS)
    return fig_car, ((ax_x_car, ax_y_car, ax_z_car, ax_dist_car), (ax_length_car, ax_width_car,
                                                                   ax_height_car, _))


def init_gt_ped_plot():
    fig_ped, ((ax_x_ped, ax_y_ped, ax_z_ped), (ax_length_ped, ax_width_ped,
                                               ax_height_ped)) = plt.subplots(2, 3, figsize=(20, 10))
    YLABEL_PER_FRAME = "Num points per frame"
    YLABEL_STATS = "Probability"
    ax_x_ped.set_xlabel('GT Pedestrian X [m]')
    ax_x_ped.set_ylabel(YLABEL_STATS)
    ax_y_ped.set_xlabel('GT Pedestrian Y [m]')
    ax_y_ped.set_ylabel(YLABEL_STATS)
    ax_z_ped.set_xlabel('GT Pedestrian Z [m]')
    ax_z_ped.set_ylabel(YLABEL_STATS)
    ax_length_ped.set_xlabel('GT Pedestrian length [m]')
    ax_length_ped.set_ylabel(YLABEL_STATS)
    ax_width_ped.set_xlabel('GT Pedestrian width [m]')
    ax_width_ped.set_ylabel(YLABEL_STATS)
    ax_height_ped.set_xlabel('GT Pedestrian height [m]')
    ax_height_ped.set_ylabel(YLABEL_STATS)
    return fig_ped, ((ax_x_ped, ax_y_ped, ax_z_ped), (ax_length_ped, ax_width_ped,
                                                      ax_height_ped))


def init_pred_car_plot():
    fig_car_pred, ((ax_x_car_pred, ax_y_car_pred, ax_z_car_pred), (ax_length_car_pred, ax_width_car_pred,
                                                                   ax_height_car_pred)) = plt.subplots(2, 3, figsize=(20, 10))
    ax_x_car_pred.set_xlabel('Pred Car X [m]')
    ax_y_car_pred.set_xlabel('Pred Car Y [m]')
    ax_z_car_pred.set_xlabel('Pred Car Z [m]')
    ax_length_car_pred.set_xlabel('Pred Car length [m]')
    ax_width_car_pred.set_xlabel('Pred Car width [m]')
    ax_height_car_pred.set_xlabel('Pred Car height [m]')
    return fig_car_pred, ((ax_x_car_pred, ax_y_car_pred, ax_z_car_pred), (ax_length_car_pred, ax_width_car_pred,
                                                                          ax_height_car_pred))


def init_pred_ped_plot():
    fig_ped_pred, ((ax_x_ped_pred, ax_y_ped_pred, ax_z_ped_pred), (ax_length_ped_pred, ax_width_ped_pred,
                                                                   ax_height_ped_pred)) = plt.subplots(2, 3, figsize=(20, 10))
    ax_x_ped_pred.set_xlabel('Pred Pedestrian X [m]')
    ax_y_ped_pred.set_xlabel('Pred Pedestrian Y [m]')
    ax_z_ped_pred.set_xlabel('Pred Pedestrian Z [m]')
    ax_length_ped_pred.set_xlabel('Pred Pedestrian length [m]')
    ax_width_ped_pred.set_xlabel('Pred Pedestrian width [m]')
    ax_height_ped_pred.set_xlabel('Pred Pedestrian height [m]')
    return fig_ped_pred, ((ax_x_ped_pred, ax_y_ped_pred, ax_z_ped_pred), (ax_length_ped_pred, ax_width_ped_pred,
                                                                          ax_height_ped_pred))


def set_stats_to_title(ax_x, peak_x, average_x):
    ax_x.title.set_text(f'Peak: {peak_x:.2f}, Average: {average_x:.2f}')


def copy_bar_plot(ax_orig, filename):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    for container in ax_orig.containers:
        for patch in container.patches:
            x_pos = patch.get_x()
            height = patch.get_height()
            width = patch.get_width()
            ax.bar(x_pos, height, width=width, color=patch.get_facecolor())
    ax.set_xlim(ax_orig.get_xlim())
    ax.set_ylim(ax_orig.get_ylim())
    ax.set_xlabel(ax_orig.get_xlabel(), fontsize=16)
    ax.set_ylabel(ax_orig.get_ylabel(), fontsize=16)
    # Create a new legend for the new figure
    handles, labels = ax_orig.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.savefig(filename)
    fig.savefig(filename[:-4] + ".svg")
    return fig, ax


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Run Dataset Analysis-------------------------')

    wandb.init(config=vars(cfg), project="st3d", name=args.run_name)

    # Dataset configs
    eval_configs = get_all_configs(cfg)
    eval_config_rep = list(eval_configs.values())[0]

    eval_datasets = list()
    for eval_config in eval_configs.values():
        eval_set, eval_loader, eval_sampler = build_dataloader(
            dataset_cfg=eval_config,
            class_names=eval_config.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=False, workers=1,
            logger=logger, training=args.is_train,
            model_ontology=cfg.get('ONTOLOGY', None)
        )
        eval_dataset = dict(dataset_class=eval_set, loader=eval_loader, sampler=eval_sampler)
        eval_datasets.append(eval_dataset)
        logger.info(f'Total number of samples: \t{len(eval_loader)}')

    model = None
    if args.ckpt:
        eval_dataset_rep = eval_datasets[0]
        model = build_network(model_cfg=cfg.MODEL, num_class=len(
            cfg.CLASS_NAMES), dataset=eval_dataset_rep['dataset_class'])
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        logger.info("Model loaded")

    # Change color map.
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap("tab10").colors)

    fig_point_all, ((ax_x_point_all, ax_y_point_all, ax_z_point_all, ax_intensity_point_all),
                    (ax_num_points_point_all, ax_num_voxels_point_all, ax_num_points_in_voxel_point_all, ax_dist_point_all)) = init_point_plot()
    fig_gt_car_all, ((ax_x_gt_car_all, ax_y_gt_car_all, ax_z_gt_car_all, ax_dist_gt_car_all), (ax_length_gt_car_all, ax_width_gt_car_all,
                                                                                               ax_height_gt_car_all, _)) = init_gt_car_plot()
    fig_gt_ped_all, ((ax_x_gt_ped_all, ax_y_gt_ped_all, ax_z_gt_ped_all), (ax_length_gt_ped_all, ax_width_gt_ped_all,
                                                                           ax_height_gt_ped_all)) = init_gt_ped_plot()
    if model:
        fig_pred_car_all, ((ax_x_pred_car_all, ax_y_pred_car_all, ax_z_pred_car_all), (ax_length_pred_car_all, ax_width_pred_car_all,
                                                                                       ax_height_pred_car_all)) = init_pred_car_plot()
        fig_pred_ped_all, ((ax_x_pred_ped_all, ax_y_pred_ped_all, ax_z_pred_ped_all), (ax_length_pred_ped_all, ax_width_pred_ped_all,
                                                                                       ax_height_pred_ped_all)) = init_pred_ped_plot()

    for eval_dataset in eval_datasets:
        dataset_name = eval_dataset['loader'].dataset.dataset_ontology
        print("dataset onotology:", dataset_name)
        hist_x = None
        bins_x = None
        hist_y = None
        bins_y = None
        hist_z = None
        bins_z = None
        hist_intensity = None
        bins_intensity = None
        hist_num_points = None
        bins_num_points = None
        hist_num_points_in_voxel = None
        bins_num_points_in_voxel = None
        hist_num_voxels = None
        bins_num_voxels = None
        hist_dist = None
        bins_dist = None
        hist_x_car = None
        hist_y_car = None
        hist_z_car = None
        hist_length_car = None
        hist_width_car = None
        hist_height_car = None
        hist_x_car_pred = None
        bins_x_car_pred = None
        hist_y_car_pred = None
        bins_y_car_pred = None
        hist_z_car_pred = None
        bins_z_car_pred = None
        hist_length_car_pred = None
        bins_length_car_pred = None
        hist_width_car_pred = None
        bins_width_car_pred = None
        hist_height_car_pred = None
        bins_height_car_pred = None
        X_INDEX = 1
        Y_INDEX = 2
        Z_INDEX = 3
        INTENSITY_INDEX = 4
        BOX_X_INDEX = 0
        BOX_Y_INDEX = 1
        BOX_Z_INDEX = 2
        BOX_LENGTH_INDEX = 3
        BOX_WIDTH_INDEX = 4
        BOX_HEIGHT_INDEX = 5
        BINS = 50
        BINS_SIZE = 100
        RANGE_XY = (-100, 100)
        RANGE_Z = (-10, 10)
        RANGE_INTENSITY = (0, 1)
        RANGE_BOX_SIZE = (0, 8)
        RANGE_NUM_POINTS = (20000, 200000)
        RANGE_NUM_VOXELS = (0, 100000)
        RANGE_NUM_POINTS_IN_VOXEL = (0, 10)
        RANGE_DIST = (0, 100)
        MAX_SAMPLE = 5

        progress_bar = tqdm.tqdm(total=len(eval_dataset['loader']), leave=True, desc='eval', dynamic_ncols=True)
        car_class_list = ["Vehicle", "Car", "car", "waymo:Vehicle",
                          "pandaset:Car", "lyft:car", "nuscenes:car", "kitti:Car"]
        pedestrian_class_list = ["Pedestrian", "pedestrian", "waymo:Pedestrian",
                                 "pandaset:Pedestrian", "lyft:pedestrian", "nuscenes:pedestrian", "kitti:Pedestrian"]

        for idx, data_dict in enumerate(eval_dataset['loader']):
            hist_x_curr, bins_x_curr = np.histogram(data_dict['points'][:, X_INDEX], bins=BINS, range=RANGE_XY)
            hist_y_curr, bins_y_curr = np.histogram(data_dict['points'][:, Y_INDEX], bins=BINS, range=RANGE_XY)
            hist_z_curr, bins_z_curr = np.histogram(data_dict['points'][:, Z_INDEX], bins=BINS, range=RANGE_Z)

            with_intensity = data_dict['points'].shape[1] > INTENSITY_INDEX
            if with_intensity:
                range = RANGE_INTENSITY
                norm_factor = 256.0 if eval_dataset['loader'].dataset.dataset_ontology in ('lyft', 'nuscenes') else 1
                hist_intensity_curr, bins_intensity_curr = np.histogram(
                    data_dict['points'][:, INTENSITY_INDEX] / norm_factor, bins=BINS, range=RANGE_INTENSITY)

            hist_num_points_curr, bins_num_points_curr = np.histogram(
                data_dict['points'].shape[0], bins=BINS, range=RANGE_NUM_POINTS)

            hist_num_voxels_curr, bins_num_voxels_curr = np.histogram(
                data_dict['voxels'].shape[0], bins=BINS, range=RANGE_NUM_VOXELS)

            hist_num_points_in_voxel_curr, bins_num_points_in_voxel_curr = np.histogram(
                data_dict['voxel_num_points'], bins=RANGE_NUM_POINTS_IN_VOXEL[-1], range=RANGE_NUM_POINTS_IN_VOXEL)

            dist = np.linalg.norm(data_dict['points'][:, X_INDEX:Y_INDEX], axis=1)
            hist_dist_curr, bins_dist_curr = np.histogram(dist, bins=BINS, range=RANGE_DIST)

            car_class_index = -1
            pedestrian_class_index = -1
            for index, class_name in enumerate(eval_dataset['loader'].dataset.class_names):
                if class_name in car_class_list:
                    car_class_index = index + 1  # +1 because of background class
                if class_name in pedestrian_class_list:
                    pedestrian_class_index = index + 1  # +1 because of background class
            mask_car = data_dict['gt_boxes'][:, :, 7] == car_class_index  # accesing gt_classes index.
            mask_ped = data_dict['gt_boxes'][:, :, 7] == pedestrian_class_index  # accesing gt_classes index.

            hist_x_car_curr, bins_x_car_curr = np.histogram(
                data_dict['gt_boxes'][mask_car, BOX_X_INDEX], bins=BINS, range=RANGE_XY)
            hist_y_car_curr, bins_y_car_curr = np.histogram(
                data_dict['gt_boxes'][mask_car, BOX_Y_INDEX], bins=BINS, range=RANGE_XY)
            hist_z_car_curr, bins_z_car_curr = np.histogram(
                data_dict['gt_boxes'][mask_car, BOX_Z_INDEX], bins=BINS, range=RANGE_Z)
            hist_length_car_curr, bins_length_car_curr = np.histogram(
                data_dict['gt_boxes'][mask_car, BOX_LENGTH_INDEX], bins=BINS_SIZE, range=RANGE_BOX_SIZE)
            hist_width_car_curr, bins_width_car_curr = np.histogram(
                data_dict['gt_boxes'][mask_car, BOX_WIDTH_INDEX], bins=BINS_SIZE, range=RANGE_BOX_SIZE)
            hist_height_car_curr, bins_height_car_curr = np.histogram(
                data_dict['gt_boxes'][mask_car, BOX_HEIGHT_INDEX], bins=BINS_SIZE, range=RANGE_BOX_SIZE)
            dist_car = np.linalg.norm(data_dict['gt_boxes'][mask_car, BOX_X_INDEX:BOX_Y_INDEX], axis=1)
            hist_dist_car_curr, bins_dist_car_curr = np.histogram(dist_car, bins=BINS, range=RANGE_DIST)

            hist_x_ped_curr, bins_x_ped_curr = np.histogram(
                data_dict['gt_boxes'][mask_ped, BOX_X_INDEX], bins=BINS, range=RANGE_XY)
            hist_y_ped_curr, bins_y_ped_curr = np.histogram(
                data_dict['gt_boxes'][mask_ped, BOX_Y_INDEX], bins=BINS, range=RANGE_XY)
            hist_z_ped_curr, bins_z_ped_curr = np.histogram(
                data_dict['gt_boxes'][mask_ped, BOX_Z_INDEX], bins=BINS, range=RANGE_Z)
            hist_length_ped_curr, bins_length_ped_curr = np.histogram(
                data_dict['gt_boxes'][mask_ped, BOX_LENGTH_INDEX], bins=BINS_SIZE, range=RANGE_BOX_SIZE)
            hist_width_ped_curr, bins_width_ped_curr = np.histogram(
                data_dict['gt_boxes'][mask_ped, BOX_WIDTH_INDEX], bins=BINS_SIZE, range=RANGE_BOX_SIZE)
            hist_height_ped_curr, bins_height_ped_curr = np.histogram(
                data_dict['gt_boxes'][mask_ped, BOX_HEIGHT_INDEX], bins=BINS_SIZE, range=RANGE_BOX_SIZE)

            if model:
                load_data_to_gpu(data_dict)
                with torch.no_grad():
                    pred_dicts, _ = model.forward(data_dict)
                annos = eval_dataset['loader'].dataset.generate_prediction_dicts(
                    data_dict, pred_dicts, cfg.CLASS_NAMES,
                )

                annos_car = [anno for anno in annos if np.isin(anno['name'], car_class_list).any()]
                annos_car_boxes = np.array([anno['boxes_lidar'] for anno in annos_car])
                hist_x_car_pred_curr, bins_x_car_pred_curr = np.histogram(
                    annos_car_boxes[:, :, BOX_X_INDEX], bins=BINS, range=RANGE_XY)
                hist_y_car_pred_curr, bins_y_car_pred_curr = np.histogram(
                    annos_car_boxes[:, :, BOX_Y_INDEX], bins=BINS, range=RANGE_XY)
                hist_z_car_pred_curr, bins_z_car_pred_curr = np.histogram(
                    annos_car_boxes[:, :, BOX_Z_INDEX], bins=BINS, range=RANGE_Z)
                hist_length_car_pred_curr, bins_length_car_pred_curr = np.histogram(
                    annos_car_boxes[:, :, BOX_LENGTH_INDEX], bins=BINS, range=RANGE_BOX_SIZE)
                hist_width_car_pred_curr, bins_width_car_pred_curr = np.histogram(
                    annos_car_boxes[:, :, BOX_WIDTH_INDEX], bins=BINS, range=RANGE_BOX_SIZE)
                hist_height_car_pred_curr, bins_height_car_pred_curr = np.histogram(
                    annos_car_boxes[:, :, BOX_HEIGHT_INDEX], bins=BINS, range=RANGE_BOX_SIZE)

                annos_ped = [anno for anno in annos if np.isin(anno['name'], ped_class_list).any()]
                annos_ped_boxes = np.array([anno['boxes_lidar'] for anno in annos_ped])
                hist_x_ped_pred_curr, bins_x_ped_pred_curr = np.histogram(
                    annos_ped_boxes[:, :, BOX_X_INDEX], bins=BINS, range=RANGE_XY)
                hist_y_ped_pred_curr, bins_y_ped_pred_curr = np.histogram(
                    annos_ped_boxes[:, :, BOX_Y_INDEX], bins=BINS, range=RANGE_XY)
                hist_z_ped_pred_curr, bins_z_ped_pred_curr = np.histogram(
                    annos_ped_boxes[:, :, BOX_Z_INDEX], bins=BINS, range=RANGE_Z)
                hist_length_ped_pred_curr, bins_length_ped_pred_curr = np.histogram(
                    annos_ped_boxes[:, :, BOX_LENGTH_INDEX], bins=BINS, range=RANGE_BOX_SIZE)
                hist_width_ped_pred_curr, bins_width_ped_pred_curr = np.histogram(
                    annos_ped_boxes[:, :, BOX_WIDTH_INDEX], bins=BINS, range=RANGE_BOX_SIZE)
                hist_height_ped_pred_curr, bins_height_ped_pred_curr = np.histogram(
                    annos_ped_boxes[:, :, BOX_HEIGHT_INDEX], bins=BINS, range=RANGE_BOX_SIZE)

            if hist_x is None:
                hist_x = hist_x_curr
                bins_x = bins_x_curr
                hist_y = hist_y_curr
                bins_y = bins_y_curr
                hist_z = hist_z_curr
                bins_z = bins_z_curr
                if with_intensity:
                    hist_intensity = hist_intensity_curr
                    bins_intensity = bins_intensity_curr
                hist_num_points = hist_num_points_curr
                bins_num_points = bins_num_points_curr
                hist_num_voxels = hist_num_voxels_curr
                bins_num_voxels = bins_num_voxels_curr
                hist_num_points_in_voxel = hist_num_points_in_voxel_curr
                bins_num_points_in_voxel = bins_num_points_in_voxel_curr
                hist_dist = hist_dist_curr
                bins_dist = bins_dist_curr
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
                hist_dist_car = hist_dist_car_curr
                bins_dist_car = bins_dist_car_curr
                hist_x_ped = hist_x_ped_curr
                bins_x_ped = bins_x_ped_curr
                hist_y_ped = hist_y_ped_curr
                bins_y_ped = bins_y_ped_curr
                hist_z_ped = hist_z_ped_curr
                bins_z_ped = bins_z_ped_curr
                hist_length_ped = hist_length_ped_curr
                bins_length_ped = bins_length_ped_curr
                hist_width_ped = hist_width_ped_curr
                bins_width_ped = bins_width_ped_curr
                hist_height_ped = hist_height_ped_curr
                bins_height_ped = bins_height_ped_curr
                if model:
                    hist_x_car_pred = hist_x_car_pred_curr
                    bins_x_car_pred = bins_x_car_pred_curr
                    hist_y_car_pred = hist_y_car_pred_curr
                    bins_y_car_pred = bins_y_car_pred_curr
                    hist_z_car_pred = hist_z_car_pred_curr
                    bins_z_car_pred = bins_z_car_pred_curr
                    hist_length_car_pred = hist_length_car_pred_curr
                    bins_length_car_pred = bins_length_car_pred_curr
                    hist_width_car_pred = hist_width_car_pred_curr
                    bins_width_car_pred = bins_width_car_pred_curr
                    hist_height_car_pred = hist_height_car_pred_curr
                    bins_height_car_pred = bins_height_car_pred_curr
                    hist_dist_car_pred = hist_dist_car_pred_curr
                    bins_dist_car_pred = bins_dist_car_pred_curr
                    hist_x_ped_pred = hist_x_ped_pred_curr
                    bins_x_ped_pred = bins_x_ped_pred_curr
                    hist_y_ped_pred = hist_y_ped_pred_curr
                    bins_y_ped_pred = bins_y_ped_pred_curr
                    hist_z_ped_pred = hist_z_ped_pred_curr
                    bins_z_ped_pred = bins_z_ped_pred_curr
                    hist_length_ped_pred = hist_length_ped_pred_curr
                    bins_length_ped_pred = bins_length_ped_pred_curr
                    hist_width_ped_pred = hist_width_ped_pred_curr
                    bins_width_ped_pred = bins_width_ped_pred_curr
                    hist_height_ped_pred = hist_height_ped_pred_curr
                    bins_height_ped_pred = bins_height_ped_pred_curr
            else:
                hist_x += hist_x_curr
                hist_y += hist_y_curr
                hist_z += hist_z_curr
                if with_intensity:
                    hist_intensity += hist_intensity_curr
                hist_num_points += hist_num_points_curr
                hist_num_voxels += hist_num_voxels_curr
                hist_num_points_in_voxel += hist_num_points_in_voxel_curr
                hist_dist += hist_dist_curr
                hist_x_car += hist_x_car_curr
                hist_y_car += hist_y_car_curr
                hist_z_car += hist_z_car_curr
                hist_length_car += hist_length_car_curr
                hist_width_car += hist_width_car_curr
                hist_height_car += hist_height_car_curr
                hist_x_ped += hist_x_ped_curr
                hist_y_ped += hist_y_ped_curr
                hist_z_ped += hist_z_ped_curr
                hist_length_ped += hist_length_ped_curr
                hist_width_ped += hist_width_ped_curr
                hist_height_ped += hist_height_ped_curr
                hist_dist_car += hist_dist_car_curr
                if model:
                    hist_x_car_pred += hist_x_car_pred_curr
                    hist_y_car_pred += hist_y_car_pred_curr
                    hist_z_car_pred += hist_z_car_pred_curr
                    hist_length_car_pred += hist_length_car_pred_curr
                    hist_width_car_pred += hist_width_car_pred_curr
                    hist_height_car_pred += hist_height_car_pred_curr
                    hist_x_ped_pred += hist_x_ped_pred_curr
                    hist_y_ped_pred += hist_y_ped_pred_curr
                    hist_z_ped_pred += hist_z_ped_pred_curr
                    hist_length_ped_pred += hist_length_ped_pred_curr
                    hist_width_ped_pred += hist_width_ped_pred_curr
                    hist_height_ped_pred += hist_height_ped_pred_curr

            dataset_name = eval_dataset['loader'].dataset.dataset_ontology
            progress_bar.set_postfix_str(dataset_name)
            progress_bar.update()

            if idx * args.batch_size >= MAX_SAMPLE:
                print("Breaking after {} samples".format(MAX_SAMPLE))
                break

        # initialize a matplotlib plot
        fig, ((ax_x, ax_y, ax_z, ax_intensity), (ax_num_points, ax_num_voxels,
              ax_num_points_in_voxel, ax_dist)) = init_point_plot()
        if hist_intensity is None:
            ax_intensity.axis('off')

        fig_car, ((ax_x_car, ax_y_car, ax_z_car, ax_dist_car), (ax_length_car, ax_width_car,
                  ax_height_car, _)) = init_gt_car_plot()

        fig_ped, ((ax_x_ped, ax_y_ped, ax_z_ped), (ax_length_ped, ax_width_ped,
                                                   ax_height_ped)) = init_gt_ped_plot()

        if model:
            fig_car_pred, ((ax_x_car_pred, ax_y_car_pred, ax_z_car_pred), (ax_length_car_pred, ax_width_car_pred,
                                                                           ax_height_car_pred)) = init_pred_car_plot()
            fig_ped_pred, ((ax_x_ped_pred, ax_y_ped_pred, ax_z_ped_pred), (ax_length_ped_pred, ax_width_ped_pred,
                                                                           ax_height_ped_pred)) = init_pred_ped_plot()

        # find peak of histogram
        peak_x = bins_x[np.argmax(hist_x)]
        peak_y = bins_y[np.argmax(hist_y)]
        peak_z = bins_z[np.argmax(hist_z)]
        if hist_intensity is not None:
            peak_intensity = bins_intensity[np.argmax(hist_intensity)]
        peak_num_points = bins_num_points[np.argmax(hist_num_points)]
        peak_num_voxels = bins_num_voxels[np.argmax(hist_num_voxels)]
        peak_num_points_in_voxel = bins_num_points_in_voxel[np.argmax(hist_num_points_in_voxel)]
        peak_dist = bins_dist[np.argmax(hist_dist)]
        peak_x_car = bins_x_car[np.argmax(hist_x_car)]
        peak_y_car = bins_y_car[np.argmax(hist_y_car)]
        peak_z_car = bins_z_car[np.argmax(hist_z_car)]
        peak_length_car = bins_length_car[np.argmax(hist_length_car)]
        peak_width_car = bins_width_car[np.argmax(hist_width_car)]
        peak_height_car = bins_height_car[np.argmax(hist_height_car)]
        peak_x_ped = bins_x_ped[np.argmax(hist_x_ped)]
        peak_y_ped = bins_y_ped[np.argmax(hist_y_ped)]
        peak_z_ped = bins_z_ped[np.argmax(hist_z_ped)]
        peak_length_ped = bins_length_ped[np.argmax(hist_length_ped)]
        peak_width_ped = bins_width_ped[np.argmax(hist_width_ped)]
        peak_height_ped = bins_height_ped[np.argmax(hist_height_ped)]
        peak_dist_car = bins_dist_car[np.argmax(hist_dist_car)]
        if model:
            peak_x_car_pred = bins_x_car_pred[np.argmax(hist_x_car_pred)]
            peak_y_car_pred = bins_y_car_pred[np.argmax(hist_y_car_pred)]
            peak_z_car_pred = bins_z_car_pred[np.argmax(hist_z_car_pred)]
            peak_length_car_pred = bins_length_car_pred[np.argmax(hist_length_car_pred)]
            peak_width_car_pred = bins_width_car_pred[np.argmax(hist_width_car_pred)]
            peak_height_car_pred = bins_height_car_pred[np.argmax(hist_height_car_pred)]
            peak_x_ped_pred = bins_x_ped_pred[np.argmax(hist_x_ped_pred)]
            peak_y_ped_pred = bins_y_ped_pred[np.argmax(hist_y_ped_pred)]
            peak_z_ped_pred = bins_z_ped_pred[np.argmax(hist_z_ped_pred)]
            peak_length_ped_pred = bins_length_ped_pred[np.argmax(hist_length_ped_pred)]
            peak_width_ped_pred = bins_width_ped_pred[np.argmax(hist_width_ped_pred)]
            peak_height_ped_pred = bins_height_ped_pred[np.argmax(hist_height_ped_pred)]

        # save histogram to file
        np.save(f"/storage/hist_dist_{dataset_name}_tmp.npy", hist_dist)

        # compute average of histgram
        average_x = np.average(bins_x[:-1], weights=hist_x) if hist_x.sum() > 0 else 0
        average_y = np.average(bins_y[:-1], weights=hist_y) if hist_y.sum() > 0 else 0
        average_z = np.average(bins_z[:-1], weights=hist_z) if hist_z.sum() > 0 else 0
        average_intensity = np.average(bins_intensity[:-1], weights=hist_intensity) if hist_intensity is not None else 0
        average_num_points = np.average(
            bins_num_points[:-1], weights=hist_num_points) if hist_num_points.sum() > 0 else 0
        average_num_voxels = np.average(
            bins_num_voxels[:-1], weights=hist_num_voxels) if hist_num_voxels.sum() > 0 else 0
        average_num_points_in_voxel = np.average(
            bins_num_points_in_voxel[:-1], weights=hist_num_points_in_voxel) if hist_num_points_in_voxel.sum() > 0 else 0
        average_dist = np.average(bins_dist[:-1], weights=hist_dist) if hist_dist.sum() > 0 else 0
        average_x_car = np.average(bins_x_car[:-1], weights=hist_x_car) if hist_x_car.sum() > 0 else 0
        average_y_car = np.average(bins_y_car[:-1], weights=hist_y_car) if hist_y_car.sum() > 0 else 0
        average_z_car = np.average(bins_z_car[:-1], weights=hist_z_car) if hist_z_car.sum() > 0 else 0
        average_length_car = np.average(
            bins_length_car[:-1], weights=hist_length_car) if hist_length_car.sum() > 0 else 0
        average_width_car = np.average(bins_width_car[:-1], weights=hist_width_car) if hist_width_car.sum() > 0 else 0
        average_height_car = np.average(
            bins_height_car[:-1], weights=hist_height_car) if hist_height_car.sum() > 0 else 0
        average_dist_car = np.average(bins_dist_car[:-1], weights=hist_dist_car) if hist_dist_car.sum() > 0 else 0
        average_x_ped = np.average(bins_x_ped[:-1], weights=hist_x_ped) if hist_x_ped.sum() > 0 else 0
        average_y_ped = np.average(bins_y_ped[:-1], weights=hist_y_ped) if hist_y_ped.sum() > 0 else 0
        average_z_ped = np.average(bins_z_ped[:-1], weights=hist_z_ped) if hist_z_ped.sum() > 0 else 0
        average_length_ped = np.average(
            bins_length_ped[:-1], weights=hist_length_ped) if hist_length_ped.sum() > 0 else 0
        average_width_ped = np.average(bins_width_ped[:-1], weights=hist_width_ped) if hist_width_ped.sum() > 0 else 0
        average_height_ped = np.average(
            bins_height_ped[:-1], weights=hist_height_ped) if hist_height_ped.sum() > 0 else 0
        if model:
            average_x_car_pred = np.average(bins_x_car_pred[:-1], weights=hist_x_car_pred)
            average_y_car_pred = np.average(bins_y_car_pred[:-1], weights=hist_y_car_pred)
            average_z_car_pred = np.average(bins_z_car_pred[:-1], weights=hist_z_car_pred)
            average_length_car_pred = np.average(bins_length_car_pred[:-1], weights=hist_length_car_pred)
            average_width_car_pred = np.average(bins_width_car_pred[:-1], weights=hist_width_car_pred)
            average_height_car_pred = np.average(bins_height_car_pred[:-1], weights=hist_height_car_pred)
            average_x_ped_pred = np.average(bins_x_ped_pred[:-1], weights=hist_x_ped_pred)
            average_y_ped_pred = np.average(bins_y_ped_pred[:-1], weights=hist_y_ped_pred)
            average_z_ped_pred = np.average(bins_z_ped_pred[:-1], weights=hist_z_ped_pred)
            average_length_ped_pred = np.average(bins_length_ped_pred[:-1], weights=hist_length_ped_pred)
            average_width_ped_pred = np.average(bins_width_ped_pred[:-1], weights=hist_width_ped_pred)
            average_height_ped_pred = np.average(bins_height_ped_pred[:-1], weights=hist_height_ped_pred)

        set_stats_to_title(ax_x, peak_x, average_x)
        set_stats_to_title(ax_y, peak_y, average_y)
        set_stats_to_title(ax_z, peak_z, average_z)
        if hist_intensity is not None:
            set_stats_to_title(ax_intensity, peak_intensity, average_intensity)
        set_stats_to_title(ax_num_points, peak_num_points, average_num_points)
        set_stats_to_title(ax_num_voxels, peak_num_voxels, average_num_voxels)
        set_stats_to_title(ax_num_points_in_voxel, peak_num_points_in_voxel, average_num_points_in_voxel)
        set_stats_to_title(ax_dist, peak_dist, average_dist)
        set_stats_to_title(ax_x_car, peak_x_car, average_x_car)
        set_stats_to_title(ax_y_car, peak_y_car, average_y_car)
        set_stats_to_title(ax_z_car, peak_z_car, average_z_car)
        set_stats_to_title(ax_length_car, peak_length_car, average_length_car)
        set_stats_to_title(ax_width_car, peak_width_car, average_width_car)
        set_stats_to_title(ax_height_car, peak_height_car, average_height_car)
        set_stats_to_title(ax_dist_car, peak_dist_car, average_dist_car)
        set_stats_to_title(ax_x_ped, peak_x_ped, average_x_ped)
        set_stats_to_title(ax_y_ped, peak_y_ped, average_y_ped)
        set_stats_to_title(ax_z_ped, peak_z_ped, average_z_ped)
        set_stats_to_title(ax_length_ped, peak_length_ped, average_length_ped)
        set_stats_to_title(ax_width_ped, peak_width_ped, average_width_ped)
        set_stats_to_title(ax_height_ped, peak_height_ped, average_height_ped)
        if model:
            set_stats_to_title(ax_x_car_pred, peak_x_car_pred, average_x_car_pred)
            set_stats_to_title(ax_y_car_pred, peak_y_car_pred, average_y_car_pred)
            set_stats_to_title(ax_z_car_pred, peak_z_car_pred, average_z_car_pred)
            set_stats_to_title(ax_length_car_pred, peak_length_car_pred, average_length_car_pred)
            set_stats_to_title(ax_width_car_pred, peak_width_car_pred, average_width_car_pred)
            set_stats_to_title(ax_height_car_pred, peak_height_car_pred, average_height_car_pred)
            set_stats_to_title(ax_x_ped_pred, peak_x_ped_pred, average_x_ped_pred)
            set_stats_to_title(ax_y_ped_pred, peak_y_ped_pred, average_y_ped_pred)
            set_stats_to_title(ax_z_ped_pred, peak_z_ped_pred, average_z_ped_pred)
            set_stats_to_title(ax_length_ped_pred, peak_length_ped_pred, average_length_ped_pred)
            set_stats_to_title(ax_width_ped_pred, peak_width_ped_pred, average_width_ped_pred)
            set_stats_to_title(ax_height_ped_pred, peak_height_ped_pred, average_height_ped_pred)

        # finally, show the plot
        ALPHA = 0.8
        ax_x.bar(bins_x[:-1], hist_x / np.sum(hist_x), width=np.diff(bins_x), color='r', alpha=ALPHA)
        ax_y.bar(bins_y[:-1], hist_y / np.sum(hist_y), width=np.diff(bins_y), color='g', alpha=ALPHA)
        ax_z.bar(bins_z[:-1], hist_z / np.sum(hist_z), width=np.diff(bins_z), color='b', alpha=ALPHA)
        if hist_intensity is not None:
            ax_intensity.bar(bins_intensity[:-1], hist_intensity / np.sum(hist_intensity),
                             width=np.diff(bins_intensity), color='y', alpha=ALPHA)
        ax_num_points.bar(bins_num_points[:-1], hist_num_points / np.sum(hist_num_points),
                          width=np.diff(bins_num_points), color='y', alpha=ALPHA)
        ax_num_voxels.bar(bins_num_voxels[:-1], hist_num_voxels / np.sum(hist_num_voxels),
                          width=np.diff(bins_num_voxels), color='y', alpha=ALPHA)
        ax_num_points_in_voxel.bar(bins_num_points_in_voxel[:-1], hist_num_points_in_voxel / np.sum(hist_num_points_in_voxel),
                                   width=np.diff(bins_num_points_in_voxel), color='y', alpha=ALPHA)
        ax_dist.bar(bins_dist[:-1], hist_dist / np.sum(hist_dist), width=np.diff(bins_dist), color='y', alpha=ALPHA)

        num_frames = len(eval_dataset['loader']) if len(eval_dataset['loader']) < MAX_SAMPLE else MAX_SAMPLE

        ax_x_point_all.bar(bins_x[:-1], hist_x / np.sum(hist_x), width=np.diff(bins_x), alpha=ALPHA, label=dataset_name)
        ax_y_point_all.bar(bins_y[:-1], hist_y / np.sum(hist_y), width=np.diff(bins_y), alpha=ALPHA, label=dataset_name)
        ax_z_point_all.bar(bins_z[:-1], hist_z / np.sum(hist_z), width=np.diff(bins_z), alpha=ALPHA, label=dataset_name)
        if hist_intensity is not None:
            ax_intensity_point_all.bar(bins_intensity[:-1], hist_intensity / np.sum(hist_intensity),
                                       width=np.diff(bins_intensity), alpha=ALPHA, label=dataset_name)
        ax_num_points_point_all.bar(bins_num_points[:-1], hist_num_points / np.sum(hist_num_points),
                                    width=np.diff(bins_num_points), alpha=ALPHA, label=dataset_name)
        ax_num_voxels_point_all.bar(bins_num_voxels[:-1], hist_num_voxels / np.sum(hist_num_voxels),
                                    width=np.diff(bins_num_voxels), alpha=ALPHA, label=dataset_name)
        ax_num_points_in_voxel_point_all.bar(bins_num_points_in_voxel[:-1], hist_num_points_in_voxel / np.sum(hist_num_points_in_voxel),
                                             width=np.diff(bins_num_points_in_voxel), alpha=ALPHA, label=dataset_name)
        ax_dist_point_all.bar(bins_dist[:-1], hist_dist / num_frames,
                              width=np.diff(bins_dist), alpha=ALPHA, label=dataset_name)

        ax_x_car.bar(bins_x_car[:-1], hist_x_car / np.sum(hist_x_car),
                     width=np.diff(bins_x_car), alpha=ALPHA)
        ax_y_car.bar(bins_y_car[:-1], hist_y_car / np.sum(hist_y_car),
                     width=np.diff(bins_y_car), alpha=ALPHA)
        ax_z_car.bar(bins_z_car[:-1], hist_z_car / np.sum(hist_z_car),
                     width=np.diff(bins_z_car), alpha=ALPHA)
        ax_length_car.bar(bins_length_car[:-1], hist_length_car / np.sum(hist_length_car),
                          width=np.diff(bins_length_car), alpha=ALPHA)
        ax_width_car.bar(bins_width_car[:-1], hist_width_car / np.sum(hist_width_car),
                         width=np.diff(bins_width_car), alpha=ALPHA)
        ax_height_car.bar(bins_height_car[:-1], hist_height_car / np.sum(hist_height_car),
                          width=np.diff(bins_height_car), alpha=ALPHA)
        ax_dist_car.bar(bins_dist_car[:-1], hist_dist_car / np.sum(hist_dist_car),
                        width=np.diff(bins_dist_car), alpha=ALPHA)

        ax_x_gt_car_all.bar(bins_x_car[:-1], hist_x_car / np.sum(hist_x_car),
                            width=np.diff(bins_x_car), alpha=ALPHA, label=dataset_name)
        ax_y_gt_car_all.bar(bins_y_car[:-1], hist_y_car / np.sum(hist_y_car),
                            width=np.diff(bins_y_car), alpha=ALPHA, label=dataset_name)
        ax_z_gt_car_all.bar(bins_z_car[:-1], hist_z_car / np.sum(hist_z_car),
                            width=np.diff(bins_z_car), alpha=ALPHA, label=dataset_name)
        ax_length_gt_car_all.bar(bins_length_car[:-1], hist_length_car / np.sum(hist_length_car),
                                 width=np.diff(bins_length_car), alpha=ALPHA, label=dataset_name)
        ax_width_gt_car_all.bar(bins_width_car[:-1], hist_width_car / np.sum(hist_width_car),
                                width=np.diff(bins_width_car), alpha=ALPHA, label=dataset_name)
        ax_height_gt_car_all.bar(bins_height_car[:-1], hist_height_car / np.sum(hist_height_car),
                                 width=np.diff(bins_height_car), alpha=ALPHA, label=dataset_name)
        ax_dist_gt_car_all.bar(bins_dist_car[:-1], hist_dist_car / num_frames,
                               width=np.diff(bins_dist_car), alpha=ALPHA, label=dataset_name)

        ax_x_ped.bar(bins_x_ped[:-1], hist_x_ped / np.sum(hist_x_ped),
                     width=np.diff(bins_x_ped), alpha=ALPHA)
        ax_y_ped.bar(bins_y_ped[:-1], hist_y_ped / np.sum(hist_y_ped),
                     width=np.diff(bins_y_ped), alpha=ALPHA)
        ax_z_ped.bar(bins_z_ped[:-1], hist_z_ped / np.sum(hist_z_ped),
                     width=np.diff(bins_z_ped), alpha=ALPHA)
        ax_length_ped.bar(bins_length_ped[:-1], hist_length_ped / np.sum(hist_length_ped),
                          width=np.diff(bins_length_ped), alpha=ALPHA)
        ax_width_ped.bar(bins_width_ped[:-1], hist_width_ped / np.sum(hist_width_ped),
                         width=np.diff(bins_width_ped), alpha=ALPHA)
        ax_height_ped.bar(bins_height_ped[:-1], hist_height_ped / np.sum(hist_height_ped),
                          width=np.diff(bins_height_ped), alpha=ALPHA)

        ax_x_gt_ped_all.bar(bins_x_ped[:-1], hist_x_ped / np.sum(hist_x_ped),
                            width=np.diff(bins_x_ped), alpha=ALPHA, label=dataset_name)
        ax_y_gt_ped_all.bar(bins_y_ped[:-1], hist_y_ped / np.sum(hist_y_ped),
                            width=np.diff(bins_y_ped), alpha=ALPHA, label=dataset_name)
        ax_z_gt_ped_all.bar(bins_z_ped[:-1], hist_z_ped / np.sum(hist_z_ped),
                            width=np.diff(bins_z_ped), alpha=ALPHA, label=dataset_name)
        ax_length_gt_ped_all.bar(bins_length_ped[:-1], hist_length_ped / np.sum(hist_length_ped),
                                 width=np.diff(bins_length_ped), alpha=ALPHA, label=dataset_name)
        ax_width_gt_ped_all.bar(bins_width_ped[:-1], hist_width_ped / np.sum(hist_width_ped),
                                width=np.diff(bins_width_ped), alpha=ALPHA, label=dataset_name)
        ax_height_gt_ped_all.bar(bins_height_ped[:-1], hist_height_ped / np.sum(hist_height_ped),
                                 width=np.diff(bins_height_ped), alpha=ALPHA, label=dataset_name)
        if model:
            ax_x_car_pred.bar(bins_x_car_pred[:-1], hist_x_car_pred / np.sum(hist_x_car_pred),
                              width=np.diff(bins_x_car_pred), color='r', alpha=ALPHA)
            ax_y_car_pred.bar(bins_y_car_pred[:-1], hist_y_car_pred / np.sum(hist_y_car_pred),
                              width=np.diff(bins_y_car_pred), color='g', alpha=ALPHA)
            ax_z_car_pred.bar(bins_z_car_pred[:-1], hist_z_car_pred / np.sum(hist_z_car_pred),
                              width=np.diff(bins_z_car_pred), color='b', alpha=ALPHA)
            ax_length_car_pred.bar(bins_length_car_pred[:-1], hist_length_car_pred / np.sum(hist_length_car_pred),
                                   width=np.diff(bins_length_car_pred), color='y', alpha=ALPHA)
            ax_width_car_pred.bar(bins_width_car_pred[:-1], hist_width_car_pred / np.sum(hist_width_car_pred),
                                  width=np.diff(bins_width_car_pred), color='y', alpha=ALPHA)
            ax_height_car_pred.bar(bins_height_car_pred[:-1], hist_height_car_pred / np.sum(hist_height_car_pred),
                                   width=np.diff(bins_height_car_pred), color='y', alpha=ALPHA)

            ax_x_pred_car_all.bar(bins_x_car_pred[:-1], hist_x_car_pred / np.sum(hist_x_car_pred),
                                  width=np.diff(bins_x_car_pred), alpha=ALPHA, label=dataset_name)
            ax_y_pred_car_all.bar(bins_y_car_pred[:-1], hist_y_car_pred / np.sum(hist_y_car_pred),
                                  width=np.diff(bins_y_car_pred), alpha=ALPHA, label=dataset_name)
            ax_z_pred_car_all.bar(bins_z_car_pred[:-1], hist_z_car_pred / np.sum(hist_z_car_pred),
                                  width=np.diff(bins_z_car_pred), alpha=ALPHA, label=dataset_name)
            ax_length_pred_car_all.bar(bins_length_car_pred[:-1], hist_length_car_pred / np.sum(hist_length_car_pred),
                                       width=np.diff(bins_length_car_pred), alpha=ALPHA, label=dataset_name)
            ax_width_pred_car_all.bar(bins_width_car_pred[:-1], hist_width_car_pred / np.sum(hist_width_car_pred),
                                      width=np.diff(bins_width_car_pred), alpha=ALPHA, label=dataset_name)
            ax_height_pred_car_all.bar(bins_height_car_pred[:-1], hist_height_car_pred / np.sum(hist_height_car_pred),
                                       width=np.diff(bins_height_car_pred), alpha=ALPHA, label=dataset_name)

            ax_x_ped_pred.bar(bins_x_ped_pred[:-1], hist_x_ped_pred / np.sum(hist_x_ped_pred),
                              width=np.diff(bins_x_ped_pred), color='r', alpha=ALPHA)
            ax_y_ped_pred.bar(bins_y_ped_pred[:-1], hist_y_ped_pred / np.sum(hist_y_ped_pred),
                              width=np.diff(bins_y_ped_pred), color='g', alpha=ALPHA)
            ax_z_ped_pred.bar(bins_z_ped_pred[:-1], hist_z_ped_pred / np.sum(hist_z_ped_pred),
                              width=np.diff(bins_z_ped_pred), color='b', alpha=ALPHA)
            ax_length_ped_pred.bar(bins_length_ped_pred[:-1], hist_length_ped_pred / np.sum(hist_length_ped_pred),
                                   width=np.diff(bins_length_ped_pred), color='y', alpha=ALPHA)
            ax_width_ped_pred.bar(bins_width_ped_pred[:-1], hist_width_ped_pred / np.sum(hist_width_ped_pred),
                                  width=np.diff(bins_width_ped_pred), color='y', alpha=ALPHA)
            ax_height_ped_pred.bar(bins_height_ped_pred[:-1], hist_height_ped_pred / np.sum(hist_height_ped_pred),
                                   width=np.diff(bins_height_ped_pred), color='y', alpha=ALPHA)

            ax_x_pred_ped_all.bar(bins_x_ped_pred[:-1], hist_x_ped_pred / np.sum(hist_x_ped_pred),
                                  width=np.diff(bins_x_ped_pred), alpha=ALPHA, label=dataset_name)
            ax_y_pred_ped_all.bar(bins_y_ped_pred[:-1], hist_y_ped_pred / np.sum(hist_y_ped_pred),
                                  width=np.diff(bins_y_ped_pred), alpha=ALPHA, label=dataset_name)
            ax_z_pred_ped_all.bar(bins_z_ped_pred[:-1], hist_z_ped_pred / np.sum(hist_z_ped_pred),
                                  width=np.diff(bins_z_ped_pred), alpha=ALPHA, label=dataset_name)
            ax_length_pred_ped_all.bar(bins_length_ped_pred[:-1], hist_length_ped_pred / np.sum(hist_length_ped_pred),
                                       width=np.diff(bins_length_ped_pred), alpha=ALPHA, label=dataset_name)
            ax_width_pred_ped_all.bar(bins_width_ped_pred[:-1], hist_width_ped_pred / np.sum(hist_width_ped_pred),
                                      width=np.diff(bins_width_ped_pred), alpha=ALPHA, label=dataset_name)
            ax_height_pred_ped_all.bar(bins_height_ped_pred[:-1], hist_height_ped_pred / np.sum(hist_height_ped_pred),
                                       width=np.diff(bins_height_ped_pred), alpha=ALPHA, label=dataset_name)

        filename = os.path.join(args.out_dir, f'0_point_hist_{dataset_name}{args.figure_suffix}.png')
        fig.savefig(filename)
        wandb.save(filename)
        wandb.log({f'val/{dataset_name}/Point histogram': wandb.Image(filename)})
        filename_car = os.path.join(args.out_dir, f'1_gt_car_hist_{dataset_name}{args.figure_suffix}.png')
        fig_car.savefig(filename_car)
        wandb.save(filename_car)
        wandb.log({f'val/{dataset_name}/GT car histogram': wandb.Image(filename_car)})
        filename_ped = os.path.join(args.out_dir, f'2_gt_pedestrian_hist_{dataset_name}{args.figure_suffix}.png')
        fig_ped.savefig(filename_ped)
        wandb.save(filename_ped)
        wandb.log({f'val/{dataset_name}/GT pedestrian histogram': wandb.Image(filename_ped)})
        if model:
            filename_car_pred = os.path.join(args.out_dir, f'3_pred_car_hist_{dataset_name}{args.figure_suffix}.png')
            fig_car_pred.savefig(filename_car_pred)
            wandb.save(filename_car_pred)
            wandb.log({f'val/{dataset_name}/Pred car histogram': wandb.Image(filename_car_pred)})
            filename_ped_pred = os.path.join(
                args.out_dir, f'4_pred_pedestrian_hist_{dataset_name}{args.figure_suffix}.png')
            fig_ped_pred.savefig(filename_ped_pred)
            wandb.save(filename_ped_pred)
            wandb.log({f'val/{dataset_name}/Pred pedestrian histogram': wandb.Image(filename_ped_pred)})

    filename_point = os.path.join(args.out_dir, f'0_point_hist_all{args.figure_suffix}.png')
    fig_point_all.legend()
    fig_point_all.savefig(filename_point)
    wandb.save(filename_point)
    wandb.log({f'val/Point histogram': wandb.Image(filename_point)})

    filename_gt_car = os.path.join(args.out_dir, f'1_gt_car_hist_all{args.figure_suffix}.png')
    fig_gt_car_all.legend()
    fig_gt_car_all.savefig(filename_gt_car)
    wandb.save(filename_gt_car)
    wandb.log({f'val/GT car histogram': wandb.Image(filename_gt_car)})

    filename_gt_ped = os.path.join(args.out_dir, f'2_gt_pedestrian_hist_all{args.figure_suffix}.png')
    fig_gt_ped_all.legend()
    fig_gt_ped_all.savefig(filename_gt_ped)
    wandb.save(filename_gt_ped)
    wandb.log({f'val/GT pedestrian histogram': wandb.Image(filename_gt_ped)})

    if model:
        filename_pred_car = os.path.join(args.out_dir, f'3_pred_car_hist_all{args.figure_suffix}.png')
        fig_pred_car_all.legend()
        fig_pred_car_all.savefig(filename_pred_car)
        wandb.save(filename_pred_car)
        wandb.log({f'val/Pred car histogram': wandb.Image(filename_pred_car)})
        filename_pred_ped = os.path.join(args.out_dir, f'4_pred_pedestrian_hist_all{args.figure_suffix}.png')
        fig_pred_ped_all.legend()
        fig_pred_ped_all.savefig(filename_pred_ped)
        wandb.save(filename_pred_ped)
        wandb.log({f'val/Pred pedestrian histogram': wandb.Image(filename_pred_ped)})

    fig_copies = []
    fig_copies.append([ax_x_point_all, f'100_point_x_hist_all{args.figure_suffix}.png', 'val/point x histogram'])
    fig_copies.append([ax_y_point_all, f'100_point_y_hist_all{args.figure_suffix}.png', 'val/point y histogram'])
    fig_copies.append([ax_z_point_all, f'100_point_z_hist_all{args.figure_suffix}.png', 'val/point z histogram'])
    fig_copies.append([ax_intensity_point_all, f'100_point_intensity_hist_all{args.figure_suffix}.png',
                      'val/point intensity histogram'])
    fig_copies.append([ax_dist_point_all, f'100_point_dist_hist_all{args.figure_suffix}.png',
                      'val/point dist histogram'])
    fig_copies.append([ax_x_gt_car_all, f'100_gt_car_x_hist_all{args.figure_suffix}.png', 'val/GT car x histogram'])
    fig_copies.append([ax_y_gt_car_all, f'100_gt_car_y_hist_all{args.figure_suffix}.png', 'val/GT car y histogram'])
    fig_copies.append([ax_z_gt_car_all, f'100_gt_car_z_hist_all{args.figure_suffix}.png', 'val/GT car z histogram'])
    fig_copies.append([ax_length_gt_car_all, f'100_gt_car_length_hist_all{args.figure_suffix}.png',
                      'val/GT car length histogram'])
    fig_copies.append([ax_width_gt_car_all, f'100_gt_car_width_hist_all{args.figure_suffix}.png',
                      'val/GT car width histogram'])
    fig_copies.append([ax_height_gt_car_all, f'100_gt_car_height_hist_all{args.figure_suffix}.png',
                      'val/GT car height histogram'])
    fig_copies.append([ax_dist_gt_car_all, f'100_gt_car_dist_hist_all{args.figure_suffix}.png',
                      'val/GT car dist histogram'])
    for ax, filename, title in fig_copies:
        path = os.path.join(args.out_dir, filename)
        copy_bar_plot(ax, path)
        wandb.save(path)
        wandb.log({title: wandb.Image(path)})

    logger.info('Dataset analysis done.')

    wandb.finish()


if __name__ == '__main__':
    main()
