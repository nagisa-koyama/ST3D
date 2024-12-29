import _init_path
import os
import torch
import time
import argparse
import wandb
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    parser.add_argument('--batch_size', type=int, default=20, required=False, help='batch size for training')
    parser.add_argument('--out_dir', type=str,
                        default='/storage', help='specify the output directory')
    parser.add_argument('--out_filename', type=str,
                        default='tsne.png', help='specify the output filename')
    parser.add_argument('--run_name', type=str, default=None, help='run name for wandb')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Run TSNE-------------------------')

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

    eval_dataset_rep = eval_datasets[0]
    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=eval_dataset_rep['dataset_class'])
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    logger.info("Model loaded")

    model_tsne = TSNE(n_components=2)

    features = None
    labels = []
    feature_extraction_start = time.time()
    for eval_dataset in eval_datasets:
        print("eval_dataset onotology:", eval_dataset['loader'].dataset.dataset_ontology)
        for idx, data_dict in enumerate(eval_dataset['loader']):
            load_data_to_gpu(data_dict)
            load_data_to_gpu_duration = time.time()
            with torch.no_grad():
                pred_dicts, ret_dict = model(data_dict)
            batched_features = data_dict['spatial_features_2d'].flatten(start_dim=1).cpu().numpy()
            if features is None:
                features = batched_features
            else:
                features = np.concatenate((features, batched_features), axis=0)
            dataset_name = eval_dataset['loader'].dataset.dataset_ontology
            labels.extend([dataset_name] * args.batch_size)
            if idx * args.batch_size >= 1000:
                print("Breaking after 1000 samples")
                break
    feature_extraction_end = time.time()
    print("Feature extraction duration[s]:", feature_extraction_end - feature_extraction_start)
    tsne = model_tsne.fit_transform(features)
    print(tsne)
    tsne_duration = time.time()
    print("tsne duration [s]:", time.time() - feature_extraction_end)

    logger.info('TSNE done.')

    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))
        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)
        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # tx = scale_to_01_range(tx)
    # ty = scale_to_01_range(ty)

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    unique_labels = set(labels)
    for label in unique_labels:
        idx = [i for i, x in enumerate(labels) if x == label]
        ax.scatter(tx[idx], ty[idx], label=label)
    ax.legend(loc='best')

    # finally, show the plot
    filename = os.path.join(args.out_dir, args.out_filename)
    plt.savefig(filename)
    wandb.save(filename)
    wandb.log({'tsne_scatter': wandb.Image(filename)})
    table = wandb.Table(columns=["x", "y", "label"], data=[(x, y, l) for x, y, l in zip(tx, ty, labels)])
    wandb.log({'tsne': table})

    wandb.finish()


if __name__ == '__main__':
    main()
