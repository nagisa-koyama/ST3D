from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector


def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    # MODEL.RANGE_BN (ANALYSIS, experiments_md 20261003_05 §5 item 60; absent = unchanged): every BatchNorm normalizes with
    # per-range statistics (pcdet/models/model_utils/range_bn.py). Installed here so training, in-run evaluation and every
    # eval script build the same layers before a checkpoint is loaded.
    if model_cfg.get('RANGE_BN', None):
        from .model_utils.range_bn import install_range_bn
        model.n_range_bn = install_range_bn(model, model_cfg.RANGE_BN.KNOTS, list(dataset.point_cloud_range),
                                            thin=model_cfg.RANGE_BN.get('THIN', 32))
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape', 'gt_names', 'gt_classes']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict', 'dann_loss'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        dann_loss = None
        if 'dann_loss' in ret_dict.keys() and ret_dict['dann_loss']:
            dann_loss = ret_dict['dann_loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict, dann_loss)

    return model_func
