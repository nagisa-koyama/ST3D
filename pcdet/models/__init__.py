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
        # AFFINE_PER_KNOT (item 71 = item 67 on the real pair; absent = False = unchanged): gamma / beta per knot too.
        apk = bool(model_cfg.RANGE_BN.get('AFFINE_PER_KNOT', False))
        model.n_range_bn = install_range_bn(model, model_cfg.RANGE_BN.KNOTS, list(dataset.point_cloud_range),
                                            thin=model_cfg.RANGE_BN.get('THIN', 32), affine_per_knot=apk)
        model.range_bn_affine_per_knot = apk
    # MODEL.COUNT_NORM_CONV (ANALYSIS, experiments_md 20261003_05 §5 item 64; absent = unchanged): every sparse convolution of
    # the 3D backbone normalized by its window's occupied-site count (pcdet/models/model_utils/count_norm_conv.py).
    if model_cfg.get('COUNT_NORM_CONV', None):
        from .model_utils.count_norm_conv import install_count_norm
        model.n_count_norm = install_count_norm(model)
    # MODEL.NORM_SWAP (experiments_md 20261003_05 §5 item 68; absent = unchanged): every BatchNorm replaced by per-element
    # LayerNorm or per-frame GroupNorm (pcdet/models/model_utils/norm_swap.py), before any checkpoint is loaded.
    if model_cfg.get('NORM_SWAP', None):
        from .model_utils.norm_swap import install_norm_swap
        model.n_norm_swap = install_norm_swap(model, model_cfg.NORM_SWAP)
        print('[norm_swap] %s: %d BatchNorm layers replaced' % (dict(model_cfg.NORM_SWAP), model.n_norm_swap), flush=True)
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
