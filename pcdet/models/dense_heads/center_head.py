import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ..model_utils import dann_utils
from ...utils import loss_utils


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

        # In-head DANN, ported from AnchorHeadMulti (pcdet/models/model_utils/dann_utils.py). Built
        # ONLY when LOSS_WEIGHTS carries 'dann_weight', so every existing CenterPoint config keeps
        # its parameter set and its checkpoints round-trip unchanged. Taps `spatial_features_2d`
        # (the 2D backbone output, before shared_conv), which is where the UADA3D marginal
        # discriminator also reads, and where AnchorHeadMulti reads when it has no shared conv.
        self.domain_discriminator = dann_utils.build_domain_discriminator(
            self.model_cfg.get('LOSS_CONFIG', None), in_channels=input_channels
        )

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            ignore_mask_list, any_ignored = [], False
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                # A NEGATIVE label is an ignore region, never a positive: the pseudo-label ignore band
                # (-1..-C, self_training_utils) and IGNORE_CLASS_LABEL (source classes outside CLASS_NAMES).
                # Indexing all_names with it directly wrapped around - -1 Car became 'Cyclist', -3 Cyc
                # became 'Car' - so until 2026-10-06 every ignore-band pseudo-label trained as a positive of
                # a permuted class. Clipping sends it to 'bg', which no head claims; label 0 is padding.
                cur_labels = cur_gt_boxes[:, -1].cpu().long().numpy()
                gt_class_names = all_names[np.clip(cur_labels, 0, None)]
                ignore_boxes = cur_gt_boxes[cur_gt_boxes[:, -1] < 0]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                ignore_mask = self.ignore_region_mask(
                    ignore_boxes, heatmap, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                any_ignored = any_ignored or bool((ignore_mask == 0).any())
                ignore_mask_list.append(ignore_mask.to(gt_boxes_single_head.device))
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            # None when nothing is ignored, so a batch without ignore regions computes exactly the loss
            # it always did (the focal loss takes mask=None).
            ret_dict['heatmap_masks'].append(torch.stack(ignore_mask_list, dim=0) if any_ignored else None)
        return ret_dict

    def ignore_region_mask(self, ignore_boxes, heatmap, feature_map_stride, gaussian_overlap=0.1, min_radius=2):
        """Per-pixel heatmap-loss weight for one sample: 0 inside an ignore region, 1 elsewhere.

        KITTI-DontCare-style ignore. An ignore box's footprint is the union of
          - every BEV pixel whose centre lies inside the rotated box, and
          - the square a positive of that size would draw its gaussian over (same radius rule as
            assign_target_of_single_head), so a small box still covers what a positive would have.
        The weight is zero in EVERY class channel there, so the box is neither a positive nor a negative
        of any class. A pixel holding a positive's peak (heatmap == 1 in any channel) is never ignored, so
        an ignore box overlapping a real object cannot delete that object.

        Vectorised over boxes on the boxes' device (the GPU in training): a pseudo-labelled frame can carry
        ~100 ignore boxes, and a per-box Python loop would cost a large share of an iteration.

        Args:
            ignore_boxes: (K, 8) [x, y, z, dx, dy, dz, heading, label<0], any device
            heatmap: (C, H, W) the sample's positive heatmap for this head
        Returns:
            (H, W) float tensor on ignore_boxes' device
        """
        num_classes, height, width = heatmap.shape
        device = ignore_boxes.device
        keep = torch.ones((height, width), dtype=torch.float32, device=device)
        boxes = ignore_boxes[(ignore_boxes[:, 3] > 0) & (ignore_boxes[:, 4] > 0)].float()
        if boxes.shape[0] == 0:
            return keep
        cell_x = float(self.voxel_size[0] * feature_map_stride)
        cell_y = float(self.voxel_size[1] * feature_map_stride)
        x0, y0 = float(self.point_cloud_range[0]), float(self.point_cloud_range[1])
        ix = torch.arange(width, device=device, dtype=torch.float32)
        iy = torch.arange(height, device=device, dtype=torch.float32)
        px, py = x0 + (ix + 0.5) * cell_x, y0 + (iy + 0.5) * cell_y   # pixel centres in metres
        radius = centernet_utils.gaussian_radius(boxes[:, 3] / cell_x, boxes[:, 4] / cell_y,
                                                 min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius).float()
        gx = torch.clamp((boxes[:, 0] - x0) / cell_x, min=0, max=width - 0.5).int().float()
        gy = torch.clamp((boxes[:, 1] - y0) / cell_y, min=0, max=height - 0.5).int().float()
        ignored = torch.zeros((height, width), dtype=torch.bool, device=device)
        for s in range(0, boxes.shape[0], 32):   # bounds the (k, H, W) temporaries
            b = boxes[s:s + 32]
            ox = px[None, None, :] - b[:, 0, None, None]
            oy = py[None, :, None] - b[:, 1, None, None]
            cos_h, sin_h = torch.cos(b[:, 6])[:, None, None], torch.sin(b[:, 6])[:, None, None]
            inside = (((ox * cos_h + oy * sin_h).abs() <= b[:, 3, None, None] / 2)
                      & ((-ox * sin_h + oy * cos_h).abs() <= b[:, 4, None, None] / 2))
            r = radius[s:s + 32, None, None]
            square = (((ix[None, None, :] - gx[s:s + 32, None, None]).abs() <= r)
                      & ((iy[None, :, None] - gy[s:s + 32, None, None]).abs() <= r))
            ignored |= (inside | square).any(dim=0)
        keep[ignored] = 0
        keep[heatmap.eq(1).any(dim=0).to(device)] = 1
        return keep

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(
                pred_dict['hm'], target_dicts['heatmaps'][idx], mask=target_dicts['heatmap_masks'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def get_domain_adversarial_loss(self):
        """The DANN term, or (None, {}) when this head has no discriminator or the batch carried
        no `domain_label` (plain train.py never stamps one; only train_st_utils does)."""
        if self.domain_discriminator is None or 'domain_preds' not in self.forward_ret_dict:
            return None, {}
        loss, tb_dict = dann_utils.domain_adversarial_loss(
            self.forward_ret_dict['domain_preds'], self.forward_ret_dict['domain_label'],
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dann_weight']
        )
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        # Domain discriminator, mirroring anchor_head_multi.py. The stale entries are cleared
        # first so an eval-mode or plain-training forward (no `domain_label`) cannot serve a
        # previous batch's predictions to get_domain_adversarial_loss().
        self.forward_ret_dict.pop('domain_preds', None)
        self.forward_ret_dict.pop('domain_label', None)
        if self.domain_discriminator is not None and 'domain_label' in data_dict:
            self.forward_ret_dict['domain_label'] = data_dict['domain_label']
            self.forward_ret_dict['domain_preds'] = dann_utils.domain_predictions(
                self.domain_discriminator, spatial_features_2d
            )

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        data_dict['batch_box_preds'] = torch.cat((pred_dicts[0]['center'], pred_dicts[0]['center_z'],
                                                 pred_dicts[0]['dim'], pred_dicts[0]['rot']), 1)
        data_dict['batch_cls_preds'] = pred_dicts[0]['hm']

        # --- BEGIN new: conditional-discriminator spatial-map export (opt-in, UADA3D port) ---
        if self.training and self.model_cfg.get('EXPORT_SPATIAL_PREDS', False):
            data_dict['cls_preds_spatial'] = torch.clamp(
                pred_dicts[0]['hm'].sigmoid(), min=1e-4, max=1 - 1e-4
            )
            data_dict['box_preds_spatial'] = data_dict['batch_box_preds']
        # --- END new ---

        if 'domain' in data_dict:
            # Skip Gaussian target assignment entirely for the (unlabeled) target-domain batch —
            # its detection loss is discarded anyway by DACenterPoint.get_training_loss().
            if self.training and data_dict['domain'] != 1:
                target_dict = self.assign_targets(
                    data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                    feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
                )
                self.forward_ret_dict['target_dicts'] = target_dict
        else:
            if self.training:
                target_dict = self.assign_targets(
                    data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                    feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
                )
                self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict
