"""IA-SSD (Zhang et al., CVPR 2022), ported from UADA3D (5de3c16) for the point-based control of the two-term
BatchNorm analysis (experiments_md/20261006_01). Two departures from UADA3D's file, both for ST3D's contracts:

* eval returns (pred_dicts, recall_dicts), as every ST3D detector does - tools/eval_utils unpacks two values;
* the backbone is built with num_class, which IASSD_Backbone requires and ST3D's shared
  Detector3DTemplate.build_backbone_3d does not pass. Overridden here so the shared template stays untouched.
"""
from .detector3d_template import Detector3DTemplate
from .. import backbones_3d


class IASSD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict
        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            num_class=self.num_class,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = getattr(backbone_3d_module, 'backbone_channels', None)
        return backbone_3d_module, model_info_dict

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()

        loss = loss_point
        return loss, tb_dict, disp_dict
