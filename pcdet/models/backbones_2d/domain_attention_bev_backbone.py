import numpy as np
import torch
import torch.nn as nn

from ...ops.domain_attention.faster_rcnn.DAResNet import DAResNet, DABasicBlock
from ..backbones_2d.base_bev_backbone import BaseBEVBackbone


class DomainAttentionBEVBackbone(BaseBEVBackbone):
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg, input_channels)

        self.da_block = DABasicBlock(512, 512)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        x = self.da_block(x)

        data_dict['spatial_features_2d'] = x
        print("x.shape:", x.shape)

        return data_dict
