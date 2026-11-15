"""Sparse UNet on spconv: Pointcept's SpUNet-v1m1 (MIT, Copyright (c) 2023 Pointcept; author Xiaoyang Wu;
pointcept/models/sparse_unet/spconv_unet_v1m1_base.py at 77f87ab), ported without Pointcept's registry, without the
encoder-only mode (its torch_geometric scatter) and with torch's trunc_normal_ in place of timm's. Layer structure,
normalisation, initialisation and the forward pass are unchanged; tests/test_pcseg_core.py pins the parameter count.
"""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import spconv.pytorch as spconv


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, in_channels, embed_channels, stride=1, norm_fn=None, indice_key=None, bias=False):
        super().__init__()
        assert norm_fn is not None
        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, embed_channels, kernel_size=1, bias=False), norm_fn(embed_channels))
        self.conv1 = spconv.SubMConv3d(in_channels, embed_channels, kernel_size=3, stride=stride, padding=1,
                                       bias=bias, indice_key=indice_key)
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(embed_channels, embed_channels, kernel_size=3, stride=stride, padding=1,
                                       bias=bias, indice_key=indice_key)
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))
        return out


def offset2batch(offset):
    """Cumulative counts [B] -> batch index per row."""
    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    return torch.repeat_interleave(torch.arange(len(offset), device=offset.device), counts)


class SpUNet(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=32, channels=(32, 64, 128, 256, 256, 128, 96, 96),
                 layers=(2, 3, 4, 6, 2, 2, 2, 2)):
        super().__init__()
        assert len(layers) % 2 == 0 and len(layers) == len(channels)
        self.num_stages = len(layers) // 2
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base_channels, kernel_size=5, padding=1, bias=False, indice_key='stem'),
            norm_fn(base_channels), nn.ReLU())

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down, self.up, self.enc, self.dec = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for s in range(self.num_stages):
            self.down.append(spconv.SparseSequential(
                spconv.SparseConv3d(enc_channels, channels[s], kernel_size=2, stride=2, bias=False,
                                    indice_key='spconv%d' % (s + 1)),
                norm_fn(channels[s]), nn.ReLU()))
            self.enc.append(spconv.SparseSequential(OrderedDict([
                ('block%d' % i, block(channels[s], channels[s], norm_fn=norm_fn, indice_key='subm%d' % (s + 1)))
                for i in range(layers[s])])))
            self.up.append(spconv.SparseSequential(
                spconv.SparseInverseConv3d(channels[len(channels) - s - 2], dec_channels, kernel_size=2, bias=False,
                                           indice_key='spconv%d' % (s + 1)),
                norm_fn(dec_channels), nn.ReLU()))
            self.dec.append(spconv.SparseSequential(OrderedDict([
                ('block%d' % i, block(dec_channels + enc_channels if i == 0 else dec_channels, dec_channels,
                                      norm_fn=norm_fn, indice_key='subm%d' % s))
                for i in range(layers[len(channels) - s - 1])])))
            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        self.final = (spconv.SubMConv3d(channels[-1], num_classes, kernel_size=1, padding=1, bias=True)
                      if num_classes > 0 else spconv.Identity())
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, batch):
        grid_coord, feat, offset = batch['grid_coord'], batch['feat'], batch['offset']
        b = offset2batch(offset)
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat, indices=torch.cat([b.unsqueeze(-1).int(), grid_coord.int()], dim=1).contiguous(),
            spatial_shape=sparse_shape, batch_size=int(b[-1]) + 1)
        x = self.conv_input(x)
        skips = [x]
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            skip = skips.pop(-1)
            x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
            x = self.dec[s](x)
        x = self.final(x)
        return x.features
