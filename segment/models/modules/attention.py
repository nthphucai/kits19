import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import diff
from torch.nn.modules.batchnorm import BatchNorm3d
from torch.nn.modules.pooling import MaxPool3d


# ===============================================================================================================
# Module name: CBAM (SPATIAL ATTENTION + CHANNEL ATTENTION)
# Soure      : https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CBAM.py
# ===============================================================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=1 / 16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()

        bottleneck = int(np.round(in_channels * ratio))
        self.se = nn.Sequential(
            nn.Linear(in_channels, bottleneck, bias=False),
            nn.ReLU(),
            nn.Linear(bottleneck, in_channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        avg_pool = self.avgpool(x).squeeze()
        max_pool = self.maxpool(x).squeeze()
        avg_channel = self.se(avg_pool)
        max_channel = self.se(max_pool)
        output = self.sigmoid(avg_channel + max_channel)
        output = output.view(b, c, 1, 1, 1)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
            ),  # if kernel_size=3, padding=1
            # nn.Conv3d(2, 1, kernel_size=kernel_size, stride=1, padding=1),
            # nn.BatchNorm3d(1),
            nn.InstanceNorm3d(1),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        max_spat, _ = torch.max(x, dim=1, keepdim=True)
        mean_spat = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_spat, mean_spat], dim=1)
        output = self.sigmoid(self.conv(result))
        return output


class CBAM3D(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, inpt):
        residual = inpt
        output = inpt * self.channel_att(inpt)
        output = output * self.spatial_att(output)
        return output + residual


# ===============================================================================================================
#                                          SEAttention - SQUEEZE AND EXCITATION
# ===============================================================================================================
class SEAttention(nn.Module):
    def __init__(self, in_channels, ratio=1 / 16, rank=3):
        super().__init__()
        self.sqz = nn.AdaptiveAvgPool2d(1) if rank == 2 else nn.AdaptiveAvgPool3d(1)
        bottleneck = int(np.round(in_channels * ratio))
        self.exc = nn.Sequential(
            nn.Linear(in_channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = [x.shape[0], x.shape[1]]
        att = self.exc(self.sqz(x).squeeze())  # b,c
        att = att.view(b, c, *[1] * len(x.shape[2:]))
        output = att * x

        return output


# ===============================================================================================================
#                                          SAttention
# ===============================================================================================================
import torch.nn.functional as func


class SAModule(nn.Module):
    def __init__(self, num_channels, rank=2):
        super(SAModule, self).__init__()
        self.num_channels = num_channels

        conv = nn.Conv2d if rank == 2 else nn.Conv3d

        self.conv1 = conv(
            in_channels=num_channels, out_channels=num_channels // 8, kernel_size=1
        )
        self.conv2 = conv(
            in_channels=num_channels, out_channels=num_channels // 8, kernel_size=1
        )
        self.conv3 = conv(
            in_channels=num_channels, out_channels=num_channels, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat_map):
        # batch_size, num_channels, height, width = feat_map.size()

        # B x C x spatial
        # B x spatial x C
        conv1_proj = self.conv1(feat_map).flatten(start_dim=2)
        conv1_proj = conv1_proj.permute(0, 2, 1)

        # B x C x spatial
        conv2_proj = self.conv2(feat_map).flatten(start_dim=2)

        # B x spatial x spatial
        relation_map = torch.bmm(conv1_proj, conv2_proj)
        attention = func.softmax(relation_map, dim=-1)

        # B x C x spatial
        conv3_proj = self.conv3(feat_map).flatten(start_dim=2)

        # B x C x spatial
        feat_refine = torch.bmm(conv3_proj, attention.permute(0, 2, 1))
        feat_refine = feat_refine.view(*feat_map.shape)

        feat_map = self.gamma * feat_refine + feat_map

        return feat_map


class GlobalAverage(nn.Module):
    def __init__(self, dims=(-1, -2, -3), keepdims=False):
        """
        :param dims: dimensions of inputs
        :param keepdims: whether to preserve shape after averaging
        """
        super().__init__()
        self.dims = dims
        self.keepdims = keepdims

    def forward(self, x):
        return torch.mean(x, dim=self.dims, keepdim=self.keepdims)

    def extra_repr(self):
        return f"dims={self.dims}, keepdims={self.keepdims}"
