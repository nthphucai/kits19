from pickle import FALSE

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment.models.modules.attention import CBAM3D, SEAttention


# ===============================================================================================================
#                                          BASIC UNET MODULES
# ===============================================================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.doubleconv(x)
        return x


# class Uppad(nn.Module):
#     """
#     change shape: x1.size = x2.size
#     This because the original paper encourage down-sampling without padding the same as the up-sampling without zero-padding, which can avoid corrupting semantic information.
#     This is the one of the reason for which the overlap-tile strategy was proposed
#     """
#     def __init__(self, x1, x2):
#         super().__init__()
#         self.x1 = x1
#         self.x2 = x2

#     def pad(self):
#         diff_Z = self.x2.size()[2] - self.x1.size()[2]
#         diff_Y = self.x2.size()[3] - self.x1.size()[3]
#         diff_X = self.x2.size()[4] - self.x1.size()[4]
#         pad = [diff_X//2, diff_X - diff_X//2, diff_Y//2, diff_Y - diff_Y//2, diff_Z//2, diff_Z - diff_Z//2]
#         x1 = F.pad(self.x1, pad)
#         return x1

# class UpSample(nn.Sequential):
#     """
#     some block of pretrained : they just have flow: Conv --> batchnorm --conv --batchnorm, did not have relu...
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, act=False):
#         super().__init__()
#         self.up   = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.norm = nn.BatchNorm3d(out_channels)
#         if act:
#           self.relu = nn.ReLU(nn.ReLU(inplace=True))

""" Batchnorm
"""


class Conv_BN_ReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm3d(out_channels, affine=False)

        if act == True:
            self.act = nn.ReLU(inplace=True)


class BN_Conv_ReLU(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, act=True
    ):
        super().__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        if act:
            self.act = nn.ReLU()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )


""" Instancenorm
"""


class Conv_IN_ReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=False)

        if act == True:
            self.act = nn.ReLU(inplace=True)


class IN_Conv_ReLU(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, act=True
    ):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channels)
        if act:
            self.act = nn.ReLU()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )


"""
change shape: x1.size = x2.size
This because the original paper encourage down-sampling without padding the same as the up-sampling without zero-padding, which can avoid corrupting semantic information. 
This is the one of the reason for which the overlap-tile strategy was proposed
"""


class Uppad(nn.Module):
    def __init__(self, x1, x2):
        super().__init__()
        self.x1 = x1
        self.x2 = x2

    def pad(self):
        diff_Z = self.x2.size()[2] - self.x1.size()[2]
        diff_Y = self.x2.size()[3] - self.x1.size()[3]
        diff_X = self.x2.size()[4] - self.x1.size()[4]
        pad = [
            diff_X // 2,
            diff_X - diff_X // 2,
            diff_Y // 2,
            diff_Y - diff_Y // 2,
            diff_Z // 2,
            diff_Z - diff_Z // 2,
        ]
        x1 = F.pad(self.x1, pad)
        return x1


"""
some block of pretrained : they just have flow: Conv --> batchnorm --conv --batchnorm, did not have relu...
"""


class UpSample(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, act=False
    ):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # self.norm = nn.BatchNorm3d(out_channels)
        self.norm = nn.InstanceNorm3d(out_channels)

        if act:
            self.relu = nn.ReLU(nn.ReLU(inplace=True))


class EncoderBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2), DoubleConv(in_channels, out_channels)
        )


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, encoder_channel, kernel_size=2, padding=0, stride=2, act=True
    ):
        super().__init__()
        self.upsample = UpSample(
            in_channels,
            in_channels // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act=act,
        )
        self.doubleconv = nn.Sequential(
            *[
                Conv_BN_ReLU(
                    encoder_channel + in_channels // 2,
                    encoder_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                Conv_BN_ReLU(
                    encoder_channel, encoder_channel, kernel_size=3, stride=1, padding=1
                ),
            ]
        )

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.doubleconv(x1)
        return x1


"""
ResBlock
"""


class ResBlock(nn.Module):
    """Proposed residual unit
    https://arxiv.org/pdf/1603.05027.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding=1,
        att_ratio=1 / 8,
        downsample=True,
    ):
        super().__init__()
        self.skip = nn.Sequential(
            IN_Conv_ReLU(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=padding
            ),
            IN_Conv_ReLU(
                out_channels, out_channels, kernel_size=3, stride=1, padding=padding
            ),
            SEAttention(out_channels),
        )

        self.identity = Conv_IN_ReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            act=False,
        )

    def forward(self, x):
        skip = self.skip(x)
        identity = self.identity(x)
        residual = identity + skip

        return residual


# ========================================================================================================================================
# Model : RES_UNET
# Source: https://arxiv.org/pdf/1908.02182.pdf
# ========================================================================================================================================
def Encoder_Res(in_channels, out_channels, layer_num, stride):
    layers = []
    layers.append(ResBlock(in_channels, out_channels, stride=stride))
    in_channels = out_channels
    for i in range(0, layer_num):
        layers.append(ResBlock(in_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


# =================================================================================================================================================
#  Model name: RESIDUAL UNET MODULES
#  Source    : https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb
# =================================================================================================================================================
class stem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.skip = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            IN_Conv_ReLU(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )
        self.identity = Conv_IN_ReLU(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=False
        )

    def forward(self, x):
        skip = self.skip(x)
        identity = self.identity(x)
        residual = identity + skip

        return residual


class Up_Concat(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode="trilinear", align_corners=True
        )

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x1 = Uppad(x1, x2).pad()
        x1 = torch.cat([x1, x2], dim=1)
        return x1


"""
Decoder Block + ResBlock
"""


class Decoder_Res(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, stride=1, padding=1):
        super().__init__()
        self.up_concat = Up_Concat(scale_factor)
        self.res_block = ResBlock(
            in_channels, out_channels, stride=stride, padding=padding
        )

    def forward(self, x1, x2):
        x = self.up_concat(x1, x2)
        x = self.res_block(x)
        return x


# **************************************************
#   Pretrained model with Resnet and EfficientNet
# ===================================================


class PretrainedBlock(nn.Module):
    def __init__(self, name, pretrained, configs):
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        self.configs = configs

    def get_channels(self):
        """Get channels from pretrained model"""
        num_channels = []
        if self.name.startswith("res"):
            num_channels.append(getattr(self.pretrained, "bn1").num_features)
            for i in range(1, 5):
                layer = getattr(self.pretrained, f"layer{i}")[-1]
                if hasattr(layer, "bn3"):
                    num_channels.append(layer.bn3.num_features)
                else:
                    num_channels.append(layer.bn2.num_features)

        elif self.name.startswith("effi"):
            for i in self.configs[self.name]:
                num_channels.append(self.pretrained._blocks[i]._bn2.num_features)

        return num_channels

    def get_features(self, x):
        """Extract features from pretrained model"""
        extract_features = []
        if self.name.startswith("resnet"):
            x = nn.Sequential(*list(self.pretrained.children())[:3])(x)
            extract_features.append(x)
            x = getattr(self.pretrained, "maxpool")(x)
            for i in range(1, 5):
                layer = getattr(self.pretrained, f"layer{i}")
                for j, bottleneck in enumerate(layer):
                    x = bottleneck(x)
                    if j == len(layer) - 1:
                        extract_features.append(x)

        elif self.name.startswith("effi"):
            x = self.pretrained._swish(
                self.pretrained._bn0(self.pretrained._conv_stem(x))
            )
            for idx, block in enumerate(self.pretrained._blocks):
                x = block(x)
                if idx in self.configs[self.name]:
                    extract_features.append(x)

        return extract_features
