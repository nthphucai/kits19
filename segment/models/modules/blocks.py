import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import diff

from segment.models.modules.attention import CBAM3D, SAModule, SEAttention


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


class EncoderBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2), DoubleConv(in_channels, out_channels)
        )


class Uppad(nn.Module):
    """
    change shape: x1.size = x2.size
    This because the original paper encourage down-sampling without padding the same as the up-sampling without zero-padding, which can avoid corrupting semantic information.
    This is the one of the reason for which the overlap-tile strategy was proposed
    """

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


class UpSample(nn.Sequential):
    """
    some block of pretrained : they just have flow: Conv --> batchnorm --conv --batchnorm, did not have relu...
    """

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
        self.norm = nn.BatchNorm3d(out_channels)
        if act:
            self.relu = nn.ReLU(nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, encoder_channel, act=False):
        super().__init__()
        self.upsample = UpSample(
            in_channels, in_channels // 2, kernel_size=2, stride=2, padding=0, act=act
        )
        self.doubleconv = DoubleConv(
            encoder_channel + in_channels // 2, encoder_channel
        )

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x1 = torch.cat([Uppad(x1, x2).pad(), x2], dim=1)
        x1 = self.doubleconv(x1)
        return x1


"""
Decoder Block + SE
"""


class DecoderBlock_SE(nn.Module):
    def __init__(self, in_channels, out_channels, encoder_channel, act=False):
        super().__init__()
        self.upsample = UpSample(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            activation=act,
        )
        self.doubleconv = DoubleConv(
            encoder_channel + in_channels // 2, encoder_channel
        )
        self.att = SEAttention(encoder_channel)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x1 = torch.cat([Uppad(x1, x2).pad(), x2], dim=1)
        x1 = self.doubleconv(x1)
        x1 = self.att(x1)
        return x1


# =================================================================================================================================================
#  Model name: RESIDUAL UNET MODULES
#  Source    : https://github.com/nikhilroxtomar/Deep-Residual-Unet/raw/e250c2f5bb1a260cbe7bed6b7232d7b8b20fe1a2//images/arch.png
# =================================================================================================================================================


class ResBlock_v1(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, padding=1):
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.BatchNorm3d(input_channels),
            nn.ReLU(),
            nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Sequential(
            nn.Conv3d(
                input_channels, output_channels, kernel_size=3, padding=1, stride=stride
            ),
            nn.BatchNorm3d(output_channels),
        )

    def forward(self, x):
        return self.doubleconv(x) + self.skip(x)


class Up_Concat(nn.Module):
    def __init__(self, in_channels, out_channels, act=False):
        super().__init__()
        self.upsample = UpSample(
            in_channels, out_channels, kernel_size=2, stride=2, act=act
        )

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x1 = torch.cat([Uppad(x1, x2).pad(), x2], dim=1)
        return x1


# ========================================================================================================================================
# Model : RES_UNET
# Source: https://arxiv.org/pdf/1908.02182.pdf
# ========================================================================================================================================
class ResBlock_v2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=1):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=padding
            ),
            # nn.InstanceNorm3d(out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=padding
            ),
            # nn.InstanceNorm3d(out_channels)
            nn.BatchNorm3d(out_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=3, padding=padding, stride=stride
            ),
            nn.InstanceNorm3d(out_channels),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.residual_block(x) + self.skip(x)
        return self.relu(output)


class Decoder_Res_v2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=2, stride=2, padding=0
            ),
            # nn.InstanceNorm3d(out_channels, affine=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.Conv3d(
                out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1
            ),
            # nn.InstanceNorm3d(out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x1 = torch.cat([Uppad(x1, x2).pad(), x2], dim=1)
        x1 = self.conv(x1)
        return x1


def _make_reslayer(block, in_channels, out_channels, layer_num, stride):
    layers = []
    layers.append(block(in_channels, out_channels, stride=stride))
    in_channels = out_channels
    for i in range(0, layer_num):
        layers.append(block(in_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


# ===============================================================================================================
#                                         CONVOLUTIONAL LAYER
# ===============================================================================================================
# (W-F+2P)/S + 1
# Example: 32x32 input, 5x5 filter, padding=0, stride=1
# (32-5+2*0)/1 + 1 = 27/1 + 1 = 28x28
# summary(model, (3, 32, 32), 4)          # summary(model, (channels, size, size), batch_size)


def cal_convLayer(d_size, w_size, h_size, kernel_size, padding, stride, batch_size):
    out_d = int((d_size - kernel_size[0] + 2 * padding) / (stride[0]) + 1)
    out_w = int((w_size - kernel_size[1] + 2 * padding) / (stride[1]) + 1)
    out_h = int((h_size - kernel_size[2] + 2 * padding) / (stride[2]) + 1)
    return batch_size, out_d, out_w, out_h


# ex: cal_convLayer(d_size=160,
#                   w_size=160, h_size=160, kernel_size=(2,1,1), padding=0, stride=(2,1,1), batch_size=5)


def inv_convLayer(out_d, out_w, out_h, kernel_size, padding, stride):
    d_size = (out_d - 1) * stride - 2 * padding + kernel_size[0]
    w_size = (out_w - 1) * stride - 2 * padding + kernel_size[1]
    h_size = (out_h - 1) * stride - 2 * padding + kernel_size[2]

    return d_size, w_size, h_size


# ex: inv_convLayer(out_d=160, out_w=160, out_h=160, kernel_size=(2, 2, 2), padding=0, stride=2)
