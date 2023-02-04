import torch.nn as nn
from efficientnet_pytorch_3d import EfficientNet3D
from numpy import diff

from segment.data.normalization import Normalization
from segment.models.medicalnet import generate_model
from segment.models.modules.attention import CBAM3D, SAModule, SEAttention
from segment.models.modules.blocks import DecoderBlock, EncoderBlock, UpSample
from segment.models.modules.common import (BN_Conv_ReLU, Decoder_Res,
                                           DoubleConv, Encoder_Res,
                                           PretrainedBlock, ResBlock, stem)

# ===============================================================================================================
#                                          BASIC UNET MODULES
# ===============================================================================================================


class BasicUnet3D(nn.Module):
    def __init__(self, in_imgs, in_channels, num_classes):
        super().__init__()
        self.norm = Normalization()
        self.enc1 = DoubleConv(in_imgs, in_channels)
        self.enc2 = EncoderBlock(in_channels, in_channels * 2)
        self.enc3 = EncoderBlock(in_channels * 2, in_channels * 4)  # 64
        self.enc4 = EncoderBlock(in_channels * 4, in_channels * 8)  # 128
        self.enc5 = EncoderBlock(in_channels * 8, in_channels * 16)  # 256

        self.dec1 = DecoderBlock(in_channels * 16, in_channels * 8, act=True)
        self.dec2 = DecoderBlock(in_channels * 8, in_channels * 4, act=True)
        self.dec3 = DecoderBlock(in_channels * 4, in_channels * 2, act=True)
        self.dec4 = DecoderBlock(in_channels * 2, in_channels, act=True)

        self.seg = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.norm(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        x6 = self.dec1(x5, x4)
        x7 = self.dec2(x6, x3)
        x8 = self.dec3(x7, x2)
        x9 = self.dec4(x8, x1)

        out = self.seg(x9)

        return out


# =================================================================================================================================================
#  Model name: RESIDUAL UNET MODULES
#  Source    : https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb
# =================================================================================================================================================


class UnetRes3D_v1(nn.Module):
    def __init__(self, in_channels, filters=[16, 32, 64, 128, 256], num_classes=3):
        super().__init__()
        self.norm = Normalization()
        self.stem = stem(in_channels, filters[0], stride=1, padding=1)
        self.enc_block1 = ResBlock(filters[0], filters[1], padding=1, stride=2)
        self.enc_block2 = ResBlock(filters[1], filters[2], padding=1, stride=2)
        self.enc_block3 = ResBlock(filters[2], filters[3], padding=1, stride=2)
        self.enc_block4 = ResBlock(filters[3], filters[4], padding=1, stride=2)

        self.bridge = nn.Sequential(
            BN_Conv_ReLU(filters[4], filters[4], kernel_size=1, stride=1, padding=0),
            BN_Conv_ReLU(filters[4], filters[4], kernel_size=1, stride=1, padding=0),
        )
        self.dec_block1 = Decoder_Res(
            filters[4] + filters[3], filters[4], stride=1, padding=1
        )
        self.dec_block2 = Decoder_Res(
            filters[4] + filters[2], filters[3], stride=1, padding=1
        )
        self.dec_block3 = Decoder_Res(
            filters[3] + filters[1], filters[2], stride=1, padding=1
        )
        self.dec_block4 = Decoder_Res(
            filters[2] + filters[0], filters[1], stride=1, padding=1
        )

        self.seg = nn.Sequential(
            *[
                SEAttention(filters[1]),
                nn.Dropout(0.5),
                nn.Conv3d(filters[1], num_classes, kernel_size=1, stride=1),
            ]
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.stem(x)
        x1 = self.enc_block1(x)
        x2 = self.enc_block2(x1)
        x3 = self.enc_block3(x2)
        x4 = self.enc_block4(x3)

        x5 = self.bridge(x4)

        x6 = self.dec_block1(x5, x3)
        x7 = self.dec_block2(x6, x2)
        x8 = self.dec_block3(x7, x1)
        x9 = self.dec_block4(x8, x)

        output = self.seg(x9)

        return output


# ========================================================================================================================================
# Model : RES_UNET
# Source: https://arxiv.org/pdf/1908.02182.pdf
# ========================================================================================================================================


layer_num = [1, 2, 3, 4, 5]
filters = [24, 60, 120, 240, 320]


class UnetRes3D_v2(nn.Module):
    def __init__(
        self, in_channels, filters=filters, layer_num=layer_num, num_classes=3
    ):
        super().__init__()
        self.norm = Normalization()
        self.res_enc1 = stem(in_channels, filters[0], stride=1, padding=1)
        self.res_enc2 = Encoder_Res(
            filters[0], filters[1], layer_num=layer_num[0], stride=(1, 2, 2)
        )
        self.res_enc3 = Encoder_Res(
            filters[1], filters[2], layer_num=layer_num[1], stride=2
        )
        self.res_enc4 = Encoder_Res(
            filters[2], filters[3], layer_num=layer_num[2], stride=2
        )
        self.res_enc5 = Encoder_Res(
            filters[3], filters[4], layer_num=layer_num[3], stride=2
        )

        self.bridge = Encoder_Res(
            filters[4], filters[4], layer_num=layer_num[4], stride=2
        )

        self.res_dec1 = Decoder_Res(filters[4] + filters[4], filters[4])
        self.res_dec2 = Decoder_Res(filters[4] + filters[3], filters[3])
        self.res_dec3 = Decoder_Res(filters[3] + filters[2], filters[2])
        self.res_dec4 = Decoder_Res(filters[2] + filters[1], filters[1])
        self.res_dec5 = Decoder_Res(
            filters[1] + filters[0], filters[0], scale_factor=(1, 2, 2)
        )

        self.seg = nn.Sequential(
            *[
                CBAM3D(filters[0]),
                # SEAttention(filters[0]),
                nn.Dropout(0.5),
                nn.Conv3d(filters[0], num_classes, kernel_size=1, bias=False),
            ]
        )

    def forward(self, x):
        x = self.norm(x)
        enc1 = self.res_enc1(x)
        enc2 = self.res_enc2(enc1)
        enc3 = self.res_enc3(enc2)
        enc4 = self.res_enc4(enc3)
        enc5 = self.res_enc5(enc4)

        enc6 = self.bridge(enc5)

        dec1 = self.res_dec1(enc6, enc5)
        dec2 = self.res_dec2(dec1, enc4)
        dec3 = self.res_dec3(dec2, enc3)
        dec4 = self.res_dec4(dec3, enc2)
        dec5 = self.res_dec5(dec4, enc1)

        output = self.seg(dec5)

        return output


# ========================================================================================================================================
# Pretrained model Efficinetnet and Resnet
# ========================================================================================================================================
pretrain_path = "/content/drive/MyDrive/Seg3D/KiTS2019/Source/outputs/checkpoints/medicalnet/resnet_50.pth"

""" DyUnet + Decoder_Res
"""
configs = {
    "resnet18": 18,
    "resnet34": 34,
    "resnet50": 50,
    "resnet101": 101,
    "efficientnet-b0": [0, 2, 4, 10, 15],
    "efficientnet-b1": [1, 3, 7, 15, 22],
    "efficientnet-b2": [1, 4, 7, 15, 22],
    "efficientnet-b3": [1, 4, 7, 17, 25],
    "efficientnet-b4": [1, 5, 9, 21, 31],
    "efficientnet-b5": [2, 7, 12, 26, 38],
    "efficientnet-b6": [2, 8, 14, 30, 44],
    "efficientnet-b7": [3, 10, 17, 37, 54],
}


class DyUnetRes3D(nn.Module):
    def __init__(self, pretrained_name, configs=configs, num_classes=3):
        super(DyUnetRes3D, self).__init__()
        self.pretrained_name = pretrained_name
        self.configs = configs

        if pretrained_name.startswith("effi"):
            pretrained = EfficientNet3D.from_name(
                pretrained_name,
                override_params={"num_classes": num_classes},
                in_channels=1,
            )
        elif pretrained_name.startswith("res"):
            pretrained = generate_model(
                pretrain_path=pretrain_path,
                model_depth=configs["resnet50"],
                shortcut_type="B",
                num_seg_classes=num_classes,
            )

        for param in pretrained.parameters():
            param.requires_grad = True

        self.pretrained_block = PretrainedBlock(
            name=pretrained_name, pretrained=pretrained, configs=configs
        )

        for param in pretrained.parameters():
            param.requires_grad = True

        ch1, ch2, ch3, ch4, final_channel = self.pretrained_block.get_channels()

        self.norm = Normalization()

        self.dec4 = Decoder_Res(final_channel + ch4, ch4)
        self.dec3 = Decoder_Res(ch4 + ch3, ch3)
        self.dec2 = Decoder_Res(ch3 + ch2, ch2)
        self.dec1 = Decoder_Res(ch2 + ch1, ch1)

        self.seg = nn.Sequential(
            *[
                # SEAttention(ch1),
                CBAM3D(ch1),
                UpSample(ch1, 32, kernel_size=2, stride=2, padding=0, act=True),
                nn.Dropout(0.5),
                nn.Conv3d(32, num_classes, kernel_size=1, bias=False),
            ]
        )

    def forward(self, x):
        x = self.norm(x)
        enc1, enc2, enc3, enc4, final_enc = self.pretrained_block.get_features(x)
        b4_dec = self.dec4(final_enc, enc4)
        b3_dec = self.dec3(b4_dec, enc3)
        b2_dec = self.dec2(b3_dec, enc2)
        b1_dec = self.dec1(b2_dec, enc1)

        out = self.seg(b1_dec)
        return out
