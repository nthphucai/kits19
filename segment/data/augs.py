import numpy as np
import torchio.transforms as transforms
from scipy import ndimage

# ====================================================================================================================================
# TRANSFORMATION
# Source: https://github.com/MIC-DKFZ/batchgenerators/blob/d0b9c45713347808e59a6ab3bb1500b58e034f74/batchgenerators/augmentations/utils.py
# ====================================================================================================================================
augs = [
    transforms.RandomFlip(axes=[0, 2], flip_probability=0.3, include=["image", "mask"]),
    transforms.OneOf(
        [
            transforms.RandomBlur(std=1, p=0.3, include=["image"]),
            transforms.RandomNoise(p=0.3, include=["image"]),
        ]
    ),
    transforms.RandomGamma(0.01, p=0.3, include=["image"]),
    transforms.RandomAffine(
        scales=0.1,
        degrees=(10, 0, 0),
        translation=24,
        image_interpolation="bspline",
        p=0.3,
        include=["image", "mask"],
    ),
    # transforms.OneOf([
    #           # transforms.RandomAnisotropy(downsampling=(1, 2), p=0.1, include=["image", "mask"]),
    #           transforms.RandomSwap(patch_size=15, num_iterations = 100, p=0.1, include=["image", "mask"])
    #           ])
]
augs = transforms.Compose(augs, include=["image", "mask"])
