from .transforms import augs
from .crop_and_pad import (crop_and_pad_if_needed, foreground_crop,
                           random_center_crop, random_crop_3D)

aug_maps = {
    "crop_and_pad_if_needed": crop_and_pad_if_needed,
    "random_crop_3D": random_crop_3D,
    "foreground_crop": foreground_crop,
    "random_center_crop": random_center_crop,
    "transforms": augs
}
