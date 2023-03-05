from .crop_and_pad import (crop_object, crop_pad, foreground_crop,
                           pad_if_needed, random_center_crop, random_crop_2D,
                           random_crop_3D)

aug_maps = {
    "crop_pad": crop_pad,
    "crop_object": crop_object,
    "pad_if_needed": pad_if_needed,
    "random_crop_3D": random_crop_3D,
    "random_crop_2D": random_crop_2D,
    "foreground_crop": foreground_crop,
    "random_center_crop": random_center_crop,
}
