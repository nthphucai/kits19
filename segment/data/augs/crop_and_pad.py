import numpy as np


def crop_pad(v, axes, crop_size=256):
    """
    :param volumne:
    :param axes:(z,y,x)
    :return:
    """
    shapes = np.array(v.shape)
    axes = np.array(axes)
    sizes = np.array(shapes[axes])

    diffs = sizes - np.array(crop_size)
    for diff, axis in zip(diffs, axes):
        left = abs(diff) // 2
        right = (left + 1) if diff % 2 != 0 else left

        if diff < 0:
            v = np.pad(
                v, [(left, right) if i == axis else (0, 0) for i in range(len(shapes))]
            )
        elif diff > 0:
            slices = tuple(
                [
                    slice(left, -right) if i == axis else slice(None)
                    for i in range(len(shapes))
                ]
            )
            v = v[slices]
        else:
            continue
    return v


def crop_object(img: np.array, label: np.array):
    """
    Crop object corresponding to ndimage.find_objects
    :param img:
    :param label:
    :return:
    """
    vol_array = img
    img = img.astype(int)
    label = label.astype(int)
    slice_z, slice_y, slice_x = ndimage.find_objects(img)[184]

    slice_z_stop = slice_z.stop

    if vol_array.shape[0] > 100:
        slice_z_stop = vol_array.shape[0] - (vol_array.shape[0] % 100)
    elif vol_array.shape[0] > 200:
        slice_z_stop = vol_array.shape[0] - (vol_array.shape[0] % 200)

    return (
        img[slice_z.start : slice_z_stop, slice_y, :],
        label[slice_z.start : slice_z_stop, slice_y, :],
    )


def pad_if_needed(img, label, axes, crop_size):
    shapes = np.array(img.shape)
    axes = np.array(axes)
    sizes = np.array(shapes[axes])
    diffs = sizes - np.array(crop_size)

    for diff, axis in zip(diffs, axes):
        left = abs(diff) // 2
        right = (left + 1) if diff % 2 != 0 else left
        if diff >= 0:
            continue
        elif diff < 0:
            img = np.pad(
                img,
                [(left, right) if i == axis else (0, 0) for i in range(len(shapes))],
            )
            if label is not None:
                label = np.pad(
                    label,
                    [
                        (left, right) if i == axis else (0, 0)
                        for i in range(len(shapes))
                    ],
                )

    if label is not None:
        return img, label

    return img, label


def random_crop_3D(img, label=None, crop_size=256):
    """
    Crop without respect to label
    :param img:
    :param label:
    :return:
    """
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * len(img.shape)
    else:
        assert len(crop_size) == len(
            img.shape
        ), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    if crop_size[0] < img.shape[0]:
        lb_z = np.random.randint(0, img.shape[0] - crop_size[0])
    elif crop_size[0] == img.shape[0]:
        lb_z = 0
    else:
        raise ValueError(
            "crop_size[0] must be smaller or equal to the images z dimension"
        )

    if crop_size[1] < img.shape[1]:
        lb_y = np.random.randint(0, img.shape[1] - crop_size[1])
    elif crop_size[1] == img.shape[1]:
        lb_y = 0
    else:
        raise ValueError(
            "crop_size[1] must be smaller or equal to the images x dimension"
        )

    if crop_size[2] < img.shape[2]:
        lb_x = np.random.randint(0, img.shape[2] - crop_size[2])
    elif crop_size[2] == img.shape[2]:
        lb_x = 0
    else:
        raise ValueError(
            "crop_size[2] must be smaller or equal to the images y dimension"
        )

    img = img[
        lb_z : lb_z + crop_size[0],
        lb_y : lb_y + crop_size[1],
        lb_x : lb_x + crop_size[2],
    ]
    if label is not None:
        label = label[
            lb_z : lb_z + crop_size[0],
            lb_y : lb_y + crop_size[1],
            lb_x : lb_x + crop_size[2],
        ]
    return img, label

    return img


def random_crop_2D(img, label=None, crop_size=256):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * len(img.shape)
    else:
        assert len(crop_size) == len(
            img.shape
        ), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"

    if crop_size[0] < img.shape[0]:
        lb_x = np.random.randint(0, img.shape[0] - crop_size[0])
    elif crop_size[0] == img.shape[0]:
        lb_x = 0
    else:
        raise ValueError(
            "crop_size[0] must be smaller or equal to the images x dimension"
        )

    if crop_size[1] < img.shape[1]:
        lb_y = np.random.randint(0, img.shape[1] - crop_size[1])
    elif crop_size[1] == img.shape[1]:
        lb_y = 0
    else:
        raise ValueError(
            "crop_size[1] must be smaller or equal to the images y dimension"
        )

    img = img[lb_x : lb_x + crop_size[0], lb_y : lb_y + crop_size[1]]
    if label is not None:
        label = label[lb_x : lb_x + crop_size[0], lb_y : lb_y + crop_size[1]]
    return img, label

    return img


def foreground_crop(img, label):
    from random import random

    target_indexs = np.where(label > 0)
    [img_d, img_h, img_w] = img.shape
    [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
    [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)

    [target_d, target_h, target_w] = np.array([max_D, max_H, max_W]) - np.array(
        [min_D, min_H, min_W]
    )

    Z_min = int((min_D - target_d * 1.0 / 2))
    Y_min = int((min_H - target_h * 1.0 / 2))
    X_min = int((min_W - target_w * 1.0 / 2))

    Z_max = int(img_d - ((img_d - (max_D + target_d * 1.0 / 2))))
    Y_max = int(img_h - ((img_h - (max_H + target_h * 1.0 / 2))))
    X_max = int(img_w - ((img_w - (max_W + target_w * 1.0 / 2))))

    Z_min = int(np.max([0, Z_min]))
    Y_min = int(np.max([0, Y_min]))
    X_min = int(np.max([0, X_min]))

    Z_max = int(np.min([img_d, Z_max]))
    Y_max = int(np.min([img_h, Y_max]))
    X_max = int(np.min([img_w, X_max]))

    return (
        img[Z_min:Z_max, Y_min:Y_max, X_min:X_max],
        label[Z_min:Z_max, Y_min:Y_max, X_min:X_max],
    )


def random_center_crop(img, label):
    target_indexs = np.where(label > 0)
    [img_d, img_h, img_w] = img.shape
    [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
    [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)

    [target_d, target_h, target_w] = np.array([max_D, max_H, max_W]) - np.array(
        [min_D, min_H, min_W]
    )

    Z_min = int((min_D - target_d * 1.0 / 2) * random())
    Y_min = int((min_H - target_h * 1.0 / 2) * random())
    X_min = int((min_W - target_w * 1.0 / 2) * random())

    Z_max = int(img_d - ((img_d - (max_D + target_d * 1.0 / 2))) * random())
    Y_max = int(img_h - ((img_h - (max_H + target_h * 1.0 / 2))) * random())
    X_max = int(img_w - ((img_w - (max_W + target_w * 1.0 / 2))) * random())

    Z_min = int(np.max([0, Z_min]))
    Y_min = int(np.max([0, Y_min]))
    X_min = int(np.max([0, X_min]))

    Z_max = int(np.min([img_d, Z_max]))
    Y_max = int(np.min([img_h, Y_max]))
    X_max = int(np.min([img_w, X_max]))

    return (
        img[Z_min:Z_max, Y_min:Y_max, X_min:X_max],
        label[Z_min:Z_max, Y_min:Y_max, X_min:X_max],
    )
