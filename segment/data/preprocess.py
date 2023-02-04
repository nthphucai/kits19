import os
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import KFold
from tqdm import tqdm

import segment.utils.parameter as para
from segment.utils.utils import multiprocess

"""
return 3D volume after preprocess
"""


class preprocess3D:
    def __init__(self, df, name, kfold=True):
        self.df = df
        self.name = name
        self.kfold = kfold

        self.vol_path = os.path.join(
            para.output_path, self.name, "{phase}_vol_path".format(phase=self.name)
        )
        self.seg_path = os.path.join(
            para.output_path, self.name, "{phase}_seg_path".format(phase=self.name)
        )
        if os.path.exists(self.vol_path) or os.path.exists(self.seg_path):
            shutil.rmtree(self.vol_path)
            shutil.rmtree(self.seg_path)
        # print('...make dirs...')
        os.makedirs(self.vol_path)
        os.makedirs(self.seg_path)

    def create_npy(self, idx):
        row = self.df.iloc[idx]
        case_id = row["case_id"]
        vol = nib.load(row["img_path"])
        msk = nib.load(row["seg_path"])

        vol_affine = vol.affine[[2, 1, 0, 3]]
        msk_affine = vol.affine[[2, 1, 0, 3]]

        vol = vol.get_fdata()
        msk = msk.get_fdata()

        vol = np.clip(vol, para.lower_bound, para.upper_bound)
        vol, msk = drop_invalid_range(volume=vol, label=msk)

        new_vol = resample(vol, np.diag(abs(vol_affine)), para.target_spacing, order=3)
        new_msk = resample(msk, np.diag(abs(msk_affine)), para.target_spacing, order=0)

        crop_vol = crop_pad(new_vol, axes=(1, 2), crop_size=256)
        crop_msk = crop_pad(new_msk, axes=(1, 2), crop_size=256)

        non_zero = new_msk.sum()
        crop_non_zero = crop_msk.sum()

        assert non_zero == crop_non_zero, "error cropping"

        np.save(
            os.path.join(
                self.vol_path, "{case_id}_imaging".format(case_id=case_id) + ".nii.gz"
            ),
            crop_vol,
        )
        np.save(
            os.path.join(
                self.seg_path,
                "{case_id}_segmentation".format(case_id=case_id) + ".nii.gz",
            ),
            crop_msk,
        )

        # print('processed vol:', crop_vol.shape)

    def run(self):
        try:
            multiprocess(self.create_npy, range(len(self.df)), workers=4)
        except Exception as e:
            print(e)

        """save to dataframe
      """
        self._save_to_df()

    def _save_to_df(self):
        dataset = []
        for file in os.listdir(self.vol_path):
            if file.startswith("case"):
                case = file.split("/")[-1].split(".")[0].split("_")[0]
                _id = file.split("/")[-1].split(".")[0].split("_")[1]
                case_id = case + "_" + _id

                vol_file = file
                seg_file = file.replace("imaging", "segmentation")
                dataset.append(
                    [
                        case_id,
                        os.path.join(self.vol_path, vol_file),
                        os.path.join(self.seg_path, seg_file),
                    ]
                )

        df = pd.DataFrame(dataset, columns=["case_id", "new_vol_path", "new_seg_path"])
        df.to_csv(os.path.join(para.csv_path, self.name + "_ds" + ".csv"))
        if self.kfold:
            split_data(df=df, n_split=10)


def split_data(df, n_split):
    kfold = KFold(n_splits=n_split, random_state=42, shuffle=True)
    for i, (train_index, val_index) in enumerate(kfold.split(df)):
        df.loc[val_index, "fold"] = i
        df.to_csv(os.path.join(para.csv_path, "train_fold.csv"), index=False)
    return df


"""
resample(voume, old_shape, new_shape, order)
"""


def resample(v, dxyz, new_dxyz, order=1):
    dz, dy, dx = dxyz[:3]
    new_dz, new_dy, new_dx = new_dxyz[:3]

    z, y, x = v.shape

    new_x = np.round(x * dx / new_dx)
    new_y = np.round(y * dy / new_dy)
    new_z = np.round(z * dz / new_dz)

    new_v = zoom(v, (new_z / z, new_y / y, new_x / x), order=order)
    return new_v


# ====================================================================================================================================
# CROP AND PAD
# Source: https://github.com/MIC-DKFZ/batchgenerators/blob/d0b9c45713347808e59a6ab3bb1500b58e034f74/batchgenerators/augmentations/utils.py
# ====================================================================================================================================
"""
Cut off the invalid area
"""


def drop_invalid_range(volume, label=None):
    zero_value = volume[0, 0, 0]
    non_zeros_idx = np.where(volume != zero_value)

    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

    if label is not None:
        return (
            volume[min_z:max_z, min_h:max_h, min_w:max_w],
            label[min_z:max_z, min_h:max_h, min_w:max_w],
        )
    else:
        return volume[min_z:max_z, min_h:max_h, min_w:max_w]


"""
crop_pad(volume, axes, crop_sz)
"""


def crop_pad(v, axes, crop_size=256):
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


"""Crop object corresponding to ndimage.find_objects
"""
# def crop_object (img: np.array, label: np.array):

#   img = img.astype(int)
#   label = label.astype(int)
#   slice_z, slice_y, slice_x = ndimage.find_objects(img)[184]

#   return img[slice_z, slice_y,  :], label[slice_z, slice_y,  :]


def crop_object(img: np.array, label: np.array):

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


"""
  Padding img
  img: (z, x, y)
  Source: https://github.com/MIC-DKFZ/batchgenerators/blob/d0b9c45713347808e59a6ab3bb1500b58e034f74/batchgenerators/augmentations/utils.py

"""


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


"""
Crop without respect to label
  
"""


def random_crop_3D(img, label=None, crop_size=256):
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


"""
Crop respect to label
"""


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
    from random import random

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
