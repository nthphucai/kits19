import numpy as np
import torch
from scipy import ndimage

from segment.data.base_class import StandardDataset
from segment.data.normalization import normalize
from segment.data.preprocess import (crop_object, crop_pad, foreground_crop,
                                     pad_if_needed, random_center_crop,
                                     random_crop_3D)

"""
Create train_dataset

"""


class Kits19Dataset(StandardDataset):
    def __init__(self, df, aug=None, phase="train"):
        super().__init__()
        self.df = df
        self.aug = aug
        self.phase = phase

        self.inptz = 80  # 128
        self.inpty = 160
        self.inptx = 160

    def get_len(self):
        return self.df.shape[0]

    def get_item(self, idx):

        row = self.df.iloc[idx]
        vol_path = row["new_vol_path"]
        seg_path = row["new_seg_path"]

        vol_arr = np.load(vol_path)
        seg_arr = np.load(seg_path)

        # patch size
        patch_size = [self.inptz, self.inpty, self.inptx]

        if self.phase == "train":
            vol_arr, seg_arr = self._preprocess(
                vol_arr, seg_arr, patch_size, phase="train"
            )
        elif self.phase == "val":
            vol_arr, seg_arr = self._preprocess(
                vol_arr, seg_arr, patch_size, phase="val"
            )

        seg_arr = self._stack_mask(seg_arr)
        vol = torch.FloatTensor(vol_arr).unsqueeze(0)
        seg = torch.FloatTensor(seg_arr)

        if self.aug is not None:
            auged = self.aug({"image": vol, "mask": seg})
            vol = auged["image"]
            seg_msk = (auged["mask"] > 0.5).int()

        return vol, seg

    def _stack_mask(self, mask_arr: np.ndarray):
        org_arr = mask_arr.copy()
        org_arr[org_arr == 0] = 0
        org_arr[org_arr == 1] = 1
        org_arr[org_arr == 2] = 0

        tumor_arr = mask_arr.copy()
        tumor_arr[tumor_arr == 0] = 0
        tumor_arr[tumor_arr == 1] = 0
        tumor_arr[tumor_arr == 2] = 1

        bground_arr = mask_arr.copy()
        bground_arr[bground_arr == 0] = 1
        bground_arr[bground_arr == 1] = 0
        bground_arr[bground_arr == 2] = 0

        stack_msk = np.stack([bground_arr, org_arr, tumor_arr])
        # stack_msk = np.stack([org_arr, tumor_arr])

        return stack_msk

    def _stack_img(self, vol_arr: np.ndarray, seg_arr: np.ndarray, crop_size):
        """first image: foreground + random crop"""
        vol_st, _ = foreground_crop(img=vol_arr, label=seg_arr)
        vol_st, _ = random_crop_3D(img=vol_st, label=seg_arr, crop_size=crop_size)
        """ second image: random crop
      """
        vol_nd, _ = random_crop_3D(img=vol_arr, label=seg_arr, crop_size=crop_size)

        stack_vol = np.stack([vol_st, vol_nd])

        return stack_vol

    def _preprocess(self, vol_arr, seg_arr, patch_size, phase="train"):
        if phase == "train":
            # nonzero = np.where(seg_arr.sum(axis=(1,2))>0)[0]
            # start, end = nonzero[[0, -1]]
            # start_slc = max(0, start-para.expand_slice)
            # end_slc   = min(end+para.expand_slice, seg_arr.shape[0]-1)
            # vol_arr = vol_arr[start_slc:end_slc+1, ...]
            # seg_arr = seg_arr[start_slc:end_slc+1, ...]

            vol_arr, seg_arr = crop_or_pad(
                vol_arr, seg_arr, crop_size=patch_size, mode="random_center_crop"
            )

        if phase == "val":
            vol_arr, seg_arr = crop_or_pad(
                vol_arr, seg_arr, crop_size=patch_size, mode="center"
            )

        return vol_arr, seg_arr


"""
  preprocess images 
"""


def crop_or_pad(vol_arr: np.array, seg_arr: np.array, crop_size, mode="center"):

    if mode == "random":
        vol_arr, seg_arr = crop_object(vol_arr, seg_arr)
        vol_arr = crop_pad(vol_arr, axes=(0,), crop_size=80)
        seg_arr = crop_pad(seg_arr, axes=(0,), crop_size=80)
        vol_arr, seg_arr = pad_if_needed(
            img=vol_arr,
            label=seg_arr,
            axes=(
                1,
                2,
            ),
            crop_size=160,
        )
        vol_arr, seg_arr = random_crop_3D(
            img=vol_arr, label=seg_arr, crop_size=crop_size
        )

    elif mode == "center":
        vol_arr, seg_arr = crop_object(vol_arr, seg_arr)
        vol_arr = crop_pad(vol_arr, axes=(0, 1, 2), crop_size=crop_size)
        seg_arr = crop_pad(seg_arr, axes=(0, 1, 2), crop_size=crop_size)

    elif mode == "random_center_crop":
        vol_arr, seg_arr = random_center_crop(img=vol_arr, label=seg_arr)
        vol_arr = crop_pad(vol_arr, axes=(0, 1, 2), crop_size=crop_size)
        seg_arr = crop_pad(seg_arr, axes=(0, 1, 2), crop_size=crop_size)

    elif mode == "x_random":
        vol_arr = crop_pad(vol_arr, axes=(0, 1), crop_size=(80, 160))
        seg_arr = crop_pad(seg_arr, axes=(0, 1), crop_size=(80, 160))
        vol_arr, seg_arr = pad_if_needed(
            img=vol_arr, label=seg_arr, axes=(2,), crop_size=160
        )
        vol_arr, seg_arr = random_crop_3D(
            img=vol_arr, label=seg_arr, crop_size=crop_size
        )

    return vol_arr, seg_arr


# from scipy import ndimage
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

# import parameter as para
# from repos.constant import np, random, torch
# from from segment.data.normalization import normalize
# from from segment.data.base_class import StandardDataset
# from from segment.data.preprocess import foreground_crop, crop_object, random_crop_3D

# """
# Create train_dataset

# """
# class Kits19Dataset(StandardDataset):
#     def __init__(self, df, aug=None, phase='train'):
#       super().__init__()
#       self.df = df
#       self.aug = aug
#       self.phase = phase

#       self.inptz = 80 #128
#       self.inpty = 160
#       self.inptx = 160

#     def get_len(self):
#       return self.df.shape[0]

#     def get_item(self, idx):

#         row = self.df.iloc[idx]
#         vol_path = row['new_vol_path']
#         seg_path = row['new_seg_path']

#         vol_arr = np.load(vol_path)
#         seg_arr = np.load(seg_path)

#         # patch size
#         patch_size = [self.inptz, self.inpty, self.inptx]

#         if self.phase == 'train':
#             # vol_arr, seg_arr = self._stack_img(vol_arr, seg_arr, patch_size, phase='train')
#           vol_arr, seg_arr = self._preprocess(vol_arr, seg_arr, patch_size, phase='train')

#         elif self.phase == 'val':
#           vol_arr, seg_arr = self._stack_img(vol_arr, seg_arr, patch_size, phase='val')
#         #   vol_arr, seg_arr = self._preprocess(vol_arr, seg_arr, patch_size, phase='test')

#         seg_arr = self._stack_mask(seg_arr)
#         vol = torch.FloatTensor(vol_arr).unsqueeze(0)
#         seg = torch.FloatTensor(seg_arr)

#         if self.aug is not None:
#             auged = self.aug({"image": vol, "mask": seg})
#             vol = auged["image"]
#             seg = (auged["mask"] > 0.5).int()

#         return vol, seg

#     def _stack_mask(self, mask_arr: np.ndarray):
#       org_arr = mask_arr.copy()
#       org_arr[org_arr == 0] = 0
#       org_arr[org_arr == 1] = 1
#       org_arr[org_arr == 2] = 0

#       tumor_arr = mask_arr.copy()
#       tumor_arr[tumor_arr == 0] = 0
#       tumor_arr[tumor_arr == 1] = 0
#       tumor_arr[tumor_arr == 2] = 1

#       bground_arr = mask_arr.copy()
#       bground_arr[bground_arr == 0] = 1
#       bground_arr[bground_arr == 1] = 0
#       bground_arr[bground_arr == 2] = 0

#       stack_msk = np.stack([bground_arr, org_arr, tumor_arr])
#       # stack_msk = np.stack([org_arr, tumor_arr])

#       return stack_msk

#     def _stack_img(self, vol_arr: np.ndarray, seg_arr: np.ndarray, crop_size, phase='train'):
#       if phase == 'train':
#         """ first image: foreground + random crop
#         """
#         # vol_st, _ = crop_or_pad(vol_arr, seg_arr, crop_size=crop_size, mode='random_center_crop')

#         """ second image: random crop
#         """
#         # vol_nd, seg_arr = crop_or_pad(vol_arr, seg_arr, crop_size=crop_size, mode='random')

#         vol_st, seg_arr = crop_or_pad(vol_arr, seg_arr, crop_size=crop_size, mode='random')

#         stack_vol = np.stack([vol_st, vol_st])

#       elif phase == 'val':
#         vol_st, seg_arr = crop_or_pad(vol_arr, seg_arr, crop_size=crop_size, mode='random')
#         stack_vol = np.stack([vol_st, vol_st])

#       return stack_vol, seg_arr

#     def _preprocess(self, vol_arr, seg_arr, patch_size, phase='train'):
#       if phase == 'train':
#         nonzero = np.where(seg_arr.sum(axis=(1,2))>0)[0]
#         start, end = nonzero[[0, -1]]
#         start_slc = max(0, start-para.expand_slice)
#         end_slc   = min(end+para.expand_slice, seg_arr.shape[0]-1)
#         vol_arr = vol_arr[start_slc:end_slc+1, ...]
#         seg_arr = seg_arr[start_slc:end_slc+1, ...]

#         vol_arr, seg_arr = crop_or_pad(vol_arr, seg_arr, crop_size=patch_size, mode='random_center_crop')

#       if phase == 'test':
#         vol_arr, seg_arr = crop_or_pad(vol_arr, seg_arr, crop_size=patch_size, mode='z_random')

#       return vol_arr, seg_arr


# """
#   preprocess images
# """
# def crop_or_pad(vol_arr: np.array, seg_arr: np.array, crop_size, mode='center'):
#   from from segment.data.preprocess import pad_if_needed, crop_pad, random_crop_3D, random_center_crop

#   if mode == 'random':
#     vol_arr, seg_arr = crop_object(vol_arr, seg_arr)
#     vol_arr = crop_pad(vol_arr, axes=(0,), crop_size=80)
#     seg_arr = crop_pad(seg_arr, axes=(0,), crop_size=80)
#     vol_arr, seg_arr = pad_if_needed(img=vol_arr, label=seg_arr, axes=(1,2,), crop_size=160)
#     vol_arr, seg_arr = random_crop_3D(img=vol_arr, label=seg_arr, crop_size=crop_size)

#   elif mode == 'center':
#     vol_arr, seg_arr = crop_object(vol_arr, seg_arr)
#     vol_arr = crop_pad(vol_arr, axes=(0,1,2), crop_size=crop_size)
#     seg_arr = crop_pad(seg_arr, axes=(0,1,2), crop_size=crop_size)

#   elif mode == 'random_center_crop':
#     vol_arr, seg_arr = random_center_crop(img=vol_arr, label=seg_arr)
#     # vol_arr, seg_arr = foreground_crop(img=vol_arr, label=seg_arr)
#     vol_arr = crop_pad(vol_arr, axes=(0,1,2), crop_size=crop_size)
#     seg_arr = crop_pad(seg_arr, axes=(0,1,2), crop_size=crop_size)

#   elif mode == 'z_random':
#     vol_arr = crop_pad(vol_arr, axes=(1,2,), crop_size=160)
#     seg_arr = crop_pad(seg_arr, axes=(1,2,), crop_size=160)
#     vol_arr, seg_arr = pad_if_needed(img=vol_arr, label=seg_arr, axes=(0,), crop_size=80)
#     vol_arr, seg_arr = random_crop_3D(img=vol_arr, label=seg_arr, crop_size=crop_size)

#   return vol_arr, seg_arr
