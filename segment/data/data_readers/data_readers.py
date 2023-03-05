from typing import Union

import numpy as np
import torch

from segment.data.base_class import StandardDataset
from segment.data.normalization import normalize
from segment.data.preprocess import preprocess_maps
from segment.utils import parameter as para


class Kits19Dataset(StandardDataset):
    def __init__(self, df, aug=None, phase="train"):
        super().__init__()
        self.df = df
        self.aug = aug
        self.phase = phase

        self.inptz = 80  # 128
        self.inpty = 160
        self.inptx = 160

        self.patch_size = [self.inptz, self.inpty, self.inptx]

    def get_len(self):
        return self.df.shape[0]

    def get_item(self, idx):
        row = self.df.iloc[idx]
        vol_path = row["new_vol_path"]
        seg_path = row["new_seg_path"]
        vol_arr = np.load(vol_path)
        seg_arr = np.load(seg_path)

        if self.phase == "train":
            vol_arr, seg_arr = self._preprocess_inputs(
                vol_arr, seg_arr, self.patch_size, mode="random_center_crop"
            )
        elif self.phase == "val":
            vol_arr, seg_arr = self._preprocess_inputs(
                vol_arr, seg_arr, self.patch_size, mode="random_center_crop"
            )

        seg_arr = self._stack_mask(seg_arr, use_bground=False)
        vol_tensor = torch.FloatTensor(vol_arr).unsqueeze(0)
        seg_tensor = torch.FloatTensor(seg_arr)

        if self.aug is not None:
            auged = self.aug({"image": vol_tensor, "mask": seg_tensor})
            vol_tensor = auged["image"]
            seg_tensor = (auged["mask"] > 0.5).int()

        return vol_tensor, seg_tensor

    def _stack_mask(self, seg_arr: np.ndarray, use_bground: False) -> np.array:
        """This function aims to stack organ mask and tumor mask into one single mask.

        Args:
            seg_arr (np.ndarray): segment array
            use_bground (False): whether to use background as one class to classify,
            in this case, there are 3 classes (background, organ, tumor)

        Returns:
            numpy.array: the segmented mask.
        """
        org_arr = seg_arr.copy()
        org_arr[org_arr == 0] = 0
        org_arr[org_arr == 1] = 1
        org_arr[org_arr == 2] = 0

        tumor_arr = seg_arr.copy()
        tumor_arr[tumor_arr == 0] = 0
        tumor_arr[tumor_arr == 1] = 0
        tumor_arr[tumor_arr == 2] = 1

        if use_bground:
            bground_arr = seg_arr.copy()
            bground_arr[bground_arr == 0] = 1
            bground_arr[bground_arr == 1] = 0
            bground_arr[bground_arr == 2] = 0
            return np.stack([bground_arr, org_arr, tumor_arr])
        else:
            return np.stack([org_arr, tumor_arr])

    @staticmethod
    def _stack_img(vol_arr: np.ndarray, seg_arr: np.ndarray, crop_size):
        vol_st, _ = preprocess_maps["foreground_crop"](vol_arr, seg_arr)
        vol_st, _ = preprocess_maps["random_crop_3D"](vol_st, seg_arr, crop_size)
        # vol_st, _ = preprocess_maps["crop_or_pad"](vol_arr, seg_arr, crop_size=crop_size, mode='random_center_crop')

        vol_nd, _ = preprocess_maps["random_crop_3D"](vol_arr, seg_arr, crop_size)
        # vol_nd, seg_arr = preprocess_maps["crop_or_pad"](vol_arr, seg_arr, crop_size=crop_size, mode='random')
        stack_vol = np.stack([vol_st, vol_nd])
        return stack_vol

    def _preprocess_inputs(
        self,
        vol_arr: np.array,
        seg_arr: np.array,
        patch_size: Union[tuple, list],
        mode: str = "center",
        segment_bground: bool = False,
    ):
        if segment_bground:
            nonzero = np.where(seg_arr.sum(axis=(1, 2)) > 0)[0]
            start, end = nonzero[[0, -1]]
            start_slc = max(0, start - para.expand_slice)
            end_slc = min(end + para.expand_slice, seg_arr.shape[0] - 1)
            vol_arr = vol_arr[start_slc : end_slc + 1, ...]
            seg_arr = seg_arr[start_slc : end_slc + 1, ...]

        vol_arr, seg_arr = self.crop_or_pad(vol_arr, seg_arr, patch_size, mode)
        return vol_arr, seg_arr

    @staticmethod
    def crop_or_pad(
        vol_arr: np.array,
        seg_arr: np.array,
        crop_size: Union[tuple, list],
        mode="center",
    ):
        if mode == "random":
            vol_arr, seg_arr = preprocess_maps["crop_object"](vol_arr, seg_arr)
            vol_arr = preprocess_maps["crop_pad"](vol_arr, axes=(0,), crop_size=80)
            seg_arr = preprocess_maps["crop_pad"](seg_arr, axes=(0,), crop_size=80)
            vol_arr, seg_arr = preprocess_maps["pad_if_needed"](
                img=vol_arr,
                label=seg_arr,
                axes=(
                    1,
                    2,
                ),
                crop_size=160,
            )
            vol_arr, seg_arr = preprocess_maps["random_crop_3D"](
                img=vol_arr, label=seg_arr, crop_size=crop_size
            )

        elif mode == "center":
            vol_arr, seg_arr = preprocess_maps["crop_object"](vol_arr, seg_arr)
            vol_arr = preprocess_maps["crop_pad"](
                vol_arr, axes=(0, 1, 2), crop_size=crop_size
            )
            seg_arr = preprocess_maps["crop_pad"](
                seg_arr, axes=(0, 1, 2), crop_size=crop_size
            )

        elif mode == "random_center_crop":
            vol_arr, seg_arr = preprocess_maps["random_center_crop"](
                img=vol_arr, label=seg_arr
            )
            # vol_arr, seg_arr = foreground_crop(img=vol_arr, label=seg_arr)
            vol_arr = preprocess_maps["crop_pad"](
                vol_arr, axes=(0, 1, 2), crop_size=crop_size
            )
            seg_arr = preprocess_maps["crop_pad"](
                seg_arr, axes=(0, 1, 2), crop_size=crop_size
            )

        elif mode == "x_random":
            vol_arr = preprocess_maps["crop_pad"](
                vol_arr, axes=(0, 1), crop_size=(80, 160)
            )
            seg_arr = preprocess_maps["crop_pad"](
                seg_arr, axes=(0, 1), crop_size=(80, 160)
            )
            vol_arr, seg_arr = preprocess_maps["pad_if_needed"](
                img=vol_arr, label=seg_arr, axes=(2,), crop_size=160
            )
            vol_arr, seg_arr = preprocess_maps["random_crop_3D"](
                img=vol_arr, label=seg_arr, crop_size=crop_size
            )

        return vol_arr, seg_arr
