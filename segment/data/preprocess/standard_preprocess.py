import os
from random import random
from typing import List

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ...utils.utils import multiprocess
from ..augs import aug_maps as AUG_MAPS
from .base_preprocess import BasePreprocess3D

class Preprocess3D(BasePreprocess3D):
    def __init__(
        self,
        data: List[dict],
        crop_size: int = 256,
        save_file_path: str = None,
        vol_path: str = None,
        seg_path: str = None,
        configs: str = None,
    ):
        self.data = data
        self.crop_size = crop_size
        self.vol_path = vol_path
        self.seg_path = seg_path
        self.configs = configs

        super().__init__(data=data, vol_path=vol_path, seg_path=seg_path)

    def create_one_item(self, item: dict) -> dict:
        case_id = item["case_id"]
        vol = nib.load(item["img_path"])
        msk = nib.load(item["seg_path"])

        vol_affine = vol.affine[[2, 1, 0, 3]]
        msk_affine = vol.affine[[2, 1, 0, 3]]

        vol = vol.get_fdata()
        msk = msk.get_fdata()

        vol = np.clip(vol, self.configs["lower_bound"], self.configs["upper_bound"])
        vol, msk = self.drop_invalid_range(volume=vol, label=msk)

        new_vol = self.resample(
            vol, np.diag(abs(vol_affine)), self.configs["target_spacing"], order=3
        )
        new_msk = self.resample(
            msk, np.diag(abs(msk_affine)), self.configs["target_spacing"], order=0
        )

        cropped_vol = AUG_MAPS["crop_and_pad_if_needed"](
            new_vol, axes=(1, 2), crop_size=self.crop_size
        )
        cropped_msk = AUG_MAPS["crop_and_pad_if_needed"](
            new_msk, axes=(1, 2), crop_size=self.crop_size
        )

        non_zero = new_msk.sum()
        crop_non_zero = cropped_msk.sum()
        assert non_zero == crop_non_zero, "error cropping"

        return {"case_id": case_id, "vol": cropped_vol, "msk": cropped_msk}

    
    def create_one_test_item(self, item: dict) -> dict:
        case_id = item["case_id"]
        vol = nib.load(item["img_path"])
        vol_affine = vol.affine[[2, 1, 0, 3]]
        vol = vol.get_fdata()

        vol = np.clip(vol, self.configs["lower_bound"], self.configs["upper_bound"])
        vol = self.drop_invalid_range(volume=vol, label=None)

        new_vol = self.resample(vol, np.diag(abs(vol_affine)), self.configs["target_spacing"], order=3)
        cropped_vol = AUG_MAPS["crop_and_pad_if_needed"](new_vol, axes=(1, 2), crop_size=self.crop_size)
     
        return {"case_id": case_id, "vol": cropped_vol}
