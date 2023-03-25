import os
import numpy as np
import nibabel as nib

from segment.data.preprocess.base_preprocess import BasePreprocess3D
from segment.data.augs import aug_maps
from segment.utils.utils import get_progress

class Postprocess3D(BasePreprocess3D):
    def __init__(self, configs: dict, prediction: dict): 
        self.config = configs
        self.preds = prediction
        
    def create(self):
        predictions = list(map(self.create_one_item, get_progress(self.preds, desc="postprocessing ")))
        case_id = [item["case_id"].split("_")[1] for item in self.preds]
        return predictions, case_id

    def create_one_item(self, item):
        pred_path = item["npy_seg_path"]
        preprocess_vol_path = item["new_vol_path"]
        vol_path = item["vol_path"]
        case_id = item["case_id"]

        vol = nib.load(vol_path)
        vol_affine = vol.affine[[2,1,0,3]]
        vol_array = vol.get_fdata()

        preprocess_vol_arr = np.load(preprocess_vol_path)

        seg_array = np.load(pred_path)        
        seg_array = aug_maps["crop_and_pad_if_needed"](seg_array, axes=(0,1,2), crop_size=preprocess_vol_arr.shape)
        seg_array = self.resample(seg_array, self.config["target_spacing"], np.diag(abs(vol_affine)), order=0)
        seg_array = aug_maps["crop_and_pad_if_needed"](seg_array, axes=(0,1,2), crop_size=vol_array.shape)
        seg_nii = nib.Nifti1Image(seg_array, vol_affine[[2, 1, 0, 3]])
        return seg_nii
