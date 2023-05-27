import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from segment.data.augs import aug_maps
from segment.data.preprocess.base_preprocess import BasePreprocess3D
from segment.utils.utils import get_progress


class Postprocess3D(BasePreprocess3D):
    def __init__(self, configs: dict, data: dict): 
        self.config = configs
        self.data = data
        
    def create(self):
        predictions = list(map(self.create_one_item, get_progress(self.data, desc="postprocessing ")))
        case_id = [item["case_id"].split("_")[1] for item in self.data]
        return predictions, case_id

    def create_one_item(self, item):
        pred_path = item["npy_seg_path"]
        preprocess_vol_path = item["new_vol_path"]
        vol_path = item["vol_path"]
        case_id= item["case_id"].split("_")[1] 

        vol = sitk.ReadImage(vol_path)
        vol_array = sitk.GetArrayFromImage(vol).transpose(-1, 0, 1)
        
        preprocess_vol_arr = np.load(preprocess_vol_path)
        seg_array = np.load(pred_path)        
        seg_array = aug_maps["crop_and_pad_if_needed"](seg_array, axes=(0,1,2), crop_size=preprocess_vol_arr.shape)
        seg_array = self.resample(seg_array, self.config["target_spacing"], vol.GetSpacing(), order=0)
        seg_array = aug_maps["crop_and_pad_if_needed"](seg_array, axes=(0,1,2), crop_size=vol_array.shape)
        seg_array = seg_array.astype(np.int16, copy=False)
        seg_array = seg_array.transpose(-2, -1, 0)
        
        # Correct dimenstion due to the differences in how the nibabel and sitk interpret the image orientation
        seg_array = np.rot90(seg_array, axes=(0,1))
        seg_array = np.flip(seg_array, axis=0)

        seg_nii = sitk.GetImageFromArray(seg_array)
        seg_nii.SetDirection(vol.GetDirection())
        seg_nii.SetOrigin(vol.GetOrigin())
        seg_nii.SetSpacing(vol.GetSpacing())

        sitk.WriteImage(seg_nii, f"predictions/prediction_{case_id}.nii.gz")

        return seg_nii

    def create_one_item_nibabel(self, item):
        pred_path = item["npy_seg_path"]
        preprocess_vol_path = item["new_vol_path"]
        vol_path = item["vol_path"]
        case_id = item["case_id"].split("_")[1] 

        vol = nib.load(vol_path)
        vol_affine = vol.affine[[2,1,0,3]]
        vol_array = vol.get_fdata()

        preprocess_vol_arr = np.load(preprocess_vol_path)

        seg_array = np.load(pred_path)        
        seg_array = aug_maps["crop_and_pad_if_needed"](seg_array, axes=(0,1,2), crop_size=preprocess_vol_arr.shape)
        seg_array = self.resample(seg_array, self.config["target_spacing"], np.diag(abs(vol_affine)), order=0)
        seg_array = aug_maps["crop_and_pad_if_needed"](seg_array, axes=(0,1,2), crop_size=vol_array.shape)
        seg_array = seg_array.astype(np.int16, copy=False)
        seg_nii = nib.Nifti1Image(seg_array, affine=vol.affine, header=vol.header)
        
        nib_save_path = f"predictions/prediction_{case_id}_nib.nii.gz"
        nib.save(seg_nii, nib_save_path)
        
        return seg_nii
