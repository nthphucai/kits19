import os
import numpy as np
import nibabel as nib

from segment.data.preprocess.base_preprocess import BasePreprocess3D
from segment.data.augs import aug_maps
from segment.utils.utils import get_progress

class Postprocess3D(BasePreprocess3D):
    def __init__(self,
        configs: dict, 
        prediction: dict, 
        predict_path: str="predictions",
    ): 
        self.config = configs
        self.preds = prediction

        if not os.path.exists(predict_path):
            os.makedirs(predict_path)
        
        self.predict_path = predict_path

    def create(self):
        predictions = list(map(self.create_one_item, get_progress(self.preds, desc="postprocessing ")))
        case_id = [item["case_id"].split("_")[1] for item in self.preds]

        pred_path = [self.predict_path + f"/prediction_{idc}.nii.gz" for idc in case_id]
        [nib.save(seg, path) for seg, path in zip(predictions, pred_path)]
        
    def create_one_item(self, item):
        pred_path = item["new_seg_path"]
        vol_path = item["vol_path"]
        case_id = item["case_id"]

        vol = nib.load(vol_path)
        vol_affine = vol.affine[[2,1,0,3]]
        vol_array = vol.get_fdata()

        seg_array = np.load(pred_path)
        seg_array = self.resample(seg_array, self.config["target_spacing"], np.diag(abs(vol_affine)), order=0)
        seg_array = aug_maps["crop_and_pad_if_needed"](seg_array, axes=(0,1,2), crop_size=vol_array.shape)
        seg_nii = nib.Nifti1Image(seg_array, vol_affine)
        return seg_nii
